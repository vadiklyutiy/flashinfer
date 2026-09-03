# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Warp-MMA BF16 GEMM with intra-CTA split-K for low-M decode shapes.

The operands are relabelled so the weight's output dimension becomes the tiled
GEMM-M: a decode GEMM with public ``M = 1`` then still has ``N / 16``
independent CTAs instead of one.  Inside a CTA the work is divided again,
``mma_warps_m`` warps over the weight rows and ``mma_warps_k`` warps over the K
tile, with the partial accumulators summed through shared memory.  This is the
structural difference from ``dense_bf16_gemm_sm100_splitk``, which splits K
across CTAs.

The kernel deliberately uses ``cp.async`` and warp-level ``mma.sync`` rather
than TMA and ``tcgen05``.  At these sizes the run time is dominated by fixed
cost, and skipping the TMEM allocation and the TMA descriptor setup is worth
more than the wider instruction.

Naming follows the layout, not the public API: ``gemm_m`` is the public N,
``gemm_n`` is the public M (token count), ``gemm_k`` is the contraction extent.
"""

from __future__ import annotations

import dataclasses
import functools
import math

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import torch as _torch
from cutlass import Boolean, const_expr
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack

MMA_INST_MNK = (16, 8, 16)
WARP_SIZE = 32
# The MMA N tiler is doubled so a B fragment fills one ldmatrix.x4.
TILE_N = MMA_INST_MNK[1] * 2
_COPY_BITS = 128
_COPY_ELEMS = 8

_SUPPORTED_WARPS_M = (1, 2, 4)
_SUPPORTED_WARPS_K = (1, 2, 4, 8)
_SUPPORTED_TILE_KS = (64, 128, 256)
# Tokens one CTA covers, in units of the MMA N of 8.  16 makes a B fragment
# fill exactly one ldmatrix.x4; 8 halves the wasted MMA columns when the batch
# is narrower than that, and 32 serves 17..32 tokens in one pass instead of
# two.
_SUPPORTED_TILE_NS = (8, 16, 32)
_MAX_SMEM_BYTES = 200 * 1024
_MAX_M = 64
# Autotuning explores a strict subset of the 26 valid (warps_m, warps_k,
# tile_k) families.  These nine are what a B200 sweep of the entire tactic
# space -- 28,910 measurements over the 16 routed shapes at M in
# {1,2,4,8,16,24,32}, repeated on a second node -- found worth measuring.
# Together with the depths below they offer a median of 23 tactics per shape
# and give up 0.8% against the best tactic in the whole space at the 90th
# percentile, 1.7% at worst.  The rule this replaced measured 20 tactics and
# gave up 2.1% and 8.6%.
#
# Families are the right unit here rather than independent warp and tile
# ranges, because they substitute for each other: every low-``warps_k`` entry
# looks individually removable, yet dropping the group costs up to 4.4% on
# K=384, where only a 64-wide K tile leaves enough tiles to fill a pipeline.
# The lone ``warps_m=4`` entry is the cheapest of them all -- it is valid only
# where N divides by 64 and K by 256, so it adds three tactics, and it halves
# the worst case by winning N=3584,K=7168 at M=24 and 32.
_AUTOTUNE_FAMILIES = (
    (1, 4, 256),
    (1, 2, 128),
    (1, 2, 256),
    (2, 2, 256),
    (1, 4, 128),
    (2, 1, 64),
    (2, 1, 128),
    (2, 2, 128),
    (4, 1, 256),
)
# Pipeline depths offered per family, each clamped to the deepest that fits.
# Winners spread over 1..12 stages, so no single depth and no occupancy-derived
# pair covers them: past 114 KB every tactic is down to one resident CTA, yet
# inside that regime depth still decides, and N=2112,K=7168 wants ten stages at
# 164 KB.  Clamping does most of the work -- these three rungs land on eleven
# distinct depths across the shapes, and adding the rungs 1, 8 and 12 back
# nearly doubles the tactic count while leaving the worst case where it is.
_AUTOTUNE_DEPTHS = (4, 8, 12)
_MAX_STAGES = 8
# SMs on a B200, which is what the token-tile ranking is calibrated against.
_SM_COUNT = 148
# Token tiles measured per token count, taken off ``_tile_n_ranking``.  Two is
# the useful number: the ranking's first choice is already optimal in 102 of
# 112 measured cases, and its second covers six of the ten misses, including
# the largest at 8.6%.  A third would add half again as many tactics to fix two
# cases worth 1.3% and 4.0% on the launch-bound K=128 shapes.
_AUTOTUNE_TILE_NS = 2
# Below this many K tiles a pipeline cannot be filled, so a narrower K tile
# that yields more of them wins.  Every measured tile_k winner follows it.
_MIN_PIPELINED_K_TILES = 6
# Half of a B200 SM's 228 KB shared memory, less a margin.  Staging past this
# drops the SM to one resident CTA, which costs more than the extra depth buys
# whenever the grid does not divide evenly into whole waves: at N=3216, K=7168
# eight stages (128 KB) measured 15.6 us against 12.7 us for six (96 KB).
_SMEM_TWO_CTAS_BYTES = 112 * 1024


@dataclasses.dataclass(frozen=True)
class SplitK2Tactic:
    """One split-k-2 specialization.

    Every field except ``tile_n`` is a function of ``(N, K)`` only, so a cached
    choice stays valid as the token count moves inside an autotuner bucket.
    ``tile_n`` is how many tokens one CTA covers, so it does depend on M -- but
    only through ``ceil(M / tile_n)``, and FlashInfer buckets M by powers of
    two, which keeps a bucket's tactic valid across the whole bucket.
    """

    mma_warps_m: int
    mma_warps_k: int
    tile_k: int
    stages: int
    tile_n: int = TILE_N


def _align_1k(nbytes: int) -> int:
    return (nbytes + 1023) // 1024 * 1024


def _g2s_thread_shape(tile_rows: int, tile_k: int, num_threads: int) -> tuple[int, int]:
    """``(rows, k_vecs)`` cp.async thread layout for a K-major operand tile.

    Prefers 8 threads along K so one row-group covers a 128-byte cache line,
    but widens when the tile has too few rows to keep every thread busy.
    """
    k_vecs_max = tile_k // _COPY_ELEMS
    k_vecs = min(max(num_threads // tile_rows, min(k_vecs_max, 8)), k_vecs_max)
    if k_vecs == 0 or num_threads % k_vecs:
        raise ValueError(f"{num_threads} threads do not split into {k_vecs} K vectors")
    rows = num_threads // k_vecs
    if rows > tile_rows or tile_rows % rows:
        raise ValueError(f"cannot tile {tile_rows}x{tile_k} over {num_threads} threads")
    return rows, k_vecs


def _smem_bytes(tactic: SplitK2Tactic, stages: int) -> int:
    tile_m = MMA_INST_MNK[0] * tactic.mma_warps_m
    num_warps = tactic.mma_warps_m * tactic.mma_warps_k
    acc_elems = MMA_INST_MNK[0] * tactic.tile_n // WARP_SIZE
    reduce_bytes = (
        num_warps * WARP_SIZE * acc_elems * 4 if tactic.mma_warps_k > 1 else 0
    )
    # SmemAllocator bump-allocates, so the three buffers do not alias.
    return sum(
        _align_1k(nbytes)
        for nbytes in (
            tile_m * tactic.tile_k * 2 * stages,
            tactic.tile_n * tactic.tile_k * 2 * stages,
            reduce_bytes,
        )
    )


def _ldmatrix_count(rows: int) -> int:
    """8x8 matrices one ``ldmatrix`` should move for a ``rows`` x 16 tile.

    The instruction moves at most four, so a wider tile is covered by issuing
    the atom more than once.
    """
    return min(4, rows * MMA_INST_MNK[2] // 64)


def effective_stages(tactic: SplitK2Tactic, k: int) -> int:
    """Pipeline depth after clamping to the K-tile count."""
    return max(1, min(tactic.stages, k // tactic.tile_k))


def validate_tactic(tactic: SplitK2Tactic, m: int, n: int, k: int) -> None:
    """Reject a split-k-2 tactic that cannot serve ``(m, n, k)``."""
    if tactic.mma_warps_m not in _SUPPORTED_WARPS_M:
        raise ValueError(f"unsupported mma_warps_m={tactic.mma_warps_m}")
    if tactic.mma_warps_k not in _SUPPORTED_WARPS_K:
        raise ValueError(f"unsupported mma_warps_k={tactic.mma_warps_k}")
    if tactic.tile_k not in _SUPPORTED_TILE_KS:
        raise ValueError(f"unsupported tile_k={tactic.tile_k}")
    if tactic.tile_n not in _SUPPORTED_TILE_NS:
        raise ValueError(f"unsupported tile_n={tactic.tile_n}")
    if tactic.stages < 1:
        raise ValueError(f"stages={tactic.stages} must be positive")
    if not 1 <= m <= _MAX_M:
        raise ValueError(f"split-k-2 GEMM requires 1 <= M <= {_MAX_M}, got {m}")
    tile_m = MMA_INST_MNK[0] * tactic.mma_warps_m
    if n <= 0 or n % tile_m:
        raise ValueError(f"N={n} must be divisible by tile_m={tile_m}")
    if k <= 0 or k % tactic.tile_k:
        raise ValueError(f"K={k} must be divisible by tile_k={tactic.tile_k}")
    if tactic.tile_k % (MMA_INST_MNK[2] * tactic.mma_warps_k):
        raise ValueError(
            f"tile_k={tactic.tile_k} must be divisible by "
            f"{MMA_INST_MNK[2] * tactic.mma_warps_k} for "
            f"{tactic.mma_warps_k}-way split-K"
        )
    num_threads = tactic.mma_warps_m * tactic.mma_warps_k * WARP_SIZE
    if num_threads > 1024:
        raise ValueError(f"{num_threads} threads exceeds the CTA limit")
    _g2s_thread_shape(tile_m, tactic.tile_k, num_threads)
    _g2s_thread_shape(tactic.tile_n, tactic.tile_k, num_threads)
    smem = _smem_bytes(tactic, effective_stages(tactic, k))
    if smem > _MAX_SMEM_BYTES:
        raise ValueError(f"{smem} bytes of SMEM exceeds {_MAX_SMEM_BYTES}")


def _tile_n_ranking(n: int, m: int) -> list[int]:
    """Token tiles, most promising first, ranked by how the grid fills the GPU.

    A wider token tile serves the batch in fewer passes, but the passes are not
    serial -- they are more CTAs in the same grid, and on a GPU that is not
    full they overlap.  So what matters is how many waves the *whole* grid
    takes, not how many passes there are, and only when two tiles tie on waves
    does the extra parallelism of the narrower one decide.  Ranking on that
    picks the measured best in 102 of 112 cases; ranking on passes alone, or
    breaking ties toward the wider tile, picks it in 27.

    ``tile_m`` is taken as one warp's 16 rows, the value ``mma_warps_m=1``
    families use, since those win almost everywhere.
    """

    def waves(tile_n: int) -> tuple[int, int, int]:
        ctas = (n // MMA_INST_MNK[0]) * -(-m // tile_n)
        return (-(-ctas // _SM_COUNT), -ctas, tile_n)

    return sorted(_SUPPORTED_TILE_NS, key=waves)


def _max_stages(
    tactic: SplitK2Tactic, k: int, budget: int, cap: int = _MAX_STAGES
) -> int:
    """Deepest pipeline that fits the K-tile count, ``budget`` bytes and ``cap``."""
    depth = min(cap, k // tactic.tile_k)
    while depth > 1 and _smem_bytes(tactic, depth) > budget:
        depth -= 1
    return depth


def default_tactic(m: int, n: int, k: int) -> SplitK2Tactic:
    """Pick a tactic without measuring.

    Takes the widest K tile that still leaves enough K tiles to fill a
    pipeline, splits it four ways across warps when the tile is wide enough to
    keep a whole ``ldmatrix.x4`` per warp, and stages as deeply as it can
    without dropping the SM to a single resident CTA.
    """
    divisors = [tile_k for tile_k in _SUPPORTED_TILE_KS if k % tile_k == 0]
    if not divisors:
        raise ValueError(f"K={k} is not a multiple of any supported tile_k")
    pipelined = [tile_k for tile_k in divisors if k // tile_k >= _MIN_PIPELINED_K_TILES]
    for tile_k in sorted(pipelined or divisors, reverse=True):
        warps_k = 4 if tile_k >= 128 else 1
        probe = SplitK2Tactic(1, warps_k, tile_k, 1)
        stages = _max_stages(probe, k, _SMEM_TWO_CTAS_BYTES)
        tactic = SplitK2Tactic(1, warps_k, tile_k, stages)
        try:
            validate_tactic(tactic, m, n, k)
        except ValueError:
            continue
        return tactic
    raise ValueError(f"no split-k-2 tactic serves (M={m}, N={n}, K={k})")


def autotune_tactics(m: int, n: int, k: int) -> list[SplitK2Tactic]:
    """Enumerate the compact tactic space used by FlashInfer autotuning.

    Each measured-useful family is offered at every depth in
    ``_AUTOTUNE_DEPTHS``, clamped to the deepest pipeline the family can hold
    on this K and token tile.  Clamping is what lets one fixed ladder serve
    every shape: at K=128 there is a single K tile, so every rung collapses
    onto depth 1.

    Token tiles are not enumerated.  ``_tile_n_ranking`` orders them well
    enough that the top two carry the choice, which keeps the space near where
    it was before ``tile_n`` existed while recovering most of what pinning it
    at 16 gave up: 1.2% at the 90th percentile against 9.3%.
    """
    tactics: list[SplitK2Tactic] = []
    try:
        tactics.append(default_tactic(m, n, k))
    except ValueError:
        return []
    deepest_rung = max(_AUTOTUNE_DEPTHS)
    for tile_n in _tile_n_ranking(n, m)[:_AUTOTUNE_TILE_NS]:
        for warps_m, warps_k, tile_k in _AUTOTUNE_FAMILIES:
            if k % tile_k:
                continue
            probe = SplitK2Tactic(warps_m, warps_k, tile_k, 1, tile_n)
            deepest = _max_stages(probe, k, _MAX_SMEM_BYTES, cap=deepest_rung)
            for rung in _AUTOTUNE_DEPTHS:
                tactic = SplitK2Tactic(
                    warps_m, warps_k, tile_k, min(rung, deepest), tile_n
                )
                try:
                    validate_tactic(tactic, m, n, k)
                except ValueError:
                    continue
                tactics.append(tactic)
    return list(dict.fromkeys(tactics))


def _make_ab_smem_layout(dtype, rows: int, tile_k: int, stages: int):
    """K-major swizzled SMEM layout for ``stages`` ``(rows, tile_k)`` tiles.

    ``Swizzle<3, 3, 3>`` on 16-bit elements is the 8-row XOR of 16-byte chunks
    that keeps ``ldmatrix`` bank-conflict free.
    """
    major = min(tile_k, 64)
    swizzle_bits = min(int(math.log2(major * dtype.width // _COPY_BITS)), 3)
    atom = cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, 3, 3),
        0,
        cute.make_layout((8, major), stride=(major, 1)),
    )
    return cute.tile_to_shape(atom, (rows, tile_k, stages), (0, 1, 2))


def _make_g2s_tiled_copy(dtype, tile_rows: int, tile_k: int, num_threads: int):
    rows, k_vecs = _g2s_thread_shape(tile_rows, tile_k, num_threads)
    return cute.make_tiled_copy_tv(
        cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            dtype,
            num_bits_per_copy=_COPY_BITS,
        ),
        cute.make_layout((rows, k_vecs), stride=(k_vecs, 1)),
        cute.make_layout((1, _COPY_ELEMS)),
    )


class SplitK2DenseGemmKernel:
    """Warp-level MMA GEMM with a cp.async pipeline and intra-CTA split-K."""

    def __init__(
        self,
        *,
        element_type,
        k_extent: int,
        tactic: SplitK2Tactic,
        use_pdl: bool,
    ) -> None:
        self.element_type = element_type
        self.mma_warps_m = tactic.mma_warps_m
        self.mma_warps_k = tactic.mma_warps_k
        self.tile_m = MMA_INST_MNK[0] * tactic.mma_warps_m
        self.tile_n = tactic.tile_n
        self.tile_k = tactic.tile_k
        self.use_pdl = use_pdl

        self.num_warps = tactic.mma_warps_m * tactic.mma_warps_k
        self.num_threads = self.num_warps * WARP_SIZE
        self.k_iters = k_extent // tactic.tile_k
        self.k_blocks_per_warp = tactic.tile_k // tactic.mma_warps_k // MMA_INST_MNK[2]
        self.stages = effective_stages(tactic, k_extent)
        self.acc_elems = MMA_INST_MNK[0] * self.tile_n // WARP_SIZE
        self.smem_bytes = _smem_bytes(tactic, self.stages)

    def _make_tiled_mma(self):
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.element_type, cutlass.Float32, MMA_INST_MNK),
            cute.make_layout((1, 1, 1)),
            permutation_mnk=(MMA_INST_MNK[0], self.tile_n, MMA_INST_MNK[2]),
        )

    @cute.jit
    def __call__(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        stream: _cuda.CUstream,
    ) -> None:
        self.kernel(
            gA,
            gB,
            gC,
            self._make_tiled_mma(),
            _make_g2s_tiled_copy(
                self.element_type, self.tile_m, self.tile_k, self.num_threads
            ),
            _make_g2s_tiled_copy(
                self.element_type, self.tile_n, self.tile_k, self.num_threads
            ),
        ).launch(
            grid=[
                cute.ceil_div(cute.size(gA, mode=[0]), self.tile_m),
                cute.ceil_div(cute.size(gB, mode=[0]), self.tile_n),
                1,
            ],
            block=[self.num_threads, 1, 1],
            smem=self.smem_bytes,
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        tiled_mma: cute.TiledMma,
        copy_a: cute.TiledCopy,
        copy_b: cute.TiledCopy,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        gemm_n = cute.size(gB, mode=[0])

        warp_idx = cute.arch.warp_idx()
        lane_idx = tidx % WARP_SIZE
        warp_m = warp_idx % self.mma_warps_m
        warp_k = warp_idx // self.mma_warps_m

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(
            self.element_type,
            _make_ab_smem_layout(
                self.element_type, self.tile_m, self.tile_k, self.stages
            ),
            byte_alignment=1024,
        )
        sB = smem.allocate_tensor(
            self.element_type,
            _make_ab_smem_layout(
                self.element_type, self.tile_n, self.tile_k, self.stages
            ),
            byte_alignment=1024,
        )

        tile_a = cute.local_tile(gA, (self.tile_m, self.tile_k), (bidx, None))
        tile_b = cute.local_tile(gB, (self.tile_n, self.tile_k), (bidy, None))
        coord_b = cute.local_tile(
            cute.make_identity_tensor(gB.shape),
            (self.tile_n, self.tile_k),
            (bidy, None),
        )

        thr_copy_a = copy_a.get_slice(tidx)
        thr_copy_b = copy_b.get_slice(tidx)
        tAgA = thr_copy_a.partition_S(tile_a)
        tAsA = thr_copy_a.partition_D(sA)
        tBgB = thr_copy_b.partition_S(tile_b)
        tBsB = thr_copy_b.partition_D(sB)
        tBcB = thr_copy_b.partition_S(coord_b)

        # Token rows past the public M would read past the activation tensor.
        pred_b = cute.make_rmem_tensor(
            cute.make_layout(
                (1, cute.size(tBcB, mode=[1]), cute.size(tBcB, mode=[2])),
                stride=(0, cute.size(tBcB, mode=[2]), 1),
            ),
            Boolean,
        )
        for rest_n in cutlass.range_constexpr(cute.size(tBcB, mode=[1])):
            for rest_k in cutlass.range_constexpr(cute.size(tBcB, mode=[2])):
                pred_b[0, rest_n, rest_k] = tBcB[(0, 0), rest_n, rest_k, 0][0] < gemm_n

        # Each warp owns one 16-row slice of the weight tile.  Row slicing is
        # safe because the offset is a whole number of 8-row swizzle atoms; the
        # K split must *not* be a slice, since cutting a Swizzle<3,3,3> layout
        # below its 64-element period rebases the XOR and silently reads the
        # wrong banks.  K is selected per k-block index instead.
        sA_warp = cute.local_tile(
            sA, (MMA_INST_MNK[0], self.tile_k, self.stages), (warp_m, 0, 0)
        )

        thr_mma = tiled_mma.get_slice(lane_idx)
        acc = cute.make_rmem_tensor(
            cute.make_layout(
                tiled_mma.partition_shape_C((MMA_INST_MNK[0], self.tile_n))
            ),
            cutlass.Float32,
        )
        acc.fill(0.0)

        tCsA = thr_mma.partition_A(sA_warp)
        tCsB = thr_mma.partition_B(sB)
        frag_a = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        frag_b = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])

        s2r_a = cute.make_tiled_copy(
            cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(False, _ldmatrix_count(self.tile_m)),
                self.element_type,
            ),
            layout_tv=tiled_mma.tv_layout_A_tiled,
            tiler_mn=(tiled_mma.get_tile_size(0), tiled_mma.get_tile_size(2)),
        )
        # A narrow token tile holds fewer than four 8x8 matrices, so it needs a
        # narrower ldmatrix than the A side does.
        s2r_b = cute.make_tiled_copy(
            cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(False, _ldmatrix_count(self.tile_n)),
                self.element_type,
            ),
            layout_tv=tiled_mma.tv_layout_B_tiled,
            tiler_mn=(tiled_mma.get_tile_size(1), tiled_mma.get_tile_size(2)),
        )
        tCsA_view = s2r_a.get_slice(lane_idx).partition_S(sA_warp)
        tCrA_view = s2r_a.get_slice(lane_idx).retile(frag_a)
        tCsB_view = s2r_b.get_slice(lane_idx).partition_S(sB)
        tCrB_view = s2r_b.get_slice(lane_idx).retile(frag_b)

        gmem = (copy_a, tAgA, tAsA, copy_b, tBgB, tBsB, pred_b)
        compute = (
            warp_k * self.k_blocks_per_warp,
            tiled_mma,
            s2r_a,
            tCsA_view,
            tCrA_view,
            s2r_b,
            tCsB_view,
            tCrB_view,
            frag_a,
            frag_b,
            acc,
        )

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        if const_expr(self.stages == 1):
            for k_iter in cutlass.range(self.k_iters, unroll=1):
                cute.arch.sync_threads()
                self._load_stage(k_iter, 0, *gmem)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                self._mma_stage(0, *compute)
        else:
            for stage in cutlass.range_constexpr(self.stages - 1):
                self._load_stage(stage, stage, *gmem)
                cute.arch.cp_async_commit_group()
            for k_iter in cutlass.range(self.k_iters, unroll=1):
                # Draining to stages-2 in flight retires the group holding this
                # iteration's tile; the barrier that follows also guarantees
                # every warp is done reading the buffer about to be refilled.
                cute.arch.cp_async_wait_group(self.stages - 2)
                cute.arch.sync_threads()
                load_iter = k_iter + self.stages - 1
                if load_iter < self.k_iters:
                    self._load_stage(load_iter, load_iter % self.stages, *gmem)
                cute.arch.cp_async_commit_group()
                self._mma_stage(k_iter % self.stages, *compute)

        # Every warp holds the same lane-to-(m, n) mapping, so summing the
        # split-K partials is elementwise across warps; indexing
        # lane-contiguously keeps both the store and the load conflict free.
        if const_expr(self.mma_warps_k > 1):
            partials = cutlass.utils.SmemAllocator().allocate_tensor(
                cutlass.Float32,
                cute.make_layout(
                    (WARP_SIZE, self.num_warps, self.acc_elems),
                    stride=(1, WARP_SIZE, WARP_SIZE * self.num_warps),
                ),
                byte_alignment=1024,
            )
            cute.arch.sync_threads()
            for i in cutlass.range_constexpr(self.acc_elems):
                partials[lane_idx, warp_idx, i] = acc[i]
            cute.arch.sync_threads()
            if warp_k == 0:
                for i in cutlass.range_constexpr(self.acc_elems):
                    for other in cutlass.range_constexpr(1, self.mma_warps_k):
                        acc[i] += partials[
                            lane_idx, warp_idx + other * self.mma_warps_m, i
                        ]

        if warp_k == 0:
            row_tile = bidx * self.mma_warps_m + warp_m
            tCgC = thr_mma.partition_C(
                cute.local_tile(gC, (MMA_INST_MNK[0], self.tile_n), (row_tile, bidy))
            )
            tCcC = thr_mma.partition_C(
                cute.local_tile(
                    cute.make_identity_tensor(gC.shape),
                    (MMA_INST_MNK[0], self.tile_n),
                    (row_tile, bidy),
                )
            )
            for i in cutlass.range_constexpr(cute.size(acc)):
                if tCcC[i][1] < gemm_n:
                    tCgC[i] = acc[i].to(self.element_type)

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def _load_stage(
        self, k_iter, stage, copy_a, tAgA, tAsA, copy_b, tBgB, tBsB, pred_b
    ) -> None:
        cute.copy(copy_a, tAgA[None, None, None, k_iter], tAsA[None, None, None, stage])
        cute.copy(
            copy_b,
            tBgB[None, None, None, k_iter],
            tBsB[None, None, None, stage],
            pred=pred_b,
        )

    @cute.jit
    def _mma_stage(
        self,
        stage,
        k_block_base,
        tiled_mma,
        s2r_a,
        tCsA_view,
        tCrA_view,
        s2r_b,
        tCsB_view,
        tCrB_view,
        frag_a,
        frag_b,
        acc,
    ) -> None:
        for j in cutlass.range_constexpr(self.k_blocks_per_warp):
            k_blk = k_block_base + j
            cute.copy(s2r_a, tCsA_view[None, 0, k_blk, stage], tCrA_view[None, 0, j])
            cute.copy(s2r_b, tCsB_view[None, 0, k_blk, stage], tCrB_view[None, 0, j])
        for j in cutlass.range_constexpr(self.k_blocks_per_warp):
            cute.gemm(tiled_mma, acc, frag_a[None, None, j], frag_b[None, None, j], acc)


def _from_dlpack_static(tensor: _torch.Tensor):
    return from_dlpack(tensor, assumed_align=32)


@functools.cache
def _get_compiled_splitk2_kernel(
    dtype,
    m: int,
    n: int,
    k: int,
    tactic: SplitK2Tactic,
    use_pdl: bool,
):
    if dtype != _torch.bfloat16:
        raise ValueError(f"split-k-2 GEMM supports BF16; got {dtype}")
    kernel = SplitK2DenseGemmKernel(
        element_type=cutlass.BFloat16,
        k_extent=k,
        tactic=tactic,
        use_pdl=use_pdl,
    )
    reprs = tuple(
        _from_dlpack_static(tensor)
        for tensor in (
            _torch.empty((n, k), dtype=dtype, device="cuda"),
            _torch.empty((m, k), dtype=dtype, device="cuda"),
            _torch.empty((m, n), dtype=dtype, device="cuda").T,
        )
    )
    stream = _cuda.CUstream(_torch.cuda.current_stream().cuda_stream)
    return cute_ext.compile(kernel, *reprs, stream)


def _validate_runtime_tensors(a, b, out, tactic: SplitK2Tactic):
    if any(not isinstance(tensor, _torch.Tensor) for tensor in (a, b, out)):
        raise ValueError("a, b, and out must be torch tensors")
    if a.ndim != 2 or b.ndim != 2 or out.ndim != 2:
        raise ValueError("split-k-2 GEMM accepts only 2D tensors")
    if a.device.type != "cuda" or b.device != a.device or out.device != a.device:
        raise ValueError("a, b, and out must be on the same CUDA device")
    if a.dtype != _torch.bfloat16 or b.dtype != a.dtype or out.dtype != a.dtype:
        raise ValueError("a, b, and out must share BF16 dtype")
    if not a.is_contiguous() or not b.T.is_contiguous() or not out.is_contiguous():
        raise ValueError("split-k-2 GEMM requires row-major A/out and column-major B")
    if any(tensor.data_ptr() % 32 for tensor in (a, b, out)):
        raise ValueError("a, b, and out must be 32-byte aligned")

    m, k = a.shape
    if b.shape[0] != k:
        raise ValueError(
            f"incompatible shapes: a is {tuple(a.shape)}, b is {tuple(b.shape)}"
        )
    n = b.shape[1]
    if out.shape != (m, n):
        raise ValueError(f"out must have shape {(m, n)}, got {tuple(out.shape)}")
    validate_tactic(tactic, m, n, k)
    return m, n, k


def run_splitk2_dense(a, b, out, pdl: bool, tactic: SplitK2Tactic):
    """Run ``A[M,K] @ B[K,N]`` with the ``mm_bf16`` layouts."""
    m, n, k = _validate_runtime_tensors(a, b, out, tactic)
    compiled = _get_compiled_splitk2_kernel(a.dtype, m, n, k, tactic, pdl)
    stream = _cuda.CUstream(_torch.cuda.current_stream(a.device).cuda_stream)
    compiled(
        _from_dlpack_static(b.T),
        _from_dlpack_static(a),
        _from_dlpack_static(out.T),
        stream,
    )
    return out


__all__ = [
    "SplitK2Tactic",
    "autotune_tactics",
    "default_tactic",
    "run_splitk2_dense",
    "validate_tactic",
]
