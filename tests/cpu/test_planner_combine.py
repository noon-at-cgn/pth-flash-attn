# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for the split-KV combine in the log2 domain.

The kernel works in the exp2 domain throughout: scores are pre-multiplied by
log2(e), probabilities are 2^(x - m), and each split stores its normalised
output O_s together with lse2_s = m_s + log2(l_s). The combine kernel merges
the splits as

    M = max_s lse2_s,   o = sum_s 2^(lse2_s - M) O_s / sum_s 2^(lse2_s - M),

skipping splits with lse2_s = -inf (empty splits). This file is the
executable fp64 specification of that merge, written against the planner's
tile partition (``split_tile_range``), so the optimised combine kernel has a
reference and the two traps (wrong log base, multiplying garbage of an empty
split) are pinned by negative controls.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pth_flash_attn.planner import split_tile_range

# KV tokens per load tile (N).
TILE = 64
LOG2E = 1.0 / math.log(2.0)


def _split_partials(
    x2: np.ndarray, v: np.ndarray, num_splits: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-split (O_s, lse2_s) for one query row; x2 are exp2-domain scores.

    Empty splits (fewer tiles than splits) get lse2 = -inf and an O full of
    NaN: the workspace comes from torch.empty, so an unwritten partial holds
    arbitrary bits, and NaN is the worst case for a combine that multiplies it.
    """
    num_tiles = -(-len(x2) // TILE)
    outs = np.full((num_splits, v.shape[1]), np.nan)
    lses = np.full(num_splits, -np.inf)
    for s in range(num_splits):
        first, last = split_tile_range(
            split=s, num_splits=num_splits, num_tiles=num_tiles
        )
        lo, hi = first * TILE, min(last * TILE, len(x2))
        if lo >= hi:
            continue
        m = x2[lo:hi].max()
        p = np.exp2(x2[lo:hi] - m)
        lses[s] = m + np.log2(p.sum())
        outs[s] = p @ v[lo:hi] / p.sum()
    return outs, lses


def _combine_log2(outs: np.ndarray, lses: np.ndarray) -> np.ndarray:
    """The combine kernel's rule, in fixed split order (deterministic)."""
    live = np.isfinite(lses)  # skip empty splits BEFORE touching their O
    w = np.exp2(lses[live] - lses[live].max())
    return (w[:, None] * outs[live]).sum(axis=0) / w.sum()


def _reference(x2: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Plain softmax in the natural-exp domain: softmax(s) with s = x2 * ln 2."""
    s = x2 / LOG2E
    p = np.exp(s - s.max())
    return p @ v / p.sum()


@pytest.mark.parametrize("length", [1, 63, 64, 65, 1000, 4097])
@pytest.mark.parametrize("num_splits", [1, 2, 7, 34, 100])
def test_log2_combine_equals_unsplit_softmax(length: int, num_splits: int) -> None:
    """Merging split partials by log2-LSE reproduces unsplit attention.

    Why: split-KV is only an execution strategy; the result must be the same
    softmax(QK^T)V up to rounding, for any split count the planner may return,
    including more splits than tiles (L=1, S=100: 99 empty splits).
    """
    rng = np.random.default_rng(length * 1000 + num_splits)
    x2 = rng.normal(0.0, 3.0, length) * LOG2E
    v = rng.normal(0.0, 1.0, (length, 16))
    got = _combine_log2(*_split_partials(x2, v, num_splits))
    np.testing.assert_allclose(got, _reference(x2, v), rtol=1e-12, atol=1e-12)


def test_negative_control_natural_log_merge_is_wrong() -> None:
    """Negative control: weighting log2-LSEs with exp() instead of exp2() fails.

    Why: mixing log bases is the classic unit bug at the kernel/combine
    boundary (e.g. if partials were ever handed to vLLM's merge_attn_states,
    which uses natural-log LSE). The test must be able to see that bug.
    """
    rng = np.random.default_rng(7)
    x2 = rng.normal(0.0, 3.0, 4097) * LOG2E
    v = rng.normal(0.0, 1.0, (4097, 16))
    outs, lses = _split_partials(x2, v, 8)
    w = np.exp(lses - lses.max())  # wrong base
    wrong = (w[:, None] * outs).sum(axis=0) / w.sum()
    assert np.abs(wrong - _reference(x2, v)).max() > 1e-3


def test_negative_control_empty_split_garbage_poisons_unskipped_merge() -> None:
    """Negative control: without skipping lse = -inf, an empty split's garbage O
    reaches the output (0 * NaN = NaN).

    Why: empty splits exist whenever S > tiles (short contexts under a
    length-independent S) and their O slots are never written. Skipping them
    before the multiply, as the combine rule says, is what makes that safe.
    """
    rng = np.random.default_rng(3)
    x2 = rng.normal(0.0, 1.0, 100) * LOG2E  # 2 tiles, 34 splits -> 32 empty
    v = rng.normal(0.0, 1.0, (100, 16))
    outs, lses = _split_partials(x2, v, 34)
    assert np.isinf(lses).sum() == 32
    w = np.exp2(lses - lses.max())  # 0 for empty splits, but 0 * NaN = NaN
    unskipped = (w[:, None] * outs).sum(axis=0) / w.sum()
    assert np.isnan(unskipped).all()
    assert np.isfinite(_combine_log2(outs, lses)).all()
