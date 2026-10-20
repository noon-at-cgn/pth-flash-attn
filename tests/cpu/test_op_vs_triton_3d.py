# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""The Triton arm of ``bench/op_vs_triton.py`` runs vLLM's split-KV (3D) kernel at q=1.

Why it matters: ``unified_attention`` falls back to its slower 2D kernel for a
single-token decode unless the caller passes ``seq_threshold_3D``,
``num_par_softmax_segments`` and the three ``softmax_segm_*`` buffers, as vLLM's
``TritonAttentionImpl`` does. Without them the q=1 ratios were about ten times too
large. The harness needs a GPU, so this test reads its source: the buffers exist
with the shapes vLLM uses and are passed to the call.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "bench" / "op_vs_triton.py"
KEYS = {
    "seq_threshold_3D",
    "num_par_softmax_segments",
    "softmax_segm_output",
    "softmax_segm_max",
    "softmax_segm_expsum",
}


def _tree() -> ast.Module:
    return ast.parse(SRC.read_text(encoding="utf-8"))


def test_the_buffers_of_the_3d_kernel_are_built_with_vllms_keys():
    dicts = [
        n
        for n in ast.walk(_tree())
        if isinstance(n, ast.Dict)
        and {k.value for k in n.keys if isinstance(k, ast.Constant)} == KEYS
    ]
    assert len(dicts) == 1


def test_the_triton_call_receives_them():
    calls = [
        n
        for n in ast.walk(_tree())
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", "") == "unified_attention"
        and any(k.arg is None for k in n.keywords)
    ]
    assert len(calls) == 1
    (star,) = [k for k in calls[0].keywords if k.arg is None]
    assert isinstance(star.value, ast.Name) and star.value.id == "seg_3d"


def test_the_threshold_follows_vllms_rule():
    """``seq_threshold_3D`` is ``MIN_LAUNCH_GRID_SIZE_2D // num_kv_heads``.

    vLLM 0.31.0 derives the threshold from the number of KV heads, not from a
    constant; a fixed value would run the 3D kernel for batches where vLLM runs
    the 2D one (and the reverse at other head counts).
    """
    rules = [
        n
        for n in ast.walk(_tree())
        if isinstance(n, ast.Assign)
        and [t.id for t in n.targets if isinstance(t, ast.Name)] == ["seg_threshold"]
    ]
    assert len(rules) == 1
    assert ast.unparse(rules[0].value) == "triton_attn.MIN_LAUNCH_GRID_SIZE_2D // HKV"
