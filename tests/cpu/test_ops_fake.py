# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for the op schema, the fake kernel and the op's CPU registration.

What is protected here:
* the schema in ``ops.py`` is the frozen schema documented in ``docs/api.md``
  and the one the C++
  library registers (one interface, three copies, all equal);
* the fake kernel of the out-variant op returns None, never touches data, and
  rejects exactly what the C++ metadata checks reject, with the same message;
* the built op, called on CPU tensors (the registration that exists to test the
  checks), passes every check and then refuses to compute with a message of
  its own, and ``torch.library.opcheck`` behaves accordingly. The kernels
  themselves are tested on the GPU (tests/gpu/test_backend_op.py).

Two parts need nothing but torch: a test-local ``torch.library`` fragment
declares the same schema under a private namespace and gets the same fake, so
the fake is testable before (and independently of) the C++ build. The parts
that need ``_C.abi3.so`` skip when it has not been built.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from pth_flash_attn import ops
from pth_flash_attn.planner import plan_launch, workspace_token_splits
from torch._subclasses.fake_tensor import FakeTensorMode

REPO = Path(__file__).resolve().parents[2]
API_DOC = REPO / "docs" / "api.md"
BINDINGS = REPO / "csrc" / "pth_attn" / "torch_bindings.cpp"
CPU_ONLY_ERROR = "computes on CUDA tensors only"
needs_op_library = pytest.mark.skipif(
    not ops.is_available(), reason="op library _C.abi3.so not built"
)

# A private namespace with the identical schema and the identical fake, and no
# real kernel: the fake is then the only implementation that can run.
_TEST_NS = "pth_flash_attn_faketest"
_TEST_LIB = torch.library.Library(_TEST_NS, "DEF")  # kept alive for the module
_TEST_LIB.define(ops.OP_SCHEMA)
torch.library.register_fake(f"{_TEST_NS}::{ops.OP_NAME}", lib=_TEST_LIB)(
    ops.pth_flash_attn_fwd_fake
)
TEST_OP = getattr(getattr(torch.ops, _TEST_NS), ops.OP_NAME).default
ARG_NAMES = [a.name for a in torch._C.parse_schema(ops.OP_SCHEMA).arguments]


def _parse(text: str) -> str:
    """Canonical string of a schema, qualified with the op namespace."""
    if "::" not in text:
        text = f"{ops.OP_NAMESPACE}::{text}"
    return str(torch._C.parse_schema(text))


def _schema_from_api_doc() -> str:
    """The op schema text of docs/api.md (first code block that declares it),
    comments stripped."""
    fenced = API_DOC.read_text().split("```")[1::2]
    blocks = [b.split("\n", 1)[1] for b in fenced]  # drop the fence's info string
    section = next(b for b in blocks if "pth_flash_attn_fwd(" in b)
    lines = [line.split("#", 1)[0].strip() for line in section.splitlines()]
    return " ".join(x for x in lines if x)


def _schema_from_bindings() -> str:
    """The m.def() text in torch_bindings.cpp (adjacent C literals joined)."""
    body = BINDINGS.read_text().split("m.def(", 1)[1].split(");", 1)[0]
    return "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', body))


def make_args(
    *,
    reqs: int = 2,
    q_len: int = 4,
    G: int = 6,
    Hkv: int = 2,
    D: int = 256,
    num_blocks: int = 4,
    block_size: int = 16,
    num_splits: int | None = None,
    window: int = 0,
    causal: bool = True,
) -> dict[str, Any]:
    """Valid CPU arguments in schema order, shaped like production.

    The caches are strided views into one vLLM int8_per_token_head buffer:
    per (block, slot, KV head) one 2D+8-byte cell [K D | k_scale f32 | V D |
    v_scale f32], so the views have the real non-contiguous strides and the
    4-byte-aligned V offset (D + 4). The launch parameters come from the
    planner, as they will in the backend.
    """
    cell = torch.zeros(num_blocks, block_size, Hkv, 2 * D + 8, dtype=torch.int8)
    T = reqs * q_len
    w_ts = workspace_token_splits(
        max_num_seqs=reqs, q_dec_max=q_len, Hkv=Hkv, D=D, num_sms=68
    )
    plan = plan_launch(
        num_tokens=T,
        num_reqs=reqs,
        max_query_len=q_len,
        G=G,
        Hkv=Hkv,
        D=D,
        num_sms=68,
        is_prefill=False,
        ws_token_splits=w_ts,
        window=window,
    )
    splits = plan.num_splits if num_splits is None else num_splits
    ws_t = splits if splits > 1 else 0
    args = {
        "out": torch.zeros(T, G * Hkv, D, dtype=torch.bfloat16),
        "q": torch.zeros(T, G * Hkv, D, dtype=torch.bfloat16),
        "k_cache": cell[..., :D],
        "v_cache": cell[..., D + 4 : 2 * D + 4],
        "k_scale": cell[..., D : D + 4].view(torch.float32)[..., 0],
        "v_scale": cell[..., 2 * D + 4 :].view(torch.float32)[..., 0],
        "block_table": torch.zeros(reqs, num_blocks, dtype=torch.int32),
        "seq_lens": torch.full((reqs,), q_len, dtype=torch.int32),
        "query_start_loc": torch.arange(0, T + 1, q_len, dtype=torch.int32),
        "max_query_len": q_len,
        "wm": plan.wm,
        "num_splits": splits,
        "softmax_scale": D**-0.5,
        "ws_o": torch.empty(ws_t, T, G * Hkv, D, dtype=torch.float32),
        "ws_lse": torch.empty(ws_t, T, G * Hkv, dtype=torch.float32),
        "window": window,
        "causal": causal,
    }
    assert list(args) == ARG_NAMES
    return args


def _with(**over: Any) -> Callable[[], dict[str, Any]]:
    return lambda: {**make_args(), **over}


def _q_and_out(shape: tuple[int, ...], dtype: torch.dtype = torch.bfloat16) -> dict:
    return {
        "q": torch.zeros(shape, dtype=dtype),
        "out": torch.zeros(shape, dtype=dtype),
    }


# The speculative-decoding drafter's attention per rank: D = 128, G = 4,
# 8 query tokens per request (bonus + 7), window 2048, non-causal.
DRAFTER: dict[str, Any] = {
    "D": 128,
    "G": 4,
    "q_len": 8,
    "window": 2048,
    "causal": False,
}

# The C++ refusal of a windowed or non-causal call off D = 128.
NEEDS_D128 = "sliding-window and non-causal attention need head size D = 128"

# (case id, argument builder, expected message; None = valid arguments).
# Each invalid case breaks exactly one metadata check of launch.cu.
CASES: list[tuple[str, Callable[[], dict[str, Any]], str | None]] = [
    ("valid_split", make_args, None),
    ("valid_s1_empty_ws", lambda: make_args(num_splits=1), None),
    (
        "q_rank",
        _with(q=torch.zeros(8, 3072, dtype=torch.bfloat16)),
        "q must be 3-D, got 2-D",
    ),
    (
        "q_dtype",
        _with(**_q_and_out((8, 12, 256), torch.float32)),
        "q must be bfloat16 or float16, got float32",
    ),
    (
        "head_size",
        _with(**_q_and_out((8, 12, 96))),
        "head size D = q.size(2) must be 64, 128 or 256, got 96",
    ),
    (
        "q_inner_stride",
        _with(q=torch.zeros(8, 12, 512, dtype=torch.bfloat16)[..., ::2]),
        "q must have stride 1 in its last dimension, got 2",
    ),
    (
        "out_dtype",
        _with(out=torch.zeros(8, 12, 256, dtype=torch.float16)),
        "out (same dtype as q) must be bfloat16, got float16",
    ),
    (
        "out_tokens",
        _with(out=torch.zeros(7, 12, 256, dtype=torch.bfloat16)),
        "out.size(0) must equal T = 8, got 7",
    ),
    (
        "k_cache_dtype",
        _with(k_cache=torch.zeros(4, 16, 2, 256, dtype=torch.uint8)),
        "k_cache must be int8, got an unsupported dtype",
    ),
    (
        "k_cache_head_size",
        _with(k_cache=torch.zeros(4, 16, 2, 128, dtype=torch.int8)),
        "k_cache.size(3) must equal D = 256, got 128",
    ),
    (
        "v_cache_blocks",
        _with(v_cache=torch.zeros(3, 16, 2, 256, dtype=torch.int8)),
        "v_cache.size(0) must equal NB = 4, got 3",
    ),
    (
        "k_scale_dtype",
        _with(k_scale=torch.zeros(4, 16, 2, dtype=torch.float16)),
        "k_scale must be float32, got float16",
    ),
    (
        "v_scale_block_size",
        _with(v_scale=torch.zeros(4, 8, 2)),
        "v_scale.size(1) must equal BS = 16, got 8",
    ),
    (
        "hq_not_multiple",
        _with(**_q_and_out((8, 13, 256))),
        "Hq = 13 must be a positive multiple of Hkv = 2",
    ),
    ("wm_value", _with(wm=3), "wm must be 1, 2 or 4, got 3"),
    (
        "group_over_64",
        _with(
            **_q_and_out((8, 130, 256)),
            wm=4,
            ws_o=torch.empty(34, 8, 130, 256),
            ws_lse=torch.empty(34, 8, 130),
        ),
        "G = Hq/Hkv = 65 exceeds the supported maximum 64",
    ),
    (
        "group_over_16wm",
        _with(**_q_and_out((8, 48, 256)), wm=1),
        "G = Hq/Hkv = 24 must be <= 16*wm = 16",
    ),
    (
        "block_table_dtype",
        _with(block_table=torch.zeros(2, 4, dtype=torch.int64)),
        "block_table must be int32, got int64",
    ),
    (
        "seq_lens_rank",
        _with(seq_lens=torch.zeros(2, 1, dtype=torch.int32)),
        "seq_lens must be 1-D, got 2-D",
    ),
    (
        "qsl_length",
        _with(query_start_loc=torch.zeros(4, dtype=torch.int32)),
        "query_start_loc.size(0) must equal num_reqs + 1 = 3, got 4",
    ),
    (
        "block_table_rows",
        _with(block_table=torch.zeros(1, 4, dtype=torch.int32)),
        "block_table has 1 rows, fewer than num_reqs = 2",
    ),
    (
        "max_query_len",
        _with(max_query_len=9),
        "max_query_len must be in [0, T = 8], got 9",
    ),
    ("num_splits_zero", _with(num_splits=0), "num_splits must be >= 1, got 0"),
    (
        "scale_nan",
        _with(softmax_scale=float("nan")),
        "softmax_scale must be finite and > 0, got nan",
    ),
    (
        "scale_negative",
        _with(softmax_scale=-1.0),
        "softmax_scale must be finite and > 0, got -1",
    ),
    (
        "scale_zero",
        _with(softmax_scale=0.0),
        "softmax_scale must be finite and > 0, got 0",
    ),
    (
        "ws_o_too_few_splits",
        _with(ws_o=torch.empty(33, 8, 12, 256)),
        (
            "ws_o must have size(0) >= num_splits = 34 and size(1) >= T = 8, "
            "got [33, 8, ...]"
        ),
    ),
    (
        "ws_o_dtype",
        _with(ws_o=torch.empty(34, 8, 12, 256, dtype=torch.bfloat16)),
        "ws_o must be float32, got bfloat16",
    ),
    (
        "ws_lse_heads",
        _with(ws_lse=torch.empty(34, 8, 6)),
        "ws_lse.size(2) must equal Hq = 12, got 6",
    ),
    (
        "window_negative",
        _with(window=-1),
        "window must be >= 0, got -1",
    ),
    (
        "window_at_d256",
        _with(window=2048),
        f"{NEEDS_D128}, got 256",
    ),
    (
        "non_causal_at_d256",
        _with(causal=False),
        f"{NEEDS_D128}, got 256",
    ),
    (
        "window_at_d64",
        lambda: make_args(D=64, G=4, Hkv=2, window=64),
        f"{NEEDS_D128}, got 64",
    ),
    (
        "non_causal_at_d64",
        lambda: make_args(D=64, G=4, Hkv=2, causal=False),
        f"{NEEDS_D128}, got 64",
    ),
    # The drafter's launch at TP=2 and TP=1: D = 128,
    # G = 4, q = 8, window 2048, non-causal; a causal window and a non-causal
    # call without window at D = 128; causal without window at D = 64.
    ("valid_drafter_tp2", lambda: make_args(**DRAFTER, Hkv=4), None),
    ("valid_drafter_tp1", lambda: make_args(**DRAFTER, Hkv=8), None),
    (
        "valid_causal_window_d128",
        lambda: make_args(D=128, G=4, Hkv=2, window=64),
        None,
    ),
    (
        "valid_non_causal_no_window_d128",
        lambda: make_args(D=128, G=4, Hkv=2, causal=False),
        None,
    ),
    ("valid_causal_d64", lambda: make_args(D=64, G=4, Hkv=2), None),
]


def _to_fake(args: dict[str, Any], mode: FakeTensorMode) -> list[Any]:
    # from_tensor keeps sizes, strides and dtype exactly; the data is dropped.
    return [
        mode.from_tensor(v) if isinstance(v, torch.Tensor) else v for v in args.values()
    ]


def _fake_error(fake_fn: Callable[..., None], args: dict[str, Any]) -> str | None:
    with FakeTensorMode() as mode:
        fake_args = _to_fake(args, mode)
        try:
            result = fake_fn(*fake_args)
        except RuntimeError as e:
            return str(e)
    assert result is None
    return None


@pytest.mark.parametrize(
    ("case", "build", "expected"), CASES, ids=[c[0] for c in CASES]
)
def test_fake_checks_metadata_like_cpp(
    case: str, build: Callable[[], dict[str, Any]], expected: str | None
) -> None:
    """Under FakeTensorMode the registered fake accepts valid calls (returning
    None) and rejects each invalid one with the C++ check's message.

    Why: torch.compile runs only the fake while tracing. If the fake accepted
    a call the kernel rejects, the error would surface later, at run time, far
    from its cause; if it rejected a valid call, compilation would break.
    Runs through a private-namespace op whose only implementation is the fake.
    """
    err = _fake_error(TEST_OP, build())
    if expected is None:
        assert err is None, err
    else:
        assert err == f"pth_flash_attn_fwd: {expected}"


def assert_fake_gives_expected(fake_fn: Callable[..., None]) -> None:
    """Raise AssertionError unless fake_fn gives every case's expected outcome."""
    for case, build, expected in CASES:
        err = _fake_error(fake_fn, build())
        want = None if expected is None else f"pth_flash_attn_fwd: {expected}"
        assert err == want, (case, err)


def _fake_ignoring_causal(*args: Any) -> None:
    # Mutant: the fake as if every call were causal (the D = 256 non-causal
    # refusal lost; the window part of the same check stays).
    named = dict(zip(ARG_NAMES, args, strict=True))
    named["causal"] = True
    ops.pth_flash_attn_fwd_fake(*named.values())


def test_negative_control_fake_ignoring_causal_is_caught() -> None:
    """Negative control: a fake that drops the non-causal D = 256 check fails.

    Why: shows the case list has teeth for the new arguments without the C++
    build (the pairwise C++ comparison below needs ``_C.abi3.so``): a fake that
    let a non-causal D = 256 call through would let torch.compile trace a call
    the kernel refuses.
    """
    assert_fake_gives_expected(ops.pth_flash_attn_fwd_fake)
    with pytest.raises(AssertionError):
        assert_fake_gives_expected(_fake_ignoring_causal)


def test_fake_needs_no_data() -> None:
    """The fake runs on meta tensors, which have no storage at all.

    Why: a fake that read a value (e.g. seq_lens[0]) would fail under
    torch.compile's FakeTensors, or worse, specialise the graph on one value.
    """
    meta = {
        k: (
            torch.empty_strided(v.shape, v.stride(), dtype=v.dtype, device="meta")
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in make_args().items()
    }
    assert TEST_OP(*meta.values()) is None


def test_schema_is_the_documented_schema() -> None:
    """ops.OP_SCHEMA, docs/api.md and the C++ m.def() text are one schema.

    Why: the schema is a frozen interface. Three copies exist (documentation,
    C++, Python); comparing parsed schemas catches any drift in a name, type,
    order or mutability annotation. Exactly out, ws_o and ws_lse are written
    (out-variant), and the op returns nothing.
    """
    canonical = _parse(ops.OP_SCHEMA)
    assert _parse(_schema_from_api_doc()) == canonical
    if BINDINGS.is_file():
        assert _parse(_schema_from_bindings()) == canonical
    schema = torch._C.parse_schema(ops.OP_SCHEMA)
    written = [
        a.name
        for a in schema.arguments
        if a.alias_info is not None and a.alias_info.is_write
    ]
    assert written == ["out", "ws_o", "ws_lse"]
    assert len(schema.returns) == 0
    # The window/causal tail: host constants, graph-safe, with the
    # causal defaults so calls written before them still resolve.
    tail = [(a.name, str(a.type), a.default_value) for a in schema.arguments[-2:]]
    assert tail == [("window", "int", 0), ("causal", "bool", True)]


# A private namespace with the same schema and a CPU kernel that records the
# arguments the dispatcher hands it after the first 15. The dispatcher passes
# a Python kernel no trailing argument that equals its schema default.
_REC_NS = "pth_flash_attn_defaultstest"
_REC_LIB = torch.library.Library(_REC_NS, "DEF")  # kept alive for the module
_REC_LIB.define(ops.OP_SCHEMA)
_RECORDED: list[tuple[Any, ...]] = []
_REC_LIB.impl(ops.OP_NAME, lambda *args: _RECORDED.append(args[15:]), "CPU")
REC_OP = getattr(getattr(torch.ops, _REC_NS), ops.OP_NAME).default


def _legacy_args() -> list[Any]:
    """The 15 arguments of a call written before window/causal existed."""
    return list(make_args().values())[:-2]


def test_legacy_15_argument_call_resolves_as_causal_without_window() -> None:
    """A call with the old 15 arguments is window = 0, causal = True.

    Why: callers written against the 15-argument form of the schema must
    keep working. The schema defaults let such a call run the causal kernel
    unchanged instead of failing at re-vendor. Checked at the dispatcher (a
    recording kernel: the legacy call and an explicit (0, True) arrive the
    same way, (0, True) being the schema's defaults, while (2048, False)
    arrives as given), through the fake under FakeTensorMode (a D = 256 call:
    any other default would be refused) and through the Python defaults.
    """
    legacy = _legacy_args()
    assert len(legacy) == 15
    _RECORDED.clear()
    for tail in ((), (0, True), (2048, False)):
        assert REC_OP(*legacy, *tail) is None
    assert _RECORDED == [(), (), (2048, False)]
    assert legacy[1].size(2) == 256  # q is D = 256: windowed calls are refused
    assert _fake_error(TEST_OP, dict(zip(ARG_NAMES, legacy, strict=False))) is None
    assert ops.pth_flash_attn_fwd_fake(*legacy) is None


@needs_op_library
def test_legacy_15_argument_call_reaches_the_op() -> None:
    """The built op accepts the 15-argument call and ends in its CPU-only error."""
    with pytest.raises(RuntimeError, match=CPU_ONLY_ERROR):
        ops.pth_flash_attn_fwd(*_legacy_args())
    op = torch.ops.pth_flash_attn.pth_flash_attn_fwd.default
    with pytest.raises(RuntimeError, match=CPU_ONLY_ERROR):
        op(*_legacy_args())


@needs_op_library
def test_loaded_op_has_the_frozen_schema() -> None:
    """The library the wrapper calls registers exactly the frozen schema."""
    op = torch.ops.pth_flash_attn.pth_flash_attn_fwd.default
    assert str(op._schema) == _parse(ops.OP_SCHEMA)


def _cpp_error(args: dict[str, Any]) -> str:
    with pytest.raises(RuntimeError) as info:
        ops.pth_flash_attn_fwd(*args.values())
    return str(info.value)


def assert_fake_matches_cpp(fake_fn: Callable[..., None]) -> None:
    """Raise AssertionError unless fake_fn and the C++ checks agree on every case.

    Valid arguments: the fake returns None and the op raises its CPU-only error
    (all checks passed). Invalid arguments: both raise, and the fake's message
    appears verbatim in the op's (which adds C++ location context around it).
    """
    for case, build, expected in CASES:
        args = build()
        fake_err = _fake_error(fake_fn, args)
        real_err = _cpp_error(args)
        if expected is None:
            assert fake_err is None, (case, fake_err)
            assert CPU_ONLY_ERROR in real_err, (case, real_err)
        else:
            assert fake_err is not None, (case, "fake accepted an invalid call")
            assert fake_err in real_err, (case, fake_err, real_err)


@needs_op_library
def test_fake_matches_cpp_case_by_case() -> None:
    """Fake and C++ checks agree on every case: same accept/reject, same message.

    Why: the fake is a hand-written mirror of C++ code. Only a pairwise
    comparison against the compiled checks shows the mirror is faithful.
    """
    assert_fake_matches_cpp(ops.pth_flash_attn_fwd_fake)


def _fake_missing_wm_check(*args: Any) -> None:
    # Mutant: the fake with the `wm in {1, 2, 4}` check deleted.
    named = dict(zip(ARG_NAMES, args, strict=True))
    if named["wm"] == 3:
        named["wm"] = 4  # behave as if any wm were acceptable
    ops.pth_flash_attn_fwd_fake(*named.values())


@needs_op_library
def test_negative_control_drifted_fake_is_caught() -> None:
    """Negative control: a fake missing one C++ check fails the parity test.

    Why: shows the pairwise comparison has teeth; a fake that silently lost a
    check (here: wm validation) must not pass as "matching the C++ checks".
    """
    with pytest.raises(AssertionError):
        assert_fake_matches_cpp(_fake_missing_wm_check)


@needs_op_library
def test_valid_cpu_call_refuses_to_compute() -> None:
    """A valid call on CPU tensors ends in the CPU-only error, not in silence.

    Why: the CPU registration exists to test the checks. A valid call must end
    in the specific error, not return (which would leave ``out`` unwritten and
    look like a result) and not fail a check (which would mean the test
    arguments, i.e. our model of a valid call, are wrong).
    """
    args = make_args()
    assert CPU_ONLY_ERROR in _cpp_error(args)
    assert not args["out"].any(), "a CPU call must not write out"


@needs_op_library
def test_opcheck_on_cpu_tensors() -> None:
    """torch.library.opcheck on CPU tensors: the checks that only need the fake
    pass; the checks that execute the real op fail with exactly the CPU error.

    Why: opcheck is the standard registration test. test_schema,
    test_faketensor and test_aot_dispatch_dynamic run the real op, which on CPU
    can only report its CPU-only error; any other failure (schema mismatch,
    missing or wrong fake) would be a real registration bug.
    test_autograd_registration passes: no input requires grad and the op
    declares no autograd formula (inference-only). The same opcheck on CUDA
    tensors, where the op computes, is tests/gpu/test_backend_op.py.
    """
    op = torch.ops.pth_flash_attn.pth_flash_attn_fwd.default
    results = torch.library.opcheck(
        op, tuple(make_args().values()), raise_exception=False
    )
    assert results["test_autograd_registration"] == "SUCCESS"
    for name in ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"):
        err = results[name]
        assert isinstance(err, Exception), (name, err)
        chain, e = [], err
        while e is not None:
            chain.append(str(e))
            e = e.__cause__ or e.__context__
        assert any(CPU_ONLY_ERROR in m for m in chain), (name, chain)
    # The fake alone, through the real op's registration: no data, no error.
    with FakeTensorMode() as mode:
        assert op(*_to_fake(make_args(), mode)) is None


def test_ops_module_makes_no_cuda_call() -> None:
    """ops.py never calls torch.cuda and importing it initialises no GPU.

    Why: vLLM-facing code must ask ``current_platform``, and an import-time
    CUDA initialisation would break forked workers and CPU-only CI.
    """
    tree = ast.parse(Path(ops.__file__).read_text())
    assert not any(
        isinstance(n, ast.Attribute) and n.attr == "cuda" for n in ast.walk(tree)
    )
    probe = "import torch, pth_flash_attn.ops\nassert not torch.cuda.is_initialized()\n"
    subprocess.run([sys.executable, "-c", probe], check=True)
