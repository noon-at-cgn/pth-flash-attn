# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Simulator equality for the primitives (ptx, dequant, smem_layout).

csrc/pth_attn/{ptx,dequant,smem_layout}.cuh are compiled twice:
- with g++ (tests/cpp/test_primitives.cu, host mode), which evaluates their
  bit and index math and prints it in the shape of the simulator's golden
  tables; every key is compared with tests/sim/ptxsim/golden.py;
- with nvcc for sm_80/86/89 (device mode), through the ptxas gate and the SASS
  check, and the cvt4 instruction counts are compared with the documented claim.

Build commands live in tests/cpp/run_primitives.sh. Needs g++ and nvcc (the
development image); skipped elsewhere.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests" / "sim"))

from ptxsim.golden import golden_tables  # noqa: E402

RUNNER = REPO / "tests" / "cpp" / "run_primitives.sh"
NVCC = Path("/usr/local/cuda/bin/nvcc")

# Golden keys the C++ side deliberately does not produce, and why.
EXCLUDED = {
    "format_version": "metadata of the golden file",
    "magic.lop3_lut_xor": "the XOR is plain C; its LUT 0x3c is checked in SASS",
    "k_ldmatrix_x4_lane": "pre-swizzle coordinates; per_d.*.k_ldsm has the offsets",
    "q_ldmatrix_x4_lane": "pre-swizzle coordinates; per_d.*.q_ldsm has the offsets",
    "v_lds32_lane": "pre-swizzle coordinates; per_d.*.v_lds32 has the offsets",
    "per_d.*.loader": "loader lane policy belongs to the loader, not here",
}

needs_gxx = pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")
needs_nvcc = pytest.mark.skipif(not NVCC.exists(), reason="needs nvcc")


def _excluded(path: str) -> bool:
    parts = path.split(".")
    for pat in EXCLUDED:
        pp = pat.split(".")
        if len(pp) == len(parts) and all(a in ("*", b) for a, b in zip(pp, parts)):
            return True
    return False


def compare(cpp, gold, path: str = "") -> list[str]:
    """Mismatches between the C++ output and the golden tables, as readable
    strings. Dicts must have the same keys (minus EXCLUDED); lists must be
    equal element by element; for long hex strings the first differing value
    is reported."""
    if isinstance(gold, dict):
        if not isinstance(cpp, dict):
            return [f"{path}: expected an object"]
        out = []
        for k in sorted(set(gold) | set(cpp)):
            p = f"{path}.{k}" if path else k
            if _excluded(p):
                continue
            if k not in cpp:
                out.append(f"{p}: missing in C++ output")
            elif k not in gold:
                out.append(f"{p}: not in golden tables")
            else:
                out += compare(cpp[k], gold[k], p)
        return out
    if isinstance(gold, str) and isinstance(cpp, str):
        if cpp == gold:
            return []
        n = min(len(cpp), len(gold))
        i = next((i for i in range(0, n, 4) if cpp[i : i + 4] != gold[i : i + 4]), n)
        a, b = cpp[i : i + 4], gold[i : i + 4]
        return [f"{path}: first difference at value {i // 4}: C++ {a!r} vs {b!r}"]
    if cpp != gold:
        return [f"{path}: C++ {str(cpp)[:120]} != golden {str(gold)[:120]}"]
    return []


def _build_and_run(out_dir: Path, include: Path | None = None) -> dict:
    cmd = ["bash", str(RUNNER), "host", str(out_dir)]
    if include is not None:
        cmd.append(str(include))
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads((out_dir / "primitives.json").read_text())


@pytest.fixture(scope="module")
def golden():
    return json.loads(json.dumps(golden_tables()))  # JSON types, like the C++ side


@needs_gxx
def test_headers_reproduce_simulator_golden_tables(tmp_path, golden):
    """The CUDA headers compute exactly what the simulator proved: all 256
    cvt4 values (and every value in every byte position), the V fragment
    words, pi and its inverse, the K/V/Q swizzle chunk maps, the Q element
    offsets, the per-lane ldmatrix/LDS.32 offsets, the d-permutation and
    epilogue columns, stage offsets for D = 64/128/256, the selector
    constants, and the Q prescale of every finite bf16 and fp16 value with its
    own and eight fixed exponents.

    Why it matters: the simulator is where the layouts and arithmetic were
    proved; this is the link that makes those proofs
    statements about the code the kernel compiles. Any difference is a bug in
    the header (or in the host shim, which would also need fixing).
    """
    cpp = _build_and_run(tmp_path)
    mismatches = compare(cpp, golden)
    assert mismatches == [], "\n".join(mismatches[:20])


MUTATIONS = {
    # id: (header, [(original text, mutated text), ...], golden key that must differ)
    "k_swizzle_mask": (
        "smem_layout.cuh",
        [("kKSwizzleMask = 7;", "kKSwizzleMask = 3;")],
        "k_chunk",
    ),
    "v_swizzle_key": (
        "smem_layout.cuh",
        [("kVSwizzleKeyScale = 2;", "kVSwizzleKeyScale = 1;")],
        "v_chunk",
    ),
    "magic_bias": (
        "dequant.cuh",
        [("kMagicBiasHalf2 = 0x64806480u;", "kMagicBiasHalf2 = 0x64006400u;")],
        "magic.cvt4",
    ),
    "v_selector": (
        "dequant.cuh",
        [("kVSelD01 = 0x5140u;", "kVSelD01 = 0x5410u;")],
        "v_b_fragments",
    ),
    # Dropping pi (both directions, so the round-trip static_assert still holds
    # and the golden comparison is what catches it).
    "pi_dropped": (
        "smem_layout.cuh",
        [
            ("return 4 * ((k & 7) >> 1) + 2 * (k >> 3) + (k & 1);", "return k;"),
            ("return 2 * (d >> 2) + 8 * ((d >> 1) & 1) + (d & 1);", "return d;"),
        ],
        "q_elem_offset",
    ),
}


@needs_gxx
@pytest.mark.parametrize("name", list(MUTATIONS))
def test_mutated_header_is_caught(tmp_path, golden, name):
    """Negative control: a changed constant in a temp copy of a header (K
    swizzle mask, V swizzle key scale, magic bias, a V selector, pi dropped)
    makes the golden comparison fail, at the expected key.

    Why it matters: proves the comparison above is sensitive to exactly the
    constants it is meant to protect, so its passing means something.
    """
    header, edits, key = MUTATIONS[name]
    inc = tmp_path / "inc"
    inc.mkdir()
    for h in ("ptx.cuh", "dequant.cuh", "smem_layout.cuh"):
        shutil.copy(REPO / "csrc" / "pth_attn" / h, inc / h)
    text = (inc / header).read_text()
    for old, new in edits:
        assert text.count(old) == 1, f"mutation anchor not unique in {header}: {old}"
        text = text.replace(old, new)
    (inc / header).write_text(text)
    cpp = _build_and_run(tmp_path / "out", include=inc)
    mismatches = compare(cpp, golden)
    assert mismatches, "mutation was not detected"
    assert any(key in m for m in mismatches), mismatches[:5]


def _run_device(out_dir: Path) -> subprocess.CompletedProcess:
    # The script calls `python` for the gate tools, which import torch: put
    # this interpreter's directory first on PATH so it is the one used.
    path = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
    return subprocess.run(
        ["bash", str(RUNNER), "device", str(out_dir)],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": path},
    )


@pytest.fixture(scope="module")
def device_reports(tmp_path_factory):
    out = tmp_path_factory.mktemp("device")
    proc = _run_device(out)
    return proc, out


@needs_nvcc
def test_probe_kernels_pass_ptxas_and_sass_gates(device_reports):
    """Every wrapper compiles for sm_80/86/89; the probes have 0 spill bytes
    and <= 255 registers; the SASS contains LDGSTS (cp.async), LDSM
    (ldmatrix) and HMMA (mma.sync) and no I2F/I2FP and no local memory.

    Why it matters: an inline-PTX wrapper with a wrong constraint or
    qualifier fails here, on the CPU, instead of on a GPU.
    """
    proc, out = device_reports
    assert proc.returncode == 0, proc.stdout + proc.stderr
    gate = json.loads((out / "ptxas_gate.json").read_text())
    assert {f["arch"] for f in gate["files"]} == {"sm_80", "sm_86", "sm_89"}
    for f in gate["files"]:
        assert f["pass"] and f["kernels"] >= 5, f


# Kernel (mangled name fragment) -> expected opcode counts: the conversion is
# 1 LOP3 + 2 PRMT + 2 HADD2 per 4 int8 values (HSUB2 is HADD2 with a negated
# immediate in SASS); the V fragment path adds 1 PRMT per 4 bytes for the
# token pairing: 4 words -> 4 LOP3 + (4 + 8) PRMT + 8 HADD2.
EXPECTED_COUNTS = {
    "10probe_cvt4E": {"LOP3": 1, "PRMT": 2, "HADD2": 2},
    "13probe_cvt4_x4E": {"LOP3": 4, "PRMT": 8, "HADD2": 8},
    "19probe_v_b_fragmentsE": {"LOP3": 4, "PRMT": 12, "HADD2": 8},
}
ARCHS = ("80", "86", "89")


@needs_nvcc
def test_cvt4_sass_instruction_counts(device_reports):
    """cvt4 compiles to exactly 1 LOP3 + 2 PRMT + 2 HADD2 per 4 bytes on every
    arch (and the V fragment path to 1 more PRMT per 4 bytes), with no I2F;
    the instructions are the designed ones: LOP3.LUT with 0x80808080 and
    immLut 0x3c, PRMT with 0x4140 / 0x4342, HADD2 with -1152.

    Why it matters: the conversion cost in the kernel's instruction budget
    (PRMT and HADD2 per warp-tile) assumes this sequence; a compiler that
    lowered it differently would change the inner-loop budget.
    """
    proc, out = device_reports
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rep = json.loads((out / "sass_counts.json").read_text())
    seen = 0
    for cub in rep["cubins"]:
        for k in cub["kernels"]:
            for frag, want in EXPECTED_COUNTS.items():
                if frag in k["name"]:
                    seen += 1
                    got = {op: k["opcode_counts"].get(op, 0) for op in want}
                    assert got == want, (cub["cubin"], k["name"], got)
    assert seen == len(EXPECTED_COUNTS) * len(ARCHS)

    sys.path.insert(0, str(REPO / "tools"))
    import sass_check

    cuobjdump = sass_check.find_cuobjdump()
    for arch in ARCHS:
        text = sass_check.disassemble(
            cuobjdump, out / "sass" / f"test_primitives.sm_{arch}.cubin"
        )
        kernels = sass_check.parse_sass(text)
        cvt = next(v for n, v in kernels.items() if "10probe_cvt4E" in n)
        lines = [f"{i.opcode} {i.operands}" for i in cvt]
        lop3 = [s for s in lines if s.startswith("LOP3")]
        assert len(lop3) == 1 and "0x80808080" in lop3[0] and "0x3c" in lop3[0]
        prmt = [s for s in lines if s.startswith("PRMT")]
        assert len(prmt) == 2, prmt
        assert any("0x4140" in s for s in prmt) and any("0x4342" in s for s in prmt)
        hadd = [s for s in lines if s.startswith("HADD2")]
        assert all("-1152, -1152" in s for s in hadd), hadd
