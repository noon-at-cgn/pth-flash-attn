# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""bench/e2e: the salt wrapper, the logprob tool and ladder.sh.

The logprob tool is tested through its pure functions, ``--help`` and the absence of a
key file.
"""

import builtins
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

E2E = Path(__file__).resolve().parents[2] / "bench" / "e2e"
WRAPPER = E2E / "ladder_salt.py"
LOGPROB = E2E / "prompt_logprob.py"
LADDER_SH = E2E / "ladder.sh"


def _load_logprob():
    spec = importlib.util.spec_from_file_location("prompt_logprob_under_test", LOGPROB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_wrapped_script_reads_the_pinned_epoch_and_its_own_arguments(tmp_path):
    script = tmp_path / "probe.py"
    script.write_text(
        "import sys, time\nprint(int(time.time()), sys.argv[1:], __name__)\n"
    )
    out = subprocess.run(
        [sys.executable, str(WRAPPER), "1760000000", str(script), "--label", "8kr1"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert out.strip() == "1760000000 ['--label', '8kr1'] __main__"


def test_a_non_integer_epoch_is_a_usage_error(tmp_path):
    script = tmp_path / "probe.py"
    script.write_text("print('ran')\n")
    run = subprocess.run(
        [sys.executable, str(WRAPPER), "now", str(script)],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 1
    assert "usage" in run.stderr
    assert "ran" not in run.stdout


def test_the_wrapped_script_can_import_modules_next_to_it(tmp_path):
    (tmp_path / "sibling.py").write_text("VALUE = 41\n")
    script = tmp_path / "probe.py"
    script.write_text("import sibling\nprint(sibling.VALUE + 1)\n")
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    out = subprocess.run(
        [sys.executable, str(WRAPPER), "1760000000", str(script)],
        capture_output=True,
        text=True,
        check=True,
        cwd=other_cwd,
    ).stdout
    assert out.strip() == "42"


# ---- prompt_logprob.py: command line -------------------------------------------------


def test_logprob_help_exits_zero_and_names_every_subcommand():
    run = subprocess.run(
        [sys.executable, str(LOGPROB), "--help"], capture_output=True, text=True
    )
    assert run.returncode == 0
    for name in ("doc", "dump", "compare", "refcompare"):
        assert name in run.stdout
    for sub, option in (
        ("doc", "--source"),
        ("dump", "--api-key-env"),
        ("compare", "--tail"),
        ("refcompare", "--blocks"),
    ):
        sub_help = subprocess.run(
            [sys.executable, str(LOGPROB), sub, "--help"],
            capture_output=True,
            text=True,
        )
        assert sub_help.returncode == 0
        assert option in sub_help.stdout


def test_logprob_never_opens_a_key_file_on_import_or_help(monkeypatch):
    opened = []
    real_open = builtins.open

    def recording_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", recording_open)
    mod = _load_logprob()
    with pytest.raises(SystemExit) as exc:
        mod.main(["--help"])
    assert exc.value.code == 0
    assert opened == []
    assert (
        "open(" not in LOGPROB.read_text().split("def post")[0]
    )  # nothing is read at module level


def test_api_key_comes_only_from_the_environment(monkeypatch):
    mod = _load_logprob()
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("MY_KEY", raising=False)
    assert mod.api_key(None) is None
    monkeypatch.setenv("VLLM_API_KEY", "abc")
    assert mod.api_key(None) == "abc"
    monkeypatch.setenv("MY_KEY", "xyz")
    assert mod.api_key("MY_KEY") == "xyz"
    monkeypatch.delenv("MY_KEY")
    with pytest.raises(SystemExit):
        mod.api_key("MY_KEY")


def test_doc_is_built_from_the_given_sources_in_sorted_order(tmp_path):
    mod = _load_logprob()
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "b.md").write_text("beta " * 10)
    (src / "a.txt").write_text("alpha " * 10)
    (src / "sub" / "c.md").write_text("gamma " * 10)
    (src / "skip.py").write_text("ignored\n")
    single = tmp_path / "one.rst"
    single.write_text("delta " * 10)
    doc = mod.build_doc(70, [str(single), str(src)])
    heads = [line for line in doc.splitlines() if line.startswith("=== ")]
    assert heads == [
        "=== one.rst ===",
        "=== src/a.txt ===",
        "=== src/b.md ===",
        "=== src/sub/c.md ===",
    ]
    assert "ignored" not in doc
    assert str(tmp_path) not in doc
    assert len(doc) == int(70 * 4.2)
    with pytest.raises(SystemExit):
        mod.build_doc(100000, [str(src)])


def test_doc_skips_a_file_whose_content_repeats_an_earlier_one(tmp_path):
    mod = _load_logprob()
    src = tmp_path / "s"
    src.mkdir()
    (src / "a.md").write_text("same text " * 3)
    (src / "b.md").write_text("same text " * 3)
    (src / "c.md").write_text("x" * 300)
    doc = mod.build_doc(40, [str(src)])
    assert [line for line in doc.splitlines() if line.startswith("=== ")] == [
        "=== s/a.md ===",
        "=== s/c.md ===",
    ]


def test_default_doc_source_is_the_docs_directory_of_this_repository():
    mod = _load_logprob()
    assert [Path(p) for p in mod.DEFAULT_SOURCES] == [
        Path(__file__).resolve().parents[2] / "docs"
    ]


# ---- prompt_logprob.py: statistics (expectations worked out by hand) -----------------


def test_block_bootstrap_of_a_constant_sample_is_that_constant():
    mod = _load_logprob()
    assert mod.block_boot_mean([0.5] * 1000, 100) == (0.5, 0.5)
    assert mod.block_boot_mean([1.0, 2.0, 3.0, 4.0], 10) == (2.5, 2.5)  # a single block


def test_block_bootstrap_is_deterministic_for_its_fixed_seed():
    mod = _load_logprob()
    sample = [0.1 + 0.001 * (i % 7) for i in range(700)]
    first = mod.block_boot_mean(sample, 64)
    assert mod.block_boot_mean(sample, 64) == first
    mean = sum(sample) / len(sample)
    assert first[0] <= mean <= first[1]
    assert first[1] - first[0] < 0.001


def test_block_bootstrap_of_two_unequal_blocks_spans_both_blocks():
    # Blocks A = fifty 0.0 and B = fifty 1.0. A resample of two blocks has mean 0, 0.5
    # or 1 with probabilities 1/4, 1/2, 1/4, so over 4000 resamples the 2.5th and 97.5th
    # percentiles are 0 and 1.
    mod = _load_logprob()
    assert mod.block_boot_mean([0.0] * 50 + [1.0] * 50, 50) == (0.0, 1.0)


def test_coarse_kl_on_a_hand_worked_pair():
    mod = _load_logprob()
    # p: tokens 1, 2 with probabilities 0.5, 0.3 (rest 0.2); q: 0.25, 0.25 plus token 3
    # (rest 0.5).
    p = [[1, math.log(0.5)], [2, math.log(0.3)]]
    q = [[1, math.log(0.25)], [2, math.log(0.25)], [3, math.log(0.25)]]
    kl_pq, kl_qp, common = mod.coarse_kl(p, q)
    # Partition {1}, {2}, rest; the rest of p is 0.2, the rest of q is 1 - 0.5 = 0.5
    # (token 3 is not shared).
    want_pq = (
        0.5 * math.log(0.5 / 0.25)
        + 0.3 * math.log(0.3 / 0.25)
        + 0.2 * math.log(0.2 / 0.5)
    )
    want_qp = (
        0.25 * math.log(0.25 / 0.5)
        + 0.25 * math.log(0.25 / 0.3)
        + 0.5 * math.log(0.5 / 0.2)
    )
    assert common == 2
    assert kl_pq == pytest.approx(want_pq, abs=1e-12)
    assert kl_qp == pytest.approx(want_qp, abs=1e-12)
    assert want_pq == pytest.approx(
        0.21801, abs=1e-4
    )  # 0.346574 + 0.054696 - 0.183258, evaluated by hand


def test_coarse_kl_without_shared_tokens_is_zero():
    mod = _load_logprob()
    kl_pq, kl_qp, common = mod.coarse_kl([[1, math.log(0.5)]], [[2, math.log(0.5)]])
    # one cell, "rest", holding all the mass on both sides
    assert (kl_pq, kl_qp, common) == (0.0, 0.0, 0)


def test_pair_metrics_on_tiny_dumps_with_strict_ids():
    mod = _load_logprob()
    x = {
        "actual": [None, -1.0, -2.0, -0.5, -3.0],
        "top1": [None, 7, 8, 9, 10],
        "top1lp": [None, -0.5, -1.0, -1.5, -2.0],
    }
    y = {
        "actual": [None, -1.5, -2.0, -0.25, -3.0],
        "top1": [None, 7, 8, 5, 10],
        "top1lp": [None, -0.5, -1.0, -1.5, -2.5],
    }
    r = mod.pair_metrics(x, y, 1, 5)
    # |delta| over positions 1..4 = 0.5, 0, 0.25, 0 -> sorted 0, 0, 0.25, 0.5
    assert r["positions"] == 4
    assert r["mean_abs_delta"] == 0.1875
    assert r["median_abs_delta"] == 0.25  # element 2 of 4
    assert r["p99_abs_delta"] == 0.5  # element int(0.99 * 4) = 3
    assert r["max_abs_delta"] == 0.5
    assert r["top1_agreement"] == 0.75  # ids differ at position 3 only
    assert r["top1_agreement_strict_ids"] == 0.75
    assert r["top1_tie_aware"] is False
    assert (
        r["ids_differ_at_equal_top1_logprob"] == 1
    )  # position 3: 9 vs 5 at the same logprob
    assert r["mean_abs_delta_top1_logprob"] == 0.125  # only position 4 differs, by 0.5
    assert "kl_lower_bound" not in r


def test_pair_metrics_is_tie_aware_when_both_dumps_have_top_k_lists():
    mod = _load_logprob()
    x = {
        "actual": [None, -1.0, -2.0, -0.5, -3.0],
        "top1": [None, 7, 8, 9, 10],
        "top1lp": [None, -0.5, -1.0, -1.5, -2.0],
        "top": 3,
        "topk": [
            None,
            [[7, -0.5], [3, -2.0]],
            [[8, -1.0]],
            [[9, -1.5], [5, -1.5]],
            [[10, -2.0]],
        ],
    }
    y = {
        "actual": [None, -1.5, -2.0, -0.25, -3.0],
        "top1": [None, 7, 8, 5, 10],
        "top1lp": [None, -0.5, -1.0, -1.5, -2.0],
        "top": 2,
        "topk": [None, [[7, -0.5]], [[8, -1.0]], [[5, -1.5], [9, -1.5]], [[10, -2.0]]],
    }
    r = mod.pair_metrics(x, y, 1, 5)
    assert r["top1_tie_aware"] is True
    assert r["top1_agreement"] == 1.0  # the sets of maximal tokens intersect everywhere
    assert r["top1_agreement_strict_ids"] == 0.75
    kl = r["kl_lower_bound"]
    assert kl["top"] == 2
    assert (
        kl["mean_common_tokens"] == 1.25
    )  # common top-K tokens per position: 1, 1, 2, 1
    # the shared tokens carry identical probabilities on both sides, so the coarse KL is
    # exactly zero
    assert kl["mean_kl_a_b"] == 0.0
    assert kl["mean_kl_b_a"] == 0.0


def _dump(path, ref_actual, deltas):
    ids = list(range(len(ref_actual)))
    actual = [None] + [a + d for a, d in zip(ref_actual[1:], deltas[1:])]
    path.write_text(
        json.dumps(
            {
                "ids": ids,
                "actual": actual,
                "top1": [None] + [1] * (len(ids) - 1),
                "top1lp": [None] + [-1.0] * (len(ids) - 1),
                "top": 1,
            }
        )
    )


@pytest.mark.parametrize(
    "new_delta, verdict, exit_code, paired, ci_ok",
    [
        (
            0.5,
            "PASS",
            0,
            0.0,
            True,
        ),  # as far from the reference as the old arm: difference 0
        (
            0.625,
            "FAIL",
            1,
            0.125,
            False,
        ),  # 0.625 > 0.5 + max(0.05, 0.002) and the difference 0.125 > margin
    ],
)
def test_refcompare_verdict_on_tiny_dumps(
    tmp_path, capsys, new_delta, verdict, exit_code, paired, ci_ok
):
    mod = _load_logprob()
    ref_actual = [None, -1.0, -2.0, -3.0, -4.0, -5.0]
    ref, old, new = tmp_path / "ref.json", tmp_path / "old.json", tmp_path / "new.json"
    _dump(ref, ref_actual, [None] + [0.0] * 5)
    _dump(old, ref_actual, [None] + [0.5] * 5)
    _dump(new, ref_actual, [None] + [new_delta] * 5)
    code = mod.main(
        ["refcompare", str(new), str(old), str(ref), "--tail", "3", "--blocks", "2"]
    )
    rep = json.loads(capsys.readouterr().out)
    assert code == exit_code
    assert rep["verdict_noninferior_point"] == verdict
    assert rep["all"]["new_vs_ref"]["positions"] == 5
    assert rep["tail"]["new_vs_ref"]["positions"] == 3  # --tail 3 -> positions 3, 4, 5
    for row in ("all", "tail"):
        assert rep[row]["noninferiority_margin"] == pytest.approx(
            0.05
        )  # 10 % of 0.5 beats the 0.002 floor
        assert rep[row]["paired_mean_diff_new_minus_old"] == paired
        assert rep[row]["paired_diff_ci95_block2"] == [
            paired,
            paired,
        ]  # every difference is the same
        assert rep[row]["noninferior_ci_upper_block2"] is ci_ok


def test_compare_passes_identical_dumps_and_fails_a_shifted_one(tmp_path, capsys):
    mod = _load_logprob()
    actual = [None, -1.0, -2.0, -3.0, -4.0, -5.0]
    a, b, c = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    _dump(a, actual, [None] + [0.0] * 5)
    _dump(b, actual, [None] + [0.0] * 5)
    _dump(c, actual, [None] + [0.25] * 5)  # mean |delta| 0.25 > 0.02 nats
    assert mod.main(["compare", str(a), str(b), "--tail", "3"]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "PASS"
    assert mod.main(["compare", str(a), str(c), "--tail", "3"]) == 1
    assert json.loads(capsys.readouterr().out)["verdict"] == "FAIL"


# ---- ladder.sh -----------------------------------------------------------------------

BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(BASH is None, reason="bash not available")


def _clean_env(**extra):
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "PORT",
            "PYTHON",
            "SALT_EPOCH",
            "OUT_DIR",
            "HYPERQWEN_DIR",
            "REPS_8K",
            "REPS_32K",
            "REPS_128K",
        }
    }
    env.update(extra)
    return env


@needs_bash
def test_ladder_sh_dry_run_prints_every_command(tmp_path):
    (tmp_path / "hq" / "bench").mkdir(parents=True)
    ladder_py = tmp_path / "hq" / "bench" / "conc_ladder.py"
    ladder_py.write_text("")
    out_dir = tmp_path / "out"
    run = subprocess.run(
        [BASH, str(LADDER_SH), "--dry-run"],
        capture_output=True,
        text=True,
        env=_clean_env(
            HYPERQWEN_DIR=str(tmp_path / "hq"), PORT="8123", OUT_DIR=str(out_dir)
        ),
    )
    assert run.returncode == 0, run.stderr
    lines = run.stdout.splitlines()
    assert len(lines) == 5 + 3 + 3
    salt_py = E2E / "ladder_salt.py"

    def expected(depth, rep, tokens):
        base = f"{out_dir}/conc_ladder_{depth}_r{rep}"
        return (
            f"PORT=8123 python3 {salt_py} 1791800000 {ladder_py} "
            f"--label {depth}r{rep} --n 1,2 --out 256 "
            f"--reps 1 --ctx-tokens {tokens} --json {base}.json > {base}.log 2>&1"
        )

    want = (
        [expected("8k", r, 8192) for r in range(1, 6)]
        + [expected("32k", r, 32000) for r in range(1, 4)]
        + [expected("128k", r, 128000) for r in range(1, 4)]
    )
    assert lines == want
    assert not out_dir.exists()  # a dry run creates nothing


@needs_bash
def test_ladder_sh_dry_run_honours_depth_arguments_and_overrides(tmp_path):
    (tmp_path / "hq" / "bench").mkdir(parents=True)
    (tmp_path / "hq" / "bench" / "conc_ladder.py").write_text("")
    run = subprocess.run(
        [
            BASH,
            str(LADDER_SH),
            "--dry-run",
            "--salt-epoch",
            "42",
            "--port",
            "9000",
            "--out-dir",
            str(tmp_path / "o"),
            "128k",
        ],
        capture_output=True,
        text=True,
        env=_clean_env(HYPERQWEN_DIR=str(tmp_path / "hq"), REPS_128K="2"),
    )
    assert run.returncode == 0, run.stderr
    lines = run.stdout.splitlines()
    assert len(lines) == 2
    assert all(
        ln.startswith("PORT=9000 python3 ")
        and " 42 " in ln
        and "--ctx-tokens 128000" in ln
        for ln in lines
    )
    assert "--label 128kr2" in lines[1]


@needs_bash
def test_ladder_sh_reports_a_missing_prerequisite(tmp_path):
    missing = subprocess.run(
        [BASH, str(LADDER_SH), "--dry-run"],
        capture_output=True,
        text=True,
        env=_clean_env(),
    )
    assert missing.returncode == 64
    assert "HYPERQWEN_DIR" in missing.stderr
    empty = subprocess.run(
        [BASH, str(LADDER_SH), "--dry-run"],
        capture_output=True,
        text=True,
        env=_clean_env(HYPERQWEN_DIR=str(tmp_path)),
    )
    assert empty.returncode == 64
    assert "conc_ladder.py" in empty.stderr


@needs_bash
def test_ladder_sh_help_exits_zero():
    run = subprocess.run(
        [BASH, str(LADDER_SH), "--help"],
        capture_output=True,
        text=True,
        env=_clean_env(),
    )
    assert run.returncode == 0
    for word in ("HYPERQWEN_DIR", "SALT_EPOCH", "--dry-run", "PORT"):
        assert word in run.stdout
