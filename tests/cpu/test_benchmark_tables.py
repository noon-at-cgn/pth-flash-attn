# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""docs/benchmarks.md is rendered from the CSV files of bench/results/.

Checked here: tables, sources, links and bold flags.
"""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "bench" / "results"
PAGE = ROOT / "docs" / "benchmarks.md"
README = ROOT / "README.md"
PR_URL = re.compile(
    r"^https://github\.com/syv-ai/HyperQwen/pull/\d+(#issuecomment-\d+)?$"
)


def _load_renderer():
    spec = importlib.util.spec_from_file_location(
        "render_tables", RESULTS / "render_tables.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


render = _load_renderer()


def _rows():  # noqa: D401
    for path in sorted(RESULTS.glob("*.csv")):
        with open(path, newline="", encoding="utf-8") as fh:
            for line, row in enumerate(csv.DictReader(fh), start=2):
                yield path.name, line, row


def _slug(heading: str) -> str:
    heading = re.sub(r"[`*_]", "", heading.strip().lower())
    return re.sub(r"\s", "-", re.sub(r"[^\w\s-]", "", heading))


def _anchors(path: Path) -> set[str]:
    return {
        _slug(m.group(2))
        for m in re.finditer(r"^(#{1,6}) (.+)$", path.read_text(encoding="utf-8"), re.M)
    }


def test_every_table_of_the_page_equals_the_rendered_one():
    have = render.page_blocks(PAGE.read_text(encoding="utf-8"))
    want = render.render_all()
    assert set(have) == set(want)
    for table_id, text in want.items():
        assert have[table_id] == text, (
            f"{table_id} differs from bench/results; run render_tables.py --write"
        )


def test_the_headline_block_is_marked_for_embedding():
    page = PAGE.read_text(encoding="utf-8")
    assert page.count("<!-- HEADLINE_TABLE:start -->") == 1
    assert page.count("<!-- HEADLINE_TABLE:end -->") == 1


def test_the_readme_headline_block_is_identical_to_the_page_block():
    page = render.page_blocks(PAGE.read_text(encoding="utf-8"))["headline"]
    readme = render.page_blocks(README.read_text(encoding="utf-8"))
    assert set(readme) == {"headline"}, "README.md embeds exactly the headline block"
    assert readme["headline"] == page, "README.md and the page differ; run --write"
    assert page == render.render_all()["headline"], "run render_tables.py --write"


def test_a_drifted_readme_headline_is_reported_and_repaired_negative_control(tmp_path):
    script = str(RESULTS / "render_tables.py")
    page = tmp_path / "benchmarks.md"
    page.write_text(PAGE.read_text(encoding="utf-8"), encoding="utf-8")
    readme = tmp_path / "README.md"
    original = README.read_text(encoding="utf-8")
    readme.write_text(original.replace("6.40", "6.41", 1), encoding="utf-8")
    cmd = [sys.executable, script, "--page", str(page), "--readme", str(readme)]
    drifted = subprocess.run([*cmd, "--check"], capture_output=True, text=True)
    assert drifted.returncode == 1
    assert "differs: headline (README.md)" in drifted.stderr
    subprocess.run([*cmd, "--write"], check=True)
    repaired = subprocess.run([*cmd, "--check"], capture_output=True, text=True)
    assert repaired.returncode == 0
    assert readme.read_text(encoding="utf-8") == original


def test_the_command_line_check_passes():
    run = subprocess.run(
        [sys.executable, str(RESULTS / "render_tables.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr


def test_the_default_output_holds_every_block_once():
    out = subprocess.run(
        [sys.executable, str(RESULTS / "render_tables.py")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert render.page_blocks(out) == render.render_all()


def test_every_csv_has_the_documented_header_and_flags():
    seen = 0
    for name, line, row in _rows():
        seen += 1
        assert list(row) == render.COLUMNS, name
        assert row["bold"] in {"0", "1"}, (name, line)
        assert row["value"] != "", (name, line)
        assert row["table"] in render.SPEC_BY_ID, (name, line)
    assert seen > 100


def test_every_row_belongs_to_a_rendered_cell():
    for name, line, row in _rows():
        spec = render.SPEC_BY_ID[row["table"]]
        assert name == spec[1], (name, line)
        assert row["arm"] in {c[0] for c in spec[4]}, (name, line, row["arm"])


def test_every_row_has_a_label_and_a_source():
    for name, line, row in _rows():
        for field in (
            "date",
            "hardware",
            "power_w",
            "software",
            "config",
            "n",
            "source_url",
        ):
            assert row[field].strip(), (name, line, field)
        assert row["power_w"].isdigit(), (name, line)
        src = row["source_url"]
        if PR_URL.match(src):
            continue
        for item in (part.strip() for part in src.split(";")):
            path, _, anchor = item.partition("#")
            assert (ROOT / path).is_file(), (
                f"{name}:{line}: source {item!r} is not a file of this repository"
            )
            if anchor:
                assert anchor in _anchors(ROOT / path), (
                    f"{name}:{line}: no heading #{anchor} in {path}"
                )


def test_published_intervals_contain_their_values():
    for name, line, row in _rows():
        if not (row["ci95_low"] and row["ci95_high"]):
            continue
        low, value, high = (
            float(row["ci95_low"]),
            float(row["value"]),
            float(row["ci95_high"]),
        )
        assert low <= value <= high, (name, line, row["row"], row["arm"])


def test_single_run_cells_are_bold_only_for_deterministic_quantities():
    for name, line, row in _rows():
        if row["bold"] == "1" and row["n"] == "1":
            assert row["unit"] == "tokens", (name, line, row["row"])


def _same_column_group(table: str, a: str, b: str) -> bool:
    # K1 holds two context lengths side by side; only values at the same length compete
    return table != "k1" or a.rsplit("_", 1)[-1] == b.rsplit("_", 1)[-1]


def test_a_bold_cell_is_the_best_value_of_its_row():
    # times (ms, us, s) are better when lower; rates, pools and accuracies when higher;
    # ratios and text are not compared
    lower, higher = {"ms", "µs", "s"}, {"tok/s", "tokens", "fraction"}
    cells: dict[tuple, list[dict]] = {}
    for _name, _line, row in _rows():
        cells.setdefault((row["table"], row["row"]), []).append(row)
    checked = 0
    for (table, label), rows in cells.items():
        for r in rows:
            if r["bold"] != "1" or r["unit"] not in lower | higher:
                continue
            rivals = []
            for o in rows:
                if (
                    o is not r
                    and o["unit"] == r["unit"]
                    and _same_column_group(table, r["arm"], o["arm"])
                ):
                    with contextlib.suppress(ValueError):
                        rivals.append(float(o["value"]))
            if not rivals:
                continue
            checked += 1
            value = float(r["value"])
            assert (
                value <= min(rivals) if r["unit"] in lower else value >= max(rivals)
            ), (table, label, r["arm"])
    assert checked > 20


def test_links_of_the_page_resolve():
    text = PAGE.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if target.startswith("http"):
            continue
        path, _, anchor = target.partition("#")
        dest = PAGE if path == "" else (PAGE.parent / path).resolve()
        assert dest.exists(), f"broken link {target}"
        if anchor and dest.suffix == ".md":
            assert anchor in _anchors(dest), f"no heading #{anchor} in {dest.name}"


@pytest.mark.parametrize("name", sorted(p.name for p in RESULTS.glob("*.csv")))
def test_every_csv_is_listed_in_the_results_readme(name):
    assert f"`{name}`" in (RESULTS / "README.md").read_text(encoding="utf-8")
