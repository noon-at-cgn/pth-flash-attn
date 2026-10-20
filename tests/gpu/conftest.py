# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Fixtures and markers of the GPU tier (the GPU correctness grid).

The tier is never collected by default (``testpaths`` lists tests/cpu and
tests/sim only); a GPU run starts it explicitly through ``tests/gpu/run_grid.sh``.
Markers (registered here, besides ``gpu``/``full`` from pyproject.toml):

* ``quick``      the minutes-long subset run first in a GPU run;
* ``sanitizer``  small shapes run under compute-sanitizer (four tools);
* ``graph``      CUDA-graph capture/replay tests (the last step of a GPU run);
* ``sanitizer_control``  planted errors that a sanitizer must report; they
  are deselected unless the ``-m`` expression names this marker, because
  without a sanitizer the planted out-of-bounds read can fault the context.

A missing CUDA device or library is a test *failure*, not a skip: in a
GPU run, a silently skipped tier would look like a pass.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from pth_testing import ref_capi

REPO = Path(__file__).resolve().parents[2]
LOADER_LIB = REPO / "build" / "loader" / "libpth_loader.so"
CONTROL = "sanitizer_control"


def pytest_configure(config: pytest.Config) -> None:
    for line in (
        "quick: minutes-long subset of the GPU correctness grid, run first",
        "sanitizer: small shapes for compute-sanitizer (memcheck, racecheck, ...)",
        "graph: CUDA-graph capture/replay equality",
        f"{CONTROL}: planted errors; only with -m {CONTROL} under compute-sanitizer",
    ):
        config.addinivalue_line("markers", line)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--grid-record",
        default=None,
        help="append one JSON line per checked case to this file (grid_report.py)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    # Everything under tests/gpu needs the device: mark it, so `-m gpu`
    # selects the whole tier and `-m "not gpu"` can never pick it up.
    here = Path(__file__).parent
    for it in items:
        if Path(str(it.fspath)).parent == here:
            it.add_marker(pytest.mark.gpu)
    if CONTROL in (config.getoption("markexpr") or ""):
        return
    keep = [it for it in items if it.get_closest_marker(CONTROL) is None]
    if len(keep) != len(items):
        config.hook.pytest_deselected(
            items=[it for it in items if it.get_closest_marker(CONTROL)]
        )
        items[:] = keep


@pytest.fixture(scope="session")
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.fail("no CUDA device visible: the gpu tier must run on a GPU")
    return torch.device("cuda", 0)


@pytest.fixture(scope="session")
def device_info(device: torch.device) -> dict[str, Any]:
    p = torch.cuda.get_device_properties(device)
    return {
        "name": p.name,
        "sm_count": p.multi_processor_count,
        "cc": f"{p.major}.{p.minor}",
        "total_memory": p.total_memory,
    }


@pytest.fixture(scope="session")
def ref_lib(device: torch.device) -> ref_capi.RefLib:
    if not ref_capi.CUDA_LIB.exists():
        pytest.fail(f"{ref_capi.CUDA_LIB} missing: run tools/build_ref_lib.sh")
    lib = ref_capi.RefLib(ref_capi.CUDA_LIB)
    if not lib.has_cuda():
        pytest.fail(f"{ref_capi.CUDA_LIB} is the host-only build")
    return lib


@pytest.fixture(scope="session")
def loader_lib(device: torch.device) -> Any:
    from grid_cuda import LoaderLib

    if not LOADER_LIB.exists():
        pytest.fail(f"{LOADER_LIB} missing: run tools/build_loader_lib.sh")
    return LoaderLib(str(LOADER_LIB))


@pytest.fixture(scope="session")
def record(
    request: pytest.FixtureRequest, device_info: dict[str, Any]
) -> Iterator[Callable[..., None]]:
    """``record(test=..., **fields)``: one JSON line per case for grid_report.py."""
    path = request.config.getoption("--grid-record")
    fh = open(path, "a") if path else None  # noqa: SIM115 (closed below)

    def write(**fields: Any) -> None:
        if fh is not None:
            fields.setdefault("device", device_info)
            fields.setdefault("pid", os.getpid())
            fh.write(json.dumps(fields, default=str) + "\n")
            fh.flush()

    yield write
    if fh is not None:
        fh.close()
