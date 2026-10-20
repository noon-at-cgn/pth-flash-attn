# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Shared-memory bank-conflict counter.

Bank model (CUDA Programming Guide, "2.3 Writing SIMT Kernels", shared memory
bank conflicts, https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/
writing-cuda-kernels.html; Nsight Compute Profiling Guide, shared memory):
32 banks; successive 32-bit words map to successive banks, so
bank(addr) = (addr / 4) mod 32. Accesses by different threads to the *same*
32-bit word do not conflict (broadcast). Accesses to *different* words in
the same bank are serialised.

How a warp's request is split into passes ("phases"/wavefronts) is not in
the PTX ISA. The model used here (the usual one; it has not been checked against Nsight
Compute `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` on the GPU):
- 4-B accesses (LDS.32, cp.async 4): the whole warp is one phase.
- 8-B accesses (LDS.64, cp.async 8): two phases of 16 lanes.
- 16-B accesses (LDS.128, cp.async 16): four phases of 8 lanes.
- ldmatrix: each 8x8 matrix (8 row addresses x 16 B) is one phase.

The headline figure is the conflict factor: wavefronts (sum over phases of
the busiest bank's distinct-word count) divided by the minimum the same
phases need (sum over phases of ceil(distinct bytes / 128)). 1.0 means
conflict-free ("1-way"). To
limit the dependence on the phase model, `test_phase_model_insensitive`
re-evaluates every LSU pattern with the whole warp as one phase and checks
that the verdict (conflict-free or not) does not change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NUM_BANKS = 32
BANK_BYTES = 4
LINE_BYTES = NUM_BANKS * BANK_BYTES  # 128

LDMATRIX_ROWS_PER_PHASE = 8
LDMATRIX_ROW_BYTES = 16


def lanes_per_phase(width: int) -> int:
    """Lanes served per phase for a width-byte access (model above)."""
    return {4: 32, 8: 16, 16: 8}[width]


@dataclass(frozen=True)
class Access:
    """One shared-memory instruction: per-lane start addresses and width."""

    addrs: tuple[int, ...]  # active lanes only, in lane order
    width: int  # bytes per lane
    lanes: tuple[int, ...] | None = None  # lane ids (default 0..len-1)
    kind: str = "lsu"  # "lsu" or "ldmatrix"


def _words(addr: int, width: int) -> list[int]:
    assert addr % min(width, BANK_BYTES) == 0
    return [w for w in range(addr // BANK_BYTES, (addr + width - 1) // BANK_BYTES + 1)]


def phases(acc: Access, single_phase: bool = False) -> list[list[int]]:
    """Split an access into per-phase lists of start addresses.

    `single_phase=True` treats the whole instruction as one phase (the
    alternative model used by `test_phase_model_insensitive`).
    """
    lanes = acc.lanes if acc.lanes is not None else tuple(range(len(acc.addrs)))
    if single_phase:
        return [list(acc.addrs)]
    if acc.kind == "ldmatrix":
        assert acc.width == LDMATRIX_ROW_BYTES
        per = LDMATRIX_ROWS_PER_PHASE
    else:
        per = lanes_per_phase(acc.width)
    groups: dict[int, list[int]] = {}
    for lane, a in zip(lanes, acc.addrs, strict=True):
        groups.setdefault(lane // per, []).append(a)
    return [groups[k] for k in sorted(groups)]


def phase_ways(addrs: list[int], width: int) -> int:
    """Max over banks of the number of distinct words requested in one phase."""
    per_bank: dict[int, set[int]] = {}
    for a in addrs:
        for w in _words(a, width):
            per_bank.setdefault(w % NUM_BANKS, set()).add(w)
    return max(len(s) for s in per_bank.values())


def phase_ideal(addrs: list[int], width: int) -> int:
    """Fewest wavefronts one phase can take: ceil(distinct bytes / 128)."""
    distinct = set()
    for a in addrs:
        distinct.update(_words(a, width))
    return -(-len(distinct) * BANK_BYTES // LINE_BYTES)


@dataclass(frozen=True)
class ConflictReport:
    wavefronts: int  # sum over phases of the busiest bank's word count
    ideal: int  # sum over phases of ceil(distinct bytes / 128)
    worst_phase_ways: int  # max over phases of the busiest bank's count

    @property
    def ways(self) -> float:
        """Conflict factor: 1.0 means conflict-free."""
        return self.wavefronts / self.ideal


def conflicts(acc: Access, single_phase: bool = False) -> ConflictReport:
    ph = phases(acc, single_phase)
    per = [phase_ways(p, acc.width) for p in ph]
    ideal = sum(phase_ideal(p, acc.width) for p in ph)
    return ConflictReport(sum(per), ideal, max(per))


def worst(accesses, single_phase: bool = False) -> ConflictReport:
    """The access with the largest conflict factor (ties: most wavefronts)."""
    reps = [conflicts(a, single_phase) for a in accesses]
    return max(reps, key=lambda r: (r.ways, r.wavefronts))


def bank_of(addr) -> np.ndarray:
    return (np.asarray(addr) // BANK_BYTES) % NUM_BANKS
