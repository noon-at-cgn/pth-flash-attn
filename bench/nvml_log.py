# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""NVML sampler for GPU benchmark runs.

Samples every GPU every 100 ms and appends one JSON object per line:

* ``{"kind": "header", ...}``: format version, interval, driver, devices
  (index, name, uuid, PCI bus id, enforced power limit), both clocks.
* ``{"kind": "sample", ...}``: per GPU per tick: SM and memory clock (MHz),
  power (W), temperature (C), clocks-event ("throttle") reason mask and names,
  GPU utilisation, and the compute processes NVML reports (pid, MiB).
  Fields NVML cannot read are ``null`` and the error text is kept in
  ``errors``: bench/stats.py rejects a block it cannot vouch for rather than
  guessing.
* ``{"kind": "footer", ...}``: tick count and how many ticks ran late.

Timestamps: ``t_mono_ns`` is ``time.monotonic_ns()`` (CLOCK_MONOTONIC), the
clock the C++ benchmarks stamp blocks with, so samples and blocks can be
matched across containers; ``t_unix_ns`` is for humans.

Ticks are scheduled at t0 + k * interval, not "sleep interval after the last
one", so the period does not drift with the time a read takes; a tick that is
already late is skipped and counted, never bunched.

The NVML calls live behind a small backend interface so the CPU tests inject
a fake; the real backend imports ``pynvml`` (``nvidia-ml-py``, in the image)
only when used. The sampler only reads: it never sets clocks, power limits or
any other device state.

Usage (needs an NVIDIA GPU and the NVML Python bindings; the benchmark
drivers start it for the length of a run)::

    python3 bench/nvml_log.py --out results/<date>-ceilings-gpu0-220w/nvml.jsonl
    # stops on SIGTERM / SIGINT, or after --duration-s
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import socket
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TextIO

LOG_FORMAT = 1
SAMPLE_INTERVAL_MS = 100  # NVML sampled at 100 ms
MIB = 1 << 20

# Same bit names as bench/stats.py (NVML nvmlClocksEventReason*); duplicated
# deliberately so the logger has no import dependency on the analysis code.
# tests/cpu/test_nvml_log.py checks the two tables agree.
THROTTLE_NAMES = {
    0x1: "gpu_idle",
    0x2: "applications_clocks_setting",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal_slowdown",
    0x40: "hw_thermal_slowdown",
    0x80: "hw_power_brake_slowdown",
    0x100: "display_clock_setting",
}


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    uuid: str
    pci_bus_id: str
    power_limit_w: float | None


@dataclass(frozen=True)
class Reading:
    sm_clock_mhz: float | None
    mem_clock_mhz: float | None
    power_w: float | None
    temp_c: float | None
    throttle_reasons: int | None
    util_gpu_pct: float | None
    compute_procs: list[dict[str, Any]] | None
    errors: dict[str, str] = field(default_factory=dict)


class Backend(Protocol):
    def init(self) -> None: ...
    def shutdown(self) -> None: ...
    def driver_version(self) -> str: ...
    def devices(self) -> list[DeviceInfo]: ...
    def read(self, index: int) -> Reading: ...


class PynvmlBackend:
    """Real NVML through pynvml. Every read is independent: one unsupported
    query yields None for that field, not a lost sample."""

    def __init__(self) -> None:
        self._nv: Any = None
        self._handles: dict[int, Any] = {}

    def init(self) -> None:
        import pynvml  # type: ignore[import-not-found]

        self._nv = pynvml
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            self._handles[i] = pynvml.nvmlDeviceGetHandleByIndex(i)

    def shutdown(self) -> None:
        if self._nv is not None:
            self._nv.nvmlShutdown()

    @staticmethod
    def _text(v: Any) -> str:
        return v.decode() if isinstance(v, bytes) else str(v)

    def driver_version(self) -> str:
        return self._text(self._nv.nvmlSystemGetDriverVersion())

    def devices(self) -> list[DeviceInfo]:
        nv = self._nv
        out = []
        for i, h in self._handles.items():
            try:
                limit: float | None = nv.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
            except nv.NVMLError:
                limit = None
            out.append(
                DeviceInfo(
                    index=i,
                    name=self._text(nv.nvmlDeviceGetName(h)),
                    uuid=self._text(nv.nvmlDeviceGetUUID(h)),
                    pci_bus_id=self._text(nv.nvmlDeviceGetPciInfo(h).busId),
                    power_limit_w=limit,
                )
            )
        return out

    def read(self, index: int) -> Reading:
        nv = self._nv
        h = self._handles[index]
        errors: dict[str, str] = {}

        def get(name: str, fn: Callable[[], Any]) -> Any:
            try:
                return fn()
            except nv.NVMLError as e:  # one field failing must not lose the sample
                errors[name] = str(e)
                return None

        reasons_fn = getattr(nv, "nvmlDeviceGetCurrentClocksEventReasons", None) or (
            nv.nvmlDeviceGetCurrentClocksThrottleReasons
        )
        power_mw = get("power_w", lambda: nv.nvmlDeviceGetPowerUsage(h))
        util = get("util_gpu_pct", lambda: nv.nvmlDeviceGetUtilizationRates(h))
        procs = get("compute_procs", lambda: nv.nvmlDeviceGetComputeRunningProcesses(h))
        return Reading(
            sm_clock_mhz=get(
                "sm_clock_mhz", lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)
            ),
            mem_clock_mhz=get(
                "mem_clock_mhz", lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM)
            ),
            power_w=None if power_mw is None else power_mw / 1000.0,
            temp_c=get(
                "temp_c",
                lambda: nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
            ),
            throttle_reasons=get("throttle_reasons", lambda: int(reasons_fn(h))),
            util_gpu_pct=None if util is None else float(util.gpu),
            compute_procs=None
            if procs is None
            else [
                {
                    "pid": int(p.pid),
                    # usedGpuMemory is None when the driver cannot attribute it
                    "used_mib": None
                    if p.usedGpuMemory is None
                    else p.usedGpuMemory / MIB,
                }
                for p in procs
            ],
            errors=errors,
        )


_CONTAINER_ID = re.compile(r"(?:docker[-/]|libpod-|cri-containerd-)([0-9a-f]{64})")


def describe_process(pid: int, proc_root: str = "/proc") -> dict[str, Any]:
    """What can be read about ``pid`` at the moment it is first seen on a GPU:
    parent pid, command line, container id (from the cgroup path).

    NVML reports host pids. The reads work only if the logger sees the host's
    /proc (``docker run --pid=host``, as bench/lib.sh starts it); otherwise the
    process is not visible and the
    result says ``"resolved": false``. Best effort and read-only: a process
    that has already exited gives the same answer.
    """
    root = Path(proc_root) / str(pid)
    out: dict[str, Any] = {"resolved": False}
    try:
        out["cmdline"] = (
            (root / "cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode(errors="replace")
            .strip()
        )
        out["comm"] = (root / "comm").read_text().strip()
        status = (root / "status").read_text()
        m = re.search(r"^PPid:\s*(\d+)", status, re.M)
        out["ppid"] = int(m.group(1)) if m else None
        cg = (root / "cgroup").read_text()
        m = _CONTAINER_ID.search(cg)
        out["container_id"] = m.group(1)[:12] if m else None
        out["resolved"] = True
    except OSError as e:
        out["error"] = type(e).__name__
    return out


def throttle_names(mask: int | None) -> list[str] | None:
    if mask is None:
        return None
    return [name for bit, name in sorted(THROTTLE_NAMES.items()) if mask & bit]


def sample_record(
    tick: int, t_mono_ns: int, t_unix_ns: int, dev: DeviceInfo, r: Reading
) -> dict[str, Any]:
    """The JSON object for one GPU at one tick (schema $defs/nvml_sample)."""
    return {
        "kind": "sample",
        "tick": tick,
        "t_mono_ns": t_mono_ns,
        "t_unix_ns": t_unix_ns,
        "gpu_index": dev.index,
        "pci_bus_id": dev.pci_bus_id,
        "sm_clock_mhz": r.sm_clock_mhz,
        "mem_clock_mhz": r.mem_clock_mhz,
        "power_w": r.power_w,
        "temp_c": r.temp_c,
        "throttle_reasons": r.throttle_reasons,
        "throttle_names": throttle_names(r.throttle_reasons),
        "util_gpu_pct": r.util_gpu_pct,
        "compute_procs": r.compute_procs,
        "errors": r.errors,
    }


class Sampler:
    """Fixed-rate sampler. Clock and sleep are injectable for tests."""

    def __init__(
        self,
        backend: Backend,
        out: TextIO,
        interval_ms: int = SAMPLE_INTERVAL_MS,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        unix_ns: Callable[[], int] = time.time_ns,
        sleep: Callable[[float], None] = time.sleep,
        describe: Callable[[int], dict[str, Any]] = describe_process,
    ) -> None:
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        self.backend = backend
        self.out = out
        self.interval_ns = interval_ms * 1_000_000
        self.clock_ns = clock_ns
        self.unix_ns = unix_ns
        self.sleep = sleep
        self.describe = describe
        self._seen: set[tuple[int, int]] = set()  # (gpu index, pid) already logged
        self.ticks = 0
        self.late_ticks = 0
        self._stop = False
        self._devices: list[DeviceInfo] = []

    def request_stop(self, *_: Any) -> None:
        """Signal-handler compatible: finish the current tick, then stop."""
        self._stop = True

    def _emit(self, obj: dict[str, Any]) -> None:
        # One line per object, flushed: a run killed mid-way keeps every
        # complete line, and a reader can follow the file while it grows.
        self.out.write(json.dumps(obj, separators=(",", ":")) + "\n")
        self.out.flush()

    def _log_new_processes(
        self, t_mono_ns: int, t_unix_ns: int, dev: DeviceInfo, r: Reading
    ) -> None:
        """One ``proc_seen`` line the first time a pid shows on a GPU, with what
        /proc says about it then (the evidence for who used the GPU is
        written at detection, not reconstructed later). The
        benchmark's own process is logged too; readers pick the foreign ones."""
        for p in r.compute_procs or []:
            key = (dev.index, int(p["pid"]))
            if key in self._seen:
                continue
            self._seen.add(key)
            self._emit(
                {
                    "kind": "proc_seen",
                    "t_mono_ns": t_mono_ns,
                    "t_unix_ns": t_unix_ns,
                    "gpu_index": dev.index,
                    "pci_bus_id": dev.pci_bus_id,
                    "pid": int(p["pid"]),
                    "used_mib": p.get("used_mib"),
                    **self.describe(int(p["pid"])),
                }
            )

    def run(
        self, duration_s: float | None = None, gpus: Sequence[int] | None = None
    ) -> int:
        self.backend.init()
        try:
            devs = self.backend.devices()
            self._devices = [d for d in devs if gpus is None or d.index in gpus]
            t0 = self.clock_ns()
            self._emit(
                {
                    "kind": "header",
                    "log_format": LOG_FORMAT,
                    "interval_ms": self.interval_ns // 1_000_000,
                    "t_mono_ns": t0,
                    "t_unix_ns": self.unix_ns(),
                    "host": socket.gethostname(),
                    "driver_version": self.backend.driver_version(),
                    "devices": [d.__dict__ for d in self._devices],
                }
            )
            deadline = None if duration_s is None else t0 + int(duration_s * 1e9)
            k = 0
            while not self._stop:
                due = t0 + k * self.interval_ns
                if deadline is not None and due > deadline:
                    break
                now = self.clock_ns()
                if now < due:
                    self.sleep((due - now) / 1e9)
                elif now - due >= self.interval_ns:
                    # More than a whole period late (host stall): skip to the
                    # next slot instead of firing a burst of catch-up samples.
                    skipped = (now - due) // self.interval_ns
                    self.late_ticks += int(skipped)
                    k += int(skipped)
                    continue
                t_mono, t_unix = self.clock_ns(), self.unix_ns()
                for d in self._devices:
                    reading = self.backend.read(d.index)
                    self._emit(sample_record(self.ticks, t_mono, t_unix, d, reading))
                    self._log_new_processes(t_mono, t_unix, d, reading)
                self.ticks += 1
                k += 1
            self._emit(
                {
                    "kind": "footer",
                    "t_mono_ns": self.clock_ns(),
                    "ticks": self.ticks,
                    "late_ticks": self.late_ticks,
                }
            )
        finally:
            self.backend.shutdown()
        return 0


def read_log(lines: Sequence[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse a log: (header, samples). Truncated last line (killed logger) is
    tolerated; any other malformed line is an error."""
    header: dict[str, Any] | None = None
    samples = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break
            raise
        if obj.get("kind") == "header":
            header = obj
        elif obj.get("kind") == "sample":
            samples.append(obj)
    if header is None:
        raise ValueError("NVML log has no header line")
    return header, samples


def _main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Sample NVML (clocks, power, temperature, throttle reasons, compute "
            "processes) of the GPUs every 100 ms and append one JSON object per "
            "line to --out (header, samples, proc_seen, footer). Read-only: it "
            "never sets clocks or power. Needs an NVIDIA GPU and the pynvml "
            "(nvidia-ml-py) package; stops on SIGTERM / SIGINT or after "
            "--duration-s. The benchmark drivers (bench/k_run.sh, "
            "bench/ceilings_run.sh) run it alongside every measurement."
        )
    )
    p.add_argument("--out", required=True, help="JSON-lines file to append to")
    p.add_argument("--interval-ms", type=int, default=SAMPLE_INTERVAL_MS)
    p.add_argument(
        "--duration-s", type=float, default=None, help="default: until SIGTERM"
    )
    p.add_argument(
        "--gpus", default="all", help="'all' or comma-separated NVML indices"
    )
    a = p.parse_args(argv)
    gpus = None if a.gpus == "all" else [int(x) for x in a.gpus.split(",")]
    with open(a.out, "a", encoding="utf-8") as f:
        s = Sampler(PynvmlBackend(), f, interval_ms=a.interval_ms)
        signal.signal(signal.SIGTERM, s.request_stop)
        signal.signal(signal.SIGINT, s.request_stop)
        return s.run(duration_s=a.duration_s, gpus=gpus)


if __name__ == "__main__":
    sys.exit(_main())
