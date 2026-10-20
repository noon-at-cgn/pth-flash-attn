# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for bench/nvml_log.py with an injected fake NVML.

The sampler's output decides which benchmark blocks are rejected
(bench/stats.py), so it must: sample at a fixed rate without drift, never
burst after a stall, keep a sample when one field fails, write lines the
schema accepts, and stop cleanly on a signal. No GPU and no NVML library are
touched: the fake backend and a fake clock make every run deterministic.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import nvml_log  # noqa: E402
import stats  # noqa: E402

DEV = nvml_log.DeviceInfo(
    0, "NVIDIA GeForce RTX 3080", "GPU-x", "00000000:01:00.0", 220.0
)


class FakeClock:
    """Monotonic ns clock that only moves when the sampler sleeps, plus an
    optional per-read cost (how long one NVML read 'takes')."""

    def __init__(self, read_cost_ns=0):
        self.t = 5_000_000_000
        self.read_cost_ns = read_cost_ns

    def now(self):
        return self.t

    def sleep(self, s):
        assert s >= 0
        self.t += int(round(s * 1e9))


class FakeBackend:
    def __init__(self, clock, readings=None, stall_at=None, stall_ns=0):
        self.clock = clock
        self.readings = readings
        self.stall_at = stall_at
        self.stall_ns = stall_ns
        self.reads = 0
        self.inited = self.shut = False

    def init(self):
        self.inited = True

    def shutdown(self):
        self.shut = True

    def driver_version(self):
        return "580.00"

    def devices(self):
        return [
            DEV,
            nvml_log.DeviceInfo(
                1, "NVIDIA GeForce RTX 3080", "GPU-y", "00000000:02:00.0", 220.0
            ),
        ]

    def read(self, index):
        self.reads += 1
        self.clock.t += self.clock.read_cost_ns
        if self.stall_at is not None and self.reads == self.stall_at:
            self.clock.t += self.stall_ns
        if self.readings is not None:
            return self.readings(index, self.reads)
        return nvml_log.Reading(
            1800.0, 9501.0, 215.0, 66.0, 0x4, 99.0, [{"pid": 7, "used_mib": 512.0}]
        )


def run(backend, clock, **kw):
    out = io.StringIO()
    s = nvml_log.Sampler(
        backend, out, clock_ns=clock.now, unix_ns=lambda: 1, sleep=clock.sleep
    )
    s.run(**kw)
    lines = out.getvalue().splitlines()
    return s, [json.loads(x) for x in lines]


def test_fixed_rate_without_drift():
    """Ticks are scheduled at t0 + k*100 ms. Even when each read costs 7 ms,
    sample k is taken at exactly t0 + k*100 ms (a 'sleep 100 ms after each
    read' loop would drift by 7 ms per tick and misalign block windows)."""
    clock = FakeClock(read_cost_ns=7_000_000)
    s, objs = run(FakeBackend(clock), clock, duration_s=1.0, gpus=[0])
    samples = [o for o in objs if o["kind"] == "sample"]
    t0 = objs[0]["t_mono_ns"]
    assert len(samples) == 11  # t0, t0+100ms, ..., t0+1000ms
    for k, smp in enumerate(samples):
        assert smp["t_mono_ns"] == t0 + k * 100_000_000
    assert s.late_ticks == 0


def test_stall_skips_ticks_instead_of_bursting():
    """The read at t0+200 ms stalls the host for 350 ms. The sampler counts
    the two slots it missed entirely (300, 400 ms), takes the overdue 500 ms
    slot as soon as it can (at 550 ms) and is back on the grid at 600 ms. It
    never fires catch-up samples at one instant, which would put several
    identical timestamps into one block's window."""
    clock = FakeClock()
    backend = FakeBackend(clock, stall_at=3, stall_ns=350_000_000)
    s, objs = run(backend, clock, duration_s=1.0, gpus=[0])
    t0 = objs[0]["t_mono_ns"]
    times = [(o["t_mono_ns"] - t0) // 1_000_000 for o in objs if o["kind"] == "sample"]
    assert times == [0, 100, 200, 550, 600, 700, 800, 900, 1000]
    assert s.late_ticks == 2
    footer = objs[-1]
    assert footer["kind"] == "footer" and footer["late_ticks"] == 2


def test_header_samples_footer_and_schema():
    """Line format: one header (devices with PCI bus ids, interval), one sample
    per GPU per tick, one footer. Every sample validates against
    results.schema.json $defs/nvml_sample, the contract with stats.py."""
    clock = FakeClock()
    backend = FakeBackend(clock)
    _, objs = run(backend, clock, duration_s=0.25)
    assert objs[0]["kind"] == "header" and objs[0]["interval_ms"] == 100
    assert [d["pci_bus_id"] for d in objs[0]["devices"]] == [
        "00000000:01:00.0",
        "00000000:02:00.0",
    ]
    samples = [o for o in objs if o["kind"] == "sample"]
    assert len(samples) == 3 * 2  # ticks at 0, 100, 200 ms; two GPUs
    schema = json.loads((ROOT / "bench" / "results.schema.json").read_text())
    sample_schema = {"$defs": schema["$defs"], "$ref": "#/$defs/nvml_sample"}
    for smp in samples:
        jsonschema.validate(smp, sample_schema)
    assert samples[0]["throttle_names"] == ["sw_power_cap"]
    assert objs[-1]["kind"] == "footer" and objs[-1]["ticks"] == 3
    assert backend.inited and backend.shut


def test_failed_field_is_null_with_error_and_rejects_downstream():
    """One unreadable field (e.g. temperature) yields null plus the error text,
    not a lost sample; stats.py then rejects the block 'nvml_missing'
    instead of assuming it was cool (fail closed)."""
    clock = FakeClock()

    def readings(_i, _n):
        return nvml_log.Reading(
            1800.0, 9501.0, 215.0, None, 0x4, 99.0, [], {"temp_c": "Not Supported"}
        )

    _, objs = run(
        FakeBackend(clock, readings=readings), clock, duration_s=0.0, gpus=[0]
    )
    smp = next(o for o in objs if o["kind"] == "sample")
    assert smp["temp_c"] is None and smp["errors"] == {"temp_c": "Not Supported"}
    block = stats.Block(
        0, 0, 0, "a", smp["t_mono_ns"], smp["t_mono_ns"] + 1, (1.0, 1.0)
    )
    ref = {stats.SM_CLOCK: 1800.0, stats.MEM_CLOCK: 9501.0}
    assert stats.block_rejection_reasons(
        block, [smp], stats.RejectionConfig(), ref
    ) == ["nvml_missing:temp_c"]


def test_request_stop_ends_after_current_tick():
    """SIGTERM handler semantics: the current tick completes, the footer is
    written, NVML is shut down: a stopped logger leaves a parseable file."""
    clock = FakeClock()
    backend = FakeBackend(clock)
    out = io.StringIO()
    s = nvml_log.Sampler(
        backend, out, clock_ns=clock.now, unix_ns=lambda: 1, sleep=clock.sleep
    )
    real_read = backend.read

    def read_then_stop(i):
        r = real_read(i)
        if backend.reads == 4:
            s.request_stop()
        return r

    backend.read = read_then_stop
    s.run(gpus=[0])  # no duration: runs until stopped
    objs = [json.loads(x) for x in out.getvalue().splitlines()]
    assert objs[-1]["kind"] == "footer" and objs[-1]["ticks"] == 4
    assert backend.shut


def test_read_log_tolerates_only_a_truncated_last_line():
    """A logger killed mid-write leaves a partial last line: tolerated. A
    malformed line in the middle means a corrupt file: an error."""
    hdr = json.dumps({"kind": "header", "interval_ms": 100})
    smp = json.dumps({"kind": "sample", "t_mono_ns": 1})
    h, s = nvml_log.read_log([hdr, smp, '{"kind": "sam'])
    assert h["interval_ms"] == 100 and len(s) == 1
    with pytest.raises(json.JSONDecodeError):
        nvml_log.read_log([hdr, "{bad", smp])
    with pytest.raises(ValueError):
        nvml_log.read_log([smp])


def test_throttle_tables_agree_with_stats():
    """The logger names throttle bits with its own copy of the NVML table (it
    must not import the analysis code); both copies must be identical."""
    assert nvml_log.THROTTLE_NAMES == stats.THROTTLE_NAMES


def test_pynvml_backend_constants_exist_in_image():
    """The real backend uses these pynvml names; nvidia-ml-py is in the image.
    Checked without initialising NVML (no driver on the CPU runner)."""
    pynvml = pytest.importorskip("pynvml")
    for name in (
        "nvmlInit",
        "nvmlDeviceGetClockInfo",
        "nvmlDeviceGetPowerUsage",
        "nvmlDeviceGetTemperature",
        "nvmlDeviceGetComputeRunningProcesses",
        "nvmlDeviceGetEnforcedPowerLimit",
        "nvmlDeviceGetPciInfo",
        "nvmlDeviceGetUtilizationRates",
        "NVML_CLOCK_SM",
        "NVML_CLOCK_MEM",
        "NVML_TEMPERATURE_GPU",
    ):
        assert hasattr(pynvml, name), name
    assert hasattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons") or hasattr(
        pynvml, "nvmlDeviceGetCurrentClocksThrottleReasons"
    )
    for bit, name in stats.THROTTLE_NAMES.items():
        camel = {
            "gpu_idle": "GpuIdle",
            "applications_clocks_setting": "ApplicationsClocksSetting",
            "sw_power_cap": "SwPowerCap",
            "hw_slowdown": "HwSlowdown",
            "sync_boost": "SyncBoost",
            "sw_thermal_slowdown": "SwThermalSlowdown",
            "hw_thermal_slowdown": "HwThermalSlowdown",
            "hw_power_brake_slowdown": "HwPowerBrakeSlowdown",
            "display_clock_setting": "DisplayClockSetting",
        }[name]
        assert getattr(pynvml, "nvmlClocksThrottleReason" + camel) == bit


# ---------------------------------------------------------------- process events


def test_a_new_pid_is_logged_once_per_gpu_with_what_proc_says(tmp_path):
    """The evidence about a foreign process is written
    at detection. One proc_seen line per (GPU, pid), however many ticks the
    process stays; pid and memory come from NVML, the rest from /proc."""
    clock = FakeClock()
    seen = []

    def describe(pid):
        seen.append(pid)
        return {
            "resolved": True,
            "ppid": 1,
            "cmdline": f"cmd {pid}",
            "container_id": None,
        }

    backend = FakeBackend(clock)
    out = io.StringIO()
    s = nvml_log.Sampler(
        backend,
        out,
        clock_ns=clock.now,
        unix_ns=lambda: 1,
        sleep=clock.sleep,
        describe=describe,
    )
    s.run(duration_s=0.25)
    events = [json.loads(x) for x in out.getvalue().splitlines()]
    events = [o for o in events if o["kind"] == "proc_seen"]
    assert [(e["gpu_index"], e["pid"]) for e in events] == [(0, 7), (1, 7)]
    assert events[0]["used_mib"] == 512.0 and events[0]["cmdline"] == "cmd 7"
    assert events[0]["pci_bus_id"] == "00000000:01:00.0"
    assert seen == [7, 7]  # described at first sight only


def test_describe_process_reads_ppid_cmdline_and_container_id(tmp_path):
    cid = "ab" * 32
    p = tmp_path / "4242"
    p.mkdir()
    (p / "cmdline").write_bytes(b"python3\0-c\0print(1)\0")
    (p / "comm").write_text("python3\n")
    (p / "status").write_text("Name:\tpython3\nPPid:\t99\nUid:\t0\n")
    (p / "cgroup").write_text(f"0::/system.slice/docker-{cid}.scope\n")
    d = nvml_log.describe_process(4242, str(tmp_path))
    assert d == {
        "resolved": True,
        "cmdline": "python3 -c print(1)",
        "comm": "python3",
        "ppid": 99,
        "container_id": cid[:12],
    }


def test_describe_process_says_unresolved_for_a_pid_it_cannot_see(tmp_path):
    d = nvml_log.describe_process(4243, str(tmp_path))
    assert d["resolved"] is False and d["error"] == "FileNotFoundError"


def test_describe_process_without_a_container_has_a_null_id(tmp_path):
    p = tmp_path / "5"
    p.mkdir()
    (p / "cmdline").write_bytes(b"x\0")
    (p / "comm").write_text("x\n")
    (p / "status").write_text("PPid:\t1\n")
    (p / "cgroup").write_text("0::/user.slice/session-3.scope\n")
    assert nvml_log.describe_process(5, str(tmp_path))["container_id"] is None


def test_read_log_ignores_proc_seen_lines():
    lines = [
        json.dumps({"kind": "header", "log_format": 1}),
        json.dumps({"kind": "proc_seen", "pid": 1}),
        json.dumps({"kind": "sample", "tick": 0}),
    ]
    hdr, samples = nvml_log.read_log(lines)
    assert len(samples) == 1
