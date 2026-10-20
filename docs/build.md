<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn -->
# Building, installing and testing

Building the op library needs a CUDA toolkit and a C++ compiler but no GPU.
Sections 1 and 2 (requirements, installation, build knobs) are for users;
sections 3 to 7 (development container, local CI script, test tiers, what has
been run on a GPU) are for contributors.

## 1. Requirements

| Requirement | Version | Notes |
|---|---|---|
| Operating system | Linux | developed and tested on Ubuntu 24.04 |
| Python | 3.10 to 3.14 | the library is built as an `abi3` extension tagged `cp310`; tested with 3.12 |
| PyTorch | 2.11 or newer at run time | the extension targets the stable libtorch ABI of 2.11; built and tested with 2.13.0 for CUDA 13.0 (`cu130`); versions between 2.11 and 2.12 are declared but untested; the library links `libcudart.so.13`, so PyTorch must be built for CUDA 13 |
| CUDA toolkit | nvcc 13 | tested with 13.0.88; its major version must match the CUDA version PyTorch was built for |
| C++ compiler | g++ with C++17 | tested with 13.3 |
| Build tools | `setuptools` 77.0.3 up to (not including) 81, `ninja` | `ninja` makes the build parallel; without it PyTorch falls back to a slow serial backend |
| GPU to run | NVIDIA sm80, sm86 or sm89 | not needed to build |

The versions marked "tested" are those of the development image
([`docker/dev.Dockerfile`](../docker/dev.Dockerfile)): PyTorch 2.13.0+cu130,
nvcc 13.0.88, g++ 13.3, Python 3.12, Ubuntu 24.04. Other combinations are
untested.

## 2. Install

Install from a checkout into an environment that already has PyTorch (with the
matching CUDA toolkit on `PATH` or in `CUDA_HOME`). `setup.py` imports `torch`
to find its headers and compiler flags, so build isolation must be off:

```bash
pip install ninja "setuptools>=77.0.3,<81"
pip install --no-build-isolation .
python -c "from pth_flash_attn import ops; print(ops.is_available())"   # True
```

An isolated build (`pip install .`) also works, but downloads the pinned build
PyTorch (`torch == 2.13.0`, see the `build-system` table of `pyproject.toml`)
into a temporary environment. The package is not published on PyPI. Optional
extras: `test` and `lint` (the pinned tools of the development image).

For development, build in place and use the source tree directly:

```bash
python setup.py build_ext --inplace            # writes python/pth_flash_attn/_C.abi3.so
PYTHONPATH=python python -c "from pth_flash_attn import ops; print(ops.is_available())"
```

**Build knobs** (all build-time; nothing is read at run time):

| Setting | Effect |
|---|---|
| `TORCH_CUDA_ARCH_LIST` | PyTorch's own variable; honoured when set. Default: `8.0;8.6;8.9` from `tools/build_flags.py`. Use `8.6` to build one architecture only. |
| `MAX_JOBS` | upper bound on parallel compile jobs (PyTorch's extension builder reads it). Lower it on machines with little memory. |
| `CUDA_HOME` | the CUDA toolkit root, if `nvcc` is not on `PATH` |

Compile time and library size depend on the toolchain and the host and are
not tabulated here. `tools/compile_cost.py` measures them per translation unit
and per architecture on your machine (`--json` writes a report).

**What the build produces.** One shared library,
`pth_flash_attn/_C.abi3.so` (installed into the package, or in place under
`python/pth_flash_attn/` for a development build). It contains SASS for exactly
sm80, sm86 and sm89 and no PTX, so a GPU of another generation is refused with a
message and never JIT-compiled. It uses only libtorch's stable C shim, links no
Python library and has no Python init function: it is loaded with
`torch.ops.load_library`, which `pth_flash_attn.ops` does at import. The
launchers are compiled with `-fno-gnu-unique`, which keeps their function-local
statics private to this library; without it a process that also loads the test
library `libpth_attn.so` fails launches with "invalid argument".

**Checking an installed copy.** From a directory outside the source tree, run
`python <checkout>/tools/check_installed_package.py`. It checks
that the package comes from `site-packages`, that the op library loads and
registers the op and its fake, that exactly one `vllm.general_plugins` entry
point named `pth_flash_attn` exists and registers `CUSTOM`, and that the backend
class imports. It needs vLLM installed. To list the architectures inside the
library, use the CUDA toolkit's `cuobjdump`.

## 3. Development container

The CPU checks run in containers with no network and no GPU, so that they are
reproducible. There are three images, all built from the repository root (network
needed for the build only). Each Dockerfile carries the same commands in its header.

| Image tag | Dockerfile and target | Contents | Used by |
|---|---|---|---|
| `pth-flash-attn-dev` | [`docker/dev.Dockerfile`](../docker/dev.Dockerfile), stage `dev` | `nvidia/cuda:13.0.3-devel-ubuntu24.04` (nvcc 13.0.88, g++ 13.3, Python 3.12) and a virtual environment in `/app/venv` with torch 2.13.0 (cu130 build), vLLM 0.31.0 and the pinned test and lint Python packages | the `build`, `test`, `cost` and `attn` groups of `tools/run_ci_local.sh` |
| `pth-flash-attn-lint` | `docker/dev.Dockerfile`, stage `lint`, on top of the dev image | adds clang-format, ShellCheck, pre-commit and git | the lint group |
| `pth-flash-attn-sanitizer` | [`docker/sanitizer.Dockerfile`](../docker/sanitizer.Dockerfile), on top of the dev image | the same, plus a build-time check that `compute-sanitizer`, Nsight Compute, `cuobjdump` and `nvdisasm` are present | the GPU script `tests/gpu/run_grid.sh` |

```bash
docker build -f docker/dev.Dockerfile --target dev -t pth-flash-attn-dev .
docker build -f docker/dev.Dockerfile --target lint \
  --build-arg LINT_BASE=pth-flash-attn-dev -t pth-flash-attn-lint .
docker build -f docker/sanitizer.Dockerfile -t pth-flash-attn-sanitizer docker
```

The first two are all that `tools/run_ci_local.sh` needs; the third is only for
the GPU tier. Never pass `--gpus` to the first two. Every package is pinned to an
exact version and the base image to a digest: `CUDA_IMAGE` in
[`docker/dev.Dockerfile`](../docker/dev.Dockerfile) is
`nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266`.
The Ubuntu packages (g++, Python, git) follow the distribution's security updates, so
two builds of the image on different days can differ in them; a report of a
result states the image digest it ran from (`docker image inspect --format '{{.Id}}'`).

## 4. Local CI: one command for every CPU check

```bash
tools/run_ci_local.sh            # every group
tools/run_ci_local.sh lint test  # only the named groups
```

It runs the groups below in order inside the images, with `--network none`, no
GPU and 4 CPUs per container, or fewer if the machine has fewer (`min(4, nproc)`;
Docker refuses a limit above the machine's CPUs; `PTH_CI_CPUS=N` overrides it,
see `tools/ci_cpus.sh`), runs every group even if an earlier one failed, prints a
PASS/FAIL summary and exits 1 if anything failed. On a 2-CPU machine the full run
takes correspondingly longer. Reports go to `build/ci/` (not committed).

| Group | What | Tool |
|---|---|---|
| `host` | every commit has a DCO `Signed-off-by:` line (runs on the host, which needs `git` and `python3`, not in a container) | `tools/repo_checks.py signoff` |
| `lint` | ruff check and format, typos, mypy, clang-format (Google style), ShellCheck, SPDX headers, no `torch.cuda` or environment variables in engine-facing code, pre-commit configuration | the lint image |
| `build` | the op extension for sm80, sm86 and sm89; load it on CPU and compare its schema with [`api.md`](api.md); the reference and loader test libraries | `tools/build_in_image.sh`, `tools/build_ref_lib.sh`, `tools/build_loader_lib.sh` |
| `test` | `pytest tests/cpu tests/sim -m 'not (full and slow)'`; `tests/cpp/run_geometry.sh` | the dev image |
| `cost` | nvcc time and size per translation unit and architecture, `.so` size per architecture; ptxas gate on the op's host translation unit | `tools/compile_cost.py`, `tools/ptxas_gate.py` |
| `attn` | the attention kernel library build (0 spills, no stack frame), the ptxas gate on its logs (at most 255 registers), the SASS lint of the instantiation translation units | `tools/build_attn_lib.sh`, `tools/attn_ci_gates.py`, `tools/sass_check.py` |

The GitHub Actions workflow in `.github/workflows/ci.yml` runs the same script on
hosted runners (there is no GPU in CI).

## 5. Test tiers

| Tier | Location | Needs | What it checks |
|---|---|---|---|
| CPU tests | `tests/cpu/` | Python, torch (CPU), scipy, pytest | reference and bound, emulator, planner, workspace formulas, op argument validation and fake op, backend selection and builder, cache geometry against the reference, benchmark tooling, repository policy tools |
| Simulator tests | `tests/sim/` | Python, numpy | `ptxsim`: per-lane models of `ldmatrix`, `mma.sync` fragments, `prmt`, the magic-number conversion, the K permutation, the V path, swizzles and bank conflicts |
| C++ tests | `tests/cpp/` | g++ (host mode); nvcc (device compile for sm80/86/89) | `cache_geometry.cuh` (exhaustive and property tests, stride fuzz), the PTX and layout primitives against the simulator's golden tables, the tile-traits `static_assert`s |
| GPU tests | `tests/gpu/` | a CUDA device (sm80, sm86 or sm89), the test libraries built by `tools/build_*_lib.sh` | the correctness grid through the scalar reference kernel and the tensor-core kernel, loader byte-exactness, `compute-sanitizer`, CUDA-graph replay, backend and production-shape tests |

`pytest` collects `tests/cpu` and `tests/sim` by default (`testpaths` in
`pyproject.toml`); `tests/gpu` is never collected unless named. Markers: `slow`
(minutes-long CPU tests), `full` (larger CPU matrices), `gpu`, `attn` (tests of the
tensor-core kernel library), `backend` (vLLM backend, plugin and production-op
tests), and the GPU subsets `quick`, `sanitizer`, `graph` and `sanitizer_control`
(`tests/gpu/conftest.py`). A per-test timeout of 600 s turns a hang into a
failure; the 128k-token cases of `tests/cpu/test_bound.py::test_bound_holds_on_emulator_long`
lift it, and CI deselects the one test marked both `full` and `slow`.

**Principles.**

- Every kernel output element must satisfy `|o_hat - o| <= B`, where `o` is
  attention computed in fp64 on the exactly dequantised cache and `B` is derived
  from the actual inputs and the kernel's tiling (`python/pth_testing/bound.py`).
  A run passes with 0 violations; the tightness `max(err / B)` is reported so
  that a bound that has become vacuous is noticed.
- A test that guards a property (the causal edge, the page edge, the swizzle, the
  split count) is paired with a deliberately broken variant that it must reject.
- The grids plant keys aligned with the query at the first and last token and at
  tile, page and split edges, so that boundary bugs have weight at any length.
- Strides come from views, never from shapes: cache fixtures cover NHD and HND
  layouts, padded pages, a storage offset and block ids above 2^31 bytes.
- In `tests/gpu`, a missing CUDA device or library is a failure, not a skip.

**CPU tiers.**

```bash
pytest tests/cpu tests/sim -m "not (full and slow)"
python -m pth_testing.emulate --tier full          # emulator vs bound report table
tests/cpp/run_geometry.sh                          # g++ tests of cache_geometry.cuh, then an nvcc compile for sm80/86/89
```

The mutation tests pass when the bound or the model rejects the broken variant.
In the emulator table the violation count must be 0 and the tightness inside
`[1e-4, 1]`. `tools/sass_diff.py before.so after.so` proves that a refactor left
every kernel's SASS identical (needs `cuobjdump`); `tools/collect_env.sh [OUT]`
captures the environment read-only for a bug report.

**Static gates** (no GPU, run by `tools/run_ci_local.sh`): the ptxas gate
(`tools/attn_ci_gates.py`: no spills, no stack frame, at most 255 registers, every
kernel family present for every architecture), the SASS lint (`tools/sass_check.py`:
no `LDL`/`STL`, no `I2F` inside loops, `LDGSTS`, `LDSM` and `HMMA.16816.F32`
present), the repository policy checks (`tools/repo_checks.py`: SPDX headers, DCO
sign-off, no environment-variable use in engine-facing code) and the exported
subset (`tools/export_upstream.py --out DIR`, `tools/prove_export_sass.sh`: the
copy with the test hooks stripped compiles to the same SASS as the repository).

## 6. The GPU tier

The GPU tier needs one CUDA device and the CPU-built test libraries:

```bash
tools/build_ref_lib.sh        # scalar reference kernel and combine: build/ref/libpth_ref.so
tools/build_loader_lib.sh     # loader test kernels: build/loader/libpth_loader.so
tools/build_attn_lib.sh       # tensor-core kernels: build/attn/libpth_attn.so (fails on any spill)

# fill the fp64 reference/bound cache (CPU, minutes; do this before occupying a GPU)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python tests/gpu/attn_grid.py expected --tier full --jobs 4

pytest tests/gpu/test_attn_grid.py -q -p no:cacheprovider      # tensor-core kernel on the correctness grid
pytest tests/gpu/test_attn_prefill.py -q                       # prefill and mixed launches (q = 2048, 8192)
pytest tests/gpu/test_backend_op.py tests/gpu/test_backend_impl.py tests/gpu/test_attn_graph.py -q
```

`tests/gpu/run_grid.sh --gpu N` runs the whole reference-kernel sequence in one
go: build, expected values, the quick and full tiers, `compute-sanitizer`
(memcheck, racecheck, synccheck, initcheck), the planted-error controls, CUDA
graph replay and a summary report (`tests/gpu/grid_report.py`). Each step is
logged with its exit code; every step runs even after a failure, so one run
yields all the evidence, and a step that runs no test counts as a failure. It
never overwrites an existing result directory; its header lists the options and
exit codes.

The correctness grid (`tests/gpu/grid_cases.py`, 328 cases, shared with
`tests/cpu/test_grid.py`) is composed of families instead of the Cartesian
product: paging edges, row tiling, GQA group sizes, head sizes 64, 128 and 256,
mixed batches with padding rows, split counts up to the planner's maximum, context
lengths up to 256k, scale extremes, dtypes, stride fuzz, production descriptors
and sliding-window and non-causal cases at D = 128. Each case has a fixed seed
derived from its name. A case passes when every valid output element is within
`bound()` for the tiling the kernel used with no NaN or infinity, rows that belong
to no request and memory outside the views are untouched (canary bytes), and the
split-count variants agree within the sum of their bounds.

**CPU use of `run_grid.sh`.** `--jobs` (the CPU workers of the bound pool, and the
CPU quota of the CPU containers) defaults to the smaller of 4 and a quarter of
`nproc`, at least 1. Every container runs with `OMP_NUM_THREADS`,
`MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` set to 1 unless the calling
environment sets them: a numerical library that starts one thread per core in each
of several workers multiplies the load by the core count (on a 24-core host,
`--jobs 8` with unrestricted torch threads reached a load average of 99). Check
`uptime` before a run on a shared host and keep `--jobs` low there.

**Kernel-log messages.** The two sanitizer controls (`ctl-memcheck`,
`ctl-initcheck`) run planted errors that `compute-sanitizer` must report (exit 86).
The planted out-of-bounds kernel of `ctl-memcheck` is a real fault while it runs,
and on bare metal the driver logs it for the GPU under test:

```text
NVRM: Xid (PCI:<bus id>): 43, pid=<pid>, name=python3, channel 0x<channel>
```

This **Xid 43 is expected** there. It is an application-level fault raised by the
planted control, not a hardware fault. To tell it from a real problem:

- **Timing.** The line appears only while the `ctl-memcheck` step runs (compare its
  timestamp with the step's start and duration in `steps.tsv`) and at no other time.
- **Process.** Its `pid` is that of the `python3` test process of that step, and no
  other process is named.
- **Nothing else.** No PCIe advanced-error-reporting message, no "fallen off the bus"
  line (Xid 79) and no other Xid accompany it.
- **Healthy afterwards.** `nvidia-smi` still answers for that GPU, and the steps
  after the control (`ctl-initcheck`, `graph`) pass.

Any Xid outside the `ctl-memcheck` step, any Xid other than 43, and an Xid 43 that
does not meet the four conditions is a real problem: stop the run and look at the
driver, the card and the host before trusting any result.

**Other GPU checks.** `tests/gpu/attn_stress.py` repeats launches and requires one
output digest (on the shipped library and on a build that inserts random sleeps at
the producer/consumer handshakes); `compute-sanitizer racecheck` cannot model the
named-barrier handshake, so this stress test is the evidence for it.
`tests/gpu/attn_ab.py` and `attn_dev_timing.py` time library variants interleaved;
the timing used for the published kernel numbers is [`bench/k_run.sh`](../bench/k_run.sh)
([`benchmark-methodology.md`](benchmark-methodology.md)).

## 7. What has been run on a GPU

The GPU tier ran on commit `bd4b3c8`, a pre-publication tree whose kernel and
package sources differ from this release only in comments, docstrings and the
text of two error messages (compiled SASS identical; `csrc/` differs in comments
only), on 2026-10-20: one RTX 3080 20 GB at 220 W,
driver 615.71.09, the dev image of section 3, `run_grid.sh --jobs 4` with one
library thread. The 328-case grid passes on the tensor-core kernel (tightness
maximum 0.905) and on the scalar reference kernel (maximum 0.876), 328/328 each,
with 0 violations and 0 non-finite values; the window family passes 70/70 on both.
The quick tier is 140 tests and the full tier 752, all passed; the four sanitizer
tools report 0 errors on 14 tests each; both planted controls are reported
(exit 86); the graph tier passes 15 tests. The scripted steps took about 21 minutes
(the CPU reference values about 4 minutes), with a 1-minute load average of at most
8.2 on a 24-core host. An external watchdog stopped `run_grid.sh` at the expected
Xid 43 of the `ctl-memcheck` step, so the script's own report step did not run:
`graph` was run alone with the same command line, and the 328/328 figures were
computed from the JUnit and record files of the steps. The containers ran through a
shim that added a CPU limit and the thread variables.

A plugin smoke test on unmodified vLLM 0.31.0 (`pip install --no-build-isolation .`,
entry point registered, `--attention-backend CUSTOM --kv-cache-dtype
int8_per_token_head`, a 1B model at head size 64, six greedy prompts) gave
token-identical output to vLLM's Triton int8 path on 5 of 6 prompts (the sixth
diverged after a near-tie at token 41 of 48) with the same KV pool size.

The shapes that ran are listed in the "Exercised on a GPU" table of the
[README](../README.md#requirements-and-limits). sm80 and sm89 builds pass the
compile, register, spill and SASS gates and the CPU suites; no GPU test run or
timing on those two architectures exists.

## 8. Troubleshooting a build

See [`troubleshooting.md`](troubleshooting.md): a CUDA major mismatch between
nvcc and PyTorch, a missing `ninja`, a missing library at import, and a GPU of an
unsupported generation.
