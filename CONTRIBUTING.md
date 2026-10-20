<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn -->
# Contributing to pth-flash-attn

Thank you for considering a contribution. This page states what is welcome,
how to set up, which checks a pull request has to pass, and what extra evidence
a kernel change needs.

## Support

This project is maintained on a best-effort basis. Issues and pull requests are
read when time allows; there is no support service, no response-time commitment
and no private support channel. Security reports follow [SECURITY.md](SECURITY.md): the
maintainers enable GitHub private vulnerability reporting for the repository, and no
email address is offered.

## What is welcome

- **Bug reports** with a reproducer: wrong output, a crash, a refusal that
  looks wrong, a build failure. The bug-report form asks for the GPU, its
  compute capability, the driver, CUDA, PyTorch and vLLM versions and the exact
  command.
- **Hardware reports.** Only sm86 has been measured and run on a GPU (RTX 3080
  20 GB for the kernel and the GPU tests, an RTX 3090 for independent end-to-end
  runs). Results on sm80 (A100) and sm89 (Ada: RTX 40xx, L4, L40S) are
  valuable: the GPU test tier passing or failing, timings with the method of
  [`docs/benchmark-methodology.md`](docs/benchmark-methodology.md) (end-to-end
  numbers follow [Performance reports](#performance-reports)), and the
  output of `tools/collect_env.sh`.
- **Integration reports and contracts for other engines.** The op is
  engine-independent; [`docs/integration.md`](docs/integration.md) states what
  an engine has to provide. Reports of what worked or failed in another engine
  or a vLLM-derived fork, and corrections to that contract, are welcome.
- **Documentation fixes**: errors, unclear passages, broken links.
- **Kernel and planner changes** are welcome after an issue has described the
  problem and the intended approach, because they are constrained by the
  correctness and performance checks below. Large unannounced changes are likely
  to be declined.

Out of scope: configurations outside the supported matrix, which the op and
the backend refuse on purpose (see the table in the [README](README.md#requirements-and-limits)),
a backward pass, and other cache formats. Open an issue before widening the
supported contract (a head size, window, dtype or layout that the op refuses
today).

## Setting up

- [`docs/build.md`](docs/build.md): requirements (PyTorch, nvcc, g++), the
  install, the architecture list and the dev image.
  It also describes the test tiers, what runs where, and how to build the test
  libraries and run the GPU tier.

All CPU checks run in the dev image built from
[`docker/dev.Dockerfile`](docker/dev.Dockerfile), offline and without a GPU:

```bash
tools/run_ci_local.sh              # every CPU check, as CI runs them
tools/run_ci_local.sh lint test    # only the named groups
```

The script runs each step even if an earlier one failed and prints a summary.
The header of the script lists the groups. CI
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the same script,
so there is one definition of the checks.

The pre-commit configuration ([`.pre-commit-config.yaml`](.pre-commit-config.yaml))
runs the lint hooks locally using the tools already installed in the lint
image, and adds the sign-off line automatically:

```bash
pre-commit install --hook-type pre-commit --hook-type commit-msg
pre-commit run --all-files
```

## Checks a pull request must pass

CI must be green. The checks, with the tool that runs each:

| Check | Tool |
|---|---|
| Python lint and format | `ruff check .` and `ruff format --check .` (configuration in `pyproject.toml`) |
| Spelling | `typos` |
| Types | `mypy` (files listed in `pyproject.toml`) |
| C++ and CUDA format | `clang-format --style=file` ([`.clang-format`](.clang-format)) |
| Shell | `shellcheck -s bash` |
| Source headers | `python tools/repo_checks.py spdx`: every source file starts with `SPDX-License-Identifier: Apache-2.0` and a `SPDX-FileCopyrightText:` line naming `noon-at-cgn` |
| Engine-facing code | `python tools/repo_checks.py engine-facing`: no `torch.cuda` call and no environment-variable read in `python/pth_flash_attn/*.py` |
| Sign-off | `python tools/repo_checks.py signoff` (see below) |
| CPU tests | `pytest tests/cpu tests/sim` and `tests/cpp/run_geometry.sh` |
| Build | the op extension for sm80, sm86 and sm89 builds without a GPU |
| Register, spill and SASS gates | `tools/ptxas_gate.py`, `tools/attn_ci_gates.py`, `tools/sass_check.py` |

## Conventions

- **Python**: line length 88, target Python 3.10, rule sets `E F UP B ISC SIM I G`
  and, in `python/`, `INP` and `D` (`pyproject.toml`). Public functions in the
  installed package need Google-style docstrings (`pydocstyle` with the
  `google` convention). Tests, tools and benchmarks are scripts and are exempt
  from the docstring rule, but a test states in its docstring the property it
  protects.
- **C++ and CUDA**: Google style, 2-space indent, 80 columns, as in
  [`.clang-format`](.clang-format). Inline PTX is allowed where it is needed.
- **Headers**: every source file carries the two SPDX lines. The project is
  Apache-2.0 and new files use `Copyright (c) 2026 noon-at-cgn`.
- **No environment variables and no `torch.cuda` in engine-facing code**
  (`python/pth_flash_attn/*.py`): the op takes everything as arguments.
- **Performance claims** need a measurement that follows
  [`docs/benchmark-methodology.md`](docs/benchmark-methodology.md): an effect
  must exceed twice the minimum detectable effect measured by an A/A run on the
  same machine, with the hardware, power limit and software versions stated.
- **British spelling and an impersonal voice** in documentation and comments.

## Performance reports

An end-to-end speed, latency or memory result is comparable with another one
only when it states the following. The issue form
[Performance report](.github/ISSUE_TEMPLATE/performance_report.yml) asks for
each item; the statistics are in
[`docs/benchmark-methodology.md`](docs/benchmark-methodology.md).

- **Same inputs in every arm.** The same weights (checkpoint and revision), the
  same prompts (the exact text, or the script and seed that generates it) and
  the same sampler (greedy, or the stated parameters and seed).
- **Warm-ups and measured runs, counted.** Warm-up requests are stated and
  discarded; the number of measured runs and of boots per arm is stated.
- **Decode rate and wall rate, separately.** Decode tokens per second over the
  decode window and end-to-end tokens per second including prefill are
  different figures. Time to first token (TTFT) is its own figure, cold and warm.
- **Spread, not one number.** The median and the range, or a confidence
  interval, over the measured runs; a difference inside the spread of an A/A
  run is not a result.
- **Memory.** Peak VRAM per card, and the KV-cache size in tokens that the
  engine reports.
- **The machine.** GPU model (and any memory modification), power limit,
  interconnect (PCIe generation, peer-to-peer on or off, NVLink), driver and
  CUDA versions, clock locks, background load on the host and the GPUs, and the
  sampling interval of any power or clock log.
- **The software.** Engine and PyTorch versions, the image digest or the
  commit of this repository, and the exact launch command.
- **Never convert kernel microseconds into tokens per second.** The attention
  kernel is a fraction of a decode step; a kernel timing is reported as a
  kernel timing, and a rate is measured end to end.
- **Clean logs.** Remove credentials, private paths and host identifiers (host
  names, addresses, serial numbers, user names) from logs and command lines
  before posting them.

## Sign-off (DCO)

Every commit must carry a `Signed-off-by:` line, certifying the
[Developer Certificate of Origin](https://developercertificate.org/):

```bash
git commit -s
```

`tools/repo_checks.py signoff` fails when a commit in the range lacks the
trailer, and the pre-commit `commit-msg` hook adds it. Commits are not otherwise
decorated: no generated trailers or attribution lines.

## Validating a kernel change

A change under `csrc/` or to the planner needs the following evidence in the
pull request description:

1. **CPU suite.** `tools/run_ci_local.sh` passes, including the emulator and
   bound tests that predict the kernel's error budget.
2. **GPU suite where a GPU is available.** The GPU tier in `tests/gpu/`
   ([`docs/build.md`](docs/build.md), section "The GPU tier") on
   the architecture tested: name the GPU and compute capability (sm80, sm86 or
   sm89). A change that cannot be run on a GPU says so and is validated by the
   CPU tiers and the gates below only.
3. **Compile gates.** `tools/ptxas_gate.py` and `tools/attn_ci_gates.py` (at
   most 255 registers, no spills, no stack frame, every kernel family on every
   architecture) and `tools/sass_check.py` (no local memory, no `I2F` in
   loops, `LDGSTS`, `LDSM` and `HMMA.16816.F32` present).
4. **SASS equivalence for refactors.** A change that claims not to alter the
   machine code is shown to do so with `tools/sass_diff.py` (or
   `tools/prove_sass_equal.sh`) on all three architectures.
5. **Performance.** A change that claims a speed-up or accepts a slow-down
   follows the performance rule above and reports the numbers with their
   labels.

### The vendored subset

A defined subset of the repository (the shipped kernel sources, the Python
package, `setup.py`, `pyproject.toml`, `tools/build_flags.py`, `LICENSE` and
`NOTICE`, and the scripts that export and prove it) is copied by downstream
projects. `python tools/export_upstream.py --list` prints it. The rules:

- Do not move, rename or split the exported paths.
- The list is checked by `tests/cpu/test_upstream_subset.py`, and the export
  with the test hooks stripped must compile to the same SASS as the repository
  (`tools/export_upstream.py`, `tools/prove_export_sass.sh`).
- Say in the pull request whether the change is comment-only, behaviour-
  preserving or functional, so that consumers know whether to re-vendor.

## Pull requests

- Keep a pull request to one topic and describe what it does and why. Fill in
  the template.
- Rebase on `main`; the history is kept linear.
- Do not include secrets, credentials, model weights or personal data in
  commits, logs or issue text, and remove private paths and host identifiers
  from logs. Benchmark logs are summarised in curated CSV
  files under [`bench/results/`](bench/results/README.md) with the labels the schema requires.

## Licence

By contributing you agree that your contribution is licensed under the
[Apache License 2.0](LICENSE) and that you have the right to submit it under
that licence (the DCO above).
