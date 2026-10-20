<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn -->
## Summary

<!-- What the change does and why. Link the issue that describes the problem, if there is one. -->

## Validation

<!-- Tick what applies; say "not applicable" and why for the rest. See CONTRIBUTING.md. -->

- [ ] `tools/run_ci_local.sh` passes (all groups), or the groups run are named:
- [ ] CPU tests: `pytest tests/cpu tests/sim` and `tests/cpp/run_geometry.sh`
- [ ] Kernel change: ptxas gate (`tools/ptxas_gate.py`, `tools/attn_ci_gates.py`) and SASS lint (`tools/sass_check.py`)
- [ ] Behaviour-preserving kernel refactor: SASS equivalence shown with `tools/sass_diff.py` or `tools/prove_sass_equal.sh` on sm80, sm86 and sm89
- [ ] Documentation updated where behaviour, limits or commands changed
- [ ] Performance claim: measured as in `docs/benchmark-methodology.md` (effect above twice the A/A minimum detectable effect), numbers labelled with hardware, power limit and software versions

## Hardware tested

<!-- GPU model, compute capability (sm80, sm86 or sm89), driver, CUDA, PyTorch and vLLM versions; which of tests/gpu ran. Write "none" if no GPU was available. -->

## Vendored subset

<!-- The subset is printed by `python tools/export_upstream.py --list`. -->

- [ ] This change does not touch the vendored subset
- [ ] It touches the subset: no exported path moved, and the change is
  - [ ] comment-only
  - [ ] behaviour-preserving
  - [ ] functional

## Sign-off

- [ ] Every commit carries a `Signed-off-by:` line (`git commit -s`), certifying the [Developer Certificate of Origin](https://developercertificate.org/)
- [ ] No secrets, credentials, private prompts or model weights are included
