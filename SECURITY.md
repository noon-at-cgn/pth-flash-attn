<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn -->
# Security policy

## Supported versions

Only the latest release of the 0.x series receives security fixes. At the time
of writing that is 0.1.x; see [CHANGELOG.md](CHANGELOG.md).

| Version | Supported |
|---|---|
| latest 0.x | yes |
| older 0.x releases | no |

The project is maintained on a best-effort basis (see
[CONTRIBUTING.md](CONTRIBUTING.md)); no response or fix time is promised.

## Reporting a vulnerability

Report privately through GitHub's private vulnerability reporting, which the
maintainers enable for this repository: open the repository's **Security** tab,
choose **Report a vulnerability** and fill in the form. If the Security tab
shows no reporting form, open a minimal public issue that gives no details of
the vulnerability and asks for a private channel; the maintainers will then enable
private reporting and ask for the details there. Otherwise do not open a public issue, pull request or discussion for a
suspected vulnerability, and do not post a reproducer publicly before a fix is
available.

A useful report contains the version or commit, the GPU model and compute
capability, the PyTorch, CUDA and driver versions, the exact call (the shapes,
dtypes and strides of every tensor argument) and what happens. No email address
is offered for reports.

## Scope

In scope:

- **Memory safety in the CUDA kernels** (`csrc/pth_attn/`): out-of-bounds
  global or shared-memory access, uninitialised reads that reach the output, or
  races, for arguments that the op accepts.
- **Argument validation in the op** (`csrc/pth_attn/ops.h`, `torch_bindings.cpp`,
  `python/pth_flash_attn/ops.py`): a combination of shape, dtype, stride,
  storage offset or integer value that passes the host-side checks but makes the
  kernel access memory outside the supplied views, including `int64` overflow in
  address computation.
- **Build and packaging**: an installed package or source distribution that
  contains or runs something other than what the repository defines.

Out of scope:

- **Inconsistent device metadata.** The op validates tensors on the host and
  does not read device memory on the host. The contents of `block_table` and
  `seq_lens` are trusted to be consistent with the cache and the query
  ([`docs/integration.md`](docs/integration.md)); an engine that passes block
  ids outside the cache or sequence lengths beyond the table can cause
  out-of-bounds reads. That is a precondition of the interface, not a
  vulnerability. A host-side check that is missing for a property the host can
  see is in scope.
- **Numerical accuracy** within or slightly outside the documented bound is a
  bug, not a vulnerability; use the bug-report form.
- **Unsupported configurations** that are refused with an error.
- **Vulnerabilities in vLLM, PyTorch, CUDA, the driver or other dependencies**:
  report them to those projects.
- **Test, benchmark and development tooling** (`tests/`, `bench/`, `tools/`,
  `docker/`) run on inputs the user controls, and the contents of the dev images.
- **Denial of service by resource use** with valid arguments (for example a very
  long sequence), and attacks that need control of the process or the GPU.

## Run-time behaviour

The package opens no network connections and reads no environment variables at
run time; the only file it loads is
its own compiled extension. The op allocates no device memory: the caller
supplies the workspaces ([`docs/integration.md`](docs/integration.md)).
