# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Selection, registration and the rejection matrix of the ``CUSTOM`` backend.

What is protected:

* the plugin registers the class under ``AttentionBackendEnum.CUSTOM`` and
  ``get_name()`` is ``"CUSTOM"``, because the attention layer resolves the
  selected class with ``AttentionBackendEnum[get_name()]``; a different name
  raises ``KeyError`` there (the negative control below shows it);
* registering twice is harmless and never replaces another plugin's class;
* ``validate_configuration`` (what vLLM's platform calls for an explicit
  selection) accepts the production configuration and gives a specific
  reason for every row of the feature matrix that is outside v1;
* an explicitly selected invalid backend ERRORS (``ValueError``), it does not
  fall back; automatic selection never picks ``CUSTOM``;
* the entry point is declared in ``pyproject.toml`` and, when the package is
  installed, resolvable by vLLM's own loader.

Runs on CPU against the installed vLLM (0.31.0 is the tested version, the code
was written against 0.30.0); the whole module is skipped when vLLM is absent.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import pytest
import tomllib
import torch

pytest.importorskip("vllm")

from pth_flash_attn import ops, plugin  # noqa: E402
from pth_flash_attn.vllm_backend import PthFlashAttnBackend  # noqa: E402
from vllm.platforms.interface import DeviceCapability  # noqa: E402
from vllm.v1.attention.backend import AttentionType  # noqa: E402
from vllm.v1.attention.backends import registry  # noqa: E402
from vllm.v1.attention.backends.registry import (  # noqa: E402
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.attention.selector import AttentionSelectorConfig  # noqa: E402

try:
    # The CUDA platform module loads libcuda at import: absent in the CPU-only
    # CI container. The platform-level tests below then skip, and run on a
    # machine with a GPU (tests/cpu is collected there too).
    from vllm.platforms.cuda import CudaPlatformBase
except ImportError:
    CudaPlatformBase = None

needs_cuda_platform = pytest.mark.skipif(
    CudaPlatformBase is None, reason="vllm.platforms.cuda needs libcuda (no GPU)"
)

REPO = Path(__file__).resolve().parents[2]
CLASS_PATH = "pth_flash_attn.vllm_backend.PthFlashAttnBackend"
SM86 = DeviceCapability(8, 6)

# The production selector config of the full-attention layers of the Qwen3.8
# hybrid model: D = 256, bf16, int8 KV, the 816-token hybrid page, decoder.
PROD = {
    "head_size": 256,
    "dtype": torch.bfloat16,
    "kv_cache_dtype": "int8_per_token_head",
    "block_size": 816,
    "use_mla": False,
    "has_sink": False,
    "use_sparse": False,
    "use_mm_prefix": False,
    "use_per_head_quant_scales": False,
    "attn_type": AttentionType.DECODER,
}


@pytest.fixture(autouse=True)
def registered(monkeypatch: pytest.MonkeyPatch):
    """Each test starts from a registry holding exactly our registration.

    The op library may be absent on a CPU checkout (``supports_compute_capability``
    needs it); the matrix tests that depend on it say so themselves.
    """
    monkeypatch.setattr(registry, "_ATTN_OVERRIDES", {}, raising=True)
    plugin.register()
    yield


def reasons(**overrides) -> list[str]:
    """``validate_configuration`` for the production config with overrides."""
    cfg = {**PROD, **overrides}
    cap = cfg.pop("device_capability", SM86)
    return PthFlashAttnBackend.validate_configuration(device_capability=cap, **cfg)


needs_op = pytest.mark.skipif(
    not ops.is_available(), reason="op library _C.abi3.so not built"
)


def test_registration_and_name() -> None:
    """CUSTOM resolves to our class and get_name() is the enum member's name."""
    assert AttentionBackendEnum.CUSTOM.get_path() == CLASS_PATH
    assert AttentionBackendEnum.CUSTOM.get_class() is PthFlashAttnBackend
    assert PthFlashAttnBackend.get_name() == "CUSTOM"
    # What Attention.__init__ does with the selected class.
    assert AttentionBackendEnum[PthFlashAttnBackend.get_name()] is (
        AttentionBackendEnum.CUSTOM
    )


def test_a_name_of_our_own_would_not_resolve() -> None:
    """Negative control: a project-specific name such as ``PTH_BACKEND`` is unknown."""
    with pytest.raises(ValueError, match="Unknown attention backend"):
        AttentionBackendEnum["PTH_BACKEND"]


def test_register_is_idempotent() -> None:
    plugin.register()
    plugin.register()
    assert AttentionBackendEnum.CUSTOM.get_path() == CLASS_PATH


def test_register_does_not_replace_another_plugin(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A CUSTOM registered by someone else stays; we warn instead of clobbering."""
    register_backend(AttentionBackendEnum.CUSTOM, "other.pkg.TheirBackend")
    with caplog.at_level("WARNING"):
        plugin.register()
    assert AttentionBackendEnum.CUSTOM.get_path() == "other.pkg.TheirBackend"
    assert "already registered" in caplog.text


def test_entry_point_is_declared() -> None:
    """pyproject.toml declares exactly one vllm.general_plugins entry point."""
    meta = tomllib.loads((REPO / "pyproject.toml").read_text())
    eps = meta["project"]["entry-points"]["vllm.general_plugins"]
    assert eps == {"pth_flash_attn": "pth_flash_attn.plugin:register"}


def test_entry_point_loads_through_vllm_when_installed() -> None:
    """When the package is installed, the entry point is loadable."""
    found = [
        ep
        for ep in importlib.metadata.entry_points(group="vllm.general_plugins")
        if ep.name == "pth_flash_attn"
    ]
    if not found:
        pytest.skip("pth-flash-attn is not pip-installed in this environment")
    assert found[0].value == "pth_flash_attn.plugin:register"
    found[0].load()()  # the very call vLLM's loader makes
    assert AttentionBackendEnum.CUSTOM.get_path() == CLASS_PATH


def test_importing_the_plugin_does_not_import_the_backend() -> None:
    """register() is cheap: no CUDA op library, no Triton modules at plugin load."""
    import subprocess
    import sys

    probe = (
        "import sys\n"
        "from pth_flash_attn import plugin\n"
        "plugin.register()\n"
        "assert 'pth_flash_attn.vllm_backend' not in sys.modules\n"
        "assert 'pth_flash_attn.ops' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", probe], check=True, cwd=REPO / "python")


@needs_op
def test_production_configuration_is_accepted() -> None:
    assert reasons() == []


@needs_op
def test_fp16_and_other_supported_head_sizes_are_accepted() -> None:
    assert reasons(dtype=torch.float16) == []
    for hs in (64, 128, 256):
        assert reasons(head_size=hs) == []


@needs_op
@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"head_size": 80}, "head_size not supported"),
        ({"head_size": 512}, "head_size not supported"),
        ({"dtype": torch.float32}, "dtype not supported"),
        ({"kv_cache_dtype": "auto"}, "kv_cache_dtype not supported"),
        ({"kv_cache_dtype": "fp8"}, "kv_cache_dtype not supported"),
        ({"kv_cache_dtype": "int4_per_token_head"}, "kv_cache_dtype not supported"),
        ({"kv_cache_dtype": "fp8_per_token_head"}, "kv_cache_dtype not supported"),
        ({"block_size": 100}, "block_size not supported"),
        ({"block_size": 8}, "block_size not supported"),
        ({"has_sink": True}, "attention sinks not supported"),
        ({"use_mm_prefix": True}, "partial multimodal token full attention"),
        ({"use_mla": True}, "MLA not supported"),
        ({"use_sparse": True}, "sparse not supported"),
        ({"use_per_head_quant_scales": True}, "per-head quant scales"),
        ({"attn_type": AttentionType.ENCODER}, "attention type"),
        ({"attn_type": AttentionType.ENCODER_ONLY}, "attention type"),
        ({"attn_type": AttentionType.ENCODER_DECODER}, "attention type"),
        # PROD is D = 256; windowed kernels exist at D = 128 only.
        ({"has_sliding_window": True}, "sliding window at head size 256"),
        ({"use_rswa": True}, "R-SWA not supported"),
        ({"use_rswa": True, "head_size": 128}, "R-SWA not supported"),
        ({"use_non_causal": True}, "non-causal attention at head size 256"),
        (
            {"has_sliding_window": True, "head_size": 64},
            "sliding window at head size 64",
        ),
        (
            {"use_non_causal": True, "head_size": 64},
            "non-causal attention at head size 64",
        ),
        ({"use_batch_invariant": True}, "batch invariance not supported"),
        ({"use_pcp": True}, "PCP not supported"),
        ({"use_dcp": True}, "DCP not supported"),
        ({"device_capability": DeviceCapability(7, 5)}, "compute capability"),
        ({"device_capability": DeviceCapability(9, 0)}, "compute capability"),
        ({"device_capability": DeviceCapability(10, 0)}, "compute capability"),
    ],
    ids=lambda v: v if isinstance(v, str) else str(sorted(v)),
)
def test_rejection_matrix(overrides: dict, reason: str) -> None:
    """Every row of the rejected feature matrix has its own reason."""
    got = reasons(**overrides)
    assert got, f"{overrides} was accepted"
    assert any(reason in r for r in got), (overrides, got)


@needs_op
def test_sm80_sm86_sm89_are_accepted() -> None:
    for cap in (DeviceCapability(8, 0), DeviceCapability(8, 6), DeviceCapability(8, 9)):
        assert reasons(device_capability=cap) == []


# ------------------------------------------------- speculative-decoding drafter

# The drafter's layers: D = 128, bf16, int8 KV, the 864-token
# page of its sliding-window group, decoder, window 2048, non-causal.
DRAFTER = {
    **PROD,
    "head_size": 128,
    "block_size": 864,
    "has_sliding_window": True,
    "use_non_causal": True,
}


@pytest.fixture
def op_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Python selection logic without the built library (CPU checkout)."""
    monkeypatch.setattr(ops, "is_available", lambda: True)


def test_window_and_non_causal_flags() -> None:
    """Sliding window and non-causal are supported; R-SWA, sinks, ALiBi,
    mm-prefix and batch invariance stay refused."""
    assert PthFlashAttnBackend.supports_sliding_window() is True
    assert PthFlashAttnBackend.supports_non_causal() is True
    assert PthFlashAttnBackend.supports_rswa() is False
    assert PthFlashAttnBackend.supports_sink() is False
    assert PthFlashAttnBackend.supports_alibi_sqrt() is False
    assert PthFlashAttnBackend.supports_mm_prefix() is False
    assert PthFlashAttnBackend.supports_batch_invariance() is False


def test_drafter_configuration_is_accepted(op_available) -> None:
    """D = 128 windowed, non-causal, alone or together: no reason."""
    assert reasons(**DRAFTER) == []
    for flags in (
        {"has_sliding_window": True},
        {"use_non_causal": True},
        {"has_sliding_window": True, "use_non_causal": True},
    ):
        assert reasons(**{**PROD, "head_size": 128, **flags}) == [], flags


@pytest.mark.parametrize("head_size", [64, 256])
def test_windowed_or_non_causal_off_d128_is_refused(op_available, head_size) -> None:
    """Only D = 128 has windowed kernels: at D = 64 and at the
    D = 256 MTP target each flag gets its own reason, and the causal
    configuration without window is still accepted at both sizes."""
    assert reasons(head_size=head_size) == []
    got = reasons(**{**DRAFTER, "head_size": head_size})
    why = f"at head size {head_size} (the windowed kernels exist for D = 128 only)"
    assert got == [f"sliding window {why}", f"non-causal attention {why}"]


def test_unsupported_head_size_gets_only_the_base_reason(op_available) -> None:
    """D = 80 with a window: vLLM's head_size reason, not a window reason too."""
    got = reasons(**{**DRAFTER, "head_size": 80})
    assert got == ["head_size not supported"]


def test_without_the_op_library_the_backend_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A checkout without _C.abi3.so must be refused at selection, not at launch."""
    monkeypatch.setattr(ops, "is_available", lambda: False)
    got = reasons()
    assert any("compute capability" in r for r in got), got


def test_adaptive_verification_is_supported() -> None:
    """The kernel reads q_len on the device: the mismatch is allowed."""
    assert PthFlashAttnBackend.supports_device_cpu_query_lens_mismatch() is True
    got = PthFlashAttnBackend.validate_configuration(
        device_capability=SM86, use_adaptive_verification=True, **PROD
    )
    assert not any("query lens" in r for r in got)


def test_cache_spec_layout_and_writer_are_tritons() -> None:
    """The backend inherits the cache contract; only the read side is ours."""
    from vllm.v1.attention.backends.triton_attn import (
        TritonAttentionBackend,
        TritonAttentionImpl,
    )

    assert issubclass(PthFlashAttnBackend, TritonAttentionBackend)
    assert (
        PthFlashAttnBackend.customize_spec.__func__
        is TritonAttentionBackend.customize_spec.__func__
    )
    assert PthFlashAttnBackend.forward_includes_kv_cache_update is False
    impl = PthFlashAttnBackend.get_impl_cls()
    assert issubclass(impl, TritonAttentionImpl)
    assert impl.do_kv_cache_update is TritonAttentionImpl.do_kv_cache_update
    assert impl._pth_key_value_caches is TritonAttentionImpl._pth_key_value_caches


def test_builder_is_not_tritons() -> None:
    """Triton's builder allocates scratch; this backend has its own."""
    from vllm.v1.attention.backend import AttentionMetadataBuilder
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder

    builder = PthFlashAttnBackend.get_builder_cls()
    assert not issubclass(builder, TritonAttentionMetadataBuilder)
    assert issubclass(builder, AttentionMetadataBuilder)


def test_kernel_block_sizes_accept_the_hybrid_page() -> None:
    """Any multiple of 16 is a kernel block size, e.g. the hybrid group's 816."""

    sizes = PthFlashAttnBackend.get_supported_kernel_block_sizes()
    assert [s.base for s in sizes] == [16]
    # The 0.31 signature (an optional spec argument) works too.
    assert [
        s.base for s in PthFlashAttnBackend.get_supported_kernel_block_sizes(None)
    ] == [16]
    assert PthFlashAttnBackend.supports_block_size(816)
    assert PthFlashAttnBackend.supports_block_size(16)
    assert not PthFlashAttnBackend.supports_block_size(24)


# ---------------------------------------------------------------- platform


@pytest.fixture
def cuda_sm86(monkeypatch: pytest.MonkeyPatch):
    if CudaPlatformBase is None:
        pytest.skip("vllm.platforms.cuda needs libcuda (no GPU)")
    """The CUDA platform's selector as on a 3080, without a GPU."""
    monkeypatch.setattr(
        CudaPlatformBase,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: SM86),
    )


def _selector(**overrides) -> AttentionSelectorConfig:
    return AttentionSelectorConfig(**{**PROD, **overrides})


@needs_op
def test_explicit_selection_returns_our_class_path(cuda_sm86) -> None:
    path = CudaPlatformBase.get_attn_backend_cls(
        AttentionBackendEnum.CUSTOM, _selector(), num_heads=12
    )
    assert path == CLASS_PATH


@needs_op
@pytest.mark.parametrize("num_heads", [16, 32], ids=["tp2_16_4", "tp1_32_8"])
def test_explicit_selection_of_the_drafter_returns_our_class_path(
    cuda_sm86, num_heads: int
) -> None:
    """``speculative_config.attention_backend=CUSTOM`` for the drafter."""
    path = CudaPlatformBase.get_attn_backend_cls(
        AttentionBackendEnum.CUSTOM, _selector(**DRAFTER), num_heads=num_heads
    )
    assert path == CLASS_PATH


@needs_op
@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"has_sliding_window": True}, "sliding window at head size 256"),
        ({"use_non_causal": True}, "non-causal attention at head size 256"),
        ({"use_rswa": True}, "R-SWA not supported"),
        ({"kv_cache_dtype": "auto"}, "kv_cache_dtype not supported"),
        ({"head_size": 80}, "head_size not supported"),
        ({"use_mm_prefix": True}, "partial multimodal token full attention"),
    ],
)
def test_explicit_invalid_selection_errors_and_names_the_reason(
    cuda_sm86, overrides: dict, reason: str
) -> None:
    """An explicitly selected invalid backend raises; it does not fall back."""
    with pytest.raises(ValueError, match="Selected backend") as info:
        CudaPlatformBase.get_attn_backend_cls(
            AttentionBackendEnum.CUSTOM, _selector(**overrides), num_heads=12
        )
    assert reason in str(info.value)


@needs_cuda_platform
def test_explicit_selection_on_an_unsupported_gpu_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        CudaPlatformBase,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(9, 0)),
    )
    monkeypatch.setattr(ops, "is_available", lambda: True)
    with pytest.raises(ValueError, match="compute capability"):
        CudaPlatformBase.get_attn_backend_cls(
            AttentionBackendEnum.CUSTOM, _selector(), num_heads=12
        )


def test_unregistered_custom_errors_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the plugin, --attention-backend CUSTOM says it is not registered."""
    monkeypatch.setattr(registry, "_ATTN_OVERRIDES", {}, raising=True)
    with pytest.raises(ValueError, match="must be registered"):
        AttentionBackendEnum.CUSTOM.get_class()


def test_automatic_selection_never_offers_custom() -> None:
    """The backend is opt-in: no platform priority list names ``CUSTOM``.

    The CUDA platform builds its candidate list from the enum members it
    spells out in ``cuda.py``; a backend that is not in that source can only be
    chosen with ``--attention-backend CUSTOM``. Read as text: importing the
    module needs libcuda.
    """
    import importlib.util

    spec = importlib.util.find_spec("vllm.platforms.cuda")
    assert spec is not None and spec.origin is not None
    assert "AttentionBackendEnum.CUSTOM" not in Path(spec.origin).read_text()
