"""Unit tests for the qwen3.5 MTP registry alias patch.

megatron-bridge 0.5.0's Qwen3_5 bridges register MTP mappings under
``...mtp.layers.*.mtp_model_layer.*`` while megatron-core builds the module as
``...mtp.layers.*.transformer_layer.*`` — the skew broke both bridge load
(None conversion task) and save (single-shard never completes). The patch in
``slime_plugins/megatron_bridge/qwen3_5_mtp_alias.py`` wraps
``MegatronMappingRegistry.__init__`` to add a ``transformer_layer``-renamed
shallow-copy alias for every such mapping. These tests pin that behavior
against future megatron-bridge bumps (if upstream fixes the naming, the alias
must keep deduping instead of double-registering).

Slime-required (needs megatron.bridge from the training image).

Usage:
    pytest tests/train/export/test_qwen3_5_mtp_alias.py -v
"""

from __future__ import annotations

import pytest

pytest.importorskip("megatron.bridge", reason="megatron-bridge not installed (training image only)")

import slime_plugins.megatron_bridge.qwen3_5_mtp_alias  # noqa: F401,E402  (applies the patch)
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry  # noqa: E402
from megatron.bridge.models.conversion.param_mapping import AutoMapping, QKVMapping  # noqa: E402

MTP_OLD = "language_model.mtp.layers.0.mtp_model_layer.self_attention.linear_proj.weight"
MTP_BUILT = "language_model.mtp.layers.0.transformer_layer.self_attention.linear_proj.weight"
HF_TARGET = "mtp.layers.0.self_attn.o_proj.weight"


def test_exact_match_alias_resolves_built_name():
    """An exact-match (no-wildcard) mtp_model_layer mapping gains a
    transformer_layer alias that resolves to the SAME HF target."""
    reg = MegatronMappingRegistry(AutoMapping(MTP_OLD, HF_TARGET))
    aliased = reg.megatron_to_hf_lookup(MTP_BUILT)
    assert aliased is not None, "built-name lookup must resolve via the alias"
    assert str(aliased.hf_param) == HF_TARGET
    # the original (dead) name still resolves — harmless, unchanged behavior
    assert reg.megatron_to_hf_lookup(MTP_OLD) is not None


def test_wildcard_qkv_alias_preserves_type_and_resolves_captures():
    """A wildcard QKVMapping alias keeps its concrete type and resolves the
    layer index into the HF q/k/v targets (resolve() builds a fresh instance,
    so the shallow copy's stale internals are never used)."""
    reg = MegatronMappingRegistry(
        QKVMapping(
            megatron_param="language_model.mtp.layers.*.mtp_model_layer.self_attention.linear_qkv.weight",
            q="mtp.layers.*.self_attn.q_proj.weight",
            k="mtp.layers.*.self_attn.k_proj.weight",
            v="mtp.layers.*.self_attn.v_proj.weight",
        )
    )
    built = "language_model.mtp.layers.0.transformer_layer.self_attention.linear_qkv.weight"
    m = reg.megatron_to_hf_lookup(built)
    assert m is not None
    assert isinstance(m, QKVMapping), f"alias must stay a QKVMapping, got {type(m)}"
    assert m.hf_param["q"] == "mtp.layers.0.self_attn.q_proj.weight"


def test_dedup_when_both_names_already_registered():
    """GLM/MIMO-style bridges register BOTH names themselves — the patch must
    not inject a duplicate transformer_layer mapping."""
    reg = MegatronMappingRegistry(
        AutoMapping(MTP_OLD, HF_TARGET),
        AutoMapping(MTP_BUILT, HF_TARGET),
    )
    mtp_mappings = [m for m in reg.mappings if "mtp" in m.megatron_param]
    assert len(mtp_mappings) == 2, (
        f"expected exactly the 2 hand-registered mappings, got "
        f"{[m.megatron_param for m in mtp_mappings]}"
    )


def test_non_mtp_mappings_untouched():
    """Mappings without mtp_model_layer must pass through with no aliases."""
    reg = MegatronMappingRegistry(
        AutoMapping("decoder.layers.*.self_attention.linear_proj.weight",
                    "model.layers.*.self_attn.o_proj.weight")
    )
    assert len(reg.mappings) == 1
    assert reg.megatron_to_hf_lookup(
        "decoder.layers.3.transformer_layer.self_attention.linear_proj.weight"
    ) is None


def test_patch_is_idempotent():
    """Re-importing the patch module must not double-wrap __init__ (which
    would double-alias every mtp mapping)."""
    import importlib

    import slime_plugins.megatron_bridge.qwen3_5_mtp_alias as alias_mod

    before = MegatronMappingRegistry.__init__
    importlib.reload(alias_mod)
    assert MegatronMappingRegistry.__init__ is before, "reload must not re-wrap"
    reg = MegatronMappingRegistry(AutoMapping(MTP_OLD, HF_TARGET))
    built_aliases = [m for m in reg.mappings if "transformer_layer" in m.megatron_param]
    assert len(built_aliases) == 1, "exactly ONE alias per mtp mapping"
