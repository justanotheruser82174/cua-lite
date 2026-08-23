"""Tests for Megatron → HF parameter name conversion (qwen2.py).

Covers pure Qwen2/3 (text-only) and Qwen3-VL (VLM) parameter name mappings.
"""
from __future__ import annotations

import sys
import types

import pytest
import torch

from lite.utils.path import project_root

# slime is a submodule, not installed in the venv, so it has to be put on sys.path to
# import at all — and doing that in a shared test process is hostile in three ways.
# All three are handled here; do not simplify this block back to a bare
# ``sys.path.insert``.
#
# 1. ``insert(0)`` shadows the repo's own ``examples/`` package with the slime
#    checkout's top-level ``examples/`` for every module that runs later in the same
#    process — that is what broke ``test_grounding_examples_protocol``. APPEND: the
#    repo root only contributes a *namespace portion* for ``slime``, which loses to
#    the real regular package under ``slime/slime/``.
# 2. That namespace portion is also why a plain import here is unreliable: any
#    earlier module whose ``importorskip("slime.utils")`` failed leaves
#    ``sys.modules["slime"]`` bound to the repo-root portion, whose ``__path__``
#    has no ``backends``. Purge before importing, or this module skips itself in
#    every full-suite run while passing in isolation.
# 3. Leaving a *working* ``slime`` behind flips the ``importorskip("slime.utils")``
#    guard in ~15 sibling modules from skip to pass, and they then hard-import
#    ``slime.utils.data`` -> ``ray``, which is training-image-only: 49 collection
#    errors. So the path entry and the imported modules are both rolled back.
_SLIME_ROOT = str(project_root() / "slime")


def _purge_slime_modules() -> None:
    for _name in [n for n in sys.modules if n == "slime" or n.startswith("slime.")]:
        del sys.modules[_name]


_purge_slime_modules()
sys.path.append(_SLIME_ROOT)
try:
    pytest.importorskip(
        "slime.backends.megatron_utils.megatron_to_hf.qwen2",
        reason="slime submodule not initialized",
    )
    from slime.backends.megatron_utils.megatron_to_hf.qwen2 import convert_qwen2_to_hf
    from slime.backends.megatron_utils.megatron_to_hf.qwen3_vl import (
        convert_qwen3vl_to_hf,
    )
finally:
    sys.path.remove(_SLIME_ROOT)
    _purge_slime_modules()


@pytest.fixture
def args():
    """Minimal args matching Qwen3-VL-4B-Instruct."""
    a = types.SimpleNamespace()
    a.hidden_size = 2560
    a.num_attention_heads = 32
    a.num_query_groups = 8
    a.kv_channels = 128
    a.vocab_size = 151936
    return a


# ── Pure Qwen2/3 (no VLM wrapper) ──────────────────────────────────────────


class TestPureQwen2:
    def test_embed_tokens(self, args):
        result = convert_qwen2_to_hf(
            args, "module.module.embedding.word_embeddings.weight", torch.zeros(1)
        )
        assert result[0][0] == "model.embed_tokens.weight"

    def test_lm_head(self, args):
        result = convert_qwen2_to_hf(args, "module.module.output_layer.weight", torch.zeros(1))
        assert result[0][0] == "lm_head.weight"

    def test_final_layernorm(self, args):
        result = convert_qwen2_to_hf(
            args, "module.module.decoder.final_layernorm.weight", torch.zeros(1)
        )
        assert result[0][0] == "model.norm.weight"

    def test_o_proj(self, args):
        result = convert_qwen2_to_hf(
            args, "module.module.decoder.layers.5.self_attention.linear_proj.weight", torch.zeros(1)
        )
        assert result[0][0] == "model.layers.5.self_attn.o_proj.weight"

    def test_qkv_split(self, args):
        # Fused QKV: (num_query_groups * (value_num_per_group + 2) * head_dim, hidden_size)
        head_dim = 128
        # value_num_per_group = 32 // 8 = 4 (queries per KV group)
        qkv_size = 8 * (4 + 1 + 1) * head_dim  # 8 * 6 * 128 = 6144
        param = torch.randn(qkv_size, 2560)
        result = convert_qwen2_to_hf(
            args, "module.module.decoder.layers.0.self_attention.linear_qkv.weight", param
        )
        names = [r[0] for r in result]
        assert names == [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
        ]
        # Check shapes: q=(num_heads*head_dim, hidden), k=v=(num_kv_heads*head_dim, hidden)
        assert result[0][1].shape == (32 * 128, 2560)  # q
        assert result[1][1].shape == (8 * 128, 2560)   # k
        assert result[2][1].shape == (8 * 128, 2560)   # v

    def test_gate_up_proj(self, args):
        param = torch.randn(19456, 2560)  # ffn_hidden * 2 = 9728 * 2
        result = convert_qwen2_to_hf(
            args, "module.module.decoder.layers.3.mlp.linear_fc1.weight", param
        )
        assert result[0][0] == "model.layers.3.mlp.gate_proj.weight"
        assert result[1][0] == "model.layers.3.mlp.up_proj.weight"
        assert result[0][1].shape[0] == 9728
        assert result[1][1].shape[0] == 9728

    def test_down_proj(self, args):
        result = convert_qwen2_to_hf(
            args, "module.module.decoder.layers.3.mlp.linear_fc2.weight", torch.zeros(1)
        )
        assert result[0][0] == "model.layers.3.mlp.down_proj.weight"

    def test_input_layernorm(self, args):
        result = convert_qwen2_to_hf(
            args,
            "module.module.decoder.layers.7.self_attention.linear_qkv.layer_norm_weight",
            torch.zeros(1),
        )
        assert result[0][0] == "model.layers.7.input_layernorm.weight"

    def test_post_attn_layernorm(self, args):
        result = convert_qwen2_to_hf(
            args, "module.module.decoder.layers.7.mlp.linear_fc1.layer_norm_weight", torch.zeros(1)
        )
        assert result[0][0] == "model.layers.7.post_attention_layernorm.weight"

    def test_q_norm(self, args):
        result = convert_qwen2_to_hf(
            args, "module.module.decoder.layers.2.self_attention.q_layernorm.weight", torch.zeros(1)
        )
        assert result[0][0] == "model.layers.2.self_attn.q_norm.weight"

    def test_k_norm(self, args):
        result = convert_qwen2_to_hf(
            args, "module.module.decoder.layers.2.self_attention.k_layernorm.weight", torch.zeros(1)
        )
        assert result[0][0] == "model.layers.2.self_attn.k_norm.weight"

    def test_unknown_raises(self, args):
        with pytest.raises(ValueError, match="Unknown parameter name"):
            convert_qwen2_to_hf(args, "module.module.bogus.weight", torch.zeros(1))


# ── Qwen3-VL (VLM) ─────────────────────────────────────────────────────────


class TestQwen3VL:
    """Qwen3-VL VLM mapping (v0.3.0).

    cua-lite routes ``model_type == "qwen3vl"`` to ``convert_qwen3vl_to_hf``
    (NOT the flat ``convert_qwen2_to_hf`` raw mapper), which emits the NESTED
    HF names that Qwen3-VL safetensors actually use: ``model.language_model.*``
    for the text tower and ``model.visual.*`` for the vision tower (verified
    against Tongyi-MAI/MAI-UI-2B's safetensors keys). The mapper strips the
    ``language_model.`` Megatron wrapper before routing.

    NOTE: cua-lite's production checkpoint export runs ``--megatron-to-hf-mode
    bridge`` (Megatron's own bridge), so this raw mapper isn't the production
    save path; bridge-mode save is exercised by the GRPO e2e test. These cases
    still pin the raw-mapper contract for the qwen3vl name routing.
    """

    # -- Vision model → model.visual.* --

    def test_vision_patch_embed(self, args):
        result = convert_qwen3vl_to_hf(
            args, "module.module.vision_model.patch_embed.proj.weight", torch.zeros(1)
        )
        assert result[0][0] == "model.visual.patch_embed.proj.weight"

    def test_vision_pos_embed(self, args):
        result = convert_qwen3vl_to_hf(
            args, "module.module.vision_model.pos_embed.weight", torch.zeros(1)
        )
        assert result[0][0] == "model.visual.pos_embed.weight"

    def test_vision_block_attn(self, args):
        result = convert_qwen3vl_to_hf(
            args, "module.module.vision_model.blocks.3.attn.qkv.weight", torch.zeros(1)
        )
        assert result[0][0] == "model.visual.blocks.3.attn.qkv.weight"

    def test_vision_merger(self, args):
        result = convert_qwen3vl_to_hf(
            args, "module.module.vision_model.merger.linear_fc1.weight", torch.zeros(1)
        )
        assert result[0][0] == "model.visual.merger.linear_fc1.weight"

    def test_vision_deepstack_merger(self, args):
        result = convert_qwen3vl_to_hf(
            args, "module.module.vision_model.deepstack_merger_list.0.norm.weight", torch.zeros(1)
        )
        assert result[0][0] == "model.visual.deepstack_merger_list.0.norm.weight"

    # -- Language model (through language_model.* wrapper) → model.language_model.* --

    def test_vlm_embed_tokens(self, args):
        """VLM embed → model.language_model.embed_tokens (nested, v0.3.0)."""
        result = convert_qwen3vl_to_hf(
            args, "module.module.language_model.embedding.word_embeddings.weight", torch.zeros(1)
        )
        assert result[0][0] == "model.language_model.embed_tokens.weight"

    def test_vlm_lm_head(self, args):
        result = convert_qwen3vl_to_hf(
            args, "module.module.language_model.output_layer.weight", torch.zeros(1)
        )
        assert result[0][0] == "lm_head.weight"

    def test_vlm_final_layernorm(self, args):
        result = convert_qwen3vl_to_hf(
            args, "module.module.language_model.decoder.final_layernorm.weight", torch.zeros(1)
        )
        assert result[0][0] == "model.language_model.norm.weight"

    def test_vlm_decoder_layer(self, args):
        result = convert_qwen3vl_to_hf(
            args,
            "module.module.language_model.decoder.layers.10.self_attention.linear_proj.weight",
            torch.zeros(1),
        )
        assert result[0][0] == "model.language_model.layers.10.self_attn.o_proj.weight"

    def test_vlm_gate_up_proj(self, args):
        param = torch.randn(19456, 2560)
        result = convert_qwen3vl_to_hf(
            args, "module.module.language_model.decoder.layers.0.mlp.linear_fc1.weight", param
        )
        assert result[0][0] == "model.language_model.layers.0.mlp.gate_proj.weight"
        assert result[1][0] == "model.language_model.layers.0.mlp.up_proj.weight"


# ── No regression: VLM paths don't break pure Qwen2 ────────────────────────


class TestNoRegression:
    def test_pure_qwen2_unaffected_by_vlm_code(self, args):
        """Pure Qwen2 names should never hit VLM branches."""
        # These should all work identically to the original code
        cases = [
            ("module.module.embedding.word_embeddings.weight", "model.embed_tokens.weight"),
            ("module.module.output_layer.weight", "lm_head.weight"),
            ("module.module.decoder.final_layernorm.weight", "model.norm.weight"),
            ("module.module.decoder.layers.0.mlp.linear_fc2.weight",
             "model.layers.0.mlp.down_proj.weight"),
        ]
        for megatron_name, expected_hf_name in cases:
            result = convert_qwen2_to_hf(args, megatron_name, torch.zeros(1))
            assert result[0][0] == expected_hf_name, f"Failed for {megatron_name}"
