# Qwen3.5-2B Megatron arg template.
#
# Derived from the HF checkpoint's config.json + text_config (see
# ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B). Cross-referenced with
# slime/slime/scripts/models/qwen3.5-4B.sh which is the closest official
# reference in the slime repo.
#
# Qwen3.5-specific flags (same across all dense sizes):
#   --spec ...get_qwen3_5_spec    — custom Megatron layer spec that handles
#                                    the hybrid linear/full-attention stack
#                                    (text_config.layer_types alternates
#                                    linear_attention × 3 + full_attention × 1).
#   --use-gated-attention         — attention block uses gated output.
#   --attention-output-gate       — Qwen3.5-specific extra output gate.
#   --apply-layernorm-1p          — required: HF Qwen3_5RMSNorm computes
#                                    output * (1 + weight) and stores weight
#                                    centered around 0. Megatron must use the
#                                    same convention so the bridge's direct-
#                                    copy AutoMapping is correct end-to-end.
#   --rotary-percent 0.25         — 25% partial RoPE (from config.json
#                                    rope_parameters.partial_rotary_factor).
#   --vocab-size 248320           — Qwen3.5 tokenizer is larger than Qwen3-VL's
#                                    151936.
#   --rotary-base 10000000        — HF config rope_theta (Qwen3.5-specific;
#                                    Qwen3-VL-*.sh hard-code 5e6, which is
#                                    wrong for Qwen3.5).
#
# tie_word_embeddings=true (from config.json) → no
# --untie-embeddings-and-output-weights flag.

MODEL_ARGS=(
   --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"

   --disable-bias-linear
   --qk-layernorm
   --group-query-attention
   --num-attention-heads 8
   --num-query-groups 2
   --kv-channels 256
   --num-layers 24
   --hidden-size 2048
   --ffn-hidden-size 6144
   --use-gated-attention

   --normalization RMSNorm
   --apply-layernorm-1p
   --position-embedding-type rope
   --norm-epsilon 1e-6
   --rotary-percent 0.25
   --swiglu
   --vocab-size 248320

   --rotary-base 10000000

   # qwen3.5 specific
   --attention-output-gate
)
