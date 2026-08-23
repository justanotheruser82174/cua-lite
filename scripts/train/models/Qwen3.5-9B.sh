# Qwen3.5-9B Megatron arg template.
#
# Derived from HF config.json. The slime repo does not ship a 9B template;
# this file interpolates between the slime-shipped 4B (slime/slime/scripts/
# models/qwen3.5-4B.sh) and 27B (slime/scripts/models/qwen3.5-27B.sh) using
# the 9B HF config numbers verbatim.
#
# See Qwen3.5-2B.sh for per-flag rationale.
#
# tie_word_embeddings=false (from config.json) → add
# --untie-embeddings-and-output-weights.

MODEL_ARGS=(
   --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"

   --disable-bias-linear
   --qk-layernorm
   --group-query-attention
   --num-attention-heads 16
   --num-query-groups 4
   --kv-channels 256
   --num-layers 32
   --hidden-size 4096
   --ffn-hidden-size 12288
   --use-gated-attention

   --normalization RMSNorm
   --apply-layernorm-1p
   --position-embedding-type rope
   --norm-epsilon 1e-6
   --rotary-percent 0.25
   --swiglu
   --untie-embeddings-and-output-weights
   --vocab-size 248320

   --rotary-base 10000000

   # qwen3.5 specific
   --attention-output-gate
)
