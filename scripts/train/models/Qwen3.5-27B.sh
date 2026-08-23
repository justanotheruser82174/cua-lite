# Qwen3.5-27B Megatron arg template.
#
# Byte-identical to the official slime reference at
# slime/scripts/models/qwen3.5-27B.sh; config.json values cross-verified.
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
   --num-attention-heads 24
   --num-query-groups 4
   --kv-channels 256
   --num-layers 64
   --hidden-size 5120
   --ffn-hidden-size 17408
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
