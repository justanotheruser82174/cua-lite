# lite.scalecua Sample IDs

Store fixed smoke and batch sample IDs here after implementation. Samples must
be selected after filtering non-empty `exclude_reason` and should cover all 10
OSWorld domains across `train` and `rl`.

`negative_control_corpus.jsonl` currently pins an oracle-backed scaffold for
loosened checkers and flush gates. Rows include self-contained evidence
summaries and, when local `.exps` artifacts are present, reference oracle replay
evidence whose no-op precheck reward is `0.0` and replay reward is `1.0`.
Every row is explicitly marked `artifact_source="oracle_noop_scaffold"` and
`gate_eligible=false` so this file cannot be mistaken for the collected rollout
negative-control corpus required by the Batch-5 gate.

This scaffold does not close `plan.md` WEAK-GATE #2 by itself: that gate still
requires collected reward-0 trajectories with initial/key-mutation frame
evidence showing the target feature is genuinely absent.
