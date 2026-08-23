# OSWorld Perturbation — Common Workflow

Perturbation reuses OSWorld eval source files to generate scalable training tasks with deterministic verifiable rewards, by systematically varying operation parameters or operation types. Domain-specific plans live in `devs/envs/lite.osworld/perturb/{domain}.md`; the corresponding code lives in `lite/gym/envs/lite/osworld/src/gen/train/perturb/{domain}.py`.

**These two files are co-evolved peers — neither is the ground truth.** The `.md` is drafted first as a starting-point plan, but implementation and validation (V1–V5) will surface issues that require updating both files together. The `.md` is a living document that evolves alongside the code; it is not a frozen spec that the code merely follows.

> **V1–V4 are not enough.** They catch static / oracle-replay bugs, but cycle 33's audit found multiple **agent-rollout-only** bugs that V1–V4 silently passed (impress `_build_row` only injecting `expected_py` into `oracle_actions` → 41/41 fail; `_make_spacing` raw paragraph idx vs visible paragraph idx asymmetry; thunderbird template polarity mismatched with perturbed value). **V5 (rollout audit, see `devs/envs/lite.osworld/validate/rollout/plan.md`) is now the ground-truth gate** — only after V5 finds 0 unresolved generator bugs is a generator considered correct. The hard constraints below encode the recurring V5 failure patterns.

---

## Prime Directive

> **ULTIMATE GOAL**: SFT/RL on `train.{perturb,synth}.jsonl` measurably improves agent score on `eval.jsonl` (full set, 369 tasks; infeasibles included since the agent is scored end-to-end at deployment).

```
4 sub-directives (each necessary, not sufficient — all must hold to approximate transfer):

(1) domain distribution match (Q1)
(2) per-domain pass rate match (Q2 / Δfeas)              all 4 ⊆ transfer to eval
(3) in-domain skill / reward landscape match
(4) internal correctness (V1–V5)
```

**Never sacrifice transfer to land any of (1)-(4).** Every change must answer "does this make perturb more like `eval.jsonl (infeasible-filtered)` on a real distribution dimension?" Q2 / row-count fudges that don't move skill or reward landscape → reject.

### (1) Domain distribution match (Q1)

Perturb's per-domain row-count share matches eval's, within ±5pp. Prevents over-/under-emphasising any domain. **See [§Macro Quantitative Targets — Q1](#q1--domain-count-ratio-matches-eval-within-5pp).**

### (2) Per-domain pass rate match (Q2 / Δfeas)

Per domain, perturb agent pass-rate is within ±10pp of eval-doable agent pass-rate (i.e. `eval.jsonl (infeasible-filtered)`). **See [§Macro Quantitative Targets — Q2](#q2--per-domain-agent-pass-ratio-matches-eval-within-10pp).** Asymmetric tolerance: perturb < eval (mild) is acceptable; perturb > eval is degradation (must fix).

### (3) In-domain skill / reward landscape match

Within each domain, perturb's evaluator distribution mirrors eval's: same `evaluator.func` types in similar ratio, same `result.type` and `options.rules` shapes, same op coverage (which actions/skills the metric rewards). A domain can satisfy Q1+Q2 numerically while completely mismatching skill mix — e.g. perturb chrome was 88% pass before cycle 34 mostly via easy URL navigation skills, while eval chrome rewards `active_tab_html_parse` / `url_path_parse` (perturb 0). Same average, different reward landscape, no transfer. **Reward landscape mismatch is the dominant root cause for the +Δfeas degradations.** See [/perturb-pass-rate-match.md](/perturb-pass-rate-match.md) per-domain plan.

### (4) Internal correctness (V1–V5)

V1–V4 PASS (static / oracle replay) + V5 PASS (rollout audit, see [`validate/rollout/plan.md`](/devs/envs/lite.osworld/validate/rollout/plan.md)). A dataset can satisfy every static check and still fail to transfer (cycle 33: impress was V1+V2+V3+V4 ✓ but eval-rollout 0/41 because `expected_py` only landed in `oracle_actions`). **V5 is the gate that approximates the prime directive** — only after V5 finds 0 unresolved generator bugs across two consecutive sweeps is a generator considered correct.

---

## Notation convention for `eval.jsonl` references

- **`eval.jsonl`** (unqualified): full eval set, 369 tasks, includes 29 infeasible. Used in the transfer prime directive (final goal).
- **`eval.jsonl (infeasible-filtered)`** / **eval doable** / **eval feasible**: 340-task subset excluding 29 infeasibles. Used for Q2 / distribution-match comparisons.
  - Per design, `train.perturb` does NOT construct infeasible tasks (perturbing an infeasible eval base would inherit the infeasibility premise; see [`apply_structural_perturbation` guard](/lite/gym/envs/lite/osworld/src/gen/train/perturb/dispatch.py)). The apples-to-apples baseline for Q2 is therefore the infeasible-filtered subset.
- **Δraw** = `perturb_pass% − eval_pass%` (full eval.jsonl). Misleading because eval auto-passes ~7% via infeasible-decline that perturb cannot replicate.
- **Δfeas** = `perturb_pass% − eval-doable_pass%` on `eval.jsonl (infeasible-filtered)`. **This is the Q2 metric we optimize against.**
- Throughout these AGENTS.md and `{domain}.md` docs: any time we compare a perturb pass-rate to an eval pass-rate, the eval side is **`eval.jsonl (infeasible-filtered)`** unless explicitly stated otherwise.

---

## Macro Quantitative Targets (dataset-wide)

These are the two top-level numerical health checks. Both run on the full `train.perturb.jsonl` against `eval.jsonl`. Domain-level details come later in [§Hard Constraints](#hard-constraints-apply-to-all-domains) and [§Validation Scripts](#validation-scripts).

### Q1 — Domain count ratio matches eval (within ±5pp)

`perturb_share(d) − eval_share(d) ∈ [−5pp, +5pp]` for every domain `d`. Prevents over- or under-emphasising any domain relative to the evaluation distribution we want to transfer to.

```python
import json
from collections import Counter

def shares(path):
    c = Counter()
    for line in open(path):
        row = json.loads(line)
        others = (row.get("metadata", {}).get("others") or {})
        c[others.get("domain", "?")] += 1
    total = sum(c.values())
    return {d: n / total for d, n in c.items()}

eval_s = shares("lite/gym/envs/lite/osworld/data/eval.jsonl")
pert_s = shares("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")
for d in sorted(set(eval_s) | set(pert_s)):
    diff = (pert_s.get(d, 0) - eval_s.get(d, 0)) * 100
    flag = " ⚠️" if abs(diff) > 5 else ""
    print(f"  {d:<22s} eval {eval_s.get(d,0)*100:5.1f}%  perturb {pert_s.get(d,0)*100:5.1f}%  Δ {diff:+5.1f}pp{flag}")
```

If a generator change pushes any domain past ±5pp, rebalance before commit (typically by adjusting variants-per-base in the over-weight domain or adding archetypes to the under-weight one).

### Q2 — Per-domain agent pass ratio matches eval (within ±10pp)

`|perturb_pass(d) − eval_pass(d)| ≤ 10pp` for every domain `d`, measured on the same model + same image during a V5 sweep. Large gaps indicate training-vs-evaluation drift — either perturb tasks are systematically easier (so SFT learns shortcuts that don't transfer) or systematically harder (training signal floors out, no useful gradient).

| Symptom | Likely root cause | Fix path |
|---|---|---|
| perturb easier than eval (perturb_pass ≫ eval_pass) | evaluator stricter on eval, looser on perturb (e.g. perturb dropped fields the eval still checks); or perturb instructions short-circuit the actual skill | Tighten perturb evaluator to match eval's per-field strictness; add longer-step variants |
| perturb harder than eval (perturb_pass ≪ eval_pass) | perturb amplified a UI-ceiling axis (e.g. all 4 variants fall in a deep dialog tree); generator bug (cycle 33 patterns: expected_py only in oracle_actions, idx asymmetry, polarity mismatch); or instruction has matrix bug (contradictory dual-target) | Run V5 root-cause triage; fix generator; if truly UI-ceiling, balance variant difficulty |

A gap >10pp is a bug to investigate, not (necessarily) to drop. Q2 is computed at full sweep completion via `validate/rollout/plan.md` §`scan`'s `domain_stats`.

---

## Hard Constraints (Apply to All Domains)

1. **No eval leakage**: every generated training task must not be exactly identical to its source eval task in the dimensions that matter for that domain (see `{domain}.md`). Both **exact** instruction match (filtered in dispatcher) and **semantic** leak — `(evaluator.func, evaluator.expected)` tuple identical to an eval row — are checked in V1.

2. **Every task must have a verifiable oracle, AND the expected file must be created during agent rollout**:
   - An initial state setup step ensures the state differs from the target before the agent acts (reward before agent acts = 0).
   - **`oracle_actions`** (run only when oracle replay is invoked, e.g. by V2): establishes the ground truth expected state for the gold path.
   - **`config` / `pre_config_steps`** (run on every agent rollout): MUST also create the expected-file artifact (e.g. `/tmp/perturb_expected_<base>.<ext>`) when the evaluator compares against it. Putting the expected-file writer ONLY in `oracle_actions` is a silent V5-only bug: V2 oracle smoke passes (oracle replay ran the writer), but agent rollouts read None for the expected file and score 0 unconditionally. Pattern: `perturb_config_step=_make_config_step(expected_py)` passed to `make_perturb_row`, OR insert via `cfg.insert(open_idx, _make_config_step(expected_py))` (see `libreoffice_calc.py:_emit_a1_variant` for the working calc pattern, and `libreoffice_impress.py:_build_row` for the cycle-33 fix).
   - An evaluator compares the agent's output against the expected state.
   - Operations without a deterministic oracle (AI-generated content, external file dependencies) are excluded.
   - See `{domain}.md` for the domain-specific oracle implementation.

3. **Instruction ordinal/index must match what the eval reads AND what a human reader sees**:
   - When the instruction says "fifth paragraph" / "third slide" / "row 5", the eval target index must reference the **visible / counted-by-human** position, not `paragraphs[]` raw indices that count blank separator paragraphs LO inserts on docx export. Example fix: `_make_uppercase` / `_make_spacing` advance past empty paragraphs to find the Nth visible one (see `libreoffice_writer.py:_make_spacing`, cycle-33 fix).
   - When generating multiple ops on the same archetype, pick **distinct** target indices via `rng.sample`, never `[rng.choice(pool) for _ in range(N)]` — sampling with replacement can produce contradictory instructions like "set picture height on slide 6 to 8 cm AND set picture height on slide 6 to 20 cm" (see `libreoffice_impress.py:_t1_7ae48c60`, cycle-33 fix).

4. **Instruction polarity must match the perturbed value direction**:
   - When a boolean-pref perturb flips eval value `false → true` (or vice versa), the verb (`enable`/`disable`) **and** the surrounding narrative must both shift. Asking the agent to "enable X" with anti-X narrative ("I'd rather compose clean replies") makes every variant fail because the agent reads the narrative and disables. Pattern: split `instruction_templates` into `instruction_templates_enable` / `instruction_templates_disable` and pick by `verb` (see `thunderbird.py:mail.identity.id1.auto_quote`, cycle-33 fix).

5. **All perturbable action types for the domain must have training samples**: see `{domain}.md` for the required coverage and perturbation strategies.

6. **No infeasible rows in train.perturb**: agents should never learn to "give up" on training tasks. V1 flags any train-split row with `evaluator.func == "infeasible"` as a hard fail.

7. **Drop only on structural infeasibility, not on agent step budget**: when an agent rollout truncates (`max_steps` exhausted), that's tunable — the right fix is raising `max_steps` for that task family, NOT dropping the perturb archetype. Drop only when the eval contract is genuinely impossible: redirect rewrites synthetic URL params, IMAP gloda async race, model-level safety refusal, eval threshold tighter than the gold can satisfy, schema absent on the target desktop, etc. (See `validate/rollout/plan.md` §scan for the full structural-vs-budget triage.)

---

## URL hallucination prevention (cycle 35a+)

> **`src_url` MUST be copied verbatim from `eval.jsonl`** — never reconstructed from the 8-char `short_id`, never hand-typed, never inferred from a HuggingFace path pattern. The 8-char prefix is a digest of the full task_id; the trailing UUID portion of HF storage URLs is independent and cannot be recovered from short_id alone. There is no implicit cross-check at codegen time, so a hallucinated UUID flows through V1 (well-formed URL) and V2 (oracle-replay can succeed if the wrong file happens to be a structurally-compatible artifact, or fails as `cp rc=1` which is masked by retries) before surfacing as silent V5 FALSE_NEG (expected file never built → score 0 unconditionally).
>
> **Pattern**: load `ev_by_id[base_tid]["metadata"]["config"]` → walk `download` steps → copy `files[i]["url"]` into the perturb spec. If the spec needs a different VM destination basename than the HF storage filename (e.g. f8cfa149's `file.xls` ↔ `cell_search.xls`), keep `src_url` pointing at the real HF URL and override `src_basename` separately.
>
> **Direct trigger** (cycle-35a-fix): 9 of 125 multi_apps perturb rows had hallucinated UUID suffixes in `src_url`, all caught only by V5. See [`multi_apps.md` cycle 35a section](/devs/envs/lite.osworld/perturb/multi_apps.md#cycle-35a-updates).

---

## Optimal Workflow

The workflow has two phases: **plan** (steps 0–1) and **execute** (steps 2–7). The plan phase must be completed and stabilized before writing any code.

```
PLAN PHASE (do this first, before any code)
  |
  v
Step 0: Inspect source files / app state
  -> See {domain}.md Step 0 for domain-specific inspection script
  -> Goal: understand what each source file or setting actually contains
  |
  v
Step 1: Draft per-task plan in {domain}.md (starting point — will evolve)
  -> For each eval task: decide perturbable or not, and why
  -> For perturbable tasks: design the resampling space based on actual content
  -> Review: are all required action types covered? Do assumed objects exist?
  -> Iterate until the plan is stable
     (this is the hardest part — errors here propagate everywhere)
  IMPORTANT: `{domain}.md` and `{domain}.py` are co-evolved peers.
  Any change to either — op pools, per-task params, broken/skip lists,
  oracle mechanics — must be reflected in the other immediately.
  The `.md` is NOT a frozen spec; it is updated in sync with the code
  through every V1–V4 iteration cycle.
  Every `{domain}.md` must include a `> **Keep in sync with code.**` blockquote
  at the top of its per-task plan/strategy section to make this invariant visible
  at point of use.

EXECUTE PHASE
  |
  v
Step 2: Implement core code (keep {domain}.md in sync as you go)
  -> Value pools + perturbation logic
  -> Per-task builders (aligned with {domain}.md plan; update md if plan changes)
  -> Instruction templates
  |
  v
Step 3: Generate
  -> uv run python -m lite.gym.envs.lite.osworld.src.gen.train --track perturb --domain {domain}
  -> Quick sanity: row count, split by variant type
  |
  v
Step 4: V1 Static Check (seconds)
  -> required fields, unique task_ids, no "save the file", oracle list non-empty
  -> FAIL -> fix code -> back to Step 3
  |
  v
Step 5: V2 Oracle Smoke Test (~5 min, docker)
  -> Tests: does oracle script run without crashing (rc=0)?
     Does the initial state setup actually differ from target (trivial_pass check)?
  -> Does NOT test agent capability or task quality
  -> FAIL -> fix code -> back to Step 3
  |
  v
Step 6: V3 Instruction Distribution (seconds)
  -> Compare perturb vs eval on: polite%, save-file%, multi-step%, avg word count
  -> Out of range -> adjust _build_instruction probabilities -> back to Step 3
  |
  v
Step 7: V4 Task Quality (one-time after initial implementation)
  -> V4a: op coverage check — all required action types covered? (seconds)
     FAIL -> fix code AND update {domain}.md -> back to Step 2
  -> V4b: perturb-eval match verification — feasibility + distribution match
     FAIL -> fix code AND update {domain}.md -> back to Step 2
  |
  v
Step 8: V5 Rollout Audit (the ground-truth gate)
  -> Run `scripts/rollout.py` against train.perturb in a NEW LOG_ROOT.
     Follow `devs/envs/lite.osworld/validate/rollout/plan.md` §audit-loop:
     scan -> group_by_root_cause -> spawn read-only diagnostic subagents
     -> apply fixes (.py source, NOT .jsonl) -> regen -> delete affected
     summaries -> restart_rollout (same LOG_ROOT, idempotent).
  -> Catches what V1-V4 cannot:
       * silent eval-bug-on-rollout (expected file only in oracle_actions)
       * raw-vs-visible idx asymmetry (instruction "fifth" vs eval idx)
       * polarity mismatch (verb=enable, narrative=disable)
       * UI-vs-eval mismatch (Chrome cookie schema, LO Save-As cliff,
         VS Code Copilot signin modal, GNOME schema drift)
  -> Exit gate: 0 unresolved generator bugs across all signals
     (FALSE_NEG / FALSE_POS / TRIVIAL_PASS / VARIANT_HOMOGENEITY_ZERO /
      INFEASIBLE_CLAIM_TRAIN — see plan.md §Trigger taxonomy).
  -> A second SWEEP (fresh LOG_ROOT, after sweep-1 fixes are committed)
     confirms no new bugs surface.
  |
  v
Full generate + update sha256 + commit
  -> uv run pytest (confirm pre-existing failures unchanged)
```

**Commit conditions (hard constraints)**:
- V2 all pass (100%)
- V3 polite in [30%, 40%], save = 0%
- V4a op coverage check: all required action types present, action-count distribution matches eval (±10pp), per-op share ratio 0.3×–3×
- V4b perturb-eval match verification done at least once
- V5 rollout audit: 0 unresolved HOMO_ZERO / INFEASIBLE_TRAIN / generator bugs in **two consecutive independent sweeps** (different LOG_ROOTs); flagged FALSE_NEGs are either confirmed agent-ceiling or fixed-and-verified

---

## Validation Scripts

### V1 Static Check

```python
import json
from collections import Counter

rows = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
domain_rows = [r for r in rows if "{domain}" in r["task_id"]]  # adjust filter per domain

required = {"task_id", "instruction", "metadata"}
missing = [r["task_id"] for r in domain_rows if not required <= r.keys()]
ids = [r["task_id"] for r in domain_rows]
dups = [id for id, cnt in Counter(ids).items() if cnt > 1]
save_leaks = [r["task_id"] for r in domain_rows if "save the file" in r["instruction"].lower()]
oracle_errors = [r["task_id"] for r in domain_rows
    if not isinstance((r.get("metadata", {}).get("others") or {}).get("oracle_actions"), list)
    or len((r.get("metadata", {}).get("others") or {}).get("oracle_actions", [])) == 0]

for label, lst in [("missing fields", missing), ("duplicate ids", dups),
                   ("save leaks", save_leaks), ("oracle errors", oracle_errors)]:
    print(f"[{'FAIL' if lst else 'OK  '}] {label}: {lst or 'ok'}")
```

### V2 Oracle Smoke Test

```bash
uv run python devs/envs/lite.osworld/validate/oracle/validate.py \
    --fixtures lite/gym/envs/lite/osworld/data/train.perturb.jsonl \
    --filter {domain} --concurrency 4 --retries 3 \
    --report /tmp/validate_{domain}.report.jsonl 2>&1 | tee /tmp/validate_{domain}.log
```

```python
import json
rows = [json.loads(l) for l in open("/tmp/validate_{domain}.report.jsonl")]
total = len(rows)
passed = sum(1 for r in rows if r["passed"])
trivial = [r for r in rows if "trivial_pass" in r.get("message", "")]
oracle_fail = [r for r in rows if not r["passed"] and "trivial" not in r.get("message", "")]
print(f"{passed}/{total} passed, {len(trivial)} trivial_pass, {len(oracle_fail)} oracle failures")
for r in oracle_fail:
    print(r["task_id"], ":", r["message"][:120])
```

Common failure modes:

| Failure | Cause | Fix |
|---|---|---|
| `trivial_pass` | Initial state setup left state == target | Set a different initial value |
| `rc=1` FileNotFoundError | Wrong file path in task | Check file_path field |
| `rc != 0` | Oracle script crashed | Check error message; fix oracle or exclude the task |

See `{domain}.md` for domain-specific failure modes and fixes.

### V3 Instruction Distribution

```python
import json, re

all_eval = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/eval.jsonl")]
eval_rows = [r for r in all_eval if "{domain}" in r["task_id"]]
all_perturb = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
perturb_rows = [r for r in all_perturb if "{domain}" in r["task_id"]]

_POLITE_RE = re.compile(r"^(please|could you|can you|i need|i want|i'd like|i would like)", re.I)
_SAVE_RE   = re.compile(r"save the file", re.I)
_MULTI_SEP = re.compile(r"\. (Also|Additionally|Then|Next),")

def stats(rows):
    instrs = [r["instruction"] for r in rows]
    n = len(instrs)
    return {
        "polite":    f"{sum(bool(_POLITE_RE.search(s)) for s in instrs)/n:.0%}",
        "save":      f"{sum(bool(_SAVE_RE.search(s))   for s in instrs)/n:.0%}",
        "multi_sep": f"{sum(bool(_MULTI_SEP.search(s)) for s in instrs)/n:.0%}",
        "avg_words": f"{sum(len(s.split()) for s in instrs)/n:.1f}",
    }

print("eval   :", stats(eval_rows))
print("perturb:", stats(perturb_rows))
```

Target: polite 30–40%, save 0%, avg_words within +-3 of eval baseline.

### V4a Op Coverage Check

See `{domain}.md` for the domain-specific required action type list and coverage check script.

### V4b Perturb-Eval Match Verification

See `{domain}.md` for the domain-specific analysis data and per-task checklist.

---

## Instruction Style (Common Rules)

`_build_instruction(ops, rng)` behavior — applies to all domains:

| Scenario | Behavior |
|---|---|
| Single op | `"<Op sentence>."` |
| Two ops (40%) | `"<op1>, and <op2>."` — fused into one sentence |
| Two ops (60%) / three+ ops | random separator (`. Also, ` / `. Additionally, ` / `. Then, ` / `. Next, ` / `. `); 2nd+ parts start lowercase |
| Single sentence with first word a verb (40%) | prepend `Could you help me/Please/I need to/Can you/I want to/I'd like to` |
| Any case | **Never append "Save the file."** |
