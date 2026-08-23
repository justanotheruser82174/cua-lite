# OSWorld Synth — Common Workflow

Synth generates training tasks **from scratch** (instructions + source files + oracle + evaluator), as opposed to `perturb/` which mutates existing eval rows. Code lives in `lite/gym/envs/lite/osworld/src/gen/train/synth/{domain}.py`; per-domain plans in `devs/envs/lite.osworld/synth/{domain}.md`. The two are **co-evolved peers** — same rule as on the perturb side ([perturb/AGENTS.md](/devs/envs/lite.osworld/perturb/AGENTS.md)). Every change to one lands in the same commit as the matching change to the other.

---

## Prime Directive

> **ULTIMATE GOAL**: SFT/RL on `train.{perturb,synth}.jsonl` measurably improves agent score on `eval.jsonl` (full set, 369 tasks; infeasibles included since the agent is scored end-to-end at deployment).

**Seven directives, ordered by importance to a high-quality training set** (most important → least). Each item is a hard quality gate at the level above it, a soft preference at the level below.

```
(1) Per-row internal correctness         (1a) task feasibility (tools + oracle reachable)
                                         (1b) eval_fn ⇄ op alignment
                                         (1c) V1–V5 validation pipeline PASS
(2) Instruction ⇄ source consistency     (2a) asset + content-anchor consistency
                                         (2b) F6 ordinal safety
(3) Real-source + structural diversity   (3a) ≥25% real per content-evaluable domain
                                         (3b) no template clones (structural axes vary)
                                         (3c) file-first task design (因地制宜)
(4) Skill landscape match (per domain)   evaluator.func ratio mirrors eval (±15pp)
(5) Task difficulty cap                  synth ≤ eval-doable feasibility band
(6) Domain distribution + total volume   per-domain row share ±5pp; ≥1500 rows total
(7) Instruction style match              synth phrasing close to eval same-domain rows
```

**(1a) → (1b)** is the natural read order: first ask "*can* this row run end-to-end?", then "*does* the evaluator score the right thing?", then "did V1–V5 actually confirm both empirically?". (1a) and (1b) are static (read code + source); (1c) is dynamic (run the pipeline).

**Three tiers** (by what failure of each costs at training time):

- **(1)–(3) quality floors** — violating any of these silently degrades training even when the rest pass. NEVER ship a row that fails (1)–(3).
- **(4)–(5) transfer-shape gates** — they decide WHAT skill the model learns and whether trajectories are well-formed. Drift is recoverable via a follow-up cycle.
- **(6)–(7) shape preferences** — improvable but **fixable downstream** (training-time weighted sampling for (6); model robustness for (7)). Lowest priority.

**Operational rule**: when a fix improves a higher item at the cost of a lower one, **take the fix**. Q1 widening to -10pp with all rows correct is strictly better than Q1 ±2pp with 30% buggy rows.

**Worked example (Round-6d)**: trimmed multi_apps 203→176 jsonl rows to rebalance bucket distribution toward eval's actual `compare_text_file` / `compare_images` ratio. Q1 widened from -11.1pp → -14.0pp as a side-effect. **Correct**: served (4) skill-landscape match; hit only (6) volume.

### (1) Per-row internal correctness

Three checks, ascending in cost: a static feasibility walkthrough, a static evaluator-correctness check, then a dynamic V1–V5 confirmation. (1a) and (1b) are reasoning over code+source before any rollout; (1c) is the empirical complement that catches what static reasoning missed.

#### (1a) Task feasibility (static)

Before adding any row, **mentally walk through a rollout** and confirm every step has a concrete answer:

> pre_config builds source ✓ → agent does X clicks → save → postconfig captures result → evaluator reads result vs expected → returns 1.0

Four sub-checks the walkthrough must pass:

- **Tool present in container**: every CLI / library the task touches is in the Dockerfile baseline OR explicitly listed under `Installed` (see [Tool-dependency contract](#tool-dependency-contract-transfer-to-eval-consideration)). Rows depending on un-installed tools are **infeasible** until the tool ships, even if everything else looks correct.
- **Source file constructable in pre_config**: the heredoc that materializes the source file must run cleanly inside the container with the libraries available there (no host-only deps, no missing assets). Stage assets via `_stage_asset(host_push)` for files that can't be inlined.
- **Oracle achieves the gold state**: replaying `oracle_actions` from the source state must produce a sink state that the evaluator scores 1.0. If you can't show how oracle gets to gold, the row is infeasible regardless of how clever the instruction is.
- **Agent can plausibly do it in ≤10 turns**: distinct from the difficulty cap ((5)) — this is a binary "is there ANY trajectory of clicks / commands that solves it?" check. UI-only ops with no shell oracle (multi-cursor selections, accessibility-tree-only state, presenter-console internal toggles) often **fail this check** and must be dropped, not deferred.

If any step has no concrete answer, mark the row deferred or dropped, not implemented. (1a) failures are the cheapest to catch — minutes of reading vs hours of sweep.

#### (1b) eval_fn ⇄ op alignment (static)

The cited `evaluator.func` must actually score the operation the instruction asks the agent to do. A misaligned func either FALSE-PASSES (rewards no agent action) or FALSE-FAILS (penalises correct work) — both poison training.

- **Counter-example (iter 2)**: writer row 161 cited `evaluate_strike_through_last_paragraph` for a 4th-paragraph strikethrough; the func ONLY checks the last paragraph → FALSE-PASS regardless of agent action. Fix: use `compare_docx_strict` (always-on strikethrough check) when target ≠ last.
- **General test**: read the eval func's source — does it inspect the field the instruction asks the agent to change? "Bold paragraph 5" but func only reads paragraph 3 → mismatch.
- **Common traps**: bespoke funcs with hard-coded paragraph/cell/slide indices (last-paragraph / first-cell / slide-2-only checks); `examine_*` flags on `compare_docx_strict` that gate which char-format dimensions get checked.
- **Where to read func source**:
  - `lite/gym/envs/lite/osworld/src/eval/metrics.py` — local custom funcs (e.g. `compare_docx_strict`, `compare_references`)
  - `.venv/lib/python3.12/site-packages/desktop_env/evaluators/metrics/{basic_os,chrome,docs,general,gimp,libreoffice,pdf,others,slides,table,thunderbird,utils,vlc,vscode}.py` — upstream OSWorld funcs
  - If a func is in NEITHER → fabricated → row must be re-cited or dropped.

#### (1c) V1–V5 validation pipeline (dynamic)

V1–V4 PASS (static / oracle replay) + V5 PASS (rollout audit, see [`validate/rollout/plan.md`](/devs/envs/lite.osworld/validate/rollout/plan.md)). A dataset can satisfy every (1a)/(1b) check and still fail empirically: stale source assumption, race condition in postconfig, font-substitution drift, openpyxl XML round-trip artifact. **V5 is the gate that approximates the prime directive** — only after V5 finds 0 unresolved generator bugs across two consecutive sweeps is a generator considered correct.

### (2) Instruction ⇄ source consistency

The instruction IS the SFT label. Ambiguous instructions → noisy labels → degraded training even when the evaluator is correct. Two sub-checks: **the instruction must be reasonably designed for the source**, AND **ordinal references must be unambiguous (F6 safety)**.

#### (2a) Asset + content consistency

- **Asset existence**: every concrete asset reference (`gutenberg/sherlock-adventures.txt`, `photos/wildlife/tiger-closeup.jpg`, `data/csv/oil-wti-daily.csv`) must exist in `assets/synth/MANIFEST.csv`. Counter-example (iter 1): multi_apps row 338 referenced `gutenberg/sherlock.txt` (file is `sherlock-adventures.txt`).
- **Source supports the operation**: if instruction says "bold paragraph 5", source must have ≥5 visible body paragraphs. If "merge cells A1:C1", source must have ≥3 columns. If "italicize the dialogue paragraph", source must contain a clearly-identifiable dialogue paragraph (not a fact-sheet that has zero dialogue).
- **Quote anchors actually appear**: any instruction phrase like `the paragraph beginning 'Sun Tzŭ said: The art of war'` requires the source to literally contain that text. The generator must inject the quoted lexeme into the source file (or pick a source slice that already contains it). Don't hand-wave with "this generally appears in the first 10 paragraphs of Treasure Island" — pin the slice.
- **Categorical anchors actually exist**: "highlight the warning paragraph" requires the source to have a paragraph that is identifiably a warning (by content, by pre-applied highlight, or by HR/policy genre). "Sort by Income" requires an Income column.
- **Difficulty matches source shape**: a "5×5 table" task on a paragraph-only source requires the agent to first create the table — that's a different skill from "edit existing cells". Spell out which.
- **Operational test**: write the instruction, then read the source spec — could a human do the task on this exact source without inferring missing pieces? If not, either fix the source spec OR rewrite the instruction.

#### (2b) F6 ordinal-paragraph safety

Any instruction referencing "the {ord} paragraph / line / slide / sheet / cell" must use ONE of three anti-ambiguity strategies (see writer.md §"Source-file template diversity (1) Ordinal-paragraph heuristic"):

- **(a) Quote anchor**: `"the paragraph beginning 'X'"` — agent uses Find to locate, no ordinal counting needed. Most robust.
- **(b) Body-only source + explicit `body_idxs`**: source spec declares `body_idxs=[0..n]` and explicitly notes "no title, no heading, no separator paragraphs". The Nth visible paragraph then equals `paragraphs[body_idxs[N-1]]` deterministically.
- **(c) Visual anchor**: `"the highlighted paragraph"`, `"the paragraph reading 'CHAPTER I'"`, `"the warning row in red"` — anchor by visual property the agent can identify without counting.

**Banned**: bare ordinals on documents that MAY have title / heading / separator paragraphs — writer perturb's `_make_spacing` (raw idx → 0/N when LO inserts a blank separator) is a known instance of this trap; the same trap exists in synth Round 5 templates if not audited.

### (3) Real-source + structural diversity

Two sub-checks. Both protect against the model overfitting to a "synth aesthetic" (lorem-ipsum text + geometric placeholder shapes + scalar-shuffled clones) that doesn't exist at eval time.

#### (3a) Real-source ratio

**Per content-evaluable domain, ≥25% of active plan rows reference real-world assets** (Project Gutenberg, Wikipedia HTML snapshots, FRED CSVs, Census, NASA, OpenClipArt, GitHub MIT code, arxiv PDFs from `assets/synth/`).

- **Content-evaluable**: eval funcs read file CONTENT (text / image pixels / table cells / code). Currently: chrome, calc, impress, writer, multi_apps, os, vs_code, gimp.
- **Structurally exempt**: eval funcs read profile / player state (NOT file content). Real source contributes no signal. Currently: thunderbird (prefs.js / filter rules / folder structure), vlc (player config / playback state). Real audio/video would be cosmetic.
- **Counts as real**: row references a file from `assets/synth/MANIFEST.csv` with a real `source_url` (no `(generated inline)` entries).
- **Does NOT count**: rows that say "synth Times New Roman 12pt 8-para draft" with no asset reference, even if the prose mentions a realistic-sounding genre.

#### (3b) Structural diversity (no template clones)

Two rows that differ only in scalar values (`Sales-Rep 1..N`, `Q1..Q4`, `Item A..Z`) count as 1 effective row, not N. Synth must vary on **structural axes**, not just numeric/string fill:

- **docx (writer / multi_apps)**: font family, font size, line spacing, heading hierarchy, title-page presence, section count, page orientation, paragraph style mix, inline-formatting density, embedded objects, footer/header, separator paragraphs. See writer.md §"Visual-template diversity".
- **xlsx (calc / multi_apps)**: rows × cols, dtype mix, header presence, formula presence, merged cells, frozen panes, named ranges, multi-sheet, charts, page layout.
- **pptx (impress)**: slide count, layout mix, master slide presence, embedded image/video, transitions, notes presence, theme.
- **images (gimp / chrome)**: real-world subject (portrait / landscape / product / architecture / wildlife / etc.) — grayscale gradients are NOT diverse, only different test patterns are.
- **Operational test**: pick 5 random rows of any single template; check the `_make_source_*` calls produce **structurally** different files (not just different scalar fills). If the diff is only `rng.randint(...)` outputs → pseudo-diversity → split the structural axis (e.g., add a row that swaps Default style for Quote style).

#### (3c) File-first task design (因地制宜)

The task's instruction must be the **most natural thing to ask on this specific source file's actual content** — not a generic skill grafted onto whatever file happens to be lying around. (3b) catches rows whose source FILES are structurally cloned; (3c) catches rows whose TASK is generic even when the source file is unique.

**The litmus question** — ask before drafting any new template:

> *"If I swapped this template's `_make_source_*` body for an arbitrary file with the same structural shape, would the task instruction still make sense?"*

- If **yes** (any-file works) → boring template, the file is decorative. Reject and redesign so the instruction depends on the file's actual semantic content.
- If **no** (instruction only makes sense given this content) → content-specific; keep.

**Failure modes the litmus catches**:
1. *"Sort the data by column X ascending"* on any 5-col xlsx → boring (sort is content-agnostic). Eval's sort tasks are tied to specific data semantics ("sort movies by release year").
2. *"Apply currency format to the price column"* on any xlsx with a price-named column → boring. Eval's numfmt tasks reference specific financial reports.
3. *"Make the first paragraph bold"* on any docx → boring. Eval's writer-format tasks reference specific document genres (essay, fact sheet, instruction guide).
4. *"Add a Sheet2 and copy column X"* on any single-sheet xlsx → boring. Eval's Sheet2 tasks include domain-specific copy targets ("copy the Revenue column for Q3 analysis").

**Per template, list ≥3 concrete eval task_ids that motivate it** AND the specific content axis of each (not just dimensions: also semantic domain — finance / education / healthcare / hospitality / transit / etc.). A content-specific template builds its instruction text from the file's actual semantic content (column names that mean something, paragraph topics that anchor the request).

**Anti-eval-mimicry corollary** — file-first does NOT mean copying the eval's exact instruction text. See [Synth design § Don't mirror eval too tightly](#dont-mirror-eval-too-tightly-anti-eval-mimicry) for the full statement: synth's job is to teach the SKILL, not to replicate the eval INSTANCE. Two rows teaching the same skill should differ on **both** axes — different file genre AND different invocation domain.

### (4) Skill landscape match (per domain)

Within each domain, synth's evaluator distribution mirrors eval's: same `evaluator.func` types in similar ratio, same `result.type` and `options.rules` shapes, same op coverage (which actions/skills the metric rewards). A domain can satisfy domain-share parity (item (6)) while completely mismatching skill mix — synth could route every task to a single dominant `func` while eval relies on a long tail. Same row-count, different reward landscape, no transfer.

**Tolerance**: each `evaluator.func` share within ±15pp of its eval-doable share, OR the absolute synth count is ≥ ⌈eval_count × 0.5⌉ (whichever is laxer). Long-tail funcs with `eval_count = 1` only need `synth_count ≥ 1`.

### (5) Task difficulty cap

Each synth task must be solvable by the same agent class as eval, in roughly the same number of turns. Too-hard rows produce broken trajectories (SFT) and sparse reward (RL) — pulling the train distribution rightward of eval and trading transfer for noise.

**Per-row rule**: when in doubt, ask "is this *obviously* harder than the closest eval task in the same domain?"

- **Acceptable**: synth slightly easier than eval, or in the same band.
- **Reject**: synth obviously harder. Anti-examples:
  - Compound multi-step instructions when eval gives one-step.
  - GUI ops requiring obscure menu paths (Format > Watermark, Drop Caps tab) when eval uses common menus.
  - File formats requiring tools not pre-installed (see [Hard Constraints #15](#hard-constraints) and [Tool-dependency contract](#tool-dependency-contract-transfer-to-eval-consideration)).
  - Calc: pivot tables (openpyxl pivot-XML instability), date-fuzzy multi-CSV joins.
  - Writer: complex track-changes, multi-paragraph indexed ops without quote-anchoring.
  - Any "find the global max-AND-do-X-AND-then-Y" 3-clause compound.

**Concrete caps when proposing a row**:

| Axis | Cap |
|---|---|
| Apps involved | ≤ apps in eval analog (eval is 2-app cross-flow → synth ≤ 2 apps) |
| Distinct ops | ≤ 1.5× ops in eval analog (eval = "rename + copy" = 2 ops → synth ≤ 3 ops) |
| Run-level edits per task | ≤ 2 (subscript max 2 chars; resize max 2 named layers; reorder max 2 slides). Eval `0b17a146` = 1 char ("2" in H₂O) → synth subscript rows cap at 1–2. |
| Multi-file outputs per task | ≤ 5 (eval `9bc3cc16` is per-message export → cap synth analog at ≤3 emails) |
| UI precision | mirror eval (single-char select, exact pixel coord) — don't exceed |
| Estimated turn budget | ≤ 1.3× eval task's natural turn count; default cap 10 turns |

**Operational test**: any new row's `{domain}.md` cell content must be implementable in ≤ ~20 lines of Python in `_make_source_*` AND solvable in ≤ ~10 GUI/CLI turns by inspection. If your instruction needs a paragraph of caveats to be unambiguous, simplify the row (or split into 2 rows).

**When a row blows a cap**: (a) split into N smaller rows, (b) shrink source-file scale (smaller table, fewer slides), or (c) drop and pick a different eval anchor.

**Why not pass-rate match (Q2)?** Pass-rate is a noisy lagging indicator that requires V5 rollouts to measure. The design-time difficulty cap above is upstream and cheap. Chasing exact ±10pp pass-rate parity was found to incentivise bad fixes (drop hard rows, inflate trivial rows). The rollout-pass distribution will broadly track eval if (a) instructions are not infeasibly-hard and (b) sources are not infeasibly-malformed. `Δfeas` (synth vs eval pass rate) remains a **diagnostic only** — see [Notation convention](#notation-convention-for-evaljsonl-references).

### (6) Domain distribution + total volume

**Domain share**: `synth_share(d) − eval_share(d) ∈ [−5pp, +5pp]` for every domain `d`. Prevents over-/under-emphasising any one domain.

**Volume target**: 1500–2000 rows total. Synth is sized larger than perturb because synth's primary value is file-shape diversity, and matching eval's per-domain `files/task` ratio (multi_apps 1.85, libreoffice_impress 1.21, …) at scale means one synth task per meaningfully-different file shape.

- **Cap-and-redesign, not inflate**: scale the *number of templates*, not the *rows per template*. Adding rows on top of pseudo-diverse templates makes the dataset worse.
- **Lowest priority because fixable downstream**: domain-weight imbalance can be corrected at training time with weighted sampling. Quality issues (1)–(3) cannot.

```python
import json
from collections import Counter

def shares(path):
    c = Counter(json.loads(l)["metadata"]["others"].get("domain", "?") for l in open(path))
    total = sum(c.values())
    return {d: n / total for d, n in c.items()}

eval_s  = shares("lite/gym/envs/lite/osworld/data/eval.jsonl")
synth_s = shares("lite/gym/envs/lite/osworld/data/train.synth.jsonl")
for d in sorted(set(eval_s) | set(synth_s)):
    diff = (synth_s.get(d, 0) - eval_s.get(d, 0)) * 100
    flag = " ⚠️" if abs(diff) > 5 else ""
    print(f"  {d:<22s} eval {eval_s.get(d,0)*100:5.1f}%  synth {synth_s.get(d,0)*100:5.1f}%  Δ {diff:+5.1f}pp{flag}")
```

### (7) Instruction style match

Synth-row instructions should **read like eval-side instructions in the same domain**. Not byte-identical, but in the same register / verbosity / persona. **Lowest priority** because instruction-conditional models tolerate phrasing drift well — but worth a check when adding new templates so the dataset doesn't develop a synth-specific "voice" the model can fingerprint.

Calibrate against eval samples per domain:

```
jq -c 'select(.metadata.others.domain=="<domain>") | .instruction[0:120]' lite/gym/envs/lite/osworld/data/eval.jsonl | head -10
```

Common patterns:
- **vs_code / multi_apps**: first-person request, "Please help me ..." / "I'd like to ..." prefix, often with motivation.
- **libreoffice_writer**: imperative or first-person, frequently mentions specific quoted text.
- **libreoffice_impress**: imperative, dense, often parametrises slide indices and exact values.
- **chrome / vlc / thunderbird**: conversational with hedges ("Could you ...", "I find ...", "Help me ...").
- **os**: situational, stating the user's setup and what they want fixed.

**Anti-patterns**:
- All-imperative robot voice ("Set X. Set Y. Set Z.") when eval same-domain rows are first-person.
- Boilerplate prefix repeated across many synth rows ("In this task, you must ...") when eval doesn't use one.
- Missing the "why" — most eval rows include a one-line motivation; synth rows that drop it shift the distribution.

This is a **soft constraint** — unblock real-asset / skill-coverage work first; revisit instruction style only when (1)–(6) are healthy.

---

## Notation convention for `eval.jsonl` references

- **`eval.jsonl`** (unqualified): full eval set, 369 tasks, includes 29 infeasible. Used in the transfer prime directive (final goal).
- **`eval.jsonl (infeasible-filtered)`** / **eval doable** / **eval feasible**: 340-task subset excluding 29 infeasibles. Used for distribution-match comparisons (Q1 + skill landscape).
  - **Synth NEVER constructs `infeasible` rows.** Even when an eval domain is heavily infeasible-weighted (e.g. gimp 37%, vs_code 22%, os 19%), do NOT add `func: infeasible` rows to synth. Refusal-recognition is out of scope for synth design — see §Hard Constraints #6.
- **Δfeas** = `synth_pass% − eval-doable_pass%` on `eval.jsonl (infeasible-filtered)`. **Diagnostic only**, not an optimization target — [Prime Directive §(5) Task difficulty cap](#5-task-difficulty-cap) replaced pass-rate matching with a design-time difficulty cap (chasing exact ±10pp Δfeas was found to incentivize bad fixes; see §(5) rationale).
- Throughout these AGENTS.md and `{domain}.md` docs: any time we compare a synth distribution to an eval distribution, the eval side is **`eval.jsonl (infeasible-filtered)`** unless explicitly stated otherwise.

---

## Scaler architecture

The scaler in `synth/catalog.py` does **one** thing: per-domain cross-domain volume rebalance (PD (6)). It deliberately does NOT enforce intra-domain skill ratio — that's author's responsibility per PD (4a) + PD (4d).

**Algorithm** — `_rescale_for_volume(templates)`:

1. Per domain `d`, compute `target = round(TARGET × eval_count[d] / Σ eval_count)` — a share of one **global** row cap, `TARGET` in `synth/catalog.py`. There is no per-domain multiplier. `TARGET` ships as `math.inf`, so every `target` is `inf`, Stage B below never fires, and every domain reports `UNDER`; re-derive with `uv run python -c "from lite.gym.envs.lite.osworld.src.gen.train.synth import catalog; print(catalog.TARGET, len(catalog.ALL_TEMPLATES))"`.
2. **Stage A** — every FileTask gets `n_rows = min(2, distinct_param_count)`. `distinct_param_count` is probed by running `param_fn(0..3)` and counting unique instructions.
3. **Stage B** — if `sum(n_rows) > target`: downgrade 2-Param FileTasks to 1-Param (sorted by `template_id` for determinism) until `sum(n_rows) ≤ target`. Single-Param FileTasks are never downgraded.
4. If still over after all 2-Param downgraded → emit FLAG `OVER need_manual_comment` to stderr. Author should comment FileTasks in `<domain>.py` to trim further.
5. If under → emit FLAG `UNDER`. Author should add new FileTasks (cap-2×2 limits per-File contribution to ≤4 rows).

**cap-2×2** is the structural ceiling (PD (3b) anti-clone guard):
- Each `File` declares ≤2 `FileTask`s (enforced in per-domain `_emit_templates`)
- Each `FileTask` carries ≤2 `Param`s (enforced via `SYNTH_CAP_PARAMS_PER_TASK` slicing in `_to_synth_template`)
- → max 4 rows per File; **scaling volume = adding Files, not inflating per-File density**

**What the scaler does NOT do:**

- No `_TAXONOMY_EVAL_CLASS_K` global K-weight dict
- No `_EVAL_FUNC_TO_SKILL_CLASS` alias dict
- No `skill_class_of()` helper
- No intra-domain skill-class bucketing
- No silent-zero footgun — no code path mass-zeroes templates when `N > target`; every FileTask is preserved

**Author responsibility** (not auto-enforced):

- **PD (4a) intra-domain skill ratio**: write per-skill FileTask counts proportional to eval skill distribution
- **PD (4d) evaluator metric selection**: use eval-side `evaluator.func` names directly (e.g. `evaluate_strike_through_last_paragraph` rather than `compare_docx_strict + examine_strike`) so synth and eval bucket-align without aliasing

**Audit workflow** — see [Audit tooling §measure_gap.py](#audit-tooling-measure_gappy) below.

- For OVER buckets: comment FileTasks in `<domain>.py`
- For UNDER buckets: add new FileTasks

---

## Design principles (Iter-13 redesign, preserved from former REDESIGN.md §2)

Six principles override every existing convention; they're the only acceptance criteria for any new or refactored template. Each carries its parent Prime Directive tag.

### Principle 0 — File-first, NOT operation-first  *(PD 3c)*

**Outranks all others.** Synth-side enforcement of PD (3c) file-first task design.

- **Wrong** (operation-first): "I want a filter-and-copy-to-sheet skill → build a generic xlsx → tweak it for 5 paraphrases of the same filter task."
- **Right** (file-first): "Here is a real xlsx (or one we author with intent — PnL table, gradebook, attendance). What reasonable tasks could a real user do on **this specific file**?"

Consequence — **synth row = file × tasks × params** (3-axis cartesian):
- **Axis 1 — file**: primary diversity carrier. Many distinct source files, each with its own structural shape.
- **Axis 2 — tasks**: per file, ops are **constrained by what the file affords** (daily-oil-prices invites moving-averages; pnl invites cross-sheet aggregation). Capped at ≤2 per file.
- **Axis 3 — params**: per (file, task), rotate concrete arguments (filter value, target row, color). Capped at ≤2 per task.

Variation comes primarily from **different files**, not from re-skinning the same op. Many small templates (high count, low per-template variant count) — exactly "宁可多 templates, variation 少一点" intuition.

### Principle 1 — 因地制宜 (grounded in source)  *(PD 2a + 3c)*

The seed's chosen op must be **physically possible** on the specific source. Every pool entry must be **derivable from the source builder** at template-design time.

- ❌ Rotating `[(1,1), (15,8), (5,10)]` reposition pool when the pptx canvas is 24×14 cm — `(15,8)` pushes content off-slide.
- ❌ Rotating `filter_value="Tanaka"` when the source xlsx has no row named "Tanaka".

Rule: if you can't show "yes this exists in the source", reject the pool entry.

### Principle 2 — Variation has to be semantically meaningful  *(PD 3b)*

Different paraphrases of the same instruction are **not** a variation axis. Per-seed differences must produce a **structurally different gold artifact** AND a **visibly different instruction**.

- ❌ reposition `(1,1)` vs `(1.1, 1.0)` — sub-cm invisible
- ✅ reposition `(1,1)` vs `(14, 6)` — clearly opposite corners
- ❌ blur radius `[3.0, 3.1, 3.2]` — eye can't tell
- ✅ blur radius `[3, 8, 20]` — subtle / strong / extreme

### Principle 3 — Prefer many templates × few variants over one × many  *(PD 3b + 6 cap-and-redesign)*

Split by skill / parameter cluster instead of forcing one template to cover everything.

- ❌ One filter template with 8 variants rotating `[(Region,N), (Region,S), (Salesperson,Smith), (Amount>1000), ...]`
- ✅ Three templates each with 2-3 variants:
  - `sheet2_filter_by_region`: `{N, S, E, W}`
  - `sheet2_filter_by_status`: `{Pending, Shipped, Cancelled}`
  - `sheet2_filter_by_amount_threshold`: `{>500, >1000, >2500}`

Cheaper to read, easier to debug, each carries a distinct skill specialisation.

### Principle 4 — Coordinates / parameters move visibly  *(PD 3b)*

Numeric op params should differ by ≥1 unit at the scale used in the GUI:

| Surface | Min meaningful Δ |
|---|---|
| Slide cm position | 2 cm |
| Slide pt size | 4 pt |
| Cell column / row | non-adjacent (skip ≥1) |
| Image dimension % | 30% (≥0.3× or ≥1.3×) |
| Color RGB | distinct hues (not shades) |
| Filter values | semantically different (not synonyms) |

### Principle 5 — Eval rule must follow seed  *(PD 1b)*

When the op rotates `new_sheet_name`, `target_pt`, `RGB`, etc., the evaluator's `expected` field has to rotate in lockstep. Every closure that captures a literal target must be rebuilt inside `_eval(p)` reading from `p`.

---

## Anti-patterns to eliminate (preserved from former REDESIGN.md §3)

Coding patterns that caused the 2026-05-10 audit bug cluster — forbidden going forward:

1. **Module-level lambda closing over literals**
   ```python
   gold_builder=lambda s, e: _gold_sheet2_filter(
       s, e, new_sheet_name="North Orders", filter_col_idx=2, filter_value="North",
   ),
   ```
   The lambda captures `"North Orders"` at file-load time — no per-seed call can change it. ⇒ Move literal capture INTO `_eval(p)`.

2. **`_params(seed)` returning fixed body + seed-only-on-instruction**
   ```python
   def _params(seed):
       return {"instr": instructions[seed % len(instructions)]}
   ```
   Seed never touches gold / src / eval. ⇒ Seed must affect op params, not just paraphrase.

3. **`rng = random.Random(seed ^ ...)` then unused** — dead in dozens of factories. Drop the unused rng line.

4. **`n_rows=N` without giving the factory a way to vary the op** — turns one task into N byte-clones. ⇒ Either match `len(Param)` exactly to actual variant count, or remove the inflation.

5. **Eval rule list bound at module load**
   ```python
   rules=_RULE_SHEET_NAME_AND_DATA_NAMED("Forecast"),
   ```
   Frozen sheet name; can't rotate. ⇒ Build rules inside `_eval(p)`.

---

## Audit tooling: `measure_gap.py`

[`devs/envs/lite.osworld/measure_gap.py`](/devs/envs/lite.osworld/measure_gap.py) is the **canonical quantitative gap tool**. It replaces the previous "manual snippet in PD (6)" / `scripts/audit_synth_distribution.py` (which was never authored). Run it after every synth regen:

```bash
uv run python devs/envs/lite.osworld/measure_gap.py                # all 10 domains
uv run python devs/envs/lite.osworld/measure_gap.py --domain os    # subset
uv run python devs/envs/lite.osworld/measure_gap.py --calibration  # only manual-pinned cells
uv run python devs/envs/lite.osworld/measure_gap.py --min-count 3  # suppress eval_fn singletons
```

**What it does**:

- Loads `train.synth.jsonl` + `eval.jsonl`, **filters out infeasibility on both sides** per [`Notation` §eval-feasible](#notation-convention-for-evaljsonl-references).
- Runs **per-domain dimension classifiers** — each domain has 4–6 dimensions tailored to its skill landscape (writer's `examine_*` decomposition, calc's `compare_table.options.rules` decomposition, impress's `compound multi-fn` count, etc.).
- For each `(domain, dimension, category)` cell, prints `synth_n / synth% / eval_n / eval% / Δpp / status / cal`:
  - `✓ ⚠️ 🔴`  status by |Δpp| < 5 / 15 / ≥15 (or ❌ for eval-only-zero-synth coverage holes).
  - `✓cal / ❌cal`  whether the quant value matches the manual baseline pinned in `MANUAL_TARGETS` (±5pp). ❌cal = data drifted from manual; investigate or update baseline.

**Cross-domain dimensions** (applied to every domain):
- `atom_count` — count of `+`-joined sub-fns in `evaluator.func`. Catches eval's compound evaluators (synth almost always atom_1).
- `result_type` — `evaluator.result.type` bucket. Side-channel signal (`vm_wallpaper` / `googledrive_file` / `accessibility_tree` / `is_in_vm_clickboard` / etc.) invisible at eval-fn-name level.

**Domain-specific dimensions** (highlights — see code for full list):
- **os**: `instruction_style` (backtick_leak) · `skill_scope` (gui_settings/shell_pipeline/file_edit) · `system_target` (gsettings_dconf / sys_daemon / userspace_desktop / userspace_dotfile) · `difficulty_nl2bash`
- **chrome**: `eval_fn_family` · `url_leak` · `slot_resolution` (config-preresolved URL vs agent-must-navigate) · `relative_time` (`rule_relativeTime` detection)
- **writer**: `skill_class` (decomposes `compare_docx_strict` via `examine_*` flags + handles direct upstream fns) · `target_anchor` (ordinal vs quote_anchor vs doc_wide) · `evaluator_pattern` (synth dual-pattern + compound multi-property guard)
- **calc**: `save_protocol` (open + Ctrl+S presence) · `skill_class` (decomposes `compare_table.options.rules[*].type`) · `rule_combo` (chained skill shapes) · `source_provenance` (curated_real_xlsx vs synth_inline_openpyxl)
- **impress**: `comparator_strictness` (color_tolerant / position_tolerant / strict) · `op_family` (title_style / image / background / table) · `rgb_leak` · `slide_anchor` (title_text vs ordinal)
- **multi_apps**: `apps_per_task` (apps_le_1 / apps_2 / apps_3plus) · `app_combination` (instruction-kw ∪ eval-fn inferred) · `tool_leak` (backticked pdftk/pandoc/IM/ffmpeg)
- **gimp / vlc / thunderbird / vs_code**: `instruction_leak` (Ctrl+L / menu-path / pref-key / backticked JSON-key) + per-domain `skill_class` mapped from eval_fn
- **thunderbird**: `async_flush` (pkill_kill_signal is **synth-invented**, eval canonical is close_window_only — polarity-corrected v2)
- **vlc**: `media_source` (local_file / remote_url_media / hls_m3u8)

**Critical writer-specific note**: `compare_docx_strict` is one synth fn that covers ~8 different skills via `examine_*` flags. **eval-fn-name matching is impossible for writer**; the classifier decomposes via `examine_*` AND falls back to direct upstream fn names. See [`libreoffice_writer.md`](/devs/envs/lite.osworld/synth/libreoffice_writer.md) for the dual-pattern table.

**When to run**:
- After every `train.synth.jsonl` regen.
- Before every commit that changes a `<domain>.py` generator.
- Before every sweep launch — to confirm no manual baseline drifted.

**Acceptance criteria** (replaces the old `Q-fn` / `Q-app` / `Q-scale` / `Q-metric` 4-gate):
- All ❌cal cells investigated — either classifier bug, or update manual baseline.
- 🔴 cells (|Δpp| ≥ 15) are tagged with a bridge action in the per-domain `.md` plan.
- ❌ coverage holes (eval% > 0, synth% = 0) are tagged with a "Bridge: add N templates" line.
- Infeasibility synth count = 0 (script warns otherwise — see `is_infeasibility` filter).

**Per-domain top-gap snapshots** are pinned at the top of each `{domain}.md` and updated alongside any classifier or generator change.

**Why simpler than legacy**:

- Single source of truth: `len(Param)` in `<domain>.py` directly controls emission
- No K-weight tuning loop (legacy required cycle-after-cycle K bumps to surface new templates)
- Adding/removing template has predictable effect — see code, know output rows
- No silent failures — under-volume domains FLAG explicitly instead of accepting under-coverage

---

## Synth design

### File-first, not skill-first (因地制宜)

Definition + litmus test live in [Prime Directive (3c) File-first task design](#3-real-source--structural-diversity). This sub-section archives concrete eval references that motivate the rule.

**Eval reference inspection** (2026-05-09 dump in `/tmp/eval_dl/`) — every eval file is a real document with specific semantic content; the task is only meaningful given that content:

- calc `01b269ae`: 30×5 movie list with N/A blanks → "fill blanks with value above"
- calc `8b1ce5f2`: 33×6 calendar grid → "highlight weekend columns red"
- writer `6f81754e`: 100-line train signaling timetable → "convert to table"
- writer `8472fece`: 2-paragraph Dolch sight word list → specific reading-aid task
- impress `455d3c66`: 1 slide with 11 text frames + 2 tables → element-specific edit

Pseudo-diversity is a related anti-pattern: N tasks all sharing one `_make_source_*` body with `rng` salt shuffling values inside an otherwise-identical structure. Pseudo-diverse rows pass the Q3 fingerprint check (every heredoc differs by salt) but are functionally one-task-in-N-disguises; the model learns the template, not the skill. **Inflating row count without genuine structural variation makes the dataset worse, not better** — see [Prime Directive (3b)](#3-real-source--structural-diversity).

#### Visual-template diversity (orthogonal to semantic content)

Beyond varying *what's in the file* (semantic content), source files must also vary on **visual-template axes** — what the file *looks like* in the app. Two docx files containing different semantic content but both rendered as default Liberation Serif 12pt single-spacing without headings teach the model that "writer task = default-styled doc". When eval drops a doc with a custom font, multiple headings, embedded images, and non-default line spacing, the agent struggles because synth never showed it that variant.

**Per-domain visual axes** (each Cat 1 multi-variant template should rotate through these, not just text content):

- **writer**: font family / size / line spacing / heading hierarchy / title-page presence / section count / paragraph style / inline format density / embedded objects / footer-header / empty-separator paragraphs. **Canonical example: see [`libreoffice_writer.md` §"Source-file template diversity"](/devs/envs/lite.osworld/synth/libreoffice_writer.md#source-file-template-diversity-mandatory-before-any-plan-table-row-is-implemented)** — the most fully-fleshed-out per-axis matrix; other domains adapt this pattern.
- **calc**: font / cell border styles / column widths / freeze panes / sheet zoom / number-format defaults / theme colors / row-height variation
- **impress**: slide master / theme / font family / background colors / per-slide layout family (Title-only, Title+Content, Two Content, Blank, etc.) / transition defaults / aspect ratio
- **chrome**: pre-existing tab count / Bookmark folder structure / cookie state / Preferences key density / History sqlite size
- **gimp**: image format / channel mode / layer count + naming / canvas size / DPI
- **multi_apps**: each constituent file rotates through its domain's visual axes

**Builder-side requirement**: each `_make_source_*` helper picks a visual-template variant via `rng.choice(_VISUAL_TEMPLATES)` AND emits whatever metadata downstream needs (e.g., docx writer emits `body_idxs` per the F6 ordinal heuristic, see writer.md). Per-row contracts in the plan-table call out the visual values used so the implementer can verify the template diversity at V1.

The combinatorics matter: 6 visual templates × 4 semantic content variants × 5 Cat 1 templates = 120 visually + semantically distinct source files for that domain alone. This is what closes the file-shape gap to eval at scale.

#### Don't mirror eval too tightly (anti-eval-mimicry)

A separate failure mode from "any-file boring" is **eval-mimicry**: synth generates N variants of the *exact instruction frame* eval happened to sample. Eval tests subscript on H₂O once → synth proposes 4 chemistry tasks (CO₂, C₆H₁₂O₆, H₂SO₄, NaHCO₃). All chemistry. The skill "apply subscript run formatting" lives in many naturally-occurring contexts eval did not sample: math indices (x₁, x₂, aₙ, Σᵢ), CS array notation in pseudo-code, footnote/citation markers, music chord-voicing notation, even apparel sizing labels. The evaluator (`compare_docx_strict examine_subscript=True`) grades the *outcome shape*, not the instruction text.

The principle:

> **Synth's job is to teach the SKILL, not to replicate the eval INSTRUCTION.** Eval task_ids seed the choice of skill + evaluator + grading mechanics; from there, generate variants drawn from the broader natural-occurrence space of that skill, not from rephrasings of the eval instance.

This is *complementary* to file-specificity, not a substitute:
- file-specific = the file's content motivates the task (don't be generic about what the file is)
- anti-eval-mimicry = the instruction is drawn from the broader skill domain (don't be narrow about how the skill is invoked)

A synth row should be both: its file is concretely chosen for what it is, AND its instruction is drawn from a context the eval didn't already cover. Two rows that both teach "subscript formatting" should differ on **both** axes — different file genre AND different invocation domain (one chemistry fact-sheet, one math problem set, one sheet music with chord inversions).

How to apply:

1. **Catalog the skill broadly**, not from eval text.
   *"What naturally-occurring contexts demand subscript run formatting?"* → chemistry molecules, math indices, math sequences, summation/integral bounds, vector components, programming subscripts in pseudo-code, music chord voicings, footnote markers, language-specific markers (Japanese furigana proxies), apparel size labels (XS₂, M₃), license-plate codes, version suffixes (v₁.₂).
2. **Pick variants spanning the catalog**, not iterating the eval instance.
   For 4 subscript variants: 1 chemistry (eval-anchor) + 1 math + 1 CS + 1 music or footnote. Each invokes the same `examine_subscript=True` grading.
3. **Verify the evaluator still grades correctly** for each variant. If a variant requires a different `func` (e.g., math problem set with LaTeX would need different parsing), it's a separate row not a variant.
4. **Per template, ensure the variants are mutually distinct on the invocation domain axis**, not just file-genre axis. 4 chemistry-genre files = same invocation domain × 4 file shapes = eval-mimicry. 1 chemistry + 1 math + 1 music + 1 footnote = 4 invocation domains.

**Quick eval-mimicry check before writing any plan-table row**:
- *"If I removed the eval task_id citation, would this row look like a copy of that eval task?"* If yes, the row is a paraphrase, not a contribution — redesign.
- *"Could a domain expert in an unrelated field (musician, programmer, accountant) point to a real document where this exact instruction would arise?"* If no, the row is eval-mimicry.

#### Tool-dependency contract (transfer-to-eval consideration)

Eval tasks generally use **only baseline shell tools + bundled GUI apps** — no specialized CLI utilities. Synth rows MUST match this distribution: a row that depends on a tool not in the OSWorld container is (a) infeasible at training time, AND (b) shape-mismatched against eval (eval doesn't teach those tool-flavored solutions).

**Container baseline (verified in `lite/gym/envs/lite/osworld/docker/Dockerfile`)**:

| Category | Available |
|---|---|
| GUI apps | LibreOffice (calc/writer/impress), Chrome, VSCode, Thunderbird, VLC, GIMP |
| Python libs | openpyxl, python-pptx, python-docx, beautifulsoup4, lxml, mutagen, pymupdf, pyautogui, flask, pynput |
| Fonts | msttcorefonts, fonts-noto-cjk, fonts-crosextra-{carlito,caladea} (matching eval Chinese-font fallbacks) |
| Shell baseline | bash, sed, awk, grep, sort, uniq, head, tail, wc, cut, tr, find, tar, gzip, zip, unzip, curl, ls, cp, mv, rm, chmod, chown, mkdir, df, free, ps, top, ping, dig, stat, du, file, base64, diff, comm, paste, shuf |
| **Installed** | ffmpeg, ImageMagick (`convert` / `montage` / `identify`), pandoc, poppler-utils (`pdftoppm` / `pdftotext` / `pdfinfo`), pdftk-java, jq |
| **NOT installed** | exiftool, rsvg-convert, Calibre (`ebook-convert`) |

**Hard rule**: any plan-table row using a NOT-installed tool is **infeasible** until either (a) the tool is added to the Dockerfile **OR** (b) the row is redesigned to use baseline.

ffmpeg / ImageMagick / pandoc / poppler-utils / pdftk-java / jq are all installed in the Dockerfile and live in `cua-lite/lite.osworld:latest`. See `vlc.py` (snapshot/audio-extract/video-rotate/trim) and `multi_apps.py` (Cat J pandoc, Cat K ImageMagick, Cat L pdftotext/pdfinfo, Cat M pdftoppm) for active examples; `jq` is used by `os.py`'s `asset_jq_*` family. Re-add a binary to the **NOT installed** list above only if it is removed from the Dockerfile.

Note: pygame is NOT on the agent-visible Python path — the snake eval imports a local `pygame.py` stub written to the task cwd by `src/gen/eval/multi_apps.py`, not system pygame.

**Tools that should NOT be added** (too heavy or out-of-eval-shape): Calibre (`ebook-convert`, ~150MB + GUI deps), exiftool (specialized), rsvg-convert (use `python+cairosvg` via pip instead).

#### Evaluator-func mapping (per skill type)

The choice of eval `func` in a plan-table row is NOT free — it must match what the skill changes in the file. The wrong `func` either silently ignores the change (false-pass at eval time) or false-fails on irrelevant noise (LO round-trip artifacts on untouched runs). Mirror perturb's mapping:

**Writer** (per `perturb/libreoffice_writer.py:_build_evaluator`):
| Skill type | Right `func` | Why |
|---|---|---|
| Format-only ops on a paragraph (bold / font name / font size / color / highlight / italic / strikethrough / spacing) | `compare_docx_strict` + `examine_<field>=True` for the targeted field, others False | text content unchanged → `compare_docx_files` would mis-pass; need run-level format inspection. Set ONLY the targeted examine flag to True; set others to False to avoid LO round-trip false-fails on untouched runs. |
| Text-change ops (find/replace, uppercase, lowercase, append paragraph) | `compare_docx_files` | text changes → run-level checks would false-fail on LO round-trip style.name renames |
| Subscript / superscript | `compare_docx_files` ∧ `compare_subscript_contains` (compound) | mirrors eval `0b17a146` exactly |
| Page-level structure (page break, blank page) | `contains_page_break` | structural |
| Table insert / convert-text-to-table | `compare_docx_tables` | structural |
| Image insert | `compare_docx_images` | structural |
| Whole-doc font / default-font change | `compare_font_names` / `find_default_font` | doc-scope, not per-paragraph |
| PDF export | `compare_pdfs` | output format |
| Page numbers in footer | `has_page_numbers_in_footers` | structural |

**Calc** (per `perturb/libreoffice_calc.py:_build_evaluator`):
| Skill type | Right `func` | Why |
|---|---|---|
| Cell value / formula change | `compare_table` `rules=[sheet_data]` | values |
| Cell format (number_format, fill color, font color, bold) | `compare_table` `rules=[sheet_data, style <prop>]` (style rule with `props=['fill','font','number_format']`) | format props |
| Sheet structural (rename, add, reorder) | `compare_table` `rules=[sheet_name]` | sheet-meta |
| Chart insert | `compare_table` `rules=[chart]` | embedded chart |
| Freeze panes | `compare_table` `rules=[freeze_panes]` | window state |
| Single cell formula text | `check_cell` | targeted check |
| CSV export companion | `compare_csv` (alongside compare_table on xlsx) | multi-file |

**Impress** (per `perturb/libreoffice_impress.py:_build_row`):
| Skill type | Right `func` | Why |
|---|---|---|
| Per-shape format (color / font / size / alignment) | `compare_pptx_files` + `examine_<field>=True` for targeted (others False) | shape-prop inspection |
| Text content change | `compare_pptx_files` (no examine flags) | text |
| Position / repositioning (title to bottom, table to bottom, right-align) | `compare_pptx_files` + `examine_<position>=True` | position prop |
| Speaker notes | `compare_pptx_files` + `examine_note=True` | notes prop |
| Image resize | `compare_pptx_files` + `examine_image_size=True` (and/or `examine_modify_height=True`) | shape geometry |
| Slide reorder | `compare_pptx_files` (default — slide-id ordering checked) | slide-list |
| Table insert | `compare_pptx_files` (table-presence + dims) | shape-list |

**Multi_apps**: pick the per-app rule above based on the SINK file's skill type. Image-into-docx → `compare_docx_images`. xlsx-to-docx-table → `compare_docx_tables`. TB profile state → `check_thunderbird_folder` / `compare_csv`. PDF output → `compare_pdfs`.

**Common mistake**: using `compare_docx_files` for "make paragraph X bold" — text doesn't change, so this silently passes regardless of agent action. Same trap with `compare_pptx_files` (no examine flags) for "set font color" — passes even if color unchanged.

When writing a plan-table row, name the targeted field FIRST, then look up the right func + examine flag combination from the table above. If the targeted field isn't covered (e.g., "make a paragraph italic but only the second sentence"), the skill is too granular — redesign or drop.

#### Why file-specificity is structurally easier in Cat 2 than Cat 1

The litmus test is the same for both categories, but the failure rate is asymmetric — and that asymmetry is the structural reason for the Cat 2 dominant default:

- **Cat 1 = file is the *variable axis* over a fixed skill**. The skill stays fixed, the file moves. The path of least resistance is to write one `_make_source_*` body and salt it with `rng` — exactly the boring-template anti-pattern. Making Cat 1 content-specific requires actively writing 4 truly different file bodies for 4 truly different content domains every time; the gravitational pull is toward "works on any file with N rows".
- **Cat 2 = file is *constructed for one specific perturb-orthogonal eval task***. The file shape and content are determined by what that eval task needs to demonstrate the skill against the specific evaluator. Content-specificity is the *default*, not extra discipline — there is nowhere to slip into "any file works" because the file exists to instantiate one particular eval skill.

Empirically (calc + writer audit, 2026-05-09): every flagged "any-file boring" template lived on the Cat 1 side or in the borderline file-driven Cat 2 entries (`synth_calc_freeze_panes`, `synth_writer_insert_empty_table`, `synth_writer_export_pdf_keep_name`). True Cat 2 entries anchored to a specific eval task (`synth_calc_create_sheet2_with_data` per 1273e544/26a8440e/0cecd4f3, `synth_writer_csv_text_to_table` per 936321ce, `synth_writer_insert_image_at_cursor` per 6ada715d) passed the litmus by construction.

This means the Cat 2 dominant default has two reinforcing rationales, not one:
1. **Coverage gain** (the original argument): Cat 2 closes a 0→1 training signal for a perturb-uncovered skill, strictly larger marginal transfer than another file shape for an already-trained skill.
2. **Litmus-test compliance** (newly explicit): Cat 2 is content-specific by construction; Cat 1 must fight gravity to stay content-specific. Choosing Cat 2 is the easier path to high-quality data, not just the higher-leverage path.

Operational consequence: if a Cat 1 candidate cannot articulate ≥4 SEMANTICALLY-distinct file bodies (not just dimensionally-different) before being written, demote to Cat 2 (find an uncovered eval skill instead) — don't promise content-specificity at the design stage and produce templates at implementation.

### Two categories

Every synth template declares its category in `{domain}.md`.

> **Heavy default lean: Cat 2 (~70-80%).** Cat 2 closes a coverage gap (0 → 1 training signal for a skill perturb can't reach), which is a strictly larger marginal transfer gain than Cat 1's "more file-shapes for an already-covered skill". Cat 1 is also a TRAP: it makes it easy to write mindless template repetition — same `_make_source_*` body × N rng salt — which inflates Q3 fingerprint diversity without adding training signal. Cat 1 is justified ONLY when ALL of: (a) eval shows ≥3 distinct file shapes for that skill (proof real diversity matters in distribution), (b) perturb covers <4 shapes, AND (c) you can articulate ≥4 STRUCTURALLY-DISTINCT source files (not just rng-salt-shuffled scalars) that genuinely teach different file-shape interpretations of the same skill. If any condition fails, demote to Cat 2 (find a perturb-orthogonal skill instead). The dataset gains more from filling 1 perturb gap than from 5 more file shapes for an already-trained skill.

**Cat 1 — File-driven design for perturb-known skills.** Start from a realistic, structurally-distinct source file (specific sheet layout / paragraph pattern / slide layout family / image content); end with the most natural task on THAT file, where the underlying skill happens to be perturb-covered (e.g. "set font name", "rename sheet", "navigate URL", "rotate image"). Why: perturb's source-file pool is small (1 base × N paraphrases per archetype = 1 underlying file × N text-only variants); Cat 1 widens the file-shape tail so the model learns the skill in many genuine file contexts, not just memorising a single layout. **Quality bar: ≥4 structurally-distinct file shapes per template, AND eval has ≥3 file shapes for that skill** — if eval only has 1-2 shapes for the skill, Cat 1 over-invests in a low-variance dimension and the marginal SFT gain is wasted; pivot to Cat 2 instead.

**Cat 2 — Perturb-orthogonal skill complement.** Start from a skill that exists in eval but is structurally infeasible for perturb (op is bound to specific data structure); end with the minimal-but-realistic source file that demonstrates it. Examples:

| Skill | Why perturb can't | What synth does |
|---|---|---|
| `=SUM(B2:B10)` total | Op intrinsic to specific cells | Synth xlsx whose B2:B10 has a numeric series; agent writes `=SUM(...)` in B11; eval reads B11.value |
| Add chart with series + title | Chart construction is data-bound | Synth source workbook with clean data range; eval reads `worksheet._charts` |
| Pivot tables, named ranges, freeze panes | Op requires data structure designed for it | Synth builds fitted source files |
| Subscript/superscript runs in docx, TOC, footnote insertion | Run-level format ops require boundaries to exist | Synth seeds runs with the right structure |
| Chrome multi-step bookmark trees / pref hierarchy | Some keys/structures must exist before agent can act | Synth pre-seeds Preferences/Bookmarks for the path eval reads |

Cat 2 candidates come directly from perturb's coverage gap (`perturb.md` Audit 2 "真 gap" rows). The file-first rule still applies — even when skill drives the *choice* of file, the file must be realistically structured. A `["Total"]` in cell A1 + ask "compute the total" is degenerate: the model learns the wording, not the skill.

**No Cat 3.** Difficulty bands (small/medium/large file, boundary cases) are subsumed into Cat 1 as a `_make_source_*` parameter axis (`n_rows`, `n_paragraphs`, `image_size`). One template spanning easy→hard cleanly is better than three sibling templates each locked to one difficulty.

### Per-domain Cat 1 / Cat 2 allocation guidance

Initial recommendations based on perturb's current skill coverage (P/S = skills perturb covers / skills in eval). These are **starting points** — refine via Step 0 audit per domain. The split is "% of new synth template SLOTS to allocate to Cat N", not absolute row count.

| Domain | Perturb P/S (rough) | Recommended split | Rationale |
|---|---:|---|---|
| `chrome` | ~0.85 | Cat 1 30% / Cat 2 70% | Archetypes B/P/T/J already deep on covered skills; Cat 2 priorities = multi-tab/window state, form-autofill, downloads, devtools-bound, multi-step navigation flows |
| `libreoffice_calc` | ~0.70 | Cat 1 25% / Cat 2 75% | High-priority Cat 2 = SUM/AVERAGE formula, PivotTable, named-range, advanced freeze (`B3`/`D5`), conditional formatting, data validation, sparkline. Cat 1 only for ops where eval shows ≥3 file shapes |
| `libreoffice_impress` | ~0.65 | Cat 1 20% / Cat 2 80% | UI-difficulty floor + sweep2 11/11 HOMO_ZERO = AGENT_CAP confirms Cat 1 over-investment yields zero training signal. Cat 2: chart insertion / table addition / master-slide footer / hyperlinks / smart-art / animation |
| `libreoffice_writer` | ~0.75 | Cat 1 30% / Cat 2 70% | Cat 2: TOC, footnote, multi-section page geometry, regex find-replace, tabstops, custom header text, table-cell font, image positioning |
| `multi_apps` | ~0.50 | Cat 1 20% / Cat 2 80% | Cross-app combinations inherently sparse; uncovered app pairs (chrome→TB, chrome→VLC, image batch→archive, PDF→form-fill, calc→TB attachment) dominate the transfer-gap. Most synth ROI is here |
| `gimp` | ~0.70 | Cat 1 30% / Cat 2 70% | Cat 2: layer-blend modes, color-picker fill, crop/canvas resize, path/vector ops, smart-fill/heal. Cat 1 only for ops where image-content domain genuinely matters in eval (rare) |
| `os` | ~0.65 | Cat 1 30% / Cat 2 70% | Cat 2: systemd user services, xdg-mime defaults, cron jobs, `find -exec` patterns, tar/gzip pipelines, env-var manipulation, locale settings |
| `thunderbird` | ~0.60 | Cat 1 30% / Cat 2 70% | Cat 2: Local Folder hierarchy, multi-line HTML signature, calendar entries, address book, message-move-between-folders |
| `vlc` | ~0.65 | Cat 1 30% / Cat 2 70% | Cat 2: subtitle burn-in, audio extraction, format conversion via Convert/Save, equalizer presets, snapshot capture, playlist save |
| `vs_code` | ~0.75 | Cat 1 30% / Cat 2 70% | Cat 2: .code-workspace save, format-document with extension, snippet creation, multi-cursor edits, terminal-task creation, debug-config setup |

**Hard rule**: when in doubt, **PICK CAT 2**. The asymmetric cost of mistakes:
- Wrong-Cat-2 (skill not in eval) = wasted budget but no negative transfer
- Wrong-Cat-1 (template-repetition) = pseudo-diversity that *masks* real eval-train mismatch from V5 audit + dilutes the training signal on covered skills

If a Cat 1 candidate doesn't pass the 3-condition test (eval shape diversity + perturb under-coverage + ≥4 genuine structural shapes you can articulate), demote it. Find a Cat 2 instead — there are always more eval skills than perturb covers.

**Validation hook**: after assigning a template to Cat 1, compute `eval_shapes_for_skill(skill)` — count distinct source-file fingerprints in eval rows targeting the same skill. If `< 3`, demote the template to Cat 2 status (find a different perturb-orthogonal skill to cover) instead of inflating the Cat 1 budget.

---

## Q3 — Source-file fingerprint diagnostic (synth-specific)

For every domain, the unique-file-content fingerprint ratio in `train.synth.jsonl` (distinct synthesised-file fingerprints / synth tasks) must reach **≥80% of eval's ratio**:

| Domain | eval files/task | synth target |
|---|---:|---:|
| chrome | 0.02 | ≥0.02 |
| gimp | 0.85 | ≥0.68 |
| libreoffice_calc | 0.98 | ≥0.78 |
| libreoffice_impress | 1.21 | ≥0.97 |
| libreoffice_writer | 1.13 | ≥0.90 |
| multi_apps | 1.85 | ≥1.48 |
| os | 0.29 | ≥0.23 |
| thunderbird | 0.27 | ≥0.22 |
| vlc | 0.47 | ≥0.38 |
| vs_code | 0.52 | ≥0.42 |

Use sha256 of the synthesis heredoc body (computed at generation time) — counting URL/path strings is not enough.

```python
import json, hashlib
from collections import defaultdict
fp, cnt = defaultdict(set), defaultdict(int)
for l in open("lite/gym/envs/lite/osworld/data/train.synth.jsonl"):
    d = json.loads(l); dom = d["metadata"]["others"].get("domain", "?")
    cnt[dom] += 1
    for s in d["metadata"].get("config", []) + d["metadata"].get("pre_config_steps", []):
        body = s.get("parameters", {}).get("command")
        if isinstance(body, str) and any(k in body for k in ("openpyxl", "Presentation", "Document", "Image.new")):
            fp[dom].add(hashlib.sha256(body.encode()).hexdigest()[:12])
for d in sorted(cnt):
    print(f"  {d:<22s} {cnt[d]:>4d} tasks  {len(fp[d]):>4d} unique fingerprints  ratio={len(fp[d])/cnt[d]:.2f}")
```

> **Q3 is necessary but not sufficient.** It catches identical heredocs; structural pseudo-diversity (heredocs that differ only in `rng` salt) needs the manual eyeball pass — pick 5 random tasks per template, check the source files are *different kinds of file*, not "the same kind, different scalar values".

---

## Hard Constraints

**Inherited from [perturb/AGENTS.md §Hard Constraints](/devs/envs/lite.osworld/perturb/AGENTS.md) (1–7, verbatim)**:

1. No eval leakage (exact + semantic-clone).
2. Verifiable oracle — **and the expected file must be created during agent rollout**, not only `oracle_actions`. (`_make_config_step(expected_py)` must land in `metadata.config` / `pre_config_steps`.)
3. Instruction ordinal/index matches what eval reads AND what a human reader sees.
4. Instruction polarity matches the perturbed value direction.
5. All perturbable action types covered.
6. **No infeasible rows in train (synth OR perturb).** Eval has many `func: infeasible` tasks (gimp 37%, vs_code 22%, os 19%, chrome 5%, vlc 12%) — these test refusal-recognition. **Do NOT mirror that distribution in synth.** The infeasible-distribution gap is intentional and out of scope; refusal-recognition is not a synth design target. When you find an eval-vs-synth gap and it is the `infeasible` row count, ignore the gap. All distribution comparisons (Prime Directive (4) skill landscape, (6) Q1 share) always use `eval.jsonl (infeasible-filtered)` (see [Notation convention](#notation-convention-for-evaljsonl-references)).
7. Drop only on structural infeasibility, not on agent step budget.

**Synth-specific (8–17)**:

8. **Every task declares Cat 1 or Cat 2** in `{domain}.md`.
9. **Source files must be synthesized in-container, not downloaded**. Each template owns a `_make_source_<format>(rng, params)` helper (via `openpyxl` / `python-pptx` / `python-docx` / `PIL` / json) emitted into `pre_config_steps`. No remote URLs.
10. **File-native operations, not skill-shoehorn**. Each task's operation is the most natural thing to do on its source file's actual structure — derived from inspecting the file's content inside the task builder. Two synth tasks must differ in *what they ask the agent to do*, not just in `rng`-shuffled values inside an otherwise-identical file.
    - ❌ **Pseudo-diversity**: one template body × N rng salt → identical structure, different fingerprint.
    - ❌ **Skill-shoehorn**: pick skill, dump on any file (e.g. "italicize paragraph 3" when paragraph 3 is empty).
    - ✅ **Native**: workbook with B2:B10 numeric → "compute SUM in B11"; docx whose paragraph 3 is a quote → "italicize that quote"; pptx with chart on slide 5 → "change the chart title".
11. **File diversity ≥ 80% of eval's per-domain ratio** (Q3 above) — and constraint #10 must hold independently.
12. **Cat 2 templates justify against perturb gap** (Audit 2 真 gap row or perturb-dropped archetype). If perturb covers it, prefer adding to perturb (Cat 1) instead of duplicating in synth.

**Tool-mismatch & oracle-architecture (13–17, lessons from perturb sweep2 audit)**:

13. **LO-normalize all xlsx/pptx/docx golds built via openpyxl/python-pptx/python-docx**. The eval's `read_cell_value` (calc) reads `<v>` cached values directly from XML; openpyxl-saved formula cells lack `<v>` and crash with `NoneType float()`. The agent's LO-saved result HAS `<v>` (LO recomputes on save), so eval crashes only on the gold side → silent FALSE_NEG. Mandatory tail for every calc gold-py:

    ```python
    # _LO_NORMALIZE_TAIL: re-save via soffice headless so `<v>` cached values
    # and datetime serialization match LO's own save format. Critical for any
    # formula-bearing source xlsx (Date+Diff column, Budget+Diff, etc.).
    import os as _os, subprocess as _sp, tempfile as _tf, shutil as _sh
    _td = _tf.mkdtemp()
    try:
        _sp.run(["soffice", "--headless", "--norestore", "--nofirststartwizard",
                 "--convert-to", "xlsx", "--outdir", _td, expected_path],
                capture_output=True, env={**_os.environ, "DISPLAY": ":1"}, timeout=120)
        _conv = _os.path.join(_td, _os.path.basename(expected_path))
        if _os.path.exists(_conv):
            _sh.copy(_conv, expected_path)
    finally:
        _sh.rmtree(_td, ignore_errors=True)
    ```

    Calc only: pptx/docx evals use python-pptx/python-docx (same lib both sides → no XML-level mismatch), so LO normalize is calc-specific. Direct trigger: sweep2 calc 6054afcb — sort gold's formula cells crashed eval until LO normalize was added; isolation test confirmed C2=`=B2-A2` returns `-24.0` post-fix vs CRASH pre-fix.

14. **Eval must NEVER read from a path that only `oracle_actions` creates**. Oracle runs only in `validate.py` (oracle-replay); during real training rollout, only `metadata.config` runs at setup time. Eval reading from `/tmp/perturb_a2_<short>_ls.txt` (created by `oracle: ls -R > path`) returns empty/missing → score 0 unconditionally. Direct trigger: sweep2 multi_apps `_build_a2_check_include_exclude` / `_build_a3_vscode_filemanager` / `_build_a10_batch_jpg_to_png` — ~22 tasks scored 0 before fix. Acceptable patterns:

    | Pattern | Where eval reads | Source of artifact |
    |---|---|---|
    | ✅ Direct command at eval time | `vm_command_line: "ls -R '<dir>'"` | computed live from agent-modified state |
    | ✅ `pre_config_steps` writes expected | `vm_file: /tmp/expected_<short>.X` | `_make_config_step(expected_py)` runs at setup |
    | ✅ Agent writes the result via UI | `vm_file: /home/user/Desktop/X.xlsx` | agent's natural app save |
    | ❌ Oracle creates eval-read artifact | `vm_file: /tmp/<short>_ls.txt` | `oracle_actions` only — broken in training |

15. **Sample slots without replacement when distinctness matters**. `[rng.choice(pool) for _ in range(N)]` can pick the same slide/paragraph/cell twice → contradictory ops on same target ("set slide 6 alignment center AND justify"). Use `rng.sample(pool, N)` instead. Direct trigger: sweep2 impress `_t1_05dd4c1d` / `_t1_04578141` / `_t1_3161d64e` / `_t1_5c1a6c3d`. Exception: when the N slots run different ops (e.g. `set_font_color` on s1 + `set_font_style` on s2), `rng.choice` is fine — different ops on the same slide compose, not contradict.

16. **Instruction wording must match eval's actual scope/path**. "For this repo" / "for my project" / "while working on this project" implies workspace-scope; if the eval reads `~/.config/Code/User/keybindings.json` (USER scope), the agent gets confused and over-cautiously refuses ("VS Code does not support repository-specific keybindings natively"). Same hazard for "in this folder" vs system-wide, "default profile" vs current profile, etc. Direct trigger: sweep2 vs_code `_THEME_TEMPLATES` / `_KB_CREATE_TEMPLATES` line 67/331; agent on 930fdb3b wasted entire trajectory claiming infeasibility.

17. **Template-substitution string transforms must respect every `{var}` context**. If `_stylize_a1_instruction` lowercases the first letter of `{instr}` to embed mid-sentence, but a wrapper ends with a period before `{instr}` (`"...deadline this afternoon. {instr}"`), the lowercase produces a sentence fragment ("...this afternoon. append a..."). Detect the preceding character; only lowercase when the wrapper context is genuinely mid-clause (preceded by `,` / `:` / `then` etc). Direct trigger: sweep2 calc `_stylize_a1_instruction` long-narrative wrapper — 6 tasks. Generalization: any string transform on a substituted value MUST verify behavior at every wrapper site, not just one.

---

## Asset placement rule — `assets/synth/` vs inline `.py` heredoc

**Decision boundary** for every fixture used by a synth `_make_source_*` helper.

### Put it in `assets/synth/` ONLY if at least one is true:

- **(A) Binary / not literal-rebuildable**: real `.jpg` / `.png` / `.svg` / `.pdf` / `.mp4` / `.wav` / `.mp3`. Cannot be expressed as a Python string.
- **(B) Real external content (>1 KB) requiring license/attribution**: Project Gutenberg books, Wikipedia HTML snapshots, FRED CSVs, NASA images, GitHub MIT-licensed code. The `MANIFEST.csv` source URL is the reproducibility / license artifact — that's the *reason* it's a file.
- **(C) Multi-row reuse with non-trivial regen cost**: same blob referenced by ≥3 plan rows AND regenerating costs ≥30 lines of code or a network call. (Rare — most reuse is fine inline via a shared helper.)

### Put it inline in `.py` (heredoc / f-string) — NEVER as a file — if any apply:

- **(a) Small (<2 KB) text boilerplate**: eml headers, vCard, tiny `.ini` / `.json` / `.csv` / `.gitignore` / single-paragraph docx body.
- **(b) `synth-original` with no external license / no source URL**: if you would write `(generated inline)` in the MANIFEST source-URL column, the file shouldn't exist — write the literal in the helper instead.
- **(c) Should be parameterized**: Subject lines, order numbers, prices, names — f-string is more flexible than a frozen fixture and gives free row-to-row diversity.

### Negative examples (deleted 2026-05-09)

- `email/eml/sample-newsletter.eml` (333 B, synth-original) → inline in `_make_source_thunderbird_*` heredoc.
- `email/eml/sample-receipt.eml` (270 B, synth-original) → inline + parameterize order #/total.
- `email/vcard/contact-sample.vcf` (253 B, synth-original) → inline + rng-pick names from a list.

All three failed (a), (b), and (c) — bytes-on-disk gave zero benefit over a `textwrap.dedent('''…''')` in the helper.

### Rule of thumb

If you are **writing** the fixture content yourself, write it in code. If you are **fetching** it from a URL or it came from a real-world source, save it as a file with a MANIFEST entry. The MANIFEST source-URL column should never read `(generated inline)` — that is the smell.

---

## Inherited from perturb (workflow / validation / instruction style)

Three concerns shared with perturb — synth follows the same scripts and conventions, with one synth-only addition each.

### Workflow

Mirrors [perturb/AGENTS.md §Optimal Workflow](/devs/envs/lite.osworld/perturb/AGENTS.md) (Steps 0–8, V1–V5). Synth-specific differences:

- **Step 0**: enumerate (a) eval's source-file shapes per domain and (b) perturb gaps from `perturb.md` Audit 2. (a) feeds Cat 1; (b) feeds Cat 2.
- **Step 1**: per-template `{domain}.md` entry declares Cat 1/2, names `_make_source_*` helper, lists rng-varied **structural** axes (not scalar), points at the eval skill-sig or perturb gap covered.
- **Step 2**: implement `_make_source_*` (rng-driven structural variation) → per-template builder → `pre_config_steps` writing the source file → evaluator → `oracle_actions`.
- **Steps 4–7 (V1–V4)**: same as perturb + V1 adds the Q3 fingerprint check.
- **Step 8 (V5)**: roll `train.synth + train.perturb` together (`--splits train`); same scan / subagent.diagnose / fix-and-restart loop.

**Commit conditions**: Prime Directive (1)–(7) all green (with the priority-tier discipline — (1)–(3) are hard floors); [Q3 fingerprint diagnostic](#q3--source-file-fingerprint-diagnostic-synth-specific) in range; V1 hard checks pass; V2 100%; V3 in range; V4 done; V5 0 unresolved generator bugs across two consecutive sweeps.

### Validation scripts

V1–V5 scripts in [perturb/AGENTS.md §Validation Scripts](/devs/envs/lite.osworld/perturb/AGENTS.md) apply verbatim — only the data path changes (`train.perturb.jsonl` → `train.synth.jsonl`). The Q3 fingerprint diagnostic above is the synth-specific addition.

### Instruction-style mechanics

Same `_build_instruction(ops, rng)` rules as perturb — see [perturb/AGENTS.md §Instruction Style](/devs/envs/lite.osworld/perturb/AGENTS.md). Synth inherits no-`save the file`, polite/multi-step ratios, verb-prefix wrapping. The per-domain register / phrasing tone is governed by [Prime Directive (7) Instruction style match](#7-instruction-style-match).

---

## Pipeline reference

> **Refactor anchor — read this first.** Synth is being largely rewritten as part of this cycle. When deciding what to keep vs. what to replace, split by layer:
>
> - **Underlying logic / scaffolding (从 commit `1fb870a9` 的现有 synth 代码继承)**: how a synth task is shaped end-to-end — existing synth task templates as reference baseline, how `pre_config_steps` materializes a source file in the container, how `expected_py` produces the gold file, the `make_synth_row` dispatcher contract, file-driven design. Even when the heredoc body is rewritten, mirror the *shape* from the synth code at `1fb870a9`.
> - **Implementation details (默认 follow perturb HEAD)**: when an implementation choice is uncertain or a synth-side pattern feels stale, defer to perturb HEAD as primary — `postconfig` (dialog handlers + `LO_SAVE_POSTCONFIG`), `evaluator` strictness flags (`examine_*`), `oracle_actions` + `expected_py`-into-`metadata.config`, `_LO_NORMALIZE_TAIL` for calc gold, instruction style pass, F1–F12 mitigations. Perturb is the battle-tested side; do not re-derive these in synth.
>
> Rule of thumb: if it determines *what* the synth task is and *how* the source/expected files reach the container → look at synth `1fb870a9`. If it determines *whether the rollout grades correctly* → look at perturb HEAD.

### Pipeline overview (4 phases)

```
┌── Phase 0: GENERATION (host, offline) ──────────────────────────────────────┐
│  generator builds heredoc strings:                                          │
│    config_py    = "from openpyxl import load_workbook; ...                  │
│                    wb.save('/home/user/Desktop/X.xlsx')"   ← initial state  │
│    expected_py  = "from openpyxl import load_workbook; ...                  │
│                    wb.save('/tmp/perturb_expected_<short>.xlsx')" ← gold    │
│                                                                             │
│  _make_config_step(py) → {"type": "execute", "parameters":                  │
│    {"command": "python3 << 'PYEOF'\n<py>\nPYEOF", "shell": True}}           │
│                                                                             │
│  make_synth_row / make_perturb_row builds JSONL row:                        │
│    metadata.config = [                                                      │
│       pre_config_steps...,                                                  │
│       _make_config_step(config_py),    ← writes initial source file         │
│       _make_config_step(expected_py),  ← writes /tmp/expected_*.xlsx (gold) │
│       {type: open / launch, ...}                                            │
│    ]                                                                        │
│    metadata.evaluator = {                                                   │
│       postconfig: LO_SAVE_POSTCONFIG,            ← dialog-handler patches      │
│       func: compare_table | compare_docx_strict | ...,                      │
│       result:   {type: vm_file, path: /home/user/Desktop/X.xlsx},           │
│       expected: {type: vm_file, path: /tmp/expected_<short>.xlsx}           │
│    }                                                                        │
│    metadata.others.oracle_actions = [...]    ← gold trajectory (oracle replay) │
└─────────────────────────────────────────────────────────────────────────────┘
                                   ▼ (rollout time)
┌── Phase 1: RESET — host pushes config into container ───────────────────────┐
│  docker run cua-lite/lite.osworld:latest                                             │
│  for step in metadata.config: dispatch_action(computer, step)               │
│    download → host fetches URL, pushes bytes to container path              │
│    execute  → bash -c "python3 <<PYEOF ... PYEOF" inside container          │
│               (config_py + expected_py write Desktop/X.xlsx + /tmp/expected)│
│    launch / open → start LO/Chrome/etc., focus the file                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                   ▼
┌── Phase 2: AGENT ROLLOUT (in container) ────────────────────────────────────┐
│  agent ◄── screenshot ── [container display]                                │
│  agent ─── click/type ──►[Computer API]                                     │
│  up to max_steps OR agent.terminate() / report_infeasible()                 │
│  agent edits Desktop/X.xlsx in place                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                   ▼
┌── Phase 3: POSTCONFIG (after agent) ────────────────────────────────────────┐
│  for step in evaluator.postconfig: dispatch_action(computer, step)          │
│    LO_SAVE_POSTCONFIG:                                                         │
│      activate_window LibreOffice → Esc (close stray dialog) → activate      │
│      → ctrl+s → dismiss "Keep Current Format" / "already exists" if present │
│  Now Desktop/X.xlsx contains the agent's final saved state.                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                   ▼
┌── Phase 4: EVAL (host pulls + compares) ────────────────────────────────────┐
│  _get_result(computer, evaluator.result):                                   │
│    type=vm_file → PULL Desktop/X.xlsx → host cache_dir                      │
│    type=active_tab_info | cookies | accessibility_tree → query CDP/sqlite   │
│  _get_expected(computer, evaluator.expected):                               │
│    type=vm_file → PULL /tmp/expected_*.xlsx → host cache_dir                │
│    type=rule    → literal expected dict from JSONL (no pull)                │
│  metric_fn(result_data, expected_data, **options) ← runs ON HOST            │
│    e.g. compare_table(result.xlsx, expected.xlsx, rules=[...])              │
│         compare_docx_strict(result.docx, expected.docx,                     │
│                             examine_font_name=True, examine_color=False)    │
│  returns reward ∈ {0.0, 1.0}                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Per-piece source-of-truth split

| Piece | Reference | Why |
|---|---|---|
| `make_synth_row` entry + dispatcher | [`synth/_utils.py:make_synth_row`](/lite/gym/envs/lite/osworld/src/gen/train/synth/_utils.py) **@ `1fb870a9`** | Public contract — pinned. |
| `pre_config_steps` shape (write file → container) | `synth/{vs_code,os,chrome}.py` **@ `1fb870a9`** (e.g. `synth/vs_code.py:779`, `synth/chrome.py:907`, `synth/os.py:1748`) | Heredoc-into-`execute` mechanics are stable. Mirror the shape; rewrite the heredoc body for file-first synthesis. |
| `_input_file_step` plain-text helper | [`synth/os.py:_input_file_step`](/lite/gym/envs/lite/osworld/src/gen/train/synth/os.py) **@ `1fb870a9`** | Reusable across domains. |
| `postconfig` (Save-As cliff) | **`common.py:LO_SAVE_POSTCONFIG` @ HEAD** | Dismisses "Keep Current Format" + "already exists". Synth at `1fb870a9` lacks this — must update to HEAD pattern. |
| Evaluator per-field strictness (`examine_*`) | **`perturb/libreoffice_*.py:_build_evaluator` @ HEAD** | Per-op `examine_*` toggles. |
| `oracle_actions` + `expected_py`-into-config | **`perturb/libreoffice_calc.py:_emit_a1_variant` (canonical) + `perturb/libreoffice_impress.py:_build_row` @ HEAD** | Hard Constraint #2: `_make_config_step(expected_py)` must land in `metadata.config`, NOT only in `oracle_actions`. The `1fb870a9` synth code may have the wrong pattern — audit and migrate. |
| `_LO_NORMALIZE_TAIL` for calc gold | **`perturb/libreoffice_calc.py:_LO_NORMALIZE_TAIL` @ HEAD** | F1 mitigation: gold-py runs `soffice --headless --convert-to xlsx` so cached `<v>` values + datetime serialization match LO's format. Synth at `1fb870a9` predates this — required for every new calc gold builder. |
| Instruction style | **`synth/_utils.py:_apply_style_pass` + per-domain `_*_TEMPLATES` @ HEAD** | V3 distribution targets. |

### Operational guidance

- When porting a synth template: look at the **same domain's perturb file at HEAD** for canonical `evaluator` / `postconfig` / `oracle_actions`. Don't dig through synth git history — synth is the lagging side.
- Cite reference commit / file:line in `{domain}.md` per-template entries so the next refactor pass can verify quickly.
- `common.py` is shared between synth and perturb (currently only `NOISE_CANDIDATES` + `LO_SAVE_POSTCONFIG`). Track-specific helpers live in `synth/_utils.py` and `perturb/_utils.py`. If a helper is needed by both tracks, promote it to `common.py` rather than duplicating across `_utils.py` files.

---

## Recurring failure-mode taxonomy (carry over from perturb sweeps)

Every cycle of perturb V5 audits surfaces patterns that would equally bite synth. Internalize this table BEFORE writing a new template — most synth bugs fit one of these slots:

| # | Failure mode | What it looks like | How to avoid in synth |
|---|---|---|---|
| F1 | **Tool-mismatch (openpyxl gold ↔ LO save)** | gold has no `<v>` for formulas → `read_cell_value` crashes → silent FALSE_NEG | apply `_LO_NORMALIZE_TAIL` (constraint #13) to every calc gold-py |
| F2 | **Eval depends on oracle artifact** | training rollout sees empty/missing path → score 0 unconditionally | route eval through `vm_command_line` computed live OR through a `pre_config_steps`-created file (constraint #14) |
| F3 | **Sampling with replacement** | two slots pick same slide → contradictory ops in instruction | use `rng.sample` for distinct-slot semantics (constraint #15) |
| F4 | **Wording-vs-eval-scope mismatch** | "for this repo" with user-scope eval → agent claims infeasibility | match instruction phrasing to eval's actual scope/path (constraint #16) |
| F5 | **String-transform context blind-spot** | lowercase-after-period in narrative wrappers → grammatically broken instructions | verify transform across ALL wrapper contexts (constraint #17) |
| F6 | **Visible-vs-raw index asymmetry** | "fifth paragraph" indexed at `paragraphs[4]` but LO inserts blank-separator paragraphs | filter to visible/non-empty paragraphs before indexing (perturb hard constraint #3) |
| F7 | **Polarity mismatch** | template "I'd rather compose clean replies" + perturb sets `auto_quote=true` → narrative says disable, op says enable | split templates per polarity (perturb hard constraint #4) |
| F8 | **`save the file` instruction leak** | trains agents to do an extra Save-As that pollutes file state | NEVER append; rely on `LO_SAVE_POSTCONFIG` (perturb instruction style) |
| F9 | **Agent-cap masquerading as bug** | HOMO_ZERO/FALSE_NEG looks like a generator bug but agent simply can't iterate over multiple shapes/paragraphs | classify via screenshot inspection in `subagent.diagnose` BEFORE editing generator |
| F10 | **Live-site / non-determinism** | cars.com Cloudflare, Google reCAPTCHA, IMAP gloda race | don't emit the variant; if it must ship, tag `exclude_reason="upstream_live_site_drift"` from [exclude_reasons.py](/lite/gym/envs/lite/osworld/exclude_reasons.py) |
| F11 | **Source xlsx merged cells / charts on round-trip** | openpyxl preserves but may unmerge / drop chart on save | for sources that have these, build minimal source from scratch instead of loading-then-modifying |
| F12 | **Postconfig save race** | `Ctrl+S` fires while a modal dialog is up → agent's edits never flush | use `LO_SAVE_POSTCONFIG` from `common.py` (dialog-handlers); never inline a custom Ctrl+S sequence |

When designing a new synth template, run through F1-F12 explicitly. The cost of pre-screening is minutes; the cost of catching it post-V5 is a sweep cycle.

---

## Required `## Synth task plan` section in every `{domain}.md`

Every per-domain plan MUST include a single canonical table titled `## Synth task plan` that an implementer can read top-to-bottom without cross-referencing other sections. This is the implementation contract. The table replaces (does not supplement) ad-hoc "Cat 1 templates" / "Cat 2 templates" tables — those become exposition, the plan table is the source of truth.

**Required columns** (in this order, no omissions):

| # | Cat | Source file(s) (concrete) | Synth task instruction | Eval skill mirrored | Eval `func` (+ key options) | Eval task_id citations |
|---|---|---|---|---|---|---|

**Column rules**:

1. **#** — sequential row id within the domain (`1`, `2`, … `N`).
2. **Cat** — `1` (file-driven, perturb-known skill) or `2` (perturb-orthogonal). Per the Cat 2 dominant default, ≥70% of rows should be Cat 2.
3. **Source file(s) (concrete)** — list every file the synth `pre_config_steps` materializes, with their semantic content (NOT just dimensions). For multi-file tasks, list both/all. Example good: "xlsx 5×30 sales-rep × 6-month matrix (Sales Rep / Jan / Feb / Mar / Apr / May / Jun cols, 30 rep rows with int sales)". Example bad: "xlsx 5×30 with header".
4. **Synth task instruction** — the actual instruction-template text the synth row will emit, verbatim or near-verbatim. Must reference specific column names / paragraph topics / image subjects from the source file (litmus test: would this instruction make sense on an arbitrary file with the same shape? If yes, redesign).
5. **Eval skill mirrored** — what eval skill this synth row teaches. One sentence. Should match an actual eval task's skill, not a fabricated one.
6. **Eval `func` (+ key options)** — the evaluator function name and the options that make grading work (e.g., `compare_table` + `rules=[{type:'sheet_data', sheet_idx0:0}]`; `compare_docx_strict` + `examine_subscript=True`; `check_thunderbird_prefs` on key X).
7. **Eval task_id citations** — 1-3 verified eval task_ids that motivate this synth row. Use the 8-hex suffix (e.g., `1273e544`, `26a8440e`). Mark `BROKEN_CITATION` if the cited task does not exist or does not match the proposed eval `func`. A row with `BROKEN_CITATION` MUST be either anchored to a real task or removed before implementation.

**Multi-variant rule**: if a synth template has 4 file-content variants (per the file-specificity litmus test), emit one row per variant. The plan table counts CONCRETE file shapes, not template families. A 5-template plan with 4 variants each = 20 rows in the plan.

**Template-shorthand exception (added when scaling toward ≥1000 plan rows)**: when scaling out template-mode variants (perturb-style TYPE_1/TYPE_2 patterns), the plan-table may use **batched range rows** of the form `| 201-210 | ... | <one-line description> covering all 10 variants | ... |` — each such row counts as N implied rows in the plan-row total. This shorthand is permitted IF AND ONLY IF: (a) the row text enumerates each variant concretely (e.g., `10 rows: csv→json, json→csv, yaml→json, ...` listing each parameter inline), AND (b) the implementer can read the row top-to-bottom and produce the N concrete instructions without further design work. Plan totals report both individually-numbered count and implied count separately. Otherwise, the one-row-per-shape rule stands.

**Plan table goes BEFORE** any `## Cat 1 templates` / `## Cat 2 templates` exposition tables, which become commentary on the plan rather than the primary deliverable.

#### Source mechanism — required annotation per plan-table row

Every plan-table row's `Source file(s)` cell MUST end with an explicit `[mech: <X>]` tag declaring how the file is produced at task-setup time. Three valid mechanisms:

- **`[mech: prog]`** — fully programmatic in `pre_config_steps` heredoc (python-docx / python-pptx / openpyxl / PIL / reportlab / ffmpeg / shell script writes the file from scratch). No external download.
- **`[mech: hf]`** — file lives in the `cua-lite/lite.osworld-assets` HuggingFace dataset (downloaded to `.cache/assets/synth/` by install.sh); staged into the container via `_stage_asset()`/`host_push` in `pre_config_steps`. Use for **real photos, scanned documents, complex graphics, real audio/video** that programmatic synthesis would render visually generic.
- **`[mech: prog+hf]`** — a programmatic container (e.g., pptx framework built via python-pptx) that EMBEDS one or more `hf`-downloaded real assets (e.g., real floor-plan png). Common for impress/multi_apps rows that need realistic embedded media.

**Why this matters**: synth that's 100% programmatic produces visually-generic content (geometric shapes, lorem-ipsum-style text) — even if the *task instruction* is content-specific, the *file content* doesn't carry real-world signal the model learns from. Eval's source files are real (real photos, real form text, real spreadsheet data); train should match that distribution. Heavy `prog`-only allocation is a covert pseudo-diversity failure: file content varies in dimensions but not in semantic richness.

**Allocation guideline (per domain)**:

| Domain | `prog` % | `hf` % | `prog+hf` % | Notes |
|---|---:|---:|---:|---|
| calc | ~95% | ~0% | ~5% | xlsx is mostly tabular data; programmatic + curated text snippets is realistic. |
| writer | ~70% | ~5% | ~25% | docx body text programmatic, but image-insert rows + scanned-doc genres need real images. |
| impress | ~30% | ~5% | ~65% | pptx framework programmatic, but slide images / photos must be real for portfolio/recipe/property/etc. genres to be content-specific. |
| chrome | ~85% | ~10% | ~5% | mock HTML programmatic, but a few rows need real product images / page screenshots. |
| gimp | ~5% | ~95% | ~0% | gimp eval IS about real image content; programmatic synthetic shapes don't teach the skill. |
| multi_apps | ~50% | ~25% | ~25% | mixed — depends on row's source-file types. |
| vlc | ~75% | ~25% | ~0% | ffmpeg synthetic clips for most; some rows need real music / real video for audio-extract / rotate skills to test realistic content. |
| thunderbird | ~95% | ~0% | ~5% | profile state is structural; email body text programmatic. |
| vs_code | 100% | ~0% | 0% | source code files are programmatic. |
| os | 100% | ~0% | 0% | filesystem state is programmatic. |

**Hard rule**: if a plan-table row depends on the file's *visual realism* (image-content task, photo-portfolio impress deck, real-scanned-document writer task, etc.) and yet declares `[mech: prog]`, that's a content-specificity violation — redesign to `[mech: hf]` or `[mech: prog+hf]`, or drop the realism requirement.

**`hf` asset bundle** — hosted on HuggingFace at [`cua-lite/lite.osworld-assets`](https://huggingface.co/datasets/cua-lite/lite.osworld-assets); `install.sh` downloads it (pinned revision) into `<env>/.cache/assets/synth/`. Only `MANIFEST.csv` / `README.md` / `download.sh` remain in-repo under [`data/assets/synth/`](/lite/gym/envs/lite/osworld/data/assets/synth/). See [`assets/synth/README.md`](/lite/gym/envs/lite/osworld/data/assets/synth/README.md) for the canonical layout + sourcing rules.

Lifecycle:

1. **Obtain (runtime/codegen)**: `install.sh` downloads the bundle into `.cache/`; both runtime (`dispatch.py`) and codegen (`synth/`) resolve it through `asset_root()` (`src/utils/assets.py`, `.cache/` → in-repo `data/` fallback). Assets reach the container via host-side push (`host_push`) — no `file://`/HTTP fetch at task time.
2. **Publish**: rebuild the bundle from source URLs with `data/assets/synth/download.sh`, upload to `cua-lite/lite.osworld-assets`, then bump `ASSETS_REV` in `scripts/install.sh`. Path layout is stable across the HF repo and the local cache.

Sourcing priority (per the README):
- **CC0 / public domain** (Wikimedia Commons CC0, Unsplash, OpenClipArt, FreeFloorPlans, NASA, gov open-data) — always preferred.
- **CC-BY / Unsplash License** — fine; record attribution in MANIFEST.
- **AI-generated** (SD / DALL-E) — for scarce categories, with prompt saved in MANIFEST.csv for reproducibility.
- **HARD RULE: NEVER reuse OSWorld eval source files** in train assets. Identical files in train and eval = train/eval contamination → eval no longer measures generalization. Source train assets independently. V1 hash-compares `assets/synth/` against `/tmp/eval_dl/` and fails on any sha256 match.
- **NEVER live external URL hotlink** — F10 instability.

Per-template in `{domain}.md` plan-table rows: `[mech: hf]` cells cite the relative path under `assets/synth/` (e.g., `assets/synth/photos/architecture/floor-plan-studio-01.png`). Manifest CSV entry is required before the row can be implemented; un-manifested file references = V1 failure.

Bundle build (populating the directory + MANIFEST.csv) is a **prerequisite** to implementing any `[mech: hf]` row. Until per-asset entries land in the bundle, those rows are deferred.

---

## Per-domain audit loop (run before commit, every cycle)

Run this 6-step loop per domain whenever the plan table changes (new rows, eval fixes, perturb refactor, sweep findings). The loop is iterative — re-run until none of the 6 steps surfaces an issue. Do NOT commit a per-domain plan that has not passed all 6 steps in the same cycle. Subagents are appropriate for steps 1-3 (read-heavy landscape work); steps 4-6 are critical-judgment audits done by the main agent (subagent reliability is ~50% on subjective audits).

**Step (1) — Landscape scan (3 axes)**

Build the current view of:
- **eval-fn landscape**: which `func` + `options` combinations does eval actually use in this domain? (Run a one-shot `python -c` over `eval.jsonl` per domain to enumerate.) Catch every custom `func` (e.g., writer's `check_highlighted_words`, `evaluate_strike_through_last_paragraph`, `check_italic_font_size_14`) — many are one-off.
- **skill landscape**: which skills does eval test? Group by `func` family and by instruction theme. Cross-list against perturb's coverage (saturated vs gap).
- **app-combination landscape (multi_apps only)**: which app pairs does eval test (chrome→docx, xlsx→csv, image→archive, TB→csv, …)? Which pairs are uncovered or underrepresented?

Output: a fresh enumeration committed to `{domain}.md` § "Verified eval gap inventory" (or similar). Do NOT carry forward stale enumerations from prior cycles — eval/perturb both move.

**Step (2) — Mismatch → synth design**

For each mismatch identified in step 1:
- skill in eval but NOT in perturb → candidate Cat 2 row
- skill in perturb already + eval has ≥3 distinct file shapes for it → candidate Cat 1 row (file-shape diversification)
- skill in perturb already + eval has ≤2 file shapes → drop (no marginal value)
- skill in perturb but with known F1-F12 failure → synth complements with the F-mitigated variant

Apply the Cat 2 dominant default (70-80%). The mismatch table directly drives the new plan rows; do NOT propose rows that have no mismatch backing.

**Step (3) — File diversity matches eval**

Profile eval source files (download via per-perturb `Step 0` script into `/tmp/eval_dl/{domain}/`). Measure:
- For .pptx: slide count, layout family, text-frame-per-slide distribution, image/table/chart counts, notes presence
- For .docx: visible paragraph count, table count, image count, section count, heading hierarchy
- For .xlsx: sheet count, row × col, merged cells, formula presence, datetime cells

Per template-family in the plan, ensure the proposed source-file variants span the same axes eval actually exercises. A template with 4 source variants where all are "1-section docx with 5 paragraphs" fails this step; eval has docs with 1-100 paragraphs, 0-13 tables, 0-9 images, 1-5 sections.

**Step (4) — Audit content-specificity (litmus test)**

For each plan-table row, run the litmus question from §"The 'any-file vs content-specific' litmus test":

> *"If I swap this template's `_make_source_*` body for an arbitrary file with the same structural shape, does the task instruction still make sense?"*

If yes → redesign or drop. The instruction must depend on the file's actual semantic content (column names that mean something, paragraph topics that anchor the request, image subjects that motivate the action). Repetitive template instances (4 chemistry subscript rows, 4 "fill bg with color" rows, 4 "title to bottom" rows) all fail this test — see §"Don't mirror eval too tightly".

**Step (5) — Audit value + feasibility**

For each plan-table row:
- **Value**: does this row close a real gap (Cat 2) or genuinely diversify a saturated skill (Cat 1 with ≥3 eval shapes)? If neither, drop.
- **Feasibility**: does the cited eval `func` actually grade this outcome shape? (Cross-check `func` against perturb's `_build_evaluator` and against eval.jsonl.) Does the postconfig path (LO_SAVE_POSTCONFIG, dialog handlers) reach the evaluator without losing state? Are there F1-F12 trip-wires that pre-screen would catch?
- **BROKEN_CITATION sweep**: any row whose cited task_id doesn't exist or whose `func` doesn't match the cited task — flag and either re-anchor or drop.

**Step (6) — Audit difficulty (synth ≤ eval analog)**

Apply the §"Difficulty cap (synth ≤ eval analog)" caps to every row. **Reject any row that is obviously harder than its cited eval anchor** — too-hard synth rows produce FALSE_NEG noise at rollout, not training signal, and pull the train distribution rightward of eval. Concrete checks:

- Apps involved ≤ apps in eval analog (eval is 2-app → synth must be ≤2 apps)
- Distinct ops ≤ 1.5× eval analog
- Run-level edits ≤ 2 (subscript max 2 chars, resize max 2 named layers, reorder max 2 slides)
- Multi-file outputs ≤ 5
- UI precision (single-char select, exact pixel coord) — mirror eval, don't exceed
- Estimated turn count ≤ 1.3× eval task's natural turn count; default cap 10 turns

When a row blows a cap: (a) split into N smaller rows, (b) shrink the source-file scale, or (c) drop and pick a different anchor. The purpose isn't to make synth easy — it's to keep synth's training distribution centered on the SAME difficulty band as eval, not 2× harder.

**Loop exit criterion**: a per-domain plan passes the audit when steps 1-6 each produce zero new findings on the same iteration. Treat any new finding as evidence the loop must run again. Iterate until the loop is silent, THEN commit.

---
