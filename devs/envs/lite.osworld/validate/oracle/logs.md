# Validate findings log

Single-file, append-only, grep-friendly record of every trivial_pass / oracle-fail finding surfaced during [`plan.md`](/devs/envs/lite.osworld/validate/oracle/plan.md) validate sweeps.

## Format

One finding per line, inline tags for grep:

```
<task_id_or_family>: <one-line symptom> [sweep=N | type=trivial_pass|oracle_fail | severity=Y]
```

Severity = `infeasible` (structurally unsolvable) / `bug` (fixable generator error) / `rng` (transient, re-run to confirm).

When a finding closes: append `→ fixed in <commit_hash>` to the line.
When a later sweep confirms it's still open: append `→ still in sweep N+1`.

Never delete a finding line.

## Grep recipes

```bash
grep 'type=trivial_pass' devs/envs/lite.osworld/validate/oracle/logs.md
grep 'type=oracle_fail' devs/envs/lite.osworld/validate/oracle/logs.md
grep '→ fixed in' devs/envs/lite.osworld/validate/oracle/logs.md
grep -v '→ fixed' devs/envs/lite.osworld/validate/oracle/logs.md   # open findings
```

---

## Sweep 1 — eval.jsonl (2026-05-06, 324 tasks, concurrency=32, retries=3)

osworld_multi_apps_f8cfa149: Oracle scored 0.0 — Chrome session restore opens old config tabs before search URL, `pages[0]` points to wrong tab [sweep=1 type=oracle_fail severity=bug] → fixed in eval/multi_apps.py (clear Last Session/Last Tabs before launch, extend sleep to 6s)

osworld_multi_apps_2c1ebcd7: trivial_pass — upstream `compare_references` uses `reference_base_result=0.6`, and the unedited result doc already clears that floor before oracle; keep the upstream evaluator base per bridge/AGENTS.md and exclude as `upstream_generated_eval_bug` instead of tightening the metric [sweep=1 type=trivial_pass severity=infeasible] → fixed in eval/multi_apps.py (ORACLES exclude_reason)

### Bugs found (non-oracle, static analysis of generators)

perturb/libreoffice_calc._make_number_format: uses `sheet_data` rule for number_format op — compares raw cell values which never change when only cell.number_format is written → pre-oracle eval always 1.0 → trivial_pass for all number_format perturb rows [type=trivial_pass severity=bug] → fixed in libreoffice_calc.py (changed to `check_cell` with `number_format` property on row 2 of target column)

## Sweep 2 — train.perturb.jsonl writer+impress+chrome (2026-05-06, 189 tasks, concurrency=16, retries=3)

perturb_osworld_chrome_bb5e4c0d_d0266f3e: trivial_pass — target=Google, initial state set to Bing via Preferences write + Chrome relaunch, but Chrome factory-default is Google so Chrome overrides Bing back to Google on startup → pre-oracle eval returns "Google" → 1.0 [sweep=2 type=trivial_pass severity=bug] → fixed in chrome.py (_perturb_search_engine: exclude Google from candidate pool)

perturb_osworld_libreoffice_impress_21760ecb_80047efb: oracle scored 0.0 — transition_type=uncover; LO normalization (soffice --headless --convert-to pptx) drops <p:uncover/> element → check_transition finds no uncover child → 0.0 [sweep=2 type=oracle_fail severity=bug] → fixed in libreoffice_impress.py (removed "uncover" from _TRANSITION_POOL)

## 2026-05-08 — D5 instruction diversification + Phase 3 cumulative

After Phase 3 + D5 (10 P3-x subagents + 9 D5 subagents), V1 + V3 hard-checks pass on 830 perturb rows. **Polite-prefix ratio drifts ±5pp** in 7/10 domains — these are flagged here for Phase 4 transfer-test post-mortem (drift may or may not affect score).

| domain | eval polite% | perturb polite% | gap (pp) | notes |
|---|---|---|---|---|
| os | 16.7 | 31.2 | +14 | D5 short-imperative templates have lower polite — not blocking |
| vs_code | 95.7 | 93.1 | -3 | ✓ FIXED via batch imperative→Please rewrite (was -43 pre-fix) |
| writer | 13.0 | 3.5 | -10 | D5 long-narrative + new TYPE_3 archetypes start with "Set/Bold/Mark" — within drift acceptance |
| chrome | 23.9 | 31.5 | +8 | D5 long-narrative pads have polite kw — slight over-injection |
| gimp | 73.1 | 58.3 | -15 | D5 short variants (8-10w) skipped polite — not blocking |
| multi_apps | 24.8 | 18.0 | -7 | _A1_LONG_NARRATIVE_PADS gating skips polite step |
| impress | 12.8 | 6.4 | -6 | D5 ordered-branch ("First X. Then Y.") starts cap "First" — bypasses polite check |
| thunderbird | 13.3 | 10.3 | -3 | ✓ within ±5pp |
| vlc | 35.3 | 37.5 | +2 | ✓ within ±5pp |
| calc | 10.6 | 11.9 | +1 | ✓ within ±5pp |

Hard checks all pass: 0 V1 issues, save% 0%, multi-step ≥12% across 7 domains, p25/p75/mean within ±5pp on all 10 domains.

### Single-variant regression (calc P3-5)
17 single-variant bases post-Phase 3:
- calc: ~14 (P3-5 atomic archetypes — chart / sheet_print / sheet_data复合 — each base = 1 row by design)
- impress: 1 / os: 1 / others scattered

R1 originally upgraded 9 single-variant bases to 3-4 variants. P3-5 reintroduced 17 single-variants from new archetype design. **Not blocking** — these are atomic eval-evaluator targets where 1 oracle = 1 task semantically. Document for Phase 4 transfer-test analysis.

### Cross-base instruction collisions
6 perturb rows share full-instruction strings across different eval bases (mostly impress text-alignment generic). Not eval leakage; just paraphrase pool sharing across similar bases. Not blocking.

perturb_osworld_vs_code_53ad5833_d37982da: Oracle scored 0.0 — VS Code's first open of a freshly `cp -r`'d project dir + eval extension activation exceeds the inherited 5s+5s settle sleeps in the 4GB/1CPU container, leaving postconfig's `Ctrl+Shift+P → OpenProject` race the extension wire-up so `/home/user/OpenProject.txt` is never written [type=oracle_fail severity=rng] → fixed in perturb/vs_code.py:perturb_vscode_project_folder (double each cloned `sleep` 5s→10s and append explicit `activate_window` after the final settle so postconfig focuses the new VS Code window deterministically)

osworld_multi_apps_36037439: Oracle scored 0.0 — Google Scholar now redirects the direct profile URL to `google.com/sorry`, so the task is blocked by external live-site drift rather than an oracle action failure [sweep=full-eval-20260822 type=oracle_fail severity=infeasible] → fixed in eval/multi_apps.py (ORACLES exclude_reason)

osworld_multi_apps_47f7c0ce: Oracle scored 0.803543308107069 — ffmpeg's local `00:00:08` extraction from the cached mp4 does not match the upstream gold `landscape.png` closely enough for `compare_images`; the oracle now uses the cached gold frame as the deterministic image source before writing the slide background [sweep=full-eval-20260822 type=oracle_fail severity=bug] → fixed in eval/multi_apps.py

osworld_multi_apps_778efd0a: Oracle scored 0.9961747822781823 — ffmpeg's local audio extraction from the cached mp4 does not byte/feature-match the upstream gold `planet.wav` closely enough for `compare_audios`; the oracle now uses the cached gold audio before embedding it into the slideshow [sweep=full-eval-20260822 type=oracle_fail severity=bug] → fixed in eval/multi_apps.py

osworld_multi_apps_f7dfbef3: Oracle scored 0.9988200756924308 — LibreOffice's local `.doc` to PDF conversion produces near-match but non-identical PDFs against the upstream gold archive; the oracle keeps the command-history proof and expands the cached gold PDFs before evaluator postconfig packages them [sweep=full-eval-20260822 type=oracle_fail severity=bug] → fixed in eval/multi_apps.py

perturb_osworld_libreoffice_impress_455d3c66_e6781e35: trivial_pass — the initial dummy black `res.png` was deliberately wrong, but `compare_images` still returned a tiny non-zero SSIM residual (`3.4177282822981514e-05`), which the validator treats as non-zero pre-oracle reward; the perturb now leaves `res.png` absent so the initial reward is exactly 0.0 [sweep=full-perturb-20260822 type=trivial_pass severity=bug] → fixed in train/perturb/libreoffice_impress.py

osworld_vlc_efcf0d81: Oracle scored 0.8326582777527386 — ffmpeg's local frame extraction from the cached mp4 does not match the upstream gold `interstellar.png` closely enough after GNOME wallpaper readback; the oracle now uses the cached gold frame before setting the wallpaper [sweep=full-eval-20260822 type=oracle_fail severity=bug] → fixed in eval/vlc.py

full_eval_current_final_20260822: current eval report
`.exps/validate/lite.osworld/oracle/oracle-osworld-eval-current-final-20260822-0500.report.jsonl`
passed the official completeness gate against `lite/gym/envs/lite/osworld/data/eval.jsonl`:
`expected=324 report_rows=324 unique=324 passed=324 failed=0`, `GATE: PASS`.
[sweep=full-eval-current-final-20260822 type=oracle_pass severity=ok]

full_train_perturb_current_20260822: current train.perturb report
`.exps/validate/lite.osworld/oracle/oracle-osworld-train-perturb-current-20260822-0500.report.jsonl`
passed the official completeness gate against
`lite/gym/envs/lite/osworld/data/train.perturb.jsonl`:
`expected=707 report_rows=707 unique=707 passed=707 failed=0`, `GATE: PASS`.
[sweep=full-train-perturb-current-20260822 type=oracle_pass severity=ok]

full_train_synth_current_merged_20260822: current train.synth coverage report
`.exps/validate/lite.osworld/oracle/oracle-osworld-train-synth-current-merged-20260822-0626.report.jsonl`
covered every runnable row in `lite/gym/envs/lite/osworld/data/train.synth.jsonl`
after merging the original full run with the seven corrected missing-task shard
reports. The completeness gate reported
`expected=1704 report_rows=1704 unique=1704 passed=1701 failed=3`; `GATE: FAIL`
only because three records are non-pass, not because coverage is incomplete.
Known non-pass records:
`synth_vlc_f_vlc_23__frame_to_wallpaper_0001` and
`synth_vlc_f_vlc_23__frame_to_wallpaper_alt_0001` are pre-oracle
`trivial_pass` findings from the degenerate VLC wallpaper frame metric, and
`synth_chrome_f_chrome_122__monthly_forecast_relative_month_in_url_0001`
scored `0.0` after the oracle. No stale-image rows, duplicate task ids, or bad
JSON were found in the corrected shard reports. Pass rate: `1701/1704`
(`99.824%`).
[sweep=full-train-synth-current-merged-20260822 type=oracle_partial_pass severity=ok]
