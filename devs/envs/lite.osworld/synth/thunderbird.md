# Thunderbird — Synth Plan

> Keep in sync with code. Implementation: [`thunderbird.py`](/lite/gym/envs/lite/osworld/src/gen/train/synth/thunderbird.py).
> Common workflow: [`AGENTS.md`](/devs/envs/lite.osworld/synth/AGENTS.md). Cross-reference: [`perturb/thunderbird.md`](/devs/envs/lite.osworld/perturb/thunderbird.md).

## Current quant-gap snapshot (`measure_gap.py` v2)

Run `uv run python devs/envs/lite.osworld/measure_gap.py --domain thunderbird` for live numbers. Synth N=67, eval N=14 (1 infeasibility filtered).

| Dim | Synth | Eval | Δpp | Status | Bridge |
|---|---:|---:|---:|:-:|---|
| `async_flush.pkill_kill_signal` | 0% | 0% | 0 | ✓ | Resolved — `_folder_evaluator` postconfig now mirrors eval a10b69e1 exactly (bare `ls -R`, no `_kill_tb_step`). Oracle still kills TB before writing folder dirs; folder state is already on-disk by eval time |
| `async_flush.other_postconfig` | 35.8% | 35.7% | +0.1 | ✓ | Resolved — folder check_list rows now route to `other_postconfig` (bare `ls -R` without close_window), matching eval distribution |
| `instruction_leak.pref_key_leak` | 0% | 0% | 0 | ✓ | Resolved — 13 instructions rewritten to user-intent voice (no `mail.*`/`browser.*` keys, no `about:config`, no XUL ids like `threadPaneBox`). Each instruction structurally distinct |
| `async_flush.close_window_only` | 64.2% | 57.1% | +7 | ⚠️ | Aligned — eval canonical |
| Pref-key family coverage | uneven | various | — | ⚠️ | Synth clusters on safebrowsing/html_compose; eval probes `mail.identity.*.signature_text`, `applyIncomingFilters`, dark-mode theme, `mailnews.reply_*`. Diversify pref-keys |
| `check_accessibility_tree` | 0% | 2 rows | -14 | ❌ structurally exempt | Live a11y-tree state not generable from train — keep as known unaddressable per AGENTS.md |

**Quant-correction**: gap.md initially framed `pkill` as a missing safeguard ("synth should add pkill"). **v2 inverted**: quant shows eval has 0% pkill (uses profile snapshots, doesn't need force-kill); synth's pkill is synth-invented over-aggression. **Cycle-thunderbird-dom-fix**: removed `_kill_tb_step()` from `_folder_evaluator` postconfig (the only remaining pkill site) so the 20 check_list rows now mirror eval a10b69e1 exactly. Oracle still kills TB before writing the folder dirs; the folder state persists on disk for eval to read.

## Current shape

**35 Files / 70 FileTasks / 48 current rows after scaler / 5 `eval_class` buckets.**

| eval_class | FileTasks | Files | Eval func | Notes |
|---|---|---|---|---|
| `folder_filter` | 16 | F_TB_01..08 | `check_thunderbird_filter` | msgFilterRules.dat rule records |
| `folder_list` | 8 | F_TB_09..12 | `check_list` | top-level Local Folders pairs |
| `nested_folder` | 8 | F_TB_13..16 | `check_list` | `parent.sbd/<child>` hierarchies |
| `folderTree_mode` | 16 | F_TB_17..20, F_TB_31..34 | `check_json` | xulstore.json — folderTree mode AND non-mode `<elem>.<attr>` |
| `pref_setting` | 22 | F_TB_21..30, F_TB_35 | `check_thunderbird_prefs` | prefs.js bool / int + signature (plain + HTML) |

Param-level `eval_kind` dispatcher (`_build_evaluator`, 8 kinds): `filter`, `folder`, `nested_folder`, `xulstore`, `xulstore_attr`, `signature`, `pref_bool`, `pref_int`. The dispatcher is finer-grained than `eval_class` because `pref_setting` covers three prefs.js shapes (signature regex, bool eq, int eq) and `folderTree_mode` covers two xulstore shapes (mode-key and arbitrary `<elem>.<attr>`).

`eval_count[thunderbird] = 10` doable rows is this domain's share input to the scaler's **global** cap, `TARGET` in `synth/catalog.py` (`target = round(TARGET × 10 / Σ eval_count)`) — there is no per-domain multiplier. `TARGET` ships as `math.inf`, so `target` is `inf`, Stage B never fires, and all 48 rows emit. Set `TARGET` finite to activate it: `_rescale_for_volume` then downgrades 2-Param FileTasks → 1-Param deterministically until `sum(n_rows) ≤ target`.

## Architecture / design notes

**Eval files/task ratio**: 0.27. Thunderbird eval is dominated by profile state (prefs.js, msgFilterRules.dat, signature files, xulstore.json, folder hierarchy).

**Eval evaluator-func mix**:
- `check_thunderbird_prefs` (prefs.js `user_pref(...)` lines)
- `check_thunderbird_filter` (msgFilterRules.dat record parsing)
- `check_list` (`ls -R` of Mail/Local Folders/ regex)
- `check_json` (xulstore.json folderTree mode + attrs)

**File-as-topic architecture**. All variants reuse the standard profile path `TB_PROFILE_DIR` from `synth/_utils.py` (`/home/user/.thunderbird/t5q2a5hp.default-release/`).

- Each `File` = one structurally distinct TB-profile state shape (one filter rule set, one local-folder pair, one xulstore mode, one signature, one set of `prefs.js` toggles). `src(seed)` returns the pre_config wipe back to the OPPOSITE state.
- Each `FileTask` = one `(file, task)` → one `SynthTemplate`. `gold(file, **gold_args)` returns the oracle steps that mutate the profile to the target state.
- Each `Param` = one concrete `(gold_args, eval_kind, eval_args, instr)` 4-tuple. Cap `≤2 Param` per FileTask, `≤2 FileTask` per File → max 4 rows per File before scaler downgrades.

**Postconfig**: `_TB_PREF_POSTCONFIG` closes Thunderbird before eval reads the profile (mirrors eval.jsonl 08c73485 / 5203d847). Oracle steps first run `_kill_tb_step` (`pkill -f thunderbird`) so the running TB's quit-time flush can't clobber the heredoc edits.

### File ledger (35 Files)

| Range | Count | `setup_class` | Builder | Purpose |
|---|---|---|---|---|
| F_TB_01..08 | 8 | `tb_filter` | `_src_filter_clean` / `_gold_write_filter` | one msgFilterRules.dat record per FileTask |
| F_TB_09..12 | 4 | `tb_local_folder` | `_src_folder_clean` / `_gold_create_folders` | top-level Local Folders pair |
| F_TB_13..16 | 4 | `tb_nested_folder` | `_src_nested_clean` / `_gold_create_nested` | nested hierarchy (`parent.sbd/<child>` ×2) |
| F_TB_17..20 | 4 | `tb_xulstore` | `_src_xulstore_clean` / `_gold_set_xulstore_mode` | folderTree mode (smart / unified / unread / favorite / recent) |
| F_TB_21..24 | 4 | `tb_signature` | `_src_signature_clean` / `_gold_set_signature` | plain 2-line signature |
| F_TB_25..30 | 6 | `tb_pref` | `_src_pref_opposite` / `_gold_set_pref` | prefs.js bool/int toggles (applyIncomingFilters, phishing.detection, compose_html, request_return_receipt, disable_remote_image, use_idle) |
| F_TB_31..34 | 4 | `tb_xulstore_attr` | `_src_xulstore_attr_clean` / `_gold_set_xulstore_attr` | xulstore.json non-mode attrs (folderPaneBox width/collapsed, messengerWindow sizemode, threadPaneBox collapsed) |
| F_TB_35 | 1 | `tb_signature_html` | `_src_signature_clean` / `_gold_set_signature_html` | HTML signature (same `htmlSigText` pref slot, different content axis) |

Each File pins a PRIMARY surface for Param 0 and lists `extras` / `extra_folders` / `extra_parents` covering Param-1 surfaces; the cleaner sweeps the union so no Param trivial-passes regardless of seed. Param 1 always targets a structurally different surface (PD 3b — no paraphrase clones).

**Verified eval landscape** (15 thunderbird tasks, 13 doable):
- Perturb-covered: `08c73485` `f201fbc3` (prefs bool), `3f28fe4f` (signature), `5203d847` `9b7bc335` (filters), `a10b69e1` `9bc3cc16` (local folders / backup), `dfac9ee8` (account removal), `3f49d2cc` (folderTree.mode)
- Not perturbable (dropped): `dd84e895` (run_sqlite3 star folder), `15c3b339` (a11y wizard outlook account), `7b1e1ff9` (a11y profile mgmt page), `d38192b0` (infeasible attach-pdf no-send), `10a730d5` (theme dark-only), `a1af9f1c` (infeasible)

## Implementation references

- `thunderbird.py` — `File` / `FileTask` / `Param` dataclasses; `_build_evaluator` 8-kind dispatcher.
- `synth/_utils.py` — `TB_PROFILE_DIR`, `DOMAIN_PRE_CONFIG → TB_PROFILE_URL`. `synth/thunderbird.py` — `_TB_PREF_POSTCONFIG`, `_kill_tb_step`.
- [AGENTS.md §Scaler architecture](/devs/envs/lite.osworld/synth/AGENTS.md#scaler-architecture-cycle-41--design-5) — per-domain volume only.
- [AGENTS.md §Per-domain Cat 1 / Cat 2 allocation guidance](/devs/envs/lite.osworld/synth/AGENTS.md#per-domain-cat-1--cat-2-allocation-guidance) — Cat 1 30% / Cat 2 70%; perturb-orthogonal gaps are `folder_list` + `nested_folder` + `folderTree_mode`.

## Bridge plan / outstanding work

The quant snapshot is the canonical bridge plan; items it does not cover:

- **`check_accessibility_tree`** — live a11y-tree state not generable from train (structurally exempt per AGENTS.md).
- **Per-message export (right-click → Save As loop)** — exhausts 15-turn budget (F9 agent-cap). Bulk filter / folder ops are the lever.

## Cycle-recurring failures to avoid (thunderbird-specific)

- **F7 (polarity)**: enable/disable verb pairs MUST match the prefs.js boolean target direction. The per-File `target` field locks polarity per-Param.
- **gloda async race**: messages indexed asynchronously; eval may run before gloda finishes. `_TB_PREF_POSTCONFIG` close_window + 1s sleep is the established mitigation. Oracle steps run `_kill_tb_step` first to defeat the quit-time flush race.
- **Profile path hardcoded**: `TB_PROFILE_DIR` from `synth/_utils.py`. If the container's profile name differs, all eval fails — lock via container-image audit.
- **Pref-key verification**: all pref keys must be UI-reachable, not speculative — `mail.server.default.applyIncomingFilters` (real, UI-toggleable) not `enabledMailCheckOnStartup` (speculative). Same for filter action labels (`"MarkFlagged"` matches the C++ enum, `"Mark flagged"` doesn't).
- **XUL ids verified**: TB persists the folder pane under `folderPane`, NOT `folderPaneBox`. Any xulstore element id must be verified against a live TB profile before adding a FileTask.

## Pipeline reference

`pre_config_steps` (from `File.src`) downloads the TB profile tarball via `DOMAIN_PRE_CONFIG → TB_PROFILE_URL` and writes profile prefs.js / msgFilterRules.dat / Local Folders metadata / xulstore.json to the OPPOSITE state. Launching `thunderbird` is handled by the gym setup. Oracle steps (`FileTask.gold`) first kill TB, then write the target state via `python3 << 'PYEOF'` heredoc with `json.dumps` for value escaping. Postconfig closes the TB window. Eval reads profile files directly via the `check_thunderbird_*` / `check_list` / `check_json` evaluators.
