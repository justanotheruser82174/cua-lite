# VS Code — Synth Plan

> Keep in sync with code. Implementation: [`vs_code.py`](/lite/gym/envs/lite/osworld/src/gen/train/synth/vs_code.py).
> Common workflow: [`AGENTS.md`](/devs/envs/lite.osworld/synth/AGENTS.md). Cross-reference: [`perturb/vs_code.md`](/devs/envs/lite.osworld/perturb/vs_code.md).

## Current quant-gap snapshot (`measure_gap.py` v2)

Run `uv run python devs/envs/lite.osworld/measure_gap.py --domain vs_code` for live numbers. Synth N=84, eval N=18 (5 infeasibility filtered). **vs_code is the structurally healthiest domain** — op-family proportions track upstream within ±5pp.

| Dim | Synth | Eval | Δpp | Status | Bridge |
|---|---:|---:|---:|:-:|---|
| `instruction_leak.key_leak` | 38.1% | 0% | +38 | 🔴 | Strip backticked JSON keys (`editor.wordWrap`, `files.autoSave`, etc.) from 32 instructions. Convert to abstract user voice |
| `skill_class.settings_json` | 39.3% | 38.9% | +0 | ✓ | Aligned — but **obscure-key tail invisible**. Synth clusters on ~10 popular keys (editor.fontSize, wordWrap, formatOnSave); eval probes niche keys (workbench.editor.wrapTabs, debug.focusEditorOnBreak, python.analysis.diagnosticSeverityOverrides, files.exclude glob). Add 3-5 obscure-key templates |
| `skill_class.ext_marketplace` | 19.0% | 16.7% | +2 | ✓ | Aligned (sub-bucket of v2-split `is_extension_installed`) |
| `skill_class.file_exists_grep` (sub-split) | 0% | partial | -X | ❌ | Eval has 2 rows where `is_extension_installed` actually runs `ls ~/Desktop \| grep test.py` (file-exists check, not extension check). Synth uses `compare_text_file` for file-create — add file-exists-grep template variant |
| `skill_class.file_line_edit` (sub-split) | low | 2/18 | — | ⚠️ | Synth `compare_text_file` rows are dominated by file-create-from-template; eval has 2 line-range edit rows (`compare_text_file 0ed39f63` text→test, `ec71221e` indent line 2-10). Add 3 line-edit templates |

**v2 fix**: `vs_code_skill_class` now sub-classifies `is_extension_installed` into `{ext_marketplace, ext_vsix_local, file_exists_grep, extensions_other}` via `result.cmd` inspection — surfaces the file-exists-grep coverage hole that v1 had hidden under `extensions`.

**v2 fix**: `compare_text_file` / `compare_python_pure_text` / `diff_text_file` sub-classified into `{file_create_template, file_line_edit, file_edit_other}` via instruction-verb regex — surfaces synth's file-create over-rep vs eval's line-edit pattern.

## Current shape

**63 spec entries → 75 historical active templates → 92 current jsonl rows** (every template is `n_rows=1`; one extra factory has no spec list — the local-VSIX install).

| Factory | Spec list | Count | `eval_class` | Eval evaluator-func |
|---|---|---|---|---|
| `_make_settings_template` | `_SETTINGS_SPECS` | 19 | `json_setting` | `check_json_settings` (single-key bool/int/string set) |
| `_make_settings_multi_template` | `_SETTINGS_MULTI_SPECS` | 7 | `json_setting` | `check_json_settings` (multi-key expected) |
| `_make_keybinding_template` | `_KEYBINDING_SPECS` | 6 | `json_setting` | `check_json_keybindings` |
| `_make_file_create_template` | `_FILE_CREATE_SPECS` | 6 | `text_edit` | `compare_text_file` (synthetic gold bytes) |
| `_make_file_edit_template` | `_FILE_EDIT_SPECS` | 2 | `text_edit` | `compare_text_file` (synthetic source + transform) |
| `_make_asset_file_edit_template` | `_ASSET_FILE_EDIT_SPECS` | 14 | `text_edit` | `compare_text_file` (real OSS asset + container-side transform) |
| `_make_gitignore_create_template` | `_GITIGNORE_CREATE_SPECS` | 6 | `text_edit` | `compare_text_file` (gitignore asset bytes as gold) |
| `_make_real_settings_template` | `_REAL_SETTINGS_SPECS` | 3 | `json_setting` | `check_json_settings` (real cpython JSONC as initial state) |
| `_make_extension_install_template` | `_EXTENSION_SPECS` | 5 | `extension` | `is_extension_installed` (marketplace) |
| `_make_vsix_install_template` | — (singleton) | 1 | `extension` | `is_extension_installed` (local VSIX, network-independent) |
| `_make_workspace_template` | `_WORKSPACE_SPECS` | 6 | `compare_config` | `is_extension_installed` (save-as `ls\|grep -F`) OR `check_json_settings` (`.code-workspace` blocks) |
| **Historical table total** | — | **75** | 4 buckets | — |

Current generated total: **92 rows** in `train.synth.jsonl`; the live-module snapshot above was `json_setting` 35, `text_edit` 28, `extension` 6, `compare_config` 6.

## Architecture / design notes

**Spec-dict factories (NOT FileTask/Param).** vs_code is the only domain that does not use the cap-2×2 `File` / `FileTask` / `Param` shape. Each `_make_<group>_template(spec)` factory consumes a row-spec dict from the matching `_<GROUP>_SPECS` constant and emits a `SynthTemplate`. The scaler therefore cannot auto-trim vs_code via 2-Param→1-Param downgrade — manual comment-out is the only lever; the scaler emits `OVER need_manual_comment` when above target.

**Eval files/task ratio**: 0.52. VS Code eval mixes file edits, user-scope settings/keybindings, extension installation, and workspace state.

**Eval evaluator-func mix**:
- `check_json_settings` (`~/.config/Code/User/settings.json`)
- `check_json_keybindings` (`~/.config/Code/User/keybindings.json`)
- `is_extension_installed` (`code --list-extensions | grep <id>`, and `ls | grep -F <basename>.code-workspace` for save-as)
- `compare_text_file` / `compare_config` (file edits + `.code-workspace` block subset)

**Key state-axis variation**: workspace presence (no folder / has-folder / multi-folder workspace); initial settings.json state (empty / minimal / pre-populated with real cpython JSONC); initial keybindings.json state; file-edit target type and language (`.py` / `.js` / `.ts` / `.go` / `.rs` / `.java` / `.json` / `.md` / `.txt`); extension state (none / marketplace install / local VSIX).

**Staged code assets (real OSS bundle)** — 21 files across 8 categories under `assets/synth/code/`, every byte cited in `_ASSET_FILE_EDIT_SPECS`, `_GITIGNORE_CREATE_SPECS`, or `_REAL_SETTINGS_SPECS`:

| Category | Files |
|---|---|
| `python/` | `flask-app.py` (BSD-3 Flask), `requests-api.py` (Apache-2.0), `django-utils.py` (BSD-3), `numpy-init.py` (BSD-3), `pytest-init.py` (MIT) |
| `javascript/` | `express-index.js` (MIT) |
| `typescript/` | `vscode-uri.ts`, `typescript-types.ts` (MIT / Apache-2.0) |
| `go/` | `gin-handler.go`, `chi-mux.go` (MIT) |
| `rust/` | `serde-lib.rs`, `tokio-lib.rs` (MIT) |
| `java/` | `mockito-bdd.java` (MIT) |
| `gitignore/` | `Python.gitignore`, `Node.gitignore`, `Java.gitignore`, `Go.gitignore`, `Rust.gitignore`, `C++.gitignore`, `Swift.gitignore` (CC0 github/gitignore) |
| `vscode-config/` | `python-settings.json` (MIT vscode-python `.vscode/settings.json`, JSONC) |

Pre-config calls `_stage_asset(host_push)` to push the host file into the container at the agent path. For find-replace / comment-toggle / indent rows, the matching container-side python heredoc (`_container_transform_step`) re-applies the spec transform against the just-staged bytes and writes gold to `/tmp/expected_<id>.txt`; eval-side `compare_text_file` byte-compares.

### Group-level walk-through (read alongside the spec lists)

- **`_SETTINGS_SPECS`** (19): bool / int / string single-key toggles. Pre-config writes the OPPOSITE state; oracle kills `code` then merges target key via `python3 << PYEOF` (JSONC-aware). Mirrors `276cc624` (font-size), `982d12a5` (color-theme).
- **`_SETTINGS_MULTI_SPECS`** (7): two-or-more key sets — autosave, exclude patterns, paired minimap / terminal-font keys.
- **`_KEYBINDING_SPECS`** (6): user-scope keybindings.json append. Mirrors `930fdb3b`.
- **`_FILE_CREATE_SPECS`** (6) & **`_FILE_EDIT_SPECS`** (2): synthetic-content file ops (`test.py`, `main.py`, README, requirements.txt, .env, Dockerfile; replace-all + indent-by-tab). Mirrors `57242fad`, `0ed39f63`, `ec71221e`.
- **`_ASSET_FILE_EDIT_SPECS`** (14): real-asset find-replace, comment-toggle, indent / tab→spaces across Python / JS / TS / Go / Rust.
- **`_GITIGNORE_CREATE_SPECS`** (6): "create the standard GitHub `<lang>` `.gitignore` template at `<path>`". The github/gitignore asset is the gold itself; instructions name the canonical template (3 phrasing variants per row) WITHOUT leaking `/tmp/expected_<id>.txt`.
- **`_REAL_SETTINGS_SPECS`** (3): seed user settings.json with the upstream cpython JSONC `python-settings.json`; task adds one new key. Validates that the agent preserves surrounding real-config keys.
- **`_EXTENSION_SPECS`** (5) + **VSIX singleton** (1): marketplace + local-VSIX. Pre-config purges `~/.vscode/extensions/<ext-id>-*` for trivial-pass guard. The VSIX row builds `undefined_publisher.test` on disk and is network-independent.
- **`_WORKSPACE_SPECS`** (6, `eval_class=compare_config`): `.code-workspace` save-as, multi-folder add, with-settings / with-extensions / with-launch block subsets. Pre_config never pre-creates the gold; oracle materialises via heredoc.

**Cat 2 templates (eval-task-grounded)**:

| Factory | Eval task_id | Evaluator |
|---|---|---|
| `_make_workspace_template` (save-as) | 5e2d93d8 | `is_extension_installed` (`ls <parent>/ \| grep -F <basename>.code-workspace`) — `-F` flag for literal `.` |
| `_make_workspace_template` (folders / settings / extensions / launch) | 6ed0a554 (+ Cartesian) | `check_json_settings` on `.code-workspace` top-level blocks |
| `_make_file_edit_template` (replace-all) | 0ed39f63 | `compare_text_file` |
| `_make_file_edit_template` (indent) | ec71221e | `compare_text_file` (leading whitespace) |
| `_make_file_create_template` | 57242fad | `compare_text_file` |
| `_make_real_settings_template` | 70745df8 | `check_json_settings` |
| `_make_extension_install_template` | eabc805a, 4e60007a | `is_extension_installed` (marketplace) |
| `_make_vsix_install_template` | 0512bb38 | `is_extension_installed` (local VSIX, no network) |

Rejected templates: `synth_vscode_command_palette_macro`, marketplace-formatter rows, multi-cursor / Outline / Search-in-files (UI-only).

## Implementation references

- `vs_code.py` — 11 spec-dict factories listed above; `_stage_asset(host_push)` for OSS bundle; `_container_transform_step` for asset-edit gold.
- `_vscode_welcome_dismiss_postconfig` (welcome modal pre-dismiss) — required by every new synth task.
- `_vscode_extension_postconfig` (extension activation race) — used by extension rows.
- [AGENTS.md §Scaler architecture](/devs/envs/lite.osworld/synth/AGENTS.md#scaler-architecture-cycle-41--design-5) — vs_code is the lone non-FileTask domain and is trimmed by manual comment-out when scaler reports `OVER need_manual_comment`.
- [AGENTS.md §Per-domain Cat 1 / Cat 2 allocation guidance](/devs/envs/lite.osworld/synth/AGENTS.md#per-domain-cat-1--cat-2-allocation-guidance) — Cat 1 ≈ 30% / Cat 2 ≈ 70%.

## Bridge plan / outstanding work

The quant snapshot is the canonical bridge plan; items it does not cover:

- **Multi-cursor / Outline / global-search rows** (eval rows #33, #35, #84, #93-#95, #106, #107) — UI-only, no shell oracle. DROPPED, not deferred.
- **Marketplace-extension-dependent format-document** — need Prettier / Black / language-server formatter installed in the VM. Deferred until a pre-installed `.vsix` bundle lands in the GNOME baseline.
- **Large-asset byte-equality find-replace** (`vs_code_fr_numpy_array_to_ndarray` 927 lines, `_flask_app_to_application` 1625 lines, `_gin_handler_engine_to_app` 833 lines) — `compare_text_file` byte-equality fragile against autocomplete / trim / autoformat. Re-enable only after a content-aware evaluator (assert exactly-N-replacements + before/after substrings) lands.

## Cycle-recurring failures to avoid (vs_code-specific)

- **F4 (wording-scope)**: the hardest-hit failure mode for vs_code. Audit every instruction string for "this repo" / "this project" / "my workspace" / "in this folder" when the eval is user-scope.
- **Welcome modal at turn_00**: every new synth task must include `_vscode_welcome_dismiss_postconfig` in pre/postconfig.
- **Extension install timing**: `is_extension_installed` may race extension activation. Use `_vscode_extension_postconfig` with sleep.
- **Byte-equality fragility on long assets**: `compare_text_file` against >500-line gold is fragile (incidental autocomplete / trailing-whitespace-trim / autoformat drop diverges bytes). Prefer short or transform-tight assets until a content-aware evaluator lands.
- **Save-as regex literal dots**: `grep -F` (fixed-string) so a sibling like `project_code-workspace` doesn't FALSE-PASS the `.code-workspace` save-as evaluator.
- **Gold-path leak**: gitignore instructions must NOT cite `/tmp/expected_<id>.txt` — name the canonical github/gitignore template instead.

## Pipeline reference

`pre_config_steps` writes the initial settings.json / keybindings.json / source files, pushes any OSS asset via `_stage_asset(host_push)`, launches `code <folder>` (always with explicit folder unless explicitly testing welcome state); agent acts via UI; oracle kills `code` then merges JSON / overwrites file via python heredoc; eval reads JSON / files / extension list directly.

The authoritative row inventory lives **in the `_<GROUP>_SPECS` lists in `vs_code.py`** — one spec dict per row, keyed by `template_id`.
