# VS Code — Perturbation Plan

Domain-specific plan for `vs_code`. Common workflow is in [`AGENTS.md`](/devs/envs/lite.osworld/perturb/AGENTS.md).

Code: `lite/gym/envs/lite/osworld/src/gen/train/perturb/vs_code.py`

> **Keep in sync with code.** Op pools, instruction templates, and per-task
> detection logic in this doc and `perturb/vs_code.py` are co-evolved peers —
> any change to one must be reflected in the other.

---

## Instruction Style (vs_code-specific override)

The vs_code eval corpus is **100% polite** (every one of the 23 eval tasks
opens with "Please help me..." / "Please modify..." / "I want to..."). To
avoid a domain-shift in instruction tone — which would otherwise hurt
on-domain transfer for vs_code dev-IDE-config requests — vs_code's perturb
templates **override** the generic [AGENTS.md](/devs/envs/lite.osworld/perturb/AGENTS.md)
"polite 30–40%" target with a vs_code-specific target of **polite 80–95%**.

| metric | eval baseline | vs_code perturb target |
|---|---|---|
| polite (`Please / Could you / I'd like to / Can you / I want to / I need`) | 100% | 80–95% |
| `save the file` literal | 0% | 0% (must stay) |
| avg words per instruction | 16.7 | 15–18 |

### Paraphrase pool design

Each op uses a **5–6 entry** paraphrase pool, with **4–5 entries** opening
with a polite prefix and **1 entry** in plain imperative form (preserving
diversity so the agent doesn't overfit to a single opener pattern):

- `"Please help me <op> ..."`
- `"Could you help me <op> ..."`
- `"I'd like to <op> ..."`
- `"Can you help me <op> ..."`
- `"I want to <op> ..."`
- `"<imperative op>"` (one per pool — bare verb opener)

Templates also weave in **dev-workflow context** so the instruction reads
like a real IDE configuration request from a developer:

- `"... for my dev workflow"`
- `"... in this repo"` / `"... for this project"`
- `"... I'm setting up a new repo"`
- `"... so I can debug my project"`
- `"... my eyes are tired today"` (theme switch)
- `"... before I start coding"`

This matches the eval narrative (developers asking for IDE setup help)
and keeps avg word count in the 15–18 band.

### Save-the-file ban

vs_code does **not** require any "Save the file" suffix:
- `settings.json` and `keybindings.json` are auto-saved by the oracle write.
- File-edit tasks (`text_replace`, `text_indent`) rely on validate.py's
  pkill + oracle `cp gold→source`, so VS Code's in-memory dirty buffer
  cannot undo the change. The eval's postconfig `Ctrl+S` is then a no-op.

---

## Step 0: Understand Eval Tasks

```python
"""Run from repo root: uv run python this_script.py"""
import json
rows = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/eval.jsonl")]
vsc = [r for r in rows if "_vs_code_" in r["task_id"]]
print(f"Total vs_code tasks: {len(vsc)}")
for r in vsc:
    tid = r["task_id"].split("_vs_code_")[-1]
    ev = r["metadata"]["evaluator"]
    func = ev.get("func","?") if isinstance(ev,dict) else str([e.get("func") for e in ev])
    result = ev.get("result",{}) if isinstance(ev,dict) else {}
    rt = result.get("type","") if isinstance(result,dict) else ""
    excl = r["metadata"]["others"].get("exclude_reason","")
    print(f"  {tid[:8]} [{func}/{rt}] excl={excl!r}")
    print(f"    {r['instruction'][:80]}")
```

---

## Task Type Definitions

| task_type | Perturb fn | Description | Value pool |
|---|---|---|---|
| `json_setting` | `perturb_vscode_setting` | settings.json numeric/string values | Per-key pools |
| `extension` | `perturb_vscode_extension` | Marketplace extension install | 7 extensions |
| `keybinding` | `perturb_vscode_keybinding` | Keyboard shortcut key binding | 8 keys |
| `boolean_setting` | `perturb_vscode_bool_setting` | Boolean settings.json values | `true` ↔ `false` × 4 paraphrase variants |
| `workspace_folder` | `perturb_vscode_workspace` | `.code-workspace` folder list | 10 folder names |
| `project_folder` | `perturb_vscode_project_folder` | VS Code open project folder | 8 project names |
| `create_file` | `perturb_vscode_create_file` | Create new file at target dir (`ls \| grep` evaluator) | 8 .py names + 4 generic (.md/.txt/.json) × 3 dirs |
| `text_replace` | `perturb_vscode_text_replace` | Find/replace word in plain-text file | 8 word pairs × synth source |
| `text_indent` | `perturb_vscode_text_indent` | Increase indent on a contiguous line range | 6 (start,end) ranges × synth Python source |
| `save_workspace` | `perturb_vscode_save_workspace` | Save project as `.code-workspace` (P3-4 base coverage for `5e2d93d8`) | 6 names × 4 dirs |
| `files_exclude` | `perturb_vscode_files_exclude` | `files.exclude` glob pattern (P3-4 base coverage for `c6bf789c`) | 7 patterns |
| `lint_severity` | `perturb_vscode_lint_severity` | `python.analysis.diagnosticSeverityOverrides` (P3-4 base coverage for `e2b5e914`) | 6 diagnostics × 4 severities |

### JSON Setting Keys

| setting key | Value pool | Notes |
|---|---|---|
| `files.autoSaveDelay` | 200/300/500/750/1000/1500/2000/3000/5000 ms | Exclude orig |
| `editor.wordWrapColumn` | 40/50/60/72/80/100/120/140 chars | Exclude orig |
| `workbench.colorTheme` | 8 theme names (Visual Studio Dark/Light, Monokai, etc.) | Exclude orig |

### Extension Pool

7 extensions: `ms-python.python`, `njpwerner.autodocstring`, `esbenp.prettier-vscode`, `dbaeumer.vscode-eslint`, `ms-vscode.cpptools`, `golang.go`, `rust-lang.rust-analyzer`

Only tasks whose `result.command` contains `list-extensions` are eligible. Non-marketplace extensions (local VSIX, file paths) are excluded.

### Boolean Settings

| setting key | true instruction | false instruction |
|---|---|---|
| `debug.focusEditorOnBreak` | "Keep cursor on editor at breakpoint" | "Keep cursor on debug console at breakpoint" |
| `workbench.editor.wrapTabs` | "Enable tab wrapping across multiple lines" | "Disable tab wrapping — single scrollable row" |

> **R1 audit fix**: each boolean base has only one alternative value
> (true ↔ false), so the original generator emitted a single variant per
> base — leaving `9439a27b` and `9d425400` as single-variant bases (weak
> training signal). The current generator samples 4 distinct paraphrase
> templates from `_BOOL_SETTING_TEMPLATES` and tags each variant with a
> `paraphrase_idx` knob so task_ids stay unique while the target value
> is held constant. This upgrades the two bases from 1 → 4 variants
> each.

### Save-Workspace Pool (P3-4 5e2d93d8)

- **Workspace names**: `myrepo`, `devspace`, `codebox`, `labwork`, `scratch`, `playground`
- **Parent dirs**: `/home/user/project`, `/home/user/Documents`, `/home/user/Desktop`, `/home/user/repo`
- Eval pair `(project, /home/user/project)` excluded; sample 4 unique `(name, dir)` pairs.
- Oracle: `mkdir -p {dir}` then `echo '{"folders": [{"path": "."}], "settings": {}}' > {dir}/{name}.code-workspace` — matches eval-row oracle pattern (single folder entry + empty settings).
- Detection: same `is_extension_installed` evaluator and `ls dir | grep` shell pattern as `create_file`, but `expected` ends in `.code-workspace`. The `create_file` archetype's `_REGULAR_FILE_EXTS` whitelist explicitly excludes `.code-workspace` so it falls through to this archetype.

### Files-Exclude Pool (P3-4 c6bf789c)

- **Patterns**: `**/__pycache__`, `**/.git`, `**/node_modules`, `**/.DS_Store`, `**/*.pyc`, `**/.idea`, `**/.vscode`
- Eval pattern excluded; sample 4 unique patterns.
- Evaluator does strict dict-eq on `data["files.exclude"]` (the upstream `check_json_settings` checks `data[key] != value`), so the oracle writes a single-key dict `{<pattern>: true}` matching the eval shape exactly.

### Lint-Severity Pool (P3-4 e2b5e914)

- **Diagnostics**: `reportMissingImports`, `reportUndefinedVariable`, `reportUnusedImport`, `reportGeneralTypeIssues`, `reportOptionalMemberAccess`, `reportPrivateUsage`
- **Severities**: `none`, `warning`, `information`, `error`
- Cross-product (6 × 4 = 24 pairs) minus eval `(reportMissingImports, none)`; sample 4 unique pairs.
- Evaluator targets `python.analysis.diagnosticSeverityOverrides` with strict dict-eq, so oracle writes a single-key nested dict matching the eval shape.

### Keybinding Keys

8 keys: `ctrl+j`, `ctrl+k`, `ctrl+m`, `ctrl+shift+j`, `ctrl+shift+e`, `ctrl+shift+k`, `ctrl+shift+m`, `ctrl+alt+n`

7 commands: focusActiveEditorGroup, terminal.focus, toggleSidebarVisibility, togglePanel, formatDocument, quickOpen, commentLine

Negation (remove) tasks: instruction must say `command: "-cmd"` (with leading dash). This is critical — just deleting the line doesn't satisfy `check_json_keybindings` (cycle 27 audit fix).

### Workspace / Project Folder Pools

- **workspace_folder**: 10 names — `data1/data2/src1/src2/logs1/logs2/files1/files2/tests1/docs1`; pre-creates `mkdir -p /home/user/{new_folder}` and a README.txt in `pre_config_steps`
- **project_folder**: 8 names — `project/workspace/mycode/devdir/source/codebase/sandbox/prototype`; copies original project dir to new name in both oracle and `perturb_config_step` (audit pattern: vs_code.perturb_53ad5833_named_folder_not_created)

### Create-File Pools

- **Python file pool** (when eval target ends in `.py`): `hello.py / main.py / utils.py / app.py / script.py / demo.py / lib.py / module.py`
- **Generic pool** (when eval target is non-`.py`): `notes.md` (markdown) / `readme.md` (markdown) / `todo.txt` (text) / `data.json` (JSON)
- **Path pool**: `/home/user/Desktop` / `/home/user/Documents` / `/home/user/Downloads`
- Sampling excludes the eval's `(filename, path)` pair

### Text-Replace Pool

- **Word pairs** (find → replace): `(foo,bar) / (apple,grape) / (dog,cat) / (light,dark) / (yellow,green) / (hello,goodbye) / (fast,slow) / (up,down)`
- Synthesizes a 6-line plain-text source containing the find word in multiple sentences (deterministic per pair). Gold = `source.replace(find, repl)`.
- Source path: `/home/user/Desktop/vscode_replace_text.txt` (preserves base eval target path).

### Text-Indent Pool

- **Source**: 12-line synthetic Python script (function with print statements + arithmetic).
- **Range pool** (start,end): `(2,5) / (3,7) / (4,9) / (5,10) / (2,8) / (6,11)`. Gold = source with one tab prepended on lines `[start,end]`.
- Source path: `/home/user/Desktop/test.py` (preserves base eval target path).

---

## Oracle Mechanics (VS Code)

### JSON Setting

`_build_settings_oracle`: Python heredoc that reads existing settings.json, merges in new setting, writes back.

### Extension

Oracle: `code --install-extension {ext_id} --force`. Also updates `result.command` to grep the new extension ID.

### Keybinding

`_build_keybinding_oracle`: Python heredoc that reads existing keybindings.json, appends new binding dict.

### Boolean Setting

Same as JSON setting via `_build_settings_oracle`.

### Workspace Folder

Oracle: `cp` command to adjust existing oracle commands (replacing old folder path with new). `pre_config_steps` creates `/home/user/{new_folder}/README.txt`.

### Project Folder

Oracle: prepends a `cp -r /home/user/{orig} /home/user/{new}` step. `perturb_config_step` does the same copy at setup time so VS Code can open the folder before oracle runs.

The base eval's oracle uses two 5s sleeps around the post-launch escape press — sufficient when `/home/user/project` is the canonical path with a warm folder cache, but the perturb variants open a freshly cp -r'd directory whose first VS Code load + extension activation can exceed 10s in the 4GB / 1 CPU container. The perturb fn doubles each `sleep` action in the cloned oracle (5s → 10s) and appends an explicit `activate_window` step so postconfig's `Ctrl+Shift+P → OpenProject` always lands on the new VS Code window. This eliminates the V2 oracle smoke flake on `perturb_53ad5833_d37982da` and the other three project-folder variants.

### Create File

`perturb_vscode_create_file`: 2-step shell oracle — `mkdir -p {new_path}` then `touch {new_path}/{new_name}`. `result.command` is rewritten to `["ls", new_path, "|", "grep", new_name]` and `expected.rules.expected` is updated to the new filename. Same-extension constraint: `.py` eval targets sample from Python pool only (preserves "python file" semantics in instruction).

### Text Replace / Text Indent

Both follow the **Impress source+gold** pattern: the perturb fn synthesizes a fresh source file and computes the gold deterministically from it.

- `perturb_config_step` writes both `{source_path}` (overrides the eval's downloaded file) and `/tmp/perturb_expected_{tid}.txt` (the gold) via a single shell heredoc.
- Evaluator switches `expected` from `cloud_file` → `vm_file` pointing to `/tmp/perturb_expected_*`; `result` keeps the original VS Code-edited file path.
- `oracle_actions = [cp gold → source]` runs after validate.py kills VS Code, so the eval's postconfig `ctrl+s` is a no-op (file on disk already matches gold).

### Evaluator Functions

| func | What it checks |
|---|---|
| `check_json_settings` | settings.json or workspace JSON key-value pairs |
| `is_extension_installed` | `code --list-extensions` output contains extension ID |
| `check_json_keybindings` | keybindings.json contains expected binding (including negation) |
| `compare_config` | VS Code open folder or settings.json value |

---

## Perturbation Strategy

**TYPE_1 only**: each perturbable task maps to one setting/operation type. Perturb resamples from the pool excluding the eval's target value.

**Key constraints**:
- `is_extension_installed`: only tasks where `result.command` contains `list-extensions` AND extension ID looks like `publisher.name` (not a file path) are eligible.
- `workspace_folder`: only tasks with ≥2 folder entries where at least one matches `_FOLDER_NAMES` are eligible.
- Workspace instruction templates list ALL folders explicitly (not just the changed one), to prevent the agent from leaving stale folders (cycle 27 audit pattern `6ed0a554`).

---

## Per-task Plan

> **Keep in sync with code.** Detection logic (which perturb fn handles which evaluator func/result combination) must match `perturb/vs_code.py`.

### Perturbable (18 tasks)

| tid | eval func / result | sub-type | eval target | Rows |
|---|---|---|---|---|
| `276cc624` | `check_json_settings` / settings.json | `wordWrapColumn` | 50 chars | 4 |
| `70745df8` | `check_json_settings` / settings.json | `autoSaveDelay` | 1000ms | 4 |
| `4e60007a` | `is_extension_installed` | extension | autoDocstring | 4 |
| `eabc805a` | `is_extension_installed` | extension | Python | 4 |
| `930fdb3b` | `check_json_keybindings` | keybinding (create) | ctrl+j → focusEditorGroup | 4 |
| `ea98c5d7` | `check_json_keybindings` | keybinding (remove) | -ctrl+f → list.find | 4 |
| `9439a27b` | `check_json_settings` / settings.json | `debug.focusEditorOnBreak` | true (R1: 1 → 4 paraphrase variants) | 4 |
| `9d425400` | `check_json_settings` / settings.json | `wrapTabs` | true (R1: 1 → 4 paraphrase variants) | 4 |
| `6ed0a554` | `check_json_settings` / .code-workspace | workspace folder | [data1, data2] → swap one | 4 |
| `53ad5833` | `compare_config` / vscode_config | project folder | "project" | 4 |
| `982d12a5` | `compare_config` / vm_file | colorTheme in settings.json | 4 |
| `0512bb38` | `is_extension_installed` | extension (local VSIX → marketplace) | 4 |
| `57242fad` | `is_extension_installed` (file presence via `ls \| grep`) | create_file (`test.py` → other Python file in another dir) | 4 |
| `0ed39f63` | `compare_text_file` | text_replace (synth source + gold via `str.replace`) | 4 |
| `ec71221e` | `compare_text_file` | text_indent (synth source + gold via line-range tab prefix) | 4 |
| `5e2d93d8` | `is_extension_installed` (file presence via `ls \| grep`) | save_workspace (P3-4: project → workspace `.code-workspace` save-as) | 4 |
| `c6bf789c` | `check_json_settings` / settings.json | files_exclude (P3-4: `files.exclude` glob pattern) | 4 |
| `e2b5e914` | `check_json_settings` / settings.json | lint_severity (P3-4: Python diagnostic severity override) | 4 |

> **Note `0512bb38`**: eval asks to install a local VSIX file. The extension perturb function replaces this with marketplace extension installs (changing instruction to "install Go extension" etc.). The `undefined_publisher.test` ID passes the marketplace filter (`"." not in ID` check doesn't catch `.test`). Generated rows are valid training examples for marketplace extension install.

> **Note `57242fad`**: upstream eval reuses `is_extension_installed` evaluator name for a file-presence check (`ls /home/user/Desktop | grep test.py`). `perturb_vscode_create_file` detects this pattern (no `list-extensions` in command, expected name has an extension dot) and resamples filename + target dir.

### Not Perturbable (5 infeasible)

| tid | Reason |
|---|---|
| `7aeae0e2` | infeasible |
| `7c4cc09e` | infeasible |
| `847a96b6` | infeasible |
| `971cbb5b` | infeasible |
| `dcbe20e8` | infeasible |

> **P3-4 update**: the previously-unhandled `5e2d93d8` / `c6bf789c` /
> `e2b5e914` bases are now covered by dedicated archetypes
> (`save_workspace` / `files_exclude` / `lint_severity`) — see the
> "Per-task Plan" table above. This closes the last 3 feasible base
> coverage gaps for vs_code (15 → 18 covered / 18 feasible).

---

## V4a Coverage Check

```python
"""V4a coverage check — vs_code.
Run from repo root: uv run python this_script.py
"""
import json, re
from collections import Counter

REQUIRED_TASK_TYPES = {
    "json_setting", "extension", "keybinding_create", "keybinding_remove",
    "boolean_setting", "workspace_folder", "project_folder", "create_file",
    "text_replace", "text_indent",
    # P3-4 base coverage additions:
    "save_workspace", "files_exclude", "lint_severity",
}

all_eval = {r["task_id"]: r for r in (json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/eval.jsonl"))}
all_perturb = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
vsc_perturb = [r for r in all_perturb if "_vs_code_" in r["task_id"]]

def _oracle(r):
    return "\n".join(
        a.get("parameters", {}).get("command", "")
        for a in (r["metadata"].get("others") or {}).get("oracle_actions", [])
        if isinstance(a.get("parameters", {}).get("command"), str)
    )

def _task_type(r):
    oracle = _oracle(r)
    new_ev = r["metadata"]["evaluator"]
    func = new_ev.get("func","") if isinstance(new_ev,dict) else ""
    result = new_ev.get("result",{}) if isinstance(new_ev,dict) else {}
    result_path = result.get("path","") if isinstance(result,dict) else ""
    result_type = result.get("type","") if isinstance(result,dict) else ""

    if func == "is_extension_installed":
        # Distinguish save_workspace (.code-workspace) / create_file (ls|grep)
        # / extension (list-extensions).
        result_cmd = " ".join(str(c) for c in result.get("command", []))
        if "ls" in result_cmd and "grep" in result_cmd and "list-extensions" not in result_cmd:
            exp = new_ev.get("expected", {})
            rules = exp.get("rules", {}) if isinstance(exp, dict) else {}
            exp_name = rules.get("expected", "")
            if isinstance(exp_name, str) and exp_name.endswith(".code-workspace"):
                return "save_workspace"
            return "create_file"
        return "extension"
    if func == "compare_text_file":
        # text_replace: vscode_replace_text.txt; text_indent: test.py
        if "vscode_replace_text" in result_path: return "text_replace"
        if result_path.endswith("/test.py"): return "text_indent"
        return "_unknown"
    if func == "check_json_keybindings":
        # Remove entries have command starting with "-" (e.g. "-list.find").
        # Don't use '"-' in oracle: oracle uses Python single-quote strings, not JSON.
        exp = new_ev.get("expected", {})
        if isinstance(exp, list): exp = exp[0] if exp else {}
        rules = exp.get("rules", {}) if isinstance(exp, dict) else {}
        cmd = rules.get("expected", {}).get("command", "") if isinstance(rules.get("expected", {}), dict) else ""
        return "keybinding_remove" if cmd.startswith("-") else "keybinding_create"
    if func == "compare_config" and result_type == "vscode_config": return "project_folder"
    if func in ("check_json_settings", "compare_config") and "settings.json" in result_path:
        exp = new_ev.get("expected",{})
        if isinstance(exp,list): exp = exp[0] if exp else {}
        rules = exp.get("rules",{}) if isinstance(exp,dict) else {}
        exp_settings = rules.get("expected", {})
        # compare_config can ship expected as a JSON-encoded string.
        if isinstance(exp_settings, str):
            try: exp_settings = json.loads(exp_settings.strip())
            except (json.JSONDecodeError, ValueError): exp_settings = {}
        if isinstance(exp_settings, dict):
            if "folders" in exp_settings: return "workspace_folder"
            if "files.exclude" in exp_settings: return "files_exclude"
            if "python.analysis.diagnosticSeverityOverrides" in exp_settings:
                return "lint_severity"
            for k in exp_settings:
                if isinstance(exp_settings[k], bool): return "boolean_setting"
        return "json_setting"
    if func == "check_json_settings" and "settings.json" not in result_path:
        return "workspace_folder"
    return "_unknown"

type_counts = Counter(_task_type(r) for r in vsc_perturb)
missing = REQUIRED_TASK_TYPES - set(type_counts.keys())
print(f"Total vs_code perturb rows: {len(vsc_perturb)}")
print(f"[{'FAIL' if missing else 'OK  '}] missing task types: {missing or 'none'}")
total = len(vsc_perturb) or 1
for t in sorted(REQUIRED_TASK_TYPES | set(type_counts.keys())):
    cnt = type_counts.get(t, 0)
    flag = " <-- MISSING" if cnt == 0 else ""
    print(f"  {t:<30}  {cnt:>4}  {cnt/total:>6.1%}{flag}")
```

Targets:
- All 13 task types present
- Total rows: **69 current** (older ~72-row target was 4 × 18 perturbable bases; later drops/refactors changed the generated total) — was 54 before P3-4 + R1.

---

## V4b Perturb-Eval Match Verification

### Part A: Instruction Clarity

```python
import json, random

rows = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
vsc = [r for r in rows if "_vs_code_" in r["task_id"]]

def _oracle(r):
    return "\n".join(
        a.get("parameters", {}).get("command", "")
        for a in (r["metadata"].get("others") or {}).get("oracle_actions", [])
        if isinstance(a.get("parameters", {}).get("command"), str)
    )

rng = random.Random(0)
for r in rng.sample(vsc, min(15, len(vsc))):
    print(f"[{r['task_id'].split('_vs_code_')[-1].split('_')[0]}]")
    print(f"  INSTR : {r['instruction']}")
    print(f"  ORACLE: {_oracle(r)[:200].strip()}")
    print()
```

What to verify per row:

| check | What to verify |
|---|---|
| keybinding remove | Instruction says `command: "-cmd"` (with leading dash), not just "delete the line" |
| keybinding create | Key string in instruction matches key in oracle binding dict |
| extension | Extension name in instruction matches `ext_id` in oracle install command |
| workspace | Instruction lists ALL folder names explicitly (not just the new one) |
| project | Project folder name in instruction matches oracle `cp` command path |
| json setting | Setting value in instruction matches value in oracle heredoc |

### Part B: Feasibility

```python
import json, re

all_perturb = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
vsc_perturb = [r for r in all_perturb if "_vs_code_" in r["task_id"]]

def _oracle(r):
    return "\n".join(
        a.get("parameters", {}).get("command", "")
        for a in (r["metadata"].get("others") or {}).get("oracle_actions", [])
        if isinstance(a.get("parameters", {}).get("command"), str)
    )

issues = []
for r in vsc_perturb:
    oracle = _oracle(r)
    new_ev = r["metadata"]["evaluator"]
    func = new_ev.get("func","") if isinstance(new_ev,dict) else ""

    # extension: must install via marketplace ID
    if func == "is_extension_installed":
        if not re.search(r'code --install-extension [a-z][\w-]+\.[a-z]', oracle):
            issues.append((r["task_id"], "extension oracle doesn't use publisher.name format"))

    # keybinding remove: must have '-' prefix in command string
    if func == "check_json_keybindings":
        if '"command": "-' not in oracle and "'command': '-" not in oracle:
            # could be create type, which is OK
            pass

print(f"[{'FAIL' if issues else 'OK  '}] feasibility: {len(issues)} issues")
for tid, reason in issues[:10]:
    print(f"  {tid}: {reason}")
```

---

## V4c Eval Leakage Check

```python
import json

all_eval = {r["task_id"]: r for r in (json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/eval.jsonl"))}
all_perturb = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
vsc_perturb = [r for r in all_perturb if "_vs_code_" in r["task_id"]]

def _oracle(r):
    return "\n".join(
        a.get("parameters", {}).get("command", "")
        for a in (r["metadata"].get("others") or {}).get("oracle_actions", [])
        if isinstance(a.get("parameters", {}).get("command"), str)
    )

leakage = []
for r in vsc_perturb:
    src = r["metadata"]["others"].get("source", "")
    base_tid = src.replace("perturb:", "")
    eval_row = all_eval.get(base_tid)
    if not eval_row:
        continue
    if r["instruction"].strip() == eval_row["instruction"].strip():
        leakage.append((r["task_id"], "instruction identical to eval"))

print(f"[{'FAIL' if leakage else 'OK  '}] eval leakage: {len(leakage)} violations")
for tid, reason in leakage[:10]:
    print(f"  {tid}: {reason}")
```

---

## V4d Inter-Variant Uniqueness

```python
import json, re
from collections import defaultdict

all_perturb = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
vsc_perturb = [r for r in all_perturb if "_vs_code_" in r["task_id"]]

def _oracle(r):
    return "\n".join(
        a.get("parameters", {}).get("command", "")
        for a in (r["metadata"].get("others") or {}).get("oracle_actions", [])
        if isinstance(a.get("parameters", {}).get("command"), str)
    )

by_source = defaultdict(list)
for r in vsc_perturb:
    src = re.sub(r"_[0-9a-f]{8}$", "", r["task_id"])
    by_source[src].append(r)

instr_dups = []
oracle_dups = []
for src, rows in by_source.items():
    if len(set(r["instruction"] for r in rows)) < len(rows):
        instr_dups.append(src)
    if len(set(_oracle(r) for r in rows)) < len(rows):
        oracle_dups.append(src)

print(f"[{'FAIL' if instr_dups  else 'OK  '}] duplicate instructions: {len(instr_dups)} sources")
print(f"[{'FAIL' if oracle_dups else 'OK  '}] duplicate oracle code:  {len(oracle_dups)} sources")
```

---

## Expected Output

- 18 perturbable tasks → **69 current rows** (after R1 boolean expansion + P3-4 base coverage, with later drops/refactors)
- 13 task types covered
- Key audit fixes encoded:
  - keybinding remove: `-command` negation in oracle and instruction
  - workspace: full folder list in instruction (not just new folder)
  - project folder: folder pre-created at setup time via `perturb_config_step`
  - extension: `0512bb38` local VSIX passes marketplace filter (`.test` not in excluded extensions) → generates marketplace extension install variants
  - create_file: detects upstream's mis-labeled `is_extension_installed` evaluator (file presence via `ls | grep`); resamples filename + dir excluding eval pair; explicitly excludes `.code-workspace` so save-workspace tasks fall through to the dedicated archetype
  - text_replace / text_indent: synth source + deterministic gold via `perturb_config_step` heredoc; oracle `cp` runs after VS Code is killed so postconfig `ctrl+s` is a safe no-op
  - **R1 (boolean_setting)**: 1 alternative value × 4 paraphrase templates × `paraphrase_idx` knob → 4 unique-task_id variants per base, replacing the previous single-variant emission for `9439a27b` / `9d425400`
  - **P3-4 (save_workspace / files_exclude / lint_severity)**: dedicated archetypes for the 3 previously-uncovered feasible bases (`5e2d93d8` / `c6bf789c` / `e2b5e914`); each emits 4 variants and shares the upstream evaluator (`is_extension_installed` for save_workspace; `check_json_settings` for the other two) but with task-specific instruction context and oracle shape
- V2 pass-rate target: **100%**
