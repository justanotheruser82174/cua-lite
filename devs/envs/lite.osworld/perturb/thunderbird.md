# Thunderbird — Perturbation Plan

Domain-specific plan for `thunderbird`. Common workflow is in [`AGENTS.md`](/devs/envs/lite.osworld/perturb/AGENTS.md).

Code: `lite/gym/envs/lite/osworld/src/gen/train/perturb/thunderbird.py`

> **Keep in sync with code.** This `.md` and the matching `.py` are co-evolved peers. Any change to op pools, instruction templates, polite/word-count targets, oracle mechanics, or per-task rows must be reflected in both files in the same change.

---

## Step 0: Understand Eval Tasks

```python
"""Run from repo root: uv run python this_script.py"""
import json
rows = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/eval.jsonl")]
tb = [r for r in rows if "_thunderbird_" in r["task_id"]]
print(f"Total thunderbird tasks: {len(tb)}")
for r in tb:
    tid = r["task_id"].split("_thunderbird_")[-1]
    ev = r["metadata"]["evaluator"]
    func = ev.get("func","?") if isinstance(ev,dict) else str([e.get("func") for e in ev])
    excl = r["metadata"]["others"].get("exclude_reason","")
    exp = ev.get("expected",{}) if isinstance(ev,dict) else {}
    if isinstance(exp,list): exp = exp[0] if exp else {}
    rules = exp.get("rules",{}) if isinstance(exp,dict) else {}
    print(f"  {tid[:8]} [{func}] excl={excl!r}")
    print(f"    {r['instruction'][:80]}")
```

---

## Task Type Definitions

| task_type | Perturb fn | Description | Value pool |
|---|---|---|---|
| `pref_setting` | `perturb_thunderbird_pref` | about:config preference tasks | See sub-types below |
| `filter_setting` | `perturb_thunderbird_filter` | Email filter tasks | Folder/keyword/email pools |
| `local_folder` | `perturb_thunderbird_local_folders` | Local folder creation | 8 folder name pairs |
| `backup_path` | `perturb_thunderbird_backup_path` | Email backup target directory (check_list/cache_file) | 6 path names |
| `star_folder` | `perturb_thunderbird_star_folder` | Folder to star all emails (run_sqlite3) | Currently emits 0 rows; non-eval folders race IMAP/gloda indexing |
| `unified_inbox` | `perturb_thunderbird_unified_inbox` | Folder-pane mode in xulstore.json (`check_json`) | 3 modes (favorite/unread/recent) |
| `remove_account` | `perturb_thunderbird_remove_account` | Outlook account removal (`check_csv`) | 3 paraphrase variants of the same removal |

### Pref Setting Sub-types

| sub-type | Detection | Value pool | Notes |
|---|---|---|---|
| theme | `extensions.activeThemeID` in expect_rules | Restricted to `"dark"` only | Default and "light" themes vacuously match the prefs.js seed; only "dark" requires actual agent action (cycle 27 audit) |
| signature | `mail.identity.id1.htmlSigText` in expect_rules | 8 names × 8 affiliations (separate lines) | Instruction must explicitly say "first line / second line" — inline slash separators cause single-line agent output |
| boolean | `mail.server.default.applyIncomingFilters`, `mail.imap.use_status_for_biff`, `mail.identity.id1.auto_quote` | `true` ↔ `false` | Perturb removes secondary prefs from evaluator expect to avoid unnamed-pref violation |

### Filter Setting Sub-types

| sub-type | Detection | Value pool |
|---|---|---|
| move-to-folder | `action == "Move to folder"` | 8 folders × 8 keywords |
| forward | `action == "Forward"` | 7 email addresses |

### Local Folder Pairs

8 pairs: WORK/PERSONAL, OFFICE/HOME, BUSINESS/ACADEMIC, CLIENTS/INTERNAL, PROJECTS/ADMIN, REPORTS/ARCHIVES, DRAFTS/SENT, FINANCE/TRAVEL

### Backup Path Pool

6 candidates (excluding eval value `emails.bak`): `inbox_backup`, `mail_archive`, `email_export`, `inbox.bak`, `mail_backup`, `eml_backup`

### Star Folder Pool

Retired from emission. The three non-eval candidates (INBOX id=1, Drafts id=7,
Sent id=10) are IMAP-backed in the standard profile and race async gloda
indexing, so the generator keeps the archetype documented but emits 0 rows for
now. Folder IDs come from the `global-messages-db.sqlite` `folderLocations`
table in the standard profile.

### Unified-inbox Folder-mode Pool (P3-4)

3 candidates (`favorite`, `unread`, `recent`) excluding eval value `smart`. The mode `all` is also a valid Thunderbird folder-pane value but is excluded because the seeded `xulstore.json` already has `[chrome://...messenger.xhtml][folderTree][mode] = "all"`; an instruction asking for "all folders" mode would be vacuously satisfied (trivial-pass) without any agent action. Each kept mode requires the agent to actively flip away from the seed.

### Remove-account Pool (P3-4)

The standard profile contains a single Outlook IMAP account (`anonym-x2024@outlook.com`); we cannot perturb the *target email* without first seeding additional accounts (out of scope for this archetype). The archetype therefore generates **3 paraphrase variants of the same removal** — distinct instruction wording with a stable (oracle, evaluator) pair — to expand the training signal for the `check_csv` evaluator type. `knob_assignment` includes a `paraphrase` index so each row's task_id hash is unique.

---

## Oracle Mechanics (Thunderbird)

### Pref Setting

All oracle actions kill Thunderbird first (`pkill -f thunderbird; sleep 3`) so prefs.js isn't overwritten on exit, then write the new pref values via Python heredoc.

- **theme**: uses `perturb_config_step` (NOT `pre_config_steps`) to seed the opposite theme after the base config extracts the tarball. Seeds compact-light theme so agent must flip to dark.
- **signature**: replaces name/affiliation strings in the oracle's sed/python command.
- **boolean**: builds new oracle from scratch (pkill → Python heredoc that filters old key and appends new `user_pref`). Strips secondary prefs from the copied evaluator to prevent unnamed-pref violations.

### Filter Setting

Replaces folder name + keyword strings (move filter) or email address (forward filter) in both evaluator and oracle commands.

### Local Folder

Replaces folder name patterns (e.g. `\bCOMPANY\.msf\b`) in both the evaluator regex list and the oracle mkdir/touch commands. Instructions explicitly name the `Local Folders` account (not IMAP account) to prevent wrong sidebar click (cycle 27 audit fix).

### Backup Path

Replaces `/home/user/<orig_path>` with `/home/user/<new_path>` in the evaluator `postconfig` ls command (and its `stdout` key) and in the oracle shell commands. Eval `result.path` is also updated to `<new_path>.ls`.

### Star Folder

Updates the `folderID = <id>` literal in the evaluator's SQL query and in the oracle python3 command. Folder IDs are stable constants from the profile's sqlite DB.

### Unified Inbox (P3-4 real-gap, `check_json`)

Targets eval `3f49d2cc`. The evaluator reads `xulstore.json` and asserts `[messenger.xhtml][folderTree][mode]` matches `\bsmart\b` via `check_json`. Perturb rewrites both:
- The evaluator's `ref` regex from `\bsmart\b` to `\b<new_mode>\b` (regex method `re` preserved).
- The oracle python3 heredoc's literal `"smart"` assignment to `"<new_mode>"`.

Rows produced: 3 per eval row (matches pool size).

### Remove Account (P3-4 real-gap, `check_csv`)

Targets eval `dfac9ee8`. The evaluator dumps decrypted Thunderbird logins via `firefox_decrypt.py` to a CSV and asserts the `(url, user)` pair for the removed account is NOT present (`unexpect`). Perturb keeps the oracle and evaluator pinned to the eval's single Outlook account (the standard profile has no other IMAP account to delete) and varies only the instruction wording across 3 paraphrase variants. The `paraphrase` knob_assignment field gives each variant a unique task_id.

### Dropped Account Setup (15c3b339)

`perturb_thunderbird_account_setup` replaces the email address in the
`check_accessibility_tree` xpath rule + the wizard instruction, but it is no
longer in `_INTERNAL_FNS`: the current code drops it because the task causes
GPT-5.4 to safety-refuse typing the literal string `password` into a password
field. Current `train.perturb.jsonl` emits **0** rows for this source.

### Evaluator Functions

| func | What it checks |
|---|---|
| `check_thunderbird_prefs` | prefs.js key-value pairs via regex match |
| `check_thunderbird_filter` | Filter rules: action, actionValue, condition keywords |
| `check_list` | File patterns (`.msf` for folders; `cache_file` for backup) |
| `run_sqlite3` | SQL query result against `global-messages-db.sqlite` |
| `check_accessibility_tree` | UI tree xpath match (account_setup: generated; 7b1e1ff9: returns [] — no email in XPath) |

---

## Instruction Style

**Targets** (V3 distribution check, matches eval baseline):
- `polite` prefix rate: **13–20%** (eval baseline = 13%; eval uses mostly imperative + narrative).
- `save the file` rate: **0%** (Thunderbird tasks have no file-save semantics).
- `avg_words`: **25–30** (eval baseline = 28.7).
- `multi_sep` rate: ~0% (eval has no multi-step separators).

**Style guide for templates** (each op pool exposes a 5-template paraphrase pool):
- 4 of 5 templates are imperative/declarative ("In Thunderbird, …", "Configure Thunderbird so …", "Open Thunderbird's account settings and …", "Build a filter that …").
- 1 of 5 templates uses a polite/first-person prefix (`Please …`, `Could you …`, `I want to …`, `I need …`). This yields ~20% polite per pool but rng sampling across 30 rows lands in the 13–20% target band.
- Each template adds **task context / motivation** (managing multiple accounts, archiving off-server, batch flagging for review, low-light work, metered connection, team correspondence format) so descriptions feel like the eval's narrative style ("…I work late into the night and use Thunderbird frequently…") rather than telegraphic commands.
- Each template names concrete UI surfaces (sidebar, account settings, advanced config editor, new-account wizard, identity settings, folder pane) and the parameter values to make execution unambiguous.
- Boolean-pref templates use `{verb}` lowercased mid-sentence ("In Thunderbird, enable …"); the surrounding template provides sentence-initial capitalization.

**Polite-rate guardrail**:
- `BACKUP`, `LOCAL_FOLDER`, `THEME` pools have 0/5 polite templates — these contribute multiple rows per task so a single polite template would over-represent in the overall mix.
- `BOOLEAN×3` (applyIncomingFilters / use_status_for_biff / auto_quote), `SIGNATURE`, `FORWARD`, `UNIFIED_INBOX`, `REMOVE_ACCOUNT` pools have 1/5 polite templates each.
- `FILTER` has 2/5 polite templates (`Please …` and `I need …`); the move-filter task contributes 4 rows so the second polite variant gets sampled often enough to balance the polite-free `BACKUP` / `LOCAL_FOLDER` contributions.
- Aggregate target across the 32 emitted perturb rows (post P3-4 + R1, after dropping setup/star): ~13–20% polite. Per-seed variance is acceptable; aggregate is what matters for V3.

---

## Perturbation Strategy

**TYPE_1 only (same task type, new value)**: each perturbable task maps to a single operation type. Perturb resamples excluding the eval's value.

**Key constraints**:
- Theme: restricted to "dark" candidate only to prevent trivial pass.
- Boolean prefs: flips the single targeted pref; strips secondary prefs from evaluator so agent can satisfy it via a single setting change.
- Local folder: requires ≥2 `.msf` pattern entries in `check_list` to be eligible.
- Backup path: distinguished from local_folder by `result.type == "cache_file"` in evaluator.
- Star folder: documented but currently dropped; the non-eval profile folders race async gloda indexing.

---

## Per-task Plan

### Source Plan (11 recognized tasks; 9 emit rows, 2 are dropped)

| tid | func | sub-type | eval value | Rows |
|---|---|---|---|---|
| `08c73485` | `check_thunderbird_prefs` | boolean | `applyIncomingFilters=true` and `use_status_for_biff=false` | 2 (one per pref) |
| `3f28fe4f` | `check_thunderbird_prefs` | signature | name\naffiliation regex | 4 |
| `f201fbc3` | `check_thunderbird_prefs` | boolean | `auto_quote=false` | **4** (R1: paraphrase variants when single-pref) |
| `5203d847` | `check_thunderbird_filter` | move-to-folder | folder=Promotions, keyword=discount | 4 |
| `9b7bc335` | `check_thunderbird_filter` | forward | email=anonym-x2024@outlook.com | 4 |
| `a10b69e1` | `check_list` | local_folder | COMPANY + UNIVERSITY | 4 |
| `9bc3cc16` | `check_list`/cache_file | backup_path | path=emails.bak | 4 |
| `dd84e895` | `run_sqlite3` | star_folder | folder=Bills (id=13) | 0 — dropped: only "Bills" is local; other folders race async gloda indexing |
| `15c3b339` | `check_accessibility_tree` | account_setup | email address in XPath | 0 — dropped: safety refusal on literal `password` field text |
| `3f49d2cc` | `check_json` | unified_inbox (**P3-4 real gap**) | `mode = "smart"` regex | 3 (favorite/unread/recent) |
| `dfac9ee8` | `check_csv` | remove_account (**P3-4 real gap**) | unexpect `anonym-x2024@outlook.com` | 3 (paraphrase variants) |

> **Note `08c73485`**: has TWO boolean prefs in the eval (`applyIncomingFilters=true` + `use_status_for_biff=false`). Perturb generates one row per pref (2 total), each row strips the other pref from the evaluator so agent only needs to flip one setting.

> **Note `f201fbc3` (R1-thunderbird)**: has only ONE boolean pref (`auto_quote=true`). To avoid degrading to a single-row training signal, the boolean perturb fn detects single-pref bases and emits 4 paraphrase variants drawn from the same 5-template pool (`rng.sample` without replacement); `knob_assignment` includes a `paraphrase` index so each variant gets a unique task_id hash. Multi-pref bases (`08c73485`) keep 1 row per pref (no paraphrase key) to avoid over-representing this archetype.

> **Note `9bc3cc16`**: distinguished from local_folder (`a10b69e1`) by `result.type == "cache_file"` in the evaluator; both use `check_list`. The task backs up inbox emails as `.eml` files to a named directory.

> **Note `dd84e895`**: folder pool is bounded by folders in the standard profile's sqlite DB. The non-eval folders are IMAP-backed and race async gloda indexing, so the current generator drops this archetype instead of emitting rows.

### Not Perturbable (6 tasks: 2 infeasible, 2 active but unhandled, 2 safety/race drops)

| tid | Reason |
|---|---|
| `10a730d5` | **Eval targets dark theme** (`'ref': 'dark'`); only candidate would be dark (leaks eval value); light/default excluded as trivial-pass — 0 valid candidates |
| `a1af9f1c` | infeasible |
| `d38192b0` | infeasible (excluded in eval.jsonl) |
| `7b1e1ff9` | `check_accessibility_tree` — profile management UI state; `perturb_thunderbird_account_setup` returns [] since XPath contains no email address |
| `15c3b339` | account setup is generated by code but dropped from `_INTERNAL_FNS`; safety refusal on literal `password` field text |
| `dd84e895` | star folder is generated by code but dropped from `_INTERNAL_FNS`; non-eval folders are IMAP-backed and race async gloda indexing |

> **Closed in P3-4** (was "active but unhandled" pre-P3-4): `3f49d2cc` (check_json/unified_inbox) and `dfac9ee8` (check_csv/remove_account) — see Per-task Plan above.

---

## V4a Coverage Check

```python
"""V4a coverage check — thunderbird.
Run from repo root: uv run python this_script.py
"""
import json, re
from collections import Counter

REQUIRED_TASK_TYPES = {"signature", "boolean_pref", "move_filter", "forward_filter", "local_folder", "backup_path", "unified_inbox", "remove_account"}

all_perturb = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
tb_perturb = [r for r in all_perturb if "_thunderbird_" in r["task_id"]]

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
    if func == "run_sqlite3": return "star_folder"
    if func == "check_accessibility_tree": return "account_setup"
    if func == "check_json": return "unified_inbox"
    if func == "check_csv": return "remove_account"
    if func == "check_list":
        result = new_ev.get("result", {})
        # Distinguish local_folder (thunder-local-folder.ls) from backup_path by result path.
        # Both have result.type=="cache_file"; .msf check alone is insufficient.
        result_path = result.get("path", "") if isinstance(result, dict) else ""
        if "local-folder" in result_path:
            return "local_folder"
        return "backup_path"
    if func == "check_thunderbird_filter":
        # Disambiguate by evaluator action field, not "@" in oracle:
        # move_filter oracle contains nobody@Local%20Folders which would
        # falsely trigger a bare "@" check.
        exp = new_ev.get("expected", {})
        if isinstance(exp, list): exp = exp[0] if exp else {}
        rules = exp.get("rules", {}) if isinstance(exp, dict) else {}
        expect_list = rules.get("expect", [])
        action_type = (
            expect_list[0].get("action", "") if isinstance(expect_list, list) and expect_list else ""
        )
        return "forward_filter" if action_type == "Forward" else "move_filter"
    if func == "check_thunderbird_prefs":
        if "activeThemeID" in oracle: return "theme"
        if "htmlSigText" in oracle: return "signature"
        return "boolean_pref"
    return "_unknown"

type_counts = Counter(_task_type(r) for r in tb_perturb)
missing = REQUIRED_TASK_TYPES - set(type_counts.keys())
print(f"Total thunderbird perturb rows: {len(tb_perturb)}")
print(f"[{'FAIL' if missing else 'OK  '}] missing task types: {missing or 'none'}")
total = len(tb_perturb) or 1
for t in sorted(REQUIRED_TASK_TYPES | set(type_counts.keys())):
    cnt = type_counts.get(t, 0)
    flag = " <-- MISSING" if cnt == 0 else ""
    print(f"  {t:<25}  {cnt:>4}  {cnt/total:>6.1%}{flag}")
```

Targets:
- All 8 emitted task types present
- Total rows: **32** (2+4+4+4+4+4+4+0+0+3+3) — post P3-4 + R1, after dropping setup/star

---

## V4b Perturb-Eval Match Verification

### Part A: Instruction Clarity

```python
import json, random

rows = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
tb = [r for r in rows if "_thunderbird_" in r["task_id"]]

def _oracle(r):
    return "\n".join(
        a.get("parameters", {}).get("command", "")
        for a in (r["metadata"].get("others") or {}).get("oracle_actions", [])
        if isinstance(a.get("parameters", {}).get("command"), str)
    )

rng = random.Random(0)
for r in rng.sample(tb, min(15, len(tb))):
    print(f"[{r['task_id'].split('_thunderbird_')[-1].split('_')[0]}]")
    print(f"  INSTR : {r['instruction']}")
    print(f"  ORACLE: {_oracle(r)[:200].strip()}")
    print()
```

What to verify:
- Signature tasks: instruction says "first line" / "second line", not "Name / Affil" inline
- Filter tasks: folder name and keyword in both instruction and oracle match
- Local folder tasks: instruction explicitly says "Local Folders account" (not IMAP)

### Part B: Feasibility

```python
import json, re

all_perturb = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
tb_perturb = [r for r in all_perturb if "_thunderbird_" in r["task_id"]]

def _oracle(r):
    return "\n".join(
        a.get("parameters", {}).get("command", "")
        for a in (r["metadata"].get("others") or {}).get("oracle_actions", [])
        if isinstance(a.get("parameters", {}).get("command"), str)
    )

issues = []
for r in tb_perturb:
    oracle = _oracle(r)
    new_ev = r["metadata"]["evaluator"]
    func = new_ev.get("func","") if isinstance(new_ev,dict) else ""

    # Theme: must be dark
    if func == "check_thunderbird_prefs" and "activeThemeID" in oracle:
        if "compact-dark" not in oracle:
            issues.append((r["task_id"], f"theme oracle doesn't set compact-dark"))

    # Boolean pref: no secondary prefs in evaluator expect
    if func == "check_thunderbird_prefs" and "activeThemeID" not in oracle and "htmlSigText" not in oracle:
        exp = new_ev.get("expected",{})
        if isinstance(exp,list): exp = exp[0]
        rules = exp.get("rules",{}) if isinstance(exp,dict) else {}
        expect = rules.get("expect",{})
        if isinstance(expect,dict) and len(expect) > 1:
            issues.append((r["task_id"], f"evaluator expects {len(expect)} prefs but instruction names 1"))

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
tb_perturb = [r for r in all_perturb if "_thunderbird_" in r["task_id"]]

leakage = []
for r in tb_perturb:
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
tb_perturb = [r for r in all_perturb if "_thunderbird_" in r["task_id"]]

# Oracle commands may be either str (shell=True) or list (argv form, e.g.
# ["python3", "-c", "..."]). Both forms must be normalized to a comparable
# string so list-form oracles like dd84e895 (star_folder) aren't reported as
# false-positive duplicates by the str-only collapse.
def _oracle(r):
    parts: list[str] = []
    for a in (r["metadata"].get("others") or {}).get("oracle_actions", []):
        cmd = a.get("parameters", {}).get("command", "")
        if isinstance(cmd, list):
            parts.append("\x1f".join(str(p) for p in cmd))
        elif isinstance(cmd, str):
            parts.append(cmd)
    return "\n".join(parts)

# Sources that intentionally share a single (oracle, evaluator) across
# rows and only vary instruction wording — these must NOT be flagged as
# oracle-duplicates. Currently:
#   - dfac9ee8 (remove_account): 3 paraphrase variants of the same Outlook
#     account removal (only one Outlook account in the standard profile).
#   - f201fbc3 (boolean / auto_quote): single-pref base; R1 paraphrase
#     variants reuse the same write-prefs.js oracle.
_PARAPHRASE_ONLY_SOURCES = {
    "perturb_osworld_thunderbird_dfac9ee8",
    "perturb_osworld_thunderbird_f201fbc3",
}

by_source = defaultdict(list)
for r in tb_perturb:
    src = re.sub(r"_[0-9a-f]{8}$", "", r["task_id"])
    by_source[src].append(r)

instr_dups = []
oracle_dups = []
for src, rows in by_source.items():
    if len(set(r["instruction"] for r in rows)) < len(rows):
        instr_dups.append(src)
    if src in _PARAPHRASE_ONLY_SOURCES:
        continue
    if len(set(_oracle(r) for r in rows)) < len(rows):
        oracle_dups.append(src)

print(f"[{'FAIL' if instr_dups  else 'OK  '}] duplicate instructions: {len(instr_dups)} sources")
print(f"[{'FAIL' if oracle_dups else 'OK  '}] duplicate oracle code:  {len(oracle_dups)} sources")
```

---

## Expected Output

- 9 emitted perturb tasks → **32 rows** (2+4+4+4+4+4+4+0+0+3+3) post P3-4 + R1
- 8 task types covered (boolean_pref, signature, move_filter, forward_filter, local_folder, backup_path, unified_inbox, remove_account)
- `10a730d5` (theme): excluded — eval already targets dark; non-dark candidates (light/default) are trivial-pass
- `15c3b339` (account_setup): 0 rows (dropped from `_INTERNAL_FNS`; safety refusal on literal `password` field text)
- Signature (`3f28fe4f`): 4 rows (pool sampling)
- Boolean (`08c73485`): 2 rows (one per pref — generates a row for each of the two boolean prefs in eval)
- Boolean (`f201fbc3`, R1): 4 rows (single-pref base → 4 paraphrase variants of the same flip)
- Boolean: secondary prefs stripped from evaluator so agent only needs to flip one setting
- Star folder (`dd84e895`): 0 rows (dropped from `_INTERNAL_FNS`; non-eval folders are IMAP-backed and race async gloda indexing)
- Unified inbox (`3f49d2cc`, P3-4): 3 rows (favorite/unread/recent — `smart` is eval value, `all` is trivial-pass because the seeded `xulstore.json` already has `folderTree.mode = "all"`)
- Remove account (`dfac9ee8`, P3-4): 3 paraphrase variants (single Outlook account in standard profile, can't vary target email)
- V2 pass-rate target: **100%**
