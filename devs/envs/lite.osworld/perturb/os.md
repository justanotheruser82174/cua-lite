# OS — Perturbation Plan

Domain-specific plan for `os`. Common workflow is in [`AGENTS.md`](/devs/envs/lite.osworld/perturb/AGENTS.md).

Code: `lite/gym/envs/lite/osworld/src/gen/train/perturb/os.py`

---

## Cycle 35a updates

### SSH user variants 4 → 2

`_perturb_ssh_user` previously emitted `min(4, len(candidates))` rows per ssh-user base. Across the eval distribution this produced a **2.34× over-representation** of the ssh skill — only **1 ssh row** in eval (out of 24 feasible os bases ≈ 4.2% target share), but perturb shipped 4 → 16.7% domain share for that single skill. Trimmed to `min(2, len(candidates))`. New share ratio relative to eval: **~0.97×**, well inside the ±0.5× envelope.

### Architecture-level infeasible guard

The cross-domain audit confirmed `apply_structural_perturbation` (in [`perturb/dispatch.py:70`](/lite/gym/envs/lite/osworld/src/gen/train/perturb/dispatch.py)) is correctly filtering all `evaluator.func == "infeasible"` bases before emission: **0 infeasible perturb rows** observed across all 707 current dataset-wide rows (10 domains). No code change needed — this section documents the verified invariant for cycle 35a+.

---

## Step 0: Understand Eval Tasks

```python
"""Run from repo root: uv run python this_script.py"""
import json
rows = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/eval.jsonl")]
os_tasks = [r for r in rows if "_os_" in r["task_id"]]
print(f"Total os tasks: {len(os_tasks)}")
for r in os_tasks:
    tid = r["task_id"].split("_os_")[-1]
    ev = r["metadata"]["evaluator"]
    func = ev.get("func","?") if isinstance(ev,dict) else str([e.get("func") for e in ev])
    result = ev.get("result",{}) if isinstance(ev,dict) else {}
    rt = result.get("type","") if isinstance(result,dict) else ""
    cmd = result.get("command","") if isinstance(result,dict) else ""
    excl = r["metadata"]["others"].get("exclude_reason","")
    print(f"  {tid[:8]} [{func}/{rt}] excl={excl!r}")
    print(f"    {r['instruction'][:80]}")
```

---

## Task Type Definitions

OS perturb covers five operation types:

| task_type | Perturb fn | Description | Value pool |
|---|---|---|---|
| `file_operation` | `perturb_file_operation` | Volume / dir rename / file check / SSH user | Multiple sub-pools |
| `system_query` | `perturb_system_query` | gsettings boolean toggle | `true` ↔ `false` |
| `permission` | `perturb_permission` | File permission octal code | 644/755/600/700/664/775/444/640 |
| `gnome_favorites` | `perturb_gnome_favorites` | GNOME favorite app removal | 10 apps |
| `large_text` | `perturb_large_text` | Accessibility text scaling | 4 instruction templates |
| `check_moved_jpgs` | `perturb_check_moved_jpgs` | Rename destination dir for recursive jpg copy | 5 dest names (jpg_archive / photos_jpg / all_jpgs / jpg_collection / cpjpg2) |
| `paraphrase_coverage` | `perturb_paraphrase_coverage` | Instruction-only paraphrases for fixed-target bases (closes 13 uncovered feasible eval bases) | Per-base paraphrase pool keyed on 8-char eval id |

### Sub-pools for `file_operation`

| sub-type | Detection | Value pool |
|---|---|---|
| volume | `exact_match` + `pactl`+`Volume` in result command | 10/25/40/50/60/75/80/90/100% |
| dir rename | `exact_match` + `Directory exists` in result command | 8 suffixes: `_Feb_1/_Mar_1/_Apr_1/_v2/_backup/_final/_Jan_1/_v3` |
| file check | `exact_match` + `File exists` in result command | 10 base names |
| SSH user | `check_include_exclude` + "ssh"+"user" in instruction | 9 names: alice/bob/charlie/dave/eve/frank/grace/henry/iris |

---

## Oracle Mechanics (OS)

### Volume

`_perturb_volume`: replaces `str(orig_vol)` with `str(new_vol)` in both oracle command and evaluator expected value. Uses a per-task local RNG (seeded by task_id + vol×207) for instruction selection to ensure JSONL idempotency.

### Directory Rename

`_perturb_dir_rename`: extracts target dir from evaluator `result.command`, infers source dir from oracle `mv` command, generates new target by stripping trailing `_N` and appending a new suffix.

### Permission

`perturb_permission`: builds inline shell eval command (`stat --format=%a`) instead of the original external eval.sh download (fixes vacuous-pass bug for chmod -R 444). Also prepends a `pre_config_steps` .bashrc snippet that auto-cd's the terminal to testDir.

### GNOME Favorites

`perturb_gnome_favorites`: generates a new starting app list (new_remaining + app_to_remove) seeded via `perturb_config_step` that runs AFTER the base config, so the gsettings seed doesn't get overwritten.

### Boolean Toggles

`_perturb_gsettings_toggle`: regex-flips verb in instruction (`enable` ↔ `disable`), updates expected value, replaces `true`/`false` in oracle. Skip if instruction couldn't be safely inverted (e.g., double-negation DND phrasing) or contradictory inversion.

### Large Text

`perturb_large_text`: generates instruction variants from 4 templates without changing the oracle or evaluator (oracle remains idempotent; task is purely instruction phrasing).

### check_moved_jpgs (`23393935`, P3-6 真 gap #2)

`perturb_check_moved_jpgs`: keeps the eval's 4-jpg expected list intact and
perturbs only the **destination directory name** (eval uses `cpjpg`). Each
variant rewrites:

- `evaluator.result.path`  →  `/home/user/Desktop/<new_dest>`
- oracle's `mkdir -p` and `find … -exec cp` target dir
- the config step that pre-creates `~/Desktop/cpjpg`
- the instruction (3 templates: imperative / polite / narrative)

We deep-copy the eval row before mutating its config so each variant has its
own independent `~/Desktop/<new_dest>` setup step.

### paraphrase_coverage (P3-6 base-coverage closure)

`perturb_paraphrase_coverage`: for fixed-target eval bases that admit no
value pool (download-driven `eval.sh` checks, `vm_terminal_output`, single-
target `is_utc_0`, compound `exact_match+exact_match`, regex-uninvertible
boolean toggles), we ship 1-2 instruction paraphrases per base. The
oracle and evaluator are deep-copied unchanged. The dispatcher's exact-
string leak filter still drops any paraphrase that happens to equal the
eval instruction.

Routing key: 8-char hex prefix from `osworld_os_<8char>` task IDs.

### Evaluator Functions

| func | What it checks |
|---|---|
| `exact_match` | Shell command output matches expected string (volume %, gsettings value, etc.) |
| `check_include_exclude` | Shell output includes required patterns (permissions, SSH user in /etc/passwd) |
| `check_gnome_favorite_apps` | gsettings favorite-apps list matches expected |
| `is_utc_0` | Timezone is UTC+0 — checks `timedatectl` line 4 ends with `+0000)` (perturbed via paraphrase only; one target value) |
| `check_moved_jpgs` | Set-equality between `{children[].name}` of `result.path` and `expected` jpg list |
| compound `[exact_match, exact_match]` | OR-conjunction across 2 separate exact_match getters (perturbed via paraphrase only) |

---

## Perturbation Strategy

**TYPE_1 only (same operation, new value)**: each perturbable task's operation type is fixed; perturb resamples the target value excluding the eval's value.

**No-leakage guarantee**: `candidates = [v for v in POOL if v != orig_value]` or equivalent exclusion in each sub-function.

**skip_false for boolean toggles**: gsettings toggles produce exactly 1 row (the inverse boolean).

---

## Instruction Style

> **Keep in sync with code.** Template pools and the polite/imperative ratio
> below must match `_VOLUME_TEMPLATES` / `_RENAME_TEMPLATES` / `_PERMISSION_TEMPLATES` /
> `_FAVORITES_REMOVE_TEMPLATES` / `_TEXT_SCALE_TEMPLATES` / `_SSH_USER_TEMPLATES`
> in [`os.py`](/lite/gym/envs/lite/osworld/src/gen/train/perturb/os.py).

OS perturb instructions are sampled from per-op paraphrase pools rather than
generated from a knob-based template. Each pool ships **5 paraphrases**:

- **3 imperative + narrative** (e.g., "Prepping this tree before pushing to a shared server — change every regular file under the current directory to 644, recursively.")
- **2 polite + narrative** (e.g., "Could you set permissions to 644 for every regular file under the current directory tree? I'm normalizing the working copy after a noisy checkout.")

Each pool template includes a brief OS-context **motivation** (cleaning workspace,
sharing the laptop, onboarding a teammate, accessibility setup, locking down a
working copy) so style matches eval rows like "I'm cleaning up my workspace
before sharing the laptop, can you find and delete all .log files older than 7 days?"

V3 distribution targets (mean across seeds; per-seed values vary stochastically):

| metric  | eval | perturb (mean) | target |
|---------|------|----------------|--------|
| polite  | 38%  | ~33%           | 30-45% |
| save    | 0%   | 0%             | 0%     |
| avg_words | 18.5 | ~18.2        | 16-20  |

Exception: `_perturb_gsettings_toggle` regex-flips the
verb in the base eval instruction (no paraphrase pool); those 2 tasks currently
produce 0 rows (excluded — see Per-task Plan).

`_perturb_file_check` uses `instruction.replace(filename, new_filename)` (no
paraphrase pool), but the only eval task that hits this path (`5ea617a3`) is
excluded for unrelated infeasibility reasons, so the file_check pool is never
exercised in practice.

---

## Per-task Plan

> **Keep in sync with code.** Routing logic (which perturb fn handles which eval func+result combination) must match the code.

### Perturbable via value-pool (8 tasks)

| tid | eval func / detection | eval value | Perturb pool | Rows |
|---|---|---|---|---|
| `28cc3b7e` | `exact_match` / pactl volume | 100 | 9 other levels (10–90%) | 4 |
| `e0df059f` | `exact_match` / Directory exists | `todo_list_Jan_2` | 8 suffix variants | 4 |
| `4d117223` | `check_include_exclude` / permission | 644 | 7 other octal codes | 4 |
| `5812b315` | `check_include_exclude` / SSH user | "charles" | 8 other usernames | 4 |
| `ec4e3f68` | `check_gnome_favorite_apps` | remove vim | up to 4 other apps from 10-app pool | 2 |
| `3ce045a0` | `check_include_exclude` (compound) / large text | enable | 4 instruction template variants | 4 |
| `23393935` | `check_moved_jpgs` (P3-6 真 gap #2) | dest=`cpjpg` | 5 alt dest names → 3 sampled | 3 |

> **Note `28cc3b7e`**: a `; sleep 2;` in the base eval config's pulseaudio command is stripped by the perturb generator so config is normalized (cycle 28b fix).

> **Note `ec4e3f68`**: generates 2 rows (not 4). Only 2 of the 10 favorite apps are in the current favorites list and are thus removable without gaps.

### Perturbable via paraphrase-only (12 tasks, P3-6 base-coverage closure)

These bases have a fixed target (no value pool) and were previously listed
as "active but unhandled". `perturb_paraphrase_coverage` adds 1-2 instruction
paraphrases each, keeping oracle + evaluator unchanged. Closes 12 of the
13 prior uncovered feasible bases (the 13th, `23393935`, is now handled by
the dedicated `perturb_check_moved_jpgs` value-pool fn above).

| tid | eval func | Why no value pool | Paraphrases |
|---|---|---|---|
| `13584542` | `check_include_exclude` / vm_terminal_output (132x43) | Terminal size string fixed | 2 |
| `37887e8c` | `check_include_exclude` / external eval.sh (mtime + gzip) | eval.sh hardcoded | 1 |
| `4127319a` | `check_include_exclude` / vm_terminal_output (54 lines) | Line count fixed | 2 |
| `5c1075ca` | `check_include_exclude` / external eval.sh (`*failed.ipynb`) | eval.sh hardcoded | 1 |
| `5ced85fc` | `check_include_exclude` / external eval.sh (1<br/>2<br/>3<br/>) | content fixed | 1 |
| `5ea617a3` | `exact_match` / file existence (poster_party_night.webp) | filename fixed by config download | 2 |
| `6f56bf42` | `check_include_exclude` / external eval.sh (file1→dir1/2/3) | eval.sh hardcoded | 2 |
| `94d95f96` | `check_include_exclude` / `which spotify` | package name fixed | 2 |
| `a4d98375` | `exact_match` / gsettings lock-enabled true | Naive verb-flip lands on eval string → leak; ship paraphrase | 2 |
| `f9be0997` | `exact_match` / gsettings DND show-banners=false | Regex-flip can't safely invert "do not disturb" | 2 |
| `b6781586` | `is_utc_0` (P3-6 真 gap #1) | Only one target value (UTC+0); pre-seeds non-UTC TZ to avoid trivial-pass | 3 |
| `bedcedc4` | compound `[exact_match, exact_match]` (P3-6 真 gap #3) | OR-of-2-getters; idle-delay=0 OR idle-dim=false | 2 |

### Not Perturbable (5 infeasible)

| tid | Reason |
|---|---|
| `4783cc41` | infeasible |
| `a462a795` | infeasible |
| `b3d4a89c` | infeasible |
| `c288e301` | infeasible |
| `fe41f596` | infeasible |

---

## V4a Coverage Check

```python
"""V4a coverage check — os.
Run from repo root: uv run python this_script.py
"""
import json, re
from collections import Counter

REQUIRED_TASK_TYPES = {"volume", "permission", "ssh_user", "dir_rename", "gnome_favorites", "large_text"}

all_perturb = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
os_perturb = [r for r in all_perturb if "_os_" in r["task_id"]]

def _oracle(r):
    return "\n".join(
        a.get("parameters", {}).get("command", "")
        for a in (r["metadata"].get("others") or {}).get("oracle_actions", [])
        if isinstance(a.get("parameters", {}).get("command"), str)
    )

def _task_type(r):
    oracle = _oracle(r)
    instr = r["instruction"].lower()
    new_ev = r["metadata"]["evaluator"]
    func = new_ev.get("func","") if isinstance(new_ev,dict) else ""
    if "pactl" in oracle and "volume" in oracle: return "volume"
    # Check ssh_user before permission: ssh_user oracle also contains "chmod 755 /home/..."
    if "useradd" in oracle or "adduser" in oracle: return "ssh_user"
    if "chmod" in oracle and re.search(r'\b[0-7]{3}\b', oracle): return "permission"
    if "mv" in oracle and "Desktop" in oracle: return "dir_rename"
    if "favorite-apps" in oracle: return "gnome_favorites"
    if "gsettings" in oracle and ("true" in oracle or "false" in oracle): return "gsettings_toggle"
    if "large" in instr or "enlarg" in instr or "text" in instr: return "large_text"
    return "_unknown"

type_counts = Counter(_task_type(r) for r in os_perturb)
missing = REQUIRED_TASK_TYPES - set(type_counts.keys())
print(f"Total os perturb rows: {len(os_perturb)}")
print(f"[{'FAIL' if missing else 'OK  '}] missing task types: {missing or 'none'}")
print()
total = len(os_perturb) or 1
for t in sorted(REQUIRED_TASK_TYPES | set(type_counts.keys())):
    cnt = type_counts.get(t, 0)
    flag = " <-- MISSING" if cnt == 0 else ""
    print(f"  {t:<25}  {cnt:>4}  {cnt/total:>6.1%}{flag}")
```

Targets:
- All 6 task types present (gsettings_toggle excluded — see Per-task Plan)
- Total rows: **~39** = 22 (value-pool: volume 4 + dir_rename 4 + permission 4 + ssh_user 4 + favorites 2 + large_text 4) + 3 (check_moved_jpgs) + 14 (paraphrase coverage: 10 × 1 + 2 × 2)
  - Note: V4a's `_task_type` heuristic doesn't recognize the new `check_moved_jpgs` and `paraphrase_coverage` rows → they fall through to `_unknown`. This is expected; coverage of the 6 base task types is still satisfied.
- Actual current count: **34 rows**. The "~39 rows" target above and older 48-row audit note predate later drops/refactors; use generated `train.perturb.jsonl` as source of truth.

---

## V4c Eval Leakage Check

```python
import json

all_eval = {r["task_id"]: r for r in (json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/eval.jsonl"))}
all_perturb = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
os_perturb = [r for r in all_perturb if "_os_" in r["task_id"]]

leakage = []
for r in os_perturb:
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
os_perturb = [r for r in all_perturb if "_os_" in r["task_id"]]

def _oracle(r):
    return "\n".join(
        a.get("parameters", {}).get("command", "")
        for a in (r["metadata"].get("others") or {}).get("oracle_actions", [])
        if isinstance(a.get("parameters", {}).get("command"), str)
    )

by_source = defaultdict(list)
for r in os_perturb:
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

- 19 perturbable feasible tasks → **34 current rows** (older 48-row note predated later drops/refactors)
- 5 infeasible tasks excluded
- 6 base task types covered (volume, permission, ssh_user, dir_rename, gnome_favorites, large_text) plus check_moved_jpgs + paraphrase_coverage (P3-6 additions)
- `f9be0997` (gsettings DND): now covered via paraphrase_coverage (regex-inversion still skipped)
- `a4d98375` (gsettings screen lock): now covered via paraphrase_coverage (boolean-toggle still skipped due to leak)
- `ec4e3f68` (gnome_favorites): 2 rows (not 4) — only 2 apps in the current favorites are removable
- V2 pass-rate target: **100%**

### P3-6 changelog

Phase 3-B P3-6 added two perturb fns to close the os domain ratio gap (0.92 → 1.6+):

1. `perturb_check_moved_jpgs` (真 gap #2 — 23393935): perturbs the destination dir name; oracle still copies all 4 jpgs; evaluator path follows. 3 rows.
2. `perturb_paraphrase_coverage`: 1-3 instruction paraphrases per fixed-target base. Closes 12 of 13 prior uncovered feasible bases (the 13th, 23393935, is owned by `perturb_check_moved_jpgs`). 23 rows after D5 expansion. Includes the two remaining 真 gap bases — `b6781586` (is_utc_0, 真 gap #1) and `bedcedc4` (exact_match+exact_match, 真 gap #3).

`is_utc_0` evaluator: parses `timedatectl status` line 4 — checks if it ends with `+0000)`. The eval task's oracle installs a synthetic `/usr/local/bin/timedatectl` shim that emits the canonical 8-line output reading `/etc/timezone`, so the evaluator works regardless of whether systemd's real `timedatectl` ships in the OSWorld container. We don't need to dispatch a docker probe — the upstream eval task itself proves the path is functional.

**AUDIT (2026-05) — `is_utc_0` trivial-pass guard**: The eval row's config does NOT pre-seed a non-UTC timezone, so on snapshots where `/etc/timezone == UTC` already, the initial state passes `is_utc_0` before the agent runs (vacuous reward). The synth side documented and fixed this same pitfall in `_set_utc_params` (cycle 27). `perturb_paraphrase_coverage` now injects a `pre_config_steps` entry (via `_PARAPHRASE_PRE_CONFIG_STEPS["b6781586"]`) that links `/etc/localtime` to `America/Los_Angeles` and writes `/etc/timezone` accordingly, so the initial state always FAILS the evaluator. The eval oracle still resets `/etc/timezone` to `UTC` and installs the timedatectl shim; the pre-seed is overwritten before the evaluator runs.

**AUDIT (Cycle 32 / rollout) — `is_utc_0` agent-path unwinnable**: The eval base's `oracle_actions[2]` installs the timedatectl shim, but oracle_actions are NOT executed during real agent rollout — only on synth/oracle replay paths. So real agents on the no-systemd OSWorld container saw raw `timedatectl` output ("Failed to connect to bus: Host is down") regardless of what they did to `/etc/timezone` → `is_utc_0` always returned 0. **Fix**: extended `_PARAPHRASE_PRE_CONFIG_STEPS["b6781586"]` to include the shim install as a second pre-config step (after the LA seed), so the shim is present when the agent runs and `is_utc_0` reads `/etc/timezone` correctly. Direct trigger: HOMO_ZERO 3/3 in the train.perturb rollout audit.
