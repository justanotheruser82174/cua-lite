# VLC — Perturbation Plan

Domain-specific plan for `vlc`. Common workflow is in [`AGENTS.md`](/devs/envs/lite.osworld/perturb/AGENTS.md).

Code: `lite/gym/envs/lite/osworld/src/gen/train/perturb/vlc.py`

---

## Step 0: Understand Eval Tasks

```python
"""Run from repo root: uv run python this_script.py"""
import json
rows = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/eval.jsonl")]
vlc = [r for r in rows if "_vlc_" in r["task_id"]]
print(f"Total vlc tasks: {len(vlc)}")
for r in vlc:
    tid = r["task_id"].split("_vlc_")[-1]
    ev = r["metadata"]["evaluator"]
    func = ev.get("func","?") if isinstance(ev,dict) else str([e.get("func") for e in ev])
    result = ev.get("result",{}) if isinstance(ev,dict) else {}
    rt = result.get("type","") if isinstance(result,dict) else ""
    excl = r["metadata"]["others"].get("exclude_reason","")
    exp = ev.get("expected",{}) if isinstance(ev,dict) else {}
    if isinstance(exp,list): exp = exp[0] if exp else {}
    rules = exp.get("rules",{}) if isinstance(exp,dict) else {}
    print(f"  {tid[:8]} [{func}/{rt}] excl={excl!r}")
    print(f"    {r['instruction'][:80]}")
```

---

## Task Type Definitions

| task_type | Perturb fn | Description | Value pool |
|---|---|---|---|
| `config_setting` | `perturb_vlc_config` | vlcrc key-value pairs | Per-key pools (see below) |
| `recordings_folder` | `perturb_vlc_recordings_folder` | Recording storage path | 7 absolute paths |
| `slider_color` | `perturb_vlc_slider_color` | Volume slider color type — supports `"blackish"`↔`"match"` swap | 4 named "match" RGB palettes (P3-4) |
| `output_filename` | `perturb_vlc_output_filename` | Rename agent's output media file (compare_{images,videos,audios}) | 6 names × 3 dirs per task |
| `playing` | `perturb_vlc_playing` | Vary the file name VLC plays for `is_vlc_playing` (file_name rule) — P3-4 real gap | 6 mp4 names (cp from eval source) |
| `fullscreen` | `perturb_vlc_fullscreen` | Vary the video shown fullscreen for `is_vlc_fullscreen` — P3-4 real gap | 6 mp4 names (cp from eval source) |

### Config Setting Keys (mapped to evaluator func names)

| eval func | vlcrc key | Value pool | Notes |
|---|---|---|---|
| `check_qt_bgcone` | `qt-bgcone` | 0 / 1 | Boolean toggle |
| `check_global_key_play_pause` | `global-key-play-pause` | 0 / 1 | 0 → empty value (`key=`); 1 → `key=Space` |
| `check_play_and_exit` | `play-and-exit` | 0 / 1 | Boolean toggle |
| `check_qt_max_volume` | `qt-max-volume` | 100/125/150/175/200/250/300/400 | Only under Tools→Preferences→All→Interface→Qt→Max Volume Displayed |
| `check_qt_minimal_view` | `qt-minimal-view` | 0 / 1 | Boolean toggle |
| `check_one_instance_when_started_from_file` | `one-instance-when-started-from-file` | 0 / 1 | Boolean toggle |

### Recordings Folder Paths

`/home/user/Desktop` / `/home/user/Documents` / `/home/user/Videos` / `/home/user/Downloads` / `/home/user/Music` / `/home/user/Pictures` / `/home/user/Public`

Instructions must spell out the full absolute path and menu route (Tools→Preferences→Show settings: All→Input/Codecs→Record directory or filename). Bare folder-name templates caused relative-path failures (cycle 27 audit).

---

## Oracle Mechanics (VLC)

### Config Setting

`perturb_vlc_config` writes the vlcrc key-value pair:

1. `mkdir -p ~/.config/vlc`
2. `touch ~/.config/vlc/vlcrc || vlc --reset-config` to ensure file exists
3. `sed -i '/^#?{key}=/d' vlcrc && echo '{key}={value}' >> vlcrc`

**Initial state seeding** (`perturb_config_step`): seeds vlcrc with a NON-target initial value so the default VLC state doesn't already pass the evaluator without agent action (audit pattern: `vlc.perturb_default_value_collision`).

**Special case**: `global-key-play-pause=0` → write `global-key-play-pause=` (empty) not literal `0`, matching VLC's "disabled" convention.

### Recordings Folder

`perturb_vlc_recordings_folder` replaces the `input-record-path` value in vlcrc and the `recording_file_path` in the evaluator.

### Slider Color

`perturb_vlc_slider_color` supports both directions of the `check_qt_slider_colours` rule:

- **eval target = "blackish"** (e.g. `d06f0d4d`): perturb to a **"match"** rule with one of 4 named RGB palettes — `neon green` / `ocean blue` / `warm sunset` / `neutral grey`. Each has 4 RGB triples (12 ints, semicolon-separated) and **none are blackish-eligible** (every value ≥ 100), so collisions with the eval rule are impossible. Oracle writes `qt-slider-colours=<palette rgb>` to vlcrc; pre-config seeds a different non-target RGB so the agent must actually edit the value.
- **eval target = "match"**: would perturb to "blackish" (no eval row exercises this branch yet — kept for forward compatibility).

**P3-4 closure**: this two-branch design is what fixes the previous "0 rows from `d06f0d4d`" and adds `check_qt_slider_colours` to the perturb skill set.

### Output Filename (compare_{images,videos,audios})

`perturb_vlc_output_filename` targets 3 specific tasks where the agent's job is to produce a media file (snapshot / flipped video / extracted audio) and the eval byte-compares against a cloud-hosted gold:

- **`fba2c100`** (compare_images): snap a paused-frame image
- **`aa4b5023`** (compare_videos): flip an upside-down video upright
- **`8f080098`** (compare_audios): extract MP3 audio from a music video

The gold media bytes are fixed (cloud_file URL stays the same); only the **destination filename + dir** is perturbed. Oracle becomes `wget {gold_cloud_url} -O {new_path}/{new_name}` instead of the original path. Instructions retain the underlying verb (snap / flip / extract) so the agent still does the real task — only the output location changes.

Per task: 6 filename candidates × 3 dir candidates = 18 (filename, dir) combos, sample 4 excluding the eval's original combo.

### is_vlc_playing (file_name)

`perturb_vlc_playing` covers `is_vlc_playing` rows where the eval rule is `type: file_name` (only `59f21cfb` is feasible — `bba3381f` is an HLS-stream URL row marked infeasible). Strategy:

1. Find the original mp4 path from `eval_row.metadata.config[*].download.files[].path`.
2. **Post-eval-config** (via `perturb_config_step`, NOT `pre_config_steps`): `cp -f '<orig_path>' '<orig_dir>/<new_name>'` so a real playable file lives at the new path before the oracle acts (zero extra download — same byte content as eval source). Critical: must be appended AFTER eval's `download` step. Pre-pending it would cp a non-existent source and silently leave new_path missing → oracle launches VLC on a missing file → status stays `stopped` → eval scores 0.
3. Update `evaluator.expected.rules.file_name = <new_name>`.
4. Oracle: `pkill -9 -f vlc`, then `vlc --extraintf http --http-password password --no-video-title-show --no-audio '<new_path>'`, then sleep 8. Mirrors eval's oracle structure so the result getter `vlc_playing_info` (HTTP `requests/status.xml`) can read the playing meta.filename. Sleep 8 (not 5) gives VLC's HTTP server enough time to expose `<state>playing</state>` after launch in CI containers.

Up to 2 file_name variants per source. The upstream evaluator (`vlc.py:21`) does `basename(meta.filename)` matching, so any local mp4 with the matching filename satisfies the test at oracle-replay time.

### is_vlc_fullscreen

`perturb_vlc_fullscreen` covers `is_vlc_fullscreen` rows. The evaluator only checks `vm_window_size == vm_screen_size` for the vlc app — it does NOT inspect the playing file. So perturbation = vary the played video's filename so each variant is a fresh "open this file then go fullscreen" scenario.

1. **Post-eval-config** (via `perturb_config_step`): `cp -f` original mp4 to new filename. Same rationale as is_vlc_playing — pre-config would run before download.
2. Oracle: pkill vlc → write minimal vlcrc (`qt-privacy-ask=0`) → relaunch with `DISPLAY=:1 vlc --start-time=15 '<new_path>'` → sleep 5 → `xdotool key f` (toggle fullscreen) → sleep 5. Matches eval's oracle structure exactly.

Up to 2 file variants per source.

### Evaluator Functions

| func | What it checks |
|---|---|
| `check_qt_bgcone` | `qt-bgcone` value in vlcrc |
| `check_global_key_play_pause` | `global-key-play-pause` value |
| `check_play_and_exit` | `play-and-exit` value |
| `check_qt_max_volume` | `qt-max-volume` value |
| `check_qt_minimal_view` | `qt-minimal-view` value |
| `check_one_instance_when_started_from_file` | `one-instance-when-started-from-file` value |
| `is_vlc_recordings_folder` | `input-record-path` in vlcrc |
| `check_qt_slider_colours` | Slider color rule — `blackish` (all RGB < 100) or `match` (literal RGB string) |
| `is_vlc_playing` | Plays a file_name (HTTP `requests/status.xml` → meta.filename basename) |
| `is_vlc_fullscreen` | VLC window size == screen size |

---

## Perturbation Strategy

**TYPE_1 only (same setting, new value)**: perturb resamples from the value pool excluding the eval's target value.

**No-leakage guarantee**: `candidates = [v for v in spec["values"] if v != orig_val]`.

**Boolean toggles** (5 settings): only 1 candidate value each → R1 fallback cycles 4 distinct paraphrase templates with the same value, producing **4 rows per base** (instructions differ; oracle is identical across the paraphrase set; `paraphrase_idx` knob is added to make task_ids hash-distinct). See `perturb_vlc_config` "R1 fallback" comment.

**Max volume** (`check_qt_max_volume`): 7 other values → up to 4 rows.

**Slider color** (`check_qt_slider_colours`): `d06f0d4d` eval target = "blackish" → P3-4 fix: perturb to **"match"** rule using 4 named RGB palettes → up to 4 rows. (Previously 0 rows.)

**is_vlc_playing** (P3-4 real gap): 2 file_name variants per source via `cp` of the eval source mp4.

**is_vlc_fullscreen** (P3-4 real gap): 2 file variants per source via `cp` + xdotool press F.

---

## Instruction Style

> **Keep in sync with code.** The paraphrase pools in `perturb/vlc.py` (`_VLC_CONFIG_SETTINGS[*]["instruction_templates"]`, `_RECORDING_FOLDER_TEMPLATES`, `_SLIDER_COLOR_TEMPLATES`, `_OUTPUT_RENAME_SPECS[*]["instr_templates"]`) are the authority for instruction wording.

**Targets** (V3 distribution match against eval baseline):

| metric | eval | perturb target | mechanism |
|---|---|---|---|
| polite | 41% | 36-46% (eval ±5pp) | 2/5 polite templates per pool (Could you / Please) |
| save | 6% | **0%** (hard rule) | never include "Save the file" |
| multi_sep | 0% | 0% | single-sentence narrative templates |
| avg_words | 29.0 | 25-35 (eval ±5pp; plan v2.4 standard 30.4 ±5pp) | each template carries narrative pad (motivation + context) |

**Paraphrase pool design** — each op has **5 templates**:
- 3 imperative starters (`Enable …`, `Set …`, `Configure …`, `Snap …`, `Rotate …`, …)
- 2 polite starters (`Could you …`, `Please …`)
- Each template carries a short motivation phrase to pad word count to ~26-30 (e.g. "for a tutorial slide deck", "for my offline workout playlist", "the camera was mounted upside down during the shoot", "I run VLC alongside lecture notes").

**Boolean-toggle templates** use `{verb}` (capitalized for sentence-initial) and `{verb_lower}` (mid-sentence). Some pools also use `{verb_lower2/3/5}` for the secondary clause so polarity stays grammatical (e.g. `closes instantly` ↔ `stays open`, `hidden` ↔ `visible`). The verb dict is built per-row in `perturb_vlc_config`.

**Cycle 31 polarity-bug audit**: prior templates embedded narrative pads
that only made sense for the *eval-target* polarity (e.g. "I prefer a
cleaner empty-playback background" while *enabling* the splash-screen
cone, or "compare two clips side by side" while *enabling* single-
instance). Since each boolean-toggle base has only one candidate value
(the opposite of eval target), every row produced a contradictory
instruction. Templates have been rewritten with **polarity-neutral
narrative pads** (e.g. "I'm tuning the player chrome", "I'm dialling
in the launch behaviour"); polarity-aware tail clauses still use the
`{verb_lower2/3/5}` slots, but only in grammatical contexts:
- `verb_lower2` ("fire system-wide" / "stop firing globally") → only
  used in bare-verb infinitive position ("to {verb_lower2}", "set to
  {verb_lower2}").
- `verb_lower3` ("closes instantly" / "stays open") → only used as a
  3rd-person-singular finite verb with a 3rd-person subject ("the
  player window {verb_lower3}").
- `verb_lower5` ("hidden" / "visible") → only used as a predicate
  adjective ("the toolbar is {verb_lower5}").

Removed unused `verb_lower4` slot. The template `to {verb_lower3}`
("to closes instantly") was ungrammatical and has been restructured.

---

## Per-task Plan

> **Keep in sync with code.** The `_VLC_CONFIG_SETTINGS` dict in `perturb/vlc.py` is the authority for which eval funcs are perturbable and what value pools they use.

### Perturbable (13 tasks)

| tid | eval func | eval target | Perturb pool | Rows |
|---|---|---|---|---|
| `215dfd39` | `check_qt_bgcone` | 0 (disabled) | 1 value × 4 paraphrases (R1) | 4 |
| `386dbd0e` | `check_global_key_play_pause` | 0 (disabled) | 1 value × 4 paraphrases (R1) | 4 |
| `5ac2891a` | `check_play_and_exit` | 0 (disabled) | 1 value × 4 paraphrases (R1) | 4 |
| `9195653c` | `check_qt_max_volume` | 200 | 100/125/150/175/250/300/400 → up to 4 | 4 |
| `a5bbbcd5` | `check_qt_minimal_view` | 1 (enabled) | 1 value × 4 paraphrases (R1) | 4 |
| `f3977615` | `check_one_instance_when_started_from_file` | 0 (disabled) | 1 value × 4 paraphrases (R1) | 4 |
| `8ba5ae7a` | `is_vlc_recordings_folder` | `/home/user/Desktop` | 6 other paths | 4 |
| `d06f0d4d` | `check_qt_slider_colours` | blackish | 4 named "match" RGB palettes (P3-4) | 4 |
| `fba2c100` | `compare_images` (snapshot) | `/home/user/Desktop/interstellar.png` | 18 (name,dir) combos minus eval | 4 |
| `aa4b5023` | `compare_videos` (flipped) | `/home/user/1984_…_Commercial.mp4` | 18 combos | 4 |
| `8f080098` | `compare_audios` (audio extract) | `/home/user/Desktop/Baby Justin Bieber.mp3` | 18 combos | 4 |
| `59f21cfb` | `is_vlc_playing` (file_name) | Rick Astley mp4 | 6 mp4 names → 2 (P3-4) | 2 |
| `8d9fd4e2` | `is_vlc_fullscreen` | (UI state, no value) | 6 mp4 names → 2 (P3-4) | 2 |

> **R1 (single-variant base upgrade — Phase 3-C)**: the 5 boolean toggles
> (`215dfd39 / 386dbd0e / 5ac2891a / a5bbbcd5 / f3977615`) used to emit
> 1 row each (only 1 non-eval candidate value). They now emit 4 rows
> via paraphrase cycling — same oracle, different instruction text,
> `paraphrase_idx` added to the knob_assignment so task_ids stay
> hash-distinct. Rationale: training signal scales with paraphrase
> diversity even when the underlying op is binary-toggle.

> **P3-4 (real-gap closure)**: `d06f0d4d` (slider_colours), `59f21cfb`
> (is_vlc_playing), `8d9fd4e2` (is_vlc_fullscreen) all moved from "not
> perturbable" to perturbable. The slider_colours fix uses the
> evaluator's "match" rule with literal RGB palettes (no value
> collision with "blackish"). is_vlc_playing/fullscreen vary the
> played file via `cp` of the eval source mp4 — zero extra download
> cost, byte-identical content.

### Not Perturbable (4 tasks: 3 infeasible, 1 active but unhandled)

| tid | Reason |
|---|---|
| `7882ed6e` | infeasible (Stranger Things — protected content) |
| `bba3381f` | infeasible (live HLS stream — non-reproducible) |
| `cb130f0d` | infeasible (auto-adjust brightness/contrast — no oracle) |
| `efcf0d81` | `compare_images` (wallpaper from video frame) — needs new gold per perturbed time-point |

---

## V4a Coverage Check

```python
"""V4a coverage check — vlc.
Run from repo root: uv run python this_script.py
"""
import json, re
from collections import Counter

REQUIRED_FUNCS = {
    "check_qt_bgcone", "check_global_key_play_pause", "check_play_and_exit",
    "check_qt_max_volume", "check_qt_minimal_view",
    "check_one_instance_when_started_from_file", "is_vlc_recordings_folder",
    "check_qt_slider_colours",
    "compare_images", "compare_videos", "compare_audios",
    "is_vlc_playing", "is_vlc_fullscreen",
}

all_eval = {r["task_id"]: r for r in (json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/eval.jsonl"))}
all_perturb = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
vlc_perturb = [r for r in all_perturb if "_vlc_" in r["task_id"]]

def _source_func(r):
    src = r["metadata"]["others"].get("source", "").replace("perturb:", "")
    ev = all_eval.get(src)
    if not ev: return "unknown"
    evaluator = ev["metadata"]["evaluator"]
    return evaluator.get("func","?") if isinstance(evaluator,dict) else "?"

func_counts = Counter(_source_func(r) for r in vlc_perturb)
missing = REQUIRED_FUNCS - set(func_counts.keys())
print(f"Total vlc perturb rows: {len(vlc_perturb)}")
print(f"[{'FAIL' if missing else 'OK  '}] missing setting types: {missing or 'none'}")
total = len(vlc_perturb) or 1
for f in sorted(REQUIRED_FUNCS | set(func_counts.keys())):
    cnt = func_counts.get(f, 0)
    flag = " <-- MISSING" if cnt == 0 else ""
    print(f"  {f:<45}  {cnt:>4}  {cnt/total:>6.1%}{flag}")
```

Targets:
- All 7 setting funcs + check_qt_slider_colours + 3 compare_* funcs + is_vlc_playing + is_vlc_fullscreen present (12 funcs total)
- Total rows: **44 current** (older ~48-row target predates later drops/refactors; R1 boost on 5 boolean bases + P3-4 real gaps remains the design)

---

## V4b Part C: Distribution Match

```python
"""V4b Part C — distribution match (vlc).
Run from repo root: uv run python this_script.py
"""
import json, re
from collections import Counter, defaultdict

all_eval = {r["task_id"]: r for r in (json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/eval.jsonl"))}
all_perturb = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
vlc_perturb = [r for r in all_perturb if "_vlc_" in r["task_id"]]

def _oracle(r):
    return "\n".join(
        a.get("parameters", {}).get("command", "")
        for a in (r["metadata"].get("others") or {}).get("oracle_actions", [])
        if isinstance(a.get("parameters", {}).get("command"), str)
    )

# Rows per source task
by_source = defaultdict(int)
for r in vlc_perturb:
    src = re.sub(r"_[0-9a-f]{8}$", "", r["task_id"])
    by_source[src] += 1

print("Rows per source task:")
for src, n in sorted(by_source.items()):
    print(f"  {src.split('_vlc_')[-1]}: {n}")

# Max volume values
vol_ctr = Counter()
for r in vlc_perturb:
    oracle = _oracle(r)
    m = re.search(r'qt-max-volume=(\d+)', oracle)
    if m: vol_ctr[int(m.group(1))] += 1
if vol_ctr:
    print(f"\nMax volume values drawn: {dict(vol_ctr)}")
    print(f"  200 (eval value): {'present (LEAKAGE)' if 200 in vol_ctr else 'absent (OK)'}")

# Recording paths
path_ctr = Counter()
for r in vlc_perturb:
    oracle = _oracle(r)
    m = re.search(r'input-record-path=(/home/user/\w+)', oracle)
    if m: path_ctr[m.group(1)] += 1
if path_ctr:
    print(f"\nRecording paths drawn: {dict(path_ctr)}")
    eval_path = "/home/user/Desktop"
    print(f"  {eval_path} (eval value): {'present (LEAKAGE)' if eval_path in path_ctr else 'absent (OK)'}")
```

---

## V4c Eval Leakage Check

```python
import json, re

all_eval = {r["task_id"]: r for r in (json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/eval.jsonl"))}
all_perturb = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
vlc_perturb = [r for r in all_perturb if "_vlc_" in r["task_id"]]

def _oracle(r):
    return "\n".join(
        a.get("parameters", {}).get("command", "")
        for a in (r["metadata"].get("others") or {}).get("oracle_actions", [])
        if isinstance(a.get("parameters", {}).get("command"), str)
    )

leakage = []
for r in vlc_perturb:
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
vlc_perturb = [r for r in all_perturb if "_vlc_" in r["task_id"]]

def _oracle(r):
    return "\n".join(
        a.get("parameters", {}).get("command", "")
        for a in (r["metadata"].get("others") or {}).get("oracle_actions", [])
        if isinstance(a.get("parameters", {}).get("command"), str)
    )

by_source = defaultdict(list)
for r in vlc_perturb:
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

- 13 perturbable task entries → **44 current rows** (older ~48-row target predates later drops/refactors)
  - **5 boolean toggles** (`215dfd39 / 386dbd0e / 5ac2891a / a5bbbcd5 / f3977615`): 4 rows each via R1 paraphrase cycling (was 1 row each pre-R1)
  - `check_qt_max_volume`: up to 4 rows
  - `is_vlc_recordings_folder`: up to 4 rows
  - `check_qt_slider_colours` (`d06f0d4d`): 4 rows (P3-4: "match" RGB palettes; was 0 rows)
  - `compare_images` / `compare_videos` / `compare_audios` (`fba2c100` / `aa4b5023` / `8f080098`): up to 4 rows each — relocate output file, gold media unchanged
  - `is_vlc_playing` (`59f21cfb`): 2 rows (P3-4: file_name perturbation)
  - `is_vlc_fullscreen` (`8d9fd4e2`): 2 rows (P3-4: video filename perturbation)
- 12 evaluator funcs covered (was 10 pre-P3-4)
- V2 pass-rate target: **100%**
