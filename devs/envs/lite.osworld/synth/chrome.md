# Chrome — Synth Plan

> Keep in sync with code. Implementation: [`chrome.py`](/lite/gym/envs/lite/osworld/src/gen/train/synth/chrome.py).
> Common workflow: [`AGENTS.md`](/devs/envs/lite.osworld/synth/AGENTS.md). Cross-reference: [`perturb/chrome.md`](/devs/envs/lite.osworld/perturb/chrome.md).

## Current quant-gap snapshot (`measure_gap.py` v2)

Run `uv run python devs/envs/lite.osworld/measure_gap.py --domain chrome` for live numbers. Synth N=160, eval N=43 (3 infeasibility filtered).

| Dim | Synth | Eval | Δpp | Status | Bridge |
|---|---:|---:|---:|:-:|---|
| `url_leak.url_leaked` | 0% | 0% | 0 | ✓ | Cycle-47: stripped raw `https://...` from F-CHROME-1/3/14/15/16/17/18/72 instructions |
| `relative_time.relative_time` | 5.6% | 11.6% | -6 | ⚠️ | Cycle-47: added 3 compound templates (F-CHROME-95/96/104) + 3 3-atom (F-CHROME-105/106/107) using `rule_relativeTime`; classifier only counts compound shape (eval matches use list-form `expected`) |
| `atom_count.atom_2` | 7.5% | 11.6% | -4 | ✓ | Cycle-47: added compound `func:[cdjo, cdjo]` templates (F-CHROME-95..104, F-CHROME-102/103) |
| `atom_count.atom_3plus` | 2.5% | 7.0% | -4.5 | ✓ | Cycle-47: added 3-atom templates (F-CHROME-105/106/107) — OD/pax/dates split |
| `result_type.active_tab_html_parse` | 4.4% | 14.0% | -10 | ⚠️ | Cycle-47: added F-CHROME-100..103/108/109 staging deterministic HTML pages, eval reads class fields via `active_tab_html_parse` |
| `result_type.active_tab_url_parse` | 38.8% | 14.0% | +25 | 🔴 | Over-represented; will drop as html_parse and compound atoms grow further |
| `slot_resolution.config_preresolved_url` | 71.2% | 79.1% | -8 | ⚠️ | Aligned — both halves pre-resolve URLs in config |
| `eval_fn_family.tabs` | 33.1% | 30.2% | +3 | ✓ | Aligned — was +10 in cycle 46 baseline |

**Quant-correction**: gap.md previously claimed eval "never pre-resolves URL"; actually eval pre-resolves 79% — the slot-resolution gap is small. Cycle-47 headline: every former ❌ cell now ⚠️ or ✓.

## Current shape

**45 `File`s × 45 `FileTask`s → 132 current jsonl rows** (historical scaler snapshot: 80 rows). Each `File` encodes one structurally distinct chrome profile-state shape (bookmarks / cookies / history / preferences / desktop / open-tabs / staged-asset / URL-decoy); each `FileTask` is one operation on that shape with ≤2 `Param`s. Cap-2×2 (`SYNTH_CAP_TASKS_PER_FILE = SYNTH_CAP_PARAMS_PER_TASK = 2`).

| eval_class | FileTasks | Eval `func` (built by the matching `_gold_*`) |
|---|---:|---|
| `check_direct_json_object` | 15 | `check_direct_json_object` over `active_tab_url_parse` — built by `_gold_url_query` |
| `is_expected_url_pattern_match` | 7 | `is_expected_url_pattern_match` over `active_url_from_accessTree` — built by `_gold_url_pattern` |
| `config_setting` | 6 | `exact_match` on Preferences keys (DNT / SafeBrowsing / profile.name / startup / search engine) |
| `cookies` | 5 | `is_cookie_deleted` on per-domain Cookies sqlite |
| `active_tab` | 5 | `is_expected_active_tab` over `active_tab_info` — built by `_gold_navigate_active_tab` |
| `shortcut` | 3 | `is_shortcut_on_desktop` — built by `_gold_create_shortcut` |
| `bookmark` | 2 | `is_expected_bookmarks` on Bookmarks JSON tree |
| `tabs` | 1 | `is_expected_tabs` over `open_tabs_info` — built by `_gold_open_tabs` |
| `history` | 1 | `check_history_deleted` on per-domain History sqlite |
| **Historical table total** | **45** | |

Current generated total: **132 rows** in `train.synth.jsonl`; the table above is a historical evaluator-bucket snapshot.

**File inventory**: Files 1-23 build on-disk chrome state (Bookmarks JSON / Preferences patches / History+Cookies sqlite / Desktop entries / pre-opened tabs incl. real-asset Wikipedia via `_stage_asset`). Files 24-45 are URL decoys — 24-28, 31-32, 34-41 are URL-query decoys against real-site `base_url`s (rentalcars / amazon / indeed / github / booking / macys / zappos / united / jetblue / kayak / walmart / target / ebay / yelp / redfin); Files 29-30, 33, 42-45 are URL-pattern decoys (DMV / wiki / gov / united-baggage / IRS / GitHub / StackOverflow).

## Architecture / design notes

**Eval files/task ratio**: 0.02. Chrome eval is dominated by browser state (cookies / Preferences / Bookmarks / open tabs), NOT source files — synth's value-axis is **state-shape diversity**, not file-shape.

**Key state-axis variation**: pre-existing tabs (0/1/many decoy URLs); pre-seeded bookmarks (empty/single/nested); profile state (default/custom name/multi-profile); cookie state (clean/specific domains pre-set); Preferences keys (default/partial overrides); window count / tab grouping.

**Shared infrastructure**:

- `_CHROME_RESTART_POSTCONFIG` — `pkill chrome` → relaunch with `--remote-debugging-port=1337` → `sleep 3`. Used by every config-setting / bookmark / cookies / history / shortcut FileTask so the file edit isn't clobbered by chrome's quit-time rewrite.
- `_SESSION_CLEANUP_CMD` — `rm -f` of `Last Session` / `Last Tabs` / `Current Session` / `Current Tabs` between kill and relaunch. Without this, chrome's session-restore reintroduces the decoy tab(s) alongside the gold URL → tab-count mismatch.
- `_pick_instr(instructions, seed)` — returns `{"_skip": True}` once seed exceeds `len(instructions)`, so the scaler never emits byte-identical rows.
- `_stage_asset(rel_path, dest)` — copies pre-bundled Wikipedia HTML snapshots (`assets/synth/html/wikipedia/*.html`) onto the guest VM (22 articles).
- `_gold_url_query(base_url, gold_query, parse_keys)` — oracle relaunches chrome with `_build_url(base_url, gold_query)` (urlencoded). Eval is `check_direct_json_object` over `active_tab_url_parse`.
- `_gold_url_pattern(gold_url, pattern)` — oracle relaunches chrome with `gold_url`; eval is `is_expected_url_pattern_match` over `active_url_from_accessTree`.

Real-site `base_url`s for the `check_direct_json_object` decoys are picked from eval's task pool (mirrors eval tasks `1704f00f` rentalcars, `2888b4e6` macys, `47543840` budget). URL-pattern files use wiki/GitHub/StackOverflow path shapes (stable) with a few riskier gov/IRS/United variants behind loose regex.

## Implementation references

- `chrome.py` §I (`File` / `Param` / `FileTask` dataclasses, mirrors `libreoffice_*.py` / `vs_code.py`).
- [AGENTS.md §Scaler architecture](/devs/envs/lite.osworld/synth/AGENTS.md#scaler-architecture-cycle-41--design-5) — cross-domain volume rebalance.
- [AGENTS.md §Per-domain Cat 1 / Cat 2 allocation guidance](/devs/envs/lite.osworld/synth/AGENTS.md#per-domain-cat-1--cat-2-allocation-guidance) — Cat 1 30% / Cat 2 70% (perturb already covers settings/bookmarks/URL patterns; synth gains are stateful ops + cdjo/url_pattern decoys).
- Installed `desktop_env.evaluators.metrics.general` — `check_direct_json_object`.
- Installed `desktop_env.evaluators.metrics.chrome` — `is_expected_url_pattern_match`,
  `is_expected_active_tab`, `is_expected_tabs`, `is_expected_bookmarks`,
  `is_cookie_deleted`, `check_history_deleted`, `is_shortcut_on_desktop`.

## Bridge plan / outstanding work

The quant snapshot above is the canonical bridge plan; items it does not cover:

- **Restore-closed-tab (Ctrl+Shift+T)** — chrome's session-restore reads in-memory state; oracle cannot deterministically replay. Deferred.
- **Per-host zoom (`partition.per_host_zoom_levels`)** — keyed by per-process partition_id; chrome rebuilds on launch, file-write does not persist. Deferred.
- **Tab grouping / pinning / new window** — `chrome.tab_group` state is not user-writable from disk; needs CDP injection. Deferred.
- **GitHub HTML bundle** — `assets/synth/html/github/*.html` not yet staged; Wikipedia subset is implemented. Add when bundle lands.

## Cycle-recurring failures to avoid (chrome-specific)

- **Live-site instability (F10)**: NEVER design synth tasks against external HTTP; serve all content as `file://` HTML, staged Wikipedia assets, or as the *gold* URL the oracle relaunches with after the agent acts on the decoy.
- **Cookie/profile schema (F4)**: align with perturb's `_chrome_pref_postconfig` / `_chrome_cookie_*` helpers; eval reads via specific sqlite schemas and fails silently on mismatch.
- **Session restore re-introduces decoy tabs**: every oracle that relaunches chrome must emit `_SESSION_CLEANUP_CMD` between `pkill` and the next launch.
- **Welcome modal at turn_00**: every postconfig launch needs `--no-first-run --no-default-browser-check`.

## Pipeline reference

`pre_config_steps` writes JSON Preferences / SQLite cookies / Bookmarks files via heredocs into `/home/user/chrome-data/Default/...`; launches `google-chrome --remote-debugging-port=1337 --user-data-dir=/home/user/chrome-data`; agent acts via UI; postconfig (`_CHROME_RESTART_POSTCONFIG`) kills + relaunches so chrome reads the just-written state; eval reads sqlite/json directly.

For URL-shape FileTasks the oracle owns the relaunch — pre_config seeds a decoy state, oracle pkills, clears session files, then relaunches with the gold URL.
