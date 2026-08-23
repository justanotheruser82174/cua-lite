# Chrome — Perturbation Plan

Domain-specific plan for `chrome`. Common workflow is in [`AGENTS.md`](/devs/envs/lite.osworld/perturb/AGENTS.md).

Code: `lite/gym/envs/lite/osworld/src/gen/train/perturb/chrome.py`

> **Keep in sync with code.** Every value pool, instruction template, helper, dispatcher, archetype spec dict (`_BOOKMARK_URL_VARIANTS`, `_PREFS_TASKS`, `_TAB_TASKS`, `_J1_TASKS`, `_J2_VARIANTS`, `_J3_VARIANTS`), and `_INTERNAL_FNS` entry in `chrome.py` must agree with the tables in this document. The `.md` and `.py` are co-evolved peers — divergence silently produces wrong training tasks.

---

## Step 0: Understand Eval Tasks

No source files to download. Run this to inspect per-task evaluator and knob details:

```python
"""Run from repo root: uv run python this_script.py"""
import json
rows = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/eval.jsonl")]
chrome = [r for r in rows if "_chrome_" in r["task_id"]]
print(f"Total chrome tasks: {len(chrome)}")
for r in chrome:
    tid = r["task_id"].split("_chrome_")[-1]
    instr = r["instruction"][:90]
    func = r["metadata"].get("evaluator", {}).get("func", "?")
    result = r["metadata"].get("evaluator", {}).get("result", {})
    result_type = result.get("type", "") if isinstance(result, dict) else ""
    print(f"  {tid}: [{func}] result_type={result_type} | {instr}")
```

---

## Setting / Task Type Definitions

Chrome perturb has three flavors:
- **Existing TYPE_1 (per-setting resample)**: each base maps to a single setting type; perturb resamples a target value from a fixed pool that excludes the eval's value. See "Existing TYPE_1 perturbations" below.
- **Archetypes B/P/T (added 2026-05-08, Phase 1)**: cover bookmark URLs, Preferences keys, and tab/URL state. Each archetype has a hand-curated variant set; the eval target is unchanged or paraphrased. See "Archetype B/P/T" sections below.
- **Archetype J (added 2026-05-08, Phase 3 P3-2)**: covers the `check_direct_json_object` skill-sig gap (URL query-param checks). Attaches to selected cdjo eval bases (which are themselves Not-Perturbable due to live-web result getters) and synthesizes a feasible URL-state oracle that exercises the same evaluator. See "Archetype J" section below.

### Existing TYPE_1 perturbations

| task_type | Perturb fn | Description | Value pool |
|---|---|---|---|
| `enable_do_not_track` | `perturb_browser_setting` | DNT boolean pref in Preferences JSON | `[True]` — skip_false: Chrome normalizes DNT off; "disable" variant always trivial_pass |
| `enable_safe_browsing` | `perturb_browser_setting` | Safe Browsing boolean pref | `[True]` — skip_false: Chrome normalizes SafeBrowsing back to on |
| `data_delete_automacally` | `perturb_browser_setting` | Clear data on exit boolean | `[True, False]`, but the eval=true base emits 0 rows because `skip_false` blocks the trivial-pass false direction |
| `profile_name` | `perturb_browser_setting` | Chrome profile display name | 18-name pool: Alice/Bob/Charlie/David/Emma/Frank/Grace/Henry/Isabella/Jack/Kate/Leo/Maria/Noah/Olivia/Peter/Quinn/Rachel |
| `check_font_size` | `perturb_browser_setting` | Appearance font size dropdown | 5 discrete values: Very small(9px) / Small(12px) / Medium(16px) / Large(20px) / Very large(24px) |
| `default_search_engine` | `perturb_browser_setting` | Default search engine | Google/Microsoft Bing/DuckDuckGo/Yahoo/Ecosia/Brave/Startpage |
| `bookmark_bar_folders_names` | `perturb_bookmark_folder` | Bookmark bar folder name | Work/Reading/Shopping/Travel/News/Research/Recipes/Music/Movies/Sports/Finance/Education |
| `is_cookie_deleted` | `perturb_cookie_domain` | Cookie domain to delete | .amazon.com/.google.com/.facebook.com/.twitter.com/.reddit.com/.youtube.com/.linkedin.com |
| `new_startup_page` | `perturb_startup_page` | Nuisance startup URL to clear | funbrain.com/fakegamingsite.net/annoyingads.example/popups.local/homestartup.dev/mybadhomepage.io/spammy-default.test/weirdstartpage.com |
| `check_history_deleted` | `perturb_history_keyword` | Browsing history keyword to delete | youtube/facebook/twitter/reddit/amazon/instagram/linkedin/pinterest |
| `is_shortcut_on_desktop` | `perturb_desktop_shortcut` | Desktop shortcut name | Play Puzzle Game 2048/Google Maps/Wikipedia/YouTube Music/Weather App/Calculator Online/Gmail Inbox/Translate |

### New archetypes (2026-05-08)

> **Keep in sync with code.** Every helper in `_BOOKMARK_URL_VARIANTS` / `_PREFS_TASKS` / `_TAB_TASKS` and dispatcher in `_INTERNAL_FNS` (chrome.py) must match this section. The `.md` and `.py` are co-evolved peers.

| archetype | Perturb fn | Eval base(s) | Variants per base | Mechanism |
|---|---|---|---|---|
| **B** Bookmark URL | `perturb_bookmark_url` | `7a5a7856` | 4 | pkill chrome → seed empty Bookmarks JSON → relaunch with target_url as active tab; oracle writes Bookmarks JSON containing only `{name=url, type=url, url=target}` in `bookmark_bar.children` |
| **P** Preferences keys | `perturb_preferences_keys` | `030eeff7` (DNT), `9656a811` (SafeBrowsing), `93eabf48` (color_scheme) | 4 (paraphrased instructions) | seed OPPOSITE state in Preferences then relaunch chrome; oracle writes the eval target value into Preferences. `99146c54` (auto-clear) is intentionally dropped in code because the setting is enterprise-policy-shaped, not reachable from the Chrome Settings UI |
| **T1** Tab/URL state (active_tab + url_pattern) | `perturb_navigate_url` | `59155008`, `a96b564e`, `f0b971a1`, `0d8b7de3`, `9f935cce`, `a728a36e`, `c1fa57f3`, `f3b19d1e` | 4 (target URL or regex pattern) | preserve eval base's start-URL `chrome_open_tabs`; oracle launches chrome with target URL `[launch, sleep 5]` |
| **T2** Tab/URL state (open_tabs) | `perturb_open_tabs` | `06fe7178` | 4 (different 3-URL sets) | seed 2 of 3 URLs as `chrome_open_tabs`; agent opens the missing third; oracle launches chrome with all 3 URLs |
| **J1** check_direct_json_object (single, URL params) | `perturb_check_direct_json_object` | `f5d96daf` | 4 (param dict per variant) | oracle launches chrome with a constructed URL whose query string contains explicit literal values; evaluator uses `active_tab_url_parse` + `rule` (literal `expected` dict, no `rule_relativeTime`) |
| **J2** check_direct_json_object compound (cdjo + cdjo) | `perturb_check_direct_json_object_compound` | `7f52cab9` | 4 | oracle URL contains both parse_keys subsets in query string; evaluator is compound `[cdjo, cdjo]` with two `active_tab_url_parse` getters reading disjoint subsets of the same URL |
| **J3** check_direct_json_object compound (cdjo + url_pattern) | `perturb_check_direct_json_object_pattern` | `368d9ba4` | 4 | oracle URL contains both query params (cdjo) AND a path that matches the regex (url_pattern); evaluator is compound `[cdjo, is_expected_url_pattern_match]` |

---

## Oracle Mechanics (Chrome)

### Initial State

All Chrome perturbations operate on Chrome's `Preferences` JSON file at `/home/user/chrome-data/Default/Preferences` (with fallback to `/home/user/.config/google-chrome/Default/Preferences`). The initial state is seeded via `perturb_config_step` (or `pre_config_steps`) that runs **after** the base eval config.

Each setting type has a specific seeding strategy:

| task_type | Initial-state seeding (`perturb_config_step` / `pre_config_steps`) |
|---|---|
| boolean prefs | `pkill chrome; sleep 2` → write opposite bool into Preferences → relaunch Chrome. Required because Chrome running in memory overwrites the file with its in-memory value otherwise. |
| `profile_name` | No pre-seed needed; name starts at base eval's profile name ("Thomas"). |
| `check_font_size` | No pre-seed; base eval's font size is already at 20px (Large). Oracle writes new size. |
| `default_search_engine` | `pkill chrome; sleep 2` → seed BASE engine `template_url_data` → relaunch Chrome with `--user-data-dir=/home/user/chrome-data`. |
| `bookmark_bar_folders_names` | No pre-seed; agent creates the new folder from scratch. |
| `is_cookie_deleted` | Launch Chrome briefly to create DB → pkill → INSERT seed cookie row for new domain → relaunch. |
| `new_startup_page` | `perturb_config_step`: append the nuisance `startup_urls=[nuisance_url]` write after the base jq write so it wins. |
| `check_history_deleted` | `pre_config_steps`: seed `update_browse_history` with new_keyword entries + base youtube entries. |
| `is_shortcut_on_desktop` | No pre-seed; agent creates the shortcut for new name. |

### Oracle

Each perturb function deep-copies the base eval's oracle and substitutes the new target value:

| task_type | Oracle action |
|---|---|
| boolean prefs | Python heredoc: set/remove the pref key in Preferences JSON at the correct nested path |
| `profile_name` | Replace `"name"` field value in Preferences JSON |
| `check_font_size` | `sed`-style regex `s/"default_font_size":\s*\d+/"default_font_size": N/` on Preferences |
| `default_search_engine` | Python heredoc patching `default_search_provider_data.template_url_data` with correct `short_name`/`keyword`/`url` for target engine |
| `bookmark_bar_folders_names` | Modify Bookmarks JSON to include a folder with `names=[new_name]` |
| `is_cookie_deleted` | SQLite DELETE on Cookies DB for `host_key LIKE '%new_domain%'` |
| `new_startup_page` | Python heredoc setting `restore_on_startup=5`, `startup_urls=[]` |
| `check_history_deleted` | SQLite DELETE on History DB WHERE `url LIKE '%new_keyword%'` |
| `is_shortcut_on_desktop` | Write `.desktop` file with `Name=new_name` field |

### Evaluator Functions

| eval func | What it checks |
|---|---|
| `exact_match` / `enable_do_not_track` | `Preferences["enable_do_not_track"] == "true"/"false"` |
| `exact_match` / `enable_safe_browsing` | `Preferences["safebrowsing.enabled"]` |
| `exact_match` / `data_delete_automacally` | `Preferences["browser.clear_data.browsing_data_lifetime.enabled"]` |
| `exact_match` / `profile_name` | `Preferences["profile.name"]` (case-insensitive grep in perturb variant) |
| `check_font_size` | `Preferences["webkit.webprefs.default_font_size"]` by value equality |
| `match_in_list` / `default_search_engine` | `short_name` in `default_search_provider_data.template_url_data` |
| `is_expected_bookmarks` | Bookmarks JSON folder names list |
| `is_cookie_deleted` | SQLite Cookies DB has no rows for target domain |
| `exact_match` / `new_startup_page` | `Preferences["session.restore_on_startup"] == 5` |
| `check_history_deleted` | History DB has no rows matching keywords |
| `is_shortcut_on_desktop` | `.desktop` file with correct `Name=` exists on Desktop |

---

## Perturbation Strategy

**TYPE_1 only (same setting, new value)**: every perturbable chrome task has exactly one setting type. Perturb resamples from the value pool with the eval's known value excluded. **No-leakage guarantee**: each function builds `candidates = [x for x in POOL if x != orig_value]` before sampling — the generated value is always structurally different from the eval task's target. For boolean prefs, `skip_false=True` prevents generating the "disable" variant when Chrome normalizes that state back to on (which would produce trivial_pass).

---

## Archetype B — Bookmarks URL (4 rows)

**Eval base**: `osworld_chrome_7a5a7856` (currently in "Not Perturbable" — moved to "Perturbable Tier-A1" by this archetype).

The eval `is_expected_bookmarks` checks `rules.urls` against the set of URLs found in `bookmark_bar.children` of `/home/user/chrome-data/Default/Bookmarks`. The base eval has the agent bookmark `https://jalammar.github.io/illustrated-transformer/`; perturb retargets each variant at a different `target_url`.

| variant | target_url | distractor extra_open_tab |
|---|---|---|
| 1 | `https://www.python.org/` | `https://docs.python.org/3/` |
| 2 | `https://en.wikipedia.org/wiki/Linux` | (none) |
| 3 | `https://github.com/python/cpython` | `https://github.com/torvalds/linux` |
| 4 | `https://stackoverflow.com/help/asking` | (none) |

Helpers (chrome.py):
- `_build_bookmark_url_oracle(target_url)` — Bookmarks JSON heredoc that writes `{name=url, type=url, url=target}` as the only `url`-typed child of `bookmark_bar`.
- `_build_bookmark_url_perturb_config(target_url, extras)` — `pkill chrome` → seed empty Bookmarks JSON → relaunch chrome with `[*extras, target_url]` (target_url last so it's the active tab).

**Setup**: Empty Bookmarks JSON is seeded so the eval set check `set(urls) == {target_url}` is unambiguous (no leftover bookmarks from prior runs).

**Postconfig**: `pkill chrome 2>/dev/null; sleep 8; true` — sleep 8 (not 2) is required so Chrome flushes the Bookmarks file on SIGTERM (cycle 29 H3 timing).

**Evaluator**: deep-copy of base, with `expected.rules.urls = [target_url]` only. Postconfig replaced with the cycle 29 H3 pkill-then-sleep variant.

---

## Archetype P — Preferences keys (12 rows = 3 axes × 4 variants)

| axis | base eval | pref_path | seed_strategy |
|---|---|---|---|
| `dnt` | `osworld_chrome_030eeff7` | `enable_do_not_track` | `set_false` |
| `safe_browsing` | `osworld_chrome_9656a811` | `safebrowsing.enabled` | `set_false` |
| `color_scheme` | `osworld_chrome_93eabf48` | `browser.theme.color_scheme` (1=light, 2=dark) | `set_dark` |

**Strategy**: always perturb *to* the eval target (true / 1=light) and seed the OPPOSITE state via `perturb_config_step`. Chrome resets `true→false` (so "disable" variants vacuously trivial-pass) but does NOT reset `false→true` or `dark→light` against the prefs file written before launch. This makes the perturb→eval-target direction stable. `clear_on_exit` (`99146c54`) is not part of Archetype P in current code.

Each axis has 4 paraphrased instructions (mostly imperative with 1–2 polite forms) — the *structural* eval target is fixed per axis but the *natural-language paraphrase* varies, giving 4 rows per base for SFT diversity.

Helpers (chrome.py):
- `_build_prefs_seed_step(pref_path, seed_strategy)` — `pkill -f google-chrome; sleep 2` → write OPPOSITE state to `Preferences` (both `.config/google-chrome` and `chrome-data` paths) → relaunch.
- `_build_prefs_oracle(pref_path, axis)` — python3 heredoc that sets the eval target value at the correct nested path (writes both Preferences locations so the eval reader's chrome-data-first fallback works).

**Postconfig**: `pkill chrome 2>/dev/null; sleep 8; true` — same cycle 29 H3 timing as Archetype B.

**Oracle runs before postconfig** (`oracle_after_postconfig=False`, the default) — the oracle must execute while Chrome is alive, write Preferences, and then postconfig kills Chrome so the evaluator reads the file off disk.

---

## Archetype T — Tab/URL state (36 rows = 8 base × 4 variants for T1, 1 base × 4 variants for T2)

T1 covers `is_expected_active_tab` (exact-URL match) and `is_expected_url_pattern_match` (regex match). T2 covers `is_expected_tabs` (set of currently-open tab URLs).

### T1 active_tab (4 base × 4 = 16 rows)

| base | start URL (kept as `chrome_open_tabs`) |
|---|---|
| `osworld_chrome_59155008` | `https://www.babycenter.com/child` |
| `osworld_chrome_a96b564e` | `https://www.flightaware.com/` |
| `osworld_chrome_f0b971a1` | `https://www.nfl.com/` |
| `osworld_chrome_0d8b7de3` | `https://drugs.com` |

Each variant is a target URL drawn from a stable, non-redirecting pool (mirror of synth `_NAVIGATE_TARGET_URLS`). Evaluator: `expected.rules.url = target_url`. Oracle: `[launch chrome with target_url, sleep 5]` — exact mirror of the eval base oracle shape.

**URL stability rule**: every active_tab + open_tabs target URL must satisfy `final_url == original` after `curl -L`. Redirecting URLs (e.g. `https://www.rust-lang.org/learn` → `https://rust-lang.org/learn/`) cause `is_expected_active_tab` / `is_expected_tabs` strict equality to fail. URLs known to redirect: any `www.rust-lang.org/*` (use Wikipedia article instead). Audit script: `for url in <pool>; do curl -sLo /dev/null -w "%{url_effective}" $url; done`.

**`goto_prefix` rule (active_tab variants)**: the eval `result.goto_prefix` is *prepended* to the address-bar text from the accessibility tree to reconstruct the URL. Chrome strips ONLY the `www.` subdomain — for `https://en.wikipedia.org/wiki/Linux` the bar shows `en.wikipedia.org/wiki/Linux`. If the base eval's prefix is `https://www.` and the variant's target URL has a non-`www` subdomain (e.g. `en.`, `docs.`, `news.`), reconstruction yields `https://www.en.wikipedia.org/...` whose normalized netloc differs from the expected URL → `compare_urls` fails. **Fix (post-9ca9d14d audit)**: `_swap_url_in_evaluator` derives `goto_prefix` from `target_url` itself via `_goto_prefix_for_url(target_url)` — `https://www.` if host starts with `www.`, else `https://`. Without this fix, T1 active_tab variants for bases with `goto_prefix="https://www."` (`59155008`, `f0b971a1`, `0d8b7de3`) fail on every non-www target.

### T1 url_pattern (4 base × 4 = 16 rows)

| base | start URL |
|---|---|
| `osworld_chrome_9f935cce` | `https://www.justice.gov/` |
| `osworld_chrome_a728a36e` | `https://www.dmv.virginia.gov/` |
| `osworld_chrome_c1fa57f3` | `https://www.united.com/en/us` |
| `osworld_chrome_f3b19d1e` | `https://premier.ticketek.com.au/` |

Each variant is a `(regex_pattern, oracle_url, description)` tuple drawn from synth `_NAVIGATE_URL_PATTERN_TARGETS`. Evaluator: `expected.rules.expected = [pattern]`. Oracle: `[launch chrome with oracle_url, sleep 5]`.

**Pattern caveat**: regex must be specific enough to NOT match the start URL (e.g. `en\.wikipedia\.org/wiki/Linux` is fine because the start URL is justice.gov; `justice\.gov` would be wrong because it matches the start page itself).

**Redirect caveat (a728a36e v3)**: the original variant 3 of `a728a36e` was `(www\.rust-lang\.org/learn, https://www.rust-lang.org/learn, ...)`. Audit (2026-05-08) confirmed via `curl -IL` that this URL 301-redirects to `https://rust-lang.org/learn/` (drops `www.`, adds trailing slash); the post-redirect URL no longer matches the `www\.rust-lang\.org/learn` regex → variant always fails. Replaced with the Wikipedia article on Rust which is stable.

### T2 open_tabs (1 base × 4 = 4 rows)

Eval base: `osworld_chrome_06fe7178`. Variants are 3-URL sets; setup seeds the first 2 (omitting the last); evaluator: `expected.rules.urls = all 3`. Agent must open the third.

**Setup tweak**: drop the eval base's `chrome_close_tabs` step (it would close `tripadvisor.com` which we don't seed) and replace `chrome_open_tabs.urls_to_open` with the seed subset.

Helpers (chrome.py):
- `_build_navigate_oracle(target_url)` — `[launch google-chrome ... target_url, sleep 5]` (mirror of `_navigate_url_oracle` in synth/chrome.py).
- `_build_open_tabs_oracle(urls)` — single launch step opening all URLs as tabs.
- `_swap_url_in_evaluator`, `_swap_pattern_in_evaluator`, `_swap_urls_in_evaluator` — deep-copy + targeted field replacement.
- `_replace_chrome_open_tabs(eval_row, urls)` — deep-copy eval_row and replace the `chrome_open_tabs.urls_to_open` value.

### T-archetype skipped bases

| tid | reason |
|---|---|
| `121ba48f` | `is_added_to_steam_cart` — live web interaction, can't be seeded offline |
| `e1e75309` | `compare_pdfs` — print-to-PDF, file content depends on live page rendering |
| `6766f2b8` | `is_in_list` — file unzip task, not a Chrome navigation task |
| `12086550` | `is_expected_active_tab_approximate` — fuzzy matcher relies on accessibility tree state, not stable enough for synthetic targeting |

---

## Archetype J — check_direct_json_object on URL query params (12 rows)

**Eval bases**: 3 of the 16 chrome `check_direct_json_object` eval rows (audit v2.4 Phase 3 P3-2). All 16 cdjo eval bases use live-web result getters (`active_tab_url_parse` / `active_tab_html_parse` / `url_dashPart` / `url_path_parse` / `gotoRecreationPage_and_get_html_content`) and remain "Not Perturbable" at the structural level. The J archetype borrows the task_id slot for skill-sig coverage and synthesizes a feasible URL-state variant.

**Strategy**: build a real URL with explicit static query params; oracle launches chrome with that URL via `[pkill, launch, sleep 6]`; evaluator uses `active_tab_url_parse` + `rule` (literal expected dict). All variants avoid `rule_relativeTime` so the oracle is reproducible across days.

### J1 single cdjo (1 base × 4 = 4 rows)

| base | parse_keys | result_extra | URL skeleton |
|---|---|---|---|
| `f5d96daf` (apple compare) | `modelList` | `split_list=True` + `expected.ignore_list_order=True` | `https://www.apple.com/?modelList={comma_list}` (root path, see note) |

Variant tables live in `chrome.py` as `_J1B_VARIANTS` / `_J1B_INSTRUCTIONS`. The table has 4 distinct param dicts that differ from each other AND from the eval base's expected values. Instructions paraphrase the search query; the URL builder + evaluator are deterministic from the param dict.

Dropped J1 candidates: `82279c77` (cars.com) hits a Cloudflare interstitial on fresh Chrome; `82bc8d6a` (kayak) and `f79439ad` (kiwi) were removed after rollout validation showed URL rewrite/live-site instability.

**Root-path rule (J1b apple)**: oracle audit (2026-05-08) found that the natural sub-path
`apple.com/shop/buy-iphone/iphone/compare?modelList=...` issues a server-side redirect that strips
the literal query key our eval reads (`modelList` → dropped entirely). J1b therefore targets the
**root path** (`https://www.apple.com/?...`), which preserves arbitrary query strings without rewrite.
The eval contract is purely URL-state (`active_tab_url_parse` + `parse_qs`), so only the query params
matter, not the destination page content.

### J2 compound cdjo + cdjo (1 base × 4 = 4 rows)

**Eval base**: `7f52cab9` (drip coffee maker — eval is cdjo+cdjo with one `active_tab_url_parse` and one `active_tab_html_parse`).

**Strategy**: replace both result getters with `active_tab_url_parse`, parsing disjoint key subsets (`{q, sort}` and `{category, condition}`) of the same URL. Both checks pass when the single oracle URL contains all 4 query params.

URL skeleton: `https://www.bestbuy.com/site/searchpage.jsp?q={q}&sort={sort}&category={cat}&condition={cond}`

Variants in `_J2_VARIANTS`. **Form encoding caveat**: URL builder uses `+` for spaces in `q` (e.g. `drip+coffee+maker`); `parse_qs` in the evaluator decodes `+` back to space, so the literal `expected.q` value uses spaces (`"drip coffee maker"`). Mismatch between URL form and expected literal is a common bug — `_j2_url` handles the encoding.

### J3 compound cdjo + is_expected_url_pattern_match (1 base × 4 = 4 rows)

**Eval base**: `368d9ba4` (Manchester weather forecast — eval is `[url_dashPart with rule_relativeTime, active_url_from_accessTree with rule]`).

**Strategy**: build a stable URL that simultaneously satisfies a cdjo check on URL query params AND a regex match on the URL path. Oracle URL: `https://en.wikipedia.org/wiki/{City}?city={city}&lang={lang}&section={section}`. Both:
- `active_tab_url_parse` extracts `city`, `lang`, `section` from query → matches literal expected
- `active_url_from_accessTree` returns the URL → regex matches `en\.wikipedia\.org/wiki/{City}`

Variants in `_J3_VARIANTS` (Berlin / Tokyo / Rome / Sydney with different sections). The `goto_prefix` is `https://` (not `https://www.`) since Wikipedia uses `en.` subdomain.

### Helpers (chrome.py)

- `_build_cdjo_evaluator(parse_keys, expected_dict, result_extra, expected_extra)` — builds a single `check_direct_json_object` evaluator using `active_tab_url_parse`. `result_extra` adds `replace`/`split_list`; `expected_extra` adds `ignore_list_order`/`expect_in_result`.
- `_build_cdjo_oracle(target_url)` — `[pkill chrome, launch with target_url, sleep 6]`. Mirrors synth `_filter_search_oracle` structure.
- `_build_compound_cdjo_cdjo_evaluator(p)` (J2) — two-element `func`/`result`/`expected` lists; both result getters are `active_tab_url_parse` on the same URL.
- `_build_compound_cdjo_pattern_evaluator(p)` (J3) — `[cdjo, is_expected_url_pattern_match]`; second result getter is `active_url_from_accessTree`.

### J archetype skipped cdjo bases

| tid | reason |
|---|---|
| `2888b4e6` | `url_path_parse` is Macy-specific (`get_macys_product_url_parse` reads `mens-clothing` / `Top_style` / `Sleeve_length` — too specialized for stable URL synthesis) |
| `6c4c23a1`, `9f3f70fc`, `cabb3bae`, `b4f95342`, `fc6d8143` | `active_tab_html_parse` requires a live page DOM with specific class hierarchy — no stable offline oracle |
| `b7895e80`, `da46d875`, `47543840` | compound forms where ≥2 entries use `active_tab_html_parse` |
| `1704f00f` | compound 2× cdjo but second uses `rule_relativeTime` — would need date computation at oracle time |

---

## Per-task Plan

> **Keep in sync with code.** Every change to value pool or resampling logic must be reflected in `perturb/chrome.py` immediately, and vice versa. The table and the code are the single joint source of truth — divergence silently produces wrong training tasks.

### Perturbable tasks

| tid | eval func / result_type | eval value | TYPE_1 resampling space | rows/task |
|---|---|---|---|---|
| `030eeff7` | `exact_match` / `enable_do_not_track` | `true` | structural TYPE_1: skip_false → 0 rows; **Archetype P (`dnt`)**: 4 paraphrased rows | 4 |
| `9656a811` | `exact_match` / `enable_safe_browsing` | `true` | structural TYPE_1: skip_false → 0 rows; **Archetype P (`safe_browsing`)**: 4 paraphrased rows | 4 |
| `99146c54` | `exact_match` / `data_delete_automacally` | `true` | structural TYPE_1: `skip_false` blocks the false direction; Archetype P `clear_on_exit` is dropped in code as agent-ceiling/UI-unreachable | 0 |
| `93eabf48` | `match_in_list` / `chrome_color_scheme` | `["light","system"]` | **Archetype P (`color_scheme`)**: 4 paraphrased rows (perturb to light, seed dark) | 4 |
| `2ae9ba84` | `exact_match` / `profile_name` ("Thomas") | "Thomas" | structural: 17-name pool excluding "Thomas" → 4 rows | 4 |
| `af630914` | `check_font_size` | oracle writes 24px; eval range min=16 | both excluded: candidates are 9/12/20px, sample up to 3 | 3 |
| `bb5e4c0d` | `match_in_list` / `default_search_engine` | "Bing" | 6 other engines: Google/DuckDuckGo/Yahoo/Ecosia/Brave/Startpage, sample up to 4 | 4 |
| `2ad9387a` | `is_expected_bookmarks` / folder | "Favorites" | `perturb_bookmark_folder` emits 4 folder-name variants; there is no current global Tier-list block | 4 |
| `7a5a7856` | `is_expected_bookmarks` / `bookmark_bar_websites_urls` | `https://jalammar.github.io/...` | **Archetype B**: 4 target URLs, each with optional distractor extra tab | 4 |
| `7b6c7e24` | `is_cookie_deleted` / `.amazon.com` | `.amazon.com` | dropped from `_INTERNAL_FNS`; raw SQLite cookie seeds are not visible in Chrome's UI | 0 |
| `3299584d` | `exact_match` / `new_startup_page` | funbrain.com → NTP | 7 other nuisance URLs, sample up to 4 | 4 |
| `44ee5668` | `check_history_deleted` / `youtube` | youtube | 7 other keywords: facebook/twitter/reddit/amazon/instagram/linkedin/pinterest, sample up to 4 | 4 |
| `35253b65` | `is_shortcut_on_desktop` | "Play Puzzle Game 2048" | 7 other names, sample up to 4 | 4 |
| `59155008` | `is_expected_active_tab` / `active_url_from_accessTree` | babycenter URL | **Archetype T1 (active_tab)**: 4 target URLs | 4 |
| `a96b564e` | `is_expected_active_tab` / `active_tab_info` | flightaware URL | **Archetype T1 (active_tab)**: 4 target URLs | 4 |
| `f0b971a1` | `is_expected_active_tab` / `active_url_from_accessTree` | nfl URL | **Archetype T1 (active_tab)**: 4 target URLs | 4 |
| `0d8b7de3` | `is_expected_active_tab` (compound `or`) | drugs.com URLs | **Archetype T1 (active_tab)**: 4 target URLs (uses base evaluator unchanged in shape; only swaps `expected.rules.url`) | 4 |
| `9f935cce` | `is_expected_url_pattern_match` | justice.gov regex | **Archetype T1 (url_pattern)**: 4 (pattern, url, description) variants | 4 |
| `a728a36e` | `is_expected_url_pattern_match` | dmv.virginia.gov regex | **Archetype T1 (url_pattern)**: 4 variants | 4 |
| `c1fa57f3` | `is_expected_url_pattern_match` | united.com regex | **Archetype T1 (url_pattern)**: 4 variants | 4 |
| `f3b19d1e` | `is_expected_url_pattern_match` | ticketek regex | **Archetype T1 (url_pattern)**: 4 variants | 4 |
| `06fe7178` | `is_expected_tabs` / `open_tabs_info` | 3-URL set | **Archetype T2 (open_tabs)**: 4 different 3-URL sets, seed first 2, agent opens 3rd | 4 |

> ⚠️ **DNT and SafeBrowsing (`030eeff7`, `9656a811`)**: structural perturb's `skip_false=True` blocks the value-flip variant. Archetype P now adds 4 paraphrased rows per base where the eval target is unchanged but the natural-language instruction varies — the seed flips initial state to OPPOSITE so the agent must do real work.
>
> ⚠️ **`99146c54` (data_delete_automacally)**: structural perturb emits 0 rows because `skip_false` blocks the disable variant; Archetype P's `clear_on_exit` axis was removed after validation showed the Chrome UI cannot flip the enterprise-policy-shaped setting.
>
> ℹ️ **`2ad9387a` (bookmark_bar_folders_names)**: current code has no old global Tier-list framework gate; `perturb_bookmark_folder` emits the four folder-name rows directly.
>
> ℹ️ **`7a5a7856` (bookmark URL)**: same eval `func` (`is_expected_bookmarks`) but in `bookmark_bar_websites_urls` mode — Archetype B handles this case directly with a Bookmarks-JSON oracle (no Tier-3 framework dependency).
>
> ℹ️ **`af630914` (font_size)**: excludes **two** values from the pool: `orig_min=16` (the range evaluator's min threshold) and `24` (the eval oracle's actual written value). Without excluding the oracle value, perturb generates 24px which passes the eval's range check (24 > 16) — genuine leakage. Fixed in `_perturb_font_size` by also reading the oracle's `"default_font_size"` field and excluding it.

### Not Perturbable

| tid | Reason |
|---|---|
| `12086550` | `is_expected_active_tab_approximate` — fuzzy accessibility-tree state, not stable enough for synthetic targeting |
| `121ba48f` | `is_added_to_steam_cart` — live web interaction, requires real Steam state |
| `1704f00f` | `check_direct_json_object` (compound) — live search result; second cdjo uses `rule_relativeTime` |
| `2888b4e6` | `check_direct_json_object` — `url_path_parse` is Macy-specific |
| `368d9ba4` | `check_direct_json_object` (compound) — live weather result. **Used as topical seed by Archetype J3** (cdjo+url_pattern, 4 rows) |
| `3720f614` | `infeasible` — marked infeasible in eval |
| `47543840` | `is_expected_url_pattern_match` (compound) — live rental search |
| `480bcfea` | `infeasible` — Chrome UI rollback not possible |
| `6766f2b8` | `is_in_list` — file unzip task, not a Chrome navigation task |
| `6c4c23a1` | `check_direct_json_object` — live flight search; result getter is `active_tab_html_parse` |
| `7f52cab9` | `check_direct_json_object` (compound) — live shopping result. **Used as topical seed by Archetype J2** (cdjo+cdjo, 4 rows) |
| `82279c77` | `check_direct_json_object` — live car search. **Dropped from J1a**: cars.com URL params are live-site fragile |
| `82bc8d6a` | `check_direct_json_object` — live flight search. **Dropped from J1c**: kayak URL params carry literal-date drift |
| `9f3f70fc` | `check_direct_json_object` — live shopping result; result getter is `active_tab_html_parse` |
| `ae78f875` | `infeasible` — Google results-per-page not UI-changeable |
| `b070486d` | `is_expected_url_pattern_match` (compound) — live medical search |
| `b4f95342` | `check_direct_json_object` — live booking result |
| `b7895e80` | `check_direct_json_object` (compound) — live hotel search |
| `cabb3bae` | `check_direct_json_object` — live shopping result; result getter is `active_tab_html_parse` |
| `da46d875` | `is_expected_url_pattern_match` (compound) — live transit booking |
| `e1e75309` | `compare_pdfs` — print-to-PDF, file content depends on live page rendering |
| `f5d96daf` | `check_direct_json_object` — live product comparison. **Used as topical seed by Archetype J1b** (apple compare URL, 4 rows) |
| `f79439ad` | `check_direct_json_object` — live flight search. **Dropped from J1d**: kiwi URL params carry literal-date drift |
| `fc6d8143` | `check_direct_json_object` — live flight search; result getter is `active_tab_html_parse` |
| `clear_history_*` group | `check_history_deleted` / synth tasks — synth, not eval-base |

---

## V4a Setting Coverage Check

Three checks: **(1)** all required setting types appear in perturb or have a current-code skip reason (`skip_false`, dropped archetype, or unhandled/de-scoped branch — see Per-task Plan); **(2)** rows-per-task count distribution; **(3)** each setting type's share of total rows.

> `knob_assignment` is not stored in output rows. Setting type is derived by looking up the base eval task via `metadata.others.source`.

```python
"""V4a setting coverage check — chrome.
Run from repo root: uv run python this_script.py
"""
import json, re
from collections import Counter, defaultdict

all_eval = {r["task_id"]: r for r in (json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/eval.jsonl"))}
all_rows = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
rows = [r for r in all_rows if "_chrome_" in r["task_id"]]

def _setting_type(row):
    src = row["metadata"]["others"].get("source", "")
    base_tid = src.replace("perturb:", "")
    ev = all_eval.get(base_tid)
    if not ev:
        return "unknown"
    func = ev["metadata"]["evaluator"].get("func", "")
    result = ev["metadata"]["evaluator"].get("result", {})
    rt = result.get("type", "") if isinstance(result, dict) else ""
    if func == "exact_match" and rt in ("enable_do_not_track", "enable_safe_browsing", "data_delete_automacally"):
        return "bool_setting"
    if func == "exact_match" and rt == "profile_name":
        return "profile_name"
    if func == "check_font_size":
        return "font_size"
    if func == "match_in_list" and rt == "default_search_engine":
        return "search_engine"
    if func == "is_cookie_deleted":
        return "cookie_domain"
    if func == "exact_match" and rt == "new_startup_page":
        return "startup_url"
    if func == "check_history_deleted":
        return "history_keyword"
    if func == "is_shortcut_on_desktop":
        return "shortcut_name"
    return f"{func}/{rt}"

# bool_setting, 2ad9387a, 99146c54, DNT/SB all produce 0 rows by design
REQUIRED_SETTINGS = {
    "profile_name",
    "font_size",
    "search_engine",
    "cookie_domain",
    "startup_url",
    "history_keyword",
    "shortcut_name",
}

setting_counts = Counter(_setting_type(r) for r in rows)

missing = REQUIRED_SETTINGS - set(setting_counts.keys())
print(f"[{'FAIL' if missing else 'OK  '}] missing settings: {missing or 'none'}")
print()

total = len(rows)
print(f"Total chrome perturb rows: {total}")
print(f"  {'setting':<22}  {'count':>6}  {'share':>7}")
for k in sorted(REQUIRED_SETTINGS | set(setting_counts.keys())):
    cnt = setting_counts.get(k, 0)
    flag = " <-- MISSING" if cnt == 0 else ""
    print(f"  {k:<22}  {cnt:>6}  {cnt/total:>6.1%}{flag}")

# Rows per source task
print()
by_source = defaultdict(int)
for r in rows:
    src = re.sub(r"_[0-9a-f]{8}$", "", r["task_id"])
    by_source[src] += 1
dist = Counter(by_source.values())
print("Rows per source task distribution:")
for n, cnt in sorted(dist.items()):
    print(f"  {n} rows: {cnt} source tasks")
```

Targets (existing TYPE_1 only — Archetypes B/P/T/J are scored under their own keys via the dispatcher fns):
- Active TYPE_1 sources present; `cookie_domain` and `data_delete_automacally`
  are documented as 0-row drops rather than counted as missing defects
- `profile_name` produces 4 rows
- `font_size` produces 3 rows (excludes range-min=16px AND eval oracle=24px)
- `bookmark_folder`, `search_engine`, `startup_page`, `history_keyword`, and
  `shortcut_name` produce exactly 4 rows each
- TYPE_1 sub-total: **27 rows** (6×4 + 3)

Archetype B/P/T/J row counts (verify via task_id source-base lookup):
- Archetype B → 4 rows (base `7a5a7856`)
- Archetype P → 12 rows (3 bases × 4)
- Archetype T1 → 32 rows (8 bases × 4)
- Archetype T2 → 4 rows (base `06fe7178`)
- Archetype J1 → 4 rows (base `f5d96daf`)
- Archetype J2 → 4 rows (base `7f52cab9`)
- Archetype J3 → 4 rows (base `368d9ba4`)
- Grand total: **91 rows** (27 TYPE_1 + 52 B/P/T + 12 J), current generated JSONL count

---

## V4b Perturb-Eval Match Verification

### Part A: Instruction Clarity

Manual inspection: sample ~20 rows and verify each instruction accurately and unambiguously describes the oracle action.

```python
import json, random

rows = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
chrome = [r for r in rows if "_chrome_" in r["task_id"]]

def _oracle_cmd(r):
    for a in (r["metadata"].get("others") or {}).get("oracle_actions", []):
        cmd = a.get("parameters", {}).get("command", "")
        if isinstance(cmd, str) and cmd.strip():
            return cmd[:200]
    return ""

def _config_cmd(r):
    for s in r["metadata"].get("config", []):
        if s.get("type") == "execute":
            cmd = s.get("parameters", {}).get("command", "")
            if isinstance(cmd, str) and "perturb" in cmd.lower():
                return cmd[:200]
    return ""

rng = random.Random(0)
for r in rng.sample(chrome, min(20, len(chrome))):
    knob = r.get("knob_assignment", {})
    print(f"[{r['task_id'].split('_chrome_')[-1].split('_')[0]}]  knob={knob}")
    print(f"  INSTR : {r['instruction']}")
    print(f"  ORACLE: {_oracle_cmd(r)}")
    print()
```

What to look for per row:

| Check | What to verify |
|---|---|
| Setting type | Instruction verb/noun matches setting (e.g., "profile name" ↔ `"name":` in Preferences) |
| Target value | Value named in instruction matches oracle (e.g., "DuckDuckGo" ↔ `"short_name": "DuckDuckGo"`) |
| Boolean direction | "enable" ↔ `True`, "disable" ↔ `False` / key removed |
| Numeric match | Font size label + px in instruction ↔ `"default_font_size": N` in oracle |
| Domain | Cookie domain in instruction ↔ `host_key LIKE '%domain%'` in oracle |
| Keyword | History keyword in instruction ↔ `url LIKE '%keyword%'` in oracle |
| No save leak | Instruction must not say "save the file" |
| Grammar | Natural English, no broken phrasing |

### Part B: Feasibility

For each perturbable task, verify the oracle can write the expected value and Chrome will read it:

```python
"""V4b Part B — feasibility check (chrome).
Run from repo root: uv run python this_script.py
"""
import json, re

# knob_assignment is not stored in rows; extract target values from oracle text.
# Patterns for each setting type:
_FONT_SIZE_PAT   = re.compile(r'"default_font_size":\s*(\d+)')
_ENGINE_NAMES    = ["Google", "Microsoft Bing", "DuckDuckGo", "Yahoo", "Ecosia", "Brave", "Startpage"]
_COOKIE_DOMAINS  = [".amazon.com", ".google.com", ".facebook.com", ".twitter.com",
                    ".reddit.com", ".youtube.com", ".linkedin.com"]
_HISTORY_KWS     = ["youtube", "facebook", "twitter", "reddit", "amazon", "instagram", "linkedin", "pinterest"]
_SHORTCUT_NAMES  = ["Play Puzzle Game 2048", "Google Maps", "Wikipedia",
                    "YouTube Music", "Weather App", "Calculator Online", "Gmail Inbox", "Translate"]
_STARTUP_URLS    = ["funbrain.com", "fakegamingsite.net", "annoyingads.example",
                    "popups.local", "homestartup.dev", "mybadhomepage.io",
                    "spammy-default.test", "weirdstartpage.com"]

all_eval  = {r["task_id"]: r for r in (json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/eval.jsonl"))}
all_perturb = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
chrome_perturb = [r for r in all_perturb if "_chrome_" in r["task_id"]]

def _oracle_text(r):
    return "\n".join(
        a.get("parameters", {}).get("command", "")
        for a in (r["metadata"].get("others") or {}).get("oracle_actions", [])
        if isinstance(a.get("parameters", {}).get("command"), str)
    )

def _setting_type(row):
    src = row["metadata"]["others"].get("source", "")
    base_tid = src.replace("perturb:", "")
    ev = all_eval.get(base_tid)
    if not ev:
        return "unknown"
    func = ev["metadata"]["evaluator"].get("func", "")
    result = ev["metadata"]["evaluator"].get("result", {})
    rt = result.get("type", "") if isinstance(result, dict) else ""
    if func == "check_font_size":      return "font_size"
    if func == "match_in_list":        return "search_engine"
    if func == "is_cookie_deleted":    return "cookie_domain"
    if func == "check_history_deleted":return "history_keyword"
    if func == "exact_match" and rt == "new_startup_page": return "startup_url"
    if func == "is_shortcut_on_desktop": return "shortcut_name"
    return f"{func}/{rt}"

issues = []

for r in chrome_perturb:
    oracle = _oracle_text(r)
    stype  = _setting_type(r)

    # font_size: oracle must write "default_font_size": N where N is a valid discrete size
    if stype == "font_size":
        m = _FONT_SIZE_PAT.search(oracle)
        if not m:
            issues.append((r["task_id"], "font_size oracle missing default_font_size=N"))
        elif int(m.group(1)) not in (9, 12, 16, 20, 24):
            issues.append((r["task_id"], f"font_size oracle has invalid px={m.group(1)}"))

    # search_engine: oracle must contain one of the known engine names
    elif stype == "search_engine":
        if not any(e in oracle for e in _ENGINE_NAMES):
            issues.append((r["task_id"], "search engine name not found in oracle"))

    # cookie_domain: oracle must contain a cookie domain
    elif stype == "cookie_domain":
        if not any(d.lstrip(".") in oracle for d in _COOKIE_DOMAINS):
            issues.append((r["task_id"], "no cookie domain found in oracle"))

    # history_keyword: oracle must contain a history keyword
    elif stype == "history_keyword":
        if not any(kw in oracle for kw in _HISTORY_KWS):
            issues.append((r["task_id"], "no history keyword found in oracle"))

    # startup_url: oracle must set restore_on_startup = 5
    elif stype == "startup_url":
        if "restore_on_startup" not in oracle:
            issues.append((r["task_id"], "startup oracle missing restore_on_startup"))

    # shortcut_name: oracle must mention a shortcut name
    elif stype == "shortcut_name":
        if not any(n in oracle for n in _SHORTCUT_NAMES):
            issues.append((r["task_id"], "no shortcut name found in oracle"))

print(f"[{'FAIL' if issues else 'OK  '}] feasibility: {len(issues)} issues")
for tid, reason in issues:
    print(f"  {tid}: {reason}")
```

Manual checks per setting type:

| setting type | What to verify |
|---|---|
| boolean prefs | `perturb_config_step` kills Chrome before writing → Chrome can't overwrite our change on next Preferences flush |
| `profile_name` | Evaluator uses case-insensitive grep (`grep -ic '"name".*"name"'`), not exact_match, so agent typing lowercase still passes |
| `check_font_size` | Oracle uses exact `{"type": "value", "value": N}` rule, not range — discrete dropdown values only |
| `default_search_engine` | `perturb_config_step` seeds BASE engine + relaunches with `--user-data-dir=/home/user/chrome-data`; evaluator's `postconfig` does `pkill chrome; sleep 8` to flush Preferences before reading |
| `is_cookie_deleted` | seed INSERT uses the full 20-column schema for Chrome's Cookies DB (cycle 28b). Cycle 32 (rollout audit): `is_secure=1, is_httponly=1, samesite=0, source_scheme=1, source_port=443` to mirror the working amazon-seed values — Chrome 147 silently rejects `samesite=-1, source_scheme=0` rows on load, so the seeded cookie was invisible to the agent's UI delete |
| `check_history_deleted` | Both `new_keyword` entries and base youtube entries are seeded; agent must delete only the new keyword rows |
| `new_startup_page` | `perturb_config_step` appends the nuisance URL write after the base config so it survives into first launch |

### Part C: Distribution Match

**Invariant — pool-level, not sampling-level.** The value pool for each setting type must cover the eval's value space. With only 1 eval task per setting type, "distribution match" means the perturb pool spans all plausible values an agent might encounter, not that perturb frequencies precisely match eval.

Three checks: **(1)** all non-leaking pool values are reachable (no gaps in the candidate list); **(2)** no single value dominates (pool is roughly uniform); **(3)** coverage of allowed edge values (font size keeps 9px and intentionally excludes the eval-leaking 24px).

```python
"""V4b Part C — distribution match (chrome).
Run from repo root: uv run python this_script.py
"""
import json, re
from collections import Counter, defaultdict

# knob_assignment is not stored in rows; extract values from oracle text.
# For distribution checks, we verify rows-per-task and oracle value diversity.
_FONT_SIZE_PAT = re.compile(r'"default_font_size":\s*(\d+)')
_ENGINE_PAT    = re.compile(r'"short_name":\s*"([^"]+)"')
_COOKIE_PAT    = re.compile(r'host_key.*?[\'"]\.?([a-z]+\.[a-z]+)[\'"]', re.S)
_HISTORY_PAT   = re.compile(r"(?:url LIKE|LIKE)\s*'%([a-z]+)%'")
_STARTUP_PAT   = re.compile(r"(?:startup_urls|nuisance).*?['\"]([a-z0-9.-]+)['\"]", re.S)
_SHORTCUT_PAT  = re.compile(r'Name=([^\n]+)')

all_eval  = {r["task_id"]: r for r in (json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/eval.jsonl"))}
all_perturb = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
chrome_perturb = [r for r in all_perturb if "_chrome_" in r["task_id"]]

def _setting_type(row):
    src = row["metadata"]["others"].get("source", "")
    ev = all_eval.get(src.replace("perturb:", ""))
    if not ev:
        return "unknown"
    func = ev["metadata"]["evaluator"].get("func", "")
    rt = (ev["metadata"]["evaluator"].get("result", {}) or {}).get("type", "")
    if func == "check_font_size":      return "font_size"
    if func == "match_in_list":        return "search_engine"
    if func == "is_cookie_deleted":    return "cookie_domain"
    if func == "check_history_deleted":return "history_keyword"
    if func == "exact_match" and rt == "new_startup_page": return "startup_url"
    if func == "is_shortcut_on_desktop": return "shortcut_name"
    if func == "exact_match" and rt == "profile_name": return "profile_name"
    return f"{func}/{rt}"

def _oracle_text(r):
    return "\n".join(
        a.get("parameters", {}).get("command", "")
        for a in (r["metadata"].get("others") or {}).get("oracle_actions", [])
        if isinstance(a.get("parameters", {}).get("command"), str)
    )

# C1: rows per source task
by_source: dict[str, list] = defaultdict(list)
for r in chrome_perturb:
    src = re.sub(r"_[0-9a-f]{8}$", "", r["task_id"])
    by_source[src].append(r)

print("C1: Rows per source task:")
for src, rows in sorted(by_source.items()):
    stype = _setting_type(rows[0]) if rows else "?"
    print(f"  {src.split('_chrome_')[-1]} ({stype}): {len(rows)} rows")

# C2/C3: value diversity per setting type (via oracle text)
by_type: dict[str, list] = defaultdict(list)
for r in chrome_perturb:
    by_type[_setting_type(r)].append(r)

print()
print("C2/C3: Value diversity per setting type (extracted from oracle):")
for stype, rows in sorted(by_type.items()):
    oracle_vals = []
    for r in rows:
        oracle = _oracle_text(r)
        if stype == "font_size":
            m = _FONT_SIZE_PAT.search(oracle)
            oracle_vals.append(m.group(1) if m else "?")
        elif stype == "search_engine":
            m = _ENGINE_PAT.search(oracle)
            oracle_vals.append(m.group(1) if m else "?")
        elif stype == "history_keyword":
            m = _HISTORY_PAT.search(oracle)
            oracle_vals.append(m.group(1) if m else "?")
        elif stype == "shortcut_name":
            m = _SHORTCUT_PAT.search(oracle)
            oracle_vals.append(m.group(1).strip() if m else "?")
    ctr = Counter(oracle_vals)
    total = len(oracle_vals) or 1
    print(f"\n  {stype} (n={len(rows)}):")
    for v, cnt in sorted(ctr.items(), key=lambda x: -x[1]):
        flag = "  ← HIGH (>40%)" if cnt / total > 0.4 else ""
        print(f"    {str(v):<30} {cnt:>4} ({cnt/total:>5.1%}){flag}")

# C3: Font size extremes
print()
print("C3: Font size candidates (9px must appear; 24px must not leak)")
font_oracles = [_oracle_text(r) for r in by_type.get("font_size", [])]
font_pxs = [int(m.group(1)) for o in font_oracles if (m := _FONT_SIZE_PAT.search(o))]
print(f"  sizes drawn: {sorted(set(font_pxs))}")
print(f"  9px:  {'YES' if 9  in font_pxs else 'NO ← extreme not covered'}")
print(f"  24px: {'NO ← correctly excluded' if 24 not in font_pxs else 'YES ← leakage'}")
```

Targets:
- C1: `profile_name` produces 4 rows; all other active setting types produce 4 rows each
- C2: No single value exceeds 40% share within its setting type
- C3: Font size includes 9px and intentionally excludes 24px (the eval oracle value)

---

## V4c Eval Leakage Check

Verify no perturb row is functionally identical to its source eval task.

### Guarantees by design

Each perturb function builds `candidates = [x for x in POOL if x != orig_value]` before sampling. This ensures:

- **profile_name** (`2ae9ba84`): eval=`"Thomas"` → perturb excludes `"Thomas"` → 17 candidates
- **font_size** (`af630914`): eval range min=`16`, eval oracle writes `24` → perturb excludes both → 3 candidates (9/12/20)
- **search_engine** (`bb5e4c0d`): eval=`"Bing"` (or `["Bing","Microsoft Bing"]`) → perturb excludes both → 6 candidates
- **bookmark_folder** (`2ad9387a`): eval folder name excluded from 12-name pool
- **cookie_domain** (`7b6c7e24`): dropped from `_INTERNAL_FNS`; raw SQLite cookie seeds are not visible in Chrome's UI
- **startup_page** (`3299584d`): eval nuisance URL is `funbrain.com`; perturb pool excludes it → 7 other URLs
- **history_keyword** (`44ee5668`): eval=`youtube` → 7 other keywords
- **desktop_shortcut** (`35253b65`): eval=`"Play Puzzle Game 2048"` → 7 other names
- **data_delete_automacally** (`99146c54`): eval=`true`, `skip_false=True`; Archetype P `clear_on_exit` is also dropped → 0 rows (no leakage possible)
- **DNT/SafeBrowsing** (`030eeff7`, `9656a811`): eval=`true`, `skip_false=True` → 0 rows (no leakage possible)

### V4c automated check

> `knob_assignment` is not stored in rows. Leakage is checked by comparing instruction and oracle text between each perturb row and its source eval row. For each setting type, the perturb row must differ in both instruction and oracle from the eval.

```python
"""V4c eval leakage check — chrome.
Run from repo root: uv run python this_script.py
"""
import json, re

all_eval    = {r["task_id"]: r for r in (json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/eval.jsonl"))}
all_perturb = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
chrome_perturb = [r for r in all_perturb if "_chrome_" in r["task_id"]]

_FONT_SIZE_PAT   = re.compile(r'"default_font_size":\s*(\d+)')
_ENGINE_PAT      = re.compile(r'"short_name":\s*"([^"]+)"')
_PROFILE_PAT     = re.compile(r'"name"[^:]*:[^"]*"([A-Za-z]+)"')
_COOKIE_DEL_PAT  = re.compile(r"host_key.*?['\"]\.?([a-z.]+)['\"]", re.S)
_HISTORY_DEL_PAT = re.compile(r"LIKE '%([a-z]+)%'")
_SHORTCUT_PAT    = re.compile(r'Name=([^\n]+)')
_NAMES_IN_INSTR  = re.compile(r'"([A-Za-z ]+)"')

def _oracle_text(r):
    return "\n".join(
        a.get("parameters", {}).get("command", "")
        for a in (r["metadata"].get("others") or {}).get("oracle_actions", [])
        if isinstance(a.get("parameters", {}).get("command"), str)
    )

def _eval_oracle_text(eval_row):
    return "\n".join(
        a.get("parameters", {}).get("command", "")
        for a in (eval_row["metadata"].get("others") or {}).get("oracle_actions", [])
        if isinstance(a.get("parameters", {}).get("command"), str)
    )

leakage = []

for r in chrome_perturb:
    src = r["metadata"]["others"].get("source", "")
    base_tid = src.replace("perturb:", "")
    eval_row = all_eval.get(base_tid)
    if eval_row is None:
        continue

    p_oracle  = _oracle_text(r)
    ev_oracle = _eval_oracle_text(eval_row)
    p_instr   = r["instruction"]
    ev_instr  = eval_row["instruction"]

    func   = eval_row["metadata"]["evaluator"].get("func", "")
    result = eval_row["metadata"]["evaluator"].get("result", {}) or {}
    rt     = result.get("type", "") if isinstance(result, dict) else ""

    # font_size: oracle px must differ from eval's px
    if func == "check_font_size":
        pm = _FONT_SIZE_PAT.search(p_oracle)
        em = _FONT_SIZE_PAT.search(ev_oracle)
        if pm and em and pm.group(1) == em.group(1):
            leakage.append((r["task_id"], f"font_size leakage: both use {pm.group(1)}px"))

    # search_engine: oracle engine must differ from eval's engine
    elif func == "match_in_list":
        pm = _ENGINE_PAT.search(p_oracle)
        em = _ENGINE_PAT.search(ev_oracle)
        if pm and em and pm.group(1).lower() == em.group(1).lower():
            leakage.append((r["task_id"], f"search_engine leakage: both use '{pm.group(1)}'"))

    # profile_name: instruction name must differ from eval's name
    elif func == "exact_match" and rt == "profile_name":
        ev_expected = eval_row["metadata"]["evaluator"].get("expected", {}).get("rules", {}).get("expected", "")
        # Any quoted name in perturb instruction that matches eval's expected name is leakage
        for m in _NAMES_IN_INSTR.finditer(p_instr):
            if m.group(1).lower() == ev_expected.lower():
                leakage.append((r["task_id"], f"profile_name leakage: '{m.group(1)}' == eval expected '{ev_expected}'"))
                break

    # cookie_domain: oracle domain must differ from eval's
    elif func == "is_cookie_deleted":
        ev_domains = eval_row["metadata"]["evaluator"].get("expected", {}).get("rules", {}).get("domains", [])
        if isinstance(eval_row["metadata"]["evaluator"].get("expected"), list):
            ev_domains = (eval_row["metadata"]["evaluator"]["expected"][0] or {}).get("rules", {}).get("domains", [])
        pm = _COOKIE_DEL_PAT.search(p_oracle)
        if pm:
            p_dom = pm.group(1).lstrip(".")
            for ev_dom in ev_domains:
                if p_dom == ev_dom.lstrip("."):
                    leakage.append((r["task_id"], f"cookie_domain leakage: both use '{p_dom}'"))

    # history_keyword: oracle keyword must differ from eval's
    elif func == "check_history_deleted":
        ev_kws = eval_row["metadata"]["evaluator"].get("expected", {}).get("rules", {}).get("keywords", [])
        if isinstance(eval_row["metadata"]["evaluator"].get("expected"), list):
            ev_kws = (eval_row["metadata"]["evaluator"]["expected"][0] or {}).get("rules", {}).get("keywords", [])
        pm = _HISTORY_DEL_PAT.search(p_oracle)
        if pm and pm.group(1) in ev_kws:
            leakage.append((r["task_id"], f"history_keyword leakage: oracle deletes eval keyword '{pm.group(1)}'"))

    # shortcut_name: oracle Name= must differ from eval's name
    elif func == "is_shortcut_on_desktop":
        ev_name = eval_row["metadata"]["evaluator"].get("expected", {}).get("rules", {}).get("name", "")
        if isinstance(eval_row["metadata"]["evaluator"].get("expected"), list):
            ev_name = (eval_row["metadata"]["evaluator"]["expected"][0] or {}).get("rules", {}).get("name", "")
        pm = _SHORTCUT_PAT.search(p_oracle)
        if pm and pm.group(1).strip().lower() == ev_name.lower():
            leakage.append((r["task_id"], f"shortcut_name leakage: both use '{ev_name}'"))

    # General: instruction must not be identical to eval instruction
    if p_instr.strip() == ev_instr.strip():
        leakage.append((r["task_id"], "instruction identical to eval (no change applied)"))

print(f"[{'FAIL' if leakage else 'OK  '}] eval leakage: {len(leakage)} violations")
for tid, reason in leakage[:10]:
    print(f"  {tid}: {reason}")
```

---

## V4d Inter-Variant Uniqueness

Within the N variants from the same source eval task, every row must be distinct in both instruction and oracle.

```python
"""V4d inter-variant uniqueness — chrome.
Run from repo root: uv run python this_script.py
"""
import json, re
from collections import defaultdict

all_perturb = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
chrome_perturb = [r for r in all_perturb if "_chrome_" in r["task_id"]]

def _oracle_text(r):
    return "\n".join(
        a.get("parameters", {}).get("command", "")
        for a in (r["metadata"].get("others") or {}).get("oracle_actions", [])
        if isinstance(a.get("parameters", {}).get("command"), str)
    )

by_source = defaultdict(list)
for r in chrome_perturb:
    src = re.sub(r"_[0-9a-f]{8}$", "", r["task_id"])
    by_source[src].append(r)

instr_dups  = []
oracle_dups = []
knob_dups   = []

for src, rows in by_source.items():
    instrs  = [r["instruction"] for r in rows]
    oracles = [_oracle_text(r) for r in rows]
    knobs   = [str(r.get("knob_assignment", {})) for r in rows]

    if len(instrs) != len(set(instrs)):
        instr_dups.append((src, instrs))
    if len(oracles) != len(set(oracles)):
        oracle_dups.append((src, oracles))
    if len(knobs) != len(set(knobs)):
        knob_dups.append((src, knobs))

print(f"[{'FAIL' if instr_dups  else 'OK  '}] duplicate instructions:     {len(instr_dups)} source tasks")
print(f"[{'FAIL' if oracle_dups else 'OK  '}] duplicate oracle code:      {len(oracle_dups)} source tasks")
print(f"[{'FAIL' if knob_dups   else 'OK  '}] duplicate knob assignments: {len(knob_dups)} source tasks")
for src, rows in knob_dups[:3]:
    print(f"  {src}:")
    for k in rows: print(f"    {k}")
```

Note: `random.sample` (used in all perturb functions) guarantees no duplicate values within a task's generated rows. Duplicate instructions can still occur if instruction templates coincidentally produce the same string for different values — the check above catches this.

---

## V4e Instruction-Oracle Value Consistency

The instruction names the concrete target value; the oracle code must agree.

```python
"""V4e instruction-oracle value consistency — chrome.
Run from repo root: uv run python this_script.py
"""
import json, re

# knob_assignment is not stored in rows; extract the target value from oracle text
# and verify it also appears in the instruction.
_FONT_SIZE_PAT = re.compile(r'"default_font_size":\s*(\d+)')
_ENGINE_PAT    = re.compile(r'"short_name":\s*"([^"]+)"')
_COOKIE_DEL_PAT= re.compile(r"LIKE\s+'%([a-z.]+)%'")
_HISTORY_DEL_PAT=re.compile(r"LIKE '%([a-z]+)%'")
_SHORTCUT_PAT  = re.compile(r'Name=([^\n]+)')

_FONT_LABELS = {9: "Very small", 12: "Small", 16: "Medium", 20: "Large", 24: "Very large"}

all_eval  = {r["task_id"]: r for r in (json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/eval.jsonl"))}
all_perturb = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
chrome_perturb = [r for r in all_perturb if "_chrome_" in r["task_id"]]

def _oracle_text(r):
    return "\n".join(
        a.get("parameters", {}).get("command", "")
        for a in (r["metadata"].get("others") or {}).get("oracle_actions", [])
        if isinstance(a.get("parameters", {}).get("command"), str)
    )

def _setting_type(row):
    src = row["metadata"]["others"].get("source", "")
    ev = all_eval.get(src.replace("perturb:", ""))
    if not ev:
        return "unknown"
    func = ev["metadata"]["evaluator"].get("func", "")
    rt   = (ev["metadata"]["evaluator"].get("result", {}) or {}).get("type", "")
    if func == "check_font_size":       return "font_size"
    if func == "match_in_list":         return "search_engine"
    if func == "exact_match" and rt == "profile_name": return "profile_name"
    if func == "is_cookie_deleted":     return "cookie_domain"
    if func == "check_history_deleted": return "history_keyword"
    if func == "exact_match" and rt == "new_startup_page": return "startup_url"
    if func == "is_shortcut_on_desktop": return "shortcut_name"
    return f"{func}/{rt}"

def _check(r):
    instr  = r["instruction"]
    oracle = _oracle_text(r)
    stype  = _setting_type(r)
    errors = []

    # font_size: oracle → px; instruction must mention label + px
    if stype == "font_size":
        m = _FONT_SIZE_PAT.search(oracle)
        if not m:
            errors.append("default_font_size not in oracle")
        else:
            px = int(m.group(1))
            label = _FONT_LABELS.get(px, "")
            if label and label not in instr:
                errors.append(f"font label '{label}' not in instruction (px={px})")
            if str(px) not in instr:
                errors.append(f"font px '{px}' not in instruction")

    # search_engine: oracle → engine name; instruction must match
    elif stype == "search_engine":
        m = _ENGINE_PAT.search(oracle)
        if not m:
            errors.append("engine short_name not in oracle")
        else:
            engine = m.group(1)
            bare = engine.replace("Microsoft ", "")
            if engine not in instr and bare not in instr:
                errors.append(f"engine '{engine}' not in instruction")

    # profile_name: instruction has name in quotes; oracle must write same name
    elif stype == "profile_name":
        # Extract quoted names from instruction — at least one must appear in oracle
        names_in_instr = re.findall(r'"([A-Za-z]+)"', instr)
        if not names_in_instr:
            errors.append("no quoted name found in instruction")
        else:
            if not any(n in oracle for n in names_in_instr):
                errors.append(f"none of instruction names {names_in_instr} found in oracle")

    # cookie_domain: oracle DELETE LIKE '%domain%'; instruction must mention domain
    elif stype == "cookie_domain":
        m = _COOKIE_DEL_PAT.search(oracle)
        if not m:
            errors.append("no LIKE '%domain%' pattern in oracle")
        else:
            domain = m.group(1)
            if domain not in instr:
                errors.append(f"cookie domain '{domain}' not in instruction")

    # history_keyword: oracle DELETE LIKE '%kw%'; instruction must mention keyword
    elif stype == "history_keyword":
        m = _HISTORY_DEL_PAT.search(oracle)
        if not m:
            errors.append("no LIKE '%kw%' pattern in oracle")
        else:
            kw = m.group(1)
            if kw not in instr:
                errors.append(f"history keyword '{kw}' not in instruction")

    # startup_url: instruction mentions the nuisance URL; oracle sets restore_on_startup=5
    elif stype == "startup_url":
        if "restore_on_startup" not in oracle:
            errors.append("oracle missing restore_on_startup")
        # URL itself is in perturb_config_step seed, not oracle — instruction names it
        # No oracle↔instruction URL cross-check needed (asymmetric by design)

    # shortcut_name: oracle Name=...; instruction must mention same name
    elif stype == "shortcut_name":
        m = _SHORTCUT_PAT.search(oracle)
        if not m:
            errors.append("Name= not in oracle")
        else:
            name = m.group(1).strip()
            if name not in instr:
                errors.append(f"shortcut name '{name}' not in instruction")

    return errors

failures = [(r["task_id"], errs) for r in chrome_perturb if (errs := _check(r))]
print(f"[{'FAIL' if failures else 'OK  '}] instruction-oracle consistency: {len(failures)} rows with mismatches")
for tid, errs in failures[:10]:
    print(f"  {tid}:")
    for e in errs: print(f"    {e}")
```

Known acceptable exceptions:
- `startup_url`: the oracle sets `restore_on_startup=5` (NTP mode); the nuisance URL only appears in `perturb_config_step` (initial state seed), not in the oracle. The V4e check is relaxed for this setting type — the URL→oracle path is intentionally asymmetric.
- `profile_name`: these rows use quoted target names and the oracle writes the new name. The check extracts quoted names from the instruction — at least one should match the oracle write.

---

## Expected Output

Existing TYPE_1 perturbations:
- `profile_name` (`2ae9ba84`): **4 rows** (structural)
- `bookmark_bar_folders_names` (`2ad9387a`): **4 rows** (`perturb_bookmark_folder`; no current global Tier-3 block)
- `font_size` (`af630914`): **3 rows** (excludes range-min=16px and oracle=24px → candidates: 9/12/20)
- `search_engine`, `startup_page`, `history_keyword`, `shortcut_name` (4 bases): **4 rows each = 16 rows**
- `cookie_domain` (`7b6c7e24`): **0 rows** (dropped from `_INTERNAL_FNS`; raw SQLite cookie seeds are not visible in Chrome's UI)
- TYPE_1 sub-total: **27 rows** (current generated JSONL: 4+4+3+4+4+4+4)

New archetypes (2026-05-08, Phase 1 B/P/T):
- **Archetype B** (1 base × 4 variants): **4 rows**
- **Archetype P** (3 bases × 4 paraphrased instructions): **12 rows**
- **Archetype T1** (8 bases × 4 variants): **32 rows**
- **Archetype T2** (1 base × 4 variants): **4 rows**
- B/P/T sub-total: **52 rows**

New archetypes (2026-05-08, Phase 3 P3-2 J):
- **Archetype J1** (1 base × 4 variants): **4 rows** — single `check_direct_json_object` on `active_tab_url_parse`
- **Archetype J2** (1 base × 4 variants): **4 rows** — compound `cdjo + cdjo`
- **Archetype J3** (1 base × 4 variants): **4 rows** — compound `cdjo + is_expected_url_pattern_match`
- J sub-total: **12 rows**

**Total chrome perturb rows: 91** (27 existing TYPE_1 + 52 B/P/T + 12 J), matching the current byte-locked JSONL.

All perturb values differ from their source eval target where structural perturbation is used (`candidates = [x for x in POOL if x != orig_value]`). Archetype P intentionally keeps the eval target identical and only paraphrases the instruction — the OPPOSITE seed state ensures the agent must do real work; no leakage because eval target ≠ initial state.

V2 pass-rate target: **100%**
