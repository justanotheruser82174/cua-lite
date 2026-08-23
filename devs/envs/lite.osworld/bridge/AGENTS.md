# lite.osworld ↔ osworld bridge

**North star: lite.osworld ≈ osworld.** lite.osworld re-hosts the official OSWorld eval
split on our `sandbox.linux` container instead of the QEMU VM; the target is that the
*same task + same actions* be **behaviorally and visually indistinguishable** from the
official `osworld` VM — operationally, indistinguishable in **what the agent observes (the
screenshot stream) and the task outcome (reward)**; divergences invisible to both (audio,
clipboard, sub-frame timing) are out of scope unless they surface there. Everything here
measures the lite↔osworld divergence and drives it toward zero. Corollary (the §2 discipline): a change must move lite *toward* osworld —
never introduce a NEW divergence (even a "nicer" behavior osworld lacks), and never "fix"
a quirk osworld also has (that pushes lite further away).

Sections: **§1** isolated build · **§2** what-to-fix scope · **§3** cross-replay (the
parity measurement) · **§4** execution plan · **§5** gap tracker (the divergence ledger).

**Harness files in this dir** (run from repo root; the docker/KVM-free ones are smoke-tested):
- [`task_map.py`](/devs/envs/lite.osworld/bridge/task_map.py) — the lite↔osworld task-id join
  (full-UUID; 325 in-scope pairs after dropping the 44 `exclude_reason` rows; no docker). ✅ smoke-tested
- [`visual_diff.py`](/devs/envs/lite.osworld/bridge/visual_diff.py) — per-turn **SSIM (structure)
  + CIEDE2000 ΔE (color — SSIM is grayscale-blind to equal-luminance palette shifts)** + region
  masks + contact-sheet evidence (`lite/utils/image.py` has no diff util). ✅ smoke-tested
  (SSIM is grayscale-blind to color — see §Diff & adjudication protocol)
- [`cross_replay.py`](/devs/envs/lite.osworld/bridge/cross_replay.py) — bidirectional
  cross-substrate driver; reads the same `lite_message.tool_calls` as
  [`../validate/rollout/replay_trajectory.py`](/devs/envs/lite.osworld/validate/rollout/replay_trajectory.py)
  but via its own `_load_kept_turns` (keeps per-turn identity — deliberately NOT `_load_actions`, which
  discards it), adds the other-substrate target (env_key; a divergent lite image via `LITE_OSWORLD_CONFIG`,
  §1) + recorded-pace inter-turn timing + `visual_diff`. ✅ end-to-end validated — first real cross-substrate sweep run
  (16 same-substrate floors + 6 terminal-heavy pairs, both directions) surfaced GAP-02…05 (§Gaps).
  Needs a KVM host with **both** env images fresh (on a divergent branch, a §1 private build first).

---

# Setup

Goal: run a **divergent-source `lite.osworld`** on a Docker host where **another
session is already serving it**, without disturbing that session. (Freshness
mechanism: [/docs/envs.md#image-build-and-freshness](/docs/envs.md#image-build-and-freshness),
[`backend/freshness.py`](/lite/gym/utils/backend/freshness.py).)

## Why the default collides

Freshness rides on a **shared image tag**: the additive `cua-lite/lite.osworld:latest`
and its base `cua-lite/sandbox.linux:latest` are keyed by env, not by session/source.

- **Tag content is read live at each `docker run`.** DEDICATED spawns go through
  `docker_run_detached`, which takes a plain tag string and does **not** re-check
  freshness ([`docker.py`](/lite/gym/utils/backend/docker.py)). A running server passed its
  `ensure_services` gate once and cached it, so rebuilding `:latest` gives it **no
  error** — its next `reset()` silently launches your image. (Silent, worse than a stale-error.)
- **A stock build overwrites `:latest`.** `install.sh` / sandbox `install.sh` build
  the hardcoded `:latest` tag (sandbox `build` and lite.osworld `rebuild` also
  `docker image rm -f` it first); the base is shared with `lite.demo` / `lite.cuagym`,
  widening the blast radius.
- **Base divergence is silent.** The image freshness check reads the **additive** image's
  `lite.src_hash` label only — never which base it was `FROM`. An additive built on a
  stale base still passes.

## Isolated build (private tags)

`src_hash()` is tag-independent, so a private tag with the right label passes the
gate. Build both layers privately, point your run at the additive tag, serve on your
own port (containers are named `-<port>`). From repo root:

```bash
# 1) private base from YOUR sandbox sources → :mybase (never touches :latest)
DOCKER_BUILDKIT=1 docker build -t cua-lite/sandbox.linux:mybase \
  --label "lite.src_hash=$(uv run python -m lite.gym.utils.backend.freshness hash lite.demo)" \
  -f lite/gym/sandbox/docker/Dockerfile.linux lite/gym/sandbox

# 2) private additive FROM your base via --build-arg (not a Dockerfile edit, so H is
#    unchanged and the label stays truthful) → :mine
H=$(uv run python -m lite.gym.utils.backend.freshness hash lite.osworld)
DOCKER_BUILDKIT=1 docker build -t cua-lite/lite.osworld:mine \
  --label "lite.src_hash=$H" --build-arg BASE_TAG=mybase \
  -f lite/gym/envs/lite/osworld/docker/Dockerfile lite/gym/envs/lite/osworld
```

Point the run at `:mine` **via `LITE_OSWORLD_CONFIG`** — a whole-config override read
at import — **not** `--env-kwargs '{"image":...}'`. Why (validated this session): the
freshness gate keys on the module constant `_IMAGE = CFG.env_kwargs["computer"]["image"]`,
resolved at **import** from config; it's checked at the env-level `ensure_services` gate
(fires in direct mode too, *before* the instance exists) and again at `reset()`. A per-run
`--env-kwargs image` only reaches `self._image` → the container **spawn** (`sandbox/base.py:582`),
never the gate — so on a stale `:latest` the gate rejects it before it ever spawns.
`<PREFIX>_CONFIG` changes `_IMAGE` **itself**, so gate AND spawn both use `:mine`:

```bash
# whole-config override = env default.yaml with computer.image → :mine
sed 's#lite.osworld:latest#lite.osworld:mine#' \
  lite/gym/envs/lite/osworld/configs/default.yaml > /tmp/mine.yaml
export LITE_OSWORLD_CONFIG=/tmp/mine.yaml
uv run python scripts/rollout.py --model-id gpt-5.5 --env-id lite.osworld \
  --splits eval --head 1 --config-path scripts/configs/gpt/default/lite.osworld.yaml \
  --env-kwargs '{"max_steps":4}'          # then serve on your own port, or run direct
```

Validated end-to-end this way: build `:mine` → gpt-5.5 rollout (4 turns, container booted)
→ open-loop replay → per-turn SSIM ≈ 1.0 (same-substrate baseline), `:latest` untouched.

- Only `:mine` is `docker run` at runtime; `:mybase` is build-time only.
- Base **not** diverged? Skip step 1, pass `--build-arg BASE_TAG=latest`.
- **Never** run the stock `install.sh` / sandbox `install.sh` while the other session
  is live (see the collision above) — use `install.sh provision` for the docker-free
  deps/assets/catalogs, and the two private `docker build` lines above for the image.
- The gate checks `label == H` on the additive only; using the right base is on you
  (the `BASE_TAG` you pass).

---

# Eval-split migration fixes (issue [#157](https://github.com/cua-lite/cua-lite/issues/157))

`lite.osworld` EVAL-split visual audit (GPT-5.5, 30-step authoritative). Tracks
**only** bugs **our migration introduced** (OSWorld VM → our `sandbox.linux`
container). Upstream OSWorld defects are **out of scope** — note, don't fix.

## Scope — the decision test

> **Test 1 (what diverged):** *"Would this fail identically on the original OSWorld VM?"*
> Yes → UPSTREAM (**shared quirk — PRESERVE, don't fix**). Only fails on our container →
> MIGRATION (**fix — it's a divergence**).
> **Test 2 (parity ≠ quality):** a change must make lite behave **more like osworld** — never
> add behavior osworld lacks (even if "nicer"), never remove a quirk osworld has. Both push lite away.
> **Test 3 (MINIMAL container-compat only) — the governing principle:** the migrated **setup, postconfig,
> AND eval (evaluator func / getter / expected / result)** must stay **verbatim to the upstream OSWorld
> task JSON** unless the **container substrate FORCES a change** — and then make the **smallest** change that
> works. For every divergence ask: *"is this forced by the container? is it the minimal such change?"* If
> not forced → revert to upstream. Over-modification is itself a MIGRATION bug (it silently changes outcomes
> vs the VM). **Confirmed instance:** the postconfig save-segment was rewritten far past the forced minimum
> (§A) → the GAP-A clobber + stray-key + wrong-window risks. The audit lens applies to all three surfaces:
> diff lite vs `osworld_evaluation_examples/examples/<domain>/<uuid>.json` and justify every delta as forced+minimal.

**Three-surface divergence quantification (lite `eval.jsonl` vs upstream `osworld_evaluation_examples`, 369 tasks) —
keep-rule audit COMPLETE (keep only if container-FORCED & minimal; else revert toward upstream):**

- **eval grading — 8/369 dict-divergent (NOT 368/369; that counted func-NAME only). Plus a hidden scoring layer.**
  - Layer-1 (evaluator dict = func/expected/result/options): **8 diverge** — 6 FORCED-COMPAT, 1 COSMETIC, **2 OVER-DIVERGENCE**:
    - `multi_apps_2c1ebcd7` — `reference_base_result` **0.6→0.95** (`gen/eval/multi_apps.py:427`, NO comment). Verified vs
      upstream. Score = `(sim-base)/(1-base)`, pass iff `sim≥base` → this demands **95% similarity for any credit**;
      undocumented, no container mechanism. **STRICTER → revert to 0.6.**
    - `chrome_93eabf48` — upstream `[match_in_list, is_expected_url_pattern_match]` (2 rules incl. active-tab
      `^chrome://settings/appearance/?$`) → lite single `match_in_list` (light/system only). Getter swap
      `chrome_appearance_mode_ui`→`chrome_color_scheme` is FORCED (robust on-disk pref-read vs fragile a11y-UI).
      **CONTAINER-PROBE VERIFIED (downgraded OVER-DIV→FORCED):** the on-disk pref read REQUIRES flushing Chrome, and lite's
      flush SIGTERM-kills Chrome (`runner.py:973-1002`, `_flush_chrome_profile`; NO relaunch) → the debug port dies and the
      active-tab URL is unrecoverable (before flush the tab WAS `chrome://settings/appearance`; after, port refused; a
      relaunch shows only `chrome://newtab`, `session=None`, and Chrome never restores `chrome://` pages). So check (b) could
      not pass under lite's read flow → dropping it is FORCED, not gratuitous. Caveat: it was a DESIGN CHOICE — keeping the
      live a11y read (like upstream) would have retained (b); given the on-disk read, dropping (b) is unavoidable. KEEP.
    - FORCED-COMPAT (keep): `chrome_368d9ba4` (geo-redirect: expected URL loosened `/manchester/`→incl `/london/`),
      `chrome_f0b971a1` (NFL renamed page — data correction, config launches the new URL), `gimp_06ca5602` (postconfig
      6→0: fragile Export-dialog automation replaced by in-place overwrite read; same func+gold), `impress_550ce7e7`
      (`examine_shape:false` — LO re-anchors placeholder geometry on save; documented), `multi_apps_26150609` (result
      +`pygame.py` — headless import; see setup over-div below). COSMETIC: `multi_apps_788b3701` (URL percent-encoded).
  - Layer-2 (HIDDEN): `src/eval/metrics.py` **re-implements ~15 upstream eval funcs under the SAME name**; `runner.py:156`
    resolves lite's module BEFORE upstream → **~50 eval tasks are scored by lite code despite a byte-identical evaluator
    dict** (invisible to a dict diff). Almost all are LOOSER container false-negative fixes (LO/GIMP/Thunderbird
    round-trip drift), each documented w/ a bugs.md §N ref + a "genuinely-wrong still fails" guard → broadly FORCED-COMPAT,
    but a real, deliberate departure from strict OSWorld scoring (net-loosening) a reviewer should know. The 2 highest-volume
    funcs are SAFE: `compare_table` (62 eval tasks) and `compare_pptx_files` (53) delegate to upstream verbatim on the eval
    split (their divergent branches need `number_format`/`examine_color_rgb`/`examine_shape`, set by 0 eval tasks). One
    override is STRICTER: `compare_references` (= the `2c1ebcd7` task).
    - **REIMPL guard-audit (5 self-written overrides adversarially deep-audited — "can a genuinely-wrong answer pass?"):**
    13 funcs shadow upstream by NAME (0 lite-only new interfaces); WRAPPERs delegate to `_upstream()` then widen (safe by
    construction). The 5 that DON'T delegate (REIMPL) were audited:
    - `check_config_status` — **GUARD SOUND.** Faithful scan reimpl; memsize tolerance (`_gimp_memsize_to_bytes`, 1024-based,
      correct) is pure OR-add after exact-equality → one-directional; no reachable FP (only real numeric keys tile-cache-size
      + undo-levels, latter never unit-suffixed).
    - `check_json_settings` — **GUARD SOUND** (coupling caveat). Strict superset (present keys exact; only {absent whitelisted
      key == default} newly passes); whitelist values correct. FP contained by a setup-seeding invariant, verified dataset-wide
      (eval/synth rescue never fires; all 8 perturb rescue-firing rows seed the opposite value). Recommend a generator lint.
    - `compare_line_spacing` — **GUARD SOUND.** Upstream ALREADY has the text-gate + para-count; lite's only delta is None→1.0
      (single↔single merge, one-directional). Non-blocking: text-gate uses strict-upstream `compare_docx_files` not lite's
      lenient wrapper → latent FN only if a future spacing task also types text.
    - `compare_epub` — **GUARD SOUND** (deployed conversion-only tasks; `max(positional,content)` fail→pass; strips only
      OPF/NCX metadata noise, never prose). 2 latent out-of-scope holes: dc:date/meta-cover over-strip (metadata grading),
      missed-edit masking (edit-during-conversion) — neither reachable on the current pure-conversion tasks.
    - `check_contrast_increase_and_structure_sim` — **GUARD HOLE + OVER-DIVERGENCE.** SSIM math bit-identical to upstream +
      contrast-increase direction preserved (not a regression), BUT threshold 0.65→0.40 admits wrong-but-std↑ structure-preserving
      ops (posterize/brightness-with-clip/hist-eq: SSIM 0.45-0.62 pass@0.40, fail@0.65) — reward-hackable (feeds train).
      **Container-probe VERIFIED the loosening is NOT version-forced:** lite GIMP == osworld VM GIMP == `2.10.30-1ubuntu0.1`
      (byte-identical), same non-legacy GEGL formula. On the real task asset (f723c744/berries.png) the DEFAULT non-legacy op
      scores SSIM 0.757 even at MAX contrast → clears 0.65 → upstream 0.65 does NOT false-fail → **0.40 is unnecessary here**;
      the doc's cited low numbers (0.63/0.55/0.45) match the LEGACY op (measured 0.67/0.60/0.55), i.e. the rationale conflates
      legacy vs non-legacy. **Recommend: revert eval 0.40→0.65** (closes the hole, keeps the legit edit passing); verify the
      train contrast tasks' source images (flatter images may legitimately dip below 0.65) before a global revert.
- **setup `config`: 52 changed** — **45 COSMETIC** + **4 FORCED-COMPAT** + **2 REDUNDANT** + **1 OVER-DIVERGENCE**
  (all 7 non-cosmetic container-probe VERIFIED; the "5 forced / 2 over-div" split was the pre-verification classification):
  - COSMETIC (45): template literalization — `{SCREEN_WIDTH_HALF}`→960 / `{CLIENT_PASSWORD}`→user etc. baked at codegen;
    `src/utils/dispatch.py:57` (`_replace_templates`, mirrors OSWorld `replace_screen_env_in_command`) re-substitutes identically at
    runtime → byte-different, behaviorally identical. Zero-cost to revert to placeholder form; no functional reason to.
  - FORCED-COMPAT (4, container-probe VERIFIED, keep): `chrome_44ee5668`/`7b6c7e24` (empty container Chrome profile under
    `--user-data-dir=/home/user/chrome-data` — pristine ships only `Preferences` 40B, NO History/Cookies DB → must launch to
    create the schema (bare INSERT → "no such table: urls"), pkill to unlock (INSERT while running → "database is locked";
    cookies don't lock but pkill is anti-clobber), sqlite-seed, relaunch to hand the agent a live browser. NECESSARY + near-
    minimal; only trimmable bit is `7b6c7e24`'s inherited `chrome_open_tabs(amazon)` which seeds 0 persistent cookies in-container
    → the manual INSERT does the real work). `chrome_93eabf48` (setup side: appends a `Preferences` `color_scheme2`-pop so the
    "dark mode is ON" precondition actually holds — container Chrome's default differs from upstream's snapshot; single JSON key,
    minimal — tied to the same on-disk-read design as its eval-dict entry above). `os_28cc3b7e` (no systemd,
    PID1=supervisord → `pactl` fails cold "Connection refused"; `pulseaudio --start` fixes it — VERIFIED NECESSARY).
  - **REDUNDANT (2, container-probe VERIFIED — no-op on the current image, NOT forced; harmless, revert = pure cleanup):**
    `os_5ea617a3` — the appended `mkdir/chown/mv` Trash fallback: `gio trash` actually WORKS in this session (gvfsd-trash
    daemon up, gio 2.72.4, rc=0, populates `Trash/{files,info}`) → the fallback never fires (downgraded from FORCED). `os_bedcedc4`
    — the prepended `gsettings ... idle-dim true`: base default is idle-dim=**true** (standard GNOME schema default, MATCHES a
    real VM — NOT `false` as the `gen/eval/os.py:357-361` comment claims → **that comment is STALE**, likely an older base image); the
    true trivial-pass vector is `idle-delay=0`, already neutralized by **upstream's own** `idle-delay 300` step → lite's prepend
    is a `true→true` no-op that changes no eval semantics (downgraded from OVER-DIV).
  - **OVER-DIVERGENCE (1, container-probe VERIFIED):** `multi_apps_26150609` — prepends writing `Desktop/snake/pygame.py`
    **stub into the agent-visible task folder** the agent must debug. VERIFIED NOT forced: real pygame 2.6.1 is pip-installed and
    runs headless (Xvfb + SDL-dummy + even no-display), and the task's `test.py` is import-only (no `pygame.init`/display); the
    stub's keycodes are byte-identical to real pygame and scoring is identical (both `False` = buggy pre-agent state). So the
    stub is unnecessary AND pollutes the agent's project folder → move it to eval-time (and drop `pygame.py` from `result.path`).
- **postconfig: 146/204 changed** — dominant surface; **LO/VS_CODE slim-down DONE (§A).** Remaining overwrites audited:
  chrome `_CHROME_RESTART_POSTCONFIG`/`cookie_postconfig` are **train/synth-only** (idempotent reload, KEEP); per-task eval
  postconfig (chrome `93eabf48` sleep1→5, os `5812b315` apt hygiene) benign KEEP; **GIMP `already exists` handler (9 eval +
  synth-`gimp_export_as_postconfig`) — lite-added (upstream has none), verified ZERO side-effect (Export picker="Export
  Image", format="Export Image as PNG" → no `already exists` collision, unlike LO ^Save$); live/dead of GIMP's overwrite-confirm
  title unconfirmed after 3 probes (couldn't reliably trigger the collision) → harmless either way; keep/remove is a low-stakes
  minimalism call.**

So the "didn't keep upstream" problem is **concentrated in postconfig** (now slimmed), mild in setup (45/52 cosmetic),
and in eval it is small at the DICT layer but non-trivial at the hidden **metrics.py scoring layer** (net-loosening, documented).

**Over-divergence shortlist — ALL REVERTED (user instruction; tests green, catalogs regenerated):**
1. `multi_apps_2c1ebcd7` — `reference_base_result` 0.95 → **0.6** (upstream) at `multi_apps.py:427`. **DONE.** (Kept the
   separate, justified `metrics.py:856` no-heading→0 `compare_references` fix — that's not an over-divergence.)
2. `multi_apps_26150609` — **DONE.** Dropped the `config_append` that wrote the agent-visible `snake/pygame.py` stub +
   removed `pygame.py` from `evaluator.result.path`/`dest` → setup now = upstream 4 steps, real pygame (verified headless) is
   used. (The stub copy in `oracle_actions` — oracle self-verify replay, invisible to the agent/eval — was left as-is.)
3. `check_contrast_increase_and_structure_sim` — `ssim_threshold` 0.40 → **0.65** (upstream) at `metrics.py:915` + docstring
   rewritten. **DONE.** Container-probe proved this was NOT version-forced (lite GIMP == VM GIMP == 2.10.30) and the 0.40 band
   was reward-hackable; restoring 0.65 closes the hole (also applies to the 7 train gimp contrast tasks — flatter train source
   images should be spot-checked, but berries.png clears 0.65 at max contrast).
4. `check_brightness_decrease_and_structure_sim` — wrapper DEFAULT `threshold` 0.15 → **0.03** (upstream) at `metrics.py:897`
   + docstring/module-header cleanup. **DONE.** Probe on the real asset: legit non-legacy brightness-decrease MSE ≈ 0.00001 ≪ 0.03
   (upstream normalizes both to brightness 128 → pure brightness change cancels), so 0.03 doesn't false-fail; the old "0.046/0.072"
   doc numbers don't reproduce; [0.03,0.15) was reward-hackable (posterize/solarize/wash-then-darken 0.033-0.116). Fixes eval
   `7a4deb26` + 6 train.synth brightness tasks (all `options=null` → use the default). **IMPORTANT — NOT touched:** eval
   `multi_apps_4c26e3f3` explicitly carries `options.threshold=0.15` and so does its UPSTREAM evaluator → that 0.15 is upstream
   parity, passed through unchanged (a probe-caught correction: the sub-audit had wrongly suggested dropping it).
Downgraded by probes (NOT over-divergences, left as-is): `chrome_93eabf48` (dropped URL check) → **FORCED** (flush kills Chrome;
URL unrecoverable). `os_bedcedc4` (idle-dim prepend) → **REDUNDANT no-op + STALE comment** (base default already true; upstream
idle-delay=300 handles trivial-pass). `os_5ea617a3` Trash mv-fallback → **REDUNDANT** (gio trash works). The three redundant/
downgraded items change no behavior; the `gen/eval/os.py:357-361` comment is factually stale and worth correcting.

MIGRATION examples (lite diverges from the VM — fix): env-vendored CLI unreachable on the user
channel; config not flushed/persisted like the VM; controller-vs-session routing;
umask/ownership; agent/env python split; a tool not baked; pulseaudio/gsettings
session-bus; container egress-IP/anti-bot; soffice/xvfb quirks. UPSTREAM examples (shared with
the osworld VM — PRESERVE; "fixing" them would diverge lite from osworld): LibreOffice /
openpyxl / EMU-geometry comparator over-strictness, gold bugs, OCR misses, ambiguous instructions.

## Headline (30-step authoritative)

**No migration-induced *evaluator* regression.** The substrate (flush, session-bus,
config persistence, LibreOffice save+dialog automation) works — e.g. 23 calc tasks
pass with the identical postconfig save sequence. **0 landable eval-code
(comparator/getter) migration bugs.** The one actionable eval-list item: extend
`exclude_reason` to rows *unwinnable in the container* (anti-bot egress-IP), which no
comparator change can rescue.

## Migration classes observed

### B. Container egress-IP anti-bot / geolocation redirect — ACTION: extend `exclude_reason`

Chrome/Scholar live-site tasks where the agent builds the *correct* URL but the
datacenter egress IP is blocked (Cloudflare/Akamai/CAPTCHA) or geo-redirected
(`scholar.google.com`→`.com.hk`). Confirmed at 30 steps (not step-starvation). No
comparator un-blocks a live site — exclude these (a true infra fix = residential/US
egress proxy). The 8 rows below currently have **no** `exclude_reason` — the list is
incomplete:

| task_id | block |
|---|---|
| `osworld_chrome_82279c77` | cars.com Cloudflare (+ upstream getter drift `list_price_max`→`maximum_price`) |
| `osworld_chrome_6c4c23a1` | delta.com "Access Denied" |
| `osworld_chrome_1704f00f` | rentalcars.com CAPTCHA wall |
| `osworld_chrome_b7895e80` | TripAdvisor IP block (52.177.10.152) |
| `osworld_chrome_cabb3bae` | Kohls Akamai "Access Denied" |
| `osworld_multi_apps_36037439` | Scholar `.com.hk` redirect + anti-bot (reached correct profile) |
| `osworld_multi_apps_58565672` | pre-seeded scholar tab diverted → exact-tab-list unsatisfiable |
| `osworld_multi_apps_5990457f` | scraping disabled → live Scholar vs frozen 2023 gold |

**Live container `curl` probe (`probe_antibot.py`, egress `169.229.219.180`):** cars.com **403**
(Cloudflare/captcha), delta.com **444** (Access Denied), tripadvisor **403** (captcha), kohls.com
**403** (Access Denied) — **hard-blocked**; rentalcars.com **200** and scholar **200** (no
`.com.hk` redirect) — **reachable from THIS host** though blocked from the §2 datacenter IP. Confirms the
block is **egress-IP-dependent**, i.e. a deployment/network property — **NOT a substrate/MIGRATION gap**
(the osworld VM behind the same egress fails identically). So the decision test → **UPSTREAM/infra**:
`exclude_reason` when the deployment's egress is blocked (the 4 hard-blocked always; the geo/soft ones
conditionally). A real "fix" is a residential/US egress proxy, not a comparator or container change.

### A. Postconfig save-dialog clobber — MIGRATION (lite diverged the postconfig), confirmed by upstream-diff — **FIXED**

> **STATUS: FIXED.** `LO_SAVE_POSTCONFIG` slimmed to `ctrl+s + alt+f,s` (all 4 dialog handlers dropped under
> the keep-rule — see the slim-down table below). Regenerated + lock refreshed + 18 contract tests pass.

`osworld_libreoffice_impress_0a211154`. **Root cause with evidence — diffed lite's `evaluator.postconfig`
(eval.jsonl) against the UPSTREAM OSWorld task JSON** (`.venv/.../osworld_evaluation_examples/examples/
libreoffice_impress/0a211154-….json`):

| | postconfig (the pre-eval save sequence) |
|---|---|
| **upstream (VM runs this)** | 4 steps, GENTLE: activate → sleep → `pyautogui.hotkey('ctrl','s')` → sleep. Just one ctrl+s. |
| **lite.osworld (eval.jsonl)** | ~11 steps, AGGRESSIVE: `ctrl+s` → **also** `alt+f,s` (File→Save) → then **Return-dismiss** `Changed by Others` / `Keep Current Format` / `already exists` / `Save`. |

So it is **NOT the VM's problem** and NOT a comparator — **lite's eval generator rewrote the postconfig
to be more aggressive.** When the agent edited the pptx **on disk** (outside LibreOffice), the file is
"changed by others"; lite's postconfig hits **Return on "Changed by Others" = Save-Anyway**, writing
LibreOffice's **stale in-memory buffer** back over the agent's on-disk edits → revert. Upstream's lone
`ctrl+s` never force-dismisses that dialog, so the **VM does not clobber**. → **MIGRATION.**
Caveat: only bites the **on-disk-edit agent strategy**; a GUI edit (the intended path) saves cleanly with
`ctrl+s` and never raises "Changed by Others". Why lite diverged: likely the container needed the
File-menu path / dialog handling to save reliably — the side effect is the clobber.
Fix direction (if adjudicated): make the "Changed by Others" branch keep the DISK version (Reload), or
match upstream's gentle single `ctrl+s`. Fix location: the eval generator's impress postconfig authoring.
(Sibling agent-artifact `impress_e4ef0baf` — agent `ctrl+s` clobbers its own terminal-python edits; that
one is agent-ceiling, not this.)

**How the two lite envs handle the upstream postconfig — OPPOSITE philosophies (recorded for future audits):**

| | lite.osworld | lite.scalecua |
|---|---|---|
| mechanism | `src/gen/eval/postconfig.py::normalize_postconfig` → **rewrites** the upstream `ctrl+s` save-segment into `LO_SAVE_POSTCONFIG`/`VS_CODE_SAVE_POSTCONFIG` (`src/gen/common.py`) | `src/osworld/verify.py::_run_postconfig` → **dispatches the upstream postconfig ~as-is** (only special-cases chrome `pkill`→profile-flush) |
| what it changes | the **ACTIONS**: adds `alt+f→s` menu-nav fallback (container `ctrl+s` is swallowed by a focused inner widget) + conditional Return-dismiss of `Changed by Others`/`Keep Current Format`/`already exists`/`^Save$` | the **EVALUATION**: `_capture_pre_postconfig_state` (snapshot BEFORE postconfig) + a ~20-fn `_repair_*_result` family (pptx/xlsx/chrome-settings/clipboard/url/…) that reconstructs the true result |
| goal | reliably **flush the agent's edits to disk** in the container | don't let postconfig **corrupt the scored result** |
| A-style clobber | **yes** — CBO "Save Anyway" saves LO's stale buffer over an on-disk edit (class-A) | **structurally avoided** — reads the agent's real on-disk artifact; the forced save doesn't decide the score (also: scalecua's RL fixtures often carry EMPTY postconfig — oracle writes the result — so it leans on save less) |

**Implication for A:** an alternative class-A fix is to adopt scalecua's pattern (pre-postconfig snapshot +
`_repair_generated_pptx_result`-style result reconstruction) instead of the aggressive save. Maintainer's call.

**Postconfig slim-down — DONE (keep-rule applied). Keep an extra op ONLY IF it rescues a real case AND has
ZERO side-effect in every scenario (GUI-solved AND terminal/out-of-band-solved); else drop to match upstream.**
Upstream LO postconfig = bare `activate → sleep → ctrl+s → sleep`, ZERO dialog handling. `LO_SAVE_POSTCONFIG`
(common.py) is now `activate → sleep → ctrl+s → sleep → alt+f → sleep → s → sleep` — upstream's `ctrl+s` + the
one justified addition (`alt+f,s`). All four dialog handlers were removed. Per-op verdict (each verified):

| op | lite-added? | rescues | side-effect | verdict |
|---|---|---|---|---|
| `ctrl+s` | no (upstream) | the save | — | **KEEP** (parity) |
| `alt+f,s` | yes | ctrl+s **swallowed** by a focused sidebar combobox (Properties→Orientation), live-repro `probe_swallow.py` — menubar accelerator bypasses inner-widget focus capture | none: no-op on a clean doc; terminal-edit case forces a stale save → CBO modal (now un-confirmed) BLOCKS it → on-disk edit survives (`probe_clobber.py`) | **KEEP** |
| `"Changed by Others"→Return` | yes | GUI CBO flush | **YES — clobber**: auto-"Save Anyway" writes LO's stale buffer over an agent's on-disk/terminal edit (`probe_clobber.py`: marker gone) | **DROP** |
| `"Keep Current Format"→Return` | yes | — | none, but **rescues NOTHING**: `WarnAlienFormat=false` in lite's baked profile → dialog never appears (`probe_keepformat.py` = 0) | **DROP** (dead) |
| `"already exists"→Return` | yes | — | none, but **rescues NOTHING on this substrate**: lite's LO titles the Save-As replace-confirm **"Save"**, not "…already exists" (`probe_saveas.py`) → never matches | **DROP** (dead) |
| `"^Save$"→Return` | yes | the real "Save"-titled confirm | **YES**: the Save-As **file-picker is itself titled exactly "Save"** (`probe_saveas.py`) → matcher grabs the picker and Returns → prematurely accepts a Save-As the agent left open (wrong/spurious filename) | **DROP** |

Key evidence (`probe_saveas.py`, on lite's baked LO): after `Ctrl+Shift+S` the file-picker window title is
exactly **"Save"**; after Enter the replace-confirm is **also "Save"** and there is **no "already exists"**
window. So (a) `already exists` is a dead matcher here, and (b) any `^Save$` matcher can't hit the confirm
without also hitting the picker → the "agent left a Save-As confirm open" case is **un-rescuable without a
side-effect** → intentionally NOT rescued (upstream/VM doesn't rescue it either → parity).
VS Code (`VS_CODE_SAVE_POSTCONFIG`): `ctrl+s + alt+f,s`, no dialog handlers — same single kept op; no CBO-class
auto-confirm (a stale-disk save just fails with a passive notification, can't clobber). **KEEP as-is.**
GIMP (`gimp_export_as_postconfig`, synth-only + the 9 gimp-eval 'already exists' inserts): a **different app** —
the LO "Save"-title finding does NOT transfer. **NEEDS a GIMP-specific window-title probe** before applying the
keep-rule; left as-is pending that.
- **Latent bug (still open):** `normalize_postconfig` `segment is None` fallback (postconfig.py) returns the BARE
  template, **discarding the task's original tail steps**. All 5 `--convert-to csv` calc tasks currently hit the
  segment-found path (tail preserved), but a future LO task whose `ctrl+s` isn't right after a `LibreOffice`
  activate would silently lose its tail.
- **Open (lower priority):** `activate_window("LibreOffice")` strict:false vs upstream's exact-title strict:true —
  could focus the wrong window in multi-window (multi_apps) tasks; not yet reverted.

### C. Container `user` lacks general sudo — substrate gap

`osworld_os_5812b315` (create SSH user), `osworld_os_94d95f96` (install Spotify) need
root. **Live probe of BOTH substrates (corrects an earlier imprecise "VM grants full sudo"):**
the real difference is **sudo-GROUP membership, not passwordless-ness** —
- **VM** (`probe_vm_sudo.py`): `user` ∈ groups `… sudo …`; `sudo -n true` → **"a password is
  required"** (so the VM is NOT passwordless either) — but the agent CAN sudo by typing the
  OSWorld credential (`user`:`password`).
- **lite** (`probe_sudo.py`): `user` ∈ group `user` only (NOT in `sudo`); `lite/gym/envs/lite/osworld/docker/Dockerfile:236-259`
  grants only 3 scoped NOPASSWD shims (timedatectl / hostname / powerprofilesctl) → general
  `sudo` is unauthorized even with a password.

So parity-fix (if adjudicated to fix, not exclude) = **match the VM's mechanism**: `usermod -aG
sudo user` + set `user:password` — NOT add NOPASSWD (that would diverge; the VM requires the
password). Still a policy call (widen sudo vs `exclude_reason`), not a comparator.

### D. Suspects — RESOLVED by live container probe (both NOT a container-WM migration gap)

Both class-D "suspect container-WM" hypotheses were **disproved by probing the live `:mine`
container directly** (the AGENTS.md fallback: when open-loop replay can't show it, probe the
container). The container runs GNOME Shell / mutter, which maintains EWMH correctly:

- `osworld_vlc_8d9fd4e2` — **NOT a WM gap.** Drove VLC to true fullscreen via `env.step([key f])`:
  `_NET_CLIENT_LIST` still lists the VLC window, `wmctrl` shows it at `0,0 1920×1080`, and the
  Flask `/window_size app_class_name=vlc` returns **`{width:1920, height:1080}`** — correct. (Windowed
  before = 1850×1016; fullscreen after = 1920×1080, as expected.) So `_NET_CLIENT_LIST` does NOT go
  empty on fullscreen here. → **UPSTREAM / agent-ceiling**, not MIGRATION.
- `osworld_multi_apps_02ce9a50` — **NOT a blank-capture gap.** `gnome-screenshot -w` produced a
  non-blank 1850×1053 PNG (grayscale extrema 24–255, content). → the reward-0 is **UPSTREAM (easyocr
  miss)** or agent-ceiling, not a container WM/screenshot migration bug.

(Probe scripts: `probe_classd.py`, `probe_vlc_fs2.py`. If a future audit re-flags these, re-probe
the same way before assuming a WM gap.)

## UPSTREAM — note-only (DO NOT FIX)

Fail identically on the original VM: `chrome_f0b971a1` (`compare_urls` exact path),
`gimp_e2dd0213` (`check_textbox_on_leftside` left-5%), `calc_0a2e43bf` (`compare_table`
`DataFrame.equals` on helper cells), `calc_1954cced` (`pivot_table` `_pivots=[]`; agent
typed values), `impress_04578141` (exact-color comparator on an exact-color
instruction), `thunderbird_9bc3cc16` (subject-named filename regex), `multi_apps_7ff48d5b`
(fuzzy_place_math 3-token), `multi_apps_a82b78bb` (liked_authors exact whitelist),
`multi_apps_48c46dc7` (CLI-chrome no CDP port), `multi_apps_82e3c869` (unzip hardcodes
`Desktop/`), `multi_apps_f7dfbef3` (bash_history grep glob-only), `vlc_aa4b5023`
(ambiguous "main screen"), plus the general `compare_table`/`compare_pptx`
chart/format strictness. Everything else reward-0 is **agent-ceiling**.

---

# Bridging lite.osworld ↔ official osworld (cross-replay)

Goal: **close the substrate gap** between `lite.osworld` (`sandbox.linux` container)
and official `osworld` (VM-in-Docker on `Ubuntu.qcow2`). Both run the **same 369
upstream tasks** (1:1 by UUID) and share **`LiteDesktopActionSpace`**, so a recorded
action list is portable to the other substrate — replay the same inputs, diff the
outputs. This makes the §2 decision test **empirical**.

## Task-id correspondence

Keys differ per env, so **join on the full OSWorld UUID**, never on the key string:

| env | registry key | full UUID |
|---|---|---|
| `osworld` | `osworld@<uuid>` | the key suffix itself |
| `lite.osworld` | `lite.osworld@osworld_<domain>_<uuid[:8]>` | eval-row `metadata.osworld_id` |

Build the `(lite_key, osworld_key)` pairs from the committed `tasks.json` alone
(no eval.jsonl, no env import, no docker) — the lite key is the eval generator's
formula (`src/gen/eval/__main__.py:177`), and `exclude_reason` rows are dropped so
the harness scope excludes **all infeasible / blocked / anti-bot** tasks:

```python
import json, pathlib
tasks = json.load(open("lite/gym/envs/osworld/data/tasks.json"))["tasks"]
pairs = [(f'lite.osworld@osworld_{t["domain"]}_{t["task_id"][:8]}', f'osworld@{t["task_id"]}')
         for t in tasks if not t.get("exclude_reason")]          # 369 tasks → 325 pairs (44 exclude_reason dropped)
# uuid[:8] derivation is collision-free across the 369; the full UUID is the authoritative key.
```

**Don't unify the ids.** Keys are already env-qualified (no ambiguity), and
`osworld_` is load-bearing on lite.osworld (separates migrated eval from `synth_` /
`perturb_` train) but tautological on `osworld`; lite's ids are also locked into the
published HF `cua-lite/Lite.OSWorld` corpus. Join on the UUID instead. (Optional
one-liner nicety, no rename/regen: surface `osworld_id` into lite's live
`metadata.others`.)

## Method — bidirectional open-loop action replay

The agent trace is **fixed** (recorded GPT-5.5 actions); only the executing substrate
changes, so any divergence is the substrate, not the model.

> **Reuse, don't reinvent** — a single-task replay driver already exists:
> [`replay_trajectory.py`](/devs/envs/lite.osworld/validate/rollout/replay_trajectory.py)
> (loads `lite_message.tool_calls` from `03_actions.json`, replays in a fresh container,
> saves per-turn screenshots, runs the evaluator). **Adapt** it into the bidirectional
> cross-substrate diff below — don't rewrite from scratch. Caveat: it's **stale on
> resolution** (its docstring assumes an old 1024×768 GPT config; the current GPT config is
> env-native 1920×1080, no agent-side resize), so re-derive env_kwargs from the live config,
> not its literals. The snippet below just illustrates the core loop.

**A. Record on X** (`--save-data` writes per-turn dirs; on by default):

```bash
# single task; to sweep, drop --task-id and add:
#   --splits eval --filter "lambda m: not m.others.get('exclude_reason')"   # excludes all infeasible/blocked/anti-bot
# using a §1 private image? export LITE_OSWORLD_CONFIG=/tmp/mine.yaml first (NOT --env-kwargs image — §1)
uv run python scripts/rollout.py --model-id gpt-5.5 --env-id lite.osworld \
  --task-id osworld_chrome_1704f00f --config-path scripts/configs/gpt/default/lite.osworld.yaml
uv run python scripts/rollout.py --model-id gpt-5.5 --env-id osworld \
  --task-id 1704f00f-79e6-43a7-961b-cedd3724d5fd --config-path scripts/configs/gpt/default/osworld.yaml
```

Writes `.logs/rollout/gpt-5.5/<env_id>/<ts>/<split>/<task_id>/sample_00/`, where
`<split>` = `task` for a pinned `--task-id`, `eval` when sweeping `--splits eval`.
Current canonical trajectory images are stored under `sample_00/images/*.png`.
Per-turn debug layout is `turn_NNNN/prompt_images/*.png` and matching
`turn_NNNN/prompt_images_annotated/<same-name>.png` when debug artifacts are
enabled, plus `01_prompt.txt`, `02_response.txt`, `03_actions.json`,
`04_results.json`, and optional `05_timing.json`.
Replay the `lite_message.tool_calls` field — the canonical nested
`LiteToolCall` list (`id`, `type`, `function.name`, `function.arguments`;
normalized [0,1000] coords) fed to `env.step()`. **Not** its
sibling `executed_actions` (env-lowered: per-env vocab + pixel coords, not portable).

**B. Replay X's actions on Y** (open-loop, no model; run in the env's venv, e.g.
`uv run --extra quick-start python replay.py`):

```python
import asyncio, json, pathlib
import lite.gym as gym
from lite.infer.debug.log_layout import turn_dirs

async def replay(env_key, sample_dir, out_dir, **make_kwargs):   # extra gym.make kwargs, e.g. max_steps
    turns = turn_dirs(pathlib.Path(sample_dir))
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(env_key, max_steps=len(turns) + 1, **make_kwargs)
    obs = await env.reset()
    img = obs.image                                      # None on capture timeout
    for t in turns:
        if img:
            (out_dir / f"{t.name}.png").write_bytes(img)
        msg = json.loads((t / "03_actions.json").read_text())["lite_message"]
        actions = msg.get("tool_calls") or [{
            "id": "replay_noop",
            "type": "function",
            "function": {"name": "noop", "arguments": {}},
        }]
        res = await env.step(actions)                    # same canonical calls, other substrate
        img = next((r.image for r in res.results if r.image), None)
        if res.terminated or res.truncated:
            break
    await env.close()

asyncio.run(replay("osworld@1704f00f-79e6-43a7-961b-cedd3724d5fd",
    ".logs/rollout/gpt-5.5/lite.osworld/<ts>/task/osworld_chrome_1704f00f/sample_00",
    "/tmp/replay/1704f00f"))
# osworld→lite: replay("lite.osworld@osworld_chrome_1704f00f", "<osworld sample dir>", "/tmp/replay/…")
#   for a §1 private lite image, export LITE_OSWORLD_CONFIG=/tmp/mine.yaml first (NOT image= — §1)
```

Do **both directions**: lite→osworld catches what the container tolerates but the VM
doesn't; osworld→lite catches the reverse (a tool/permission the VM grants).

**C. Diff & adjudicate.** Align by turn index (current
`turn_NNNN/prompt_images/*.png`, or legacy images through the shared debug layout
reader, ↔ replay `turn_NNNN.png`); a divergence at turn *k* localizes to the action at *k*. Then run the
**adjudication protocol** below — the metrics only *surface* candidates; **your own vision is the verdict.**

## Diff & adjudication protocol

The single place the scoring + judgment rule lives (§3.C, §4 Phase 3, and §5's scoreboard all refer here):

1. **Two metrics, one flag.** Per turn, `visual_diff.py` scores **SSIM** (grayscale structure) **AND
   CIEDE2000 ΔE** (CIELAB color, mean + p95). **Flag a turn if `SSIM < 0.98` OR `ΔE-p95 > 3`.** They are
   complementary — SSIM catches structure/layout; ΔE catches palette/theme. **SSIM is grayscale-blind** to
   equal-luminance color shifts (proven: a same-luminance gray→green swap = **SSIM 1.0000 but ΔE-p95 36.5**),
   which is exactly why ΔE is not optional.
2. **First-divergence only, vs the task's own noise floor.** Open-loop replay drifts *even same-substrate*:
   once one action lands on a slightly different screen it **cascades**. So (a) each task's **same-substrate
   replay is its noise floor** — NOT ≈1.0 (measured 0.75 for `vlc`/`vs_code`); a cross-substrate flag counts
   only if it **exceeds that task's floor**; (b) only the **FIRST** flagged turn per trajectory is meaningful
   (everything after is cascade).
3. **The metric SURFACES; your VISION is the VERDICT — always, no exceptions.** A flag (even a huge cross-drop
   on a clean floor) is only a prompt to open `visual_diff.contact_sheet` (orig | replay | abs-diff) and
   **look** — you have vision (no OCR, no external VLM; *you* are the judge). The number alone lies both ways:
   - **Big cross-drop ≠ gap** (necessary, not sufficient): a clean-floor task can crater from pure DRIFT —
     `gimp_06ca5602` floor 0.9987 → cross 0.49 (a transient open menu), `chrome_030eeff7` 0.92 → 0.46
     (navigation diverged: Settings vs a Google search); both render identically.
   - **Small / colored gaps score ~clean**: the terminal titlebar `Terminal` vs `user@host:~` (GAP-03) scores
     SSIM 0.9944 / ΔE 0 — "clean" by metric, real by eye.
4. **The scalar is per-domain BANNABLE; the visual check is NOT.** Where the floor is <~0.9, or the domain is
   banner/menu/omnibox-driven, the number is noise — turn it OFF and adjudicate by eye. Per-domain trust
   (first sweep, gpt-5.5, 16 tasks):

   | domain | same-substrate floor | scalar | note |
   |---|---|---|---|
   | `os` | ~1.0 (0.86 when a shortcut re-rolls) | **use** | terminal-open tasks drift when `ctrl+alt+t` re-rolls (GAP-02) |
   | `gimp`, `libreoffice_calc` | 0.999 | use for structure, **verify by eye** | transient menus + GAP-04 banners crater cross |
   | `chrome` | 0.86–0.92 | weak | omnibox/nav + network content drift |
   | `vlc`, `vs_code` | 0.75 | **ban** | floor already collapsed; cross ≈ floor ⇒ read nothing into it |
   | `libreoffice_*`, `multi_apps` | 0.98–0.999, cross ~0.5 (GAP-04 banners) | **ban → visual-only** | banner offset derails coord clicks |

   The scalar earns trust only on `os`/`gimp`/`calc` structure — and even there a clean floor can cross-drop
   from drift (rule 3).
5. **Masks.** Taskbar clock (fixed region), capture-time cursor glyph, network content = noise → mask them
   (`visual_diff.DEFAULT_MASKS`, currently empty; measure the 1920×1080 rects). Pin timezone/locale for dates.
6. **Reward is a corroborating secondary** for FUNCTIONAL gaps (run the evaluator post-replay) — but evaluators
   are **per-substrate** (§2), so a reward delta may be evaluator-impl drift: use it to *attribute*, not to score.

## What it surfaces

- **Functional gaps** — same actions succeed on one substrate, fail on the other
  (screenshots + reward diverge). E.g. the §2 class-C sudo gap: identical actions go
  through on osworld, stall at the privileged step on lite.osworld. Same shape for any
  missing tool, umask/ownership, or session-bus difference. (Reward corroborates — see
  protocol rule 6.)
- **Visual gaps** — cosmetic differences that don't fail the task but shift the
  agent's visual distribution (eval fidelity + SFT realism); only a screenshot diff
  finds them. Known: lite.osworld's terminal shows `user@host`/path in plain uncolored
  text vs the VM's classic Ubuntu palette (green user@host, blue path). Sweep siblings:
  window theme, cursor, fonts, wallpaper, icons.

## Caveats

- **Action-lowering is NOT shared** — lite.osworld denormalizes [0,1000]→px and maps keys
  in [`sandbox/base.py`](/lite/gym/sandbox/base.py) + [`keys.py`](/lite/gym/utils/backend/keys.py);
  osworld lowers in-container via upstream `desktop_env` pyautogui (`docker/server.py`). The
  same `LiteToolCall` can land on a different pixel or key, so unit-check lowering parity (feed
  identical actions, compare each substrate's computed px/key) — else a lowering mismatch reads
  as a substrate gap.
- **Match the recording rollout** — replay with the SAME `env_kwargs` the rollout used
  (pass its `--config-path`): coords were chosen at the rollout's agent-side view. GPT's
  config is env-native 1920×1080 with **no** `resolution` resize; other models may downscale
  (`resolution` env_kwarg) — mismatch it and clicks land on the wrong widget. Both substrates
  render 1920×1080, so frames are directly comparable — assert equal size, never rescale.
- **Inter-turn pacing** — a rollout has natural LLM-latency gaps between turns; open-loop replay
  fires back-to-back and can race the GUI (a dropdown still open when the next click fires →
  false divergence). Add a small inter-turn sleep (~1s; more for Impress/multi_apps), as
  `replay_trajectory.py` does.
- **Diff metric / drift / floor / scalar-ban** → all consolidated in **§Diff & adjudication protocol** above
  (thresholds, SSIM color-blindness, first-divergence-vs-floor, clean-floor-not-sufficient, per-domain ban).
- **Host** — `osworld` needs KVM (`/dev/kvm` + `/dev/net/tun`) + the derived image;
  run on a KVM host with both envs installed ([README](/lite/gym/envs/osworld/README.md)).

---

# Execution plan

§2 is a list of migration-gap **hypotheses** from a reward-only audit; §3 is the
**engine that falsifies them and finds what reward can't** (visual gaps; the class-D
suspects a single-substrate audit couldn't resolve). So one harness does both — don't
re-audit by hand what it can diff. Sequence:

### Phase 0 — isolated build + provision  ✅ BOTH sides DONE (validated)
Per §1: private lite `:mybase`+`:mine` AND osworld `:mine` (symlinked shared `Ubuntu.qcow2`, no
re-download) built, label-stamped → gate passes; runs pointed at via `LITE_OSWORLD_CONFIG=/tmp/mine.yaml`
/ `OSWORLD_CONFIG=/tmp/osworld_mine.yaml`; shared `:latest` untouched; docker-free prereqs via
`install.sh provision`. Same-substrate rollout→replay→diff validated (SSIM≈1.0) AND a first real
**cross-substrate** sweep run on a KVM host (16 same-substrate floors + 6 terminal-heavy pairs, both
directions) — see §Gaps GAP-02…05. (Smoke used direct-mode `sem=2` VM throttle; the real 325-pair
sweep should run through the **env-server** for admission, concurrency, and reaping, not this.)

### Phase 1 — exclusion fixes (harness-free, ship now)
Extend `exclude_reason` to the 8 anti-bot rows in §2's class-B table, and confirm
**all infeasible tasks are marked** (they define the harness's exclusion set). Edit
the eval generator (`lite/gym/envs/lite/osworld/src/gen/eval/`: `infeasible` is now DERIVED from upstream
`evaluator.func == "infeasible"` in `__main__._is_infeasible`; other reasons via per-domain ORACLES `exclude_reason`) then regen — never hand-edit JSONL (see
[/devs/envs/lite.osworld/lite.osworld.md](/devs/envs/lite.osworld/lite.osworld.md)).
`osworld/data/tasks.json` already carries `exclude_reason` (blocked/infeasible/google_auth),
so the mechanism is proven. Also reconcile the two envs' exclude sets — osworld `tasks.json`
vs the lite generator diverge (the 8 anti-bot rows are the known delta) — so the harness scope
is symmetric. **Independent of everything below.** (Defer the class-C sudo policy call to
Phase 3 — how many tasks it unblocks.)

### Phase 2 — cross-replay harness  ✅ scaffolded + validated (same-substrate)
Built in [`bridge/`](/devs/envs/lite.osworld/bridge/): `task_map.py` (325-pair UUID join,
exclude-filtered), `visual_diff.py` (SSIM **+ ΔE** + mask + contact-sheet), `cross_replay.py`
(own `_load_kept_turns` — keeps turn identity, NOT `replay_trajectory._load_actions`; open-loop replay
on either substrate + diff, both directions). Same-substrate smoke (build→rollout→replay→diff) passed:
SSIM ≈ 1.0, ΔE-p95 0 — the noise floor; a first real cross-substrate sweep has since run (§Gaps GAP-02…05).
**Hardening still worth doing:**
- **Action-lowering parity GATE (blocker)** — the two envs don't share lowering code (lite:
  `sandbox/base.py`+`keys.py`; osworld: in-container pyautogui). Unit-check identical `LiteToolCall`s
  → identical px/key on both substrates FIRST; a mismatch corrupts every diff (scoreboard = noise).
- **Clock/cursor masks** → `visual_diff.DEFAULT_MASKS` (measure the 1920×1080 rects; the smoke
  showed the taskbar clock is the dominant non-determinism).
- **`task_map` coverage-assert** (each derived lite key ∈ registry) + **exclude-union** (Phase 1).
- Scale-out: the first sweep used a direct-mode `sem=2` VM throttle; the real 325-pair sweep must run
  through the **env-server** (admission + concurrency + reaping), not direct mode.

### Phase 3 — sweep both directions (concrete)
For each in-scope pair from `task_map.load_pairs()` — record on each substrate, cross-replay
both ways, diff (from repo root):

```bash
# record (lite via the §1 private image; osworld on a KVM host)
LITE_OSWORLD_CONFIG=/tmp/mine.yaml uv run python scripts/rollout.py --model-id gpt-5.5 \
  --env-id lite.osworld --task-id <lite_task> --config-path scripts/configs/gpt/default/lite.osworld.yaml
uv run python scripts/rollout.py --model-id gpt-5.5 --env-id osworld \
  --task-id <uuid> --config-path scripts/configs/gpt/default/osworld.yaml
# cross-replay both directions + diff (writes contact sheets; prints per-turn SSIM + ΔE)
LITE_OSWORLD_CONFIG=/tmp/mine.yaml uv run python devs/envs/lite.osworld/bridge/cross_replay.py \
  <lite_task> --lite-rollout <lite sample_00> --osworld-rollout <osworld sample_00> -o /tmp/bridge/<uuid>
```

Flag a turn on **SSIM < 0.98 OR ΔE-p95 > 3** (or a reward/functional divergence); **append each to
§5 live** (per task; resumable) with the mandatory evidence. This resolves §2's open items (confirm
class C / class A; disambiguate the class-D suspects by whether screenshots diverge) and finds new
visual gaps. Prioritize domains by the real lite-vs-osworld eval-score delta
([devs/exps/eval/README.md](/devs/exps/eval/README.md)) — biggest score gap first.

### Phase 4 — apply fixes by class
- **Functional → MIGRATION fix** (widen sudo, bake a tool, fix the container WM so
  `gnome-screenshot -w` / `_NET_CLIENT_LIST` match the VM, fix the postconfig save) or
  `exclude_reason` when unfixable.
- **Visual → substrate-fidelity backlog** (theme the terminal to Ubuntu, align window
  decorations/cursor/fonts) → edit `lite/gym/sandbox/docker` or
  `lite/gym/envs/lite/osworld/docker`, rebuild `:mybase`/`:mine` (Phase 0), re-diff. The additive
  Dockerfile already encodes VM-matched parity (`en_HK.UTF-8` locale, Yaru-green accent, pinned
  fonts) — add new visual matches there beside them, and derive each target by **probing the pinned
  reference image** (`happysixd/osworld-docker`, the authoritative look), not by guessing "nicer".

### Phase 5 — re-verify
Re-diff the fixed tasks (gap gone) + a fresh GPT-5.5 `lite.osworld` eval rollout (no
regression). Fold the final classification back into #157. Freeze ~5–10 known-matching tasks
(one per domain) as a `live`-marked substrate-parity smoke test so a future `sandbox.linux`
edit can't silently reintroduce a gap.

## Landability

| Item | Harness? | KVM? | Ship |
|---|:-:|:-:|---|
| §1 isolated build | no | no (only when the osworld side runs) | now |
| Phase 1 exclusions | no | no | now, standalone |
| Phase 2 harness | — | yes | then |
| Phase 3–4 sweep + fixes | yes | yes | after harness |
| Phase 5 re-verify | yes | yes | last |

---

# Gap tracker

Live log of gaps from the §3 sweep and §2 audit. **This is a DIVERGENCE LEDGER, not a fix log.**
The bridge's job is to **record, root-cause, and PROPOSE a resolution — it does NOT apply fixes.**
Which divergences to fix (and how), which to accept as UPSTREAM, and which to `exclude` is the
**maintainer's adjudication**, made from this ledger. So the bridge advances a block only up to
**`root-caused` (+ a proposed resolution)**; `fixed`/`verified` are stamped later, *after* someone
adjudicates and implements. Do NOT edit `sandbox.linux` / eval generators to "close" a gap as part
of the sweep — surface it here and stop.

**Append a `GAP-NN` block per task/batch as found — don't wait for the full sweep** (so a mid-run
stop loses nothing). **Evidence is mandatory** on every block: both substrates' screenshot paths +
the diff artifact (or, for a visual-only catch, the contact sheet + what you saw), the diverging turn
index + action, and (functional) reward on both sides. **Status:** `open` → `root-caused` →
[maintainer adjudicates] → `fixed` → `verified` (or `wontfix`). **Class:** `functional` | `visual`.
**Proposed resolution:** `MIGRATION-fix` | `exclude` | `UPSTREAM/accepted` (proposal only — the
maintainer decides).

**Parity scoreboard (the north-star KPI)** — track confirmed gaps each sweep, **over the in-scope
(non-excluded) split**, and drive to 0. The metric is **candidate → you adjudicate → confirmed**,
NOT an auto-count (open-loop drift produces false flags, §3):
- (a) **candidate** = the **FIRST** turn (per trajectory) flagged by SSIM < 0.98 OR ΔE-p95 > 3, and
  only if it **exceeds that task's own same-substrate drift** (the 2×2 diagonal is the per-task noise
  floor — not ≈1.0). Everything after the first flag is cascade, ignore it.
- (b) **confirmed** = **you open the contact sheet and look** (your vision — no OCR/VLM): is it a real
  substrate gap (functional, e.g. "command not found"; or visual, e.g. the terminal palette) or just
  replay drift (a menu opened instead of a terminal)? Only real gaps enter the ledger.
- (c) **reward-agreement** — corroborating secondary for FUNCTIONAL gaps (run the evaluator post-replay);
  a delta may also be per-substrate evaluator-impl drift (§3), so use it to attribute, not to score.

A gap leaves the ledger only at `verified` (a re-diff shows it closed). **Excluded** tasks
(unfixable container preconditions, e.g. anti-bot) are **not** in the denominator — excluding one
concedes a divergence, it doesn't close it.

## Template

```
### GAP-NN — <title>
- **task:** lite.osworld@osworld_<domain>_<uuid[:8]>  ↔  osworld@<uuid>
- **class:** functional | visual
- **direction/step:** lite→osworld | osworld→lite ; diverges at turn <k> (<action>)
- **evidence (required):** orig <turn_k/prompt_images/*.png or shared-layout fallback> | replay <turn_k.png> | diff <path> | reward orig/replay <r0>/<r1> (functional)
- **root cause:** <substrate difference>
- **VM-too? (decision test):** no → MIGRATION | yes → UPSTREAM
- **proposed resolution:** MIGRATION-fix (<widen sudo / bake tool / theme terminal / fix WM / fix save>) | exclude (`exclude_reason: <reason>`) | UPSTREAM/accepted  _(proposal only — maintainer decides)_
- **fix location:** <lite/gym/sandbox/docker | lite/gym/envs/lite/osworld/... | eval generator + regen>
- **status:** open | root-caused  _(bridge stops here)_ | fixed | verified | wontfix
- **verify:** <re-diff result / re-rollout reward>
```

## Worked example (from §2, pre-harness)

```
### GAP-01 — container `user` lacks general sudo
- **task:** lite.osworld@osworld_os_5812b315 ↔ osworld@<uuid>; also os_94d95f96
- **class:** functional
- **direction/step:** osworld→lite ; diverges at the privileged install / user-create step
- **evidence (required):** **live container probe (`probe_sudo.py`)** — `user` is in group `user` only (no sudo group); `sudo -ln` lists ONLY 3 NOPASSWD shims (`/usr/local/bin/{hostname,powerprofilesctl,timedatectl}`); `sudo -n useradd` / `sudo -n apt-get install` → **"sudo: a password is required"** (blocked). VM grants full sudo → passes. Confirms Dockerfile:236-259.
- **root cause:** sandbox base puts `user` in **no sudo group** (only 3 NOPASSWD shims). The VM's
  `user` IS in the `sudo` group and sudoes with the OSWorld password. **Not** a passwordless-ness diff —
  both require a password; the diff is group membership.
- **VM-too? (decision test):** no (VM `user` ∈ sudo group) → MIGRATION
- **proposed resolution:** MIGRATION-fix **matching the VM** (`usermod -aG sudo user` + `user:password`,
  NOT NOPASSWD) OR `exclude_reason` — **policy call, maintainer to adjudicate** (widen sudo weakens the
  container's deliberate isolation).
- **fix location:** `lite/gym/sandbox/docker` (add-to-sudo-group + password) or eval generator (exclude).
- **status:** root-caused (confirmed both substrates by live probe)
- **verify:** rerun the privileged step with the password on both — expect VM pass, lite unauthorized (current).
```

## Gaps

<!-- append filled GAP-NN blocks here as the sweep finds them -->

_First sweep: gpt-5.5, 16 lite tasks (same-substrate floor) + 6 terminal-heavy pairs (cross, both
directions), on the `--delay 30` `:mine` build. Recordings:
`.logs/rollout/gpt-5.5/lite.osworld/20260722_190106_250447` and `.../osworld/20260722_190612_643450`._

### GAP-02 — `ctrl+alt+t` chord silently dropped (lite MORE flaky than osworld)
- **task:** all terminal-opening tasks (`os_*`, `multi_apps_*`); measured on 8 lite + 4 osworld real-rollout chords + a fresh-env ablation
- **class:** functional
- **direction/step:** intrinsic to the action (not a replay artifact); the chord either opens a terminal or is lost
- **evidence (required):** ablation (fresh env, wmctrl detection) — lite zero-interval chord **5/10 cold (≈50%), 25% under load**; `--delay 30` **7/8 (88%)**, `--delay 100` **8/8**; osworld pyautogui.hotkey **7/8**. Real rollout with `--delay 30`: lite 7/8 opened, osworld 4/4. Root screenshots: `catt_after_montage` (lite), `osw_catt_after` (osworld).
- **root cause:** `xdotool key <chord>` fires all events zero-interval; the main key's KeyPress races the async modifier-state (XkbStateNotify) update in gsd/mutter, so the keybinding grab misses. Load-modulated (worse cold/under-load). NOT D-Bus server activation (pre-start didn't help), NOT grab-registration (settle didn't help), NOT a replay bug.
- **VM-too? (decision test):** the *flakiness* is UPSTREAM (osworld's pyautogui path is ~88% too) — BUT lite's single zero-interval chord was **worse** than osworld (≈50% vs ≈88%): that delta is MIGRATION-introduced.
- **superseded resolution:** PR #171 reverts the `server.py` chord delay and instead warms the
  `gnome-terminal` client path at sandbox boot. Treat the `--delay 30` measurements above as
  historical ablation evidence, not current code behavior.
- **fix location:** `lite/gym/sandbox/docker/Dockerfile.linux` (`warm-terminal.sh`) plus
  `lite/gym/sandbox/exec_stdio/server.py` (zero-delay `key` branch) — needs image rebuild.
- **status:** pending fresh verification on the rebuilt warm-terminal image.
- **verify:** re-ablation on the rebuilt image; confirm first-turn `ctrl+alt+t` opens reliably enough
  for lite/osworld parity.

### GAP-03 — terminal prompt is plain + titlebar static (container `user` has NO ~/.bashrc)
- **task:** lite.osworld@osworld_os_23393935 ↔ osworld@23393935-… (every gnome-terminal task)
- **class:** visual (two symptoms, one root cause + one fix; subsumes what would be GAP-06 "prompt color")
- **direction/step:** lite→osworld ; the terminal window (prompt row + titlebar)
- **evidence (required):** cropped prompt compare (`term_prompt_cmp.png`) — **lite prompt is plain WHITE** `user@user-virtual-machine:~$`; **osworld is the classic Ubuntu colored** prompt (green `user@host`, blue `~`). **Titlebar:** lite static **`Terminal`**, osworld dynamic **`user@user-virtual-machine: ~`**. Scalar is near-blind (both are small/colored regions: titlebar cross SSIM 0.9944 / ΔE 0 on floor 0.9998) — VISUAL catch.
- **root cause:** the container `user` has **no `~/.bashrc`** (base pre-created the home so `/etc/skel` was never copied). bash falls back to the bare default PS1 → plain prompt AND no `\e]0;\u@\h: \w\a` title escape → gnome-terminal's static profile title. The osworld VM's `user` carries the **stock Ubuntu `~/.bashrc`** (identical to the container's own `/etc/skel/.bashrc`) which colors the prompt (`*-256color` TERM path) and sets the window title.
- **VM-too? (decision test):** no (the VM renders colored + dynamic) → MIGRATION
- **resolution:** MIGRATION-fix — **maintainer adjudicated: FIX.** Seed the stock `/etc/skel/.bashrc` (+ `.profile`/`.bash_logout`) into `/home/user` **unchanged** (faithful to the VM, not "nicer").
- **fix location:** `lite/gym/envs/lite/osworld/docker/Dockerfile` (after the `.Xauthority` step). **Applied** — needs `:mine` rebuild.
- **status:** **verified** ✅
- **verify:** rebuilt `:mine` (`~/.bashrc` now present, 3771 B = stock skel); opened a terminal → prompt colored (green `user@host`, blue `~`) + titlebar dynamic `user@user-virtual-machine:…` — matches the VM. Evidence `verify_term_crop.png`.

### GAP-04 — LibreOffice first-run infobars present on osworld, suppressed on lite
- **task:** lite.osworld@osworld_multi_apps_00fa164e ↔ osworld@00fa164e-… (all `libreoffice_*` / `multi_apps` LibreOffice tasks)
- **class:** visual (+ it is the dominant open-loop DRIFT source for these domains)
- **direction/step:** lite→osworld ; diverges at the first LibreOffice Open dialog (turn_03), cascades
- **evidence (required):** osworld's LibreOffice shows **"Help us make LibreOffice even better!"** + **"Your donations support…"** infobars; lite (baked profile) shows none. `multi_apps_00fa164e` floor **0.9998** → cross **0.51–0.56**; **generalizes — `libreoffice_calc_01b269ae` floor 0.9987 → cross 0.47–0.50** (same banners, Calc). The banners push menus/dialogs down, so lite's recorded click coords land wrong on osworld and the trajectory derails. See `cross_00fa164e` / `cross2_visual` montages. **Confirmed across Writer + Calc ⇒ all `libreoffice_*`.**
- **root cause:** lite bakes a LibreOffice profile that suppresses the first-run tip/donation infobars; osworld's fresh VM shows them every snapshot.
- **VM-too? (decision test):** yes (the original VM shows them) → UPSTREAM
- **proposed resolution:** **UPSTREAM / accepted — DO NOT FIX** (lite is intentionally cleaner; re-adding clutter + layout jitter is not worth it). **Maintainer-confirmed: don't fix.** Consequence to record: this is a **known cross-substrate drift source** for `libreoffice_*`/`multi_apps` → per §Caveats those domains are **scalar-OFF, visual-only**; benign for closed-loop eval (agent adapts to lite's own layout).
- **fix location:** n/a (accepted)
- **status:** wontfix

### GAP-05 — GNOME quick-settings (system) menu content differs
- **task:** lite.osworld@osworld_os_28cc3b7e ↔ osworld@28cc3b7e-…
- **class:** visual
- **direction/step:** lite→osworld & osworld→lite ; top-right quick-settings menu
- **evidence (required):** osworld menu richer (**Wired Connected / Balanced / Lock** + a "Software Updates available" notification); lite sparser (volume / Settings / Power Off). Same-substrate floor **0.9999** → cross **0.9646** (both directions) — large enough that the **scalar DID flag it** (< 0.98) AND visually confirmed (`cross_visual_l2o`).
- **root cause:** the container's GNOME quick-settings applets / notification stack differ from the VM (missing NetworkManager "Wired", power-profiles "Balanced", screen-lock entries).
- **VM-too? (decision test):** no → MIGRATION
- **proposed resolution:** MIGRATION-fix candidate (align the container's gnome-shell quick-settings applets) — **maintainer to adjudicate** (agents rarely use this menu; low priority).
- **fix location:** `lite/gym/sandbox/docker` (gnome-shell extensions / applet set).
- **status:** root-caused


# Fix Plan
