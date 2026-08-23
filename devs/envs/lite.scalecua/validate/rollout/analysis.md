# lite.scalecua Rollout Analysis

This file is the durable, reviewable summary for rollout result analysis. Raw
per-trajectory artifacts stay under `.exps/validate/lite.scalecua/**`; do not
commit screenshots, prompt parquet files, or full rollout logs.

Historical notes below may cite retired warm-pool validation shards. Treat them
as old evidence labels only; new ScaleCUA validation should not add warm-pool
flags or warm-pool closeout criteria.

## Artifact Layout

- Active machine-readable audit:
  `.exps/validate/lite.scalecua/batch/<active-1000-root>.visual_audit*.jsonl`
- Active scan queue:
  `.exps/validate/lite.scalecua/batch/<active-1000-root>.audit_queue.jsonl`
- Rollout command/evidence index:
  `devs/envs/lite.scalecua/validate/rollout/logs.md`
- Durable taxonomy, aggregate results, and action items:
  this file.

## Canonical Fields

- `domain` means `metadata.others.domain`, matching `lite.osworld`.
- `related_app_domain` is only an audit hint derived from ScaleCUA
  `metadata.scalecua.related_apps` / `snapshot`; never use it for split/domain
  sampling or registry filtering.
- Final success rates are computed from `visual_label`, not raw reward.
- Any disagreement between visual state and reward must be triaged against the
  rollout screenshots/actions and the relevant SCALE-CUA task JSON,
  generated getter/metric, `desktop_env.evaluate()` flow, and
  `osworld_env/worker` patch/dispatcher behavior.

## Visual Labels

| Label | Meaning | Counts as success? |
| --- | --- | --- |
| `true_success` | Reward and visual state agree that the task succeeded. | yes |
| `true_failure` | Reward and visual state agree that the task failed. | no |
| `partial_success` | Partial reward and visual/action evidence show only part of the task criteria were satisfied. | no |
| `false_success` | Reward says success but visual/task state is wrong. | no; blocks gate until explained |
| `false_failure` | Reward says failure or partial, but visual/task state appears fully correct. | pending; requires evaluator probe or upstream issue |
| `setup_failure` | Reset/setup did not create the intended task state. | no; adapter/setup bug unless upstream task is bad |
| `action_parse_failure` | Model action was parsed/executed incorrectly by CUA-Lite. | no; adapter/action bug |
| `transient_failure` | External/network/model/server transient caused an otherwise valid run to fail. | no for this attempt; may be rerun |
| `not_visually_decidable` | Screenshot cannot prove the hidden/config state being evaluated. | excluded from visual numerator until probed |
| `ambiguous_needs_evaluator_probe` | Visual evidence is insufficient or stale; run a targeted evaluator/file probe. | excluded until resolved |

## Superseded Diagnostic Batch 2026-07-14

Command evidence is recorded in `logs.md`; raw artifacts are in
`.exps/validate/lite.scalecua/batch/gpt-5.5-300`.

Current interim notes:

- Batch uses a dedicated env-server on port `30178`.
- Rollout worker concurrency is `8`.
- Prompt-data was 300 tasks: 150 `train`, 150 `rl`, 15 per
  `metadata.others.domain` per split.
- Visual audit started while rollout was still running.
- First confirmed evaluator mismatch:
  `scalecua_osworld_train_chrome_6766f2b8_8a72_417f_a9e5_56fcaa735837_task_verify_2`
  is a likely `false_failure`; Chrome shows `Hello Extensions 1.0` loaded but
  reward is `0`.
- First evaluator compatibility issue observed in server logs:
  ScaleCUA overlay file getter attempted `env.controller.*` against the
  CUA-Lite container handle. This must be classified per affected task as
  adapter compatibility, not model failure, unless a shim or normalized getter
  already covers it.
- This diagnostic batch was stopped after the compatibility issue was confirmed.
  It is not an acceptance gate and must not be used for success-rate reporting.

## Batch 2026-07-14 1000-Task Gate

Acceptance prompt artifacts:

- prompt data: `.exps/validate/lite.scalecua/batch/gpt-5.5-1000.prompt.parquet`
- manifest: `.exps/validate/lite.scalecua/batch/gpt-5.5-1000.manifest.json`

Sampling target: 1000 runnable tasks, 500 `train` and 500 `rl`, exactly
50 tasks per `metadata.others.domain` per split.

No 1000-task acceptance root is currently active while the Chrome profile alias
fix is being targeted-rerun. Restart the gate only after the targeted rerun
passes and 30283 is cleanly shut down.

### Pre-gate Diagnostic Fix: Canonical Getter Precedence

The first 1000-task attempt on env-server port `30179` was stopped before it
became an acceptance run. It exposed a systematic evaluator issue, not a model
failure: a Chrome password-manager task visually reached
`chrome://password-manager/passwords`, but reward was `0` because a generated
ScaleCUA getter for `active_url_from_accessTree` parsed the accessibility tree
only and overrode the more robust `lite.osworld` CDP-first active URL getter.

Fix: `lite.scalecua.src.osworld.verify` now sends explicit OSWorld canonical
result/expected types to `lite.osworld.src.eval.runner` before consulting
ScaleCUA overlay getters. Overlay getters remain active for generated/custom
types outside that allowlist.

Validation evidence:

- unit/static regression: `100 passed, 10 skipped`;
- targeted rollout:
  `.exps/validate/lite.scalecua/gpt-5.5-active-url-fix-20260714`;
- targeted task:
  `scalecua_osworld_train_chrome_12086550_11c0_466b_b367_1d9e75b3910e_task_verify_14`;
- reward after fix: `1.0`;
- visual check: final screenshot shows Google Password Manager at
  `chrome://password-manager/passwords`.

The partial pre-fix rollout artifacts were moved to
`.exps/validate/lite.scalecua/batch/gpt-5.5-1000-pre-canonical-getter-fix-20260714`
and must not be included in acceptance reporting.

### Pre-gate Diagnostic Fix: Python Command Output Cleanup

The second 1000-task attempt on env-server port `30179` was also stopped before
acceptance. It exposed another evaluator compatibility issue: a Chrome settings
task visually selected the requested setting, but reward was `0` because
`execute_python_command()` injected `pyautogui` for every command. Importing
`pyautogui` printed `Xlib.xauth` warnings before the Chrome Preferences path,
and generated getters treated the combined warning+path text as an invalid file
path.

Fix: the DesktopEnv adapter now injects the pyautogui prefix only for commands
that reference `pyautogui`, and strips known Xlib warning lines from command
stdout/stderr before returning them to generated getters.

Validation evidence:

- targeted rollout:
  `.exps/validate/lite.scalecua/gpt-5.5-python-command-fix-20260714`;
- targeted task:
  `scalecua_osworld_train_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_task_verify_8`;
- reward after fix: `1.0`;
- visual check: final screenshot shows the requested Chrome setting state.

The partial pre-fix rollout artifacts were moved to
`.exps/validate/lite.scalecua/batch/gpt-5.5-1000-pre-python-command-cleanup-fix-20260714`
and must not be included in acceptance reporting.

### Pre-gate Compatibility Fix: Official Getter and Score Semantics

Code audit against SCALE-CUA official `DesktopEnv.evaluate()` found remaining
drift before restarting the acceptance batch:

- official-only bare Chrome getters such as `enabled_experiments`,
  `chrome_language`, and `enable_enhanced_safety_browsing` must fall back to
  upstream `desktop_env.evaluators.getters` after ScaleCUA overlay lookup;
- local eval must preserve official raw/partial scores instead of binarizing;
- non-`infeasible` tasks with a final FAIL action must return `0` before metric
  evaluation.

Fix: `judges.resolve_getter()` now has upstream getter fallback,
`evaluate_scalecua_task()` matches official single/multi metric aggregation, and
`audit_queue.py` queues `reward_partial` rows for visual review.

Validation evidence:

- unit/static regression: `16 passed`;
- broader family/warm-pool regression: `105 passed, 10 skipped`.

### Pre-gate Compatibility Fix: VLC HTTP Auth and Infeasible Args

The next 1000-task attempt was stopped after 25 completed trajectories because a
code audit found that the sample would later hit non-canonical VLC generated
getters using `requests.get(..., auth=('', password))`. The request router sent
kwargs into the container through JSON, which converted the auth tuple to a list;
in-container `requests` expects a tuple or auth object. The same audit found that
`func == "infeasible"` handled dict action arguments but not raw JSON-string
tool-call arguments.

Fix: request kwargs now preserve tuples with an explicit marker and decode them
recursively inside the container, while `_reported_infeasible()` accepts
JSON-string `terminate(status="failure")` and `[infeasible]` response arguments.

Validation evidence:

- sampled task scan: 78 VLC result getters, 38 non-canonical VLC generated
  getters, first non-canonical hit at prompt index 402;
- unit/static regression: `18 passed`;
- broader family/warm-pool regression: `107 passed, 10 skipped`.

The partial pre-fix rollout artifacts were moved to
`.exps/validate/lite.scalecua/batch/gpt-5.5-1000-pre-vlc-auth-fix-20260714`
and must not be included in acceptance reporting.

### Pre-gate Compatibility Fix: Extension Path Getter and vm_file Bytes

The pre-VLC-auth visual audit identified two additional false-failure risks:

- extension-install tasks with `find_unpacked_extension_path` can look correct
  in Chrome but fail if ScaleCUA train/rl is routed to the narrower base runner
  instead of the official/upstream DesktopEnv getter;
- generated Excel metrics such as `check_xlsx_formula__a2439b25` can visually
  pass but return `0` when a `vm_file` local path is passed to a metric that
  explicitly expects `bytes`.

Fix: `find_unpacked_extension_path` was removed from the ScaleCUA base-runner
allowlist, and metric invocation now converts `vm_file` local paths to bytes
only when the resolved metric first parameter is annotated `bytes`.

Validation evidence:

- unit/static regression: `20 passed`;
- broader family/warm-pool regression: `109 passed, 10 skipped`.

### Pre-gate Compatibility Fix: Chrome Internal Active URL Fallback

The env-server port `30279` attempt was stopped after 41 completed
trajectories and is diagnostic only. It exposed a narrower active URL edge case
than the earlier canonical-getter issue:

- task
  `scalecua_osworld_train_chrome_12086550_11c0_466b_b367_1d9e75b3910e_task_verify_41`
  asks to navigate to Chrome's bookmark manager;
- final screenshot showed `chrome://bookmarks`;
- evaluator expected `chrome://bookmarks/` through
  `active_url_from_accessTree` + `is_expected_active_tab_approximate`;
- the page focuses the in-page bookmark search box, so AT-SPI can expose
  non-URL text even when CDP still knows the active internal page URL.

Fix: `active_url_from_accessTree` remains AT-first for normal pages, but
`lite.scalecua` now has a narrow CDP fallback for Chrome internal pages. It
canonicalizes a matching `chrome://...` URL, ignores omnibox popup/newtab
pseudo-pages, and refuses to guess when multiple non-popup internal pages are
present.

Validation evidence:

- unit/static regression after the first implementation: `36 passed`;
- broader family/warm-pool regression after the first implementation:
  `125 passed, 10 skipped`;
- old diagnostic root:
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000` completed 41 rows and must
  not be used for success-rate reporting;
- diagnostic restart:
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-activeurlfix`;
- same bookmarks task returned reward `1.0` on the fresh `30280` diagnostic
  run.

### Pre-gate Compatibility Fix: Narrow No-AT Chrome Internal Fallback

The env-server port `30280` attempt was stopped after 11 completed
trajectories. It confirmed the bookmark-manager reward fix, but code review
found the no-accessibility-tree branch was still too broad: if a normal web
task temporarily produced no AT XML while a single `chrome://...` page existed,
the fallback could return that internal page URL and create a false success.

Fix: the no-AT CDP fallback is now allowed only when the evaluator target is a
Chrome internal page (`goto_prefix == ""` or a `chrome://...` prefix). Normal
web URL checks still require the AT-derived path. Raw AT URLs such as
`chrome://bookmarks` may still be canonicalized to `chrome://bookmarks/` when
CDP has a matching internal page candidate.

Validation evidence:

- unit/static regression: `37 passed`;
- broader family/warm-pool regression: `126 passed, 10 skipped`;
- diagnostic root:
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-activeurlfix` completed 11
  rows and must not be used for success-rate reporting;
- diagnostic restart root:
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-internalfallbackfix`;
- diagnostic restart env-server: port `30281`, token
  `lite-scalecua-1000-final-20260714`;
- same bookmarks task returned reward `1.0` on the `30281` diagnostic run.

VLC minimal-view diagnostic note: task
`scalecua_osworld_train_chrome_215dfd39_f493_4bc3_a027_8a97d72c61bf_task_verify_37`
is not an evaluator migration bug when reward is `0` after pressing `Ctrl+H`.
Its evaluator reads persisted `vlcrc` via `vlc_config` and
`check_qt_minimal_view`; the screenshot can show current minimal UI without
proving `qt-minimal-view=1` was saved across the postconfig relaunch.

### Pre-gate Abort: Request Shim Script and VLC Auth Alignment

The env-server port `30281` attempt was stopped after 26 completed trajectories
and is diagnostic only. Raw rewards before abort were 14 exact successes, 11
failures, and 1 partial reward. Visual audit wrote two shards:

- `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-internalfallbackfix.visual_audit.shard_a.jsonl`
  with 26 rows;
- `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-internalfallbackfix.visual_audit.shard_b.jsonl`
  with 20 rows.

Root causes and decisions:

- `EvalEnvShim.request_in_container()` generated a `python3 -c` request script
  with incorrectly indented `try`/`except` output blocks. Existing tests covered
  tuple/bytes encoding but did not compile the actual generated script. The
  fix factors script construction into `_build_request_script()` and adds a
  compile-level regression test.
- `vlc_playing_info` used `curl --user :password`, while `lite.osworld`
  dispatch/server and base runner use the `vlc` password. The fix aligns
  `lite.scalecua` with the base runner fallback sequence: no auth, `:vlc`,
  then `:a`.
- Explicit evaluator `postconfig` now runs before terminal failure/infeasible
  checks for every split, and a per-evaluation deep copy carries
  `_postconfig_done` into base OSWorld eval so postconfig is not run twice.
- Chrome startup and extension visual `false_failure` rows are not direct
  evidence to replace official getters. The relevant generated getters read
  Chrome `Preferences` / extension settings, so screenshots can show a current
  UI state before Chrome has flushed the persisted state. These cases must be
  tagged `ambiguous_needs_evaluator_probe` or `not_visually_decidable` until a
  targeted file/getter probe proves migration drift.

Validation evidence:

- unit/static regression: `39 passed`;
- broader family/warm-pool regression: `128 passed, 10 skipped`;
- cleanup left no `lite-env-30281` / `lite.scalecua` containers.

### Pre-gate Abort: Chrome Profile Alias Drift

The env-server port `30282` attempt was stopped after 50 completed
trajectories and is diagnostic only. Raw rewards before abort were 18 exact
successes, 29 failures, and 3 partial rewards. Visual audit wrote:

- `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-requestshimfix.visual_audit.shard_a.jsonl`
  with 20 rows;
- `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-requestshimfix.audit_queue.jsonl`
  with the live queue at the time of abort.

Root cause:

- `lite.osworld` setup and Chrome launch redirect the real Chrome profile from
  `/home/user/.config/google-chrome` to `/home/user/chrome-data`.
- ScaleCUA generated and upstream fallback getters still use official DesktopEnv
  paths such as `/home/user/.config/google-chrome/Default/Preferences`,
  `~/.config/google-chrome`, and
  `os.path.join(HOME, '.config', 'google-chrome', ...)`.
- Screenshots for extension/startup tasks can look correct while reward is `0`
  because the generated getter read the stale official path instead of the
  profile Chrome actually used.

Fix:

- `EvalEnvShim` now aliases official Chrome/Chromium profile paths to
  `/home/user/chrome-data` for `get_file`, `get_vm_file`,
  `get_vm_file_content`, directory tree requests, and all command-entry helpers
  (`run_bash_script`, `execute_python_command`, `execute`, `run_command`).
- `lite.scalecua.verify` deep-aliases evaluator config strings before handing
  base-runner result types such as `vm_command_line` / `vm_command_error` to
  `lite.osworld`, so this fix does not modify `lite.osworld`.
- The alias helper covers absolute `/home/user` and `/home/ubuntu` paths,
  `~`/`$HOME` paths, Snap/Chromium paths, and upstream split-join Python forms.

Validation evidence before targeted rollout:

- unit/static regression: `42 passed`;
- broader family/warm-pool regression: `131 passed, 10 skipped`;
- cleanup left no `lite-env-30282-*` containers.

Targeted rerun in progress:

- env-server: port `30283`, token `lite-scalecua-chromealias-20260714`;
- prompt data:
  `.exps/validate/lite.scalecua/targeted/chrome-alias.prompt.parquet`;
- rollout root:
  `.exps/validate/lite.scalecua/targeted/gpt-5.5-chrome-alias`;
- tasks:
  `scalecua_osworld_train_chrome_6766f2b8_8a72_417f_a9e5_56fcaa735837_task_verify_2`,
  `scalecua_osworld_train_chrome_3299584d_8f11_4457_bf4c_ce98f7600250_task_verify_11`,
  and
  `scalecua_osworld_train_chrome_6766f2b8_8a72_417f_a9e5_56fcaa735837_task_verify_66`.

### Pre-gate Diagnostic Fix: Universal Chrome Profile Flush

The extension/startup targeted rerun proved that Chrome profile path aliasing
alone was insufficient. Some generated and base-runner getters read
`Preferences`, `Local State`, extension manifests, bookmarks, or history while
Chrome is still running. Screenshots can already show the desired UI state while
the on-disk profile files have not flushed yet.

Fix: `lite.scalecua.verify` now terminates Chrome before any known
Chrome-profile-backed result type or config-path marker is evaluated. This is
kept local to `lite.scalecua`; `lite.osworld` is unchanged.

Validation evidence:

- targeted root:
  `.exps/validate/lite.scalecua/targeted/gpt-5.5-profileflush`;
- targeted tasks:
  extension version, startup-page removal, and unpacked-extension-path probes;
- result: `3/3` reward `1.0`;
- the startup-page probe logged
  `Current startup URLs: ['http://news.ycombinator.com/']` and matched the
  expected removal of `http://reddit.com/`.

### Pre-gate Diagnostic Fix: Recreation.gov Current Grid + AT Fallback

The `30288` diagnostic root was stopped before acceptance after code audit
found that the generated `recreation_devilsgarden_html__...` getter would use a
host-side Playwright/CDP connection that is not routable from the CUA-Lite
host. A local CDP parser fixed the infrastructure issue, but the first targeted
rerun returned reward `0` even though screenshots showed the current
Recreation.gov availability grid. Root cause: the official generated task still
expected the old `camp-sortable-column-header` class, while the current site
renders a different grid structure.

The next 1000-task attempt on `30291` exposed a second version of the same
problem: a final screenshot for `task_verify_29` visibly showed the Devils
Garden availability grid, but the CDP page content available to the evaluator
did not contain the grid text, so the result could still be a false failure.

Fix: `lite.scalecua.verify` now handles the Devil's Garden hashed result type
locally, reads the active Recreation.gov page through container CDP, recognizes
both the old class marker and the current campsite grid text/date structure, and
falls back to the container AT-SPI tree when CDP content is incomplete. The AT
fallback is intentionally narrow: it requires Devils Garden/Recreation.gov
location evidence, campsite grid terms, multiple date columns, and availability
status markers. It returns the official-shaped fields required by
`check_recreation_html_element__fa1e76...`, including
`reservation_table_present`, `has_availability_data`, `reservation_dates`,
`dates_sorted`, and `earliest_reservation_identified`.

Validation evidence:

- unit/static regression: `48 passed`;
- broader family/warm-pool regression: `137 passed, 10 skipped`;
- pre-fix targeted root:
  `.exps/validate/lite.scalecua/targeted/gpt-5.5-recreationcdp-ready`,
  result `0/2` with reward `0.0`; screenshots showed the grid, so this root
  is diagnostic only;
- post-fix targeted root:
  `.exps/validate/lite.scalecua/targeted/gpt-5.5-recreationgridfix`,
  result `2/2` reward `1.0`;
- AT-fallback targeted root:
  `.exps/validate/lite.scalecua/targeted/gpt-5.5-recreation-atfallback`,
  result `2/2` reward `1.0`;
- visual check: both AT-fallback final screenshots show the Devils Garden
  availability table/grid, including Juniper Basin available on WED 15.

### Stopped Diagnostic Root: 1000-Task Profileflush + Recreation Grid Fix

The root below is diagnostic only and must not be counted as acceptance:

`.exps/validate/lite.scalecua/batch/gpt-5.5-1000-profileflush-recreationgridfix`

It was stopped after the CDP/AT mismatch above was found. Final diagnostic
queue:

- env-server port `30291`;
- token `lite-scalecua-1000-profileflush-recreationgridfix-20260714`;
- rollout concurrency `8`;
- env-server `max-live-envs=12`;
- queue artifact:
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-profileflush-recreationgridfix.audit_queue.final-diagnostic.jsonl`;
- scanned `54`, completed `46`, in-progress at interrupt `8`;
- reward `1.0`: `22`, reward `0.0`: `21`, partial reward: `3`;
- completed coverage was only `train/chrome`, so it cannot satisfy the
  per-domain 1000-task gate.

Early visual notes:

- Chrome startup-page task
  `scalecua_osworld_train_chrome_3299584d_8f11_4457_bf4c_ce98f7600250_task_verify_11`
  now returns reward `1.0`, confirming the profile flush fix in the active
  batch.
- Macy's URL/filter tasks currently fail while staying on Google search
  result pages or due live-site access behavior. Classify these as
  true/agent/live-site failures unless later evaluator probes show URL parsing
  drift.
- Chrome profile-name task
  `scalecua_osworld_train_chrome_2ae9ba84_3a0d_4d4c_8338_3a1478dc5fe3_task_verify_1`
  visually shows `Sarah` in the focused field but returns reward `0`. Because
  `profile_name` is already covered by the profile-flush gate, this is likely
  a UI commit/blur ambiguity and should be tagged
  `ambiguous_needs_evaluator_probe` or `false_failure` only after a targeted
  persisted-profile probe.
- Recreation.gov tasks
  `task_verify_26` and `task_verify_29` are superseded by the AT-fallback
  targeted root above.

Cleanup evidence:

- env-server port `30291` was stopped;
- `docker ps -a --format '{{.Names}} {{.Status}}' | rg '^lite-env-30291-'`
  returned no containers;
- `ss -ltnp | rg ':30291'` returned no listener.

### Stopped Diagnostic Root: 1000-Task Profileflush + Recreation AT Fallback

The root below is diagnostic only and must not be counted as acceptance:

`.exps/validate/lite.scalecua/batch/gpt-5.5-1000-profileflush-recreation-atfallback`

It was stopped after live visual/code audit found two evaluator compatibility
bugs:

- `default_search_engine` can return Chrome's localized built-in name
  `Yahoo! Hong Kong`, while generated Yahoo tasks expect `Yahoo` / `Yahoo!`;
- generated Apple comparison tasks place `ignore_list_order` inside
  `rules.expected`, while the generated overlay metric reads the flag from the
  top-level rules dict.

Fixes landed in `lite.scalecua` only:

- normalize Yahoo-family `default_search_engine` results to `Yahoo!`;
- hoist `check_direct_json_object` `expected.ignore_list_order` to
  `rules.ignore_list_order` before calling the generated metric.

Validation evidence:

- `uv run --no-sync pytest tests/gym/envs/lite/test_scalecua.py -q`:
  `50 passed, 3 warnings`;
- broader family/warm-pool regression:
  `139 passed, 10 skipped, 3 warnings`.

Final diagnostic queue:

- env-server port `30293`;
- token `lite-scalecua-1000-profileflush-recreation-atfallback-20260714`;
- queue artifact:
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-profileflush-recreation-atfallback.audit_queue.final-diagnostic.jsonl`;
- scanned `61`, completed `56`, in-progress at interrupt `5`;
- reward `1.0`: `30`, reward `0.0`: `23`, partial reward: `3`;
- completed coverage: `train/chrome=50`, `train/gimp=6`.

Early manual visual audit:

- artifact:
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-profileflush-recreation-atfallback.visual_audit.early.manual.jsonl`;
- labels: `true_success=3`, `true_failure=2`,
  `ambiguous_needs_evaluator_probe=1`;
- confirmed diagnostic mismatches include the Yahoo localized-name false
  failure and Apple `ignore_list_order` metric-shape false failure.

Cleanup evidence:

- env-server port `30293` was stopped;
- `docker ps -a --format '{{.Names}} {{.Status}}' | rg '^lite-env-30293-'`
  returned no containers;
- `ss -ltnp | rg ':30293'` returned no listener.

### Stopped Diagnostic Root: 1000-Task Compatfix

The root below is diagnostic only and must not be counted as acceptance:

`.exps/validate/lite.scalecua/batch/gpt-5.5-1000-compatfix`

It was stopped after visual audit found a systematic Chrome profile-name false
failure: `scalecua_osworld_train_chrome_2ae9ba84_3a0d_4d4c_8338_3a1478dc5fe3_task_verify_1`
showed the profile name input set to `Sarah`, but reward stayed `0.0`.
`lite.osworld` perturb tasks already add a `Tab`/`Escape` blur before killing
Chrome, because Chrome only commits the focused profile-name input after blur.
The initial `lite.scalecua` profile flush killed Chrome without that blur.

Fix landed in `lite.scalecua` only:

- `_flush_chrome_profile` sends `xdotool key Tab`, `xdotool key Escape`, then
  `pkill -TERM chrome`;
- tests assert the blur happens before the Chrome kill for `profile_name`,
  extension, and other Chrome profile-backed getters.

Validation evidence:

- `uv run --no-sync pytest tests/gym/envs/lite/test_scalecua.py -q`:
  `51 passed, 2 warnings`;
- broader family/warm-pool regression:
  `140 passed, 10 skipped, 2 warnings`.

Final diagnostic queue:

- env-server port `30294`;
- token `lite-scalecua-1000-compatfix-20260714`;
- rollout concurrency `8`;
- env-server `max-live-envs=12`;
- final queue artifact:
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-compatfix.audit_queue.final-diagnostic.jsonl`;
- scanned `48`, completed `45`, in-progress at interrupt `3`;
- reward `1.0`: `27`, reward `0.0`: `16`, partial reward: `2`;
- completed coverage: `train/chrome=45`.

Cleanup evidence:

- env-server port `30294` was stopped;
- `docker ps -a --format '{{.Names}} {{.Status}}' | rg '^lite-env-30294-'`
  returned no containers after targeted cleanup;
- `ss -ltnp | rg ':30294'` returned no listener.

### Stopped Diagnostic Root: 1000-Task Profileblur Compatfix

The root below is diagnostic only and must not be counted as acceptance:

`.exps/validate/lite.scalecua/batch/gpt-5.5-1000-profileblur-compatfix`

It was stopped after visual audit found GIMP filter/dialog false failures and a
key normalization action-bridge failure:

- GIMP tasks visually opened the requested filter dialogs, but official
  `action-history` evaluation returned reward `0` after postconfig closed the
  application before the history file reliably flushed.
- One LibreOffice Calc trajectory emitted a lone `+` keypress, which CUA-Lite
  incorrectly left as literal `+`; `keys.to_xdotool` rejects separator-like
  tokens and the trajectory failed before evaluation.

Final diagnostic queue:

- fresh env-server port `30295`;
- token `lite-scalecua-1000-profileblur-compatfix-20260714`;
- rollout concurrency `8`;
- env-server `max-live-envs=12`;
- final queue artifact:
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-profileblur-compatfix.audit_queue.final-diagnostic.jsonl`;
- scanned `138`, completed `130`, in-progress at interrupt `8`;
- reward `1.0`: `73`, reward `0.0`: `49`, partial reward: `8`;
- completed coverage by sampled domain: `train/chrome=50`,
  `train/gimp=50`, `train/libreoffice_calc=30`;
- cleanup: port `30295` stopped, `lite-env-30295-*` removed, no `:30295`
  listener.

Fix status:

- `lite.scalecua` now captures the pre-postconfig active GIMP window for
  `action-history` evaluators and augments only known GIMP filter/dialog
  history tokens when the window evidence matches.
- Historical correction: this key-plus compat fix made lone `+` usable as a
  literal glyph. Current Lite storage uses `+`; `plus` is only an accepted raw
  alias before normalization.
- Regression slice after both fixes: `668 passed, 10 skipped`.

### Current Acceptance Run: 1000-Task GIMP Window + Key-Plus Compatfix

The active acceptance root is:

`.exps/validate/lite.scalecua/batch/gpt-5.5-1000-gimpwindow-keyplus-compatfix`

Run configuration:

- fresh env-server port `30297`;
- token `lite-scalecua-1000-gimpwindow-keyplus-compatfix-20260714`;
- rollout concurrency `8`;
- env-server `max-live-envs=12`;
- `max-attempts=1`, so each row represents one trajectory;
- cleanup must only touch `lite-env-30297-*`;
- visual audit must be written to
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-gimpwindow-keyplus-compatfix.visual_audit.jsonl`.

Live audit checkpoint after the first 40 completed trajectories:

- raw rewards: `21` exact successes, `16` failures, `3` partial rewards;
- visual labels: `true_success=17`, `false_failure=6`, `true_failure=9`,
  `partial_success=3`, `not_visually_decidable=4`, `blocked_upstream=1`;
- the observed `false_failure` rows are not action-bridge regressions. They are
  narrow live-site drift cases where the screenshot and response show the
  intended page/content but the official URL/filter expectation is stale:
  Virginia DMV path changes, DOJ Forms filter id `401 -> 376`, FlightAware
  category/error drift, and United special-assistance path changes;
- continue the current batch unless a migration/evaluator bridge bug appears.
  If these live-site drift classes repeat heavily, choose one explicit policy
  before the next acceptance run: add narrow URL aliases with tests, or assign
  `exclude_reason` to the affected generated/live-site rows and regenerate the
  1000-task prompt sample.

### Diagnostic Run: 1000-Task Postconfig-Hardened

The active diagnostic root is:

`.exps/validate/lite.scalecua/batch/gpt-5.5-1000-postconfig-hardened`

Run configuration:

- env-server port `30314`;
- token `lite-scalecua-1000-postconfig-hardened-20260714`;
- rollout concurrency `8`;
- env-server `max-live-envs=12`;
- `max-attempts=1`;
- prompt data:
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-instructionfilter.prompt.parquet`.

Current status checkpoint:

- completed trajectories: `254`;
- raw rewards: `138` exact successes, `102` failures, `14` partial rewards;
- completed domains so far: `chrome=49`, `gimp=50`,
  `libreoffice_calc=50`, `libreoffice_impress=50`,
  `libreoffice_writer=49`, `multi_apps=6`;
- this root is diagnostic for clean accounting because the catalog and adapter
  have changed while it was running. Notable post-launch changes include
  `missing_reference_asset`, `upstream_live_site_drift`,
  `instruction_setup_mismatch`, path-reference materialization, and the exact
  `upstream_generated_eval_bug` filter for the Calc sort-range row.

Visual audit status:

- shard 02 produced 30 rows and found two confirmed generated-judge concerns
  after code audit: the Calc sort-range row is now filtered as
  `upstream_generated_eval_bug`; the Impress title-format getter remains a
  recorded latent upstream fragility but is not filtered without a positive
  evaluator probe.
- shard 03 produced 30 rows:
  `true_success=14`, `true_failure=4`, `suspected_false_failure=6`,
  `partial=6`, `suspected_false_success=0`. The six suspected false failures
  are currently under code audit against task cache, generated getter/metric
  code, rollout actions, and screenshots.

Clean accounting rule:

- Do not use this root for the final >=70% acceptance rate.
- After code/filter changes settle, regenerate or revalidate prompt data with
  `check_prompt_data.py`, start a fresh env-server, and run the 1000-task gate
  with `--concurrency 8`.

## Required Final Summary

When the batch completes, add:

- raw reward summary by split and `metadata.others.domain`;
- partial reward summary by split and `metadata.others.domain`;
- visual label counts by split and `metadata.others.domain`;
- success rate from visual labels overall, for `train`, and for `rl`;
- list of every `false_success`, `false_failure`, `setup_failure`,
  `action_parse_failure`, `transient_failure`, and evaluator compatibility
  issue;
- targeted fixes or upstream issue links for each non-model failure class;
- cleanup evidence for the 1000-task env-server and current active
  `lite-env-<port>-*` containers.
