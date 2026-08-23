# lite.scalecua Rollout Logs

Historical entries may mention old fixed-tag or `:latest` experiments. Current
ScaleCUA rollout uses the normal `cua-lite/lite.osworld:latest` lifecycle and a
fresh dedicated env-server for large batches.

Historical commands below may include retired warm-pool flags or warm-pool test
names. Do not copy those into new validation runs; current env-server validation
uses cold instances or explicit warm singleton paths only.

Each entry should include date/commit, asset identity, base image ID/digest,
command, task id, split/source/domain, log root, reward/result summary, visual
inspection notes, and failure classification.

Current rollout logs store canonical trajectory images under
`sample_*/images/*.png`, referenced by `trajectory.parquet` and
`04_results.json` image refs. When debug artifacts are enabled, per-turn prompt
image caches live under `turn_*/prompt_images/<basename>.png` with matching
overlays under `turn_*/prompt_images_annotated/<same basename>.png`; turn
metadata uses `01_prompt.txt`, `02_response.txt`, `03_actions.json`,
`04_results.json`, and optional `05_timing.json`. Older entries below may cite
legacy per-turn `images/`, `annotated/`, `05_results.json`, `06_timing.json`,
`result_images/`, `00_screenshot.png`, or `04_screenshot_annotated.png`; those
names are historical evidence only, not current layout guidance.

## 2026-07-14 gpt-5.5 rl Smoke

Commit: `7922529b`

Asset identity:

```json
{"components":{"generated_judges":"osworld/judge_functions/generated_tasks","generated_tasks":"osworld/generated_tasks","rl_judges":"osworld/judge_functions/rl_tasks","rl_tasks":"osworld/rl_tasks"},"file_cache_revision":"711e0811642364e7aa8f10a8918367d0b626d578","local_eval_manifest":{"file_count":548,"path":"SCALE-CUA/osworld_eval/evaluation_examples","sha256":"640554f7c3f79d9f51cd95f7baf84223a50bb06d5f55bdb613237ce2cdb7e9d0"},"repo":"extreme1228/ScaleCUA","revision":"77d7174d45d36e3c355269699d7f59a90a714ce6"}
```

Deprecated historical base image from an old validation experiment:
`cua-lite/lite.osworld:latest`
`sha256:401b160c983fcf3c47a0480f286b6f29eb73d7d153e35a075a755011d3a0276e`

Do not reuse this digest as current evidence. Current ScaleCUA validation uses
the normal `cua-lite/lite.osworld:latest` lifecycle and a fresh dedicated
env-server.

Task:
`scalecua_osworld_rl_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_traj_verify_0`

Instruction: `Set Chrome's default font size to Very Large.`

Split/source/domain: `rl`, `rl_tasks`, `chrome`

Direct command:

```bash
env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
  uv run --no-sync python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id lite.scalecua \
  --splits rl \
  --head 1 \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --concurrency 1 \
  --max-attempts 1 \
  --config-path scripts/configs/gpt/default/lite.scalecua.yaml \
  --env-kwargs '{"max_steps": 1, "loop_detect": 2}' \
  --save-data true --save-video false --save-gif false \
  --log-root .exps/validate/lite.scalecua/gpt-5.5-direct-rl-smoke-20260714-rerun
```

Direct log root:
`.exps/validate/lite.scalecua/gpt-5.5-direct-rl-smoke-20260714-rerun`

Direct result: `Valid: 1/1`, `episode_return=0.0`, `terminated=false`,
`truncated=true`.

Direct visual inspection of this legacy 2026-07-14 log layout:

- `00_screenshot.png`: GNOME desktop and Chrome new tab are visible; address bar
  focused; no blank/black screen, crash dialog, auth wall, or missing desktop.
- `03_actions.json`: GPT emitted only the `screenshot` tool call; no desktop
  action was executed because this smoke caps `max_steps` at 1.
- `04_screenshot_annotated.png`: only the screenshot marker is shown; no
  incorrect click/scroll/type target.
- `05_results.json`: reward `0.0`, truncated `true`, no executed actions. This
  is expected for a reset/screenshot transport smoke and is not a task-success
  judgement.

Direct cleanup: no matching `lite.scalecua`/`lite.osworld` container remained
after rollout.

Server command:

```bash
env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
  uv run --no-sync python scripts/serve_env.py \
  --host 127.0.0.1 --port 30177 \
  --env-ids lite.scalecua \
  --token lite-scalecua-smoke-20260714 \
  --max-live-envs 2 --idle-ttl-sec 120 \
  --reset-concurrency 1 --warm-pool-spawn-concurrency 1

CUA_LITE_ENV_SERVER_URL=http://127.0.0.1:30177 \
CUA_LITE_ENV_SERVER_TOKEN=lite-scalecua-smoke-20260714 \
  uv run --no-sync python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id lite.scalecua \
  --splits rl \
  --head 1 \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --concurrency 1 \
  --max-attempts 1 \
  --config-path scripts/configs/gpt/default/lite.scalecua.yaml \
  --env-kwargs '{"max_steps": 1, "loop_detect": 2}' \
  --save-data true --save-video false --save-gif false \
  --log-root .exps/validate/lite.scalecua/gpt-5.5-server-rl-smoke-20260714
```

Server log root:
`.exps/validate/lite.scalecua/gpt-5.5-server-rl-smoke-20260714`

Server result: `/envs/lite.scalecua` returned available with splits
`eval/eval_full/rl/train`; server logs show `POST /instances`, `reset`, `step`,
and `DELETE /instances`; rollout summary reports `Valid: 1/1`,
`episode_return=0.0`, `terminated=false`, `truncated=true`.

Server visual inspection of this legacy 2026-07-14 log layout:

- `00_screenshot.png`: same healthy GNOME + Chrome new-tab setup as direct mode.
- `03_actions.json`: GPT emitted only `screenshot`; no desktop action was
  executed under the one-step cap.
- `04_screenshot_annotated.png`: no bad click/scroll/type annotation.
- `05_results.json`: reward `0.0`, truncated `true`, no executed actions;
  consistent with the smoke cap.

Server cleanup: port `30177` was released. The server-side container initially
remained as exited after shutdown and was explicitly removed with
`docker rm -f -v`; final Docker check had no `lite-env-30177` or
`lite.scalecua` matches.

Failure classification: none for reset/screenshot transport. This entry does
not satisfy the full trajectory success-rate gate.

## 2026-07-14 gpt-5.5 rl Reward-1 Probe

Task:
`scalecua_osworld_rl_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_traj_verify_0`

Instruction: `Set Chrome's default font size to Very Large.`

Command:

```bash
env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
  uv run --no-sync python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id lite.scalecua \
  --task-id scalecua_osworld_rl_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_traj_verify_0 \
  --concurrency 1 \
  --max-attempts 1 \
  --config-path scripts/configs/gpt/default/lite.scalecua.yaml \
  --save-data true --save-video false --save-gif false \
  --log-root .exps/validate/lite.scalecua/gpt-5.5-direct-rl-successprobe-20260714
```

Log root:
`.exps/validate/lite.scalecua/gpt-5.5-direct-rl-successprobe-20260714`

Result: `Valid: 1/1`, `episode_return=1.0`, `mean_episode_return=1.0`,
`terminated=true`, `truncated=false`.

Visual/action inspection:

- The trajectory entered `chrome://settings/appearance`, opened the `Font size`
  dropdown, and selected `Very large`.
- Legacy `turn_05/04_screenshot_annotated.png` shows the click target on the
  `Very large` dropdown option.
- Legacy `turn_06/00_screenshot.png` shows `Font size` set to `Very large`.
- `turn_06/05_results.json` reports reward `1.0` and termination.

Cleanup: the rollout emitted an async `docker rm` timeout warning, but a
post-run Docker check found no matching `lite.scalecua`/`lite.osworld`
container.

Failure classification: none.

## 2026-07-14 Superseded 300-task Diagnostic Prompt-data

Command:

```bash
env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
  uv run --no-sync python devs/envs/lite.scalecua/validate/rollout/make_batch_prompt_data.py \
  --per-domain 15 \
  --seed 20260714 \
  --output .exps/validate/lite.scalecua/batch/gpt-5.5-300.prompt.parquet \
  --manifest .exps/validate/lite.scalecua/batch/gpt-5.5-300.manifest.json
```

Artifacts:

- `.exps/validate/lite.scalecua/batch/gpt-5.5-300.prompt.parquet`
- `.exps/validate/lite.scalecua/batch/gpt-5.5-300.manifest.json`

Validation:

- total rows: 300
- split counts: `train=150`, `rl=150`
- every OSWorld domain has exactly 15 `train` rows and 15 `rl` rows
- no sampled row has `metadata.others.exclude_reason`

This 300-task batch was later superseded by the 1000-task gate. The partial
300 rollout is diagnostic only: it exposed the generated evaluator
`env.controller.*` compatibility issue and must not be used for acceptance
success-rate reporting.

## 2026-07-14 1000-task Batch Prompt-data

Command:

```bash
env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
  uv run --no-sync python devs/envs/lite.scalecua/validate/rollout/make_batch_prompt_data.py \
  --per-domain 50 \
  --seed 20260714 \
  --output .exps/validate/lite.scalecua/batch/gpt-5.5-1000.prompt.parquet \
  --manifest .exps/validate/lite.scalecua/batch/gpt-5.5-1000.manifest.json
```

Artifacts:

- `.exps/validate/lite.scalecua/batch/gpt-5.5-1000.prompt.parquet`
- `.exps/validate/lite.scalecua/batch/gpt-5.5-1000.manifest.json`

Validation:

- total rows: 1000
- split counts: `train=500`, `rl=500`
- every OSWorld domain has exactly 50 `train` rows and 50 `rl` rows
- no sampled row has `metadata.others.exclude_reason`

This completes the 1000-task prompt-data generation gate. Full rollout must run
through a fresh dedicated env-server with concurrency 8. While it runs, every
completed trajectory must be visually audited against screenshots/actions; any
reward/visual mismatch must be checked against the corresponding SCALE-CUA task
JSON and generated getter/metric code before being classified as model failure,
adapter bug, evaluator bug, upstream blocked task, or transient.

## 2026-07-14 Pre-gate 1000-task Abort and Canonical Getter Fix

Initial 1000-task env-server:

```bash
env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
  uv run --no-sync python scripts/serve_env.py \
  --host 127.0.0.1 --port 30179 \
  --env-ids lite.scalecua \
  --token lite-scalecua-1000-20260714 \
  --max-live-envs 12 --idle-ttl-sec 600 \
  --reset-concurrency 4 --warm-pool-spawn-concurrency 4
```

The attempt was stopped and moved to:

```text
.exps/validate/lite.scalecua/batch/gpt-5.5-1000-pre-canonical-getter-fix-20260714
.exps/validate/lite.scalecua/batch/gpt-5.5-1000.pre-canonical-getter-fix-20260714.audit_queue.jsonl
```

Root cause from screenshots plus SCALE-CUA code:

- task
  `scalecua_osworld_train_chrome_12086550_11c0_466b_b367_1d9e75b3910e_task_verify_14`
  visually reached `chrome://password-manager/passwords`;
- reward was `0`;
- generated `active_url_from_accessTree` getter read only the accessibility
  tree and missed the correct Chrome internal URL;
- local `lite.osworld` already has a CDP-first active URL getter, so
  `lite.scalecua` must prefer base OSWorld handlers for canonical getter types.

Code fix:

- `lite/gym/envs/lite/scalecua/src/osworld/verify.py` now has explicit
  canonical result/expected allowlists;
- canonical types call `lite.osworld.src.eval.runner` directly;
- generated overlay getters remain responsible for ScaleCUA custom types.

Regression:

```bash
uv run --no-sync python -m py_compile \
  lite/gym/envs/lite/scalecua/src/osworld/judges.py \
  lite/gym/envs/lite/scalecua/src/osworld/verify.py

uv run --no-sync python -m pytest tests/gym/envs/lite/test_scalecua.py -q

env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
  uv run --no-sync python -m pytest \
    tests/gym/envs/lite/test_scalecua.py \
    tests/gym/test_family_declarations.py::test_declared_family_matches_audit \
    tests/gym/test_family_golden.py \
    tests/gym/test_warm_pool_contract.py::test_baked_kwargs_match_warm_md_per_env -q
```

Results:

- `11 passed` for `tests/gym/envs/lite/test_scalecua.py`;
- `100 passed, 10 skipped` for the broader family/warm-pool regression.

Targeted smoke after the fix:

```bash
CUA_LITE_ENV_SERVER_URL=http://127.0.0.1:30180 \
CUA_LITE_ENV_SERVER_TOKEN=lite-scalecua-smoke-active-url-20260714 \
  uv run --no-sync python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id lite.scalecua \
  --prompt-data .exps/validate/lite.scalecua/gpt-5.5-active-url-fix-20260714.prompt.parquet \
  --concurrency 1 --max-attempts 1 \
  --config-path scripts/configs/gpt/default/lite.scalecua.yaml \
  --save-data true --save-video false --save-gif false \
  --log-root .exps/validate/lite.scalecua/gpt-5.5-active-url-fix-20260714
```

Result: `Valid: 1/1`, `episode_return=1.0`. Final screenshot shows Google
Password Manager at `chrome://password-manager/passwords`.

Cleanup evidence:

- smoke env-server port `30180` exited cleanly;
- `docker ps -a --filter name=lite-env-30180` returned no containers.

## 2026-07-14 Pre-gate Abort and Python Command Cleanup Fix

The restarted 1000-task attempt exposed a second systematic evaluator issue and
was stopped before it became an acceptance run. The partial artifacts were moved
to:

```text
.exps/validate/lite.scalecua/batch/gpt-5.5-1000-pre-python-command-cleanup-fix-20260714
.exps/validate/lite.scalecua/batch/gpt-5.5-1000.pre-python-command-cleanup-fix-20260714.audit_queue.jsonl
.exps/validate/lite.scalecua/batch/audit_shards-pre-python-command-cleanup-fix-20260714
```

Root cause from screenshot plus SCALE-CUA getter code:

- task
  `scalecua_osworld_train_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_task_verify_8`
  visually selected the requested Chrome setting, "Delete data sites that have
  been saved to your device when you close all windows";
- reward was `0`;
- the generated getter used `execute_python_command()` to print the Chrome
  Preferences path, but the local adapter always imported `pyautogui`;
- importing `pyautogui` emitted `Xlib.xauth: warning, no xauthority details
  available`, so the getter treated warning text plus the path as a filename and
  read no Preferences file.

Code fix:

- `execute_python_command()` only injects the `pyautogui` prefix when the command
  actually references `pyautogui`;
- command stdout/stderr now strip known Xlib warning lines before being returned
  to generated getters.

Targeted smoke after the fix:

```bash
CUA_LITE_ENV_SERVER_URL=http://127.0.0.1:30181 \
CUA_LITE_ENV_SERVER_TOKEN=lite-scalecua-smoke-python-command-20260714 \
  uv run --no-sync python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id lite.scalecua \
  --prompt-data .exps/validate/lite.scalecua/gpt-5.5-python-command-fix-20260714.prompt.parquet \
  --concurrency 1 --max-attempts 1 \
  --config-path scripts/configs/gpt/default/lite.scalecua.yaml \
  --save-data true --save-video false --save-gif false \
  --log-root .exps/validate/lite.scalecua/gpt-5.5-python-command-fix-20260714
```

Result: `Valid: 1/1`, `episode_return=1.0`. Final screenshot confirms the DNT
setting state. Env-server port `30181` exited cleanly and
`docker ps -a --filter name=lite-env-30181` returned no containers.

## 2026-07-14 Official Getter and Score Semantics Fix

Code-diff audit against SCALE-CUA official `DesktopEnv.evaluate()` found two more
compatibility gaps before restarting the 1000-task gate:

- official-only bare Chrome getters referenced by generated/RL tasks were not
  reachable after overlay lookup. Examples: `enabled_experiments`,
  `chrome_language`, and `enable_enhanced_safety_browsing`.
- local aggregation still binarized scores, while official ScaleCUA returns raw
  single-metric scores, multi-`and` average/zero, and multi-`or` max/one.
  Official eval also returns `0` for non-`infeasible` tasks whose final action is
  FAIL.

Code fix:

- `judges.resolve_getter()` now resolves ScaleCUA overlay getters first and then
  upstream `desktop_env.evaluators.getters`.
- `verify.evaluate_scalecua_task()` now matches official raw/partial score
  aggregation.
- `verify.evaluate_final_fn()` now short-circuits final FAIL to `0` for
  non-`infeasible` tasks.
- `audit_queue.py` now reports and queues `reward_partial` rows separately from
  reward `0` and reward `1`.

Regression:

```bash
uv run --no-sync pytest tests/gym/envs/lite/test_scalecua.py -q

uv run --no-sync pytest \
  tests/gym/envs/lite/test_scalecua.py \
  tests/gym/test_family_declarations.py::test_declared_family_matches_audit \
  tests/gym/test_family_golden.py \
  tests/gym/test_warm_pool_contract.py::test_baked_kwargs_match_warm_md_per_env -q
```

Results:

- `16 passed` for `tests/gym/envs/lite/test_scalecua.py`;
- `105 passed, 10 skipped` for the broader family/warm-pool regression.

## 2026-07-14 Pre-gate Abort and VLC Auth Tuple Fix

The next 1000-task attempt was stopped before it reached the VLC-domain shard.
The run completed 25 trajectories (`reward=1`: 15, `reward=0`: 10,
`reward_partial`: 0) and was moved to:

```text
.exps/validate/lite.scalecua/batch/gpt-5.5-1000-pre-vlc-auth-fix-20260714
.exps/validate/lite.scalecua/batch/gpt-5.5-1000.pre-vlc-auth-fix-20260714.audit_queue.jsonl
.exps/validate/lite.scalecua/batch/gpt-5.5-1000.pre-vlc-auth-fix-20260714.visual_audit.jsonl
```

Reason for stopping:

- the 1000-task sample contains 78 VLC result getters and 38 non-canonical VLC
  generated getters;
- generated VLC status getters call `requests.get(..., auth=('', password))`
  against the VLC HTTP endpoint;
- the local request router serialized tuple kwargs through JSON as lists, so
  in-container `requests` would receive `auth=['', password]` instead of an auth
  tuple and fail with a type error;
- this had not yet affected completed trajectories because the first sampled
  non-canonical VLC getter starts at prompt index 402.

Additional compatibility gap found in the same audit: `func == "infeasible"`
accepted dict action arguments, but not raw JSON-string tool-call arguments.

Code fix:

- `judges._encode_request_value()` now preserves request tuples with an explicit
  `{"__tuple__": ...}` marker and still preserves byte payloads;
- the in-container request shim decodes tuple/bytes markers recursively before
  calling `requests.request`;
- `_reported_infeasible()` now reuses the final-failure parser and accepts
  JSON-string `terminate(status="failure")` and `[infeasible]` response args.

Regression:

```bash
uv run --no-sync pytest tests/gym/envs/lite/test_scalecua.py -q

uv run --no-sync pytest \
  tests/gym/envs/lite/test_scalecua.py \
  tests/gym/test_family_declarations.py::test_declared_family_matches_audit \
  tests/gym/test_family_golden.py \
  tests/gym/test_warm_pool_contract.py::test_baked_kwargs_match_warm_md_per_env -q
```

Results:

- `18 passed` for `tests/gym/envs/lite/test_scalecua.py`;
- `107 passed, 10 skipped` for the broader family/warm-pool regression.

Cleanup evidence:

- env-server port `30179` was stopped;
- `docker ps -a --filter name=lite-env-30179` initially showed seven exited
  containers, all from this stopped batch;
- `docker rm -f -v` removed those seven containers;
- subsequent `docker ps -a` showed no `lite-env-30179` or `lite.scalecua`
  containers.

## 2026-07-14 Pre-gate Compatibility Fix: Extension Path Getter And vm_file Bytes

Visual audit of the pre-VLC-auth partial found two likely evaluator false
failures that needed code-side verification before restarting the 1000-task
batch:

- `scalecua_osworld_train_chrome_6766f2b8_8a72_417f_a9e5_56fcaa735837_task_verify_66`
  showed the Hello Extension loaded in Chrome but reported reward `0`; its
  evaluator uses `result.type=find_unpacked_extension_path`.
- `scalecua_osworld_train_chrome_788b3701_3ec9_4b67_b679_418bfa726c22_task_verify_19`
  showed the expected Calc formulas visually but reported reward `0`; server
  logs showed `TypeError: a bytes-like object is required, not 'str'` inside
  `check_xlsx_formula__a2439b25`.

Root causes:

- `find_unpacked_extension_path` had been included in the broad OSWorld
  canonical allowlist. That routed ScaleCUA train/rl extension tasks to the
  `lite.osworld` base runner instead of the official/upstream DesktopEnv getter.
  The inconsistency came from treating same-named result types as automatically
  equivalent instead of validating each type against screenshots and official
  code.
- Official/generated `vm_file` mostly returns a local cache path, but a small
  number of generated metrics annotate their first parameter as `bytes` and
  write `result` directly. Passing the path string into those metrics causes a
  swallowed exception and false reward `0`.

Code fix:

- removed `find_unpacked_extension_path` from the ScaleCUA base-runner allowlist
  so train/rl tasks resolve it through ScaleCUA/upstream
  `desktop_env.evaluators.getters`;
- added metric input adaptation: when `result.type == "vm_file"`, the result is
  a local path, and the resolved metric first parameter is annotated `bytes`,
  read and pass file bytes; otherwise preserve path semantics.

Regression:

```bash
uv run --no-sync pytest tests/gym/envs/lite/test_scalecua.py -q

uv run --no-sync pytest \
  tests/gym/envs/lite/test_scalecua.py \
  tests/gym/test_family_declarations.py::test_declared_family_matches_audit \
  tests/gym/test_family_golden.py \
  tests/gym/test_warm_pool_contract.py::test_baked_kwargs_match_warm_md_per_env -q
```

Results:

- `20 passed` for `tests/gym/envs/lite/test_scalecua.py`;
- `109 passed, 10 skipped` for the broader family/warm-pool regression.

## 2026-07-14 Pre-gate Abort and Chrome Internal URL Fallback Fix

The env-server port `30279` attempt was stopped after 41 completed rows:

```text
.exps/validate/lite.scalecua/batch/gpt-5.5-1000
```

Observed raw rewards before abort: 17 exact successes, 21 failures, and 3
partial rewards. This root is diagnostic only and must not be used for
success-rate reporting.

Reason for stopping:

- `scalecua_osworld_train_chrome_12086550_11c0_466b_b367_1d9e75b3910e_task_verify_41`
  visually reached Chrome's bookmark manager at `chrome://bookmarks`;
- the evaluator expected `chrome://bookmarks/` through
  `active_url_from_accessTree` and `is_expected_active_tab_approximate`;
- AT-SPI can expose the in-page bookmark search focus instead of the address
  bar URL for Chrome internal pages.

Code fix:

- `active_url_from_accessTree` remains AT-first for normal pages;
- a narrow CDP fallback now canonicalizes matching `chrome://...` internal
  pages, ignores omnibox popup/newtab pseudo-pages, and refuses ambiguous
  multiple internal-page candidates.

Regression:

```bash
uv run --no-sync pytest tests/gym/envs/lite/test_scalecua.py -q

uv run --no-sync pytest \
  tests/gym/envs/lite/test_scalecua.py \
  tests/gym/test_family_declarations.py::test_declared_family_matches_audit \
  tests/gym/test_family_golden.py \
  tests/gym/test_warm_pool_contract.py::test_baked_kwargs_match_warm_md_per_env -q
```

Results:

- `36 passed` for `tests/gym/envs/lite/test_scalecua.py`;
- `125 passed, 10 skipped` for the broader family/warm-pool regression.

Cleanup evidence:

- rollout process was interrupted;
- env-server port `30279` was stopped;
- `docker rm -f -v` removed the remaining `lite-env-30279-*` container;
- subsequent `docker ps` showed no `lite-env-30279` / `lite.scalecua`
  containers.

Restarted acceptance run:

```bash
env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
  uv run --no-sync python scripts/serve_env.py \
  --host 127.0.0.1 \
  --port 30280 \
  --env-ids lite.scalecua \
  --token lite-scalecua-1000-fix-20260714 \
  --max-live-envs 12 \
  --idle-ttl-sec 600 \
  --reset-concurrency 4 \
  --warm-pool-spawn-concurrency 4
```

```bash
CUA_LITE_ENV_SERVER_URL=http://127.0.0.1:30280 \
CUA_LITE_ENV_SERVER_TOKEN=lite-scalecua-1000-fix-20260714 \
  uv run --no-sync python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id lite.scalecua \
  --prompt-data .exps/validate/lite.scalecua/batch/gpt-5.5-1000.prompt.parquet \
  --concurrency 8 \
  --max-attempts 3 \
  --config-path scripts/configs/gpt/default/lite.scalecua.yaml \
  --save-data true \
  --save-video false \
  --save-gif false \
  --log-root .exps/validate/lite.scalecua/batch/gpt-5.5-1000-activeurlfix
```

Early validation on the fresh run:

- `scalecua_osworld_train_chrome_12086550_11c0_466b_b367_1d9e75b3910e_task_verify_41`
  now returns reward `1.0`;
- `scalecua_osworld_train_chrome_215dfd39_f493_4bc3_a027_8a97d72c61bf_task_verify_37`
  still returns reward `0.0`, which is expected for a `vlc_config` task if
  `Ctrl+H` only changes current UI and does not persist `qt-minimal-view=1` to
  `vlcrc` across postconfig relaunch.

## 2026-07-14 Pre-gate Abort and Narrow Chrome Internal Fallback Fix

The `30280` restart was stopped after 11 completed rows. It confirmed the
bookmark-manager task reward changed to `1.0`, but a code-side review found
that the no-accessibility-tree Chrome-internal fallback was still too broad for
acceptance:

- normal web URL evaluators must not fall back to a single `chrome://...` CDP
  candidate when AT XML is missing;
- otherwise an ordinary web task could become a false success if Chrome happened
  to have one internal page open;
- Chrome internal pages still need a fallback when the AT-derived focused text
  is not the URL, for example `chrome://bookmarks/`.

Additional code fix:

- `active_url_from_accessTree` now allows no-AT CDP fallback only for internal
  Chrome targets (`goto_prefix == ""` or a `chrome://...` prefix);
- raw AT URLs such as `chrome://bookmarks` may still be canonicalized against a
  matching CDP candidate;
- omnibox popup/newtab pseudo-pages are still ignored, and ambiguous internal
  candidates still return no fallback.

Regression:

```bash
uv run --no-sync pytest tests/gym/envs/lite/test_scalecua.py -q

uv run --no-sync pytest \
  tests/gym/envs/lite/test_scalecua.py \
  tests/gym/test_family_declarations.py::test_declared_family_matches_audit \
  tests/gym/test_family_golden.py \
  tests/gym/test_warm_pool_contract.py::test_baked_kwargs_match_warm_md_per_env -q
```

Results:

- `37 passed` for `tests/gym/envs/lite/test_scalecua.py`;
- `126 passed, 10 skipped` for the broader family/warm-pool regression.

Cleanup evidence:

- rollout process for `gpt-5.5-1000-activeurlfix` was interrupted;
- env-server port `30280` was stopped;
- `docker rm -f -v` cleanup left no `lite-env-30280` / `lite.scalecua`
  containers in `docker ps`.

Restarted acceptance run:

```bash
env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
  uv run --no-sync python scripts/serve_env.py \
  --host 127.0.0.1 \
  --port 30281 \
  --env-ids lite.scalecua \
  --token lite-scalecua-1000-final-20260714 \
  --max-live-envs 12 \
  --idle-ttl-sec 600 \
  --reset-concurrency 4 \
  --warm-pool-spawn-concurrency 4
```

```bash
CUA_LITE_ENV_SERVER_URL=http://127.0.0.1:30281 \
CUA_LITE_ENV_SERVER_TOKEN=lite-scalecua-1000-final-20260714 \
  uv run --no-sync python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id lite.scalecua \
  --prompt-data .exps/validate/lite.scalecua/batch/gpt-5.5-1000.prompt.parquet \
  --concurrency 8 \
  --max-attempts 3 \
  --config-path scripts/configs/gpt/default/lite.scalecua.yaml \
  --save-data true \
  --save-video false \
  --save-gif false \
  --log-root .exps/validate/lite.scalecua/batch/gpt-5.5-1000-internalfallbackfix
```

Early validation on the diagnostic restart:

- `scalecua_osworld_train_chrome_12086550_11c0_466b_b367_1d9e75b3910e_task_verify_41`
  returns reward `1.0`;
- `scalecua_osworld_train_chrome_215dfd39_f493_4bc3_a027_8a97d72c61bf_task_verify_37`
  returns reward `0.0`, consistent with the hidden persisted `vlc_config`
  evaluator semantics described above.

## 2026-07-14 Pre-gate Abort and Request Shim/VLC Auth Fix

The `30281` restart was stopped after 26 completed rows:

```text
.exps/validate/lite.scalecua/batch/gpt-5.5-1000-internalfallbackfix
```

Observed raw rewards before abort: 14 exact successes, 11 failures, and 1
partial reward. This root is diagnostic only and must not be used for
acceptance success-rate reporting.

Visual audit evidence:

```text
.exps/validate/lite.scalecua/batch/gpt-5.5-1000-internalfallbackfix.visual_audit.shard_a.jsonl  # 26 rows
.exps/validate/lite.scalecua/batch/gpt-5.5-1000-internalfallbackfix.visual_audit.shard_b.jsonl  # 20 rows
```

Reason for stopping:

- code audit found `EvalEnvShim.request_in_container()` generated an invalid
  in-container Python script: the `out = {...}` block and `except` block were
  incorrectly indented, so generated getters that route `requests` through the
  container could fail with a 599-style shim response;
- `vlc_playing_info` still used `curl --user :password`, while
  `lite.osworld` dispatch and runner use the `vlc` HTTP password;
- visual audit also surfaced Chrome startup/extension reward disagreements, but
  official generated getters read Chrome `Preferences` and extension settings,
  so screenshots alone cannot prove a migration bug without a targeted
  persisted-state probe.

Code fix:

- factored request script generation into `_build_request_script()` and added a
  compile-level regression test for the exact script shape;
- aligned `vlc_playing_info` with the base runner fallback sequence: no auth,
  `:vlc`, then `:a`;
- explicit `postconfig` now runs before terminal failure/infeasible checks for
  every split, using a per-evaluation deep copy and `_postconfig_done` marker to
  avoid double-running base OSWorld postconfig.

Regression:

```bash
uv run --no-sync pytest tests/gym/envs/lite/test_scalecua.py -q

uv run --no-sync pytest \
  tests/gym/envs/lite/test_scalecua.py \
  tests/gym/test_family_declarations.py::test_declared_family_matches_audit \
  tests/gym/test_family_golden.py \
  tests/gym/test_warm_pool_contract.py::test_baked_kwargs_match_warm_md_per_env -q
```

Results:

- `39 passed` for `tests/gym/envs/lite/test_scalecua.py`;
- `128 passed, 10 skipped` for the broader family/warm-pool regression.

Cleanup evidence:

- rollout process for `gpt-5.5-1000-internalfallbackfix` was interrupted;
- env-server port `30281` was stopped;
- `docker rm -f -v` cleanup/check left no `lite-env-30281` /
  `lite.scalecua` containers in `docker ps`.

Replacement acceptance run:

```bash
env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
  uv run --no-sync python scripts/serve_env.py \
  --host 127.0.0.1 \
  --port 30282 \
  --env-ids lite.scalecua \
  --token lite-scalecua-1000-requestshimfix-20260714 \
  --max-live-envs 12 \
  --idle-ttl-sec 600 \
  --reset-concurrency 4 \
  --warm-pool-spawn-concurrency 4
```

```bash
CUA_LITE_ENV_SERVER_URL=http://127.0.0.1:30282 \
CUA_LITE_ENV_SERVER_TOKEN=lite-scalecua-1000-requestshimfix-20260714 \
  uv run --no-sync python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id lite.scalecua \
  --prompt-data .exps/validate/lite.scalecua/batch/gpt-5.5-1000.prompt.parquet \
  --concurrency 8 \
  --max-attempts 3 \
  --config-path scripts/configs/gpt/default/lite.scalecua.yaml \
  --save-data true \
  --save-video false \
  --save-gif false \
  --log-root .exps/validate/lite.scalecua/batch/gpt-5.5-1000-requestshimfix
```

## 2026-07-14 Pre-gate Abort and Chrome Profile Alias Fix

The `30282` replacement run was stopped after 50 completed rows:

```text
.exps/validate/lite.scalecua/batch/gpt-5.5-1000-requestshimfix
```

Observed raw rewards before abort: 18 exact successes, 29 failures, and 3
partial rewards. This root is diagnostic only and must not be used for
acceptance success-rate reporting.

Reason for stopping:

- code audit found that `lite.osworld` setup and launch route Chrome to
  `/home/user/chrome-data`, while ScaleCUA generated/upstream fallback getters
  still read official DesktopEnv paths such as
  `/home/user/.config/google-chrome/Default/Preferences`;
- this can make extension/startup tasks visually look correct while reward is
  `0`, because the getter reads stale `.config/google-chrome` state;
- the 1000-task prompt sample contains multiple affected generated and upstream
  fallback Chrome getter families, so continuing the batch would produce
  untrustworthy reward labels.

Code fix:

- `EvalEnvShim` aliases official Chrome/Chromium profile paths to
  `/home/user/chrome-data` for file APIs and command-entry APIs;
- `lite.scalecua.verify` aliases evaluator config strings before calling base
  `lite.osworld` result getters, covering `vm_command_line` /
  `vm_command_error` without changing `lite.osworld`;
- regression tests now cover direct `get_file`, generated
  `execute_python_command`, generated `run_bash_script`, upstream split-join
  path shapes, and base-runner config commands.

Regression:

```bash
uv run --no-sync pytest tests/gym/envs/lite/test_scalecua.py -q

uv run --no-sync pytest \
  tests/gym/envs/lite/test_scalecua.py \
  tests/gym/test_family_declarations.py::test_declared_family_matches_audit \
  tests/gym/test_family_golden.py \
  tests/gym/test_warm_pool_contract.py::test_baked_kwargs_match_warm_md_per_env -q
```

Results:

- `42 passed` for `tests/gym/envs/lite/test_scalecua.py`;
- `131 passed, 10 skipped` for the broader family/warm-pool regression.

Cleanup evidence:

- rollout process for `gpt-5.5-1000-requestshimfix` was interrupted;
- env-server port `30282` was stopped;
- `docker ps -a | rg 'lite-env-30282'` returned no containers.

Targeted post-fix rerun:

```bash
env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
  uv run --no-sync python scripts/serve_env.py \
  --host 127.0.0.1 \
  --port 30283 \
  --env-ids lite.scalecua \
  --token lite-scalecua-chromealias-20260714 \
  --max-live-envs 6 \
  --idle-ttl-sec 600 \
  --reset-concurrency 3 \
  --warm-pool-spawn-concurrency 3
```

```bash
CUA_LITE_ENV_SERVER_URL=http://127.0.0.1:30283 \
CUA_LITE_ENV_SERVER_TOKEN=lite-scalecua-chromealias-20260714 \
  uv run --no-sync python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id lite.scalecua \
  --prompt-data .exps/validate/lite.scalecua/targeted/chrome-alias.prompt.parquet \
  --concurrency 3 \
  --max-attempts 1 \
  --config-path scripts/configs/gpt/default/lite.scalecua.yaml \
  --save-data true \
  --save-video false \
  --save-gif false \
  --log-root .exps/validate/lite.scalecua/targeted/gpt-5.5-chrome-alias
```

## 2026-07-14 Pre-gate Profile Flush and Recreation Grid Fixes

Two targeted diagnostics were completed before restarting the 1000-task gate.

Profile flush targeted evidence:

- root: `.exps/validate/lite.scalecua/targeted/gpt-5.5-profileflush`;
- tasks: extension version, startup-page removal, unpacked-extension path;
- result: `3/3` reward `1.0`;
- regression: `tests/gym/envs/lite/test_scalecua.py` passed after adding
  profile-flush coverage.

Recreation.gov targeted evidence:

- pre-fix root:
  `.exps/validate/lite.scalecua/targeted/gpt-5.5-recreationcdp-ready`;
- pre-fix result: `2/2` completed with reward `0.0`; screenshots showed the
  Devils Garden availability grid, so the failure was evaluator/parser drift;
- post-fix root:
  `.exps/validate/lite.scalecua/targeted/gpt-5.5-recreationgridfix`;
- post-fix result: `2/2` reward `1.0`;
- regression:
  `uv run --no-sync pytest tests/gym/envs/lite/test_scalecua.py -q` returned
  `47 passed`;
- broader regression:
  `136 passed, 10 skipped` for the lite.scalecua + family/golden/warm-pool
  test slice.

Cleanup evidence:

- env-server port `30290` was stopped after targeted validation;
- `docker ps -a --format '{{.Names}} {{.Status}}' | rg 'lite-env-30290'`
  returned no containers;
- `ss -ltnp | rg ':30290'` returned no listener.

## 2026-07-14 Stopped 1000-task Diagnostic Run

Server command:

```bash
env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
  uv run --no-sync python scripts/serve_env.py \
  --host 127.0.0.1 \
  --port 30291 \
  --env-ids lite.scalecua \
  --token lite-scalecua-1000-profileflush-recreationgridfix-20260714 \
  --max-live-envs 12 \
  --idle-ttl-sec 900 \
  --reset-concurrency 4 \
  --warm-pool-spawn-concurrency 4
```

Rollout command:

```bash
CUA_LITE_ENV_SERVER_URL=http://127.0.0.1:30291 \
CUA_LITE_ENV_SERVER_TOKEN=lite-scalecua-1000-profileflush-recreationgridfix-20260714 \
  uv run --no-sync python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id lite.scalecua \
  --prompt-data .exps/validate/lite.scalecua/batch/gpt-5.5-1000.prompt.parquet \
  --concurrency 8 \
  --max-attempts 1 \
  --config-path scripts/configs/gpt/default/lite.scalecua.yaml \
  --save-data true \
  --save-video false \
  --save-gif false \
  --log-root .exps/validate/lite.scalecua/batch/gpt-5.5-1000-profileflush-recreationgridfix
```

Run notes:

- env-server reached `/host_status` before rollout began;
- rollout concurrency is `8`;
- diagnostic log root:
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-profileflush-recreationgridfix`;
- this root was stopped after a Recreation.gov CDP/AT mismatch was found and is
  diagnostic only;
- final diagnostic queue:
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-profileflush-recreationgridfix.audit_queue.final-diagnostic.jsonl`;
- final diagnostic counts: scanned `54`, completed `46`, in-progress at
  interrupt `8`, reward `1.0` = `22`, reward `0.0` = `21`, partial reward =
  `3`;
- cleanup: env-server port `30291` was stopped; no `lite-env-30291-*`
  containers and no `:30291` listener remained.

## 2026-07-14 Recreation.gov AT Fallback Targeted Rerun

The `30291` diagnostic root showed that `task_verify_29` could be visually
correct while CDP content still omitted the visible availability grid. The
`lite.scalecua` Recreation.gov getter now falls back to AT-SPI text after an
incomplete CDP parse.

Regression:

```bash
uv run --no-sync pytest tests/gym/envs/lite/test_scalecua.py -q

uv run --no-sync pytest \
  tests/gym/envs/lite/test_scalecua.py \
  tests/gym/test_family_declarations.py::test_declared_family_matches_audit \
  tests/gym/test_family_golden.py \
  tests/gym/test_warm_pool_contract.py::test_baked_kwargs_match_warm_md_per_env -q
```

Results:

- `48 passed` for `tests/gym/envs/lite/test_scalecua.py`;
- `137 passed, 10 skipped` for the broader family/golden/warm-pool slice.

Targeted server command:

```bash
env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
  uv run --no-sync python scripts/serve_env.py \
  --host 127.0.0.1 \
  --port 30292 \
  --env-ids lite.scalecua \
  --token lite-scalecua-recreation-atfallback-20260714 \
  --max-live-envs 6 \
  --idle-ttl-sec 600 \
  --reset-concurrency 3 \
  --warm-pool-spawn-concurrency 3
```

Targeted rollout command:

```bash
CUA_LITE_ENV_SERVER_URL=http://127.0.0.1:30292 \
CUA_LITE_ENV_SERVER_TOKEN=lite-scalecua-recreation-atfallback-20260714 \
  uv run --no-sync python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id lite.scalecua \
  --prompt-data .exps/validate/lite.scalecua/targeted/recreation-cdp.prompt.parquet \
  --concurrency 2 \
  --max-attempts 1 \
  --config-path scripts/configs/gpt/default/lite.scalecua.yaml \
  --save-data true \
  --save-video false \
  --save-gif false \
  --log-root .exps/validate/lite.scalecua/targeted/gpt-5.5-recreation-atfallback
```

Targeted result:

- root:
  `.exps/validate/lite.scalecua/targeted/gpt-5.5-recreation-atfallback`;
- tasks:
  `scalecua_osworld_train_chrome_b4f95342_463e_4179_8c3f_193cd7241fb2_task_verify_26`
  and
  `scalecua_osworld_train_chrome_b4f95342_463e_4179_8c3f_193cd7241fb2_task_verify_29`;
- result: `2/2` reward `1.0`;
- audit queue:
  `.exps/validate/lite.scalecua/targeted/gpt-5.5-recreation-atfallback.audit_queue.final.jsonl`;
- visual check: both final screenshots show the Devils Garden availability grid
  and Juniper Basin available on WED 15;
- cleanup: env-server port `30292` was stopped; no `lite-env-30292-*`
  containers and no `:30292` listener remained.

## 2026-07-14 Next 1000-task Acceptance Run

Fresh server/rollout started:

```bash
env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
  uv run --no-sync python scripts/serve_env.py \
  --host 127.0.0.1 \
  --port 30293 \
  --env-ids lite.scalecua \
  --token lite-scalecua-1000-profileflush-recreation-atfallback-20260714 \
  --max-live-envs 12 \
  --idle-ttl-sec 900 \
  --reset-concurrency 4 \
  --warm-pool-spawn-concurrency 4
```

```bash
CUA_LITE_ENV_SERVER_URL=http://127.0.0.1:30293 \
CUA_LITE_ENV_SERVER_TOKEN=lite-scalecua-1000-profileflush-recreation-atfallback-20260714 \
  uv run --no-sync python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id lite.scalecua \
  --prompt-data .exps/validate/lite.scalecua/batch/gpt-5.5-1000.prompt.parquet \
  --concurrency 8 \
  --max-attempts 1 \
  --config-path scripts/configs/gpt/default/lite.scalecua.yaml \
  --save-data true \
  --save-video false \
  --save-gif false \
  --log-root .exps/validate/lite.scalecua/batch/gpt-5.5-1000-profileflush-recreation-atfallback
```

Run notes:

- env-server reached `/host_status` before rollout began;
- rollout concurrency is `8`;
- active log root:
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-profileflush-recreation-atfallback`;
- live audit queue:
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-profileflush-recreation-atfallback.audit_queue.live.jsonl`;
- early manual visual audit:
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-profileflush-recreation-atfallback.visual_audit.early.manual.jsonl`;
- first live scan after startup: scanned `27`, completed `22`, in-progress `5`,
  reward `1.0` = `15`, reward `0.0` = `7`, partial reward = `0`;
- early manual labels: `true_success=3`, `true_failure=2`,
  `ambiguous_needs_evaluator_probe=1`;
- no `false_success` or confirmed `false_failure` in the early manual checks.

## 2026-07-14 Diagnostic Stop: Yahoo Alias + Generated ignore_list_order

The 30293 root was stopped before acceptance:

`.exps/validate/lite.scalecua/batch/gpt-5.5-1000-profileflush-recreation-atfallback`

Reason:

- visual audit showed `Yahoo! Hong Kong (Default)` in Chrome settings for
  `scalecua_osworld_train_chrome_bb5e4c0d_f964_439c_97b6_bdb9747de3f4_task_verify_52`,
  while the generated evaluator expected only `Yahoo` / `Yahoo!`;
- code audit and server logs showed Apple compare
  `scalecua_osworld_train_chrome_f5d96daf_83a8_4c86_9686_bada31fc66ab_task_verify_49`
  produced `result={'modelList': ['iphone-15', 'iphone-14', 'iphone-13']}`
  but failed because `ignore_list_order` was embedded inside `rules.expected`
  instead of top-level metric rules.

Fix:

- `lite.scalecua.verify` normalizes Yahoo-family `default_search_engine` values
  to `Yahoo!`;
- `lite.scalecua.verify` hoists generated
  `check_direct_json_object` `expected.ignore_list_order` into
  `rules.ignore_list_order` and removes it from the expected JSON payload.

Regression evidence:

- `uv run --no-sync pytest tests/gym/envs/lite/test_scalecua.py -q`:
  `50 passed, 3 warnings`;
- broader slice:
  `139 passed, 10 skipped, 3 warnings`.

Final diagnostic queue:

- artifact:
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-profileflush-recreation-atfallback.audit_queue.final-diagnostic.jsonl`;
- scanned `61`, completed `56`, in-progress `5`;
- reward `1.0` = `30`, reward `0.0` = `23`, partial reward = `3`;
- cleanup: port `30293` stopped, no `lite-env-30293-*` containers, no `:30293`
  listener.

## 2026-07-14 Diagnostic Stop: Chrome Profile-name Blur

The 30294 root was stopped before acceptance:

`.exps/validate/lite.scalecua/batch/gpt-5.5-1000-compatfix`

Reason:

- visual audit showed
  `scalecua_osworld_train_chrome_2ae9ba84_3a0d_4d4c_8338_3a1478dc5fe3_task_verify_1`
  with the Chrome profile name input visibly set to `Sarah`;
- reward was still `0.0`;
- `lite.osworld` perturb tasks for the same family add
  `xdotool key Tab; xdotool key Escape; sleep 1` before `pkill`, because Chrome
  profile-name edits can remain in the focused input until blur.

Fix:

- `lite.scalecua.verify` now blurs Chrome profile-backed state with
  `xdotool key Tab` and `xdotool key Escape` before killing Chrome for the
  profile flush;
- the change is limited to `lite.scalecua`; `lite.osworld` was not modified.

Regression evidence:

- `uv run --no-sync pytest tests/gym/envs/lite/test_scalecua.py -q`:
  `51 passed, 2 warnings`;
- broader slice:
  `140 passed, 10 skipped, 2 warnings`.

Final diagnostic queue:

- artifact:
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-compatfix.audit_queue.final-diagnostic.jsonl`;
- scanned `48`, completed `45`, in-progress `3`;
- reward `1.0` = `27`, reward `0.0` = `16`, partial reward = `2`;
- cleanup: port `30294` stopped, `lite-env-30294-*` removed, no `:30294`
  listener.

## 2026-07-14 Stopped 1000-task Profileblur Diagnostic Run

The profileblur root was stopped before acceptance after visual and code audit
found two more compatibility bugs:

- GIMP filter/dialog tasks such as Lens Distortion, Unsharp Mask, and Gaussian
  Blur were visually correct, but official `action-history` evaluation returned
  reward `0` after postconfig closed GIMP before the history file reliably
  flushed.
- A LibreOffice Calc rollout emitted a lone `+` keypress; CUA-Lite normalized it
  to literal `+`, and `keys.to_xdotool` rejected the token.

Final diagnostic queue:

- log root:
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-profileblur-compatfix`;
- artifact:
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-profileblur-compatfix.audit_queue.final-diagnostic.jsonl`;
- scanned `138`, completed `130`, in-progress `8`;
- reward `1.0` = `73`, reward `0.0` = `49`, partial reward = `8`;
- completed sampled-domain coverage: `train/chrome=50`, `train/gimp=50`,
  `train/libreoffice_calc=30`;
- cleanup: port `30295` stopped, `lite-env-30295-*` removed, no `:30295`
  listener.

Fixes added before restart:

- `lite.scalecua` captures the active GIMP window before postconfig for
  `action-history` evaluators and only applies known GIMP filter/dialog token
  fallbacks when window evidence matches.
- Historical correction: this key-plus compat fix made lone `+` usable as a
  literal glyph. Current Lite storage uses `+`; `plus` is only an accepted raw
  alias before normalization.
- Regression slice: `668 passed, 10 skipped`.

## 2026-07-14 Current 1000-task Acceptance Run

Fresh server/rollout started after the GIMP action-history and key-plus compat
fixes:

```bash
env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
  uv run --no-sync python scripts/serve_env.py \
  --host 127.0.0.1 \
  --port 30297 \
  --env-ids lite.scalecua \
  --token lite-scalecua-1000-gimpwindow-keyplus-compatfix-20260714 \
  --max-live-envs 12 \
  --idle-ttl-sec 900 \
  --reset-concurrency 4 \
  --warm-pool-spawn-concurrency 4
```

```bash
CUA_LITE_ENV_SERVER_URL=http://127.0.0.1:30297 \
CUA_LITE_ENV_SERVER_TOKEN=lite-scalecua-1000-gimpwindow-keyplus-compatfix-20260714 \
  uv run --no-sync python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id lite.scalecua \
  --prompt-data .exps/validate/lite.scalecua/batch/gpt-5.5-1000.prompt.parquet \
  --concurrency 8 \
  --max-attempts 1 \
  --config-path scripts/configs/gpt/default/lite.scalecua.yaml \
  --save-data true \
  --save-video false \
  --save-gif false \
  --log-root .exps/validate/lite.scalecua/batch/gpt-5.5-1000-gimpwindow-keyplus-compatfix
```

Run notes:

- env-server startup completed; `/host_status` was healthy;
- boot recovery reclaimed `0` prior resources;
- registry counts remained `train=20289`, `rl=2049`, `eval=361`,
  `eval_full=369`;
- rollout concurrency is `8`;
- cleanup prefix for this run is only `lite-env-30297-*`;
- visual audit target:
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-gimpwindow-keyplus-compatfix.visual_audit.jsonl`.

Live audit checkpoint after 40 completed trajectories:

- raw rewards: reward `1.0` = `21`, reward `0.0` = `16`, partial reward = `3`;
- visual audit rows written: `40`;
- visual labels: `true_success=17`, `false_failure=6`, `true_failure=9`,
  `partial_success=3`, `not_visually_decidable=4`, `blocked_upstream=1`;
- `visual_success=true` rows: `23`;
- false failures so far are concentrated in external live-site URL/filter drift:
  Virginia DMV path changes, DOJ Forms `field_component_target_id` drift,
  FlightAware category/error drift, and United accessible-travel path changes;
- Macy's generated shopping tasks in this 1000-task sample are limited to three
  train rows plus one rl row; current audited Macy rows are either true failures
  against the official URL parser or a live-site access-denied blocked row;
- resource health at this checkpoint: `/host_status` healthy, 30297 server
  process alive, `lite-env-30297-*` container count observed at `8` to `10`,
  and drift reaper repeatedly reported `orphans=0`.

## 2026-07-14 Current Display-Timeout / Instruction-Filter Run

Fresh server:

```bash
env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
  uv run --no-sync python scripts/serve_env.py \
  --host 127.0.0.1 \
  --port 30313 \
  --env-ids lite.scalecua \
  --token lite-scalecua-1000-displaytimeout-instructionfilter-20260714 \
  --max-live-envs 12 \
  --idle-ttl-sec 900 \
  --reset-concurrency 4 \
  --warm-pool-spawn-concurrency 4
```

Run root:
`.exps/validate/lite.scalecua/batch/gpt-5.5-1000-displaytimeout-instructionfilter`.

Checkpoint at 190 completed trajectories:

- raw rewards: reward `1.0` = `97`, reward `0.0` = `80`, partial reward = `13`;
- scanned `195`, completed `190`, in-progress `5`;
- completed sampled-domain coverage: `train/chrome=50`,
  `train/gimp=50`, `train/libreoffice_calc=50`,
  `train/libreoffice_impress=40`;
- rollout is still running on env-server port `30313`; do not sweep
  `lite-env-30313-*` while the server is live.

Visual audit update:

- `main_02` contact sheet:
  `/tmp/scalecua_visual_audit_writer_impress_01.png`;
- audit shard:
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-displaytimeout-instructionfilter.visual_audit.main_02.jsonl`;
- rows: `16`, focused on the newly covered LibreOffice Impress/Writer region;
- labels: `true_failure=6`, `partial_success=5`,
  `not_visually_decidable=5`;
- no confirmed new system-level migration bug in this shard;
- background-color slide tasks that look visually correct are kept as
  `not_visually_decidable`, because official evaluators inspect the saved PPTX
  slide background XML and visual screenshots cannot distinguish true slide
  background from a full-slide colored shape or unsaved state.

Official-code parity triage from current visual false-failure candidates:

- Official `SetupController._execute_setup` supports `until` plus
  `stdout`/`stderr` capture; current imported `lite.scalecua` catalogs contain
  `0` runnable `until` actions across `train`, `rl`, `eval`, and `eval_full`.
  Current dispatcher supports the stdout/stderr cases that remain in cache, so
  this is not a live setup bug for the present catalogs.
- Official setup action types remaining in the imported catalogs are covered by
  `dispatch_strict`; there are `0` unsupported action types across all four
  splits after filtering.
- Official VLC getter uses HTTP basic auth password `password`. The adapter
  previously tried no-auth / `vlc` / `a`; this was a real migration eval gap.
  Fixed by adding `:password` before the legacy fallbacks in both
  `lite.scalecua` and shared `lite.osworld` VLC getter paths.
- Official GIMP config evaluators read
  `/home/user/.config/GIMP/2.10/gimprc` or `sessionrc` after evaluator
  postconfig closes GIMP. Visual cases for default-image,
  interpolation-quality, and fullscreen were rechecked against their final
  screenshots/actions: they did not persist the requested config before
  termination, so these are model failures under official semantics, not
  migration bugs.
- GIMP action-history dialog tasks remain higher-risk because official eval
  reads the action-history file after closing GIMP. The adapter has a
  window-state fallback for visible dialogs, but current rollout rewards were
  produced by the already-running env-server process. A debug-only eval patch
  now records result, expected, and pre-postconfig window-state previews so
  targeted reruns can distinguish fallback gaps from model/window-state flakes.
- Official `active_url_from_accessTree` reads the Chrome address bar text from
  AT and prepends `goto_prefix`. The adapter follows the same AT-first
  behavior and only adds a narrow Chrome-internal CDP fallback. BabyCenter Liam
  failures are not caused by this adapter difference; they are upstream
  generated-task live-site URL-id drift and are filtered as
  `upstream_live_site_drift`.
- The hardened run exposed another live-site filter gap: generated Macy's
  product/listing rows from OSWorld id
  `2888b4e6-5b47-4b57-8bf5-c73827890774` fail under Akamai access-denied or
  Google-shopping detours, matching the inherited lite.osworld Macy's blocking
  class. The importer now assigns `upstream_live_site_drift` to 42 generated
  Macy's URL/filter rows and 3 RL browse/filter rows while leaving the RL
  bookmark/download rows runnable. Cache was reimported after the fix; this
  running 30314 rollout still contains the older prompt's sampled Macy's rows
  and should be treated as diagnostic for those tasks.
- The same official-code parity check found additional URL live-site drift,
  not importer mutation: DOJ Forms component ids (`401 -> 376` observed for
  Antitrust), Virginia DMV restructured paths, FlightAware category redirects,
  and United `special-needs -> accessibility-and-assistance.html`. The importer
  now filters these narrowly as `upstream_live_site_drift`: DOJ 35 train + 2 RL,
  DMV 40 train, FlightAware 13 train, and United 6 train. The current 30314
  rollout was launched before these cache updates and will still surface the
  sampled stale rows; use it for diagnosis, then regenerate prompt-data on a new
  env server for clean success-rate accounting.
- Official-code parity also found underspecified generated rows rather than a
  migration bug: 9 GIMP rows from
  `3c8f201a-009d-4bbe-8b65-a6f8b35bb57f` and 1 Apple HIG document row from
  `aad10cd7-9337-4b62-b704-a857848cedf2` reference a `provided URL` / `from the
  link`, but the official instruction and setup config contain no URL.
  `osworld_eval/lib_run_single.py` passes only `example["instruction"]` to the
  agent, and the RL worker does not backfill `augmented_from` or metadata. The
  importer now filters these as `instruction_setup_mismatch`. The running 30314
  rollout still contains the stale prompt rows and should treat those failures
  as diagnostic, not clean accounting.
- The 30314 rollout also exposed official generated metric reference paths:
  GIMP metrics attempted to open author-local paths such as
  `/home/lvbowen/project/AutoGen/src/envs/osworld_env/cache/.../character.png`
  or VM paths such as `/home/user/Desktop/tilearray.png` directly on the host.
  Official `DesktopEnv.evaluate()` does not materialize nested rule paths, so
  this is a ScaleCUA eval adapter gap, not a setup download gap. Fixed in
  `lite.scalecua/src/osworld/verify.py` by materializing
  `source_path` / `original_path` / `source_cache_path` into the eval cache
  before generated metrics run. Static field scan found 88 affected rows,
  including 25 author-cache rows. Regression fixture
  `oracle_gimp_author_cache_reference_flip_train_0013` passed with precheck
  reward 0.0 and oracle reward 1.0. The running 30314 rollout was launched
  before this code change, so any further `/home/lvbowen` errors from it are
  stale diagnostic signal; clean accounting needs a new env server.
- Prompt-data staleness check against the current catalog found 14 rows in
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-instructionfilter.prompt.parquet`
  that now have non-empty `exclude_reason`: 12 `upstream_live_site_drift` and
  2 `instruction_setup_mismatch`. This includes
  `scalecua_osworld_train_chrome_59155008_fe71_45ec_8a8f_dc35497b6aa8_task_verify_4`,
  whose reset error in this root is therefore stale diagnostic signal, not a
  setup migration regression. Added
  `devs/envs/lite.scalecua/validate/rollout/check_prompt_data.py`; clean 1000
  accounting must regenerate prompt-data after filter changes and pass this
  check before launching a fresh env-server.
- Regenerated clean prompt-data:
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-currentfilter-regenerated.prompt.parquet`
  with manifest
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-currentfilter-regenerated.manifest.json`.
  `check_prompt_data.py` reports 1000 rows, 50 per split/domain cell, and 0
  current `exclude_reason` hits. `make_batch_prompt_data.py` now clears
  `CUA_LITE_ENV_SERVER_URL` / `CUA_LITE_ENV_SERVER_TOKEN` before importing
  `lite.gym`, so prompt generation always uses the local current catalog rather
  than an accidentally configured remote env-server.
- Code/visual audit found one exact official generated evaluator defect in the
  current catalog:
  `scalecua_osworld_train_libreoffice_calc_eb03d19a_b88d_4de4_8a64_ca0ac66f426b_task_verify_41`.
  Its generated Excel metric checks row index `3` inside `B2:F5`, which includes
  the string label `Marks` alongside numeric marks. The importer now filters
  this exact row with `exclude_reason="upstream_generated_eval_bug"` and the
  cache was reimported; at that intermediate checkpoint train runnable count
  was `16574`. Existing prompt parquet files produced before this import are
  stale until `check_prompt_data.py` is rerun. This count was later superseded
  by the expanded 35-row generated-eval filter set below.
- Visual audit shard 03 for
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-postconfig-hardened`
  reviewed 30 unseen rows: `true_success=14`, `true_failure=4`,
  `suspected_false_failure=6`, `partial=6`, `suspected_false_success=0`.
  Output:
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-postconfig-hardened.visual_audit.shard_subagent_03.jsonl`;
  contact sheet:
  `/tmp/scalecua_postconfig_hardened_visual_audit_writer_slice_sheet.png`.
  The six suspected false failures are under code audit before any migration
  bug or upstream defect classification is accepted.
- Code audit of shard 03 resolved the six suspected false failures. Four Writer
  rows are exact official generated evaluator defects and are now filtered as
  `upstream_generated_eval_bug`:
  `scalecua_osworld_train_libreoffice_writer_936321ce_5236_426a_9a20_e0e3c5dc536f_task_verify_3`,
  `scalecua_osworld_train_libreoffice_writer_72b810ef_4156_4d09_8f08_a0cf57e7cefe_task_verify_45`,
  `scalecua_osworld_train_libreoffice_writer_6a33f9b9_0a56_4844_9c3f_96ec3ffb3ba2_task_verify_0`,
  and
  `scalecua_osworld_train_libreoffice_writer_0b17a146_2934_46c7_8727_73ff6b6483e8_task_verify_7`.
  Together with the prior Calc row, the then-current
  `upstream_generated_eval_bug` count was 5 and the imported train runnable
  count was 16570.
- The remaining shard 03 suspected migration bug was adapter-specific:
  `scalecua_osworld_train_libreoffice_writer_2b9493d7_49b8_493a_a71b_56cd1f4d6908_task_verify_34`
  called a generated getter with `config["command"]` as an argv list. The
  `_ControllerShim.run_bash_script` adapter now accepts list/tuple commands by
  shell-quoting parts before `bash -lc`, matching the generated getter shape
  without changing shared action-space utilities.
- Follow-up code audits first expanded the exact `upstream_generated_eval_bug`
  source set to 35 generated rows: 7 Calc, 20 GIMP, 3 Impress, and 5 Writer. The
  `0cecd4f3...` Calc sheet-name/order rows are deliberately not excluded; their
  official generated getter/metric wiring is valid, and the lite failure mode
  was fixed in `lite.scalecua/src/osworld/judges.py` by materializing direct VM
  file paths for `(env, config)` generated getters before calling host-local file
  APIs. That intermediate clean reimport had runnable counts
  `train=16540`, `rl=1919`, `eval=321`, and `eval_full=321`; it is superseded by
  the 38-row filter set below.
- A later visual/code subagent pass on the stale postconfig-hardened root added
  three exact upstream generated evaluator defects:
  `scalecua_osworld_train_libreoffice_calc_7e429b8d_a3f0_4ed0_9b58_08957d00b127_task_verify_2`
  has generated VLOOKUP expected areas that do not match the transported
  workbook; `scalecua_osworld_train_chrome_788b3701_3ec9_4b67_b679_418bfa726c22_task_verify_19`
  has impossible generated expected cached formula values for
  `tally_book.xlsx`; and
  `scalecua_osworld_train_vlc_778efd0a_153f_4842_9214_f05fc176b877_task_verify_26`
  expects 7 slides after duplicating a deck that actually starts with 16
  slides. The current exact `upstream_generated_eval_bug` source set is 38
  generated rows; after reimport the expected runnable counts are
  `train=16537`, `rl=1919`, `eval=321`, and `eval_full=321`.
- Superseded by the rerun3 visual/code audit patch below: the exact
  `upstream_generated_eval_bug` source set is now 41 generated rows, and four
  additional generated Chrome social-tab rows are exact local
  `proxy_required` additions. The next clean import is expected to produce
  runnable counts `train=16530`, `rl=1919`, `eval=321`, and `eval_full=321`.
- Current 30314 rollout remains diagnostic because its prompt/server predate
  several filter and adapter changes. For clean accounting, regenerate
  prompt-data against the current catalog, run `check_prompt_data.py`, and start
  a fresh env-server.

2026-07-14 follow-up:

- The stale diagnostic 30314 rollout was stopped at 477 completed
  `summary.json` files before the Dockerfile-level `timedatectl` rebuild and
  current catalog reimport. It must not be used as the acceptance 1000-task
  run.
- Direct-mode/orphan check after shutdown found no `lite-env-30314-*` or
  `lite_scalecua_oracle_validate` containers remaining.
- Visual subagent audit of priority rows 1-10 found #2-#5 were ordinary
  reward/visual successes, #7 was a real agent failure, #1 was transient setup
  capacity, and #6/#8/#9/#10 were strong `likely_migration_or_eval_bug`
  candidates spanning VS Code folder-open, text answer evaluation, unpacked
  extension/file-system output, and Calc table fill.
- Visual subagent audit of priority rows 11-20 found #12/#14 were ordinary
  agent failures, #13 was an upstream/filter candidate, and
  #11/#15/#16/#17/#18/#19/#20 were `likely_migration_or_eval_bug` candidates
  spanning spreadsheet/PPTX persistence, wrong source-domain labels, media
  transform/file checks, moved-file checks, and generated XLSX address output.
- These visual mismatches require code-side probes against the official
  generated SCALE-CUA getter/metric before either patching adapters or adding
  exact `exclude_reason`. They should not block oracle fixture development.

2026-07-14 clean 1000 reruns:

- `gpt-5.5-1000-clean-20260714` and
  `gpt-5.5-1000-clean-20260714-rerun1` are invalid accounting attempts. The
  first started before a stable env-server was available; the second missed
  `CUA_LITE_ENV_SERVER_URL` / `CUA_LITE_ENV_SERVER_TOKEN` and tried an older
  port. Do not include their errors in rollout quality statistics.
- `gpt-5.5-1000-clean-20260714-rerun2` reached real step-0 execution but the
  rollout client disappeared, leaving eight `lite-env-30316-*` containers. The
  30316 server was stopped and those eight containers were removed; a follow-up
  check found no 30316 listener, rollout process, or matching containers.
- Rerun3 was initially started as the next clean-run candidate:
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-clean-20260714-rerun3`.
  It uses dedicated env-server port 30317, token
  `lite-scalecua-1000-clean-rerun3-20260714`, prompt parquet
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-clean-20260714-rerun3/prompt.parquet`,
  `--concurrency 8`, and `--max-attempts 1`. Initial health checks showed the
  rollout process and env-server process alive, 8 `lite-env-30317-*`
  containers, and 0 `error.txt` files. Visual audit was started while the
  rollout was running, once enough completed summaries existed. Later rows in
  this log downgrade it to diagnostic after visual/code audit.
- GPT-5.5 should normally be in the 70-80% success range on supported
  OSWorld-style rows. If the next clean run lands below 70% visual success
  overall, by split, or in a large domain slice, classify that as a
  migration/eval/task alarm first, not as a model-quality conclusion. The
  required follow-up is visual audit plus official SCALE-CUA getter/metric
  comparison, targeted persisted-state probes, and oracle/no-op coverage for
  the affected setup x eval families.
- First visual-audit sidecar for rerun3 wrote
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-clean-20260714-rerun3/visual_audit.sidecar.jsonl`
  and `.md`. It reviewed 19 early Chrome rows: 10 reported reward-1 rows, all
  9 reported reward-0 rows available at that checkpoint, and no partial rows.
  It found no clear `false_success`, 8 `true_success`, 1
  `true_success_artifact`, 1 hidden-config reward-1 row that is not visually
  decidable, 3 ordinary/blocked true failures, and 6
  `false_failure`/metric-mismatch candidates. The candidates are
  `is_expected_tabs` tab canonicalization, strict
  `is_expected_active_tab_approximate` settings URL, Chrome homepage-vs-startup
  semantics, two VLC hidden-config/global-hotkey checks, and
  `is_expected_bookmarks` URL/name matching. A code-audit sidecar is assigned
  before any of these are counted as model failures.
- Rerun3 was then downgraded from clean acceptance to diagnostic. It was stopped
  at 44 completed `summary.json` files and 0 `error.txt` files after the first
  visual/code audit found catalog-level filter/evaluator issues that need
  importer changes before clean accounting. A post-stop check found no
  `30317` listener, no lite.scalecua rollout process for that root, and no
  `lite-env-30317-*` containers. The next acceptance attempt must regenerate
  prompt data after the filter/eval patches and use a new env-server port,
  token, and log root.

2026-07-14 rerun4 clean-entry refresh:

- Rerun3 visual/code audit patches were applied before starting a new clean
  root: three exact Chrome generated evaluator defects now use
  `exclude_reason="upstream_generated_eval_bug"`, and
  `scalecua_osworld_train_chrome_06fe7178_4491_4589_810f_2e2bc9502122_task_verify_{30,31,32,33}`
  now use `exclude_reason="proxy_required"` with
  `metadata.others.proxy=true`.
- Focused and full importer tests passed:
  `uv run --no-sync pytest tests/gym/envs/lite/test_scalecua.py -q` completed
  `85 passed`.
- Static validation passed:
  `uv run --no-sync python devs/envs/lite.scalecua/validate/static.py`.
- Fresh `install.sh import` produced current runnable counts
  `train=16530`, `rl=1919`, `eval=321`, `eval_full=321`; train exclusion
  counts include `upstream_generated_eval_bug=41`, `proxy_required=1225`, and
  `proxy_true_count=1425`.
- Rerun4 prompt data was regenerated at
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-clean-20260714-rerun4/prompt.parquet`
  with manifest
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-clean-20260714-rerun4/prompt.manifest.json`.
  `check_prompt_data.py` passed: 1000 rows, 50 tasks per `train`/`rl` domain,
  and 0 currently excluded tasks.
- Env-server startup for rerun4 initially wasted time because a background
  `uv run` launch was tied to the tool process group and disappeared without a
  Python traceback. The stable startup command uses `setsid` and direct
  `.venv/bin/python`:
  port `30318`, token `lite-scalecua-1000-clean-rerun4-20260714`, pid
  `836105`, log
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-clean-20260714-rerun4/server/envserver.log`.
  `/host_status` was healthy before rollout launch.
- Rerun4 rollout was launched with `setsid`, `--concurrency 8`,
  `--max-attempts 1`, `--save-data true`, `--save-video false`, and
  `--save-gif false`; log root is
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-clean-20260714-rerun4/rollout`
  and process ids observed were wrapper `863067` / python `863071`. Initial
  check showed 8 live env-server instances, 0 `summary.json`, and 0
  `error.txt`.

2026-07-14 rerun4 early visual/code audit:

- Rerun4 produced useful diagnostic failures but is no longer a clean
  acceptance root after exact filter changes. At the time of the first visual
  batches, 22 reward-0 rows were turned into contact sheets under
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-clean-20260714-rerun4/visual_batches/`.
- The first large visual subagent was closed because it did not produce a
  sidecar quickly enough. The process fix is to shard reward-0 visual review
  into small contact-sheet batches, require a sidecar/log entry, and keep code
  audit running separately.
- Code audit found no broad `lite.scalecua` adapter migration bug in the early
  Chrome failures. Confirmed upstream/generated issues:
  `scalecua_osworld_train_chrome_386dbd0e_0241_4a0a_b6a2_6704fba26b1c_task_verify_{8,9}`
  call a generated VLC metric path that references missing helper
  `is_valid_keyboard_shortcut`; they now use
  `exclude_reason="upstream_generated_eval_bug"`.
- Five exact instruction/eval mismatches now use
  `exclude_reason="instruction_eval_mismatch"`:
  `386dbd0e...task_verify_17`, `f3977615...task_verify_{66,91}`, and
  `f5d96daf...task_verify_{22,51}`. These instructions ask for how-to/list or
  comparison answers while eval requires hidden VLC config or Apple compare URL
  state.
- Remaining visually suspicious rows such as Recreation.gov URL checks,
  Ticketek FAQ/terms pages, Nike filter pages, and Google Flights/Ryanair-style
  travel rows are probe candidates, not yet filters. They require final URL/DOM
  or persisted-state probes before classification as live-site drift,
  false-failure, or model failure.
- After applying these exact filters, `install.sh import` produced
  `train=16523`, `rl=1919`, `eval=321`, `eval_full=321`; train exclusions now
  include `instruction_eval_mismatch=14` and
  `upstream_generated_eval_bug=43`. `devs/envs/lite.scalecua/validate/static.py`
  passed, and `uv run --no-sync pytest tests/gym/envs/lite/test_scalecua.py -q`
  passed with `86 passed`.

2026-07-14 rerun4 stop after stale-filter refresh:

- Additional visual/code audit and exact filter updates superseded rerun4
  before acceptance accounting. The current imported catalog after the latest
  `install.sh import` is `train=16408`, `rl=1916`, `eval=321`, and
  `eval_full=321`; train exclusions include
  `upstream_generated_eval_bug=59`.
- Rerun4 was stopped as a diagnostic root only. The rollout directory currently
  has 544 completed `summary.json` files: raw diagnostic reward count is
  `352/544 = 64.7%`, with 168 reward-0 rows and 24 non-binary/partial returns.
  Do not use this number for acceptance because the prompt parquet and server
  catalog predate the latest exact filters.
- Cleanup check after stopping rerun4 found no `30318` listener, no
  `gpt-5.5-1000-clean-20260714-rerun4` rollout process, and no
  `lite-env-30318-*`, `lite.scalecua`, or `lite_scalecua_oracle_validate`
  containers. The next acceptance attempt must regenerate prompt data from the
  current catalog and start a fresh env-server on a new port with concurrency 8.

2026-07-14 rerun4 visual/code follow-up:

- A focused visual/code audit resolved four suspicious Office rows. Two rows
  remain `needs_probe` pending final file/getter captures:
  `scalecua_osworld_train_libreoffice_calc_8b1ce5f2_59d2_4dcc_b0b0_666a714b9a14_task_verify_3`
  and
  `scalecua_osworld_train_libreoffice_impress_15aece23_a215_4579_91b4_69eec72e18da_task_verify_14`.
- One confirmed official generated evaluator defect was exact-filtered:
  `scalecua_osworld_train_libreoffice_calc_f9584479_3d0d_4c79_affa_9ad7afdd8850_task_verify_12`
  now uses `exclude_reason="upstream_generated_eval_bug"` because the generated
  evaluator checks fixed `G` cells while the visible/natural Average column is
  in `F`; sibling `_task_verify_10` was already filtered for the same family.
- One migration adapter bug was fixed instead of filtered:
  `check_pptx_shape_text__8b4cb395` now unwraps generated `{"value": ...}`
  expected rules before calling the metric. This covers
  `scalecua_osworld_train_libreoffice_impress_0a211154_fda0_48d0_9274_eaac4ce5486d_task_verify_43`.
- Fresh `install.sh import` after this follow-up reports `train=16407`,
  `rl=1916`, `eval=321`, `eval_full=321`, and
  `upstream_generated_eval_bug=60`. Static validation passed; focused tests
  passed with `151 passed`. Rerun4 remains diagnostic/stale and the next clean
  rollout still requires regenerated prompt-data plus a fresh env-server.

2026-07-14 rerun5 clean rollout start:

- Prompt data was regenerated from the current catalog at
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-clean-20260714-rerun5/prompt.parquet`
  with manifest
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-clean-20260714-rerun5/prompt.manifest.json`.
  `check_prompt_data.py` passed: 1000 rows, 50 tasks per `train`/`rl` domain,
  and 0 currently excluded rows.
- A fresh dedicated env-server is running on `127.0.0.1:30319` with token
  `lite-scalecua-1000-clean-rerun5-20260714`, pid `186151`, max live envs 8,
  reset concurrency 2, and warm-pool spawn concurrency 2. The first launch
  wrote the pid file incorrectly because of shell operator precedence, but the
  server itself stayed healthy; the pid file was corrected manually.
- Rerun5 rollout is running via that env-server with `--concurrency 8`,
  `--max-attempts 1`, `--save-data true`, `--save-video false`, and
  `--save-gif false`; rollout pid is `203364`, log root is
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-clean-20260714-rerun5/rollout`.
  The first monitor pass saw 3 completed `summary.json` files, 0 `error.txt`
  files, and raw rewards `3/3 = 100%`; this is only an initial liveness check,
  not acceptance accounting.

2026-07-14 rerun5 early visual/code audit:

- At 54 completed trajectories, `audit_queue.py` reported 37 reward-1,
  17 reward-0, and 0 partial rewards. Completed rows were still dominated by
  Chrome (`50`) with early GIMP coverage (`4`), so this is not a full-domain
  success-rate estimate.
- Visual/code audit confirmed two migration adapter bugs and two exact upstream
  filters. Fixed locally: `is_expected_bookmarks` now drops the injected
  `env` kwarg before delegating to upstream metrics, and Chrome extension
  repair treats hashed `extension_names__...` as list-valued. Exact filtered:
  `06fe7178...task_verify_34` as `upstream_live_site_drift` and
  `480bcfea...task_verify_39` as `upstream_generated_eval_bug`.
- Replaying the original `Work` bookmark rollout actions in direct mode with
  the patched code returned final reward 1.0. The earlier direct file-write
  probe is not evidence because Chrome can overwrite the profile file from
  in-memory state during postconfig relaunch.
- Fresh `install.sh import` after the fixes reports `train=16405`,
  `rl=1916`, `eval=321`, `eval_full=321`,
  `upstream_generated_eval_bug=61`, and `upstream_live_site_drift=202`.
  Static validation and the focused/full scalecua test suite passed
  (`152 passed`).
- The still-running rerun5 env-server predates these patches, so its remaining
  output is diagnostic for finding more bugs. A clean acceptance run must use a
  freshly started env-server and prompt data regenerated after the latest
  import.
