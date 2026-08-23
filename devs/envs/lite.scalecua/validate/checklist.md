# lite.scalecua Validation Checklist

This checklist tracks release gates from `../scalecua.md`. Check a box only when
the named evidence exists in the current workspace.

## Governance

- [x] `../scalecua.md` is English-only, internally consistent, and audited against
  `lite.osworld`, `lite.cuagym`, current `lite.scalecua`, and upstream
  `SCALE-CUA`.
- [x] Current split counts and candidate counts in `../scalecua.md` match the
  imported catalogs and coverage artifacts.
  Evidence: refreshed `.cache/lite.scalecua_tasks/import_report.json`,
  `.exps/validate/lite.scalecua/oracle/verified/rl.verified.closed_20260716.json`,
  and regenerated task catalogs now use RL `1,839` runnable rows, `1,839` RL
  fixture rows, and `1,839` strict verified RL fixture rows.
- [ ] Linked `devs/envs/lite.scalecua` docs and runtime READMEs are fully
  synchronized with `../scalecua.md`.

## Import And Setup/Eval Transport

- [x] ScaleCUA deterministic setup failures use a typed non-retryable
  `ScaleCuaTaskError`.
  Evidence: `uv run --no-sync pytest
  tests/gym/envs/lite/scalecua/test_scalecua_setup.py::test_scalecua_task_errors_are_nonretryable_and_round_trip
  tests/gym/envs/lite/scalecua/test_scalecua_setup.py::test_setup_rejects_excluded_task_before_dispatch
  tests/gym/envs/lite/scalecua/test_scalecua_setup.py::test_scalecua_setup_raises_on_failed_execute_result
  tests/gym/errors/test_gym_errors.py::test_is_retryable_by_exception_type -q` passed.
- [x] Classify all current `import_report.normalization_notes` exceptions:
  `active_setting_noop=2` maps to
  `scalecua_osworld_train_libreoffice_writer_f178a4a9_d090_4b56_bc4c_4b72a61a035d_task_verify_95`,
  where the metric reads LibreOffice registry state, not pseudo
  active-setting state; `get_tabs_info_noop=1` maps to
  `scalecua_osworld_train_chrome_58565672_7bfe_48ab_b828_db349231de6b_task_verify_30`
  with
	  `unsupported_schema:evaluator_postconfig_query_config`, so it is not a
	  runnable task gap; `drop_stale_rl_a462_charles_setup=7` maps to the seven RL
	  `scalecua_osworld_rl_os_a462a795_fdc7_4b23_b689_e8b6df786b78_traj_verify_{0..6}`
	  source rows: `traj_verify_0` and `traj_verify_1` are exact
	  `upstream_generated_eval_bug` excludes for stdout-newline-sensitive
	  `exact_match` evaluators, while the five remaining runnable rows have stale
	  inherited Charles-login setup removed and source-backed oracle fixtures from
	  `src/gen/oracle/domains/os.py` in `data/oracle/rl.jsonl`;
	  `gimp_export_full_path_postconfig=231`,
	  `legacy_top_level_action_parameters=6`,
	  `local_placeholder_image_download=1`,
	  `normalize_update_desktop_database=16`, and
	  `repair_root_home_test1_setup=22` are classified in `../scalecua.md` as
	  ScaleCUA importer/setup/eval compatibility normalizations. Evidence:
  `.cache/lite.scalecua_tasks/import_report.json`, focused importer tests, and
  targeted no-op/oracle validation artifacts under `.exps/validate/lite.scalecua/oracle/`.
- [x] Fresh `install.sh provision` and `validate/static.py` pass after the current
  source changes.
  Evidence: `uv run --no-sync bash
  lite/gym/envs/lite/scalecua/scripts/install.sh provision` refreshed catalogs;
  `uv run --no-sync python devs/envs/lite.scalecua/validate/static.py` passed.
- [x] Focused ScaleCUA registry/setup/eval tests pass after the current source
  changes.
  Evidence: `uv run --no-sync pytest tests/gym/envs/lite/scalecua
  tests/agents/test_registration_complete.py -q` passed.

## Oracle Closure

Current source layout: release-countable oracle rows live in
`lite/gym/envs/lite/scalecua/src/gen/oracle/domains/<domain>.py` and are
byte-locked into `data/oracle/rl.jsonl` / `train.jsonl`. Historical
`rl_auto_*` names below are retained as fixture provenance and replay-log
identifiers, not as current source file paths.

- [x] Integrate or reject `rl_auto_office_b1.py`.
  Evidence: shard is registered in
  `lite.gym.envs.lite.scalecua.src.gen.oracle.__main__.SHARDS`; generator
  check byte-locks its rows inside `data/oracle/rl.jsonl`.
- [x] Integrate or reject `rl_auto_gimp_tb_b1.py`.
  Evidence: shard is registered in
  `lite.gym.envs.lite.scalecua.src.gen.oracle.__main__.SHARDS`; generator
  check byte-locks its rows inside `data/oracle/rl.jsonl`; targeted replay
  `oracle_rl_auto_gimp_tb_b1_0089` passed no-op reward `0.0` and oracle reward
  `1.0` under
  `.exps/validate/lite.scalecua/oracle/thunderbird-nested-folder-0089-20260715/`,
  confirming the nested Thunderbird folder hierarchy is no longer generated as
  a flat folder.
- [x] Integrate or reject `rl_auto_gimp_b2.py`.
  Evidence: shard is registered in
  `lite.gym.envs.lite.scalecua.src.gen.oracle.__main__.SHARDS`; generator
  check byte-locks its rows inside `data/oracle/rl.jsonl`; full replay
  under `.exps/validate/lite.scalecua/oracle/rl-auto-gimp-b2-full-20260715/`
  passed `9/9` with no-op reward `0.0` and oracle reward `1.0`.
- [x] Integrate or reject `rl_auto_multi_apps_b1.py`.
  Evidence: shard is registered in
  `lite.gym.envs.lite.scalecua.src.gen.oracle.__main__.SHARDS`; generator
  check byte-locks its rows inside `data/oracle/rl.jsonl`; the current
  shard contains 92 rows after the 23-row evaluator-backed expansion.
- [x] Integrate or reject `rl_auto_os_files_b2.py`.
  Evidence: shard is registered in
  `lite.gym.envs.lite.scalecua.src.gen.oracle.__main__.SHARDS`; generator
  check byte-locks its rows inside `data/oracle/rl.jsonl`; the
  replay canary
  `.exps/validate/lite.scalecua/oracle/rl-auto-os-files-b2-canary-final-20260715/`
  passed `4/4` after removing unsafe terminal-output and VS Code extension
  rows from the shard.
- [x] Integrate or reject `rl_auto_thunderbird_source_b1.py`.
  Evidence: shard is registered in
  `lite.gym.envs.lite.scalecua.src.gen.oracle.__main__.SHARDS`; generator
  check byte-locks its rows inside `data/oracle/rl.jsonl`; the replay canary
  `.exps/validate/lite.scalecua/oracle/rl-auto-thunderbird-source-b1-canary-20260715/`
  passed `4/4` after skipping three unsafe account/filter candidates.
- [x] `python -m lite.gym.envs.lite.scalecua.src.gen.oracle --check` passes
  for every registered source-backed domain.
  Evidence: `uv run --no-sync python -m
  lite.gym.envs.lite.scalecua.src.gen.oracle --check` passed for all 10
  registered domains.
- [x] `rl_auto_vscode_files_b1` source-backed shard has full replay evidence.
  Evidence: full replay under
  `.exps/validate/lite.scalecua/oracle/rl-auto-vscode-files-b1-full-20260715/`
  passed `14/15`; `oracle_rl_auto_vscode_files_b1_0015` exposed an oracle
  recipe bug where the generated `test.py` had fewer than 10 lines, was fixed
  and is now folded into `src/gen/oracle/domains/vs_code.py`; targeted
  replay under
  `.exps/validate/lite.scalecua/oracle/rl-auto-vscode-files-b1-0015-fix-20260715/`
  passed `1/1`. The strict verified inventory now counts all 15 rows.
- [x] `rl_auto_gimp_b3` source-backed shard has full replay evidence.
  Evidence: worker probe under
  `.exps/validate/lite.scalecua/oracle/rl-auto-gimp-b3-full-20260715/`
  passed `4/5`; the unsafe horizontal-flip row was rejected because evaluator
  reference repair materializes `original_path` after the oracle action. The
  promoted 4-row shard is registered, byte-locked by generator `--check`, and
  full replay under
  `.exps/validate/lite.scalecua/oracle/rl-auto-gimp-b3-promote-20260715/`
  passed `4/4` with no-op reward `0.0` and oracle reward `1.0`.
- [x] `rl_auto_os_files_b1` source-backed shard has full replay evidence.
  Evidence: prior strict inventory counted `21/23` rows; targeted replay for
  `oracle_rl_auto_os_files_b1_0021` and `oracle_rl_auto_os_files_b1_0022`
  under
  `.exps/validate/lite.scalecua/oracle/rl-auto-os-files-b1-tail-0021-0022-20260715/`
  passed `2/2`, with no-op reward `0.0` and oracle reward `1.0`.
- [x] `rl_auto_vlc_b2` source-backed shard has full replay evidence.
  Evidence: worker candidate replay under
  `.exps/validate/lite.scalecua/oracle/rl-auto-vlc-b2-candidate-20260715/`
  passed `3/3` with no-op reward `0.0` and oracle reward `1.0`; the shard was
  registered, byte-locked by generator `--check`, and promoted with 3 rows.
- [x] Office oracle fixtures do not satisfy export/save-as tasks with fake
  blank artifacts when a source document is required.
  Evidence: `oracle_rl_auto_office_b2_0044` now copies
  `Downloads/Secrets-of-Monetizing-Video.pptx`, deletes the first two slides,
  and saves `Desktop/trimmed.pptx`; the row is now folded into the relevant
  `src/gen/oracle/domains/libreoffice_*` domain source, full generator
  `--check` passed, and production oracle replay under
  `.exps/validate/lite.scalecua/oracle/office-b2-trimmed-pptx-0044-20260715/`
  returned no-op reward `0.0` and oracle reward `1.0`.
- [x] Office Impress oracle helpers obey evaluator shape-index semantics when
  the upstream getter is index-based.
  Evidence: `oracle_rl_auto_office_b2_0045` now writes
  `shape_alignment(shape=1)` and `shape_color(shape=0)` for
  `impress_content_align_title_color`, matching the upstream getter; targeted
  replay under
  `.exps/validate/lite.scalecua/oracle/office-b2-0045-shape-index-fix-20260715/`
  returned no-op reward `0.0` and oracle reward `1.0`, and the 0040-0046 slice
  under
  `.exps/validate/lite.scalecua/oracle/office-b2-pptx-slice-0040-0046-shape-index-fix-20260715/`
  passed `7/7`.
- [x] Integrate or reject `rl_auto_office_b3.py`.
  Evidence: shard is registered in
  `lite.gym.envs.lite.scalecua.src.gen.oracle.__main__.SHARDS`; generator
  check byte-locks its rows inside `data/oracle/rl.jsonl`. The
  earlier full replay under
  `.exps/validate/lite.scalecua/oracle/rl-auto-office-b3-full-20260715/`
  passed the pre-insertion `33/33` shard; the fixture-id refresh replay under
  `.exps/validate/lite.scalecua/oracle/rl-auto-office-b3-0015-0034-id-refresh-20260715/`
  passed `20/20`, and the refreshed RL verified inventory now counts all 34
  current rows as verified.
- [x] RL coverage inventory has zero catalog/exclusion/duplicate problems.
  Evidence: `uv run --no-sync python
  devs/envs/lite.scalecua/validate/oracle/coverage_inventory.py --splits rl ...`
  now reports `fixture_rows: 1839`, `fixture_problem_rows: 0`,
  `duplicate_fixture_tasks: 0`, eval/setup combos `1335 / 1335`, and full
  setup/eval/postconfig combos `1363 / 1363`.
- [x] Verified oracle inventory distinguishes fixture coverage from replay
  evidence. Evidence: `uv run --no-sync python
  devs/envs/lite.scalecua/validate/oracle/verified_inventory.py --splits rl ...`
  now reports `verified_fixture_rows: 1839`, `current_fixture_rows: 1839`,
  `unverified_fixture_rows: 0`, `verified_catalog_rows: 1839`, and
  `unverified_catalog_rows: 0`; both `--require-complete` and
  `--require-catalog-complete` pass at this checkpoint.
- [x] `oracle_bulk_seed_06_*` fixtures have full replay evidence.
  Evidence: `uv run --no-sync python
  devs/envs/lite.scalecua/validate/oracle/validate.py ...` historically passed
  `26/26` for the same fixture ids under
  `.exps/validate/lite.scalecua/oracle/bulk-seed-06-full-20260715/`; the
  current aggregate layout stores those rows in `data/oracle/rl.jsonl` and
  `data/oracle/train.jsonl`.
- [x] `oracle_rl_vscode_seed_02_*` fixtures have full replay evidence.
  Evidence: `uv run --no-sync python
  devs/envs/lite.scalecua/validate/oracle/validate.py ...` historically passed
  `14/14` for the same fixture ids under
  `.exps/validate/lite.scalecua/oracle/rl-vscode-seed-02-full-20260715/`; the
  current aggregate layout stores those rows in `data/oracle/rl.jsonl`.
- [x] `oracle_rl_chrome_seed_01_*` fixtures have full replay evidence after the broad
  Chrome profile discovery shim fix.
  Evidence: `uv run --no-sync python
  devs/envs/lite.scalecua/validate/oracle/validate.py ...` historically passed
  `16/16` for the same fixture ids under
  `.exps/validate/lite.scalecua/oracle/rl-chrome-seed-01-full-shim-fix-20260715/`;
  focused Chrome profile alias tests in
  `tests/gym/envs/lite/scalecua/test_scalecua_chrome_profile.py`, the split
  ScaleCUA suite under `tests/gym/envs/lite/scalecua/`, and
  `tests/agents/test_registration_complete.py` passed, and the refreshed RL
  verified inventory counts all 16 rows in that shard as verified.
- [x] `rl_auto_office_b1.py` has full current replay evidence after fixing
  direct-file replay lifecycle, Writer title paragraph indexing/default-font
  mutations, and Impress max-font title-shape selection.
  Evidence: historical focused generation produced `103` rows inside
  `data/oracle/rl.jsonl`; the current source is folded into
  `src/gen/oracle/domains/libreoffice_*`; full replay passed `103/103` under
  `.exps/validate/lite.scalecua/oracle/rl-auto-office-b1-full-current-20260715/`
  with report
  `.exps/validate/lite.scalecua/oracle/rl-auto-office-b1-full-current-20260715.report.jsonl`.
  That snapshot reported a closed local fixture gate; the refreshed current RL
  inventory now reports `verified_fixture_rows: 1839`,
  `current_fixture_rows: 1839`, `unverified_fixture_rows: 0`, and
  `unverified_catalog_rows: 0`; the full catalog oracle closure gate is closed
  for the current import.
- [x] `rl_auto_os_files_b2` excludes the impossible RL notebook-delete row
  instead of keeping an unpassable oracle fixture.
  Evidence: `oracle_rl_auto_os_files_b2_0020` exposed that official
  `check_include_exclude` requires `*_failed.ipynb` includes while excluding the
  substring `d.ipynb`; the exact task
  `scalecua_osworld_rl_os_5c1075ca_bb34_46a3_a7a0_029bd7463e79_traj_verify_1`
  now has `exclude_reason="upstream_generated_eval_bug"`, `install.sh provision`
  and `validate/static.py` passed, `src/gen/oracle/domains/os.py` preserves the
  remaining rows in `data/oracle/rl.jsonl`, and `src.gen.oracle --check` passed
  for all registered domains.
- [x] Every promoted or source-backed oracle action, including every
  `data/oracle/*.jsonl` row counted toward RL closure, has full no-op negative
  plus oracle-positive replay evidence. Coverage inventory rows and canary-only
  rows do not satisfy this gate. Evidence:
  `verified_inventory.py --splits rl --require-complete --require-artifacts`
  passed with `verified_fixture_rows: 1839`, `current_fixture_rows: 1839`,
  `verified_catalog_rows: 1839`, `unverified_catalog_rows: 0`, and
  `unverified_fixture_rows: 0`; the final closure report is
  `.exps/validate/lite.scalecua/oracle/verified/rl.verified.closed_20260716.json`.
- [x] Unverified candidate JSONL shards are absent from
  `lite/gym/envs/lite/scalecua/data/oracle/` before recording a coverage
  checkpoint. Evidence: the directory now contains only `README.md`,
  `rl.jsonl`, and `train.jsonl`.
- [x] Every runnable RL task has promoted/source-backed no-op and oracle replay
  evidence, or an exact `exclude_reason`. Current strict evidence is
  `1839 / 1839` runnable RL catalog rows; 0 catalog rows remain.

## Rollout And Visual Audit

- [ ] Current diagnostic 1000-task root finishes without orphan env-server or
  direct-mode containers.
- [ ] Every completed trajectory in the diagnostic root has reward plus visual
  audit classification.
- [ ] One clean 1000-task root plus one independent confirmation root satisfy
  the visual success gates in `../scalecua.md`.
