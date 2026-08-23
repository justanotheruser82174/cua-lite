# lite.scalecua Oracle Findings Log

Historical entries may mention old fixed-tag or `:latest` experiments. Current
ScaleCUA oracle validation uses the normal `cua-lite/lite.osworld:latest`
lifecycle unless a specific entry explicitly records historical evidence.

Append-only, grep-friendly record of no-op, oracle replay, and fixture
development findings surfaced by
[`plan.md`](/devs/envs/lite.scalecua/validate/oracle/plan.md).

## Format

One finding per line, inline tags for grep:

```text
<fixture_id_or_task_id_or_family>: <one-line symptom> [sweep=N | type=noop|trivial_pass|oracle_fail|fixture_bug|eval_bug|setup_bug|unsupported | severity=bug|rng|unsupported|blocked]
```

When a finding closes, append `-> fixed in <commit_or_file_change>` to the
same line. When a later sweep confirms it still exists, append
`-> still in sweep N+1`. Never delete old finding lines.

Grep recipes:

```bash
grep 'type=trivial_pass' devs/envs/lite.scalecua/validate/oracle/logs.md
grep 'type=oracle_fail' devs/envs/lite.scalecua/validate/oracle/logs.md
grep 'type=fixture_bug' devs/envs/lite.scalecua/validate/oracle/logs.md
grep -v -- '-> fixed' devs/envs/lite.scalecua/validate/oracle/logs.md
```

## Findings

legacy_top_level_action_parameters: visual/code audit found official ScaleCUA legacy actions whose payload lived at the action top level instead of under `parameters`, e.g. RL Calc `3aaa4e37...traj_verify_3` postconfig `{"type":"key","key":"ctrl+s"}` / `{"type":"sleep","seconds":2}` and generated VS Code extension setup `{"type":"execute","command":[...]}`. The importer previously normalized these to empty parameters, causing bad setup/postconfig rather than agent failure. Fixed in `lite.scalecua` importer by merging known legacy top-level action fields before OSWorld postconfig normalization; cache regeneration keeps counts stable, empty-parameter scan is 0 for train/RL, and `tests/gym/envs/lite/test_scalecua.py` passes 196/196. [sweep=legacy-action-params-20260715 type=setup_bug severity=ok]

oracle_rl_auto_office_b3_0014: added generated RL Office B3 fixture for `scalecua_osworld_rl_libreoffice_calc_3aaa4e37...traj_verify_3`, the rollout-discovered legacy postconfig row. The fixture writes bold A1:A4 in the downloaded workbook, creates `/home/user/Export_Calc_to_CSV.csv`, then lets the normalized Office postconfig run. Fixed-image direct replay under `.exps/validate/lite.scalecua/oracle/rl-auto-office-b3-0014-legacy-postconfig-rerun1-20260715/` passed with no-op reward 0.0 and oracle reward 1.0; eval debug showed `all_col_a_bold=true`, `csv_exists=true`, `csv_row_count=4`, and no direct containers remained. [sweep=legacy-postconfig-office-b3-0014-20260715 type=oracle_pass severity=ok]

gimp_config_flush_adapter: visual rollout found GIMP preference rows where the UI reflected the requested theme/icon state but generated/base evaluators read `~/.config/GIMP/2.10/gimprc` before GIMP had gracefully exited. Added a ScaleCUA-only pre-getter flush for `gimp_config_file`, generated `gimp_theme*`/`gimp_icon_theme*`, and direct `gimprc`/`sessionrc` command paths; unit coverage passed and targeted fixed-image replays `oracle_rl_auto_gimp_tb_b1_{0039,0046,0051}` passed 3/3 under `.exps/validate/lite.scalecua/oracle/gimp-config-flush-*-20260715/`. [sweep=gimp-config-flush-20260715 type=eval_bug severity=ok]

rl_auto_vscode_files_b1_full: full source-backed shard replay under `.exps/validate/lite.scalecua/oracle/rl-auto-vscode-files-b1-full-20260715/` returned 14/15 pass. The failed row `oracle_rl_auto_vscode_files_b1_0015` scored 0.5 because the oracle-generated `test.py` had fewer than 10 lines, so the official getter's line-2-through-10 indentation check could not pass. Fixed `src/gen/oracle/rl_auto_vscode_files_b1.py` to generate the full bubble-sort body with docstring and enough indented lines, regenerated `data/oracle/rl_auto_vscode_files_b1.jsonl`, and targeted replay under `.exps/validate/lite.scalecua/oracle/rl-auto-vscode-files-b1-0015-fix-20260715/` passed 1/1. Refreshed `rl.verified.json`: strict artifact-backed RL evidence is now 296/1213, with 917 unverified RL fixtures remaining. [sweep=rl-auto-vscode-files-b1-full-20260715 type=oracle_fix severity=ok]

rl_auto_gimp_b3_promote: worker implemented a GIMP/Pillow source-backed shard for four RL image-transform rows. Initial 5-row probe under `.exps/validate/lite.scalecua/oracle/rl-auto-gimp-b3-full-20260715/` passed 4/5; rejected `scalecua_osworld_rl_gimp_58d3eeeb_e9d0_499f_962e_fd0db2a744d8_traj_verify_4` because evaluator reference repair materializes `original_path` from `/home/user/Desktop/heron.jpeg` after oracle action, making it unsafe for this file-transform shard. Promoted `rl_auto_gimp_b3` with 4 rows, generator `--check` passed, and full replay under `.exps/validate/lite.scalecua/oracle/rl-auto-gimp-b3-promote-20260715/` passed 4/4 with no-op reward 0.0 and oracle reward 1.0. Refreshed `rl.coverage.json`: 1217/1910 RL fixture rows, 752/1394 eval x setup, 779/1422 full combos; refreshed `rl.verified.json`: 300/1217 strict artifact-backed RL evidence. [sweep=rl-auto-gimp-b3-promote-20260715 type=oracle_pass severity=ok]

rl_auto_os_files_b1_tail: targeted replay for the two previously unverified source-backed OS filesystem rows `oracle_rl_auto_os_files_b1_0021` and `oracle_rl_auto_os_files_b1_0022` passed 2/2 under `.exps/validate/lite.scalecua/oracle/rl-auto-os-files-b1-tail-0021-0022-20260715/`, with no-op reward 0.0 and oracle reward 1.0. Refreshed `rl.verified.json`: strict artifact-backed RL evidence is now 302/1217, with 915 unverified RL fixtures remaining. [sweep=rl-auto-os-files-b1-tail-0021-0022-20260715 type=oracle_pass severity=ok]

rl_auto_vlc_b2_promote: worker implemented and promoted a 3-row VLC media artifact shard. Full candidate replay under `.exps/validate/lite.scalecua/oracle/rl-auto-vlc-b2-candidate-20260715/` passed 3/3, with no-op reward 0.0 and oracle reward 1.0 for every row; generator `--check` passed after registering `rl_auto_vlc_b2`. Refreshed `rl.coverage.json`: 1220/1910 RL fixture rows, 755/1394 eval x setup, 782/1422 full combos; refreshed `rl.verified.json`: 305/1220 strict artifact-backed RL evidence. [sweep=rl-auto-vlc-b2-candidate-20260715 type=oracle_pass severity=ok]

bulk_seed_02_0015_title_fix: fixed `oracle_bulk_seed_02_0015` after its historical replay scored 0.0 even though the no-op precheck scored 0.0. Root cause was a legacy oracle recipe bug: it created a new minimal `/home/user/Desktop/109_4.pptx` instead of modifying the downloaded ScaleCUA/OSWorld deck shape expected by `check_pptx_slide_title__24e166a4a3628d77570c2919341fa3d0`. The fixture now marks `oracle_after_postconfig=true`, lets the save-sensitive postconfig finish first, kills LibreOffice, edits slide 1's title in the existing PPTX to `Beverages`, and syncs the file. Targeted replay under `.exps/validate/lite.scalecua/oracle/bulk-seed-02-0015-title-fix-20260715/` passed 1/1 with no-op reward 0.0 and oracle reward 1.0. Refreshed `rl.verified.json`: 306/1220 strict artifact-backed RL evidence, 914 unverified RL fixtures; refreshed `current.verified.json`: 310/1437 train+RL strict artifact-backed evidence. [sweep=bulk-seed-02-0015-title-fix-20260715 type=oracle_fix severity=ok]

scalecua_osworld_rl_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_traj_verify_0: no-op negative check passed with reward=0.0; reset/final screenshots showed unchanged Chrome new tab and no no-op validate containers remained [sweep=seed-20260714 type=noop severity=bug] -> baseline ok

oracle_candidate_selector_smoke: selector generated 248 candidate rows and coverage report for non-excluded train+rl; candidates are not executable until `oracle_actions` or `oracle_trajectory` is populated [sweep=seed-20260714 type=fixture_bug severity=blocked] -> first executable seed fixtures were later promoted into the canonical aggregate fixture files under `lite/gym/envs/lite/scalecua/data/oracle/`

oracle_seed_chrome_font_size_rl_0001: first replay failed reward=0.0 after precheck reward=0.0; oracle action used `pkill -f chrome`, which can match and kill the executing shell before the Python Preferences write runs [sweep=seed-20260714 type=fixture_bug severity=bug] -> fixed before promotion by replacing `pkill -f` with exact `killall -9 -q` process names

oracle_seed_chrome_font_size_rl_0001: replay passed with precheck reward=0.0 and oracle reward=1.0; screenshots and debug payload written under `.exps/validate/lite.scalecua/oracle/smoke-seed-20260714-rerun1` [sweep=seed-20260714-rerun1 type=oracle_pass severity=ok]

oracle_seed_vlc_minimal_view_rl_0001: replay passed with precheck reward=0.0 and oracle reward=1.0; direct-mode final sweep completed after one async docker-rm timeout warning [sweep=seed-20260714-rerun2 type=oracle_pass severity=ok]

oracle_seed_vlc_max_volume_rl_0001: replay passed with precheck reward=0.0 and oracle reward=1.0 [sweep=seed-20260714-rerun2 type=oracle_pass severity=ok]

oracle_seed_vscode_font_size_rl_0001: replay passed with precheck reward=0.0 and oracle reward=1.0 [sweep=seed-20260714-rerun2 type=oracle_pass severity=ok]

oracle_gimp_action_history_gaussian_blur_train_0001: seed replay passed with precheck reward=0.0 and oracle reward=1.0; confirms the basic GIMP action-history setup/eval path is functional [sweep=gimp-seed-20260714-smoke1 type=oracle_pass severity=ok]

scalecua_metric_reference_paths: official generated/RL evaluators contain `source_path` / `original_path` / `source_cache_path` fields that generated metrics open directly on the host; field scan found 88 affected rows, including 25 author-cache `/home/lvbowen/...` rows. This is an eval-side asset transport gap, not a setup download failure [sweep=path-ref-20260714 type=eval_bug severity=bug] -> fixed in `lite.scalecua/src/osworld/verify.py` by materializing VM and author-cache reference paths into eval cache before metric invocation

oracle_gimp_author_cache_reference_flip_train_0013: replay passed with precheck reward=0.0 and oracle reward=1.0; confirms `/home/lvbowen/.../character.png` expected-rule references are fetched from `/home/user/Desktop/character.png` before `check_image_flipped__07b5058cd3df711389d2b4342d0c561c` opens them [sweep=path-ref-20260714 type=oracle_pass severity=ok]

oracle_desktop_vscode_project_wordwrap_train_0001: seed replay passed with precheck reward=0.0 and oracle reward=1.0; final sweep removed one validate container after async docker-rm timeout warning [sweep=desktop-seed-20260714-smoke1 type=oracle_pass severity=ok]

oracle_vlc_seed_train_playing_status_0001: seed replay passed with precheck reward=0.0 and oracle reward=1.0; no-op saw VLC stopped and replay saw VLC playing, exercising the official `:password` VLC status fallback [sweep=vlc-seed-20260714-smoke1 type=oracle_pass severity=ok]

bulk_seed_01_original_0001_0006: first runtime replay of the original bulk seed failed 0/6; failures were fixture/action bugs, mainly direct `openpyxl` writes racing LibreOffice postconfig save/overwrite and task/action mismatches for complex Office tasks [sweep=bulk-seed-01-original-20260714 type=fixture_bug severity=bug] -> fixed by removing the six invalid rows from `bulk_seed_01.jsonl`

oracle_bulk_seed_01_0007: replay failed reward=0.0 because the fixture wrote the wrong workbook/cells for the task instruction/evaluator; this is a fixture bug, not an eval adapter failure [sweep=bulk-seed-01-trimmed-smoke-20260714 type=fixture_bug severity=bug] -> fixed by removing the row from `bulk_seed_01.jsonl`

oracle_bulk_seed_01_0008: replay failed reward=0.0 because the fixture action did not match the task's expected cell comment target; this is a fixture bug, not an eval adapter failure [sweep=bulk-seed-01-trimmed-smoke-20260714 type=fixture_bug severity=bug] -> fixed by removing the row from `bulk_seed_01.jsonl`

oracle_bulk_seed_01_0010: replay failed reward=0.0 because `touch /home/user/Desktop/res.pdf` does not satisfy the official PDF export/file validation; this is a fixture bug, not an eval adapter failure [sweep=bulk-seed-01-trimmed-smoke-20260714 type=fixture_bug severity=bug] -> fixed by removing the row from `bulk_seed_01.jsonl`

oracle_bulk_seed_01_0009: replay passed with precheck reward=0.0 and oracle reward=1.0 for a command-line ODS age-sum output task [sweep=bulk-seed-01-trimmed-smoke-20260714 type=oracle_pass severity=ok]

oracle_bulk_seed_01_0011: replay passed with precheck reward=0.0 and oracle reward=1.0 for LibreOffice autosave interval configuration [sweep=bulk-seed-01-trimmed-smoke-20260714 type=oracle_pass severity=ok]

oracle_bulk_seed_01_0012: replay passed with precheck reward=0.0 and oracle reward=1.0 for LibreOffice Writer default heading font configuration [sweep=bulk-seed-01-trimmed-smoke-20260714 type=oracle_pass severity=ok]

bulk_seed_03_full: runtime replay started for 60 deterministic VS Code, Thunderbird, multi_apps, and Chrome/VLC configuration fixtures; first 16 VS Code fixtures passed with precheck reward=0.0 and oracle reward=1.0 before the sweep entered Thunderbird [sweep=bulk-seed-03-full-20260714 type=oracle_pass severity=ok]

bulk_seed_03_full: completed 53/60 passing. Four rows (`oracle_bulk_seed_03_0038` through `oracle_bulk_seed_03_0041`) targeted tasks now filtered as `proxy_required`; three rows (`oracle_bulk_seed_03_0057`, `oracle_bulk_seed_03_0059`, `oracle_bulk_seed_03_0060`) were trivial-pass VLC global play/pause fixtures whose no-op precheck already returned reward 1.0 [sweep=bulk-seed-03-full-20260714 type=trivial_pass severity=bug] -> fixed by removing those seven rows from `bulk_seed_03.jsonl`

bulk_seed_03_trimmed_full: runtime replay started for the trimmed 53-row `bulk_seed_03.jsonl` with concurrency 4 under `.exps/validate/lite.scalecua/oracle/bulk-seed-03-trimmed-full-20260714` [sweep=bulk-seed-03-trimmed-full-20260714 type=oracle_pass severity=ok]

bulk_seed_05_full: completed 47/53 passing under `.exps/validate/lite.scalecua/oracle/bulk-seed-05-full-20260714`; four rows were unsuitable stable fixtures because the no-op precheck already returned non-zero reward (`oracle_bulk_seed_05_0028` VLC `qt-bgcone`, `oracle_bulk_seed_05_0036` Chrome startup new-tab, `oracle_bulk_seed_05_0043` reconciled invoices partial 0.3, `oracle_bulk_seed_05_0060` copy+clipboard partial 0.5) [sweep=bulk-seed-05-full-20260714 type=trivial_pass severity=bug] -> fixed by removing those four rows from `bulk_seed_05.jsonl`

oracle_bulk_seed_05_0041: replay failed reward=0.0 because the generated getter `get_docx_file_count__bb5651c2` directly opens `config["path"]` on the host; this is the ScaleCUA generated direct-VM-path evaluator gap, not a fixture bug [sweep=bulk-seed-05-full-20260714 type=eval_bug severity=bug] -> fixed in `lite.scalecua/src/osworld/judges.py` by AST-gated VM file materialization for direct file APIs

oracle_bulk_seed_05_0050: replay failed reward=0.0 because the fixture copied only `*hk*.jpg`; the official expected files also include `hong-kong-china.jpg`, which does not contain the literal substring `hk` [sweep=bulk-seed-05-full-20260714 type=fixture_bug severity=bug] -> fixed in `bulk_seed_05.jsonl` by copying the two exact expected filenames

bulk_seed_05_trimmed: after removing four trivial-pass rows and fixing `oracle_bulk_seed_05_0050`, `bulk_seed_05.jsonl` contains 49 executable fixtures. This historical entry was blocked on a stale OSWorld image missing the Dockerfile-level `timedatectl` shim; replay under the current lifecycle should use a freshly rebuilt `cua-lite/lite.osworld:latest` image. [sweep=bulk-seed-05-targeted-fix-20260714 type=setup_bug severity=blocked]

bulk_seed_05_targeted_fix_rerun1: historical replay of `oracle_bulk_seed_05_0041` and `oracle_bulk_seed_05_0050` passed 2/2 with no-op reward 0.0 and oracle reward 1.0 after the OSWorld image included the Dockerfile-level `timedatectl` shim. This confirms both the direct VM path materialization fix for generated getters and the corrected `hk_collection` fixture action; future replay should follow the current `cua-lite/lite.osworld:latest` lifecycle. [sweep=bulk-seed-05-targeted-fix-20260714-rerun1 type=oracle_pass severity=ok]

oracle_coverage_inventory_current: `coverage_inventory.py` matched 303 committed fixture rows to current runnable `train`/`rl` rows with 0 missing/excluded references and 0 duplicate `(split,task_id)` entries; current fixtures cover 191/5346 setup x eval combinations and 193/5464 setup x eval x postconfig combinations; report written to `.exps/validate/lite.scalecua/oracle/coverage/current.coverage.json` and `.md` [sweep=coverage-inventory-20260714 type=coverage_gap severity=blocked]

oracle_all_matrix_candidates_current: `select_fixtures.py --coverage-target all-eval-setup` generated 5346 candidates covering 5346/5346 setup x eval combinations, and `--coverage-target all-full` generated 5464 candidates covering 5464/5464 setup x eval x postconfig combinations; candidate paths are `.exps/validate/lite.scalecua/oracle/candidates/all_eval_setup.candidates.current.jsonl` and `.exps/validate/lite.scalecua/oracle/candidates/all_full.candidates.current.jsonl` [sweep=coverage-inventory-20260714 type=coverage_gap severity=blocked]

oracle_action_template_coverage_current: sidecar planning clustered all 5464 full-matrix candidates by domain/setup/eval/postconfig and defined 12 oracle action templates; current 303 fixtures are all `execute`, so `bulk_seed_06+` must add template-driven execute rows plus app-automation/trajectory probes instead of only easy config writes. Sidecar path: `.exps/validate/lite.scalecua/oracle/action_template_coverage.sidecar.md` [sweep=coverage-inventory-20260714 type=coverage_gap severity=blocked]
bulk_seed_06_static: added 26 candidate-indexed deterministic `execute` fixtures spanning Chrome prefs, VS Code JSON/keybindings, VLC `vlcrc`, GIMP raster files, and OS file/settings rows; JSONL parse/current-task/duplicate/coverage checks passed, but replay validation was not run in this sweep [sweep=bulk-seed-06-static-20260714 type=coverage_gap severity=blocked] -> added `lite/gym/envs/lite/scalecua/data/oracle/bulk_seed_06.jsonl`

oracle_coverage_inventory_after_bulk_seed_06: `coverage_inventory.py` matched 329 committed fixture rows to current runnable `train`/`rl` rows with 0 missing/excluded references and 0 duplicate `(split,task_id)` entries; current fixtures cover 213/5345 setup x eval combinations and 218/5463 setup x eval x postconfig combinations; report refreshed at `.exps/validate/lite.scalecua/oracle/coverage/current.coverage.json` and `.md` [sweep=coverage-inventory-after-bulk-seed-06-20260714 type=coverage_gap severity=blocked]

oracle_all_matrix_candidates_after_filter_refresh: after the rerun3 filter/import refresh, `select_fixtures.py --coverage-target all-eval-setup` regenerated 5345 candidates covering 5345/5345 setup x eval combinations, and `--coverage-target all-full` regenerated 5463 candidates covering 5463/5463 setup x eval x postconfig combinations; candidate/report paths remain `.exps/validate/lite.scalecua/oracle/candidates/all_eval_setup.*.current.*` and `all_full.*.current.*` [sweep=coverage-inventory-after-bulk-seed-06-20260714 type=coverage_gap severity=blocked]

oracle_rl_only_inventory_current: `coverage_inventory.py --splits rl` is now the north-star RL report; it matched 109 RL fixture rows to current runnable RL tasks, skipped 220 out-of-scope train fixture rows, found 0 fixture problems and 0 duplicate `(split,task_id)` entries, and reports 109/1919 runnable RL tasks covered, 83/1396 RL setup x eval combinations covered, and 84/1420 RL setup x eval x postconfig combinations covered. Report paths: `.exps/validate/lite.scalecua/oracle/coverage/rl.coverage.json` and `.md` [sweep=rl-coverage-inventory-20260714 type=coverage_gap severity=blocked]

oracle_coverage_inventory_after_rerun4_exact_filters: after adding 5 exact `instruction_eval_mismatch` rows and 2 exact `upstream_generated_eval_bug` rows, fresh import keeps RL runnable at 1919 and reduces train runnable to 16523. `coverage_inventory.py` now reports 329/329 committed fixtures valid, 0 problems, 0 duplicates, 213/5344 setup x eval combinations covered, and 218/5462 setup x eval x postconfig combinations covered. `select_fixtures.py --coverage-target all-eval-setup` regenerated 5344 candidates and `--coverage-target all-full` regenerated 5462 candidates. RL-only inventory is unchanged at 109/1919 tasks and 83/1396, 84/1420 combo coverage [sweep=rerun4-exact-filter-refresh-20260714 type=coverage_gap severity=blocked]

oracle_rl_full_candidate_backlog_current: `select_fixtures.py --coverage-target all-tasks --splits rl` generated `.exps/validate/lite.scalecua/oracle/candidates/rl_full.candidates.current.jsonl` with all 1919 current runnable RL tasks. The report `.exps/validate/lite.scalecua/oracle/candidates/rl_full.coverage.current.json` covers 1396/1396 RL setup x eval combinations and 1420/1420 RL setup x eval x postconfig combinations. These are candidates only; they do not count as oracle fixtures until populated with actions/trajectories and replayed [sweep=rl-full-candidates-20260714 type=coverage_gap severity=blocked]

oracle_rl_seed_01_integration_intermediate: after integrating `rl_calc_seed_01`,
deduplicated `rl_writer_seed_01`, deduplicated `rl_thunderbird_seed_01`, and
`rl_vscode_seed_01`, `coverage_inventory.py` reports 386/386 committed
`train`/`rl` fixtures valid, 0 fixture problems, 0 duplicate `(split,task_id)`
entries, 258/5344 setup x eval combinations covered, and 263/5462 setup x eval
x postconfig combinations covered. The RL-only report now covers 166/1919
runnable RL tasks, 128/1396 RL setup x eval combinations, and 129/1420 RL full
setup x eval x postconfig combinations. `devs/envs/lite.scalecua/validate/static.py`
passed. This is an intermediate snapshot superseded by
`oracle_rl_seed_01_gimp_os_multi_apps_calc02_integration_current`
[sweep=rl-seed-01-integration-20260714 type=coverage_gap severity=blocked]

oracle_candidate_backlog_after_gimp_exact_filters: after adding 4 exact GIMP
`upstream_generated_eval_bug` filters, fresh import keeps RL runnable at 1919
and reduces train runnable to 16518 with `upstream_generated_eval_bug=47`.
`select_fixtures.py --coverage-target all-eval-setup` regenerated 5344/5344
candidates, `--coverage-target all-full` regenerated 5462/5462 candidates, and
`--coverage-target all-tasks --splits rl` regenerated the 1919/1919 RL-full
candidate backlog [sweep=gimp-exact-filters-20260714 type=coverage_gap severity=blocked]

oracle_rl_seed_01_chrome_vlc_impress_integration_superseded: after integrating
`rl_chrome_seed_01`, `rl_vlc_seed_01`, and `rl_impress_seed_01`,
`coverage_inventory.py` and `devs/envs/lite.scalecua/validate/static.py`
passed. This snapshot is superseded by the next integration entry, which is the
current oracle coverage source of truth.
[sweep=rl-seed-01-chrome-vlc-impress-integration-20260714 type=coverage_gap severity=blocked]

oracle_rl_seed_01_gimp_os_multi_apps_calc02_integration_superseded_by_domain_refresh: after
integrating `rl_gimp_seed_01`, `rl_os_seed_01`, `rl_multi_apps_seed_01`, and
`rl_calc_seed_02`, `coverage_inventory.py` reports 509/509 committed
`train`/`rl` fixtures valid, 0 fixture problems, 0 duplicate
`(split,task_id)` entries, 359/5344 setup x eval combinations covered, and
368/5462 setup x eval x postconfig combinations covered. The RL-only report
now covers 289/1919 runnable RL tasks, 229/1396 RL setup x eval combinations,
and 234/1420 RL full setup x eval x postconfig combinations. This snapshot is
superseded by the canonical `metadata.others.domain` refresh.
[sweep=rl-seed-01-gimp-os-multi-apps-calc02-integration-20260714 type=coverage_gap severity=blocked]

oracle_canonical_domain_refresh_current: after mapping generated/RL
`metadata.others.domain` from the canonical OSWorld example domain while
keeping stable source-domain task ids, fresh import reports canonical train
domain counts `chrome=2143`, `gimp=1092`, `libreoffice_calc=2705`,
`libreoffice_impress=2704`, `libreoffice_writer=1485`, `multi_apps=5250`,
`os=1479`, `thunderbird=827`, `vlc=1138`, `vs_code=1466`; canonical RL domain
counts `chrome=246`, `gimp=175`, `libreoffice_calc=250`,
`libreoffice_impress=250`, `libreoffice_writer=172`, `multi_apps=380`,
`os=180`, `thunderbird=117`, `vlc=135`, `vs_code=144`. `coverage_inventory.py`
now reports 509/509 committed `train`/`rl` fixtures valid, 0 fixture problems,
0 duplicate `(split,task_id)` entries, 351/5343 setup x eval combinations
covered, and 369/5475 setup x eval x postconfig combinations covered. The
RL-only report covers 289/1919 runnable RL tasks, 226/1396 RL setup x eval
combinations, and 235/1424 RL full setup x eval x postconfig combinations.
Candidate backlogs were regenerated: all-eval-setup 5343/5343, all-full
5475/5475, and RL-full 1919 tasks with 1424/1424 full combos.
[sweep=canonical-domain-refresh-20260714 type=coverage_gap severity=blocked]

oracle_rl_chrome_seed_02_and_filter_followup_superseded_by_url_seed: integrated
`src/gen/oracle/rl_chrome_seed_02.py` plus generated
`data/oracle/rl_chrome_seed_02.jsonl`, adding 18 runnable RL Chrome fixtures
from Python source. A later visual/code follow-up exact-filtered
`scalecua_osworld_train_libreoffice_calc_f9584479_3d0d_4c79_affa_9ad7afdd8850_task_verify_12`
as `upstream_generated_eval_bug` and fixed the local
`check_pptx_shape_text__8b4cb395` expected-rule adapter. Fresh import now
reports `train=16407`, `rl=1916`, `eval=321`, `eval_full=321`, with
`upstream_generated_eval_bug=60`. `coverage_inventory.py` now reports 563/563
committed `train`/`rl` fixtures valid, 0 fixture problems, 0 duplicates,
390/5325 setup x eval combinations covered, and 409/5455 setup x eval x
postconfig combinations covered. RL-only inventory covers 346/1916 runnable
RL tasks, 265/1395 setup x eval combinations, and 275/1423 full combinations.
Candidate backlogs were regenerated: all-eval-setup 5325/5325, all-full
5455/5455, and RL-full 1916 tasks with 1423/1423 full combos.
[sweep=rl-chrome-seed-02-filter-followup-20260714 type=coverage_gap severity=blocked]

oracle_rl_multi_apps_url_seed_01_current: integrated
`src/gen/oracle/rl_multi_apps_url_seed_01.py` plus generated
`data/oracle/rl_multi_apps_url_seed_01.jsonl`, adding 15 runnable RL URL
fixtures from Python source: 9 canonical `multi_apps` rows and 6 canonical
`chrome` rows. The generated-oracle byte-lock now covers
`rl_browser_desktop_seed_02`, `rl_chrome_seed_02`, and
`rl_multi_apps_url_seed_01`. Current import remains `train=16407`, `rl=1916`,
`eval=321`, `eval_full=321`, with `upstream_generated_eval_bug=60`.
`coverage_inventory.py` now reports 578/578 committed `train`/`rl` fixtures
valid, 0 fixture problems, 0 duplicates, 396/5325 setup x eval combinations
covered, and 415/5455 setup x eval x postconfig combinations covered. RL-only
inventory covers 361/1916 runnable RL tasks, 271/1395 setup x eval
combinations, and 281/1423 full combinations. Candidate backlogs were
regenerated: all-eval-setup 5325/5325, all-full 5455/5455, and RL-full 1916
tasks with 1423/1423 full combos.
[sweep=rl-multi-apps-url-seed-01-20260714 type=coverage_gap severity=blocked]

oracle_rl_vscode_seed_02_current: integrated
`src/gen/oracle/rl_vscode_seed_02.py` plus generated
`data/oracle/rl_vscode_seed_02.jsonl`, adding 14 runnable RL VS Code fixtures
from Python source. The generated-oracle byte-lock now covers
`rl_browser_desktop_seed_02`, `rl_chrome_seed_02`,
`rl_multi_apps_url_seed_01`, and `rl_vscode_seed_02`. Current import remains
`train=16407`, `rl=1916`, `eval=321`, `eval_full=321`, with
`upstream_generated_eval_bug=60`. `coverage_inventory.py` now reports 592/592
committed `train`/`rl` fixtures valid, 0 fixture problems, 0 duplicates,
402/5325 setup x eval combinations covered, and 421/5455 setup x eval x
postconfig combinations covered. RL-only inventory covers 375/1916 runnable
RL tasks, 277/1395 setup x eval combinations, and 287/1423 full combinations.
[sweep=rl-vscode-seed-02-20260714 type=coverage_gap severity=blocked]

oracle_thunderbird_nested_folder_0089_verified: targeted replay for
`oracle_rl_auto_gimp_tb_b1_0089`
(`scalecua_osworld_rl_thunderbird_a10b69e1_6034_4a2b_93e1_571d45194f75_traj_verify_1`)
now passes after the generated Thunderbird Local Folders oracle writes both the
parent folder and nested child path (`Bills`, `Bills/Receipts`) instead of a
flat `Receipts` folder. Negative no-op returned reward `0.0`; oracle replay
returned reward `1.0` with `check_list=1.0`. Artifacts:
`.exps/validate/lite.scalecua/oracle/thunderbird-nested-folder-0089-20260715/`
and report
`.exps/validate/lite.scalecua/oracle/thunderbird-nested-folder-0089-20260715.report.jsonl`.
No matching direct-mode containers remained after the replay.
[sweep=thunderbird-nested-folder-0089-20260715 type=oracle_replay severity=resolved]

oracle_thunderbird_folder_slice_0088_0093_verified: replayed the
Thunderbird Local Folders slice
`oracle_rl_auto_gimp_tb_b1_0088` through `oracle_rl_auto_gimp_tb_b1_0093`.
All six fixtures passed: no-op precheck reward `0.0`, oracle replay reward
`1.0`, and final summary `6/6 passed, 0 failed`. This covers flat local
folders, nested `.sbd` folders (`Bills/Receipts`, `PROJECTS/{Active,Archived}`),
multi-folder checks, and the hard-coded `COMPANY/PROJECTS` nested rule. The run
used direct mode with `--concurrency 3`; Docker cleanup emitted timeout warnings
for a few container removals, final sweep removed one leftover, and a post-run
Docker audit found 0 matching containers. Artifacts:
`.exps/validate/lite.scalecua/oracle/thunderbird-folder-slice-0088-0093-20260715/`
and report
`.exps/validate/lite.scalecua/oracle/thunderbird-folder-slice-0088-0093-20260715.report.jsonl`.
[sweep=thunderbird-folder-slice-0088-0093-20260715 type=oracle_replay severity=resolved]

oracle_office_b2_trimmed_pptx_0044_verified: fixed and replayed the Office B2
fake-artifact case `oracle_rl_auto_office_b2_0044`
(`scalecua_osworld_rl_libreoffice_impress_a097acff_6266_4291_9fbd_137af7ecd439_traj_verify_2`).
The task asks to delete the first two slides from
`/home/user/Downloads/Secrets-of-Monetizing-Video.pptx` and save
`/home/user/Desktop/trimmed.pptx`; the oracle now copies the source deck,
removes the first two slides, normalizes to the expected slide count, and saves
the target instead of creating a fresh blank presentation. `rl_auto_office_b2`
generator byte-lock passed, the generated-shard pytest passed, no-op precheck
returned reward `0.0`, oracle replay returned reward `1.0` with
`check_slide_count__3e661325ce185501287a57bdd6c09b52=1.0`, and a post-run
Docker audit found 0 matching containers. Artifacts:
`.exps/validate/lite.scalecua/oracle/office-b2-trimmed-pptx-0044-20260715/`
and report
`.exps/validate/lite.scalecua/oracle/office-b2-trimmed-pptx-0044-20260715.report.jsonl`.
[sweep=office-b2-trimmed-pptx-0044-20260715 type=fixture_bug severity=resolved]

oracle_rl_auto_multi_apps_b1_expansion_current: extended
`rl_auto_multi_apps_b1.py` and regenerated
`data/oracle/rl_auto_multi_apps_b1.jsonl` from 69 to 92 source-backed rows,
adding 23 RL `multi_apps` oracle fixtures for evaluator-backed text, CSV,
XLSX, image/file-info, `chrome://`, and search-query URL cases. The shard
byte-lock passed with `uv run --no-sync python -m
lite.gym.envs.lite.scalecua.src.gen.oracle --shard rl_auto_multi_apps_b1
--check`, and the generated-shard pytest passed under the project `uv`
environment. Current inventory now reports 1,096 / 1,910 runnable RL fixture
tasks, 679 / 1,394 RL setup x eval combinations, 705 / 1,422 RL setup x eval x
postconfig combinations, 0 fixture problems, and 0 duplicate fixture tasks.
[sweep=rl-auto-multi-apps-b1-expansion-current type=coverage_gap severity=blocked]

oracle_office_b2_shape_index_0045_verified: fixed and replayed
`oracle_rl_auto_office_b2_0045`
(`scalecua_osworld_rl_libreoffice_impress_a434992a_89df_4577_925c_0c58b747f0f4_traj_verify_1`).
The replay previously returned reward `0.5` because the generated oracle used
the semantic `content_shape()` helper, whose `python-pptx` wrapper identity
comparison could select the title shape and set `content_alignment` on the
wrong shape. The upstream getter for
`impress_content_align_title_color__63833ad040c5e27e84bd66675fcf45fe` reads
`slide.shapes[0]` for title color and `slide.shapes[1]` for content alignment,
so the generator now emits a family-specific `shape_color(shape=0)` plus
`shape_alignment(shape=1)` instead of changing the generic helper. Targeted
replay returned no-op reward `0.0` and oracle reward `1.0`; the 0040-0046
Office B2 PPTX slice then passed `7/7` with every row at no-op `0.0` and oracle
`1.0`. Artifact roots:
`.exps/validate/lite.scalecua/oracle/office-b2-0045-shape-index-fix-20260715/`
and
`.exps/validate/lite.scalecua/oracle/office-b2-pptx-slice-0040-0046-shape-index-fix-20260715/`.
Post-run Docker audits found 0 matching direct-mode containers.
[sweep=office-b2-shape-index-0045-20260715 type=fixture_bug severity=resolved]

oracle_rl_auto_vscode_files_b1_current: registered
`rl_auto_vscode_files_b1.py` in the source-backed generator and generated
`data/oracle/rl_auto_vscode_files_b1.jsonl` with 15 runnable RL VS Code/file
fixtures. The shard check and generated-shard pytest passed, and fresh
`coverage_inventory.py` reports 1,111 / 1,910 runnable RL fixture tasks,
691 / 1,394 RL setup x eval combinations, 717 / 1,422 RL setup x eval x
postconfig combinations, 0 fixture problems, and 0 duplicate fixture tasks.
Train+RL inventory reports 1,328 runnable fixture tasks, 816 / 5,323 setup x
eval combinations, and 851 / 5,453 setup x eval x postconfig combinations.
[sweep=rl-auto-vscode-files-b1-current type=coverage_gap severity=blocked]

oracle_rl_auto_multi_apps_b1_0074_0080_fixes: runtime replay of the 23-row
`rl_auto_multi_apps_b1` expansion initially passed 21/23. The two failures
were both migration/oracle bugs, not upstream defects. Fixture 0074 failed in
setup because the official ScaleCUA task uses bare `update-desktop-database`
and the official server tolerates its non-zero returncode; the importer now
normalizes that exact command to a user-local desktop database refresh, keeping
strict setup semantics elsewhere. Targeted replay for
`oracle_rl_auto_multi_apps_b1_0074` then passed with no-op reward `0.0` and
oracle reward `1.0`. Fixture 0080 failed because the generated oracle launched
Chrome directly with `chrome://extensions`, which left Chrome on a new-tab/crash
restore state; the generator now uses a self-contained internal-URL recipe:
launch `about:blank` with CDP, then open the internal URL through
`chrome_open_tabs`. Targeted replay for `oracle_rl_auto_multi_apps_b1_0080`
passed with actual URL `chrome://extensions/`, and the same helper was
regression-checked on `oracle_rl_auto_multi_apps_b1_0007` and `_0021`, both
passing for `chrome://settings`. Validation artifacts:
`.exps/validate/lite.scalecua/oracle/rl-auto-multi-apps-b1-0070-0092-20260715/`,
`.exps/validate/lite.scalecua/oracle/rl-auto-multi-apps-b1-0074-0080-fixes-20260715/`,
`.exps/validate/lite.scalecua/oracle/rl-auto-multi-apps-b1-0080-internal-url-fix-20260715/`,
and
`.exps/validate/lite.scalecua/oracle/rl-auto-multi-apps-b1-internal-url-regression-20260715/`.
Generator `--check`, the generated-shard pytest, full
`tests/gym/envs/lite/test_scalecua.py`, and post-run Docker orphan audit passed.
[sweep=rl-auto-multi-apps-b1-0074-0080-fixes-20260715 type=fixture_bug severity=resolved]

oracle_rl_auto_os_files_b2_current: promoted the OS/files B2 source-backed
generator as `rl_auto_os_files_b2.py` and regenerated
`data/oracle/rl_auto_os_files_b2.jsonl` with 44 runnable RL fixtures. The raw
candidate excluded three unsafe `vm_terminal_output` rows whose typed terminal
oracle could land in Files search instead of Terminal, and two VS Code
extension rows whose fake `~/.vscode/extensions/...` directory did not satisfy
`code --list-extensions`. Generator `--check` passed, static validation passed,
and replay canary
`.exps/validate/lite.scalecua/oracle/rl-auto-os-files-b2-canary-final-20260715/`
passed `4/4` with no-op reward `0.0` and oracle reward `1.0`.
[sweep=rl-auto-os-files-b2-current-20260715 type=fixture_bug severity=resolved]

oracle_rl_auto_thunderbird_source_b1_current: promoted the Thunderbird
source-backed generator as `rl_auto_thunderbird_source_b1.py` and regenerated
`data/oracle/rl_auto_thunderbird_source_b1.jsonl` with 16 runnable RL fixtures.
Three account/filter candidates remain skipped because the official-style
getter path can false-negative after deletion (`grep -ic ... || echo 0` can
produce ambiguous `0\n0` output). Generator `--check` passed, static validation
passed, and replay canary
`.exps/validate/lite.scalecua/oracle/rl-auto-thunderbird-source-b1-canary-20260715/`
passed `4/4` with no-op reward `0.0` and oracle reward `1.0`.
[sweep=rl-auto-thunderbird-source-b1-current-20260715 type=fixture_bug severity=resolved]

oracle_coverage_inventory_after_os_files_b2_thunderbird_source: after
registering `rl_auto_os_files_b2` and `rl_auto_thunderbird_source_b1`, the
RL-only coverage report matches 1,171 / 1,910 runnable RL fixtures, 727 /
1,394 RL setup x eval combinations, and 754 / 1,422 RL setup x eval x
postconfig combinations, with 0 fixture problems and 0 duplicate fixture
tasks. The train+RL report matches 1,388 runnable fixtures, 852 / 5,323 setup
x eval combinations, and 888 / 5,453 setup x eval x postconfig combinations.
`uv run --no-sync python -m lite.gym.envs.lite.scalecua.src.gen.oracle --check`,
`uv run --no-sync python devs/envs/lite.scalecua/validate/static.py`, and
`uv run --no-sync pytest -q tests/gym/envs/lite/test_scalecua.py` all passed
(`115 passed` for pytest), and a post-run Docker audit found 0 matching
`lite-env-lite_scalecua_oracle_validate` containers.
[sweep=coverage-inventory-after-os-files-b2-thunderbird-source-20260715 type=coverage_gap severity=blocked]

oracle_rl_auto_gimp_b2_current: registered `rl_auto_gimp_b2.py` and generated
`data/oracle/rl_auto_gimp_b2.jsonl` with 9 runnable RL GIMP fixtures covering
background color export, brightness, image dimensions, and layer-bbox export
families. The shard is source-backed and byte-locked by `uv run --no-sync
python -m lite.gym.envs.lite.scalecua.src.gen.oracle --shard
rl_auto_gimp_b2 --check`. Full production replay under
`.exps/validate/lite.scalecua/oracle/rl-auto-gimp-b2-full-20260715/` passed
`9/9` with no-op reward `0.0`, oracle reward `1.0`, screenshots, traces, and
eval debug payloads. Skipped families remain explicit in source comments:
`video_duration` is not a GIMP desktop capability, `gimp_xcf_layers` needs a
validated GIMP batch/XCF recipe, and evaluator sidecar text files are not
user-visible artifacts.
[sweep=rl-auto-gimp-b2-current-20260715 type=oracle_pass severity=ok]

oracle_rl_auto_office_b3_current: registered `rl_auto_office_b3.py` and
generated `data/oracle/rl_auto_office_b3.jsonl` with 33 runnable RL Office and
Office-adjacent fixtures (`multi_apps=16`, `libreoffice_impress=12`,
`libreoffice_calc=3`, `libreoffice_writer=2`). The shard is source-backed and
byte-locked by `uv run --no-sync python -m
lite.gym.envs.lite.scalecua.src.gen.oracle --shard rl_auto_office_b3 --check`.
Full production replay under
`.exps/validate/lite.scalecua/oracle/rl-auto-office-b3-full-20260715/` passed
`33/33` with no-op reward `0.0`, oracle reward `1.0`, screenshots, traces, and
eval debug payloads. The run initially exposed picture-size/index issues in
representative canaries; those were fixed before the full replay by parsing
real Desktop image names from instructions and using official getter slide
indices for picture/font checks.
[sweep=rl-auto-office-b3-current-20260715 type=oracle_pass severity=ok]

oracle_rl_auto_chrome_b2_candidate_not_counted: the unpromoted Chrome B2
candidate remained unregistered after a 10-row canary passed 8/10 but left
TripAdvisor local-page rows unverified. The unverified source and JSONL were
kept out of `data/oracle/` and out of the release generator registry so
`coverage_inventory.py` cannot overstate fixture coverage.
Current blocker: `active_tab_html_parse` with `goto_prefix=https://www.` does
not yet have a proven local-page substitute for `http://127.0.0.1...` CDP URLs.
[sweep=rl-auto-chrome-b2-candidate-20260715 type=coverage_gap severity=blocked]

oracle_coverage_inventory_after_gimp_b2_office_b3: after registering verified
`rl_auto_gimp_b2` and `rl_auto_office_b3`, and removing unverified Chrome B2
JSONL from `data/oracle/`, the RL-only coverage report matches 1,213 / 1,910
runnable RL fixture rows, 748 / 1,394 RL setup x eval combinations, and 775 /
1,422 RL setup x eval x postconfig combinations, with 0 fixture problems and 0
duplicate fixture tasks. The train+RL report matches 1,430 runnable fixtures,
873 / 5,323 setup x eval combinations, and 909 / 5,453 setup x eval x
postconfig combinations. These are fixture inventory counts, not verified
oracle counts; final closure still requires full replay evidence for every
counted RL oracle row. Static validation and full
`tests/gym/envs/lite/test_scalecua.py` passed (`115 passed`).
[sweep=coverage-inventory-after-gimp-b2-office-b3-20260715 type=coverage_gap severity=blocked]

bulk_seed_06_full_replay_current: full production replay of
`lite/gym/envs/lite/scalecua/data/oracle/bulk_seed_06.jsonl` passed `26/26`
under `.exps/validate/lite.scalecua/oracle/bulk-seed-06-full-20260715/`, with
no-op negative prechecks and oracle-positive replays for Chrome preference,
VS Code JSON/keybinding, VLC `vlcrc`, GIMP raster, and OS file/settings rows.
The shard contains 21 RL rows and 5 train rows; refreshed inventories report
RL strict verified evidence `314 / 1,219` with 905 unverified, and train+RL
strict verified evidence `323 / 1,436` with 1,113 unverified. Generator
byte-lock, `devs/envs/lite.scalecua/validate/static.py`, and
`tests/gym/envs/lite/test_scalecua.py` also passed in the same sweep.
[sweep=bulk-seed-06-full-20260715 type=oracle_pass severity=ok]

rl_vscode_seed_02_full_replay_superseded_by_chrome_seed_01_refresh: full production replay of
`lite/gym/envs/lite/scalecua/data/oracle/rl_vscode_seed_02.jsonl` passed
`14/14` under
`.exps/validate/lite.scalecua/oracle/rl-vscode-seed-02-full-20260715/`.
No-op warnings for missing VS Code `keybindings.json` and `argv.json` were
expected negative-state evidence; replay rewards were all `1.0`. Refreshed
inventories report RL strict verified evidence `328 / 1,219` with 891
unverified, and train+RL strict verified evidence `337 / 1,436` with 1,099
unverified.
[sweep=rl-vscode-seed-02-full-20260715 type=oracle_pass severity=ok] -> superseded by `rl_chrome_seed_01_broad_profile_find_fix_current`

rl_chrome_seed_01_broad_profile_find_fix_current: full production replay of
`lite/gym/envs/lite/scalecua/data/oracle/rl_chrome_seed_01.jsonl` initially
passed `15/16`; the failed row `oracle_rl_chrome_seed_01_0016` used the
official generated `font_and_dnt` getter, whose broad
`find /root /home -name "Preferences" -path "*/google-chrome/Default/*" | head
-1` profile discovery can miss lite.osworld's launched
`/home/user/chrome-data` profile. This was a ScaleCUA overlay compatibility
bug, not an upstream exclude: `lite.scalecua/src/osworld/judges.py` now aliases
that broad discovery command before `run_bash_script`/shell execution, focused
Chrome profile alias tests pass, targeted `oracle_rl_chrome_seed_01_0016`
replay passed with no-op reward `0.0` and oracle reward `1.0`, and full shard
replay under
`.exps/validate/lite.scalecua/oracle/rl-chrome-seed-01-full-shim-fix-20260715/`
passed `16/16`. Refreshed inventories report RL strict verified evidence
`351 / 1,219` with 868 unverified, and train+RL strict verified evidence
`360 / 1,436` with 1,076 unverified. Direct-mode post-run Docker audit found
0 matching scalecua containers.
[sweep=rl-chrome-seed-01-full-shim-fix-20260715 type=eval_bug severity=ok] -> fixed in `lite.scalecua/src/osworld/judges.py`

rl_fixture_replay_gate_current: targeted replay closed the remaining current RL
fixture rows. Initial final-fixes replay passed Writer title alignment,
Thunderbird filter+pref, dual-filter, and SQLite starred-message fixtures
(`4/6`) under
`.exps/validate/lite.scalecua/oracle/final-fixes-20260715.report.jsonl`.
The two remaining Thunderbird rows exposed fixture-location bugs: generated
getters use `xargs` over paths containing `Local Folders`, so the fixture must
also write account-visible files under
`ImapMail/outlook.office365.com/`. After updating
`rl_thunderbird_seed_01_10` and `rl_thunderbird_seed_01_15`, targeted replay
under
`.exps/validate/lite.scalecua/oracle/final-fixes-rerun3-20260715.report.jsonl`
passed `2/2`. `oracle_rl_writer_seed_01_0012` was removed from the fixture
path and exact-excluded because the official generated tab-stop metric accepts
only literal `RIGHT`/`END` while the getter returns `RIGHT (2)` for a correct
UNO right tab stop. Refreshed `verified_inventory.py --splits rl
--require-complete` now passes with `1213 / 1213` RL fixture rows verified and
`0` unverified fixture rows; full catalog closure remains open at `1213 / 1897`
runnable RL catalog rows.
[sweep=rl-fixture-replay-gate-current-20260715 type=oracle_fix severity=ok]

rl_auto_thunderbird_tail_b11_7b1e1ff9_troubleshooting_v6: probe evidence exists
under
`.exps/validate/lite.scalecua/oracle/thunderbird_tail_probe2_20260716/report.jsonl`
with no-op precheck reward `0.0` and replay reward `1.0`, but formal targeted
validation still failed before replay with
`post-boot setup transient (TimeoutError) (retry_after_s=30)` under both
`.exps/validate/lite.scalecua/oracle/rl_auto_thunderbird_tail_b11_v6_final_20260716/report.jsonl`
and
`.exps/validate/lite.scalecua/oracle/rl_auto_thunderbird_tail_b11_v6_final_retry2_20260716/report.jsonl`
using `cua-lite/lite.osworld:scalecua-20260715-fixedbase`, unique session ids,
concurrency `1`, and `--reset-timeout 900`. The failure is a validator/runtime
setup flake before oracle actions, likely the Thunderbird profile download
exceeding the exec-stdio per-call timeout rather than an upstream eval bug; do
not exact-exclude and do not formalize until a strict formal pass exists.
[sweep=rl-auto-thunderbird-tail-b11-v6-20260716 type=setup_bug severity=blocked]

rl_auto_thunderbird_tail_b11_7b1e1ff9_troubleshooting_v6_timeoutfix: narrowed the
formal validation failure from exec-stdio host timeout to the underlying setup
download. `lite/gym/envs/lite/osworld/src/utils/dispatch.py` now passes a
420s timeout to OSWorld `download` actions, covering the existing five
`wget --timeout=60` attempts plus backoff, and it can use an exact host-side
`xlangai/ubuntu_osworld_file_cache` asset fallback when present. The first
retry progressed past the old timeout and exposed the real transient: HF
returned no response headers, so setup could not create
`/home/user/Desktop/thunderbird-profile.tar.gz`. After `hf_hub_download`
eventually populated the exact
`.cache/hf_asset_probe/thunderbird/dd84e895-72fd-4023-a336-97689ded257c/thunderbird-profile.tar.gz`
blob, targeted strict validation used only
`cua-lite/lite.osworld:scalecua-20260715-fixedbase` and the single verify_6
candidate under
`.exps/validate/lite.scalecua/oracle/rl_auto_thunderbird_tail_b11_v6_timeoutfix_20260716/candidate.jsonl`.
Formal PASS evidence is under
`.exps/validate/lite.scalecua/oracle/rl_auto_thunderbird_tail_b11_v6_timeoutfix_20260716/report.jsonl`
and
`.exps/validate/lite.scalecua/oracle/rl_auto_thunderbird_tail_b11_v6_timeoutfix_20260716/run/oracle_rl_auto_thunderbird_tail_b11_v6_timeoutfix_0001/result.json`.
After registering as `oracle_rl_auto_thunderbird_tail_b11_0011`, the registered
row also passed strict validation under
`.exps/validate/lite.scalecua/oracle/rl_auto_thunderbird_tail_b11_v6_registered_20260716/report.jsonl`
and
`.exps/validate/lite.scalecua/oracle/rl_auto_thunderbird_tail_b11_v6_registered_20260716/run/oracle_rl_auto_thunderbird_tail_b11_0011/result.json`.
Strict gate: no-op precheck reward `0.0`, replay reward `1.0`, `result.json`
present, and screenshots `00_noop_reset.png`, `01_noop_final.png`,
`10_oracle_reset.png`, `11_oracle_after_actions.png` present. Registered as
`oracle_rl_auto_thunderbird_tail_b11_0011`; no exact exclude.
[sweep=rl-auto-thunderbird-tail-b11-v6-timeoutfix-20260716 type=setup_bug severity=ok]

train_current_clean_20260822: strict train evidence passed with
`.exps/validate/lite.scalecua/oracle/oracle-scalecua-train-current-clean-20260822-0558.report.jsonl`.
This is a 200-row oracle fixture smoke/subset, not full train-catalog coverage.
Verified inventory output
`.exps/validate/lite.scalecua/oracle/coverage/train.current-clean-single.verified.json`
reported `current_fixture_rows=200`, `strict_valid_report_rows=200`,
`verified_fixture_rows=200`, `duplicate_strict_valid_report_rows=0`,
`stale_report_rows=0`, `artifact_missing_strict_valid_report_rows=0`,
`catalog_runnable_rows=16139`, `verified_catalog_rows=200`,
`unverified_catalog_rows=15939`, and `report_problem_counts={}`.
[sweep=train-current-clean-20260822 type=oracle_pass severity=ok]

rl_current_full_20260822: full RL evidence passed with
`.exps/validate/lite.scalecua/oracle/oracle-scalecua-rl-20260822-033036.report.jsonl`.
Verified inventory output
`.exps/validate/lite.scalecua/oracle/coverage/rl.current.verified.json`
reported `current_fixture_rows=1809`, `strict_valid_report_rows=1809`,
`verified_fixture_rows=1809`, `catalog_runnable_rows=1809`,
`verified_catalog_rows=1809`, `unverified_catalog_rows=0`,
`duplicate_strict_valid_report_rows=0`, `stale_report_rows=0`,
`artifact_missing_strict_valid_report_rows=0`, and
`report_problem_counts={}`. The validator's `--require-rl-flush-fired` gate was
satisfied with RL flush evidence for both VLC and Thunderbird.
[sweep=rl-current-full-20260822 type=oracle_pass severity=ok]
