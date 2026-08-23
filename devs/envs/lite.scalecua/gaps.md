# lite.scalecua Runtime And Migration Gaps

This file primarily tracks gaps that are not confirmed SCALE-CUA upstream task
defects. Confirmed official generated instruction/eval bugs still belong in
`UPSTREAM_ISSUES.md` and should be filtered by exact
`metadata.others.exclude_reason`, but confirmed eval bugs found during rollout
triage are mirrored here when they affect the current validation lane. Gaps
here are adapter, container substrate, GUI persistence, asset materialization,
or validation-process issues that need a local fix, a targeted probe, or an
explicit unsupported policy.

## Classification Rules

- `container_runtime`: the official ScaleCUA/OSWorld VM assumes QEMU/systemd or
  VM services that the `lite.osworld` Docker substrate does not provide.
- `migration_adapter`: the task state is correct, but the lite.scalecua
  evaluator/setup adapter reads the wrong path, shape, service, or result type.
- `persistence_flush`: the GUI visually shows the requested state, but the
  evaluator reads an app config/file after close/save/relaunch and the state may
  not have flushed.
- `asset_materialization`: official generated code references author-local or
  VM-local paths that must be transported into the eval cache.
- `needs_probe`: screenshots are insufficient; rerun with file/getter/debug
  probes before deciding filter vs adapter vs true agent failure.
- `upstream_task_bug`: do not record as a gap after confirmation. Move it to
  `UPSTREAM_ISSUES.md` and add an exact exclusion if it is unsafe to patch.

## Confirmed During Current Validation Lane

### 2026-07-15 rollout false-negative generated judge fixes

- Type: `migration_adapter` / generated judge compatibility, fixed locally in
  `lite.scalecua` rather than filtered.
- Evidence root:
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-fixedbase-20260715`.
- Fixed rows/families:
  - `scalecua_osworld_train_libreoffice_writer_e246f6d8_78d7_44ac_b668_fcf47946cb50_task_verify_2`
    visually and script-verified all italic runs as red, but generated getter
    `get_docx_italic_color_info__96c1a3698fda406b269dcc52d1b2d9c9`
    called `RGBColor.hex()` on python-docx `RGBColor`. Resolution: narrow
    getter override converts `RGBColor`/OOXML color values to `RRGGBB`.
  - `scalecua_osworld_train_libreoffice_writer_adf5e2c3_64c7_4644_b7b6_d2f0167927e7_task_verify_32`
    final screenshot showed the target thesis sentence highlighted yellow, but
    generated getter
    `get_text_highlighting__1f6188764666903805a2bbde08d45ff2` only accepted
    python-docx highlight values and missed LibreOffice OOXML shading. Resolution:
    narrow getter override accepts `w:highlight` and yellow `w:shd`.
  - `scalecua_osworld_train_libreoffice_calc_3680a5ee_6870_426a_a997_eba929a0d25c_task_verify_12`
    final screenshot showed `merged_data.csv` with 10000 data rows, but
    generated getter
    `get_csv_merge_data__c66369e707b97de3ccd6da4699663fe6` referenced missing
    `_extract_unique_values_from_xlsx/_ods` helpers. Resolution: narrow getter
    override parses the merged CSV and treats source-file coverage as
    best-effort.
  - `scalecua_osworld_train_libreoffice_calc_c7c1e4c3_9e92_4eba_a4b8_689953975ea4_task_verify_33`
    final screenshot showed `G2=Initials`, `G3=QL`, `G4=TY`, `G5=LK`, but
    generated expected rows were off by one (`2/3/4`) even though row 2 is the
    header. Resolution: narrow metric wrapper accepts the one-row offset only
    when `G2` is exactly `Initials`.
- Guardrail: these are not global LibreOffice relaxations. They are keyed by
  exact generated function names and covered by focused unit tests in
  `tests/gym/envs/lite/scalecua/test_scalecua_generated_judge_repairs.py`:
  `test_scalecua_generated_docx_italic_color_getter_accepts_rgbcolor`,
  `test_scalecua_generated_docx_highlight_getter_accepts_ooxml_shading`,
  `test_scalecua_generated_csv_merge_getter_tolerates_missing_source_helpers`,
  and `test_scalecua_generated_calc_initials_metric_accepts_header_row_offset`.

### 2026-07-16 generated helper/import omissions

- Type: `migration_adapter`, fixed locally in `lite.scalecua`.
- Evidence root:
  `.exps/validate/lite.scalecua/train/gpt-5.5-train1000-filtered-profilepath-20260716`.
- Fixed symptoms:
  - generated GIMP saturation metrics called missing `measure_saturation`;
  - generated file/VLC getter
    `get_file_exists__3b8e423e430323c0078f4425aded05b9` called missing
    `_get_video_rotation`;
  - generated getter shards imported `from xml import etree` but then used
    `lxml`-style `fromstring(...).xpath(...)`.
- Resolution: the ScaleCUA overlay loader now injects the missing metric/getter
  helpers and rewires generated getter-module `etree` globals to `lxml.etree`.
  Confirmed upstream task-contract bugs remain filtered by exact
  `exclude_reason`; this helper layer only prevents direct generated-code
  exceptions.
- Guardrail: do not move these helper shims to `lite.osworld`; they are
  ScaleCUA generated-code compatibility, not shared desktop substrate.

### RL OS notebook delete substring eval bug

- Type: `upstream_task_bug`, mirrored here because it was found by the current
  oracle replay lane.
- Affected task:
  `scalecua_osworld_rl_os_5c1075ca_bb34_46a3_a7a0_029bd7463e79_traj_verify_1`.
- Evidence:
  `.exps/validate/lite.scalecua/oracle/rl-auto-os-files-b2-unverified-full-20260715.report.jsonl`
  fixture `oracle_rl_auto_os_files_b2_0020` failed after the oracle deleted
  non-failed notebooks. Official expected rules include
  `a_failed.ipynb`, `c_failed.ipynb`, and `e_failed.ipynb`, while excluding
  `d.ipynb`.
- Root cause: official `check_include_exclude` is substring-based, so every
  required `*_failed.ipynb` include necessarily contains the excluded substring
  `d.ipynb`. Reward `1.0` is unreachable without changing official evaluator
  semantics.
- Resolution: exact `exclude_reason="upstream_generated_eval_bug"` in
  `lite.gym.envs.lite.scalecua.src.utils.dataset.UPSTREAM_GENERATED_EVAL_BUG_TASK_IDS`;
  do not keep an oracle fixture for this row.

## Base Substrate Candidacy

Only gaps that improve the OSWorld desktop substrate itself should move to
`lite.osworld` setup or the base Dockerfile. ScaleCUA generated-code
compatibility must stay inside `lite.scalecua` unless the same behavior is
proven necessary for `lite.osworld` tasks.

| Gap | Base Dockerfile candidate? | Reason |
| --- | --- | --- |
| `timedatectl` no-systemd shim | Yes, canonical in the `lite.osworld` Dockerfile | OSWorld and ScaleCUA both contain timezone tasks whose official evaluator assumes VM-shaped `timedatectl` output. A shim that reflects `/etc/timezone`, `/etc/localtime`, `date +%z`, and limited NTP state is a substrate feature, not ScaleCUA-specific generated-code repair. Runtime setup should only verify image freshness. |
| Chrome profile path alias / extension getter repair | No | The mismatch is between ScaleCUA generated getters and CUA-Lite's Chrome profile layout. It belongs in `lite.scalecua` evaluator adapter and tests. |
| GIMP action-history/sessionrc fallback | Mostly no | The visible UI vs persisted config race is evaluator/postconfig behavior. Add narrow adapter/probe fixes first; only move a generic app flush utility if lite.osworld itself shows the same drift. |
| Thunderbird prefs/profile probes | No for now | This is profile/evaluator path and persistence validation, not a Dockerfile dependency. |
| VLC `vlcrc` persistence probes | No for now | This is app write/close timing or evaluator probing. A base change is premature without proving a common VLC substrate defect. |
| LibreOffice save/close handling | No | This is postconfig dispatch and file normalization, already an evaluator/setup lifecycle concern. |
| Author-local reference materialization | No | This is a ScaleCUA generated task artifact transport issue. |
| Direct-path generated getter file materialization | No | Official generated getters can directly open `config["path"]` inside the VM process. In CUA-Lite those getters run on the host, so the VM file must be downloaded in the `lite.scalecua` judge adapter; this is not an OSWorld substrate feature. |
| Direct-mode cleanup/orphan checks | No | This belongs in validation tooling and lifecycle scripts, not the image. |

The `timedatectl` shim lives in the `lite.osworld` Dockerfile, not in
per-task setup. This keeps VM-shaped timezone support in the shared OSWorld
desktop substrate and avoids rewriting a shell script on every reset. Runtime
setup only checks that the image is fresh enough and warns if an old image is
still being used. The required bar for the image artifact is:

- support `timedatectl status` with the official line shape used by OSWorld
  metrics;
- support `timedatectl show --property=NTP --value` without claiming active
  NTP unless a local shim state says so;
- support `timedatectl set-timezone <zone>` by updating `/etc/timezone` and
  `/etc/localtime`;
- either support `timedatectl set-ntp true/false` through explicit shim state
  or mark NTP-dependent rows unsupported instead of returning false success;
- run lite.osworld eval/oracle regression checks before changing the shared
  image, because the shim affects every OSWorld-family reset.

## Current Gap Register

### G-001: non-systemd `timedatectl status`

- Type: `container_runtime`
- Scope: OS timezone generated/RL rows from OSWorld id
  `b6781586-6346-41cd-935a-a6b1487918fc`.
- Symptom: Official evaluators run `timedatectl status` and parse the VM-shaped
  output. The `lite.osworld` container can have `/etc/timezone` and
  `/etc/localtime` set correctly while `timedatectl` itself fails because the
  container is not systemd-backed.
- Evidence: rollout row
  `scalecua_osworld_train_os_b6781586_6346_41cd_935a_a6b1487918fc_task_verify_29`
  showed timezone files/date offset consistent with `Asia/Dubai`, while the
  generated result command `timedatectl status` failed.
- Decision: install a shared `timedatectl` shim in the `lite.osworld`
  Dockerfile so fresh OSWorld-family containers expose VM-shaped `status`,
  `show --property=NTP --value`, `set-timezone`, and `set-ntp` behavior.
  Keep runtime setup to a lightweight freshness check, and keep any
  lite.scalecua-local evaluator fallback only as a guard for stale diagnostic
  roots or old env servers that were started before the shared image fix.
- Guardrail: do not synthesize successful NTP/systemd service semantics. Rows
  whose reward depends on `set-ntp`, `NTP service: active`, or real systemd
  state need either a real shim with setup support or an explicit unsupported
  filter after confirmation.
- Validation: unit test the fallback shape and at least one oracle/probe row
  for UTC, Asia/Tokyo, and Asia/Dubai-style offsets.

### G-002: Chrome profile path and extension getter shape

- Type: `migration_adapter`
- Scope: Chrome generated/RL rows that read `Preferences`, `Local State`,
  extension settings, History, or Bookmarks.
- Symptom: Official generated getters assume QEMU Chrome profile paths such as
  `/home/user/.config/google-chrome/Default/Preferences`; the CUA-Lite runtime
  launches Chrome with `/home/user/chrome-data`.
- Evidence: visual failures around unpacked extension tasks showed the
  extension loaded on screen, but generated extension getters returned empty
  shapes because they read the wrong profile or returned a shape mismatched with
  the generated metric.
- Decision: keep the fix inside lite.scalecua: alias Chrome profile paths,
  flush Chrome before profile-backed getters, and repair empty generated
  extension getter results from the actual Chrome `Preferences` and manifest.
- Guardrail: do not change `lite.osworld` or shared action-space utilities for
  this ScaleCUA-specific generated getter behavior without a separate
  non-regression proof.
- Validation: unit tests for profile aliasing, flush trigger, extension
  manifest/name/version/description/count shapes, plus a fresh rollout/probe on
  extension rows.

### G-003: GIMP action-history postconfig race

- Type: `migration_adapter` / `persistence_flush`
- Scope: GIMP rows whose evaluator reads
  `/home/user/.config/GIMP/2.10/action-history` after postconfig closes GIMP.
- Symptom: screenshots show the requested GIMP filter dialog or effect, but the
  final `action-history` file may not yet contain the action token after
  postconfig.
- Evidence: visual audit rows from family
  `a746add2-cab0-4740-ac36-c3769d9bfb46` showed requested dialogs such as
  Gaussian Blur or Unsharp Mask, while old reward was 0 from the action-history
  getter.
- Decision: capture pre-postconfig window state and augment only known missing
  action-history tokens when the active window/class/title proves the matching
  GIMP dialog was open.
- Guardrail: this is a narrow fallback, not a generic GIMP success override.
  Unknown dialog names require either a new targeted token mapping or a true
  failure label.
- Validation: unit coverage for token mapping and oracle rows for multiple
  GIMP filter families.

### G-004: GIMP `sessionrc` persistence after close

- Type: `persistence_flush`
- Scope: GIMP rows whose evaluator reads `sessionrc` or `gimprc` after UI
  state changes, especially family
  `d52d6308-ec58-42b7-a2c9-de80e4837b2b`.
- Symptom: screenshots can show fullscreen/rulers/menubar-like UI state, but
  reward depends on persisted config after GIMP quit.
- Evidence: visual audit marked rows `_27`, `_29`, `_32`, and `_41` as
  container/runtime or persistence suspects; rollout
  `scalecua_osworld_rl_gimp_a746add2_cab0_4740_ac36_c3769d9bfb46_traj_verify_6`
  visually showed Dark theme while the generated getter still required
  persisted `gimprc`.
- Decision: fixed in the ScaleCUA adapter by gracefully flushing GIMP before
  `gimp_config_file`, generated `gimp_theme*`, generated `gimp_icon_theme*`,
  and direct `gimprc`/`sessionrc` command/path getters. The flush uses WM_CLASS
  `Gimp`, `ctrl+q`, the Quit GIMP dialog's `alt+d` discard path, and an exit
  wait loop borrowed from lite.osworld's validated GIMP lifecycle.
- Guardrail: this is only a persistence flush before reading config files. It
  does not override metric results or infer success from screenshots; if the
  final file is wrong after flush, the task still fails.
- Validation: targeted unit coverage for base and generated GIMP config
  getters; oracle replays
  `oracle_rl_auto_gimp_tb_b1_{0039,0046,0051}` passed under fixed image
  `cua-lite/lite.osworld:scalecua-20260715-fixedbase`.

### G-005: Thunderbird prefs/profile persistence

- Type: `persistence_flush`
- Scope: Thunderbird generated/RL rows whose evaluator reads profile files such
  as `prefs.js` after UI changes.
- Symptom: final UI can show the intended setting, while reward reads stale or
  different profile preferences.
- Evidence: visual audit rows
  `scalecua_osworld_train_thunderbird_08c73485_7c6d_4681_999d_919f5c32dcfa_task_verify_17`,
  `_21`, and `_50` were visually plausible but failed prefs-file checks.
- Decision: rerun with profile-path and prefs flush probes before changing
  filters. Keep postconfig strict and run it exactly once.
- Guardrail: Thunderbird compaction prompt rows are especially easy to
  misread: `mail.prompt_purge_threshhold=false` can correspond to the UI
  "Always ask me before compacting folders automatically" checkbox being
  checked because Thunderbird stores a negative preference. Do not add an
  upstream exclusion without verifying the generated metric and prefs semantics.
- Validation: oracle fixtures for representative Thunderbird prefs and a probe
  that records UI state, profile path, `prefs.js`, and getter result.

### G-006: VLC visible UI vs persisted `vlcrc`

- Type: `persistence_flush`
- Scope: VLC rows whose evaluator checks `vlcrc` after changing hidden or
  persisted settings.
- Symptom: screenshots show a current VLC UI state such as minimal view or
  always-on-top, but the evaluator reads persisted config after relaunch.
- Evidence: visual audit rows from Chrome/VLC mixed tasks
  `215dfd39...task_verify_37`, `386dbd0e...task_verify_3`, and
  `d06f0d4d...task_verify_2` were not decidable from screenshots alone.
- Decision: rerun with direct `vlcrc` probes. If persisted config is correct
  but the getter fails, add adapter fallback. If persisted config is wrong,
  count as agent failure or add a postconfig flush only if the UI state is
  provably correct and VLC delayed the write.
- Guardrail: existing VLC auth/status fallbacks do not automatically solve
  hidden persisted config checks.
- Validation: oracle fixtures for minimal view, always-on-top, and common VLC
  preference rows.

### G-007: LibreOffice save/close/focus drift

- Type: `migration_adapter` / `persistence_flush`
- Scope: Writer/Calc/Impress generated/RL rows with postconfig save/close and
  file-based metrics.
- Symptom: a document can appear visually edited but reward reads an old file,
  a normalized conversion artifact, or an unsaved state.
- Evidence: visual/code audits flagged save/collection staleness in several
  Calc/Impress rows; separate confirmed upstream generated metric bugs are
  tracked in `UPSTREAM_ISSUES.md`, not here.
- Decision: keep robust lite.scalecua postconfig save handling for LibreOffice:
  activate generic `LibreOffice`, run `ctrl+s`/menu save, handle `Changed by
  Others`, `Keep Current Format`, overwrite, and Save dialogs, then normalize
  LO files symmetrically before strict comparison.
- Guardrail: if final files are saved correctly and generated metrics still
  contradict task semantics, move that exact row to `UPSTREAM_ISSUES.md` and
  filter with `upstream_generated_eval_bug`.
- Validation: oracle fixtures covering Writer, Calc, and Impress save paths,
  plus rollout probes that download the result file and compare generated
  getter output against visual/file truth.

### G-008: author-local reference path materialization

- Type: `asset_materialization`
- Scope: generated metrics that open `source_path`, `original_path`, or
  `source_cache_path` strings such as
  `/home/lvbowen/project/.../cache/<id>/<file>`.
- Symptom: result collection or metric execution fails on the host because the
  generated metric references author-local paths not present in the CUA-Lite
  checkout.
- Evidence: generated task JSON and metrics pass these paths through result or
  expected rules. Some files can be recovered from VM setup locations such as
  Desktop/Downloads/Pictures; others are genuinely missing public assets.
- Decision: materialize references inside lite.scalecua before metric calls by
  fetching same-basename files from likely VM locations into the eval cache.
- Guardrail: if the basename cannot be found in the VM or official HF snapshot,
  classify the exact row as `missing_reference_asset` in `UPSTREAM_ISSUES.md`
  instead of silently passing or inventing a file.
- Validation: unit tests for path repair and oracle/probe rows that exercise
  image, PDF, PPT, and office-document reference paths.

### G-009: stale diagnostic rollout roots

- Type: `validation_process`
- Scope: long-running rollout roots started before new filters or adapter fixes.
- Symptom: old prompt parquet or env-server state can continue sampling rows
  that are now excluded or evaluating with pre-fix logic, producing misleading
  reward=0 counts.
- Evidence: the current 1000 rollout root is useful as diagnostic evidence but
  predates several exact filters and adapter fixes, so its raw success rate must
  not be used as the final gate.
- Decision: keep old roots only for visual/code triage. After import/static/unit
  validation, start a fresh env server and fresh prompt parquet for the final
  1000-rollout gate.
- Guardrail: always label stale rollout roots as diagnostic in logs and reports.
- Validation: final report must include prompt generation timestamp, import
  report hash/revision, env-server token/port, filter counts, and visual audit
  coverage.

### G-010: direct-mode validation cleanup

- Type: `validation_process`
- Scope: direct-mode oracle validation and any local env containers not managed
  by the env server.
- Symptom: failed or slow `docker rm -f -v` calls can leave orphan containers or
  volumes during high-concurrency oracle runs.
- Evidence: oracle validation logs show async Docker cleanup timeout warnings;
  the drift reaper and final sweep are expected to backstop, but this must be
  audited before launching another direct-mode batch.
- Decision: do not run multiple direct-mode oracle validators concurrently.
  Before and after each direct-mode run, inspect and clean matching
  `lite_scalecua_oracle_validate` containers.
- Guardrail: large rollout should use a fresh env server with bounded
  concurrency instead of direct mode.
- Validation: each oracle report should record final pass/fail counts and a
  post-run orphan check.

### G-011: generated getters that directly open VM paths

- Type: `migration_adapter`
- Scope: generated/RL overlay getters with `(env, config)` signatures that use
  host-local file APIs such as `openpyxl.load_workbook(config["path"])`,
  `Document(config["path"])`, `Presentation(config["path"])`, `Image.open`, or
  `ZipFile` without calling `env.controller.get_file` / `get_vm_file`.
- Symptom: official OSWorld VM evaluation can open `/home/user/...` directly
  because the judge code runs inside the VM-shaped environment. In CUA-Lite the
  generated getter runs on the host, so direct reads of `/home/user/...` either
  fail or return empty fallback values. The Calc family
  `0cecd4f3-74de-457b-ba94-29ad6b5dafb6` exposed this as a migration bug, not
  an upstream generated getter/type mismatch.
- Evidence: official imported getter `get_xlsx_sheet_names__34fcb36a` reads
  `config.get("path")` with `openpyxl.load_workbook(file_path, data_only=True)`.
  The matching task JSON has a valid generated getter/metric pair, so filtering
  it as `upstream_generated_eval_bug` would hide a lite adapter gap.
- Decision: keep these rows runnable. `lite.scalecua` now materializes VM paths
  for direct-file generated getters into `<cache_dir>/_overlay_inputs/` before
  invoking the getter. The materialization is source-shape gated and skipped for
  getters that already call `env.controller.get_file` / `get_vm_file`, and for
  legacy first-argument `file_path` getters handled by `_LazyVmFilePath`.
- Guardrail: do not broaden this into generic mutation of all `config` strings.
  Only known file-path keys are rewritten, never `dest`, command strings, URL
  rules, or expected text. If a getter still fails after materialization, probe
  the generated source and VM file first; add an exact upstream exclusion only
  after proving the official getter/metric contradicts the instruction.
- Validation: unit tests cover direct `openpyxl.load_workbook(config["path"])`
  materialization, `env.controller.get_file(config["path"])` non-mutation, and
  first-argument `file_path` getter compatibility. Add oracle coverage for
  Calc sheet-name/sheet-order generated rows during the next oracle fixture
  expansion.

### G-012: official reset/final-eval settle parity

- Type: `migration_adapter` / `validation_process`
- Scope: slow-launch or persistence-sensitive domains: GIMP, LibreOffice Calc,
  LibreOffice Impress, LibreOffice Writer, Thunderbird, VLC, and Chrome rows
  whose getters depend on live CDP/window state.
- Symptom: official ScaleCUA `run_single_example` waits substantially longer
  after reset and before final eval than the current CUA-Lite rollout path. The
  lite path can therefore take the first screenshot or run final eval before a
  slow app, postconfig launch, or persistence flush reaches the same state the
  official VM runner would observe.
- Evidence: sidecar audit
  `.exps/validate/lite.scalecua/audits/setup_eval_gap_audit.sidecar.md` cites
  official `SCALE-CUA/osworld_eval/lib_run_single.py` reset and pre-eval waits,
  while `lite/gym/sandbox/base.py` and `lite.osworld` setup use shorter generic
  waits. This is a migration-timing difference, not an upstream task defect.
- Decision: do not change the running 1000 rollout root mid-flight. First add
  targeted probes/oracle rows for slow-launch domains that capture
  `00_reset.png`, app readiness, postconfig file/window state, and final getter
  results. If races reproduce, add a `lite.scalecua`-scoped readiness/settle
  policy rather than changing base `lite.osworld` latency.
- Guardrail: do not blindly add the official 60s/20s sleeps to every task. Use
  domain/app-aware readiness where possible, and document any fixed wait as a
  ScaleCUA compatibility tax with before/after evidence.
- Validation: targeted reruns must compare raw reward, screenshots, and getter
  debug payloads before/after the settle change. Oracle coverage must include
  at least one slow-launch or postconfig-sensitive row per affected domain
  before this gap can close.

## Confirmed Eval Bugs From Rerun4 Triage

These entries are mirrored here for rollout/debug triage. The exact upstream
filter source remains `UPSTREAM_ISSUES.md` and
`dataset.UPSTREAM_GENERATED_EVAL_BUG_TASK_IDS`.

- `scalecua_osworld_train_libreoffice_calc_42e0a640_4f19_4b28_973d_729602b5a4a7_task_verify_34`
  is a confirmed upstream generated eval contradiction. The instruction asks
  to put minimum values in `Sheet2!A1/B1` and also add column headers
  "Minimum Revenue" and "Minimum Expenses"; the generated getter
  `get_sheet2_min_values__7ae1d612` reads `Sheet2!A1/B1` as numeric values.
  A correct spreadsheet cannot simultaneously put headers and values in those
  two cells. Action: exact filter with
  `exclude_reason="upstream_generated_eval_bug"`; fresh import now reports
  train `upstream_generated_eval_bug=48` and train runnable `16517`.
- `scalecua_osworld_train_libreoffice_calc_a82b78bb_7fde_4cb3_94a4_035baf10bcf0_task_verify_48`
  exposed a generated bookmark evaluator wiring defect. The task requires
  saving Caltech author pages in a folder named `Caltech Researchers`, and the
  generated rule carries `names=["Caltech Researchers"]`; the generic upstream
  `is_expected_bookmarks` metric hardcodes the folder `Liked Authors`. Action:
  do not filter this row. `lite.scalecua` now resolves generic
  `is_expected_bookmarks` for `train`/`rl` through a ScaleCUA wrapper that
  respects `rule["names"]`, and `EvalEnvShim.get_bookmarks()` reads
  `/home/user/chrome-data/Default/Bookmarks` first.
- `scalecua_osworld_train_libreoffice_calc_4188d3a4_077d_46b7_9c86_23e1a036f6c1_task_verify_31`
  remains a probe, not a confirmed exclude. The generated metric reads
  `C10` without `data_only=True`, while the task text asks for a formula whose
  calculated value should be `4949.64`. Before filtering or patching, rerun
  with a downloaded workbook probe that records raw formula text, cached value,
  LibreOffice recalculation behavior, and metric output.
- `scalecua_osworld_train_libreoffice_calc_12382c62_0cd1_4bf2_bdc8_1d20bf9b2371_task_verify_45`
  exposed a generated Calc chart getter compatibility bug. The generated getter
  calls missing private helper `_parse_cell_range`, and the generated metric
  compares openpyxl chart title objects directly to strings. Action: do not
  filter this row. `lite.scalecua` now injects `_parse_cell_range` into
  generated Calc chart getter modules and wraps the matching chart metric to
  normalize rich chart-title objects before comparison. Remaining reward-0
  cases for this row should be treated as visual/file probes, not as this
  fixed helper defect.
- `scalecua_osworld_train_libreoffice_calc_2c1ebcd7_9c6d_4c9a_afad_900e381ecd5e_task_verify_20`
  exposed a domain-classification migration bug. The generated source file is
  stored under HF `generated_tasks/libreoffice_calc`, but the canonical OSWorld
  example lives under `examples/multi_apps`; using the HF directory as
  `metadata.others.domain` polluted rollout/oracle stratification. Action:
  keep task ids and `metadata.others.source_domain` stable on the HF directory,
  but map `metadata.others.domain` from the OSWorld example domain. A visual
  reward-0 case for the bold-headings evaluator remains a normal LibreOffice
  save/getter probe under G-007; it is not an exact evaluator exclusion.
- `scalecua_osworld_train_libreoffice_impress_0a211154_fda0_48d0_9274_eaac4ce5486d_task_verify_43`
  exposed a ScaleCUA adapter bug for generated PPTX text metrics. The official
  expected config is `{"value": "Game Instructions"}`, while the generated
  metric `check_pptx_shape_text__8b4cb395` compares the result string directly
  to the expected value. Action: fixed in `lite.scalecua` by narrowly unwrapping
  `{"value": ...}` only for `check_pptx_shape_text__8b4cb395`, with a unit
  regression test. Do not exact-exclude this row.

## Open Probe Queue

- Rerun4 follow-up fixes from visual/code audit:
  - `recreation_url_check__6dc3893f96ccd943c500af1756962de6` was a migration
    getter bug: the generated Playwright getter read CDP `pages[0]`, which can
    be a background tab while the visible active tab is already on
    `recreation.gov`. Action: fixed in `lite.scalecua` by routing this hashed
    result type through the active accessibility-tree URL getter and returning
    `{"url": active_url}`.
  - `dir_file_count__00db2192` for the Django clone row was a migration
    compatibility gap with the OSWorld VM shape. Action: fixed only for
    `/home/user/django` + `*.py` by verifying `.git`, Django remote URL, and
    `django/__init__.py`, then recursively counting Python files; other
    `dir_file_count` rows keep their generated semantics.
  - `check_speedtest_report__26660ad1` had an over-narrow generated regex for
    current Speedtest report text such as `Ping / idle latency: 4 ms`. Action:
    fixed with a metric wrapper scoped to this one generated metric.
  - Exact upstream/live-site filters added after visual audit:
    `a728a36e..._task_verify_57`, `b4f95342..._task_verify_45`,
    `f3b19d1e..._task_verify_{39,52,64}`, and `9f3f70fc..._task_verify_37`
    use `exclude_reason="upstream_live_site_drift"`;
    `f3b19d1e..._task_verify_86` and `f8cfa149..._task_verify_52` use
    `exclude_reason="upstream_generated_eval_bug"`.
- Probe G-012 settle parity with slow-launch GIMP, Thunderbird, VLC, and
  LibreOffice oracle rows before changing default rollout timing.
- Probe `timedatectl status` fallback on timezone generated/RL rows; separately
  identify and decide NTP/systemd rows.
- Probe representative direct-path generated getters after the next cache
  import, especially Calc sheet-name/order rows, to confirm reward 1 under
  oracle actions.
- Probe GIMP `sessionrc` rows with pre/post file captures.
- Probe Thunderbird prefs rows with UI state, profile path, and `prefs.js`.
- Probe VLC `vlcrc` rows with persisted config before and after postconfig.
- Probe LibreOffice false failures by downloading final files and comparing
  generated getter facts against task instructions.
- Probe unresolved visual-audit rows before adding any more
  `upstream_generated_eval_bug` exact filters.

## 2026-07-14 Rerun5 Confirmed Fixes And Filters

- `scalecua_osworld_train_chrome_2ad9387a_65d8_4e33_ad5b_7580065a27ca_task_verify_16`
  exposed a migration adapter bug in the ScaleCUA bookmark metric wrapper.
  The rollout screenshot and replay showed the `Work` folder on the bookmarks
  bar, but the wrapper forwarded the injected `env` kwarg into upstream
  `desktop_env.evaluators.metrics.is_expected_bookmarks`, which accepts only
  `(bookmarks, rule)`. Action: fixed in `lite.scalecua` by dropping wrapper
  options before delegating. Replay of the original rollout actions now returns
  reward 1.0.
- `scalecua_osworld_train_chrome_6766f2b8_8a72_417f_a9e5_56fcaa735837_task_verify_0`
  exposed a migration adapter bug in Chrome extension getter repair. The
  visible state had `Hello Extensions 1.0` enabled, but hashed
  `extension_names__...` fell through to singular `extension_name*` handling
  and returned a string while the generated metric expects a list. Action:
  fixed in `lite.scalecua` by treating `extension_names` as list-valued.
- `scalecua_osworld_train_chrome_06fe7178_4491_4589_810f_2e2bc9502122_task_verify_34`
  is a confirmed upstream live-site drift row. The desired tabs are visibly
  restored, but live redirects produce URLs such as `about.gitlab.com` and
  `stackoverflow.com/questions` while the official expected URLs are
  `www.gitlab.com` and `www.stackoverflow.com`. Action: exact-filter with
  `exclude_reason="upstream_live_site_drift"`, not a broad URL alias.
- `scalecua_osworld_train_chrome_480bcfea_d68f_4aaa_a0a9_2589ef319381_task_verify_39`
  is a confirmed upstream generated evaluator bug. The task asks to enable Tab
  Groups flags, but the generated expected ids are old Chrome flag names
  (`tab-groups`, `tab-groups-collapse`) while the runtime exposes current
  related ids. Action: exact-filter with
  `exclude_reason="upstream_generated_eval_bug"`, not a global flag alias.

## 2026-07-15 Shenzhen Live Address Lookup Family Filter

- `scalecua_osworld_train_multi_apps_7ff48d5b_2df2_49da_b500_a5150ffc7f18_*`
  is filtered with `exclude_reason="upstream_live_site_drift"` for all 60
  generated train rows. This is not a migration bug: the official generated
  tasks require live Shenzhen location/address lookup but evaluate against
  fixed historical lists. The RL rows under the same OSWorld id are static
  Word-document edits and remain runnable/oracle-required.

## 2026-07-15 Budget.com Live Access Restriction

- `scalecua_osworld_rl_chrome_47543840_672a_467d_80df_8f7c3b9788c9_traj_verify_{0,1,2}`
  are exact-filtered with `exclude_reason="upstream_live_site_drift"`.
  Visual rollout reached Budget.com but the live site displayed an access
  restriction/security-verification page before the expected Boston Logan car
  rental results and sort/filter URLs could be reached. The neighboring rows
  already marked `proxy_required` keep that reason; unrelated generated/Office
  rows mentioning budget data remain runnable.

## 2026-07-15 Generated OS Script/Evaluator Mismatch

- `scalecua_osworld_train_os_6f56bf42_85b8_4fbb_8e06_6c44960184ba_task_verify_{43,44,46,47}`
  are exact-filtered with `exclude_reason="upstream_generated_eval_bug"`.
  Rollout and fixed-tag oracle probing showed the requested `report.txt`
  archive state can be created, but official `eval.sh` still validates the
  original `dir*/file1` task and fails with `Failure: file1 does not exist in
  dir1.` This is upstream generated task/evaluator drift, not a ScaleCUA
  transport/runtime defect.

## 2026-07-15 Generated VS Code Open-File Evaluator Mismatch

- `scalecua_osworld_train_vs_code_53ad5833_3455_407b_bbc6_45b4c79ab8fb_task_verify_{0,1,12,23,30,31,32,33,36,37,48}`
  are exact-filtered with `exclude_reason="upstream_generated_eval_bug"`.
  Visual rollout and a fixed-tag oracle probe showed that `main.py`/`README.md`
  can be open in VS Code while the official generated evaluator still returns
  reward 0. The generated rows call `GetOpenFile` or `OpenFile` and wait for
  `/home/user/OpenFile.txt`, but the bundled official
  `vscodeEvalExtension` for this source id registers only `OpenProject`,
  `GetColorTheme`, and `GetBreakPoint`. This is an upstream generated eval
  defect, not a container parity issue, so `lite.scalecua` must filter these
  rows instead of monkey-patching a new open-file getter.

## 2026-07-16 Generated Writer Email Evaluator Mismatch

- `scalecua_osworld_train_libreoffice_writer_0a0faba3_5580_44df_965d_f562a99b291c_task_verify_{39,40,41,42}`
  are exact-filtered with `exclude_reason="upstream_generated_eval_bug"`.
  The official generated metadata assumes the DOCX contains an address like
  `service@blcup.com`, but the actual source document contains only
  `Support Contact: service blcup.com` and no regex-valid email address. The
  generated getter therefore returns `{}` even for a visually reasonable edit.

## 2026-07-16 Generated Writer Paragraph Index Mismatch

- `scalecua_osworld_train_libreoffice_writer_e528b65e_1107_4b8c_8988_490e4fece599_task_verify_57`
  is exact-filtered with `exclude_reason="upstream_generated_eval_bug"`.
  The instruction asks to italicize the second paragraph. The source DOCX
  paragraph order is: paragraph 0 title (`Question Two: Geography and Magical
  Realism`), paragraph 1 first body paragraph, paragraph 2 second body
  paragraph. The official generated evaluator hard-codes `para_idx=1` in both
  `docx_paragraph_italic__4e5c493c` and
  `check_docx_para_italic__4e5c493c`, so a visually correct edit to the second
  body paragraph is scored against the first body paragraph instead. This is an
  upstream generated task/evaluator mismatch, not a lite migration defect.

## 2026-07-16 Generated GNOME Theme How-To Rows

- `scalecua_osworld_train_chrome_f8369178_fafe_40c2_adc4_b9b08a125456_task_verify_{37,38,41,42,46,47,50,51,54,55}`
  are exact-filtered with `exclude_reason="instruction_eval_mismatch"`. These
  rows ask for an explanatory answer (`Provide steps...` / `Show me how...`) to
  install or activate GNOME themes, icons, or cursors, while the generated
  evaluator reads `gsettings*.out` and requires the desktop setting to have
  actually changed. Neighboring rows from the same source id still ask for direct
  state changes (`Install...`, `Set...`, `Apply...`) and remain runnable.

## 2026-07-16 Generated File Move Checked As Copy

- `scalecua_osworld_train_chrome_788b3701_3ec9_4b67_b679_418bfa726c22_task_verify_6`
  is exact-filtered with `exclude_reason="upstream_generated_eval_bug"`. The
  instruction asks to move `codes/main.py`, `codes/test_connection.py`, and
  `meeting_notes.md` into `Documents/Projects/OSWorld/backup`. Visual rollout
  showed the final terminal `ls` listing all three files in that destination.
  The generated getter/metric pair
  `multiple_files_exist__03426e679d8f4571bede57a16eea69a4` /
  `check_all_files_exist__03426e679d8f4571bede57a16eea69a4` is explicitly a
  copy checker: it requires the original source files to still exist and even
  logs `Source file missing - this was a MOVE, not a COPY`. Neighboring rows
  that ask to copy the same files remain runnable.

## 2026-07-16 PPTX Embedded Image Name Compatibility

- `scalecua_osworld_train_libreoffice_impress_c82632a4_56b6_4db4_9dd1_3820ee3388e4_task_verify_47`
  visually succeeded in rollout: slide 2 contained the inserted image and the
  LibreOffice Properties pane showed width `5.00 cm` and height `4.00 cm`.
  The generated metric `check_pptx_image_size__ca0cb440` still returned 0
  because it required the embedded picture name/filename to contain `none.png`.
  LibreOffice/PPTX embedding does not reliably preserve the source filename
  (`python-pptx` often sees a generic `Picture 1`/media part), so
  `lite.scalecua` now wraps `check_pptx_image_size__*`: it first runs the
  official metric unchanged, then retries without only the unstable image-name
  constraint when dimensions and slide checks are otherwise sufficient. This is
  a migration compatibility fix; hash/checksum image tasks are not relaxed.

## 2026-07-16 Generated DMV/Ticketek Live URL Drift

- `scalecua_osworld_train_chrome_a728a36e_8bf1_4bb6_9a03_ef039a5233f0_task_verify_{21,22,24,25}`
  are exact-filtered with `exclude_reason="upstream_live_site_drift"`. Visual
  rollout for `task_verify_21` reached the correct Virginia DMV Real ID page,
  but the live site canonicalized it to `https://dmv.virginia.gov/licenses-ids/real-id`
  while the generated evaluator requires the historical exact URL with
  `www.dmv.virginia.gov`. Neighboring Real ID rows that already use optional
  `www` regexes remain runnable.
- `scalecua_osworld_train_chrome_f3b19d1e_2d48_44e9_b4e1_defcae1a0197_task_verify_38`
  is exact-filtered with `exclude_reason="upstream_live_site_drift"`. Visual
  rollout reached the current Biletix/Ticketek Accessibility help article for
  accessible seating, but the generated evaluator still requires both
  `Accessible` and `FAQ` tokens in the active tab URL/title. The current help
  center route/title no longer includes `FAQ`.

## 2026-07-16 Generated Chrome Extension Source-Type Getter Compatibility

- Visual train rollout showed unpacked `Hello Extensions` tasks reaching
  `chrome://extensions` with the extension installed, while
  `extension_source_type__ae6416e4` still returned reward 0. The official
  generated getter reads the default VM Chrome Preferences path directly, but
  lite.scalecua can run Chrome through the lite profile layout. `lite.scalecua`
  now repairs empty `extension_source_type*` getter results from the discovered
  extension entries into the official metric shape
  `{extension_name: "webstore"|"unpacked"}`. This is a migration compatibility
  fix, not an upstream task exclusion.

## 2026-07-16 Generated GNOME Terminal How-To Rows

- `scalecua_osworld_train_os_13584542_872b_42d8_b299_866967b5c3ef_task_verify_{14,15,16,22,24,25,27,28}`
  are exact-filtered with `exclude_reason="instruction_eval_mismatch"`. These
  rows ask for a how-to answer (`Guide me...`, `Explain how...`, `Show me
  how...`) about persistent GNOME Terminal font, scrollback, or default-profile
  settings, while the generated evaluators require the hidden terminal profile
  state to have actually changed. Neighboring rows from the same source id that
  ask to configure, set, ensure, or help fix the state remain runnable.

## 2026-07-16 Generated Professor Notes XLSX Getter Column Bug

- `scalecua_osworld_train_libreoffice_calc_c7c1e4c3_9e92_4eba_a4b8_689953975ea4_task_verify_65`
  visually reached the intended state: `G2=Notes` and `G3=PhD advisor` for the
  Qi Liu row. The official generated getter
  `get_xlsx_cells_dict__a9a82c07` reads professor names from column `A`, but
  the spreadsheet stores row numbers in `A` and professor names in `B`. The
  official metric then awards only the header half-credit because it sees the
  first professor name as `"1"` rather than `"Qi Liu"`. `lite.scalecua` now
  repairs only this exact result type by reading names from column `B` and
  notes from column `G`, preserving the official metric shape.

## 2026-07-16 Generated GNOME Terminal GSettings Getter Compatibility

- `scalecua_osworld_train_os_13584542_872b_42d8_b299_866967b5c3ef_task_verify_30`
  visually showed the agent setting the GNOME Terminal default profile
  `cursor-shape` to `ibeam`, but the generated getter returned reward 0 because
  it depends on postconfig opening a terminal, typing a `gsettings get` command,
  and then scraping terminal output. The semantic evaluator is the same
  persistent `gsettings` state, so `lite.scalecua` now repairs the generated
  terminal result types `terminal_cursor_shape__*`, `terminal_scrollback__*`,
  `terminal_profile_name__*`, and `terminal_color_scheme__*` by reading the
  default GNOME Terminal profile directly through the desktop session bus. This
  keeps the official include/exclude metrics unchanged while removing a
  container UI synchronization false negative.

## 2026-07-16 Generated Cars.com Live URL Drift

- `scalecua_osworld_train_chrome_82279c77_8fc6_46f6_9622_3ba96f61b477_task_verify_{8,9,10,11}`
  are exact-filtered with `exclude_reason="upstream_live_site_drift"`. Visual
  rollout for `task_verify_9` reached a live-site unavailability path and then
  a fallback vehicle site while the generated evaluator still parses Cars.com
  active URL params for `zip=02101`, `maximum_distance=40`,
  `list_price_max=30000`, and `fuel_slugs[]=hybrid`. The task can no longer be
  made deterministic without a stable Cars.com fixture or proxy capture, so
  these rows join the existing exact Cars.com drift filters.

## 2026-07-16 Generated Apple Compare Instruction/Eval Mismatch

- `scalecua_osworld_train_chrome_f5d96daf_83a8_4c86_9686_bada31fc66ab_task_verify_{8,14,17}`
  are exact-filtered with `exclude_reason="instruction_eval_mismatch"`, matching
  the already-filtered same-family rows `task_verify_{22,51}`. These
  instructions ask for a text comparison of Apple product features/specs, while
  the evaluator requires an active Apple comparison URL whose `modelList`
  query params contain specific model slugs. Rows that explicitly ask to use
  Apple's comparison page/tool remain separate candidates and are not filtered
  by this exact addition.

## 2026-07-16 Generated Thunderbird Pref Method Compatibility

- 39 generated Thunderbird rows using `check_thunderbird_prefs` encode rule
  methods as `==`, `literal`, `in`, or `contains`, while the upstream OSWorld
  Thunderbird prefs metric only accepts OSWorld's `_match_value_to_rule`
  method vocabulary. Visual rollout found false negatives where the UI state was
  correct, for example
  `scalecua_osworld_train_thunderbird_3f28fe4f_5d9d_4994_a456_efd78cfae1a3_task_verify_11`
  with `Attach my vCard to messages` enabled but reward 0. `lite.scalecua` now
  wraps only Thunderbird prefs metrics for train/RL overlays: it first runs the
  original metric, then falls back to a generated-schema matcher for those
  method aliases. This is a migration/evaluator compatibility fix, not an
  exclusion.

## 2026-07-15 RL Runtime Tail Oracle Closure Suggestions

- `scalecua_osworld_rl_os_3ce045a0_877b_42aa_8d2c_b4a863336ab8_traj_verify_6`
  is not strict-oracle eligible under the fixed base image. The task asks for
  Large Text plus animations disabled, but the fixed-base no-op precheck already
  returns partial reward `0.5`; replay can reach `1.0`. Suggested exact filter:
  `exclude_reason="upstream_partial_precondition"` or the closest accepted
  local exclude reason for generated rows whose initial state partially
  satisfies the official conjunction. Do not admit this as a strict fixture with
  `expected_pre_reward=0.5` for the RL closure gate.
- `scalecua_osworld_rl_chrome_b4f95342_463e_4179_8c3f_193cd7241fb2_traj_verify_{3,4}`,
  `scalecua_osworld_rl_chrome_b7895e80_f4d1_4648_bee0_4eb45a6f1fa8_traj_verify_{0,2}`,
  and
  `scalecua_osworld_rl_chrome_da46d875_6b82_4681_9284_653b0c7ae241_traj_verify_2`
  are the remaining Chrome coverage gaps in the OS/VS Code/VLC/Chrome small-tool
  scope. They evaluate live Recreation.gov, TripAdvisor, or Microsoft Bookings
  active-tab/form state. Existing candidate notes show local file/HTTP mocks did
  not satisfy the CDP/XPath getters for the Recreation.gov and TripAdvisor
  families. Suggested exact filter if closure requires deterministic RL oracle
  coverage: `exclude_reason="upstream_live_site_drift"`.
