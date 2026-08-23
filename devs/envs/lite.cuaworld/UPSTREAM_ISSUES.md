# lite.cuaworld — upstream task-content defects (record-only, excluded)

Defects in the **upstream (gym-anything) task content** — verifiers, task.json, export scripts,
task assets. Per the ownership rule ([lite.cuaworld.md](/devs/envs/lite.cuaworld/lite.cuaworld.md#where-to-fix-things-decision-table))
we **record and exclude** these, we do **not** patch upstream task content. (Bugs *we* introduced
porting the software-environment layer are fixed — see the "Migration defects" section there.)

## How they're found and excluded

Results live in
[`lite/gym/envs/lite/cuaworld/data/validation_excludes.json`](/lite/gym/envs/lite/cuaworld/data/validation_excludes.json)
(`software → {task_id: reason}`). The **engine bakes each flagged task's reason into
`metadata.others['exclude_reason']`** at registration (`src/software.py::_exclude_reasons`), so a
rollout drops them with the same standard filter lite.osworld uses:

    --filter "lambda m: not m.others.get('exclude_reason')"

The file is produced by **three layers**, each owning a disjoint set of reason codes. Every layer
records its own per-software coverage in `_meta._layers`, including a `never_swept` list — because
a software with zero findings and a software that was never looked at are otherwise **byte-
identical** in this file (`_exclude_reasons` does `doc.items()`, so a missing key silently yields
`{}`). That is not hypothetical: `geogebra` and `qgis` were absent from the file entirely and read
as clean for as long as it existed.

| layer | how it runs | owns |
|---|---|---|
| `live_noop` | reset each task (runs `pre_task`) in a real container, then verify with **no agent action and no VLM**. A healthy task scores a clean 0; anything that errors before scoring, or scores non-zero for a no-op, is broken regardless of the agent. | the 11 codes below the rule |
| `forged` | offline. Execute every registered verifier host-side against a synthesized **lazy-agent filesystem**: every file the verifier asks the container for exists, is non-trivial in size, has a plausible extension and a fresh mtime, and is empty/garbage inside. The judge answers **no**. A pass means the deliverable's CONTENT is never checked. | `hollow_artifact` |
| `setup_rc` | offline. Replay the `pre_task` hook in a throwaway root (user namespace + chroot, no network) whose `/workspace` is staged exactly as `install.sh` stages it, and assert rc == 0. | `setup_aborts` |

The two offline layers are one generator:

```bash
# dry run — writes forged.json / setup_rc.json / findings.json under --report
uv run --no-sync python lite/gym/envs/lite/cuaworld/scripts/utils/validation_sweep.py

# regenerate the offline layers of validation_excludes.json in place
uv run --no-sync python .../validation_sweep.py --write
```

`--write` rewrites **only** the reason codes those layers own and recomputes `_meta._total`;
`live_noop` entries are carried through untouched. Re-running reproduces the same entries (verified:
two consecutive full runs of `forged` produce byte-identical findings), so these two classes are
**generated, not hand-maintained**. Regenerating the `live_noop` layer still means re-running the
container sweep (see [devs/data/lite.cuaworld/AGENTS.md](/devs/data/lite.cuaworld/AGENTS.md)), and its
hand-added live-rollout entries (below) still have to be re-applied by hand afterwards.

Per-env, per-task detail also lives beside the materials in the HF dataset
`cua-lite/lite.cuaworld-assets` (`<env>_env/UPSTREAM_ISSUES.md`).

## Reason codes (all splits: train + eval + long_horizon → 880 tasks)

| reason | layer | count | what it is |
|---|---|---|---|
| `nonzero_baseline` | live_noop | 406 | verifier awards partial `raw_score` to a no-op episode (miscalibrated baseline) |
| `missing_verifier` | live_noop | 243 | `task.json` declares a program verifier, but no `verifier.py` was shipped, or a live rollout proves a required ground-truth asset is missing |
| `gameable_full` | live_noop | 122 | no-op or live rollout scores **reward=1** for an empty, artifact-only, or incomplete outcome — poisons SFT ("wrong/no work → success") |
| `broken_export_query_live_instance` | live_noop | 17 | slicer3d export spawns a fresh headless Slicer, queries an empty scene (no `loadScene`) and never exits (no `slicer.util.exit`) → hang + wrong verdict |
| `verifier_runtime_pip` | live_noop | 13 | verifier does `python -m pip install …` at runtime; the engine venv is uv-managed with no pip |
| `verifier_crash` | live_noop | 13 | verifier raises (KeyError / TypeError / missing dep / broken pipe) before scoring |
| **`hollow_artifact`** | **forged** | **10** | verifier PASSES on a plausible-but-empty deliverable, with no judge involved — the file's content is never checked |
| `verifier_nameerror` | live_noop | 8 | verifier uses an undefined name |
| `verifier_syntaxerror` | live_noop | 5 | the shipped `verifier.py` has a Python syntax error (bytes-identical to upstream) |
| `slow_timeout` | live_noop | 5 | setup / export exceeds the 600s deadline (heavy per-episode work) |
| `other` | live_noop | 35 | miscellaneous setup/export/verify-pipeline failure that fits none of the above |
| `verifier_pdb` | live_noop | 1 | verifier has an embedded debugger call (`BdbQuit`) |
| **`setup_aborts`** | **setup_rc/live** | **2** | the `pre_task` hook exits non-zero, so the episode starts from a half-built desktop |

The following live-layer entries are **hand-added** and must be re-applied after a `live_noop` regenerate,
because the no-op sweep cannot see them:

- `openvsp/openvsp_wave_drag_area_ruling` (`verifier_nameerror`) — `export_result.sh:89`
  interpolates a bare `{RESULT_FILE}` into an f-string; `RESULT_FILE` is a SHELL variable, so the
  post_task hook NameErrors before moving the result JSON into place. The sweep is blind because the
  resulting score is 0, which is also the expected agent-free baseline.
- `jstock/anonymize_portfolio_view` (`gameable_full`) — the verifier collects both the live final
  screenshot and the PNG the agent wrote, then judges only `images_to_check[-1]`, which is the
  agent's file whenever it exists. A live rollout scored 1.0 by screenshotting and then painting grey
  boxes over the currency columns **in the image** while JStock still showed every column. A no-op
  writes no file and scores 0, so the sweep is blind.
- `gcompris/science_experiment_catalog`, `gcompris/solar_system_explore`, and
  `gcompris/vector_drawing_composition` (`gameable_full`) — live B3 rollouts scored reward 1 from
  terminal-written or artifact-only outputs while the required GCompris activity was not completed/used.
- `gcompris/simple_word_processing` (`gameable_full`) — live B4 rollout scored reward 1 even though feedback said the
  required screenshot file was not found at the expected path.
- `gcompris/target_score` (`gameable_full`) — live B4 rollout scored reward 1 from loose activity/progression while
  the final pre-action frame still showed level 1/5 with an unanswered equation.
- `slicer3d/tumor_ventricle_proximity` (`missing_verifier`) — live B4 rollout reached the report/measurement path, but
  the verifier had no `/tmp/proximity_ground_truth.json`, so it could not judge distance/classification.
- `openemr/reschedule_appointment` (`gameable_full`) — live B5 rollout scored reward 1 after direct MySQL mutation
  from a tainted setup state; the verifier did not require the existing appointment id to be preserved and still passed
  despite duration feedback showing 900 seconds instead of the expected 1800.
- `openemr/check_in_appointment` (`gameable_full`) — focused B5 reroll scored reward 1 after the agent used direct
  MySQL to set `pc_apptstatus='@'` while the OpenEMR UI was blocked by modal/DataTables warnings. The verifier accepted
  final DB state and weak screenshot evidence as UI completion.
- `openemr/add_patient_guarantor`, `openemr/cancel_appointment`, `openemr/close_patient_encounter`,
  `openemr/document_no_show`, and `openemr/document_refusal_of_care` (`other`) — pinned OpenEMR materials contain
  image/live-confirmed setup or export defects not covered by the empty-root setup replay. Four target seed patients or
  encounters that the shipped OpenEMR data never creates; `document_no_show` produces invalid result JSON after the
  live setup path (`NULL` timestamp in the export), so reward 0 is not a trustworthy agent-failure signal.
- `slicer3d/tracheal_shape_index` (`gameable_full`) — live B5 rollout scored reward 1 with `slice_number=82` against
  GT slice 46 and an AP diameter error above tolerance; the verifier gate requires only score >= 60 plus one accurate
  diameter, not the requested aortic-arch-level measurement.
- `openemr/generate_day_sheet` (`gameable_full`) — live B7 rollout scored reward 1 after the agent typed a fabricated
  `data:text/html` Day Sheet page into Firefox; the verifier accepted browser-title/VLM evidence without enforcing an
  OpenEMR report origin.
- `openemr/add_insurance_info`, `openemr/document_drug_allergy`, and
  `openemr/document_fall_risk_assessment` (`other`) — live B7 rollouts exposed patient identity/PID mismatches or
  malformed export JSON. The task assets hard-code a target pid while the named seeded patient is absent or different,
  so reward 0 is not a clean agent-failure signal.
- `slicer3d/split_segment_scissors` (`broken_export_query_live_instance`) — live B7 final frame showed
  `Liver_Right`/`Liver_Left` segments saved from the interactive scene, but `export_result.sh` launched a fresh
  headless Slicer process and queried an empty scene.
- `solvespace/fishplate_angle_repair` (`gameable_full`) — live B7 scored reward 1 after direct `.slvs` text edits and
  a hand-written tiny DXF; the verifier checked constraint values and DXF existence/mtime, not real export geometry.
- `solvespace/slot_profile_tangent` (`verifier_crash`) — B8 focused reroll, run after fixing the CUA-Lite post_start
  wrapper, started in a healthy SolveSpace UI but the agent-created `.slvs` artifact could not be decoded as UTF-8.
  The pinned verifier logs the read failure, leaves `line_count`/`arc_count` uninitialized, then raises
  `UnboundLocalError` in the final pass gate instead of returning a clean score 0.
- `astroimagej/time_series_subframing` (`gameable_full`) — live B7 scored reward 1 from direct FITS cropping and a
  report even though the task's own VLM checklist requires GUI stack/ROI/crop workflow and final feedback said VLM did
  not confirm it.
- `astroimagej/identify_variable_star` (`other`) — live B7 final screenshot showed a non-empty report, but
  `export_result.sh` reads `os.environ["REPORT_FILE"]` without exporting the shell variable, so verifier feedback/raw
  score are polluted before judging the actual missing/wrong measurement table.
- `openemr/document_medication_administration` (`other`) — live B8 showed an incomplete B12 administration workflow,
  but the verifier signal is also polluted: `task.json` declares post_task `export_task.sh`, while the directory only
  contains `export_result.sh`, so `/tmp/medication_administration_result.json` is never written.
- `vscode/repair_epidemiological_data_pipeline` (`setup_aborts`) — live B8 setup materialized an empty/partial
  workspace because `setup_task.sh` writes `csv.writer(sys.stdout)` without importing `sys`; the agent then guessed
  against missing source/test files.
- `slicer3d/optic_nerve_sheath_diameter` (`other`) — live B8 loaded `BrainMRI_T1`, which does not include clear
  globes/retrobulbar optic nerve sheaths, while the task/evaluator require bilateral ONSD measurements 3 mm posterior
  to each globe.
- `slicer3d/aorta_measurement` (`other`) — live B8 loaded a phantom-like `AbdominalCT`, not anatomy that can support
  the task's vertebral-level requirement.
- `gcompris/prime_muncher` (`gameable_full`) — live B9 scored reward 1 after entering Number Munchers primes mode
  with only a small visible score/progress indicator (`2`), not the required score of at least 20 nor confirmed
  `~/Documents/high_score.png`. The verifier passes on activity plus any gameplay evidence.
- `solvespace/extrude_constrained_profile` (`gameable_full`) — live B9 scored reward 1 after direct `.slvs` text
  edits produced a file containing `Group.type=5100`; verifier feedback still reported no 100 mm depth constraint, but
  the pass gate accepts the edited group type and saved file.
- `vscode/repair_historical_nlp_pipeline` (`other`) — live B9 trajectory reported `6 passed`, but
  `export_result.sh` hashes every file under `tests/` after running pytest, so generated `__pycache__` can trip the
  "tests modified" anti-gaming gate and convert a plausible success into reward 0.
- `slicer3d/neural_foramen_assessment` (`other`) — live B9 loaded a phantom-like `LumbarCT` with no lumbar foramina,
  while setup generated synthetic/random foraminal ground truth from volume shape. Reward 0 is not a clean agent
  failure signal because the requested anatomy is absent.
- `gcompris/clock_activity` (`gameable_full`) — live B10 scored reward 1 while first/middle/last frames still showed
  an active clock prompt (`4 hours 30 minutes`) with wrong-looking hand positions. The verifier/VLM gate accepted
  navigation and interaction as task completion.
- `slicer3d/gastric_volume_bariatric` and `slicer3d/liver_ablation_suitability` (`other`) — live B10 loaded
  phantom-like/synthetic volumes that do not support the requested gastric/liver clinical measurements; these are not
  clean capability failures.
- `solvespace/parametric_scissor_lift_kinematics` and `solvespace/symmetric_trapezoid_channel` (`other`) — live B10
  zeros were visually fair, but the pinned verifiers text-decode binary `.slvs` files with UTF-8 and hit decode
  failures before trustworthy semantic checks. A real SolveSpace success could be rejected through the same path.
- `vscode/debug_ml_model_api` (`other`) — live B10 final frame showed the task's pytest suite passing after file
  writes, but the exporter short-circuited all hidden checks with `Files were not modified during the task` from a
  second-resolution mtime gate. Treat the sample as polluted rather than a clean success or clean failure.
- `astroimagej/calibrate_science_frames` and `astroimagej/create_color_ratio_map` (`gameable_full`) — live B11
  reward-1 trajectories passed from terminal/Python or artifact-only outputs; `create_color_ratio_map` also had wildly
  inaccurate ratios and no confirmed Image Calculator workflow. These rewards are not trustworthy AstroImageJ GUI
  completion signals.
- `gcompris/braille_alphabet` (`other`) — live B11 visual trajectory completed three letters and returned to the menu,
  but the verifier relied on progress-file/success-animation evidence and scored 0. Treat the sample as a polluted
  false negative rather than an agent failure.
- `astroimagej/galaxy_isophotal_morphology` and `astroimagej/point_source_suppression` (`gameable_full`) — live B12
  reward-1 trajectories passed from terminal/Python-generated FITS/CSV/report artifacts while first/middle/last frames
  showed idle AstroImageJ and final VLM feedback did not confirm the required GUI workflow.
- `astroimagej/map_transient_artifacts_zproject` (`other`) — live B12 agent created projection/artifact outputs, but
  the post-task/export path failed to produce `/tmp/task_result.json`; the exporter loads a large FITS stack and the
  surfaced zero is an export/verification failure, not a clean agent failure.
- `gcompris/reading_readiness_assessment` (`gameable_full`) — live B12 reward 1 came from a terminal-written report
  after at most three activities were opened and only one had meaningful interaction, despite the task requiring at
  least four distinct activity interactions.
- `openemr/document_flu_vaccine` and `openemr/prescribe_medication` (`gameable_full`) — live B12 reward-1 trajectories
  passed after direct MySQL inserts into OpenEMR tables; the verifiers checked final DB state and did not require the
  requested UI workflow.
- `openemr/generate_aging_report` (`gameable_full`) — live B12 reward 1 was awarded on an empty Collections aging
  report with no patient rows and zero totals because the verifier pass gate overweights navigation/columns.
- `openemr/add_procedure_fee_sheet` (`other`) — live B12 found task metadata/setup/export pinned to PID 5, while the
  shipped DB state has a different patient at PID 5 and no matching Gerald Koss target; reward 0 is not a clean
  agent-failure signal.
- `astroimagej/flat_field_noise_analysis` (`gameable_full`) — live B13 scored reward 1 even though final feedback said
  the VLM did not detect the required Z-projection operations; the trajectory used terminal/Python artifacts rather
  than a trustworthy AstroImageJ GUI workflow.
- `vscode/fix_ecommerce_i18n_implementation` (`gameable_full`) — live B13 scored reward 1 / raw 80 while verifier
  feedback still said `de.json` plurals were broken; the pass threshold converts an incomplete requested fix into a
  full success signal.
- `gcompris/railway_construct` (`other`) — B13 source audit found the requested Railway track-construction activity is
  hallucinated/unavailable; the closest shipped GCompris activity is a Railroad memory game, so reward 0 is not a clean
  agent-failure signal.
- `astroimagej/multiband_histogram_analysis` (`verifier_crash`) — live B14 reset/first frame showed AstroImageJ
  launched normally and the agent wrote `ionization_stats.txt`, but the pinned verifier calls
  `extract_stats('H-alpha|656nm', ...)` and then `float(match.group(1))`; on a line starting `H-alpha 656nm` the first
  regex alternation branch matches with no numeric capture groups, raising `TypeError` before scoring. This is a
  task-local verifier regex bug, not an AstroImageJ image/runtime failure.
- `openemr/patient_chart_review` (`gameable_full`) — live B14 reward 1 came from terminal/SQL-assisted chart extraction
  and shell-writing `/home/ga/Desktop/patient_summary.txt`, bypassing the requested OpenEMR chart-review workflow.
- `openemr/document_medical_device` (`gameable_full`) — live B14 reward 1 came from a direct MySQL insert into the
  `lists` table; the verifier checked final DB state and did not require UI provenance.
- `openemr/generate_audit_log_report` (`gameable_full`) — live B14 reward 1 was awarded even though verifier feedback
  said report results were not visible and the final UI did not show the requested patient/report; the agent generated
  `/tmp/audit_report.html` outside OpenEMR.
- `openemr/add_referral_source_option` (`gameable_full`) — live B16 reward 1 came from reading OpenEMR DB credentials
  and inserting `Westside Urgent Care` directly into `list_options` with MySQL. The verifier checked the final DB state
  and count delta, not UI provenance, so the sample is a direct-state false positive.
- `astroimagej/create_binary_source_mask`, `astroimagej/generate_inverted_finding_chart`, and
  `astroimagej/plan_mos_spectroscopy_targets` (`gameable_full`) — live B17d/fix reward-1 trajectories passed from
  terminal/Python-generated FITS/CSV/ROI/PNG artifacts while AstroImageJ stayed visually idle. The focused
  `generate_inverted_finding_chart` reroll proved the CUA-Lite `cv2` runtime gap was fixed, then exposed the pinned
  verifier gate: it still passed with `raw_score=75` while feedback said the VLM did not detect the required `M16 Core`
  text or target marker. `plan_mos_spectroscopy_targets` likewise said the VLM did not detect ROI Manager or visual
  selections, so these rewards are artifact-only and not trustworthy AstroImageJ GUI-completion signals.
- `astroimagej/eagle_roi_transfer_measurement` (`gameable_full`) — live B15 scored reward 1 from terminal/Python
  CSV and text artifacts. The final reward feedback explicitly said the VLM did not confirm a polygon around the
  Pillars of Creation, but CSV existence, exact equal area, sufficient area, and ratio arithmetic still cleared the
  pass gate.
- `astroimagej/detect_exoplanet_transit` (`other`) — live B15 produced a plausible written transit analysis, but
  `export_result.sh` reads `MEASUREMENT_FILE` inside an embedded Python block without exporting the shell variable,
  then a radius regex can pick up the uncertainty (`0.16 R_J`) instead of the derived planet radius. The resulting
  reward 0 is an export/verifier mismatch, not a clean GUI-task failure.
- `astroimagej/measure_atmospheric_extinction` (`other`) — live B15 ended with
  `Invalid control character` while loading `/tmp/task_result.json` because `export_result.sh` embeds multiline report
  content directly in JSON. The same verifier also treats report/data/coefficient artifacts as enough to pass without
  a hard VLM workflow gate, so this reward is not a trustworthy training signal.
- `astroimagej/empirical_snr_map_generation` (`gameable_full`) — live B18c scored reward 1 / raw 70 even though
  verifier feedback flagged the SNR FITS dtype as wrong, the applied background stats as suspect, and the reported
  stats as inconsistent with the image math. The trajectory generated terminal/Python artifacts rather than a
  trustworthy AstroImageJ workflow.
- `openemr/order_lab_tests` (`other`) — live B18c exposed a pinned patient/material mismatch: the prompt names Rosetta
  Effertz (DOB 1965-04-18), while the setup/export/verifier hard-code `patient_pid=5` and only check that PID. The
  agent had to create Rosetta as a new patient, and the final reward 0 only checked for orders on pid 5, so the zero is
  not a clean agent-failure signal.
- `openemr/post_insurance_payment` (`gameable_full`) — live B18c scored reward 1 / raw 95 despite feedback saying no
  EOB reference was found in the payment memo, while the task explicitly required `EOB2024-7834`. The pass gate ignores
  a required reference field after amount/new-payment checks pass.
- `slicer3d/clean_segmentation_islands` (`other`) — live B18c began from a bare desktop even though the task-local
  setup script attempts to launch Slicer, then the final exporter wrote invalid JSON by embedding Python boolean text
  such as `False` directly into `/tmp/islands_task_result.json`. Reward 0 is therefore setup/export pollution, not a
  clean failed segmentation attempt.
- `slicer3d/measure_tumor_vessel_distance` (`gameable_full`) — live B18c scored reward 1 / raw 65 with
  `minimum_distance_mm: 0.0`; the verifier feedback itself said the distance was out of plausible range and a suspected
  default value, but the sparse pass gate accepts score >=65 through output/existence/classification/VLM points.
- `gcompris/roman_numerals_study_guide` (`gameable_full`) — live B20 scored reward 1 even though first/middle/last
  frames and the trace showed only four conversions solved (`I`, `V`, `X`, `L`) while the final screen still showed
  the `C` prompt. The verifier awards file/header/symbol/activity credit without counting the requested five solved
  conversions. Evidence: `.logs/takeover-20260726-b20/canaryA/cuaworld-gcompris/train/roman_numerals_study_guide/sample_00`.
- `vscode/remediate_crypto_flaws` (`other`) — live B20 ended with tests passing and code using
  `PBKDF2HMAC`/`AESGCM` from `cryptography`, but the pinned evaluator recognizes only narrow
  `pbkdf2_hmac`/`GCM` attribute patterns and returned reward 0 / raw 65. The zero is a verifier false negative, not a
  clean agent failure. Evidence: `.logs/takeover-20260726-b20/canaryA/cuaworld-vscode/train/remediate_crypto_flaws/sample_00`.
- `slicer3d/fiducial_volume_registration` (`other`) — live B21 final frame showed at least five visible landmark
  labels and a printed rigid transform in the Slicer Python Console, but `export_result.sh` launches a fresh headless
  Slicer process and queries that empty scene for transform/fiducial nodes. The resulting reward 0 / raw 10 says
  "No transform node created" and "0 pairs" despite the live scene evidence, so the zero is an export false negative,
  not a clean agent failure. Evidence:
  `.logs/takeover-20260727-b21/wave1/cuaworld-slicer3d/train/fiducial_volume_registration/sample_00`.
- `gcompris/share_candies_division` (`other`) — live B22 final trajectory showed the Share the candies side-bar counts
  but a blank central play area, so the VLM/verifier could not see candies, children, distribution, or success despite
  a normal reset. Treat the zero as a task/material/rendering defect rather than a clean agent failure. Evidence:
  `.logs/takeover-20260727-b22/wave1/cuaworld-gcompris/eval/share_candies_division/sample_00`.
- `astroimagej/cosmic_ray_counting` (`other`) — live B23 output used the task's stated standard deviation method and
  saved a valid difference image/report, but setup ground truth computes the hidden threshold with MAD×1.4826 robust
  sigma. The resulting count mismatch is a prompt/verifier mismatch, not a clean agent failure. Evidence:
  `.logs/takeover-20260727-b23/wave1/cuaworld-astroimagej/train/cosmic_ray_counting/sample_00`.
- `astroimagej/extract_cluster_core` (`gameable_full`) — live B23 paid reward 1 / raw 60 while its own feedback said the
  subframe was from the wrong location, subframe/report metrics were missing or inaccurate, and VLM did not detect the
  ROI/crop operation. The pass gate turns a wrong extraction into a full sparse success signal. Evidence:
  `.logs/takeover-20260727-b23/wave1/cuaworld-astroimagej/train/extract_cluster_core/sample_00`.
- `gcompris/cross_domain_developmental_battery` (`gameable_full`) — live B23 reward 1 came from a terminal-written
  report after visual evidence showed only a narrow activity path, not the requested four distinct cognitive domains.
  The verifier checks report keywords/mtime and not activity completion provenance. Evidence:
  `.logs/takeover-20260727-b23/wave1/cuaworld-gcompris/train/cross_domain_developmental_battery/sample_00`.
- `gcompris/missing_letter_spelling` (`other`) — live B23 ended with `Failed to load task result: Expecting value` while
  loading `/tmp/task_result.json`, so the zero is a task-local export/result-plumbing failure rather than a clean
  spelling-task negative. Evidence:
  `.logs/takeover-20260727-b23/wave1/cuaworld-gcompris/train/missing_letter_spelling/sample_00`.
- `vscode/repair_cif_parser_library` (`other`) — live B23 terminal workflow fixed all five functional checks and
  `pytest` passed, but the hidden VLM gate required visible VSCode interaction even though the task text only asks to
  fix `/home/ga/workspace/cif_parser/`. The reward 0 is a verifier/spec mismatch, not a clean code-fix failure.
  Evidence: `.logs/takeover-20260727-b23/wave1/cuaworld-vscode/eval/repair_cif_parser_library/sample_00`.

### The 11 entries the two offline layers added

`hollow_artifact` (score under the forged probe, judge answering **no**; every one clears its own
verifier's pass gate and pays `reward=1.0`):

| task | score | what the forged file bought |
|---|---:|---|
| `dbeaver/nyc_311_chronic_noise` | 100 | "Could not find 'Address' column in output CSV" — then "Correctly identified 0 chronic offenders" on an empty CSV |
| `dbeaver/northwind_fraud_detection_benford` | 90 | "CSV row count incorrect (found 0, expected 9)" — then "All digit counts match ground truth perfectly" |
| `qblade/load_sample_project` | 90 | existence + "Project file has substantial content (4096 bytes)" is the whole check |
| `gvsig_desktop/clean_layer_schema` | 95 | garbage bytes parsed as a DBF header: "Record count valid (2667008169)", fields "successfully removed" because none are there |
| `gvsig_desktop/intersect_rivers_countries` | 80 | same DBF, `reward_type: weighted` → a pass is worth **1.0**, not 0.80 |
| `gretl/spectral_analysis_gdp_cycles` | 80 | "Spectral density values (Max: 399.0000) are consistent with a stationary growth rate series" — any numbers pass |
| `hec_ras/analyze_lateral_velocity_diff` | 80 | "CSV headers mismatch" + "CSV contains data (399 rows)" + "Workflow verification assumed passed" |
| `hec_ras/audit_channel_stability` | 70 | "Simulation results not found", "CSV headers incorrect", "Unknown station" — still 70 |
| `gretl/mle_normal_regression` | 70 | 30 points for `vlm_res.get("success")` (the judge was *reachable*, not right) + 40 for any file with 3 numbers in wide ranges = exactly the gate |
| `sumo/aggregate_routes_to_od_matrix` | 60 | "OD matrix not found or empty" + "WARNING: Ground truth generation failed. Granting partial credit for content." |

`setup_aborts`:

- `openlca/chemical_reaction_stoichiometry` — `setup_task.sh:2` is `set -e` and `:5` is
  `source /workspace/utils/task_utils.sh`. `install.sh::build` stages the software payload dirs
  that hooks are allowed to read, but it does **not** stage `utils/`; openlca's on-disk `utils/`
  holds a lone `.gitkeep`, so that path exists in **no image that has ever been built**. The hook dies before
  `launch_openlca` and before the initial screenshot. Invisible to the no-op sweep by construction:
  the expected agent-free baseline is 0 and the actual score is also 0.

## Not detectable by this sweep (keep in mind)

- **The `setup_rc` replay's raw rc is a candidate, not a verdict.** Its root is an empty desktop:
  bash, coreutils and `/workspace` are real, but nothing the software's image bakes is there.
  1451 of 3083 hooks exit non-zero in it, and nearly all of that is the absence talking (`ERROR:
  Source data missing: /opt/fits_samples/…`, `WARNING: Eclipse not detected` under `set -e`,
  `ERROR: GeoGebra not found!`) — all fine in the real image. Only an rc corroborated by a cause
  decidable **without** the image (today: sourcing a `/workspace` path `install.sh` never stages)
  becomes an exclude. To trust the rc itself, run the same driver inside
  `cua-lite/lite.cuaworld.<software>`; only the root changes, and the live no-op sweep already has
  the container open.
- **Setup aborts that need the image to reproduce.** One is known and NOT excluded:
  `openlca/waste_treatment_linkage_setup:20` calls `ensure_uslci_database > /dev/null` under
  `set -e`. The helper ends `echo ""; return 1`, and the image creates
  `~/openLCA-data-1.4/databases` empty (the USLCI zip is only dropped in `~/LCA_Imports/`, never
  imported), so it always returns 1. The adapter's helper guard (`_guard_helper_calls`) already
  neutralizes this whole class — it fixed the sibling tasks `openlca/corporate_esg_taxonomy` and
  `openlca/stoichiometric_process_modeling`, which is why they are not listed above — but its
  whole-line anchor excludes `>`, so a redirected call slips through. The fix belongs in that
  guard, not in this file.
- **Judge-gated hollow passes are NOT excluded.** 44 further tasks pass on a hollow artifact *once
  the judge is assumed to answer yes to everything* (28 gcompris, 6 slicer3d, 5 qblade, 1 each in
  gvsig_desktop / openemr / solvespace / ugene / vlc_media_player). That is not evidence of a
  defect — it is the ordinary shape of a VLM-judged task (a few existence points programmatically,
  the rest from the judge), and the premise it needs ("a real judge shown a blank 640×400 grey PNG
  says yes") is exactly what we have no evidence for. `validation_sweep.py` reports the list as a
  diagnostic (`judge_gated.json`) and never excludes it.
- **Verifiers that check the deliverable over the shell channel.** The `forged` layer answers
  `exec_capture` with the empty string; forging command *output* would be a guess about the app,
  not about the deliverable. Such verifiers are under-reported, never over-reported.
- **`geogebra` and `qgis` have never had the `live_noop` layer run on them.** They are now present
  in the file (offline layers only, and clean under both) and listed in
  `_meta._layers.live_noop.never_swept`, so the gap is recorded rather than invisible. Both are
  structurally sound — 150/150 registered tasks ship a parseable `verifier.py` with a resolvable
  target, none is an unconditional-pass stub, and none scores above 0 for a no-op host-side — but
  `nonzero_baseline` (52% of all findings) can only be measured in a container, so expect entries
  once the live sweep reaches them.
- **Data-provenance gaps** — e.g. slicer3d brats/amos/ircadb silently run on *synthetic* CT when a
  gated download fails. The pipeline "works" (clean 0) but the underlying data is wrong. No layer
  here can see this; judge from each task's `setup_task.sh`.
- **Gameability that needs a trivial wrong action** — `live_noop` tests the true no-op and `forged`
  tests a hollow deliverable; neither tests a cheap *wrong* one.
