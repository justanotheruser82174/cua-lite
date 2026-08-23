# lite.scalecua Upstream Issues

Record upstream data or judge defects here. Do not silently patch them in the
adapter.

## Template

- Task ID:
- Split:
- Source path:
- Symptom:
- Evidence:
- Adapter decision:
- `exclude_reason` if applicable:

## Current Known Policies

- Matching OSWorld eval tasks inherit `lite.osworld` `exclude_reason` exactly.
- Google auth and proxy support are out of scope for Phase 1/2.
- Non-dict setup/postconfig actions are import failures or typed exclusions.

## Issues

- Task IDs:
  - `scalecua_osworld_train_vs_code_53ad5833_3455_407b_bbc6_45b4c79ab8fb_task_verify_0`
  - `scalecua_osworld_train_vs_code_53ad5833_3455_407b_bbc6_45b4c79ab8fb_task_verify_1`
  - `scalecua_osworld_train_vs_code_53ad5833_3455_407b_bbc6_45b4c79ab8fb_task_verify_12`
  - `scalecua_osworld_train_vs_code_53ad5833_3455_407b_bbc6_45b4c79ab8fb_task_verify_23`
  - `scalecua_osworld_train_vs_code_53ad5833_3455_407b_bbc6_45b4c79ab8fb_task_verify_30`
  - `scalecua_osworld_train_vs_code_53ad5833_3455_407b_bbc6_45b4c79ab8fb_task_verify_31`
  - `scalecua_osworld_train_vs_code_53ad5833_3455_407b_bbc6_45b4c79ab8fb_task_verify_32`
  - `scalecua_osworld_train_vs_code_53ad5833_3455_407b_bbc6_45b4c79ab8fb_task_verify_33`
  - `scalecua_osworld_train_vs_code_53ad5833_3455_407b_bbc6_45b4c79ab8fb_task_verify_36`
  - `scalecua_osworld_train_vs_code_53ad5833_3455_407b_bbc6_45b4c79ab8fb_task_verify_37`
  - `scalecua_osworld_train_vs_code_53ad5833_3455_407b_bbc6_45b4c79ab8fb_task_verify_48`
- Split: `train`
- Source path:
  `osworld/generated_tasks/vs_code/53ad5833-3455-407b-bbc6-45b4c79ab8fb_task_verify_*.json`
- Symptom: The generated evaluator asks VS Code to write
  `/home/user/OpenFile.txt` through `GetOpenFile` or `OpenFile`, so visually
  correct states with `main.py` or `README.md` open still score 0.
- Evidence: A fixed-tag oracle probe on `_task_verify_23` left `main.py`
  visibly open in VS Code, but `OpenFile.txt` was never created. Inspecting the
  official bundled `vscodeEvalExtension` for this source id showed only
  `OpenProject`, `GetColorTheme`, and `GetBreakPoint` commands; there is no
  `GetOpenFile` / `OpenFile` implementation.
- Adapter decision: Do not add a ScaleCUA-only open-file getter. Filter these
  exact generated rows and let official/overlay VS Code getters handle any
  supported commands.
- `exclude_reason` if applicable: `upstream_generated_eval_bug`.

- Task ID: `scalecua_osworld_train_chrome_59155008_fe71_45ec_8a8f_dc35497b6aa8_task_verify_4`
- Split: `train`
- Source path: `osworld/generated_tasks/chrome/59155008-fe71-45ec-8a8f-dc35497b6aa8_task_verify_4.json`
- Symptom: The official generated evaluator expects exact URL
  `https://www.babycenter.com/baby-names/details/liam-5051`, but the live site
  reached during rollout used `https://www.babycenter.com/baby-names/details/liam-2820`.
- Evidence: visual audit row in
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-displaytimeout-instructionfilter.visual_audit.main_01.jsonl`
  labels this as `false_failure`; the corresponding ScaleCUA RL task family
  uses regex `babycenter\\.com/baby-names/details/liam`, which avoids numeric
  ID drift.
- Adapter decision: Do not rewrite the official expected URL or loosen
  `is_expected_active_tab`. Filter the affected generated exact numeric
  BabyCenter rows before rollout/export.
- `exclude_reason` if applicable: `upstream_live_site_drift` for the 20
  non-proxy generated exact numeric BabyCenter URL rows; 6 overlapping proxy
  rows keep `proxy_required`.

- Task IDs:
  - `scalecua_osworld_train_chrome_06fe7178_4491_4589_810f_2e2bc9502122_task_verify_30`
  - `scalecua_osworld_train_chrome_06fe7178_4491_4589_810f_2e2bc9502122_task_verify_31`
  - `scalecua_osworld_train_chrome_06fe7178_4491_4589_810f_2e2bc9502122_task_verify_32`
  - `scalecua_osworld_train_chrome_06fe7178_4491_4589_810f_2e2bc9502122_task_verify_33`
- Split: `train`
- Source path:
  `osworld/generated_tasks/chrome/06fe7178-4491-4589-810f-2e2bc9502122_task_verify_{30,31,32,33}.json`
- Symptom: These generated reopen-tab rows seed and expect Reddit, Twitter/X,
  and Facebook tabs, but the generated JSON omitted the `proxy` flag. Live
  social sites can redirect, block, or require proxy/login handling, and strict
  `is_expected_tabs` URL equality can then report reward 0 despite a plausible
  visual completion.
- Evidence: Rerun3 visual/code audit found `_30` as a false-failure/filter
  candidate, with no lite evaluator divergence. Sibling generated rows
  `_13.._29` and `_36.._45` already demonstrate that this family can contain
  exact proxy-flagged rows, while `lite.osworld` eval keeps the base
  `06fe7178` task runnable.
- Adapter decision: Filter only these four exact generated rows as
  `proxy_required` and set `metadata.others.proxy=true` during import. Do not
  exclude the whole OSWorld id, and do not synthesize ScaleCUA eval/eval_full
  exclusions because `lite.scalecua` exposes only `train` and `rl`; evaluation
  uses the canonical `lite.osworld` eval split.
- `exclude_reason` if applicable: `proxy_required`.

- Task ID: `scalecua_osworld_train_chrome_2888b4e6_5b47_4b57_8bf5_c73827890774_task_verify_34`
- Split: `train` plus related RL browse/filter rows
- Source path: `osworld/generated_tasks/chrome/2888b4e6-5b47-4b57-8bf5-c73827890774_task_verify_*.json`
- Symptom: The official generated/RL evaluators check Macy's product/listing URL
  filters, but `macys.com` product/listing pages return Akamai `Access Denied`
  in lite/osworld datacenter runs, matching the inherited lite.osworld eval
  blocking reason for Macy's.
- Evidence: hardened rollout
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000-postconfig-hardened`
  produced reward-0 rows where the agent either remained on Google Shopping or
  hit Macy's access-denied page. The same blocking class already appears in
  `lite.osworld` eval exclusions as a `block: macys.com product listing blocked
  by Akamai anti-bot ...` reason.
- Adapter decision: Do not rewrite official Macy's URL parser/metrics and do
  not add a synthetic local Macy's mirror. Filter generated/RL rows whose reward
  depends on live Macy's URL/filter state.
- `exclude_reason` if applicable: `upstream_live_site_drift` for 42 generated
  URL/filter rows and 3 RL browse/filter rows from OSWorld id
  `2888b4e6-5b47-4b57-8bf5-c73827890774`. The RL Macy's homepage bookmark row
  and unrelated Chrome downloads row remain runnable.

- Task ID: `scalecua_osworld_train_chrome_9f935cce_0a9f_435f_8007_817732bfc0a5_task_verify_15`
- Split: `train` plus 2 related RL DOJ Forms rows
- Source path: `osworld/generated_tasks/chrome/9f935cce-0a9f-435f-8007-817732bfc0a5_task_verify_*.json`
- Symptom: The official generated/RL evaluators expect historical DOJ Forms
  filter query ids such as `field_component_target_id=401`, but current DOJ
  Forms uses different component ids for the same visible division filters.
- Evidence: hardened rollout reached the current Antitrust forms page with
  `field_component_target_id=376`; the imported evaluator exactly matches the
  HF source JSON, so this is not importer mutation.
- Adapter decision: Do not rewrite DOJ expected query parameters or loosen the
  URL-pattern metric. Filter only rows from this OSWorld id whose expected
  rules contain `field_component_target_id=...`.
- `exclude_reason` if applicable: `upstream_live_site_drift` for 35 generated
  rows and 2 RL rows. Generic DOJ pages and U.S. Courts RL rows remain runnable.

- Task ID: `scalecua_osworld_train_chrome_a728a36e_8bf1_4bb6_9a03_ef039a5233f0_task_verify_38`
- Split: `train`
- Source path: `osworld/generated_tasks/chrome/a728a36e-8bf1-4bb6-9a03-ef039a5233f0_task_verify_*.json`
- Symptom: Several official Virginia DMV generated evaluators are pinned to
  old paths or regexes such as `licenses-ids/id/applying`,
  `licenses-ids/license/applying/fees`, `licenses-ids/license/renew`,
  `vehicles/registrations`, and old `new.*resident` URLs. Current rollout lands
  on restructured DMV pages such as `moving/new-virginia#title` and
  `licenses-ids/exams/road-skills-test`.
- Evidence: hardened rollout screenshots visually reached the intended DMV
  content while official URL regexes failed; imported evaluator fields match HF
  source JSON exactly. Direct URL probes confirmed representative old paths are
  404 or redirect to different current paths, while still-current paths such as
  `locations`, `vehicles`, `forms`, `licenses-ids/real-id`,
  `licenses-ids/cdl`, and driver-license eligibility are usable.
- Adapter decision: Filter only the verified stale DMV expected path fragments;
  keep broad/current DMV rows runnable.
- `exclude_reason` if applicable: `upstream_live_site_drift` for 40 generated
  rows.

- Task ID: `scalecua_osworld_train_chrome_a96b564e_dbe9_42c3_9ccf_b4498073438a_task_verify_67`
- Split: `train`
- Source path: `osworld/generated_tasks/chrome/a96b564e-dbe9-42c3-9ccf-b4498073438a_task_verify_*.json`
- Symptom: The official FlightAware generated evaluators for category pages
  expect historical `discussions.flightaware.com/c/...` category URLs. Current
  category slugs/ids redirect or no longer match the official expected URL.
- Evidence: hardened rollout reached FlightAware discussions but failed URL
  regex for the Feature Requests/Suggestions category. Direct probe showed a
  representative category URL redirecting to a different category slug.
- Adapter decision: Filter only generated rows from this OSWorld id whose
  expected rules contain `discussions.flightaware.com/c/`; latest/top views and
  thread URL rows remain runnable unless upstream marks them `proxy_required`.
- `exclude_reason` if applicable: `upstream_live_site_drift` for 13 generated
  rows.

- Task ID: `scalecua_osworld_train_chrome_c1fa57f3_c3db_4596_8f09_020701085416_task_verify_22`
- Split: `train`
- Source path: `osworld/generated_tasks/chrome/c1fa57f3-c3db-4596-8f09-020701085416_task_verify_*.json`
- Symptom: The official United special-assistance rows expect
  `united.com/en/us/fly/travel/special-needs`, while current navigation reaches
  `united.com/en/us/fly/travel/accessibility-and-assistance.html`.
- Evidence: hardened rollout server logs show the current United URL and the
  official expected regex mismatch; imported evaluator fields match HF source
  JSON exactly.
- Adapter decision: Filter only the `special-needs` expected-path rows from this
  OSWorld id. Other United pages remain runnable unless upstream marks them
  `proxy_required`.
- `exclude_reason` if applicable: `upstream_live_site_drift` for 6 generated
  rows.

- Task ID: `scalecua_osworld_train_gimp_3c8f201a_009d_4bbe_8b65_a6f8b35bb57f_task_verify_2`,
  `..._task_verify_11`, and `..._task_verify_18`
- Split: `train`
- Source path: `osworld/generated_tasks/gimp/3c8f201a-009d-4bbe-8b65-a6f8b35bb57f_task_verify_{2,3,5,9,11,14,16,18,22,28}.json`
- Symptom: The official generated GIMP instructions reference `provided URL` or
  `from the link` / `supplied link`, but the instruction contains no URL and
  setup only opens a terminal. During rollout the agent asked for the missing
  image URL or tried to search local files, and the evaluator then failed
  because the expected output file did not exist.
- Evidence: Official `osworld_eval/lib_run_single.py::run_single_example`
  passes only the `instruction` argument to the agent, and
  `desktop_env.DesktopEnv.reset` executes only JSON `config`. Official
  `osworld_env/worker/src/session.py::OSWorldSession` similarly sets
  `self.instruction = self.task_data["instruction"]` and prepends helper
  uploads only; it does not backfill `augmented_from`, metadata, or source
  trajectory fields. The imported rows match the HF source JSON exactly.
- Adapter decision: Do not rewrite generated instructions by inferring a
  sibling `kingbird.jpeg` URL. Filter the underspecified generated rows before
  rollout/export.
- `exclude_reason` if applicable: `instruction_setup_mismatch` for 10 generated
  GIMP rows from OSWorld id
  `3c8f201a-009d-4bbe-8b65-a6f8b35bb57f`.

- Task ID: `scalecua_osworld_train_chrome_aad10cd7_9337_4b62_b704_a857848cedf2_task_verify_12`
- Split: `train`
- Source path: `osworld/generated_tasks/chrome/aad10cd7-9337-4b62-b704-a857848cedf2_task_verify_12.json`
- Symptom: The official instruction says to access the button design guidelines
  `at the provided URL`, but the instruction contains no URL and setup only
  launches Chrome.
- Evidence: Same official instruction/setup flow as above; no hidden URL
  injection exists in official OSWorld eval or RL worker paths.
- Adapter decision: Do not synthesize an Apple HIG URL in the adapter. Filter
  this underspecified row before rollout/export.
- `exclude_reason` if applicable: `instruction_setup_mismatch` for 1 generated
  row.

- Task ID: `scalecua_osworld_train_gimp_e8172110_ec08_421b_a6f5_842e6451911f_task_verify_7`
  and related generated/RL image-reference rows.
- Split: `train` / `rl`
- Source path examples:
  `osworld/generated_tasks/gimp/e8172110-ec08-421b-a6f5-842e6451911f_task_verify_7.json`,
  `osworld/generated_tasks/gimp/d68204bf-11c1-4b13-b48b-d303c73d4bf6_task_verify_35.json`,
  `osworld/generated_tasks/gimp/d68204bf-11c1-4b13-b48b-d303c73d4bf6_task_verify_74.json`,
  `osworld/rl_tasks/gimp/58d3eeeb-e9d0-499f-962e-fd0db2a744d8_traj_verify_4.json`.
- Symptom: Official generated evaluators pass `source_path`,
  `original_path`, or `source_cache_path` into generated metrics, and those
  metrics open the string directly on the host. Some strings are VM paths such
  as `/home/user/Desktop/tilearray.png`; others are author-local paths such as
  `/home/lvbowen/project/AutoGen/src/envs/osworld_env/cache/.../character.png`.
  In CUA-Lite these host paths do not exist, so otherwise correct outputs can
  score 0 with `FileNotFoundError`.
- Evidence: Official `desktop_env.DesktopEnv.evaluate()` simply calls
  `result_getter`, `expected_getter`, then the metric; it does not materialize
  paths nested inside rule dictionaries. The official task config does download
  the assets into the VM, so this is not a setup download omission.
- Adapter decision: Keep these rows runnable. `lite.scalecua` materializes
  metric reference paths into the local eval cache before calling generated
  metrics. `/home/user/...` is fetched directly from the VM. Author cache paths
  are mapped by basename to likely VM asset locations, starting with
  `/home/user/Desktop`.
- `exclude_reason` if applicable: none for this class unless the VM asset cannot
  be fetched even after the configured candidates. Regression coverage:
  `oracle_gimp_author_cache_reference_flip_train_0013` passes with no-op reward
  0.0 and oracle reward 1.0.

- Task ID: `scalecua_osworld_train_libreoffice_impress_4c26e3f3_3a14_4d86_b44a_d3cedebbb487_task_verify_*`
- Split: `train`
- Source path: `osworld/generated_tasks/libreoffice_impress/4c26e3f3-3a14-4d86-b44a-d3cedebbb487_task_verify_*.json`
- Symptom: 4 official generated rows require
  `/home/lvbowen/project/AutoGen/results/task_verify_per5_1221/run_20251221_151547/reference_slide2_image.png`
  through `expected.rules.reference_path`, but that author-results file is not
  present in the public ScaleCUA snapshot, HF task snapshot, setup downloads, or
  VM postconfig output.
- Evidence: Neighboring generated variants materialize `/tmp/reference_slide2.png`
  during setup, while these rows do not transport or generate the referenced
  image. Import-time field scan catches only `reference_path` under
  `/home/lvbowen/project/AutoGen/results/`; author cache paths under
  `.../src/envs/osworld_env/cache/...` remain runnable because they can be
  mapped to VM assets.
- Adapter decision: Do not synthesize a basename match for an artifact that does
  not exist. Filter these rows before rollout/export.
- `exclude_reason` if applicable: `missing_reference_asset`.

- Task ID: `scalecua_osworld_train_libreoffice_calc_eb03d19a_b88d_4de4_8a64_ca0ac66f426b_task_verify_41`
- Split: `train`
- Source path: `osworld/generated_tasks/libreoffice_calc/eb03d19a-b88d-4de4-8a64-ca0ac66f426b_task_verify_41.json`
- Symptom: The official generated evaluator uses result range `B2:F5` and
  expected rule `row_idx: 3` for a descending sort by marks. That row includes
  the string label `Marks` plus numeric mark values, so the generated metric can
  compare a string label against numbers and return 0/error independent of the
  agent's final state.
- Evidence: The imported task cache matches the official evaluator. Code audit
  traced the generated getter/metric to
  `.cache/lite.scalecua_tasks/judge_functions/train/verigen_getters/calc.py`
  and `.../verigen_metrics/excel.py`; the defect is in the generated range and
  metric combination, not the lite adapter.
- Adapter decision: Filter this exact row. Do not change the shared Excel
  sorting metric because that would drift from official semantics and could
  alter other valid tasks.
- `exclude_reason` if applicable: `upstream_generated_eval_bug`.

- Task ID: `scalecua_osworld_train_libreoffice_impress_4ed5abd0_8b5d_47bd_839f_cacfa15ca37a_task_verify_26`
- Split: `train`
- Source path: `osworld/generated_tasks/libreoffice_impress/4ed5abd0-8b5d-47bd-839f-cacfa15ca37a_task_verify_26.json`
- Symptom: The generated title-format getter treats the first non-empty text
  shape on each slide as the title and then checks bold/underline state. This is
  a fragile upstream heuristic for slides where decorative or body shapes can
  precede the visual title.
- Evidence: Code audit found no lite adapter semantic change: the imported
  evaluator and official generated getter are used as-is. In the current source
  `4_1.pptx`, the heuristic appears to select the intended title shapes for
  slides 3, 5, and 6, so the available rollout evidence is not enough to call
  the reward-0 case a confirmed false failure.
- Adapter decision: Record as latent upstream fragility only. Do not filter or
  patch unless a replay/evaluator probe proves the visual title was correctly
  bold+underlined while the official getter still returns 0.
- `exclude_reason` if applicable: none.

- Task IDs:
  - `scalecua_osworld_train_libreoffice_writer_936321ce_5236_426a_9a20_e0e3c5dc536f_task_verify_3`
  - `scalecua_osworld_train_libreoffice_writer_72b810ef_4156_4d09_8f08_a0cf57e7cefe_task_verify_45`
  - `scalecua_osworld_train_libreoffice_writer_6a33f9b9_0a56_4844_9c3f_96ec3ffb3ba2_task_verify_0`
  - `scalecua_osworld_train_libreoffice_writer_0b17a146_2934_46c7_8727_73ff6b6483e8_task_verify_7`
- Split: `train`
- Source paths:
  - `osworld/generated_tasks/libreoffice_writer/936321ce-5236-426a-9a20-e0e3c5dc536f_task_verify_3.json`
  - `osworld/generated_tasks/libreoffice_writer/72b810ef-4156-4d09-8f08-a0cf57e7cefe_task_verify_45.json`
  - `osworld/generated_tasks/libreoffice_writer/6a33f9b9-0a56-4844-9c3f-96ec3ffb3ba2_task_verify_0.json`
  - `osworld/generated_tasks/libreoffice_writer/0b17a146-2934-46c7-8727-73ff6b6483e8_task_verify_7.json`
- Symptom: Visual rollout audit found plausible completed Writer states that
  scored reward 0. Code audit traced each case to a generated evaluator defect
  rather than a lite migration bug: strict whitespace/table text mismatch,
  inherited font size ignored by direct `run.font.size` checks, generated
  getter references to missing helper functions, and an exact-run underline
  check that fails when LibreOffice splits the target phrase into multiple
  runs.
- Evidence: The imported evaluator code matches the official generated task
  files; the failures are in task-specific generated getter/metric assumptions.
- Adapter decision: Filter these exact rows. Do not loosen shared Writer
  metrics or add heuristic effective-style inference in the migration adapter,
  because that would diverge from official generated semantics and can change
  valid rows.
- `exclude_reason` if applicable: `upstream_generated_eval_bug`.

## 2026-07-14 Expanded Generated-Eval Exact Filters

Code review plus subagent challenges expanded the exact
`upstream_generated_eval_bug` set from 5 to 48 generated rows. The filter
boundary is intentionally narrow: rows are excluded only when the official
generated evaluator is internally inconsistent with the task or generated code
shape. Migration gaps remain runnable and must be fixed in `lite.scalecua`.

Additional exact rows now filtered:

- Calc:
  - `scalecua_osworld_train_libreoffice_calc_04d9aeaf_7bed_4024_bedb_e10e6f00eb7f_task_verify_33`
  - `scalecua_osworld_train_libreoffice_calc_42e0a640_4f19_4b28_973d_729602b5a4a7_task_verify_24`
  - `scalecua_osworld_train_libreoffice_calc_42e0a640_4f19_4b28_973d_729602b5a4a7_task_verify_34`
  - `scalecua_osworld_train_libreoffice_calc_42e0a640_4f19_4b28_973d_729602b5a4a7_task_verify_36`
  - `scalecua_osworld_train_libreoffice_calc_7e429b8d_a3f0_4ed0_9b58_08957d00b127_task_verify_2`
  - `scalecua_osworld_train_libreoffice_calc_81c425f5_78f3_4771_afd6_3d2973825947_task_verify_22`
  - `scalecua_osworld_train_libreoffice_calc_881deb30_9549_4583_a841_8270c65f2a17_task_verify_71`
  - `scalecua_osworld_train_libreoffice_calc_f9584479_3d0d_4c79_affa_9ad7afdd8850_task_verify_10`
  - `scalecua_osworld_train_libreoffice_calc_f9584479_3d0d_4c79_affa_9ad7afdd8850_task_verify_12`
  These check missing/contradictory label requirements, impossible table header
  expectations, percentage scale mismatch, or fixed output cells/header
  placements that the instruction never specifies consistently. They are
  task-specific generated evaluator
  defects, not shared Calc metric fixes.

- Chrome-source Calc:
  - `scalecua_osworld_train_chrome_788b3701_3ec9_4b67_b679_418bfa726c22_task_verify_19`
  This generated row asks for `D=row B * row C` in `tally_book.xlsx`, but the
  official expected cached values are impossible for the transported source
  workbook: `B2*C2=2307.132` and `B6*C6=16184.96`, while the generated expected
  values are `2306.592` and `16184.0`.

- Chrome browser/VLC generated evaluator rows:
  - `scalecua_osworld_train_chrome_12086550_11c0_466b_b367_1d9e75b3910e_task_verify_28`
  - `scalecua_osworld_train_chrome_386dbd0e_0241_4a0a_b6a2_6704fba26b1c_task_verify_11`
  - `scalecua_osworld_train_chrome_386dbd0e_0241_4a0a_b6a2_6704fba26b1c_task_verify_8`
  - `scalecua_osworld_train_chrome_386dbd0e_0241_4a0a_b6a2_6704fba26b1c_task_verify_9`
  - `scalecua_osworld_train_chrome_48c46dc7_fe04_4505_ade7_723cba1aa6f6_task_verify_1`
  The first row asks for Chrome's privacy/security menu but generated eval
  accepts only `chrome://settings/privacy`; the second asks for a system-wide
  VLC stop shortcut but generated eval accepts only VLC's `vlcrc`
  `global-key-stop`; the two rerun4 hotkey rows call the same generated VLC
  metric path, which references missing helper `is_valid_keyboard_shortcut`;
  the bookmark row passes `is_expected_bookmarks` rules with URLs but without
  the required `type` field. Rerun3/rerun4 visual/code audit found no lite
  adapter divergence for these rows.

- GIMP:
  - `scalecua_osworld_train_gimp_7767eef2_56a3_4cea_8c9f_48c070c7d65b_task_verify_20`
  - `scalecua_osworld_train_gimp_7767eef2_56a3_4cea_8c9f_48c070c7d65b_task_verify_25`
  - `scalecua_osworld_train_gimp_7767eef2_56a3_4cea_8c9f_48c070c7d65b_task_verify_35`
  - `scalecua_osworld_train_gimp_d52d6308_ec58_42b7_a2c9_de80e4837b2b_task_verify_54`
  - `scalecua_osworld_train_gimp_d52d6308_ec58_42b7_a2c9_de80e4837b2b_task_verify_75`
  - `scalecua_osworld_train_gimp_e8172110_ec08_421b_a6f5_842e6451911f_task_verify_64`
  - `scalecua_osworld_train_gimp_227d2f97_562b_4ccb_ae47_a5ec9e142fbb_task_verify_43`
  - `scalecua_osworld_train_gimp_227d2f97_562b_4ccb_ae47_a5ec9e142fbb_task_verify_47`
  - `scalecua_osworld_train_gimp_7b7617bd_57cc_468e_9c91_40c4ec2bcb3d_task_verify_30`
  - `scalecua_osworld_train_gimp_7b7617bd_57cc_468e_9c91_40c4ec2bcb3d_task_verify_38`
  - `scalecua_osworld_train_gimp_b148e375_fe0b_4bec_90e7_38632b0d73c2_task_verify_{0,1,8,9,16,17,28,29,30,31,32,33,34,35}`
  The `227d2f97` rows call `compare_docx_images` with a result docx plus a
  rule dict, not the two-docx signature the metric expects. The `7b7617bd`
  rows use `check_config_status` on quoted/nested `gimprc` values that its
  whitespace parser cannot represent. The newly added `7767...{20,25}` and
  `d52d...{54,75}` rows came from rerun4 visual/code audit: visible GIMP state
  satisfied the instruction or was already true initially, while generated eval
  required hidden `gimprc`/`sessionrc` persistence. The `b148e375`
  layer-creation rows check the `layer-new-name` preference instead of actual
  XCF layer existence; this can pass or fail independently of the requested
  document state.

- Impress:
  - `scalecua_osworld_train_libreoffice_impress_841b50aa_df53_47bd_a73a_22d3a9f73160_task_verify_16`
  - `scalecua_osworld_train_libreoffice_impress_9cf05d24_6bd9_4dae_8967_f67d88f5d38a_task_verify_47`
  - `scalecua_osworld_train_libreoffice_impress_9cf05d24_6bd9_4dae_8967_f67d88f5d38a_task_verify_55`
  These evaluate speaker notes or the wrong 0-based slide index for instructions
  that describe visible slide content. The earlier suspected
  `04578141...task_verify_6` row is not filtered because the JSON/code does not
  prove a wrong shape target.

- VLC-source Impress:
  - `scalecua_osworld_train_vlc_778efd0a_153f_4842_9214_f05fc176b877_task_verify_26`
  This row operates on `Minimalist_Business_Slides.pptx` but expects
  `slide_count=7` after duplicating slide 2. Visual rollout and source
  inspection show the transported deck has 16 slides, so a correct duplicate
  produces 17 slides and the generated metric can award only the duplicate-text
  half.

- Writer:
  - `scalecua_osworld_train_libreoffice_writer_ecc2413d_8a48_416e_a3a2_d30106ca36cb_task_verify_67`
  This row asks to remove all page breaks, but the generated
  `contains_page_break` metric returns 0 when expected and actual counts are
  both zero. The related `ecc...task_verify_81` row is not filtered: it is a
  weak false-positive check for text position, but it does not cause correct
  completions to fail.

Rows explicitly not filtered:

- `0cecd4f3-74de-457b-ba94-29ad6b5dafb6` Calc sheet-name/order rows. Their
  official generated getter/metric names are wired correctly. The lite failure
  mode was direct host-side reads of `/home/user/...` in `(env, config)` getters,
  now handled by `lite.scalecua` direct VM-path materialization.
- `scalecua_osworld_train_libreoffice_impress_04578141_..._task_verify_6`.
  Available JSON/code do not prove the evaluator targets the wrong visible
  text shape.
- `scalecua_osworld_train_libreoffice_writer_ecc2413d_..._task_verify_81`.
  The generated evaluator is under-specified for “top of document” but remains
  a weak pass condition rather than a hard false failure.

## 2026-07-14 Rerun4 Instruction/Eval Mismatch Exact Filters

Visual audit plus source inspection found five exact generated Chrome-source
rows whose user-facing instruction asks for an explanatory/list/compare answer,
while the generated evaluator requires mutating hidden VLC state or reaching a
browser URL state. These are filtered with
`exclude_reason="instruction_eval_mismatch"` rather than patched in the
adapter:

- `scalecua_osworld_train_chrome_386dbd0e_0241_4a0a_b6a2_6704fba26b1c_task_verify_17`:
  instruction says "Guide me in turning off the playlist sidebar in VLC";
  evaluator checks `vlc_config` rule `expected_playlist_tree=0`.
- `scalecua_osworld_train_chrome_f3977615_2b45_4ac5_8bba_80c17dbe2a37_task_verify_66`:
  instruction asks how to disable automatic subtitle loading; evaluator checks
  `vlc_config` rule `expected_subtitle_autoload=0`.
- `scalecua_osworld_train_chrome_f3977615_2b45_4ac5_8bba_80c17dbe2a37_task_verify_91`:
  instruction asks how to disable the VLC background cone display; evaluator
  checks `vlc_config` rule `expected_qt_bgcone=0`.
- `scalecua_osworld_train_chrome_f5d96daf_83a8_4c86_9686_bada31fc66ab_task_verify_22`:
  instruction asks to list and contrast iPhone 13 Pro Max and iPhone 12 Pro Max
  features/specifications; evaluator checks an Apple compare URL query
  `modelList` state.
- `scalecua_osworld_train_chrome_f5d96daf_83a8_4c86_9686_bada31fc66ab_task_verify_51`:
  instruction asks to compare iPhone 14 Pro Max and iPhone 13 Pro Max
  features; evaluator checks an Apple compare URL query `modelList` state.

Evidence came from rerun4 reward-0 visual batches and code-side audit of
`lite/gym/envs/lite/scalecua/data/train.jsonl` source paths. Neighboring rows from
the same OSWorld ids remain runnable unless separately filtered; this is an
exact task-id filter, not a family-level drop.

## 2026-07-14 Rerun5 Visual/Code Exact Filters

Two additional generated Chrome rows were confirmed during rerun5 visual audit
and code-side comparison against official SCALE-CUA evaluator code:

- `scalecua_osworld_train_chrome_06fe7178_4491_4589_810f_2e2bc9502122_task_verify_34`
  uses `exclude_reason="upstream_live_site_drift"`. The intended GitHub,
  StackOverflow, and GitLab tabs are visually restored, but live redirects
  change the evaluated URLs (`about.gitlab.com`,
  `stackoverflow.com/questions`) while the generated evaluator expects exact
  historical domains.
- `scalecua_osworld_train_chrome_480bcfea_d68f_4aaa_a0a9_2589ef319381_task_verify_39`
  uses `exclude_reason="upstream_generated_eval_bug"`. The generated evaluator
  expects old Chrome flag ids `tab-groups` and `tab-groups-collapse`; the
  runtime Chrome exposes current related Tab Groups flag ids, so a broad alias
  would hide path-sensitive flag tasks.

## 2026-07-15 RL Oracle Replay Exact Filter

Oracle replay of `rl_auto_os_files_b2` found one exact upstream RL evaluator
bug:

- `scalecua_osworld_rl_os_5c1075ca_bb34_46a3_a7a0_029bd7463e79_traj_verify_1`
  uses `exclude_reason="upstream_generated_eval_bug"`. The instruction asks to
  delete every `.ipynb` file whose filename does not contain `failed`, and the
  expected rules require `a_failed.ipynb`, `c_failed.ipynb`, and
  `e_failed.ipynb` to remain while excluding `d.ipynb`. Official
  `check_include_exclude` is substring-based, so each required
  `*_failed.ipynb` filename necessarily contains the excluded substring
  `d.ipynb`; reward `1.0` is unreachable under the official evaluator.

- `scalecua_osworld_rl_libreoffice_writer_0a0faba3_5580_44df_965d_f562a99b291c_traj_verify_6`
  uses `exclude_reason="upstream_generated_eval_bug"`. Oracle replay produced
  all non-empty paragraphs with a tab stop at `10.0cm` and getter alignment
  values of `RIGHT (2)`, but the official generated metric accepts only the
  literal strings `RIGHT` or `END`. This is a task-specific UNO enum
  normalization bug in the generated metric, not a lite migration/runtime
  defect, so the row is filtered instead of patching shared evaluator behavior.

## 2026-07-16 RL Impress Tail Exact Filters

Strict oracle closure of the remaining RL Impress tail found eight exact rows
that should be filtered with `exclude_reason="upstream_generated_eval_bug"`.
They are exact task-id filters; neighboring Impress rows remain runnable and
must keep oracle coverage.

- `scalecua_osworld_rl_libreoffice_impress_ef9d12bd_bcee_4ba0_a40e_918400f43ddf_traj_verify_{1,2,4,5}`:
  no-op precheck already returns nonzero reward (`1.0`, `0.5`, `1.0`,
  `1.0`) in the imported fixedbase state, so these UI-state rows cannot pass
  the strict no-op-negative oracle gate.
- `scalecua_osworld_rl_libreoffice_impress_550ce7e7_747b_495f_b122_acdc4d0b8e54_traj_verify_{4,5,6}`:
  the generated evaluator reads non-public `python-pptx` `Font.strike` state
  for strikethrough checks; replay cannot produce a trustworthy official
  reward-1 oracle without changing generated evaluator semantics.
- `scalecua_osworld_rl_libreoffice_impress_af2d657a_e6b3_4c6a_9f67_9e3ed015974c_traj_verify_3`:
  the generated evaluator imports `PP_PLACEHOLDER` from `pptx.util`, where it
  is unavailable, so the official generated metric is broken independently of
  the lite migration.

## 2026-07-15 Budget.com Live Access Restriction

- `scalecua_osworld_rl_chrome_47543840_672a_467d_80df_8f7c3b9788c9_traj_verify_{0,1,2}`
  use `exclude_reason="upstream_live_site_drift"`. Visual rollout reached
  `budget.com`, but the live site returned an access-restricted/security
  verification page before the expected Boston Logan rental-car search and
  price-sort result URLs could be reached. This is exact-filtered at the RL
  task-id level; `traj_verify_{3,4}` remain `proxy_required`, and unrelated
  tasks that mention budgets are not filtered.

## 2026-07-15 Shenzhen Live Address Lookup Family Filter

- `7ff48d5b-2df2-49da-b500-a5150ffc7f18` generated `train` rows use
  `exclude_reason="upstream_live_site_drift"` for all 60 generated variants.
  The source family asks the agent to search current Shenzhen public-service,
  hospital, park, library, metro, mall, school, post-office, police-station,
  and visa-endorsement-terminal locations, while the generated evaluators
  compare against hard-coded historical address/name lists in
  `AllLocations.docx`. Visual rollout confirmed a representative Baoan
  endorsement-terminal row produced plausible current web-search addresses
  that failed only because the expected addresses are stale. The three
  `rl_tasks` rows for the same OSWorld id are static Word-edit tasks and remain
  runnable; they must be covered by oracle actions rather than excluded.

## 2026-07-15 Generated OS Script/Evaluator Mismatch

- `scalecua_osworld_train_os_6f56bf42_85b8_4fbb_8e06_6c44960184ba_task_verify_{43,44,46,47}`
  use `exclude_reason="upstream_generated_eval_bug"`. These generated rows ask
  for `report.txt` copies in `archive1`, `archive2`, and `archive3`, and their
  expected rules look for `Success: report.txt exists in archive*`. However the
  official postconfig still downloads the original OSWorld `eval.sh`, which
  checks for `file1` in `dir1`, `dir2`, and `dir3`. A fixed-tag oracle probe
  that created the requested `report.txt/archive*` state returned reward `0.0`
  with debug output `Failure: file1 does not exist in dir1.`, so this is an
  upstream generated task/evaluator mismatch rather than a lite migration bug.
