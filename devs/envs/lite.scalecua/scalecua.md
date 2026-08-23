# lite.scalecua OSWorld Migration Plan

Status: implementation exists. RL `oracle_actions`/`exclude_reason` closure is
closed for the current imported `rl` catalog; large-scale visual rollout remains
a bug-discovery and fix-verification lane and must not become a separate
scorekeeping project.

Operating rule: rollout and oracle validation are not general implementation
work queues. Rollout is used only to expose migration/runtime/setup/eval bugs,
verify scoped fixes, or justify exact `exclude_reason` filters after visual and
code-side triage. Oracle actions have durable value beyond triage: a passing
oracle becomes a regression asset that prevents future drift, while a failing
oracle is potential bug evidence until it is classified as fixture error,
migration/runtime bug, upstream task defect, or transient infrastructure
failure. Do not do speculative refactors, broad cleanup, or documentation
bookkeeping while known rollout/oracle failures still need owner-layer fixes;
update docs only to encode the rule, evidence, or command that prevents the
same mistake from recurring.

Priority order:

1. **RL bug closure is the primary objective.** Every official `rl_tasks` row
   must be classified by oracle/no-op evidence:
   - correct task: add executable `oracle_actions` or `oracle_trajectory`;
   - upstream ScaleCUA task/eval defect: add exact `exclude_reason` and record
     evidence;
   - fixable runtime or migration bug: fix the owning layer first, then add
     executable oracle evidence.
2. **Rollout exists only to find and validate bugs/fixes.** Reward summaries are
   only triage queues. A rollout failure or suspicious success must become one
   of: owner-layer fix with targeted rerun/oracle evidence, exact upstream
   `exclude_reason`, transient rerun, or confirmed model failure. Rollout does
   not replace RL oracle closure and must not turn into restart churn.
3. **Oracle actions are both coverage and bug probes.** A passing oracle should
   be promoted or source-backed so the task is protected from future drift. A
   failing oracle is not discarded as a bad score; it starts a concrete
   classification loop: fixture mistake, ScaleCUA migration bug, shared
   OSWorld runtime/container parity bug, upstream task/eval defect, or
   transient infrastructure failure.
4. **Fix ownership is part of the RL closure decision, not a later cleanup.**
   - VM/container parity gaps are capabilities supported by the official
     OSWorld VM/runtime but missing from the `lite.osworld` container. Fix
     these in the `lite.osworld` Dockerfile or shared OSWorld runtime, with
     `lite.osworld` regression evidence.
   - Apply the issue-116 agent/env split to every fix. Agent-facing CLI and
     Python behavior must keep the official guest/user-terminal surface.
     Env-facing setup/eval dependencies belong in `/opt/env/venv` or
     `/opt/env/bin`, reached only by the env runner/server path. Do not repair
     evaluator import failures by making generic `python`, `python3`, `pip`, or
     `pip3` resolve differently for the agent.
   - ScaleCUA migration bugs belong in `lite.scalecua`: importer normalization,
     setup transport, DesktopEnv shim behavior, generated judge overlays,
     profile/path/material transport, and ScaleCUA-specific eval compatibility.
   - Upstream defects are not patched around. They get exact `exclude_reason`
     filters and issue/gap evidence.
5. **No independent fix stream.** Aside from fixes surfaced by oracle/no-op
   replay, rollout visual triage, or the official-code parity probe needed to
   explain those failures, do not change implementation semantics. Static tests,
   coverage inventory, and docs are verification/supporting work, not separate
   reasons to keep editing code.

Unchecked validation items in this document mean missing local evidence, not
open design questions. A gate closes only when the exact command, result counts,
date-relevant artifact path, and failure disposition are recorded.

Do not treat Python migration, generated-oracle, or fixture-generation code as
mechanical low-risk porting. These scripts encode the hidden contract between
official ScaleCUA tasks, official OSWorld VM behavior, CUA-Lite container
runtime, setup transport, evaluator stdout/file/path semantics, desktop/CDP
state, and replay evidence. A one-line path, quoting, shell, profile, or
postconfig-order mismatch can turn a solvable task into reward `0`, or create a
false reward `1`. Therefore every new generator or migration helper needs
source-backed registration, byte-lock checks, no-op/oracle replay canaries, and
owner-layer classification for failures.

Evidence vocabulary:

- **Workspace-local fixture** means a JSONL fixture row exists in this checkout.
  It is useful development evidence, but it is not release-countable by itself.
- **Source-backed fixture** means the row is generated from a registered
  `src/gen/oracle` shard and `python -m lite.gym.envs.lite.scalecua.src.gen.oracle
  --check` byte-locks it inside `data/oracle/rl.jsonl` or
  `data/oracle/train.jsonl`.
- **Release-countable oracle evidence** requires all of: non-excluded catalog
  row, promoted fixture or generated source, no-op precheck reward at the
  documented negative score, oracle replay at the documented positive score,
  screenshots/debug payloads, and clean `coverage_inventory.py` membership with
  no duplicate or missing-task problems.
- **Verified oracle row** means one fixture row has passed the full no-op
  negative and oracle-positive replay path through
  `devs/envs/lite.scalecua/validate/oracle/validate.py`, with per-fixture
  report JSONL, screenshots, trace, and eval debug payloads. Coverage inventory
  rows, source-backed generation, and canary-only replay are not verified
  oracle rows.
- **Rollout evidence** is bug-discovery evidence until every reward/screenshot
  mismatch is classified into migration fix, exact upstream `exclude_reason`,
  transient rerun, or model failure.
- **Passing oracle evidence** is drift-prevention evidence: keep the fixture,
  screenshots, debug payloads, and generator source so future setup/eval/runtime
  changes can be replayed.
- **Failing oracle evidence** is a bug probe: classify and act on it before
  treating the task as covered.

Hard gate: every oracle action in a promoted fixture, registered/source-backed
shard, `lite/gym/envs/lite/scalecua/data/oracle/` row, or final RL closure
count must become a verified oracle row before release closure. In-flight
unverified JSONL may exist only as active development/replay material; it does
not count, must stay visible in `verified_inventory.py`, and must be resolved
before a release checkpoint by one of three outcomes: no-op negative plus
oracle-positive replay passes, the row is moved back to a diagnostic
artifact path, or the task is fixed/exactly excluded. The final closure
denominator is not `coverage_inventory.py.fixture_rows`; it is the set of
non-excluded RL rows with release-countable no-op plus replay evidence.
Candidate JSONL not being actively replayed must stay outside `data/oracle/`,
because the inventory intentionally scans that directory and would otherwise
overstate coverage.

This document replaces the previous running notes. It is the governing plan for
migrating the OSWorld portion of `SCALE-CUA` into CUA-Lite as
`lite.gym.envs.lite.scalecua`. It is intentionally modeled after
`redesign.md` (refactor branch): canonical decisions
first, validation evidence rules before checklists, migration as owner-slice
milestones, and convergence defined separately from current progress.

Reference roots:

- Upstream ScaleCUA checkout: `SCALE-CUA/`
- Shared OSWorld runtime/env reference: `lite/gym/envs/lite/osworld/`
- OSWorld development and validation reference: `devs/envs/lite.osworld/`
- ScaleCUA runtime package: `lite/gym/envs/lite/scalecua/`
- ScaleCUA development package: `devs/envs/lite.scalecua/`

## 0. RL North Star And Closure Decision Tree

The official `rl_tasks` split is the release-critical universe. This section is
intentionally before catalog/image/runtime details because every downstream
decision is subordinate to RL task closure.

Current RL universe after fresh import:

- Total RL rows: 2,049
- Excluded RL rows: 210
- Runnable RL rows: 1,839
- Excluded reason breakdown: `infeasible=18`, `proxy_required=102`,
  `unsupported_schema:action_list=5`, `upstream_generated_eval_bug=63`,
  `upstream_live_site_drift=22`.

Runnable RL by canonical catalog domain:

| Domain | Runnable | Workspace-local oracle fixtures by catalog domain |
|---|---:|---:|
| `chrome` | 152 | 152 |
| `gimp` | 169 | 169 |
| `libreoffice_calc` | 242 | 242 |
| `libreoffice_impress` | 237 | 237 |
| `libreoffice_writer` | 161 | 161 |
| `multi_apps` | 340 | 340 |
| `os` | 165 | 165 |
| `thunderbird` | 110 | 110 |
| `vlc` | 129 | 129 |
| `vs_code` | 134 | 134 |
| Total | 1,839 | 1,839 |

The catalog domain is the only domain used for RL closure, rollout
stratification, and coverage accounting. Current workspace-local RL fixture
rows have zero fixture-domain/catalog-domain mismatches; if this drifts, treat
it as local fixture metadata drift, not as a different canonical domain split.

Required closure decision for every official RL row:

1. If the task is correct and supported, add executable `oracle_actions` or
   `oracle_trajectory` and replay it through production setup/eval.
2. If the task has an upstream ScaleCUA task/eval defect, add an exact
   `metadata.others.exclude_reason` and record evidence in upstream issue/gap
   docs.
3. If the task exposes a fixable gap, fix the correct owner layer first:
   - official VM/runtime behavior missing from the `lite.osworld` container:
     fix `lite.osworld` Docker/runtime and add OSWorld regression evidence;
   - ScaleCUA migration/setup/eval/judge/material transport bug: fix
     `lite.scalecua` and add ScaleCUA tests/evidence.
4. After a fix, the task returns to case 1 and needs executable oracle evidence.

There is no third class of runnable RL task without oracle evidence.

Current RL fixture/candidate inventory:

- `1,839 / 1,839` runnable RL tasks have workspace-local, source-backed oracle
  fixtures.
- Current strict artifact-backed replay evidence covers `1,839 / 1,839`
  workspace-local RL fixtures and `1,839 / 1,839` runnable RL catalog rows.
  The RL fixture replay gate is closed for the current import: unverified
  catalog rows `0`, unverified fixture rows `0`, fixture problems `0`, and
  artifact-missing strict rows `0`.
- Candidate backlog:
  `.exps/validate/lite.scalecua/oracle/candidates/rl_full.candidates.current.jsonl`
  contains one row per non-excluded RL catalog row and is regenerated after
  import changes. Candidates are
  not fixtures until actions/trajectories are implemented and replayed.
- Full RL release evidence is closed for the current imported catalog. Future
  import/source changes reopen the gate until the same `verified_inventory.py`
  closure result is reproduced.

Train/generated tasks remain in scope for large rollout and long-tail
regression, but they do not replace the RL closure gate. Rollout visual review
is a bug-fix evidence lane alongside RL closure work; it finds systematic bugs
and false success/failure patterns, but it does not substitute for
oracle/exclude closure.

Fast entry points:

| Question | Source of truth |
|---|---|
| What is the RL north star? | Section 0 |
| What is being migrated? | Section 1 |
| What are the canonical splits, pins, and current counts? | Section 2 |
| Where do importer/setup/eval/oracle responsibilities live? | Sections 3-4 and 6 |
| How are task fixes classified? | Section 5 |
| What validation is mandatory? | Sections 6-7 |
| How does large-scale rollout run? | Section 7.6 |
| What is the milestone plan? | Section 8 |
| When is this converged? | Section 10 |

### 0.1 Release Gates And Parallel Validation Summary

Release closes only when all primary and secondary gates below are satisfied.
Detailed commands remain in Section 7; this summary exists so later work does
not mistake current artifacts for completion.

| Gate | Close condition |
|---|---|
| RL closure | All 2,049 official RL rows are classified; every non-excluded RL row has release-countable oracle evidence, and every unsupported row has exact `exclude_reason` evidence. |
| Import/static | Fresh `install.sh provision`, `validate/static.py`, registry tests, and focused ScaleCUA tests pass against the pinned sources in Section 2. |
| Oracle replay | Every oracle action in promoted, source-backed, registered, or `data/oracle/` RL fixture rows passes no-op negative eval and oracle replay through production setup/eval with screenshots/debug payloads, or the row is moved out of release fixture paths. |
| Rollout bug-fix evidence | One clean 1000-task gpt-5.5 root plus one independent confirmation root run on fresh env-servers at concurrency 8; every trajectory is visually audited, zero false success/failure remains unresolved, and overall/train/RL visual success is at least 70% after migration/runtime/upstream defects are fixed or excluded. |
| Cleanup/drift | Direct-mode validators and env-server rollouts leave no orphan containers; README/dev docs/tests agree with this plan. |

Parallel validation lanes:

- Oracle generation/replay, rollout execution, visual triage, and official-code
  parity audits run in parallel when their write sets are disjoint, but parity
  audits exist to explain rollout/oracle failures and prevent regressions.
- Oracle work does not wait for rollout completion. A replay failure is triage
  input for a fixture bug, setup/eval migration bug, upstream mismatch, or
  unsupported task classification.
- Rollout work does not restart repeatedly for scoped fixes. Continue the
  current batch unless stale semantics would materially invalidate most
  remaining trajectories; rerun affected tasks separately after fixes.
- Subagents should be assigned bounded slices: domain oracle generation,
  official ScaleCUA parity audit, visual root-cause triage, or stale-doc audit.
  They must not duplicate the same unresolved slice.

### 0.2 Current Checkpoint And Active Blockers

Current source facts:

- Import artifact:
  `.cache/lite.scalecua_tasks/import_report.json`, sha256
  `7ae43cbe63e0e1d97c0d695df45103a3a9ceea313edf6a7883dfab08be0563e3`.
- Asset identity:
  `.cache/lite.scalecua_tasks/.asset_identity`, sha256
  `64920ffe12dc606114ec78bbbade4174ed83415544d703459a95dd26b083ec16`.
- RL coverage inventory:
  `.exps/validate/lite.scalecua/oracle/coverage/rl.coverage.json`.
- Train+RL coverage inventory:
  `.exps/validate/lite.scalecua/oracle/coverage/current.coverage.json`.
- RL verified oracle inventory:
  `.exps/validate/lite.scalecua/oracle/verified/rl.verified.closed_20260716.json`,
  sha256
  `aa1d76cff43188cbd0b1ebb1bb8ac0649c1cb18109cd764b9d91c9cc3eee5f6e`.
- Train+RL verified oracle inventory:
  `.exps/validate/lite.scalecua/oracle/coverage/current.verified.json`.

Active blockers before release:

- RL closure is not an active blocker for the current import: `1,839 / 1,839`
  runnable RL catalog rows have strict artifact-backed oracle evidence.
- Keep every promoted RL fixture replay-verified. Any new `data/oracle` row or
  import/source change reopens the RL gate until no-op and oracle-positive
  replay pass and `verified_inventory.py --require-artifacts` returns zero
  unverified catalog and fixture rows.
- Keep candidate-selection artifacts refreshed after every import before citing
  them as current coverage denominators.
- Complete one clean 1000-task gpt-5.5 rollout root plus one independent
  confirmation root with visual audit and root-cause classification.
- Repair linked `devs/envs/lite.scalecua` docs when they contain stale local
  counts. This document's Section 2 and Section 6 counts are the current
  checkpoint until a new coverage/import refresh is recorded.

Resolved at this checkpoint, not active blockers:

- Sixty-six RL source-backed oracle shards are registered in
  `lite.gym.envs.lite.scalecua.src.gen.oracle.__main__.SHARDS`, including
  Office, GIMP/Thunderbird, multi-app, OS/files, OS/VS Code tail,
  Thunderbird-source, VS Code, browser, Chrome, VLC, and URL seed shards.
  `python -m
  lite.gym.envs.lite.scalecua.src.gen.oracle --check` passes for all of them.
- `import_report.normalization_notes` is classified:
  `active_setting_noop=2` maps to
  `scalecua_osworld_train_libreoffice_writer_f178a4a9_d090_4b56_bc4c_4b72a61a035d_task_verify_95`,
  where the unsupported postconfig-only `active_setting` probes are normalized
  to waits and the metric reads LibreOffice registry state;
  `get_tabs_info_noop=1` maps to excluded
  `scalecua_osworld_train_chrome_58565672_7bfe_48ab_b828_db349231de6b_task_verify_30`
  with `unsupported_schema:evaluator_postconfig_query_config`;
  `drop_stale_rl_a462_charles_setup=7` maps to the seven RL
  `scalecua_osworld_rl_os_a462a795_fdc7_4b23_b689_e8b6df786b78_traj_verify_{0..6}`
  source rows: `traj_verify_0` and `traj_verify_1` are exact
  `upstream_generated_eval_bug` excludes for stdout-newline-sensitive
  `exact_match` evaluators, while the five remaining runnable rows have stale
  inherited Charles-login setup removed and source-backed oracle fixtures from
  `src/gen/oracle/domains/os.py` in `data/oracle/rl.jsonl`;
  `gimp_export_full_path_postconfig=231` rewrites
  generated GIMP export postconfig to deterministic full output paths matching
  the evaluator result path; `legacy_top_level_action_parameters=6` preserves
  official ScaleCUA legacy action payloads whose fields lived at the action
  top level instead of under `parameters`; `local_placeholder_image_download=1`
  replaces a missing external placeholder image with a local materialized
  placeholder; `normalize_update_desktop_database=16` normalizes bare desktop
  database updates into supported execute semantics; and
  `repair_root_home_test1_setup=22` repairs the known `/home/test1` setup family
  so the official evaluator's user/home checks run under sudo-compatible
  container semantics.

### 0.3 Stale, Local, And Non-Release Evidence

These artifacts are useful for development but cannot be cited as release
completion:

- Per-batch `lite/gym/envs/lite/scalecua/data/oracle/*.jsonl` files outside
  `rl.jsonl` and `train.jsonl`. They are workspace-local fixtures until
  promoted/source-backed and replayed into the canonical aggregate files.
- Unregistered `src/gen/oracle/*.py` modules. Local static fixes or canary-only
  success do not count until the shard is registered, regenerated, byte-locked,
  and replayed.
- Candidate JSONL and coverage reports. They are selection/inventory artifacts,
  not executable oracle evidence.
- Old diagnostic `gpt-5.5-1000-*` rollout roots. They become bug-fix evidence
  only after reclassification against the current catalog, fresh env-server
  root, full completion, visual audit, and known-bug disposition.
- Raw rollout rewards. Reward `1` can be false positive and reward `0` can be a
  migration/eval bug; screenshots plus code-side root-cause evidence decide.

### 0.4 Fix Ownership Snapshot

The detailed taxonomy remains in Section 5. The short rule used during triage:

- If the official OSWorld VM/runtime supports a capability and the shared
  `lite.osworld` container lacks it, fix `lite.osworld` Docker/runtime and add
  `lite.osworld` regression evidence.
- User-facing VM CLIs that an agent may reasonably call, such as
  `timedatectl`, `hostname`, and `powerprofilesctl`, belong on the normal guest
  PATH (`/usr/local/bin` or `/usr/bin`) with VM-compatible behavior. Env-facing
  helper tools used only by setup/eval/backend plumbing belong under
  `/opt/env/bin` or equivalent hidden env paths. Do not put a user-facing VM
  parity command in `/opt/env/bin`.
- Python/package fixes follow the same issue-116 split. Evaluator/setup Python
  is env-facing and should run from `/opt/env/venv/bin/python`; its packages
  must be installed into that venv at image build or env install time. Agent
  terminal `python3`, `pip`, and `pip3` stay system/guest-facing. If an
  official postconfig says `sudo pip install ...` but the result command runs
  env-facing `python`, do not add a pip wrapper or redirect `sudo pip`; either
  preinstall the true evaluator dependency into `/opt/env/venv`, or classify the
  row as an upstream incompatibility if the dependency is task/user-facing.
- Missing VM base packages assumed by official tasks/evaluators, such as
  `bc`, belong in the shared `lite.osworld` Dockerfile when they are part of
  the official VM surface. System-level desktop session parity, such as
  shell/gsettings/DBus behavior, belongs in shared `lite.osworld`
  runtime/supervisor once it is proven to be a general OSWorld capability.
- If the gap is ScaleCUA importer, setup transport, generated judge,
  DesktopEnv shim, profile/path/material transport, or XCF/generated-eval
  compatibility, fix `lite.scalecua`.
- If official ScaleCUA task/eval is internally wrong or depends on unavailable
  live/external state, add an exact `exclude_reason` and document evidence.
- If screenshots and code show the task/eval are correct but the agent failed,
  keep it as model failure and do not change task/eval semantics.

## 1. Goals And Boundaries

### 1.1 Goals

1. Migrate ScaleCUA OSWorld training tasks into CUA-Lite as
   `lite.gym.envs.lite.scalecua`, with runtime splits `train` and `rl` only.
   Evaluation-set work should use `lite.osworld`'s canonical `eval` split.
2. Reuse the `lite.osworld` runtime substrate and emulate the official
   ScaleCUA/OSWorld service surface on that container substrate. The normal
   lifecycle uses `cua-lite/sandbox.linux:latest` and
   `cua-lite/lite.osworld:latest`; install/build/validation docs must not rely
   on fixed image tags.
3. Preserve strict setup and eval transport: unsupported setup/postconfig shapes
   are excluded or fixed explicitly, not silently ignored.
4. Preserve official ScaleCUA train/RL scoring semantics, including raw partial
   rewards and official multi-metric aggregation, rather than the binary
   `lite.osworld` eval threshold.
5. Make the official `rl_tasks` split the release north star: every RL row is
   either exactly excluded or covered by executable oracle evidence.
6. Keep generated task catalogs deterministic and cache-backed. Runtime code
   reads imported JSONL catalogs, never raw HF JSON.
7. Build oracle actions by first studying `lite.osworld` oracle recipes and
   adapting the operational shape to the current ScaleCUA evaluator target.

### 1.2 Non-Goals

- Do not migrate the ScaleCUA training stack, RL worker/controller, agent
  implementations, monitor UI, or non-OSWorld suites.
- Do not create a new `lite.scalecua` Docker image in the current phase.
- Do not mirror the full ScaleCUA dataset into a new `lite.scalecua-assets`
  repository unless a later design change explicitly accepts that cost.
- Do not hide upstream task/eval defects with adapter patches.

### 1.3 Baseline Evidence Rule

Before a milestone closes, its owner must classify each discovered behavior
against these evidence sources:

- Official ScaleCUA OSWorld flow:
  `SCALE-CUA/osworld_env/worker/src/task.py`,
  `SCALE-CUA/osworld_env/worker/src/environment.py`,
  `SCALE-CUA/osworld_eval/run_scalecua_os.py`,
  `SCALE-CUA/osworld_eval/lib_run_single.py`, and
  `SCALE-CUA/osworld_eval/desktop_env/desktop_env.py`.
- Official ScaleCUA eval examples may be used for code-side parity research, but
  they are not imported into `lite.scalecua`; evaluation-set tasks come from
  `lite.osworld`.
- Shared CUA-Lite OSWorld runtime and evaluator:
  `lite/gym/envs/lite/osworld/`.
- OSWorld oracle and rollout validation process:
  `devs/envs/lite.osworld/validate/oracle/` and
  `devs/envs/lite.osworld/validate/rollout/`.
- ScaleCUA migration/runtime code:
  `lite/gym/envs/lite/scalecua/` and `devs/envs/lite.scalecua/`.

Each behavior must be one of:

- required product invariant;
- redesigned equivalent with evidence;
- exact upstream defect with `exclude_reason`;
- unsupported runtime/material dependency recorded in `gaps.md`;
- deliberate retirement with a short impact note.

Unclassified behavior blocks release because omission is not a design decision.

### 1.4 Execution Discipline

- Use subagents aggressively for disjoint work: oracle generation, official-code
  audit, visual rollout triage, import/filter audit, and replay validation.
- Keep the work bug-driven. A code change needs a concrete failing oracle/no-op
  replay, rollout visual/root-cause finding, or official ScaleCUA/OSWorld parity
  mismatch needed to explain such a failure. Do not make broad or speculative
  fixes just because they look cleaner.
- Every five minutes during long runs, verify that available subagent capacity is
  doing meaningful non-overlapping work or is deliberately idle because no
  useful disjoint task exists.
- Do not let documentation activity replace bug fixing. If rollout or oracle
  replay exposes a systematic migration bug, fix the owner slice first and then
  update this document with the principle that would have prevented drift.
- Do not restart large rollout loops just because a fix exists. Restart only
  when continuing would score materially wrong semantics. Otherwise let the
  current batch finish and re-run affected tasks or a fresh sweep later.
- A milestone is commit-ready only when it makes the new path authoritative for
  its declared scope and deletes, quarantines, or explicitly labels stale
  artifacts.

## 2. Canonical Decisions

This section is the single source of truth for names, pins, counts, and artifact
ownership. README files, dev docs, tests, rollout commands, and subagent prompts
must agree with it.

### 2.1 Runtime Splits And Catalogs

Runtime split names are exactly `train` and `rl`.
`generated_tasks` and `rl_tasks` are upstream source names only.

| Runtime split | Source | Source metadata | Rows | Current excluded | Current runnable | Catalog |
|---|---|---|---:|---:|---:|---|
| `train` | HF `extreme1228/ScaleCUA/osworld/generated_tasks/**/*.json` | `generated_tasks` | 20,289 | 4,110 | 16,179 | `lite/gym/envs/lite/scalecua/data/train.jsonl` |
| `rl` | HF `extreme1228/ScaleCUA/osworld/rl_tasks/**/*.json` | `rl_tasks` | 2,049 | 210 | 1,839 | `lite/gym/envs/lite/scalecua/data/rl.jsonl` |

Current counts are from a fresh local provision:

```bash
uv run --no-sync bash lite/gym/envs/lite/scalecua/scripts/install.sh provision
```

Evidence artifact:

- `.cache/lite.scalecua_tasks/import_report.json`
  sha256 `7ae43cbe63e0e1d97c0d695df45103a3a9ceea313edf6a7883dfab08be0563e3`.
- `.cache/lite.scalecua_tasks/.asset_identity`
  sha256 `64920ffe12dc606114ec78bbbade4174ed83415544d703459a95dd26b083ec16`.
- The import command exited `0`; unsupported rows are represented by the
  primary `exclude_reason` counts below, not by importer exceptions.

Rules:

- `metadata.source_split` and `metadata.others.source_split` both store the
  upstream source name. They validate/report provenance only; runnable
  export/rollout filtering uses only the singular non-empty
  `metadata.others.exclude_reason` field.
- `metadata.others.domain` is the canonical domain for rollout/oracle
  stratification. Do not replace it with source directory, `related_apps`, or
  `snapshot`.
- `task_id` is stable and source-derived. Example:
  `scalecua_osworld_rl_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_traj_verify_0`.
- Runtime registration reads only imported JSONL catalogs.

Export contract:

- Exported parquet files are downstream artifacts, not canonical task catalogs.
  Recreate them from registry splits instead of committing or reusing old
  parquet blindly.
- Every runnable export must include:

```bash
--filter "lambda m: not m.others.get('exclude_reason')"
```

- The runtime split names accepted by export/rollout are only `train` and `rl`.
  Use `lite.osworld` for evaluation-set rollout.

### 2.2 Task-Source Pins And Cache Layout

ScaleCUA has no Lite-owned asset bundle and no `lite.scalecua-assets` mirror.
The upstream HF snapshot is task-source input, so
`scripts/utils/tasks.sh generate` owns both the upstream snapshot fetch and the
runtime catalog generation.

Generated task catalogs live under
`lite/gym/envs/lite/scalecua/data/` and are ignored by Git. They sit next to
the tracked `catalog.lock.json` so the user-facing mental model matches other
Lite envs: split JSONL catalogs are in `data/`, while bulky upstream material
and derived judge overlays live in repo-root `.cache/lite.scalecua_tasks/`.

Pinned sources:

- HF dataset: `extreme1228/ScaleCUA`
  at `77d7174d45d36e3c355269699d7f59a90a714ce6`.
- OSWorld file cache URLs are rewritten from
  `xlangai/ubuntu_osworld_file_cache/resolve/main` to revision
  `711e0811642364e7aa8f10a8918367d0b626d578`.
- `lite.osworld` eval catalog identity is included in
  `data/catalog.lock.json` only because ScaleCUA train/RL domain mapping uses
  matching OSWorld ids where available. It does not create ScaleCUA eval splits.

Pinning caveat: `--force-download` is the path that proves the HF revision pin.
Reusing an existing or probe snapshot is currently count- and judge-presence
guarded, not a full commit-marker/content-manifest verification. Release
evidence must either come from a force-download import or add explicit snapshot
identity verification before treating a reused snapshot as pinned content.

Expected generated catalog layout:

```text
lite/gym/envs/lite/scalecua/data/
├── train.jsonl
├── rl.jsonl
├── catalog.lock.json
└── oracle/
    ├── train.jsonl
    └── rl.jsonl
```

Expected task-source cache layout:

```text
.cache/lite.scalecua_tasks/
├── import_report.json
├── .asset_identity
├── .complete
├── hf_snapshot/
└── judge_functions/
    ├── train/
    │   ├── getters.py
    │   ├── metrics.py
    │   ├── verigen_getters/
    │   └── verigen_metrics/
    └── rl/
        ├── getters.py
        ├── metrics.py
        ├── verigen_getters/
        └── verigen_metrics/
```

### 2.3 Exclusion Reasons

`exclude_reason` is the primary filter field. It must be omitted for runnable
rows and a non-empty exact string for unsupported rows. Do not create parallel
fields such as `exclude_reasons`, `blocked_reason`, or proxy-only booleans for
main filtering.

Current primary exclusion counts after the same fresh import artifact:

| Split | Reasons |
|---|---|
| `train` | `infeasible=1852`, `proxy_required=1225`, `google_auth=346`, `upstream_live_site_drift=325`, `unsupported_asset_url=131`, `upstream_generated_eval_bug=121`, `instruction_eval_mismatch=87`, `instruction_setup_mismatch=15`, `missing_reference_asset=4`, and one each of `unsupported_schema:evaluator_postconfig_query_config`, `unsupported_setup:close_all_libreoffice`, `unsupported_setup:copyfile_from_guest_to_host`, `unsupported_setup:navigate_to_chrome_extensions` |
| `rl` | `proxy_required=102`, `infeasible=18`, `upstream_live_site_drift=22`, `upstream_generated_eval_bug=63`, `unsupported_schema:action_list=5` |

Primary reason precedence for non-inherited `train` and `rl` rows is:
`google_auth`; upstream non-empty reason; `infeasible`;
`instruction_setup_mismatch`; `instruction_eval_mismatch`;
`missing_reference_asset`; `upstream_generated_eval_bug`;
`unsupported_schema:*`; `unsupported_setup:*`; `unsupported_asset_url`;
`proxy_required`; `upstream_live_site_drift`. `proxy_required` intentionally
outranks only live-site drift, not auth, infeasible, schema, setup, or asset
defects.

For evaluation-set work, use `lite.osworld`'s `eval` split directly. ScaleCUA
does not import or expose `eval`/`eval_full` catalogs.

### 2.4 Image And Install Policy

Current phase has no `lite/gym/envs/lite/scalecua/docker/Dockerfile`, no
env-local `lite.scalecua` image, and no `lite.scalecua` entry in
`lite/gym/utils/backend/freshness.py`.

Repository config fallback for this validation branch:

```text
cua-lite/lite.osworld:latest
```

ScaleCUA image lifecycle policy:

```bash
uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/install.sh rebuild
uv run --no-sync bash lite/gym/envs/lite/scalecua/scripts/install.sh rebuild
```

- `lite.scalecua` has no Dockerfile and reuses `lite.osworld`.
- Default builds and rollouts use `cua-lite/lite.osworld:latest`.
- Install scripts do not expose fixed-tag image overrides; rebuilding follows
  the repository's normal `:latest` lifecycle.
- ScaleCUA development and validation must not pass runtime image overrides;
  the environment image comes from the checked-in `lite.scalecua` config.

Installer lifecycle in `lite/gym/envs/lite/scalecua/scripts/install.sh`:

| Command | Actual behavior |
|---|---|
| no arg / `build` | call `lite.osworld/scripts/install.sh`, install small import deps, import catalogs |
| `rebuild` | call base install, install deps, force-fetch HF snapshot, import with `--force-download` |
| `provision` | install deps, delegate to `scripts/utils/tasks.sh generate`, then run catalog/runtime-cache checks |
| `import` | compatibility alias for `provision`; do not use in new docs or runbooks |
| `status` | read-only; prints cache presence, asset identity, task cache `FRESH`/`STALE`/`MISSING`, catalog row counts, report presence, and base image `FRESH`/`STALE`/`MISSING` |
| `pull` | pulls/checks the shared `lite.osworld` base image, installs import deps, generates ScaleCUA `train`/`rl` catalogs, and checks catalog/runtime-cache freshness |
| `push` | exits non-zero because there is no ScaleCUA image |

These command names borrow the CUA-Lite env lifecycle vocabulary, not
`lite.cuagym` side effects. For ScaleCUA, there is no `assets` command because
the HF snapshot is task source owned by `scripts/utils/tasks.sh`; `push` is
intentionally unsupported until a dedicated ScaleCUA image exists.

Dockerfile boundary:

- Do not add a ScaleCUA Dockerfile unless ScaleCUA-specific runtime packages or
  services become necessary.
- Missing OSWorld VM base packages that official tasks/evaluators assume
  exist, for example `bc`, are shared image dependencies and belong in
  `lite.osworld` Dockerfile package layers.
- User-facing VM parity commands belong on the normal guest PATH. Use
  `/usr/local/bin` for osworld-scoped shims such as `timedatectl`,
  `hostname`, and `powerprofilesctl`; keep `/opt/env/bin` for env-facing tools
  hidden from the guest agent, such as backend-only helpers env-vendored for
  setup/eval plumbing.
- Issue-116 Python/package boundary is mandatory for Dockerfile fixes:
  evaluator/setup packages are installed with
  `uv pip install --python /opt/env/venv/bin/python ...`; agent-facing pip
  remains `/usr/bin/pip` or `/usr/bin/pip3`. Do not add `/usr/local/bin/pip`,
  `docker/bin/pip`, PATH-marker wrappers, or dispatch rewrites that make
  generic pip resolve to the env venv.
- Do not fix official CLI stdout semantics to satisfy a broken generated
  `exact_match`. If an official CLI normally prints a trailing newline and the
  generated ScaleCUA expected string omits it, classify the row as an upstream
  generated eval defect with exact `exclude_reason`.
- Do not set global process env that pins mutable OS state. In particular,
  the image may initialize `/etc/localtime` and `/etc/timezone`, but must not
  set `ENV TZ=...` because `timedatectl set-timezone` evaluators expect
  `date +%z` to follow the mutable system timezone.
- When a shared OSWorld runtime fix touches
  `lite/gym/envs/lite/osworld/docker/*`, setup/runtime files, or image
  dependencies, rebuild via `lite.osworld/scripts/install.sh`; do not invoke
  `docker build ... docker/` directly. The OSWorld Dockerfile requires the env
  root as build context, BuildKit heredocs, the selected base image build arg,
  and the installer's image freshness label.
- If a missing capability is a shared OSWorld VM/runtime parity gap that should
  be inherited by any `lite.osworld`-based env, fix it in the `lite.osworld`
  Dockerfile/runtime with `lite.osworld` regression evidence.
- If DBus/gsettings/shell behavior is proven to be a general official VM
  desktop capability, fix the shared `lite.osworld` runtime/supervisor session
  wiring. Per-oracle `dbus-run-session` or hardcoded bus repair is diagnostic
  scaffolding only; it must not become the permanent mechanism for
  making every task pass.
- If a missing behavior is ScaleCUA generated-judge/setup/profile/asset
  compatibility, keep the fix in `lite.scalecua`.

### 2.5 Official Eval Semantics

ScaleCUA train/RL evaluation is OSWorld-like but not identical to the
`lite.osworld` eval wrapper.

Required semantics:

- `train` and `rl` use a `DesktopEnv`-compatible shim. Generated ScaleCUA judges
  expect an env object with `.controller`, `.cache_dir`, `.vm_ip`,
  `.server_port`, `.chromium_port`, `.vlc_port`, and routed VM HTTP/CDP/VLC
  behavior. A raw container handle is not sufficient.
- Official `DesktopEnv.evaluate()` returns early for `no_verify` before
  postconfig. For the supported imported ScaleCUA evaluator surface,
  postconfig runs before infeasible/FAIL terminal scoring and before metric
  evaluation. Do not short-circuit postconfig just because a final action looks
  terminal.
- Single metric returns the raw official score.
- Multi-metric `and` returns `0` on the first zero; otherwise it returns the
  average.
- Multi-metric `or` returns `1` on the first one; otherwise it returns the max.
- Do not apply the base `lite.osworld` binary `>=0.5` aggregation to
  ScaleCUA `train`/`rl`.
- Evaluation-set validation is delegated to `lite.osworld`'s canonical `eval`
  split. `lite.scalecua` keeps only train/RL scoring semantics.

## 3. Target Runtime Architecture

### 3.1 Importer And Catalog Generation

Authoritative code:

- `lite/gym/envs/lite/scalecua/src/utils/assets.py`
- `lite/gym/envs/lite/scalecua/src/utils/dataset.py`
- `lite/gym/envs/lite/scalecua/scripts/utils/import_tasks.py`

Importer responsibilities:

- Fetch/pin the HF snapshot under `.cache/lite.scalecua_tasks/hf_snapshot`.
- Normalize setup and postconfig into strict action lists.
- Rewrite OSWorld file-cache URLs to the pinned revision.
- Materialize generated judge overlays into `.cache/lite.scalecua_tasks/judge_functions/`.
- Emit per-split JSONL catalogs and `import_report.json`.
- Validate strict row counts, domain counts, source split fields, duplicate IDs,
  and non-empty `exclude_reason` values.
- Publish generated cache state atomically: stage catalogs and judge overlays,
  validate the staged tree, replace the live cache only after validation passes,
  and roll back or leave the previous live cache intact on failure.

### 3.2 Setup Transport

Authoritative code:

- `lite/gym/envs/lite/scalecua/src/osworld/setup.py`
- shared base preamble from `lite/gym/envs/lite/osworld/src/utils/setup.py`

Required behavior:

- Run the shared `lite.osworld` setup preamble with empty config first.
- Dispatch ScaleCUA `config` actions through a strict allowlist after importer
  normalization. Any normalizer exception that maps an upstream action to a
  no-op or wait must be explicitly counted in `import_report`, documented here,
  and resolved before release by implementation, exclusion, or proof that the
  no-op is semantically safe for the affected task IDs.
- Do not call `lite.osworld.src.utils.dispatch.dispatch_actions` directly for
  ScaleCUA setup, postconfig, or oracle actions. OSWorld's batch dispatcher may
  log and continue on per-action exceptions or unknown action types. ScaleCUA
  actions must go through `dispatch_strict`, so unsupported actions, nonzero
  execute results, and missing declared file outputs fail closed.
- Fail closed on unsupported actions unless the importer has already excluded
  the task with a typed reason.
- Deterministic task/setup/material failures must surface as typed
  non-retryable task errors, not generic retryable `RuntimeError`s. Add or keep
  ScaleCUA-specific error coverage so excluded tasks, invalid setup,
  postconfig failure, and deterministic eval/material failure are not retried
  as transient infrastructure failures.
- Official ScaleCUA session code injects helper upload directories before task
  config for some flows. Current imported runnable rows have no known required
  `upload_dir` dependency; if future imports expose one, implement the helper
  transport or exclude the task with an exact reason.
- Do not pre-apply target state in setup to make oracle or rollout pass.
- The current dispatch existence check for file-producing setup is only
  `test -e`; do not document non-empty file validation unless the code is
  upgraded to enforce it.

### 3.3 Eval Transport And Judge Overlays

Authoritative code:

- `lite/gym/envs/lite/scalecua/src/osworld/verify.py`
- `lite/gym/envs/lite/scalecua/src/osworld/judges.py`
- `lite/gym/envs/lite/scalecua/src/osworld/xcf.py`

Required behavior:

- Load ScaleCUA generated judge overlays for `train` and `rl`.
- Prefer canonical `lite.osworld` getters/metrics where compatible.
- Fall back to ScaleCUA overlay and official `desktop_env` getters/metrics only
  for the generated surface that requires them.
- Compatibility shims are allowed only when they restore an official generated
  judge/runtime helper surface that ScaleCUA referenced but did not ship, or
  adapt that surface to the container/exec-stdio transport without changing the
  task's semantic expectation. Current allowlisted examples include generated
  GIMP XCF helpers and Calc chart private-helper repair. True upstream
  instruction/eval defects still require exact `exclude_reason` rather than a
  semantic adapter patch.
- Preserve raw official partial scores.
- Run evaluator `postconfig` in official order.
- Emit debug details sufficient to classify failures into migration bug,
  upstream task/eval defect, unsupported dependency, or model failure.

### 3.4 Registry And Env Server

Authoritative code:

- `lite/gym/envs/lite/scalecua/main.py`
- `lite/gym/envs/lite/scalecua/configs/default.yaml`
- `scripts/configs/gpt/default/lite.scalecua.yaml`

Rules:

- `lite.scalecua` registers as a normal CUA-Lite env using the shared
  `lite.osworld` image.
- Large rollout must use a fresh env-server root and must not share a stale
  server with previous diagnostic runs.
- Direct-mode oracle validation is currently intentional: the oracle validator
  clears `CUA_LITE_ENV_SERVER_URL` and uses its own `SESSION_ID`. Do not claim
  recommended/full oracle validation uses env-server until the validator is
  changed.
- Direct-mode runs must end with orphan-container checks.

## 4. File Structure Contract

Abridged runtime source layout. This intentionally omits generated `__pycache__`
files, but includes files that affect import, registration, setup/eval, and
oracle generation:

```text
lite/gym/envs/lite/scalecua/
|-- .gitignore
|-- README.md
|-- __init__.py
|-- configs/default.yaml
|-- data/
|   |-- README.md
|   |-- assets.lock.yaml
|   `-- oracle/
|       |-- README.md
|       |-- rl.jsonl
|       `-- train.jsonl
|-- main.py
|-- scripts/
|   |-- install.sh
|   `-- utils/
|       |-- __init__.py
|       `-- import_tasks.py
`-- src/
    |-- __init__.py
    |-- gen/
    |   |-- __init__.py
    |   `-- oracle/
    |       |-- __init__.py
    |       |-- __main__.py
    |       |-- _fixtures.py
    |       `-- domains/
    |           |-- __init__.py
    |           |-- chrome.py
    |           |-- gimp.py
    |           |-- libreoffice_calc.py
    |           |-- libreoffice_impress.py
    |           |-- libreoffice_writer.py
    |           |-- multi_apps.py
    |           |-- os.py
    |           |-- thunderbird.py
    |           |-- vlc.py
    |           `-- vs_code.py
    |-- osworld/
    |   |-- __init__.py
    |   |-- judges.py
    |   |-- setup.py
    |   |-- verify.py
    |   `-- xcf.py
    `-- utils/
        |-- __init__.py
        |-- assets.py
        `-- dataset.py
```

Abridged development source layout. Generated validation artifacts under
`.exps/validate/lite.scalecua/` are intentionally omitted; `oracle/` keeps only
core validation entrypoints and logs:

```text
devs/envs/lite.scalecua/
|-- AGENTS.md
|-- UPSTREAM_ISSUES.md
|-- gaps.md
`-- validate/
    |-- README.md
    |-- checklist.md
    |-- static.py
    |-- oracle/
    |   |-- coverage_inventory.py
    |   |-- logs.md
    |   |-- plan.md
    |   |-- select_fixtures.py
    |   |-- verified_inventory.py
    |   `-- validate.py
    |-- rollout/
    |   |-- analysis.md
    |   |-- audit_queue.py
    |   |-- check_prompt_data.py
    |   |-- make_batch_prompt_data.py
    |   |-- plan.md
    |   `-- logs.md
    `-- samples/
        `-- README.md
```

Workspace-local oracle fixtures currently consist of two canonical aggregate
JSONL files in `lite/gym/envs/lite/scalecua/data/oracle/`: `rl.jsonl` with
1,839 RL fixture rows and `train.jsonl` with 217 train fixture rows. Per-batch
and per-domain shard JSONL files are not part of the published data layout;
their provenance lives in `src/gen/oracle/*.py` and each row's `source` field.
Current RL coverage inventory matches all 1,839 RL rows to runnable RL catalog
tasks with zero fixture problems, zero duplicate fixture tasks, eval/setup
combos `1,335 / 1,335`, and full setup/eval/postconfig combos
`1,363 / 1,363`.

Untracked workspace-local shards are validation inputs only; they are not
release artifacts until registered, regenerated into the canonical aggregates,
replayed, and deliberately promoted.

Promotion path:

- Prefer source-backed promotion: put the oracle recipe in
  `lite/gym/envs/lite/scalecua/src/gen/oracle/*.py`, register it in `SHARDS`,
  regenerate the JSONL shard, and byte-lock it with generator `--check`.
- One-off hand-authored JSONL is allowed only when a generator would add more
  complexity than evidence value; those rows still need no-op and replay
  artifacts before counting toward RL closure.
- Untracked workspace-local JSONL never counts as release evidence by itself.
  The final branch must either track the promoted fixture files or replace them
  with a documented asset-promotion mechanism before the RL closure gate can
  close.

Registered source-backed oracle generation currently has 10 domain entries in
`lite/gym/envs/lite/scalecua/src/gen/oracle/__main__.py`. Do not maintain a
manual shard table elsewhere; the source of truth is:

```bash
uv run python -m lite.gym.envs.lite.scalecua.src.gen.oracle --check
```

For row counts during development, load `SHARDS` from the same module and call
`build_rows(domain)`. A row becomes release-countable only after generator
byte-lock and strict no-op/oracle replay evidence.

There must be no unregistered generator modules in
`lite/gym/envs/lite/scalecua/src/gen/oracle/`. Candidate/provenance code belongs
in `.exps/` or dev notes until it passes the strict gate. Promotion requires
adding the row to the appropriate registered domain source, deterministic
regeneration, static checks, and replay through the oracle validator before it
changes the release fixture counts.

## 5. Task-Fix Ownership Boundary

Every failing task must be classified before editing code:

1. Shared OSWorld runtime parity gap.
2. ScaleCUA adapter/setup/eval transport bug.
3. Upstream ScaleCUA task/eval defect.
4. Unsupported external dependency or missing material.
5. Model failure.

Ownership:

- Shared OSWorld runtime parity gaps are capabilities that the official
  OSWorld VM/runtime supports and that all `lite.osworld`-based envs should
  inherit. Examples include VM command behavior, desktop service shims,
  missing runtime packages, or base image behavior. Fix these in
  `lite/gym/envs/lite/osworld/docker/` or shared `lite.osworld` runtime code,
  with `lite.osworld` regression tests.
- Runtime parity includes two path classes. User-facing VM commands go on the
  normal guest PATH (`/usr/local/bin` or `/usr/bin`) because the agent and
  evaluator may both call them. Env-facing setup/eval helpers go under
  `/opt/env/bin` or another backend-only path so they do not change the
  user-visible desktop surface.
- Python/package parity is governed by the issue-116 split. Env-facing
  evaluator/setup code may use `/opt/env/venv/bin/python` and dependencies
  installed into that venv; agent-facing `python3`, `pip`, and `pip3` remain the
  system guest tools. A task postconfig that installs a Python package with
  `sudo pip` does not justify a global pip wrapper when the evaluator later runs
  env Python.
- Runtime parity fixes that should have lived in Docker/runtime must be moved
  out of per-task Python monkey patches when found. Examples: base packages
  like `bc`, VM CLIs like `timedatectl`/`powerprofilesctl`, mutable timezone
  behavior, sudoers needed for those VM CLIs, and general DBus/gsettings
  session availability.
- ScaleCUA-specific compatibility belongs in `lite/gym/envs/lite/scalecua/`:
  generated judge shims, DesktopEnv compatibility, Chrome profile aliasing,
  ScaleCUA setup normalization, asset materialization, XCF handling, and
  source-specific evaluator overlays.
- Upstream ScaleCUA task/eval defects must get exact `exclude_reason` evidence
  and be documented in `devs/envs/lite.scalecua/UPSTREAM_ISSUES.md` or
  `devs/envs/lite.scalecua/gaps.md`.
- Unsupported external dependencies such as login, proxy, CAPTCHA, unavailable
  author artifacts, and live-site drift should be filtered with exact reasons
  rather than patched in setup or oracle actions.
- Avoid ScaleCUA-specific changes in generic modules such as
  `lite/infer/cli.py`, `lite/infer/rollout.py`, or shared action utilities.
  Touch shared modules only for product-level semantics with cross-env tests.

## 6. Oracle Policy

Oracle fixtures are validation artifacts, not runtime catalogs. Each fixture
must reference a real imported task by `split` and `task_id`.

Required fixture semantics:

- ScaleCUA oracle validation uses two fresh resets per fixture. The precheck
  reset captures screenshots, optionally runs fixture postconfig for
  `oracle_after_postconfig`, then sends a no-op final `response`; reward must
  equal `expected_pre_reward`, defaulting to `0.0`. The replay reset captures
  screenshots, optionally runs fixture postconfig, replays oracle
  actions/trajectory, captures post-action screenshots and eval debug, then
  sends the final response; reward must equal `expected_reward`, defaulting to
  `1.0`.
- Oracle actions may use deterministic dispatch operations such as shell/file
  edits, app config writes, or short app actions. The hard rule is no target
  pre-application in setup/postconfig and no fake artifacts that a user could
  never produce.
- Screenshots, debug payloads, and logs are evidence. A numeric reward without
  visual or debug inspection is insufficient for promotion.
- Oracles must run through production setup/eval, not a simplified verifier.
- Any reward mismatch is first assumed to be a migration bug until official
  ScaleCUA code, screenshots, and task JSON prove otherwise.
- A passing oracle is not just a green check. It should be kept as a
  source-backed or deliberately promoted regression fixture so future changes
  cannot silently drift setup/eval behavior.
- A failing oracle is not just a failed fixture. It is potential bug evidence
  and must be classified before moving on: oracle recipe mistake, ScaleCUA
  migration/setup/eval bug, shared OSWorld runtime/container parity bug,
  upstream task/eval defect requiring exact `exclude_reason`, or transient
  infrastructure failure needing isolated rerun.

Primary oracle development strategy:

- The default path for new ScaleCUA oracle coverage is to find the matching
  per-domain `lite.osworld` oracle pattern first, then adapt that pattern to the
  current ScaleCUA task. Do not design a fresh oracle approach until the closest
  OSWorld domain recipes have been inspected and ruled out.

- Before implementing a ScaleCUA oracle family, inspect the closest
  `lite.osworld` `metadata.others.oracle_actions` and generator code.
- Reuse the operational recipe: which process is stopped, which profile/config
  file is mutated, how gold files are generated, how app state is normalized,
  when postconfig runs, and which artifacts prove state.
- Recompute target values from the current ScaleCUA evaluator. Do not copy
  constants by `metadata.osworld_id` unless the evaluator goal signature and
  target state match exactly.
- Use the closest `lite.osworld` oracle fixtures and generator code directly as
  mandatory input for new batches; do not maintain a copied one-off inventory
  note in `lite.scalecua`.
- Mirror useful implementation principles back into this document or the
  relevant `devs/envs/lite.scalecua` validation document in the same pass.

Oracle source layout convergence:

- User-facing task metadata must match `lite.osworld`: `oracle_actions` and
  `oracle_after_postconfig` live only in `metadata.others`. The fixture JSONL
  schema may keep top-level replay fields because it is a dev validator input,
  not `LiteBaseMetadata`.
- `lite.osworld` keeps oracle source by domain: eval oracles live in
  `src/gen/eval/<domain>.py`, synth/perturb oracles flow through
  domain-oriented generators and shared row builders. ScaleCUA mirrors that user
  and maintenance shape with one registered oracle source per domain.
- Legacy `rl_auto_*` names remain only as fixture-id/source provenance inside
  `data/oracle/*.jsonl`; they are not source file names. Do not add new
  batch-numbered source files for normal development. Promote new coverage into
  the matching domain module.
- Current source layout:

  ```text
  lite/gym/envs/lite/scalecua/src/gen/oracle/
    __main__.py
    _fixtures.py
    domains/
      chrome.py
      gimp.py
      libreoffice_calc.py
      libreoffice_impress.py
      libreoffice_writer.py
      multi_apps.py
      os.py
      thunderbird.py
      vlc.py
      vs_code.py
  ```

- Each domain module currently stores canonical, byte-locked oracle rows because
  the historical aggregate contains both generated rows and legacy bootstrap
  rows with intentionally preserved field shapes. Future algorithmic builders
  may be introduced per domain only if `data/oracle/rl.jsonl` and
  `data/oracle/train.jsonl` remain byte-locked or are intentionally regenerated
  with fresh replay evidence.

Current broad oracle coverage:

- `train+rl` runnable tasks: 18,141.
- workspace-local fixture rows: 2,056 across canonical `rl.jsonl` and
  `train.jsonl`.
- RL eval x setup coverage: `1,335 / 1,335`.
- RL full setup x eval x postconfig coverage: `1,363 / 1,363`.
- strict artifact-backed verified RL fixtures: `1,839 / 1,839`.

Current candidate tiers. Each pair is a selection artifact plus a coverage
report; none is replay evidence:

- `smoke`: 248 candidates,
  `.exps/validate/lite.scalecua/oracle/candidates/smoke.candidates.jsonl` and
  `.exps/validate/lite.scalecua/oracle/candidates/smoke.coverage.json`.
- `recommended`: 648 candidates,
  `.exps/validate/lite.scalecua/oracle/candidates/recommended.candidates.current.jsonl`
  and
  `.exps/validate/lite.scalecua/oracle/candidates/recommended.coverage.current.json`.
- `all_eval_setup`: 5,308 candidates,
  `.exps/validate/lite.scalecua/oracle/candidates/all_eval_setup.candidates.current.jsonl`
  and
  `.exps/validate/lite.scalecua/oracle/candidates/all_eval_setup.coverage.current.json`.
- `all_full`: 5,436 candidates,
  `.exps/validate/lite.scalecua/oracle/candidates/all_full.candidates.current.jsonl`
  and
  `.exps/validate/lite.scalecua/oracle/candidates/all_full.coverage.current.json`.
- `rl_full`: one candidate per current non-excluded RL catalog row after
  regeneration,
  `.exps/validate/lite.scalecua/oracle/candidates/rl_full.candidates.current.jsonl`
  and
  `.exps/validate/lite.scalecua/oracle/candidates/rl_full.coverage.current.json`.

These coverage and candidate numbers are inventory only. Do not describe them as
"covered" in release notes unless the matching fixture rows also satisfy the
release-countable oracle evidence definition at the top of this document.
Regenerate candidate tiers after every import/filter change before quoting
their numeric counts.

## 7. Validation Strategy

### 7.1 Evidence Rules

- A plan-only recipe is not an executed eval artifact.
- Coverage counts are not evidence by themselves; executable replay artifacts
  are required.
- Static checks close only static gates.
- Candidate JSONL closes only selection/coverage gates, not oracle correctness.
- Rollout `summary.json` closes neither correctness nor task-quality gates
  without screenshot review and root-cause classification.
- Rollout triage labels may include `deps_missing`, `materials_missing`, typed
  intentional refusal, and model failure, but RL closure has only two terminal
  states: release-countable oracle evidence or exact
  `metadata.others.exclude_reason`.
- Local missing-dependency evidence does not shrink the release universe unless
  it becomes an exact `exclude_reason`.

### 7.2 Goal-To-Gate Map

| Goal | Required gate |
|---|---|
| Catalog determinism | `install.sh provision`, `static.py`, strict count/domain validation, fresh `import_report.json` |
| Runtime registration | registry tests and `gym.registry.task_ids("lite.scalecua", split="rl")` smoke |
| Setup transport | strict-dispatch tests, typed non-retryable task-error tests, plus rollout/oracle canaries for each action family |
| Eval transport | official-code parity tests, raw-score tests, postconfig-order tests |
| RL closure | every RL row exact excluded or replayed oracle-covered |
| Oracle quality | no-op negative eval, oracle replay positive eval, screenshots/debug payloads |
| Rollout bug-fix loop | one clean 1000-task root plus one independent confirmation root, each on a fresh env-server at concurrency 8, sampled as `500 train + 500 rl` with `50` tasks per split-domain cell, visual audit queues, orphan cleanup |
| Drift prevention | static tests, coverage reports, oracle replay reports, rollout audit logs, docs synchronized |

### 7.3 Static And Catalog Commands

Required after importer/filter/setup/eval changes:

```bash
uv run --no-sync bash lite/gym/envs/lite/scalecua/scripts/install.sh provision
uv run python devs/envs/lite.scalecua/validate/static.py
uv run --no-sync pytest tests/gym/envs/lite/scalecua tests/agents/test_registration_complete.py -q
```

Required after registered oracle generator changes:

```bash
uv run python -m lite.gym.envs.lite.scalecua.src.gen.oracle --check
```

### 7.4 Coverage Commands

RL-only report:

```bash
uv run python devs/envs/lite.scalecua/validate/oracle/coverage_inventory.py \
  --fixtures lite/gym/envs/lite/scalecua/data/oracle \
  --catalog-dir lite/gym/envs/lite/scalecua/data \
  --splits rl \
  --output .exps/validate/lite.scalecua/oracle/coverage/rl.coverage.json \
  --markdown .exps/validate/lite.scalecua/oracle/coverage/rl.coverage.md
```

Train+RL report:

```bash
uv run python devs/envs/lite.scalecua/validate/oracle/coverage_inventory.py \
  --fixtures lite/gym/envs/lite/scalecua/data/oracle \
  --catalog-dir lite/gym/envs/lite/scalecua/data \
  --output .exps/validate/lite.scalecua/oracle/coverage/current.coverage.json \
  --markdown .exps/validate/lite.scalecua/oracle/coverage/current.coverage.md
```

RL verified evidence report:

```bash
uv run python devs/envs/lite.scalecua/validate/oracle/verified_inventory.py \
  --fixtures lite/gym/envs/lite/scalecua/data/oracle \
  --reports .exps/validate/lite.scalecua/oracle \
  --splits rl \
  --require-artifacts \
  --output .exps/validate/lite.scalecua/oracle/coverage/rl.verified.json \
  --markdown .exps/validate/lite.scalecua/oracle/coverage/rl.verified.md
```

Train+RL verified evidence report:

```bash
uv run python devs/envs/lite.scalecua/validate/oracle/verified_inventory.py \
  --fixtures lite/gym/envs/lite/scalecua/data/oracle \
  --reports .exps/validate/lite.scalecua/oracle \
  --splits train rl \
  --require-artifacts \
  --output .exps/validate/lite.scalecua/oracle/coverage/current.verified.json \
  --markdown .exps/validate/lite.scalecua/oracle/coverage/current.verified.md
```

Inventory gate:

- Before any local fixture shard contributes to RL closure, run
  `coverage_inventory.py --splits rl --require-clean` over all fixture files.
  The command must exit `0`, which requires zero
  `fixture_task_not_in_catalog`, `fixture_split_mismatch`,
  `fixture_points_to_excluded_task`, and duplicate fixture-task problems.
- Registered generator `--check` covers only shards in `SHARDS`.
- Unregistered/local JSONL shards do not count as release evidence while they
  remain workspace-local. Before counting, they must be deliberately promoted
  or replaced by a documented asset-promotion mechanism, pass inventory with
  zero catalog problems, and have no-op plus oracle replay artifacts.
- Before marking RL oracle closure, run `verified_inventory.py --splits rl
  --require-complete --require-catalog-complete` and require exit code `0`.
  `--require-complete` proves every current RL fixture row is replay-verified;
  this currently passes for 1,839 / 1,839 current RL fixtures.
  `--require-catalog-complete` proves every non-excluded RL catalog row has
  replay-verified evidence. The current checkpoint passes for 1,839 / 1,839
  runnable RL catalog rows with strict artifact-backed replay evidence.

Candidate refresh:

```bash
uv run python devs/envs/lite.scalecua/validate/oracle/select_fixtures.py \
  --coverage-target all-tasks --splits rl \
  --output .exps/validate/lite.scalecua/oracle/candidates/rl_full.candidates.current.jsonl \
  --report .exps/validate/lite.scalecua/oracle/candidates/rl_full.coverage.current.json

uv run python devs/envs/lite.scalecua/validate/oracle/select_fixtures.py \
  --coverage-target all-eval-setup \
  --output .exps/validate/lite.scalecua/oracle/candidates/all_eval_setup.candidates.current.jsonl \
  --report .exps/validate/lite.scalecua/oracle/candidates/all_eval_setup.coverage.current.json

uv run python devs/envs/lite.scalecua/validate/oracle/select_fixtures.py \
  --coverage-target all-full \
  --output .exps/validate/lite.scalecua/oracle/candidates/all_full.candidates.current.jsonl \
  --report .exps/validate/lite.scalecua/oracle/candidates/all_full.coverage.current.json
```

### 7.5 Oracle Replay

Reference scripts:

- `devs/envs/lite.scalecua/validate/oracle/validate.py`
- `devs/envs/lite.scalecua/validate/oracle/coverage_inventory.py`
- `devs/envs/lite.scalecua/validate/oracle/verified_inventory.py`
- `devs/envs/lite.osworld/validate/oracle/validate.py`

Replay requirements:

- Run no-op negative eval and oracle-positive replay for every promoted,
  source-backed, registered, or `data/oracle/` RL fixture before it contributes
  to RL closure. A fixture row that cannot be replay-verified must be fixed,
  exactly excluded, or moved out of release fixture paths before a release
  checkpoint.
- Shard replay within one validator process using `--concurrency`. The validator
  uses a session id derived from `--artifacts` by default, or an explicit
  `--session-id`, and sweeps only matching direct-mode containers at startup,
  signal handling, and final cleanup. Never launch two validator processes with
  the same `--session-id` against the same Docker daemon.
- Because current oracle validation uses direct mode, every replay batch must
  end with orphan-container checks.
- Current `lite.scalecua` oracle validation is direct-mode only. If any linked
  dev doc suggests env-server-backed recommended/full oracle replay before the
  validator grows isolated server sessions, that doc is stale and must be
  corrected rather than followed.

### 7.6 Rollout Bug-Fix Loop And Visual Audit

Reference docs:

- `devs/envs/lite.osworld/validate/rollout/plan.md`
- `devs/envs/lite.scalecua/validate/rollout/plan.md`

Large rollout bug-fix loop:

- Treat rollout as a bug-finding and fix-verification loop, not as a separate
  task-quality scorecard. Raw success rate is a health signal only after
  reward/screenshot mismatches and systematic failures are classified.

- Regenerate prompt data from the current catalog. Old diagnostic
  `gpt-5.5-1000-*` prompt/root artifacts are not bug-fix evidence unless
  explicitly reclassified against the current catalog, a fresh env-server, full
  completion, visual audit, and known-bug disposition.
- Validate prompt data with
  `devs/envs/lite.scalecua/validate/rollout/check_prompt_data.py`.
- Start a fresh env-server for the batch. Do not reuse a stale server/root from
  smoke or diagnostic runs.
- Run gpt-5.5 rollout with concurrency 8. The 1000-task bug-fix loop is sampled as
  `500 train + 500 rl` over the current runnable catalog: 10 canonical domains
  per split and 50 tasks per split-domain cell.
- For the 1000-task loop, every trajectory needs an audit row. Reward `1`,
  reward `0`, partial rewards, and errors get separate review queues.
- Visual review must inspect screenshots and relevant ScaleCUA task/evaluator
  code. Reward `1` can still be false positive; reward `0` can still be a
  migration/eval bug.
- For reward/visual mismatches, distinguish oracle replay from agent replay.
  Oracle/no-op validation proves setup/eval/gold feasibility; it does not prove
  the recorded model trajectory produced the scored state. A `false_failure`,
  `false_success`, or `ambiguous_needs_evaluator_probe` closes only after a
  targeted getter/file probe or recorded-trajectory replay equivalent is
  documented. If no ScaleCUA replay script exists, name the probe command and
  artifact path used instead.
- The 1000-task loop uses visual audit labels, not raw `summary.json`: 1000/1000
  trajectories audited; zero unresolved `transient_failure`; zero
  `false_success`; zero `false_failure`; overall visual success rate at least
  70%; `train` visual success rate at least 70%; `rl` visual success rate at
  least 70%; every canonical domain has 100 audited trajectories total,
  exactly 50 `train` plus 50 `rl`; every split/domain cell has 50 audited
  trajectories and a recorded success rate.
- The 1000-task batch is a ScaleCUA bug-finding sample, not a substitute for the
  OSWorld rollout audit loop. The loop requires one clean 1000-task root plus an
  independent confirmation root after all source/filter/runtime fixes from the
  first root are applied. Each root must use freshly generated prompt data,
  pass `check_prompt_data.py`, run on a fresh dedicated env-server, and have its
  own visual audit JSONL. Within a root, keep the log root stable for
  monitoring; after any fix that changes task bytes, eval semantics, or runtime
  semantics, mark the current root diagnostic and start a new clean root unless
  the affected task dirs are explicitly deleted/rerun with a sentinel and
  documented in `rollout/logs.md`.
- Persist rollout analysis under `devs/envs/lite.scalecua/validate/rollout/`.
  Confirmed migration/runtime bugs go to `gaps.md` with the owner-layer fix;
  confirmed upstream task/eval defects go to `UPSTREAM_ISSUES.md` plus an exact
  `exclude_reason`; transient live-site blocks get a rerun/freeze/exclude
  decision; model failures need screenshots and code-side evidence before they
  are removed from the bug queue.
- Do not restart an active large rollout just because a scoped fix exists.
  Restart only when continuing would make most remaining trajectories score
  under materially wrong semantics. Otherwise let the batch keep producing
  evidence and rerun affected tasks separately after the fix.
- End every rollout cycle with orphan env-server/direct-mode container checks.

### 7.7 Mandatory Parallel Validation Lanes

| Lane | Work | Parallelism rule |
|---|---|---|
| S | static/import/catalog schema checks | run immediately after source changes |
| C | coverage inventory and candidate refresh | run in parallel with static checks when catalogs are stable |
| O | oracle generator implementation and replay | shard by domain/setup/eval family |
| E | official eval/setup parity probes | only to explain rollout/oracle failures or prove a candidate owner-layer fix |
| D | Docker/env-server smoke and cleanup checks | only for active rollout/oracle infrastructure or owner-layer fixes |
| R | rollout and visual triage | keep rollout running while subagents inspect completed failures |

## 8. Migration Plan

### 8.1 Current Governance Checkpoint

M1 is this document-audit checkpoint:
`devs/envs/lite.scalecua/scalecua.md` must stay English-only,
count-consistent with fresh import/coverage artifacts, and structurally clear
about RL closure, oracle evidence, rollout visual validation, and fix
ownership. This checkpoint is not an implementation release gate by itself; the
remaining implementation milestones below are still open.

### 8.2 Remaining Implementation Milestones

| Milestone | Owner scope | Parallel work | Commit gate |
|---|---|---|---|
| M2 Importer and task-source contract | `src/utils/*`, `scripts/utils/import_tasks.py`, tests | source-count audit, filter taxonomy audit, OSWorld domain-map audit | fresh import, static validation, catalog tests |
| M3 Setup/eval transport parity | `src/osworld/*`, focused tests | official DesktopEnv comparison, generated judge probes, shared runtime gap audit | setup/eval canaries pass; raw-score and postconfig-order tests pass |
| M4 Oracle source-generation expansion | `src/gen/oracle/*`, `data/oracle/*`, oracle docs | domain workers for Office, GIMP/TB, multi_apps, Chrome/OS/VLC/VSCode tails | new shards registered or explicitly rejected, generator `--check`, coverage inventory refreshed, replay canaries pass |
| M5 RL full oracle closure | canonical oracle aggregates and replay artifacts | replay shards, no-op negative probes, visual artifact inspection | all 1,839 runnable RL tasks release-countable replay-covered or exactly excluded |
| M6 1000-task rollout bug-fix loop | rollout prompt/root, env-server, audit logs | rollout runner, visual subagents, code-side root-cause subagents | one clean `500 train + 500 rl` root plus one independent confirmation root, 50 tasks per split-domain cell, concurrency 8, visual audit complete, zero false success/failure, >=70% overall/train/rl visual success after fixes/exclusions |
| M7 Release cleanup | README/dev docs/tests/cache guards | stale artifact audit, orphan cleanup audit, regression test audit | no stale prompt/root claims, no orphan containers, no unregistered evidence counted as complete |

## 9. Decision Log

Accepted:

- `lite.scalecua` reuses the `lite.osworld` runtime substrate; no new ScaleCUA
  image in this phase. The normal lifecycle uses
  `cua-lite/lite.osworld:latest`.
- Runtime splits are only `train` and `rl`.
- `rl_tasks` is the release north star.
- `exclude_reason` is singular and is the main filter field.
- Shared OSWorld runtime parity gaps may be fixed in `lite.osworld`; ScaleCUA
  generated-judge/setup/profile/asset compatibility stays in `lite.scalecua`.
- Oracle authoring must start from the closest `lite.osworld` recipe and
  generator source for the same domain/setup/eval family.
- Large rollout bug-fix evidence requires visual audit, not just reward
  summaries.

Deferred:

- A dedicated `lite.scalecua` Dockerfile.
- Env-server-backed oracle validation. Current oracle validator is direct-mode.

## 10. Definition Of Convergence

Design convergence requires:

- This document is English-only and has no stale count/artifact claims.
- Source pins, runtime splits, image policy, install lifecycle, and task-fix
  ownership are internally consistent.
- Current evidence and target gates are never mixed.
- No release claim depends only on untracked workspace-local JSONL,
  unreplayed generator output, raw rollout reward, or unaudited screenshots.
- Every valuable correction from development is reflected here or in the
  relevant `devs/envs/lite.scalecua` validation document.

Implementation convergence requires:

- Fresh `install.sh provision` and `static.py` pass.
- Registration and focused tests pass.
- `src/gen/oracle --check` passes for all registered generator shards.
- Every RL row is exactly excluded or replay-covered by oracle evidence.
- No-op negative eval and oracle replay evidence exist for every promoted,
  source-backed, registered, or `data/oracle/` RL fixture that remains in a
  release fixture path.
- One clean 1000-task gpt-5.5 root plus one independent confirmation root reach
  the rollout health threshold after systematic migration bugs are fixed and
  exact upstream defects are excluded.
- Visual audit queues for reward `1`, reward `0`, partial, and error cases are
  complete.
- Direct-mode and env-server runs leave no orphan containers.
- README/dev docs/tests agree with this plan.
