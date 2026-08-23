# Rollout Migration

Forward-migrate **rollout trajectories already published on HuggingFace** to the current
LiteSample schema. Scope is deliberately narrow: this is a **one-time repair of exactly
the user-defined migration whitelist for HF-uploaded rollout inputs** (`Lite.OSWorld`,
`Lite.CUAGym`, `Lite.CUAWorld`, `Lite.ScaleCUA`, and `WebGym`), not a general data
pipeline, not a re-collection, and not a place to redeclare data policy. Other uploaded
datasets are intentionally retired migration inputs and must be regenerated from
`lite/data/preproc` raw sources instead of migrated. It is also the *only* place such
repair may live: new runtime/export code must not repair retired flat Lite calls
(`{call_id,name,arguments}`), JSON-string `function.arguments`, or missing/alias ids.

- The pure schema transform is [upgrade.py](/devs/migration/upgrade.py).
- File iteration, dry-run and verification are [run.py](/devs/migration/run.py).
- Invariant checks are [verify.py](/devs/migration/verify.py).
- `run.py` **never mutates a remote HF repo.** Uploading is transport only and is a
  separate, explicit step after stage/verify validation.

For the collect-side pipeline (rollout → annotate → stage → upload) see
[/devs/data/lite.osworld/AGENTS.md](/devs/data/lite.osworld/AGENTS.md). This document is its
migration counterpart: same destination, different source.

## What Migration Repairs

Migration's target is the current canonical LiteSample contract for the
allow-listed published rollout rows. It is not a byte-for-byte oracle against
fresh preproc: the migrated row is expected to pass the same content gates that a
new staged row must pass, while preserving the historical data that was actually
published. Datasets outside the allow-list are not repaired here; regenerate
those from `lite/data/preproc` raw sources instead.

The migrator only repairs representations that old published rows used and current preproc no
longer emits:

| old published input | canonical migrated output |
|---|---|
| flat Lite tool calls (`{call_id,name,arguments}`) | canonical nested calls (`{id,type,function}`) |
| JSON-string `function.arguments` | dict `function.arguments` |
| missing/alias call ids | canonical `id` / `tool_call_id` pairing |
| bare GUI action runs | canonical action-batch calls (`computer(actions=[...])` / `mobile(actions=[...])`) |
| post-action `role:"user"` observation bubbles | `role:"tool"` result messages owned by the producing call's `id` via `tool_call_id` |
| old terminal markers and spellings | the shared `lite/data/utils/messages` output |
| legacy metadata names such as `extra_tools` | current tagged `LiteCUAMetadata` keys |

Migration **compacts** images, once, at the row write-out point (`_finalize_row`). Stripping
noop-only `screenshot` / `wait` turns drops a turn but no picture, so the image that turn showed
ends up referenced by nothing: `devs.data.utils.compact_row_images`
([devs/data/utils.py](/devs/data/utils.py)) — the one place allowed to
renumber an index — drops those orphans and renumbers the survivors `0..N-1`, rewriting the
`images` list and every `{"type":"image","index": N}` part in the same step (and asserting the
result is dense). References that pointed outside the ORIGINAL list are still rejected, not
repaired. Stale observation screenshots are not promoted into reference images; authored
reference images must survive as ordinary message content image parts.

Migration is **not** a no-op on already-published rows. Verified on a 6-row Lite.ScaleCUA
subset: `messages_unchanged = 0/3` and `metadata_unchanged = 0/3` for both configs — every row
is rewritten.

## What Migration Cannot Repair: An Unpaired Final Action

A published row whose episode ran out of step budget ends on an action whose observation was
never published. Migration turns observations it already has into `role:"tool"` results; it
cannot invent the one that was never published. So the final action stays unpaired, and
[verify.py](/devs/migration/verify.py) exempts exactly those call ids. Current
`validate_canonical_rows` also allows a final assistant tool call at EOF with no
`role:"tool"` result. Stage rejects non-final missing results and orphan tool
results; it should not be cited as the reason to drop these final-label rows.
The publish decision for affected historical rows is the explicit incomplete-row publish policy.

**This is a raw-boundary fact, not a reason to fabricate data.** `verify.py` records
that the source never published the final observation. Migration should not invent an
empty result or a synthetic `Done.` final to close the label.

**Measured 2026-08-08 over the allow-listed datasets, every locally cached row of the two
affected ones.** Disabling the exemption newly refuses **242 of 636** migrated rows in **39 of 46
partitions** — `Lite.CUAWorld` 235/409, `Lite.CUAGym` 7/21, `Lite.OSWorld` 0/133,
`Lite.ScaleCUA` 0/73 — and it refuses them inside `upgrade_lite_sample` itself
(`_validate_canonical_output`), so even a `--dry-run` without `--verify` raises. The exemption is
load-bearing; do not delete it. All 242 are step-budget truncations
(`terminated=False, truncated=True`, `exclude_reason` containing `incomplete`) whose last
assistant turn is a bare `computer` call — **0 of 242 folded a `terminate` onto the last action
turn**, which is what an earlier revision of this section claimed the exemption was for.

The `Lite.ScaleCUA 0/73` entry above is a locally cached sample, not the full published dataset.
Over the full published `Lite.ScaleCUA` migration, `17,953/17,953` rows migrate and `--verify`
passes, but **460/17,953** rows are excluded by the same incomplete-row publish filter used by
the published staged tree; the filtered staged output is **17,493** rows.

Current preproc emits no such row: cagui / guiodyssey keep the terminal step's screenshot as the
last action's result, ui_genie_agent skips an episode without one, and aguvis / guiact /
multimodal_mind2web drop the unpaired final step. **So the repair for an affected source is to
re-run [lite/data/preproc](/lite/data/preproc) from the raw source, not to migrate it** — for
cagui / guiodyssey the fresh-preproc row is also strictly richer, publishing the terminal
screenshot the old row dropped. Check a dataset with `--dry-run` and then stage a sample before
planning a migration.

Scope note: the user-defined migration whitelist is exactly the five HF-uploaded dataset
routes [run.py](/devs/migration/run.py) allow-lists: `Lite.OSWorld`, `Lite.CUAGym`,
`Lite.CUAWorld`, `Lite.ScaleCUA`, and `WebGym`. Everything else is refused by
`_require_allowed_lite_dataset_path`. The match is by exact dataset route, not a scratch
alias: a path under `WebGymRT`, `WebGym.copy`, or `Lite.OSWorld.copy` is still out of
scope even if it contains an allow-listed-looking child. `ScaleCUA` without the `Lite.`
published-route prefix, including `cua-lite/ScaleCUA`, is fresh preproc rather than
migration. Other uploaded repos are intentionally retired migration inputs; regenerate
them from the `lite/data/preproc` raw sources instead of adding another migration path.
A blast radius quoted over other published datasets is measuring rows migration can
never see.

The remaining compatibility tables in `devs/migration` are migration-window-only raw-boundary
readers. Path platform/task/split spellings point at the shared owners in
`LiteCUAMetadata` and `lite.data.staging.CANONICAL_SPLITS`; dialect aliases and the final-EOF
observation exemption stay local because they describe published legacy inputs, not a current
runtime contract. Delete those local tables with the one-time repair tool when the migration
window closes; do not copy them into runtime, export, upload, or training code.

## Complete Workflow

Steps 1-4 below were executed end to end on a real 6-row / 29-image Lite.ScaleCUA
subset. The upload, read-back, and SFT-export commands are the required follow-up
smokes for an approved publish cycle; run them against the exact revision and repo
you intend to publish.

### 0. Run From The Repo Root, And Freeze The Revision

`uv run` resolves the project from the working directory. `cd`-ing into a scratch dir and
running `uv run python -m devs.migration.run` fails with
`ModuleNotFoundError: No module named 'lite'`. Stay at the repo root and use absolute paths for
`-o` / `--out`.

```bash
COMMIT="$(git rev-parse --short HEAD)"
```

Freeze the code revision for the whole cycle. The migration output is only reproducible
against a named revision: `$COMMIT` tags the upload in step 5 and pins the read-back in step 6.
A resume must use the identical command at the identical revision.

### 1. Download The Source

```bash
uv run python -m lite.data.hf.download Lite.ScaleCUA \
    --org cua-lite \
    --out .data/hf/src/cua-lite/Lite.ScaleCUA
```

Use `--allow-patterns` to take a single shard while iterating. It bounds both the fetch and
the walk, so a warm HF cache holding shards from an earlier, differently-patterned pull cannot
drag those extra cohorts into the output.

**Why this step works on data that is by definition not yet canonical.** `download` is
**layout**-canonical only: it reshapes storage (HF shard groups merged into the local partition
layout, embedded image bytes extracted into the `ImageStore`, `messages` / `metadata` coerced
from their transport shape) and deliberately does **not** assert that row *content* matches the
current LiteSample schema. It cannot: `download` is the one entry point whose input is
*by definition* possibly-unmigrated — historical rows already published on HF — which is
migration's entire premise. A content gate here would demand the data be repaired before it
could be fetched *for repair*, making this very command unrunnable on exactly the rows below.

Content is gated downstream by `hf.stage`; upload and export assume that gate has already run:

| entry point | input | content-gated? |
|---|---|---|
| `hf.download` | historical HF data, possibly unmigrated | **no** — layout only |
| `hf.stage` | local log-root (fresh rollout, or post-migration) | **yes** — `validate_canonical_rows`; last content gate before upload |
| `hf.upload` | staged canonical tree | **no** — transport only: stats/card, image embedding, sharding, push |
| `export_sft` | canonical tree or trusted raw log-root | **no** — conversion/tokenization smoke, not a publish gate |

So a historical row with retired flat Lite tool calls (`{call_id,name,arguments}`),
JSON-string `function.arguments`, or missing/alias ids downloads fine; migration repairs
those before step 3's `stage` gate. Already-nested Lite rows are current-format data,
not legacy repair input: they belong in `stage`, not in this one-time migrator.

**Trap — point `run.py` at an `hf.download` root, not an `hf.stage` root.** Given a staging
tree, `run.py` picks up the sidecar `stats.json` / `repo.json` as row files and fails with
`ValueError: metadata must be an object`. The message does not name the real problem.

### 2. Migrate

```bash
uv run python -m devs.migration.run \
    .data/hf/src/cua-lite/Lite.ScaleCUA \
    -o .data/hf/migrated/cua-lite/Lite.ScaleCUA \
    --verify
```

Prints e.g. `{"files": 2, "rows": 6, "verified": 6, "dry_run": false}`. `--verify` runs
[verify.py](/devs/migration/verify.py) per row and is cheap — leave it on.

Use `--dry-run` first on an unfamiliar dataset: it runs the **full** migration (and
verification) in memory and **writes nothing** — `-o/--output` is accepted but ignored, and the
printed summary reports `"output_path": null`. So `--dry-run` answers "would this dataset
migrate and verify cleanly?"; producing the migrated tree needs a real run with `-o`.

**Trap — the migrated tree has no image store.** `run.py` walks only `.json` / `.jsonl` /
`.parquet`, so `images/` is not carried across and the next step cannot resolve image paths:

```bash
cp -r .data/hf/src/cua-lite/Lite.ScaleCUA/images \
      .data/hf/migrated/cua-lite/Lite.ScaleCUA/
```

### 3. Re-Stage — Required, Not Optional

`stage` consumes **rollout log-roots**, not canonical trees. Pointing `--log-roots` at the
migrated tree fails with `no trajectory.parquet under [...]`. So the real sequence is
`migrate → unstage → stage`:

```bash
# canonical tree -> log-root layout, ONE LOG-ROOT PER CONFIG
for CFG in rl train; do
  uv run python -m lite.data.hf.unstage \
      --dataset .data/hf/migrated/cua-lite/Lite.ScaleCUA \
      --log-root ".data/hf/logroot/$CFG" \
      --splits "$CFG" --config-names "desktop.use.$CFG"
done

uv run python -m lite.data.hf.stage \
    --log-roots .data/hf/logroot/rl .data/hf/logroot/train \
    --config-names desktop.use.rl desktop.use.train \
    --filter "lambda m: 'incomplete' not in (m.others.get('exclude_reason') or '').split(',')" \
    --name Lite.ScaleCUA \
    --out .data/hf/staged/cua-lite/Lite.ScaleCUA
```

**Trap — one log-root per config, not one call with every `--config-names`.** Passing
`--config-names desktop.use.rl desktop.use.train` to a single `unstage` writes both cohorts into
the same log-root, and `stage` maps log-roots to config names **1:1**. The merged root can only
be staged under one name, silently collapsing two cohorts into one. The loop above keeps them
separable. `--splits` is the rollout log-root subdir; using the registry split name keeps the
reconstructed root safe for resume. `unstage` also records each source row's canonical parquet
split as a transient `metadata.others["split"]` routing hint, and `stage` consumes and removes
that hint, so a restage preserves the source train/validation carve instead of re-drawing it
with this run's `--val-frac`. The current Lite.ScaleCUA published configs both source from
canonical `train`, so both restaged configs land under canonical `train`.

**`--filter` is required for full Lite.ScaleCUA.** `unstage` has no `--filter`, so the
publish-side row gate is `stage`'s. On the full published dataset, 460 of 17,953 rows fail that
gate with:

```
messages ended before role:tool result(s) for tool_call_id(s) ['call_0028']
```

Those rows end with a real GUI action whose observation was never published. Migration cannot
invent that screenshot. They are self-identifying by `exclude_reason` containing `incomplete`;
filter on the exact comma-separated tag (`.split(',')`), not on `not exclude_reason`. The latter
also drops rows tagged only with quality annotations such as `footgun:*`, `complex_shell`, or
`dependency_install`, which are canonical rows and should survive publication for downstream
consumers to filter. On the full run, expect `kept=17493 dropped_by_filter=460`.

Skipping this is not cosmetic. The upgrade path now writes partitions through
the same staging writer as fresh preprocessing, but historical migrated
artifacts may still carry pandas/pyarrow schema drift from older runs. Re-stage
migrated parquet before upload/export so migrated and freshly staged partitions
share the same canonical schema and `pyarrow.concat_tables(...)` succeeds.

As a control, the extra cycle is not itself suspected: for the all-train
Lite.ScaleCUA source carve above, `unstage → stage` over an already-staged tree
is row/schema identical. Do not generalize that to another dataset unless you
confirm the same split-routing invariants.

**Trap — `--dataset` must be the canonical `<root>/cua-lite/<name>` directory.** One level up
and image paths are derived from the wrong name.

### 4. Verify The Canonical Migration Shape

```bash
for CFG in rl train; do
  uv run python -m lite.infer.debug.log_contract ".data/hf/logroot/$CFG"
done
```

Expect **0** errors. On a real Lite.ScaleCUA migration this went from **3,320 → 0**;
the migrated rows also passed `verify.py` at **107/107**. Do not feed the nested output back
into `run.py`: migration is a one-time legacy-source repair path and rejects rows that are already
in the nested Lite storage shape.

**A clean §4 still is not a publish proof.** `verify.py` and `log_contract` are
raw/debug checks, while `stage` is the full content gate. On a final EOF unanswered
GUI call, all three should pass; `validate_canonical_rows` rejects only if the row
continues past that call without a result, or if a `role:"tool"` result is orphaned.

On the full 17,953-row Lite.ScaleCUA run, `17,953/17,953` migrated and verified while `stage`
dropped 460 rows after the explicit `incomplete` filter. Treat step 3's `stage` as the real
publish gate and read its `seen / kept / dropped_by_filter` line. The expected invariant is a
canonical migrated row: current metadata keys, valid image references, valid role sequence,
`id` / `tool_call_id` ownership, tool names, arguments, and action-batch grouping. Do not require
byte identity with a fresh preproc row when the source policy has diverged from the historical
published row.

### 5. Upload Transport

Upload the **re-staged** tree from step 3, never `run.py`'s output:

```bash
: "${HF_ORG:?set HF_ORG to your Hub user/org for private migration smoke repos}"
uv run python -m lite.data.hf.upload Lite.ScaleCUA \
    --org "$HF_ORG" \
    --private \
    --tag "$COMMIT" \
    --staging .data/hf/staged/cua-lite/Lite.ScaleCUA
```

Iterate against a private repo under your own namespace first (`--org "$HF_ORG" --private`),
and delete it afterwards. Use the release org only for
the approved final publish.

This is a transport smoke: it packages the already-staged canonical rows, embeds images,
pushes files, and tags the revision. It does not re-run migration invariants or row-content
validation; those are owned by `--verify`, `log_contract`, and step 3's `hf.stage` gate.

Pass `--tag "$COMMIT"` explicitly even though `upload` already tags by default: the default is
`git rev-parse --short HEAD` **at upload time**, and migration is a multi-step cycle. Switch
branches anywhere in the middle and the default tag names code that did not produce the data.
Freezing `$COMMIT` in step 0 removes that drift. A tag cannot carry a `-dirty` suffix, so commit
first or treat the upload as scratch. `--tag NONE` skips tagging only for throwaway smoke repos.

### 6. Read Back The Tagged Revision

```bash
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"

uv run python -m lite.data.hf.download Lite.ScaleCUA \
    --org "$HF_ORG" \
    --revision "$COMMIT" \
    --out "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA"
```

Pulling by `--revision "$COMMIT"` rather than `main` makes the cycle reproducible: it is the
same tree step 4 verified, not whatever the repo's default branch points at now. `download`
verifies layout only; row content was already checked by `stage` before publish transport.

### 7. Export An SFT Smoke

```bash
uv run python -m lite.train.export.export_sft \
    --config scripts/configs/qwen3_5/default/lite.osworld.yaml \
    --model-id Qwen/Qwen3.5-9B \
    --data-paths "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA" \
    --image-root "${CUA_LITE_DATASETS_ROOT}" \
    --filter "lambda m: not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) > 0.5" \
    --head 10 \
    --num-proc 1 \
    -o .data/sft/qwen3_5/lite-scalecua-smoke/train.parquet
```

`export_sft` is not another publish gate; it is a conversion/tokenization smoke for the exact
agent surface the trainer will consume. Use the rollout config for the source environment.
Lite.ScaleCUA rides the Lite.OSWorld desktop substrate, so the Qwen3.5 OSWorld rollout config is
the matching export config.

## Post-Stage Contract

No migration-owned representation should remain after the `migrate -> unstage -> stage` cycle.
Key order, empty `tool_calls`, terminal turns, image references, and `role:"tool"` result
ownership are canonical-output properties. A mismatch there is a migration bug, a staging bug,
or a source-policy divergence that cannot be recovered from the published row; it is not an
automatic byte-parity failure against fresh preproc.

## Gotchas Worth Knowing Before You Debug

- **`pd.read_parquet` yields numpy arrays.** `metadata_from_dict` then raises
  `ValueError: The truth value of an array with more than one element is ambiguous`. Use
  `coerce_meta` from `lite/data/staging.py` for metadata and
  `coerce_legacy_materialized_messages` from [upgrade.py](/devs/migration/upgrade.py)
  for legacy message structs. Skipping this has already produced two wrong measurements.
- **Diff exact fixtures at the JSON-string level, not with `==`.** Python dict equality ignores
  key order, which can hide construction-order drift in canonical migrated rows.
- **Terminal sidecars do not survive structural finals.** The shared preproc final message is a
  fresh content-only `Done.` turn; old terminal `inline_reasoning`, `action_description`, and
  other sidecars belong to the dropped old input marker, not to the migrated final.
- **Memory.** Parquet and image files can be hundreds of MB. Process one file at a time and
  `del data; gc.collect()` after extracting what you need.
