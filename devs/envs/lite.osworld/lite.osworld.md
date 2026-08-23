See [AGENTS.md](/devs/envs/AGENTS.md) for shared requirements.

**Goal:** Wrap [OSWorld](https://github.com/xlang-ai/OSWorld) (369 desktop tasks, 10 domains) as a cua-lite gym environment with `LiteDesktopActionSpace`, running on OSWorld Docker containers.

```bash
uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/install.sh
# Two-stage build: GNOME-Shell base + OSWorld-app additive. Use install.sh
# — it handles both stages with the right contexts:
uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/install.sh         # build if missing
uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/install.sh rebuild # force
```

---

## File layout

```
lite/gym/envs/lite/osworld/
├── main.py                          # LiteOsworldEnv, register "lite.osworld"
├── src/
│   ├── utils/                       # setup, dispatch, verify
│   ├── eval/                        # evaluator runtime (runner + metrics)
│   └── gen/
│       ├── eval/__main__.py         # OSWorld JSON → eval.jsonl
│       └── train/
│           ├── __main__.py          # Track A synth + Track B perturb CLI
│           ├── common.py           # NOISE_CANDIDATES + LO_SAVE_POSTCONFIG
│           ├── synth/               # Track A: per-domain templates
│           │   └── _utils.py        # SynthTemplate, make_synth_row
│           └── perturb/             # Track B: per-domain perturb functions
│               └── _utils.py        # KnobSpec, make_perturb_row
├── data/
│   ├── eval.jsonl                   # 369 eval tasks (sha256-locked)
│   ├── train.synth.jsonl            # Track A: programmatic generation
│   └── train.perturb.jsonl          # Track B: structural perturbation of eval
└── docker/
    └── Dockerfile                   # additive: LibreOffice/Chrome/etc + OSWorld Flask server + appearance, FROM cua-lite/sandbox.linux (shared base)
```

Validation script: [`devs/envs/lite.osworld/validate/oracle/validate.py`](/devs/envs/lite.osworld/validate/oracle/validate.py).

Regenerate data:
```bash
uv run python -m lite.gym.envs.lite.osworld.src.gen.eval
uv run python -m lite.gym.envs.lite.osworld.src.gen.train
```

Never hand-edit JSONL — fix the generation script and regenerate.

---

## Training-set design

### Two tracks

| Track | File | Description | Rows |
|---|---|---|---|
| **A. Synthetic** | `train.synth.jsonl` | Programmatic (setup, eval) pairs from templates | 1722 |
| **B. Perturbation** | `train.perturb.jsonl` | Structural variations of real eval tasks | 707 |

### Hard requirements

1. **IID distribution** — train `(setup_class, eval_class)` approximates eval. Max ratio <= 2x per domain.
2. **Positive reward** — every task emits reward=1.0 when oracle acts correctly. Evaluator Tier 1/2 only.
3. **App pre-launched** — every task config must `launch`/`open` the relevant app before the agent starts, matching eval distribution.
4. **No eval leakage** — no train instruction identical to any eval instruction.

### Evaluator tiers

**Tier 1 (safe):** `compare_table` (sheet_data/freeze/zoom), `compare_csv`, `compare_pdfs`, `check_json`, `exact_match`, `check_include_exclude` (vm_file), `compare_pptx_files` (text/structure), `compare_docx_files` (text), `check_transition`.

**Tier 2 (use with care):** `compare_table` (chart/style), `check_include_exclude` (vm_command_line), `compare_pptx_files` (color), `compare_docx_files` (formatting).

**Tier 3 (banned):** `compare_images` (pixel-level), `compare_table` (pivot_table), network-dependent, `check_accessibility_tree`, `infeasible`.

---

## Verification levels

Execute in order: **L1 → L2 → L3 → L4**.

| Level | What | Cost |
|---|---|---|
| **L1** | Row-by-row feasibility: instruction ↔ oracle, params match, app launched | Instant (no Docker) |
| **L2** | Oracle reward: setup → oracle → evaluator = 1.0 | ~30s/task |
| **L4** | Agent run: baseline agent achieves reward > 0 | ~5-10min/task |

### L2 — validate a full split

```bash
uv run python devs/envs/lite.osworld/validate/oracle/validate.py \
  --fixtures lite/gym/envs/lite/osworld/data/train.synth.jsonl --concurrency 4

uv run python devs/envs/lite.osworld/validate/oracle/validate.py \
  --fixtures lite/gym/envs/lite/osworld/data/train.perturb.jsonl --concurrency 4
```

`--concurrency 4` max (higher causes GIMP action-history race). Container safety: only kill containers with prefix `lite-env-local-_validate_*`.

### L4

```bash
uv run python scripts/rollout.py \
  --model-id gpt-5.5 --env-id lite.osworld --splits train --concurrency 4 \
  --config-path scripts/configs/gpt/default/lite.osworld.yaml
```

> **Always pass `--config-path scripts/configs/<agent>/default/<env>.yaml`.** The per-agent rollout config pins API sampling kwargs (`max_output_tokens`, `reasoning_effort`, etc.) and env defaults; without it, rollout falls back to generic defaults that shift the eval result.

Target: 1-2 tasks per template family. Distinguish agent capability issues (no fix) vs task design bugs (must fix).

---

## Lessons learned

These bugs were discovered during verification. Document here so future templates avoid them.

### 1. Retired: host/docker RNG mismatch in runtime synth

The old `param_fn(seed)` vs docker-side `synth_<cmd>(seed)` warning applied to
the runtime synthesis machinery that was purged by `45984c1a5`. Modern synth
generation is host-side at codegen time, so do not resurrect the old pattern of
calling deleted runtime `synth_*` helpers from documentation snippets.
If a new template has host/runtime parameter drift, fix the codegen owner that
builds the row instead of adding a docker synth read-back shim.

### 2. Multi-file templates: distinct seeds per file

When generating two+ documents, each must use a different seed. Same seed = identical files = trivial/contradictory task.

### 3. Instruction must embed all parameters the agent cannot infer from screen

Find-and-replace must name exact words. Multi-file ops must describe each file separately. Reorder columns must state the ordering criterion. Slide ops must state exact slide number and value.

### 4. `extra_config_steps` must NOT contain `launch`/`open` steps

`synth/_utils.py:make_synth_row` checks the full `config` list:
```python
already_has_app_open = any(s.get("type") in ("launch", "open") for s in config)
if open_cmd and not already_has_app_open:
    config.append({"type": "launch", ...})
```
If ANY `launch`/`open` step is already in `config` (including those from `extra_config_steps`), `open_command` and `DOMAIN_DEFAULT_OPEN` are silently skipped. Use `extra_config_steps` for `command`/`execute` steps only.

### 5. Perturb rename: `old_name` must be the setup source, not the evaluator target

For `_perturb_dir_rename`, the evaluator checks the **target** directory. The agent starts from the setup-created **source**. Extract `old_name` from the oracle's `mv SRC DEST` command.

### 6. Boolean toggle inversion: check for contradictions

After regex inversion, check if the instruction is self-contradictory. If `new_instruction == instruction` (no change), skip — phrasing too complex to invert safely.

### 7. VS Code perturb: update `result.command` when perturbing extension IDs

Must update both `expected["rules"]["expected"]` AND `result["command"]` (the `grep` invocation). Otherwise evaluator greps for the original extension ID and always returns 0.

### 8. `oracle_after_postconfig` for file-save tasks

When a task uses `LO_SAVE_POSTCONFIG` (Ctrl+S) AND oracle places expected file by `cp`, set `oracle_after_postconfig=True`. Otherwise LibreOffice's save overwrites the oracle-placed file.

### 9. Ordinal dictionaries must cover the actual range

Extend to >= 8 values with `dict.get(n, f"{n}th")` fallback. `{1: "first", 2: "second", 3: "third"}` breaks when slide index is 4+.

### 10. `_skip` in `param_fn` for invalid parameter combinations

When `param_fn` produces invalid params during host-side codegen, return `{"_skip": True}`. The generation loop skips that seed. Never filter by post-processing the JSONL.
