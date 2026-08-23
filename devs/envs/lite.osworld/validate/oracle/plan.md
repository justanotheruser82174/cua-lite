# OSWorld oracle validation plan

## Quick reference

| Item | Value |
|---|---|
| Task data | `lite/gym/envs/lite/osworld/data/{train.synth,train.perturb,eval}.jsonl` |
| Catalog lock | `lite/gym/envs/lite/osworld/data/catalog.lock.json` |
| Validate script | `devs/envs/lite.osworld/validate/oracle/validate.py` |
| Perturb generators | `lite/gym/envs/lite/osworld/src/gen/train/perturb/<domain>.py` |
| Synth generators | `lite/gym/envs/lite/osworld/src/gen/train/synth/<domain>.py` |
| Eval generators | `lite/gym/envs/lite/osworld/src/gen/eval/<domain>.py` |
| Perturb dispatcher | `lite/gym/envs/lite/osworld/src/gen/train/perturb/dispatch.py` → `apply_structural_perturbation` |
| Per-task instruction patches | `lite/gym/envs/lite/osworld/src/gen/train/perturb/dispatch.py` → `_INSTRUCTION_PATCHES` |
| Infeasible/excluded rows | eval `metadata.others.exclude_reason`, perturb dispatcher infeasible skip, domain-level skip/drop logic, and synth `catalog.py` → `_HARD_TEMPLATE_IDS` |
| Findings log | `devs/envs/lite.osworld/validate/oracle/logs.md` — one line per finding, never deleted |

---

## Oracle's role

The oracle's only purpose is to **prove the task is solvable**: given the correct initial state, some action sequence reaches the eval's expected result.

**Never invert the logic** — do not corrupt setup/postconfig/eval just to make oracle actions pass:
- ❌ Loosen eval so anything passes
- ❌ Pre-apply the target state in setup so oracle has nothing to do (trivial_pass)
- ❌ Use postconfig to rewrite the file into the format oracle produces rather than what an agent should produce

If oracle fails, first ask: **is the task actually solvable?** If yes, fix oracle_actions / oracle_trajectory, or fix setup so the initial state is genuinely "not done".

---

## Validation flow (per task)

> Infeasible rows are auto-skipped: the validator drops any row flagged
> `metadata.others.exclude_reason` (e.g. `"infeasible"`) or carrying neither
> `oracle_actions` nor `oracle_trajectory` — they can't be replayed and would
> only add false failures. So a plain run validates exactly the solvable set;
> no flag is required.

Each task goes through four steps:

1. **setup** — launch Docker container, run `config` steps
2. **pre-oracle eval (trivial_pass check)** — eval must return 0.0; if it returns 1.0, the initial state already satisfies the condition → invalid task
3. **oracle replay** — execute `oracle_actions` or `oracle_trajectory`
4. **post-oracle eval** — eval must return 1.0

---

### Step 1 — Run validation

> The examples below use `train.perturb` — replace with `train.synth` or `eval` as needed.

```bash
uv run python devs/envs/lite.osworld/validate/oracle/validate.py \
    --fixtures lite/gym/envs/lite/osworld/data/train.perturb.jsonl \
    --concurrency 4 --retries 3 \
    --report /tmp/validate_train_perturb.report.jsonl
```

Filter failing tasks from the report:

```bash
python3 -c "
import json
for line in open('/tmp/validate_train_perturb.report.jsonl'):
    r = json.loads(line)
    if not r['passed']:
        print(r['task_id'], '|', r['message'][:120])
"
```

Debug a single task:

```bash
uv run python devs/envs/lite.osworld/validate/oracle/validate.py \
    --fixtures lite/gym/envs/lite/osworld/data/train.perturb.jsonl \
    --filter perturb_osworld_chrome_44ee5668_3a5cf36b \
    --retries 1 \
    --report /tmp/debug.report.jsonl
```

---

### Step 2 — Triage failures

#### trivial_pass — initial state already satisfies the condition

Cause: setup/config pre-applies the target state, or the eval logic is too permissive.

Fix (in priority order):
1. Fix the generator's config so the initial state is genuinely "not done"
2. Use `sys.exit(42)` in the generator to skip variants where a valid initial state cannot be constructed (generator automatically excludes that task)
3. For structurally unfixable eval tasks, mark the eval row with the appropriate
   `metadata.others.exclude_reason`; perturb rows for infeasible eval bases are
   skipped by `perturb/dispatch.py`

**Do not** tighten eval to eliminate trivial_pass — that masks the root cause.

#### oracle fail — oracle actions cannot achieve the goal

Possible causes: wrong file path in oracle, timing issue, initial state doesn't match oracle's assumptions.

Fix:
1. Use an interactive container (see Step 3) to manually reproduce the oracle operations and confirm the task is solvable
2. Fix oracle_actions / oracle_trajectory in the generator script
3. If the task is structurally unsolvable inside Docker, route it through the
   current owner: eval `exclude_reason`, a domain-level perturb skip/drop, or
   synth `catalog.py` `_HARD_TEMPLATE_IDS`

---

### Step 3 — Debug (interactive container)

For complex failures (unexpected setup behavior, oracle timing issues, eval logic), start a persistent container and operate manually:

```bash
docker run -d --name debug-osworld \
  --memory 8g --cpus 2 \
  -p 8100:8000 -p 8101:6901 -p 5100:5000 \
  -e VNC_PW=password -e VNCOPTIONS=-disableBasicAuth \
  -e VNC_RESOLUTION=1920x1080 \
  cua-lite/lite.osworld:latest

# View desktop: http://localhost:8101
```

Manually execute setup config steps via OSWorld Flask server:

```bash
# Execute a shell command (type=execute config step)
curl -s -X POST http://localhost:5100/execute \
  -H "Content-Type: application/json" \
  -d '{"command": "ls /home/user", "shell": true}' | python3 -m json.tool

# Download a file (type=download config step)
curl -s -X POST http://localhost:5100/execute \
  -H "Content-Type: application/json" \
  -d '{"command": "wget -O /home/user/Desktop/test.docx https://example.com/test.docx", "shell": true}'

# Read file content to verify state
curl -s "http://localhost:5100/file?path=/home/user/.config/google-chrome/Default/Cookies" | xxd | head
```

Enter the container directly:

```bash
docker exec -it debug-osworld bash
```

Cleanup when done:

```bash
docker rm -f debug-osworld
```

---

### Step 4 — Fix → Regen → Revalidate

> **NEVER directly edit `.jsonl` files.** They are generated idempotently by generator scripts.
> Any hand-edit will be silently overwritten on the next regen and breaks the catalog lock.
> Always fix the `.py` generator script, then regen.
>
> **Keep `.py` ↔ `.md` in sync.** When you edit a perturb generator under
> `lite/gym/envs/lite/osworld/src/gen/train/perturb/<domain>.py`, also update its
> co-evolved spec at `devs/envs/lite.osworld/perturb/<domain>.md` (per-task tables,
> archetype rows, paraphrase pools, infeasible lists, expected row counts). The two files
> are the source of truth together — drift between them is the most common cause of stale
> assumptions in later audit cycles.

#### 4a. Fix the generator script

| Problem type | Fix location |
|---|---|
| perturb config/oracle wrong | `src/gen/train/perturb/<domain>.py` |
| perturb specific base task needs a wording-only patch | `src/gen/train/perturb/dispatch.py` → `_INSTRUCTION_PATCHES` |
| perturb base task structurally unsolvable | `src/gen/train/perturb/dispatch.py` infeasible skip, or the owning domain's skip/drop list |
| synth config/oracle wrong | `src/gen/train/synth/<domain>.py` |
| eval config/oracle wrong | `src/gen/eval/<domain>.py` or `src/gen/eval/__main__.py` |
| eval task structurally trivial | owning `src/gen/eval/<domain>.py` oracle metadata / `exclude_reason` |

Note: `task_id` suffix is `md5(knob_assignment)[:8]`. After regeneration the suffix may change due to RNG drift — this is expected and fine as long as the generator logic is correct.

#### 4b. Regenerate the affected JSONL split

```bash
# train.synth.jsonl
uv run python -m lite.gym.envs.lite.osworld.src.gen.train --track synth

# train.perturb.jsonl
uv run python -m lite.gym.envs.lite.osworld.src.gen.train --track perturb

# eval.jsonl
uv run python -m lite.gym.envs.lite.osworld.src.gen.eval
```

Update the catalog lock after each regen:

```bash
uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/utils/tasks.sh refresh-lock
uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/utils/tasks.sh check
```

#### 4c. Revalidate

Re-run Step 1 scoped to the fixed tasks:

```bash
uv run python devs/envs/lite.osworld/validate/oracle/validate.py \
    --fixtures lite/gym/envs/lite/osworld/data/train.perturb.jsonl \
    --filter <task_id_substring> \
    --retries 3 \
    --report /tmp/revalidate.report.jsonl
```

Expected: all previously failing tasks now pass. If any still fail, iterate from Step 2.
