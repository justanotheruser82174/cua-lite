# Shared Guidelines for Adding New Agents

Shared workflow for all agents. Agent-type-specific details (GPU setup, API keys, CLI commands) go in:
- **Local VLM agents:** [`local/AGENTS.md`](/devs/agents/local/AGENTS.md) + per-agent specs (`local/<agent>.md`)
- **API agents (Claude/GPT):** [`api/AGENTS.md`](/devs/agents/api/AGENTS.md) + per-agent specs (`api/claude.md`, `api/gpt.md`)

## Worktree-Based Development

All development MUST happen in a **git worktree** on a dedicated branch:

```bash
git worktree add ../cua-lite-<agent> <agent>
cd ../cua-lite-<agent>
git submodule update --init
uv sync --extra quick-start  # then run per-env scripts/install.sh
```

## Development Steps

### 1. Implement

Create the required files under `lite/agents/`. See [`local/AGENTS.md`](/devs/agents/local/AGENTS.md) or [`api/AGENTS.md`](/devs/agents/api/AGENTS.md) for the specific file list.

### 2. Test

Add test cases under `tests/agents/`.

### 3. Add Launch Entry

Add the agent to the appropriate example script. See [`local/AGENTS.md`](/devs/agents/local/AGENTS.md) or [`api/AGENTS.md`](/devs/agents/api/AGENTS.md) for details.

### 4. Verification

Run a quick sanity check with small `max_steps`. See [`local/AGENTS.md`](/devs/agents/local/AGENTS.md) or [`api/AGENTS.md`](/devs/agents/api/AGENTS.md) for the exact command.

#### 4.1 Data Flow Consistency

Ensure the data flow matches the agent's reference implementation (listed in each agent's spec file).

#### 4.2 Log Validation

Inspect per-turn artifacts under `<log-root>/.../sample_NN/turn_NNNN/`:

- `prompt_images/*.png` — optional debug prompt-image cache; absent when debug artifacts were disabled
- `prompt_images_annotated/*.png` — action overlays with the same basename as the prompt image they annotate; written only for turns with coordinate actions, and only with debug artifacts on
- `01_prompt.txt` — full prompt sent to model
- `02_response.txt` — raw model output
- `03_actions.json` — agent_message, lite_message, executed_actions
- `04_results.json` — step result: reward, terminated, truncated, results, info
- `05_timing.json` — per-turn timing data; absent when the turn recorded no timings

Canonical trajectory images live at `sample_NN/images/*.png` and are referenced
from `trajectory.parquet`; visual result images in `04_results.json` must point
back to that store with `image_index`. Legacy readers may accept per-turn
`images/`, `annotated/`, `05_results.json`, `06_timing.json`, or
`result_images/`, but do not treat those names as current layout.

When a batch is used as a reference log, pair by config path, env id, agent
family, runtime mode, and task id. Keep the complete log root and sample
directory; `summary.json` alone is only a score aggregate.

### 5. Stress Test

Run parallel evaluation across all supported environments. See [`local/AGENTS.md`](/devs/agents/local/AGENTS.md) or [`api/AGENTS.md`](/devs/agents/api/AGENTS.md) for the exact command.

#### 5.1 General Notes

- **Coordinator/subagent boundary:** The coordinator owns persistent repo edits, commits, env-server lifecycle, and rerun decisions. Subagents may audit completed logs, inspect artifacts, run probes/replays, and return fix proposals in parallel; they must not start env-servers, write persistent files, or claim a fix landed before the coordinator applies and verifies it.
- **Batch rollout discipline:** Full stress batches must run through the environment's env-server workflow. Direct rollout mode is for small smoke checks only. The blocking checks in [Action Space Verification](#52-action-space-verification) must pass before a batch is considered complete.
- **Env-specific verification:** No deadlocks, file-lock contention, or resource leaks under concurrency. Port allocation, emulator pool, and Docker container management work correctly at scale. Memory and GPU usage remain stable across all parallel instances. All instances produce valid observations and clean up on close.
- **Scope of fixes:** Don't only fix agent-side issues. If an error originates in the env layer but can be resolved by modifying cua-lite code (e.g. action dispatch, coordinate conversion, env wrappers, port conflicts), fix it too. RL training requires massive parallel sampling, so any issue that surfaces during parallel runs is worth solving now.
- **Counter reset rule:** When a task fails, fix the issue and re-run the exact failed case(s) to confirm the fix. Do not simply start a fresh batch of random tasks — the previously failed cases must pass after the fix before continuing.
- **Zombie cleanup rule:** Periodically clean up zombie or dead processes (sglang, envs, containers, emulators, etc.) between runs. Lingering processes waste resources and can cause port conflicts or OOM in subsequent tasks.

#### 5.2 Action Space Verification

> **NON-NEGOTIABLE — this is a blocking gate.** No batch is "done" until all three checks below pass for every turn in that batch. "Looks fine" without opening the actual artifacts does not count. Skipping this step lets action conversion bugs silently corrupt RL training data and waste GPU hours.
> **Parallelism:** Use subagents to analyze completed logs in parallel, but do not launch the next batch until the blocking checks for the current batch pass.

For every image-bearing debug turn, open matching `prompt_images/<name>.png` and `prompt_images_annotated/<name>.png` pairs when present and verify:

1. **Visual correctness** — each action produces the expected on-screen effect (scroll direction/units, drag vectors, key combos, click targets, etc.)
2. **Action conversion chain** — open `03_actions.json` to trace each action through `agent_message → lite_message → executed_actions`. Verify both legs of the conversion:

   **Leg 1 — Adapter output normalizes to CUA-Lite spec** (`agent_message → lite_message`):
   - Inspect `lite_message`: canonical stored key actions carry `keys: list[str]`; raw model strings are ingress only. Every field matches the type declared in `LiteDesktopActionSpace` / `LiteBrowserActionSpace` / `LiteMobileActionSpace` (e.g., `coordinate` must be `list[int]` within `[0, 1000]`; `amount` must be `int > 0`).
   - Watch for models outputting shorthand that the adapter passes through unchecked. Known edge cases:
     - `key(keys="enter")` — model outputs string instead of list; the shared action-space owner must normalize it to `["enter"]`.
     - `key(keys="ctrl+F")` — combo key as single string; the shared action-space owner must split on `+` and normalize casing → `["ctrl", "f"]`.
     - `key(keys="ctrl++")` — trailing literal plus key must normalize through the shared action-space owner to `["ctrl", "+"]`.
     - `key(keys=["ctrl+F"])` is not a canonical shortcut form; reject it or fix
       it at the raw source boundary instead of treating lists as chord strings.
   - No extra or missing required fields.

   **Leg 2 — CUA-Lite action maps to correct env command** (`lite_message → env execution`):
   - Units are consistent across the boundary (e.g., CUA-Lite `scroll.amount` is in wheel-click units; if the env API expects pixels, the env must convert using the correct factor).
   - Semantic intent is preserved (e.g., `type(text)` should only type text — it must not implicitly press Enter if the agent controls Enter via a separate `key(["enter"])` call).
   - All CUA-Lite parameters are forwarded (e.g., `scroll.amount` must not be silently dropped, `click.button` / `click.clicks` must map to the correct env-side variant).
   - Coordinate systems are converted correctly (`[0, 1000]` normalized → env-native pixel coordinates).

   If the on-screen result doesn't match intent, debug through both the adapter and the env implementation (either may contain bugs).
3. **Multi-action parsing** — when the model outputs multiple desktop/mobile actions in a single turn, they do **not** appear as separate top-level tool calls. `use`-task actions are nested inside one canonical `computer` / `mobile` action-batch call: `lite_message.tool_calls` reads `[{"id": …, "type": "function", "function": {"name": "computer", "arguments": {"actions": [{"action": "click", …}, {"action": "type", …}]}}}]`. Confirm every action the model emitted survives, in order, inside `function.arguments.actions[]`; core action-space inspection (`unpack_action_batch_call` in `lite/core/tools/action_space/batches.py`) expands them into the individual `computer.interface` calls you see in `executed_actions`

### 6. Commit

Commit on the agent branch inside the worktree. Do NOT merge into `zzh`.
