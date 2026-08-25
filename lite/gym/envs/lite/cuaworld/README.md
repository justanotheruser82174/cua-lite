# Lite.CUAWorld

`--env-id` `lite.cuaworld.ardour` · `lite.cuaworld.astroimagej` · `lite.cuaworld.blender3d` · … 40 in total (`gym.registry.registered_env_ids()`)

CUA-Lite re-host of [gym-anything](https://github.com/cmu-l3/gym-anything) (CMU
CUAWorld, MIT): **40 desktop software** (PyMOL, CAD, IDEs, GIS, sims, …) across
~25 domains, via `gym.make("lite.cuaworld.<software>@<task>")`. The cached
materials stay pristine; the runtime applies deterministic CUA-Lite execution
guards before uploading hooks, and setup-only material fixups never run on
export/post-task reward evidence hooks. The locked materials are a maintained
fork and may contain separately reviewed integration adaptations; upstream task
or verifier defects are documented rather than patched as part of this
integration. Split curation belongs in a separate reviewed materials change.
See [docs/envs.md](/docs/envs.md) for the env contract; the onboarded
list is [`main.py`](/lite/gym/envs/lite/cuaworld/main.py).

## Quick Start

Choose one install path for each software you plan to run:

```bash
# One-time repo bootstrap; rerun the env install script after any bare uv sync.
uv sync --all-extras

# Replace "pymol" with the software you plan to run; there is no default all-software install.

# Source path: provision that software's materials and build its image if missing/stale.
uv run --no-sync bash lite/gym/envs/lite/cuaworld/scripts/install.sh build pymol

# Or, published-image path: adopt that software's matching image, then provision materials.
# uv run --no-sync bash lite/gym/envs/lite/cuaworld/scripts/install.sh pull pymol

# Optional lifecycle helpers:
# uv run --no-sync bash lite/gym/envs/lite/cuaworld/scripts/install.sh status pymol       # read-only state check
# uv run --no-sync bash lite/gym/envs/lite/cuaworld/scripts/install.sh provision pymol    # materials + host verifier deps only
# uv run --no-sync bash lite/gym/envs/lite/cuaworld/scripts/install.sh rebuild pymol      # force an image rebuild
```

Requires Docker, network access to the public HF materials repo, and either package-index
access or preinstalled host verifier parser deps. `install.sh build <software>`
provisions only that software. Lifecycle verbs match lite.osworld/lite.scalecua:
plain `install.sh <software>` remains a compatibility alias for `build <software>`;
`provision` is docker-free materials + host verifier/VLM parser deps in the
active repo venv; `pull` checks the remote `lite.src_hash` first, adopts a
matching published image, then provisions; `status` is read-only. When a local
software image build is needed, the build path also ensures the shared CUAWorld
base image. Invoke it once per software you plan to run. The script does not
start an env-server and it does not configure model or judge
credentials. `uninstall.sh <software>` removes the image; `cleanup.sh` cleans
leaked direct-mode containers.

For an offline or locally edited materials checkout, bypass HF without changing
the lock:

```bash
# Set this only when validating a local materials checkout instead of the locked HF repo.
LITE_CUAWORLD_MATERIALS_REPO=/path/to/lite.cuaworld-assets \
  uv run --no-sync bash lite/gym/envs/lite/cuaworld/scripts/install.sh rebuild pymol
```

Local mode uses content-digested local-materials markers. Image freshness hashes
the selected subtree files staged into the Docker context; runtime cache
freshness hashes the full selected subtree, including `tasks/` and
`registered.json`. Image-context edits make the image stale while
`LITE_CUAWORLD_MATERIALS_REPO` is set; host-only task/verifier edits refresh the
materials cache without forcing a Docker rebuild. The local checkout is not
required to match the HF revision in the lock. Keep the variable set when
running that image. Remote image operations are disabled in this mode so local
materials cannot be confused with published images.

**Env-server mode (recommended)** — launch [`scripts/serve_env.py`](/scripts/serve_env.py) after exporting any provider
credentials the agent or host-side VLM judge needs; clients only set `CUA_LITE_ENV_SERVER_URL`:

```bash
CUAWORLD_ENVS="lite.cuaworld.pymol"  # space-separated env ids you installed
PORT=30100
uv run python scripts/serve_env.py --port "$PORT" --env-ids $CUAWORLD_ENVS
export CUA_LITE_ENV_SERVER_URL=http://$(hostname -I | awk '{print $1}'):$PORT
```

Use `localhost` only when the env-server and rollout client share the same network namespace.

**Direct mode** — `gym.make` brings the container up itself:

```python
import asyncio, lite.gym as gym
import lite.gym.envs.lite.cuaworld.main          # registers lite.cuaworld.*
env = gym.make("lite.cuaworld.pymol@abl_imatinib_binding_analysis")
asyncio.run(env.reset())
```

On first use, `CUAWorldServices.ensure` checks the `cua-lite/lite.cuaworld.<software>` image is
built and fresh against its `lite.src_hash` label (else raises with the install
command). Each episode gets a **fresh per-trajectory sandbox** container
(DEDICATED family), cleaned up when it ends.

## Direct Usage

```python
import asyncio, lite.gym as gym
import lite.gym.envs.lite.cuaworld.main          # registers lite.cuaworld.*
from lite.core.tools import make_tool_call

async def main():
    env = gym.make("lite.cuaworld.pymol@abl_imatinib_binding_analysis")
    res = await env.reset()            # boots the container, launches PyMOL
    print(res.text)         # task instruction
    res = await env.step([
        make_tool_call(
            "computer",
            {"actions": [
                {"action": "click", "coordinate": [500, 300]},
            ]},
            call_id="call_0000",
        ),
    ])
    print(res.reward, res.terminated)
    await env.close()

asyncio.run(main())
```

## Configuration

Shared registration defaults live in
[`configs/default.yaml`](/lite/gym/envs/lite/cuaworld/configs/default.yaml)
(`env_var_prefix: LITE_CUAWORLD`) — the computer baseline, fallback task count,
post-action delay, and per-step timeout, read via `env_config.load`. Swap the
whole file with `LITE_CUAWORLD_CONFIG=<abs-path | name-under-configs/>` — the
named form resolves against `configs/`, which currently ships `default.yaml`
only, so point it at an absolute path to override. Per-software `cpu` (and
`gpus` for blender3d) overrides are hardcoded at
`register_cuaworld_software(...)` in
[`main.py`](/lite/gym/envs/lite/cuaworld/main.py). See [the env config
contract](/docs/envs.md).

Task lifecycle limits start from the materials: each `task.json` carries upstream
`max_steps`, `timeout_sec`, and `hooks.pre_task_timeout`. A rollout config may
still intentionally set `env_kwargs.max_steps` (the GPT default/collect configs
use 30) to make a run's turn budget explicit and comparable; treat that cap as
part of the evaluation/collection protocol and record it with the rollout.
Timeouts remain task/verifier lifecycle guards, not model turn-budget knobs.
All fetched hooks run with the desktop session PATH, so their bare `python3` uses
the system interpreter where CUAWorld materials install dependencies; exec-stdio's
environment-only venv remains reserved for CUA-Lite helpers.

## Registered Tasks

```python
import lite.gym as gym
import lite.gym.envs.lite.cuaworld.main
print(gym.registry.task_ids("lite.cuaworld.pymol"))
```

Each env's `registered.json` carries the CUAWorld splits present in the current
gym-anything materials. Most softwares expose all three splits from the paper
(Aggarwal, Neubig, Welleck); known sparse onboarding exceptions include `knime`
and `freecad`, so query the registry instead of assuming every software has every
split:

| paper split | our split name | source (`<env>_split.json`) |
|---|---|---|
| CUAWorld **Train** | `train` | `train_tasks` |
| CUAWorld **Test**  | `eval`  | `test_tasks` |
| CUAWorld **Long**  | `long_horizon` | `additional_splits.long_horizon` |

Counts come from the `registered.json` files in the currently locked materials;
query the registry instead of copying a historical total. Select with `--splits`;
`--head N` samples the first N. If a run fixes `env_kwargs.max_steps`, keep that
cap consistent across compared runs and mention it in reported results.

## Evaluation

At episode end the task's upstream `post_task` hook and `verifier.py` run with
the full recorded trajectory. The verifier runs **host-side** (an adapter
subprocess), not in the container — it has to: the trajectory frames it samples
live on the host, and the VLM judge needs host credentials. It reaches into the
container through the `copy_from_env` / `exec_capture` RPCs in `env_info`. So a
verifier's `import` resolves against the REPO VENV, and the parser libraries the
upstream verifiers need (`ezdxf`, `nibabel`, `pyshp`, `astropy`, …) are installed
by `scripts/install.sh`, the same way lite.osworld installs its host-side
`osworld`/`desktop_env` evaluators.

Final reward follows upstream `reward_type`: sparse is binary, continuous is
`score / 100`, partial/rubric return the bounded score. **`dense`/`weighted` are
binary on a PASS (1.0, like sparse) and `score / 100` on a FAILURE, as shaped
partial credit** — dense reward would be accumulated per step by upstream
`reward_shaping`, and the locked materials declare none, so returning 0 there
discarded the verifier's verdict on 114 tasks (observed: a verifier reporting
`passed: true, raw_score: 85` scoring 0.0). Their `score` is not a percentage:
every registered, non-excluded dense/weighted verifier gates `passed` below 100
(thresholds 17-75, mean 63), and three return a raw count that caps at max_score
32/46/52 even on a flawless run — so `score / 100` on the passing side would pay
0.17-0.75 for a task the verifier certifies as solved, while the sparse tasks
alongside it in the same batch pay 1.0 for the same outcome.

Some verifiers are VLM-judged. The judge runs host-side in the evaluator process
(the env-server in remote mode, the rollout process in direct mode) and lets
litellm resolve provider credentials from that process environment. By default,
set `OPENAI_API_KEY` and leave `OPENAI_BASE_URL` unset. The default comes from
the env config loaded by the evaluator process (currently `gpt-5.5`); rollout YAML
`env_kwargs` do not change the verifier judge model. `VLM_MODEL` overrides
`LITE_CUAWORLD_VLM_MODEL`, which overrides `configs/default.yaml`.
Coverage counts are detector-dependent; run the current census before quoting a
percentage or count. Upstream defaults to a local Qwen3-VL on `localhost:8080`;
inheriting that here meant every VLM-judged verifier failed on any host without
that server, indistinguishably from an agent failure. To serve your own model
instead, name it and point at it with `VLM_MODEL` and `VLM_BASE_URL` (plus
`VLM_API_KEY` if it needs one). `VLM_MAX_RETRIES` is a compatibility name for
total provider attempts in this shim, and `VLM_TIMEOUT` is per attempt.
`LITE_CUAWORLD_VLM_MODEL` remains a CUA-Lite convenience override.
The shim preserves upstream's structured envelope contract for ordinary successful
`query_vlm` callers (`{success, response, parsed, error}`), and that envelope also
proxies common string methods to `response` for verifiers that call `.strip()` /
`.upper()` / `.lower()`, while `.get("schema_key")` and subscript reads fall back
to `parsed` for verifiers that treat the envelope as a parsed dict. Provider and
image failures raise `VLMProviderError` and are recorded through a verifier-worker
side channel, so both `except Exception` and bare `except:` verifier blocks surface
as invalid infrastructure failures rather than reward 0. For upstream JSON-returning
call sites, the shim returns the parsed object.

Verification has separate time budgets. Preparing verifier inputs (the
`post_task` hook, settle delay, and final screenshots) gets 180 seconds by
default. The verifier process gets at least 600 seconds, or enough time for the
configured `VLM_TIMEOUT` multiplied by those total attempts plus retry backoff.
`LITE_CUAWORLD_VERIFIER_TIMEOUT` explicitly overrides that verifier-process
budget. The configured `step_timeout` is the total deadline for one action or
terminal verification step; the default leaves room for preparation and the
default verifier budget. Increase `step_timeout` explicitly when configuring a
larger VLM timeout or attempt count; otherwise terminal verification fails fast as
an infrastructure timeout instead of being cancelled by the step deadline.

A one-task smoke rollout:

```bash
uv run python scripts/rollout.py --model-id gpt-5.5 \
  --env-id lite.cuaworld.pymol --splits eval --head 1 \
  --concurrency 1 --max-attempts 1 --save-data true \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --config-path scripts/configs/gpt/default/lite.cuaworld.yaml
```

## Development Notes

For normal use, prefer `install.sh build <software>` or `install.sh pull
<software>`.

<details>
<summary>Reproducible builds · layout</summary>

**Reproducible builds.** `install.sh` fetches `cua-lite/lite.cuaworld-assets` at the
commit locked in
[`data/assets.lock.yaml`](/lite/gym/envs/lite/cuaworld/data/assets.lock.yaml).
The lock file itself is part of the image freshness hash, so updating materials
is an explicit code change: bump the locked revision and rebuild affected
software images stamped with the new `lite.src_hash`.
The HF dataset, environment IDs, images, paths, configs, and environment
variables use the `lite.cuaworld` / `LITE_CUAWORLD` namespace.

**Layout.** cua-lite keeps only the engine + one register line per software; the
40 envs' content lives out-of-tree in the materials repo.

```
lite/gym/envs/lite/cuaworld/
├── main.py            # one register_cuaworld_software(...) per software (the onboarded list)
├── src/              # registration/runtime modules
│   ├── software.py   # registration core (reads configs, builds SandboxTaskConfigs)
│   ├── adapter.py    # runtime: run hooks from a file + host-side verifier bridge
│   └── vlm.py        # gym_anything.vlm shim (VLM-judged verifiers score, not crash)
├── configs/           # default.yaml (LITE_CUAWORLD_CONFIG takes an abs path)
├── scripts/           # install.sh / uninstall.sh / cleanup.sh (lifecycle)
├── docker/            # static Dockerfile + run_hooks.sh (per-software image build)
└── .cache/<software>/<upstream_env>/ # fetched materials (gitignored)
```

- [cua-lite/lite.cuaworld-assets](https://huggingface.co/datasets/cua-lite/lite.cuaworld-assets) — materials repo (HF dataset)
- Image stack: `cua-lite/lite.cuaworld.base` (the shared `lite/gym/sandbox/docker/Dockerfile.linux` built with `--build-arg USER=ga`, so the in-container unix user is `ga` — matching the upstream assets, which hardcode `ga`) → `cua-lite/lite.cuaworld.<software>` (runs that software's install hooks)
</details>

## Citation

Please cite the upstream work alongside [CUA-Lite](/README.md#citation).

```bibtex
@misc{aggarwal2026gymanything,
  title         = {Gym-Anything: Turn any Software into an Agent Environment},
  author        = {Pranjal Aggarwal and Graham Neubig and Sean Welleck},
  year          = {2026},
  eprint        = {2604.06126},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2604.06126}
}
```
