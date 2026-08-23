# Cua envs — `cua.sandbox` & `cua.bench`

Two ways to run computer-use on the **[Cua](https://github.com/trycua/cua) ecosystem** — Cua provisions
the machine (a local Docker/QEMU sandbox by default, or Cua cloud), so there's no cua-lite image to build.

- **[`cua.sandbox`](#cuasandbox--open-ended-instruction)** — *open-ended*: give a free-form instruction and drive a
  desktop/phone. No task, no reward (`reward=None`); the episode ends on the agent's `terminate` or `max_steps`.
- **[`cua.bench`](#cuabench--benchmark-tasks-gymmake)** — *task-oriented*: run [cua-bench](https://github.com/trycua/cua/tree/main/libs/cua-bench)
  benchmark tasks with cua-lite agents + scoring; reward comes from cua-bench's evaluator.

## Contents

- [Install](#install)
- [`cua.sandbox` — open-ended instruction](#cuasandbox--open-ended-instruction)
- [`cua.bench` — benchmark tasks (`gym.make`)](#cuabench--benchmark-tasks-gymmake)
  - [Backend modes (`webtop` / `local`)](#backend-modes-webtop--local)
  - [Reproducing the KiCad score (6/25)](#reproducing-the-kicad-score-625)
- [Notes & caveats](#notes--caveats)

## Install

```bash
uv run --no-sync bash lite/gym/envs/cua/scripts/install.sh   # pip-installs cua-sandbox + cua-bench (no cua-agent), pulls trycua/cua-xfce
export CUA_API_KEY="..."     # ONLY for Cua cloud; the default local mode needs no key (just Docker)
```

## `cua.sandbox` — open-ended instruction

A direct `CuaSandboxEnv` class (not `gym.make`-able). The instruction and platform are constructor
kwargs; **it runs locally by default** (Docker/QEMU, no key) — set `local=False` for Cua cloud.

```python
import asyncio
import lite.agents as agents
from lite.gym.envs.cua.sandbox import CuaSandboxEnv

env = CuaSandboxEnv(instruction="open a terminal and check the disk usage")   # local Linux container
# platform="macos" | "windows" | "android"      — pick the OS (android → mobile action space)
# local=False                                    — run on Cua cloud instead (needs CUA_API_KEY)
# image_kwargs={"kind": "vm"}                     — full VM instead of a container
# sandbox_kwargs={"region": "us-west-2", "memory_mb": 8192}   — cloud resources / runtime

agent = agents.make("gpt-5.5", env=env)
result = asyncio.run(agent.sample(env))
print(result.episode_return, result.terminated)   # episode_return is 0.0 (no evaluator); ends on terminate()/max_steps
```

Desktop platforms use `LiteDesktopActionSpace`; **Android** uses
`LiteMobileActionSpace` (`tap`/`long_press`/`swipe`/`pinch`/`system_button` →
`sb.mobile.*`).
Cua screenshots do not include a host cursor. `cua.sandbox` is constructed
directly, so its constructor default is sourced from
`sandbox/configs/default.yaml` `make_kwargs.cursor` (default true) and the env
composites the shared Linux cursor sprite in the capture path. Pass
`cursor=False` to `CuaSandboxEnv(...)` when comparing raw frames.

## `cua.bench` — benchmark tasks (`gym.make`)

cua-bench is a 3-level suite: **dataset** (`kicad`, `basic`, …) → **environment** (`154d0750`,
`click-button`, …) → **variant**. Mapping onto cua-lite's `env_id@task_id` model, **each dataset is one
env_id — the collection** — and the **task_id** is a specific task: the environment (single-variant, e.g.
`154d0750`) or `environment/variant` (e.g. `click-button/0`). The backend mode is part of the env_id
(`cua.bench.<mode>.<dataset>`). `install.sh` downloads the datasets, so they're available with no export.

```python
import asyncio, lite.gym as gym, lite.agents as agents

gym.registry.registered_env_ids()   # → ["cua.bench.local.kicad", "cua.bench.local.basic", ...] (one per dataset)
gym.registry.task_ids("cua.bench.local.kicad")   # → {"eval": ["154d0750", "1625e97a", ...]}  (task = environment)

env = gym.make("cua.bench.local.kicad@154d0750", max_steps=200)   # a real KiCad container
agent = agents.make("gpt-5.5", env=env)
result = asyncio.run(agent.sample(env))
print(result.episode_return, result.terminated)   # episode_return = cua-bench reward (solved ≥ 0.5)
```

CLI (same as any benchmark env — run a whole dataset or a single task):

```bash
uv run python scripts/rollout.py --model-id gpt-5.5 --env-id cua.bench.local.basic --splits eval --concurrency 8
uv run python scripts/rollout.py --model-id gpt-5.5 --env-id cua.bench.local.kicad --task-id 154d0750
```

### Backend modes (`webtop` / `local`)

A backend's lifecycle family is **per-env_id**, so the mode is part of the id. Each dataset registers
under the mode its environments declare (shipped datasets are all `native` → `local`):

| env_id | backend | needs | notes |
|---|---|---|---|
| **`cua.bench.webtop.<dataset>`** | in-process Playwright HTML desktop | nothing (no Docker) | PURE; fastest |
| **`cua.bench.local.<dataset>`** | local `trycua/cua-xfce` **container** | Docker | DEDICATED; cleanup-managed; concurrent OK, except KiCad (per-task apt-install → `--concurrency 1`) |

`local` boots a real desktop, so the first `reset()` is slow (container boot +
any app install). For concurrent local runs, use env-server mode; it handles
per-session cleanup.

`cua.bench` registers `bench/configs/default.yaml` `make_kwargs.cursor`
(default true) through `gym.make` for env-owned cursor compositing on returned
screenshots. The webtop and local modes should be validated with both
`cursor=True` and `cursor=False` when changing capture or action dispatch.

### Reproducing the KiCad score (6/25)

trycua reports **GPT-5.5 = 6/25 solved, mean 0.240** on the cua-bench KiCad EDA suite
([blog](https://github.com/trycua/cua/blob/main/blog/evaluating-gemini-3.5-flash-on-computer-use.md)).
KiCad is the dataset **`cua.bench.local.kicad`** (25 tasks, one per environment; each a real container).
After `install.sh`, run gpt-5.5 on each task and tally (scoring matches cua-bench's own `runners.py`:
reward from `evaluate()`, solved = `reward >= 0.5`).

```python
import asyncio, lite.gym as gym, lite.agents as agents

async def run(task_id):
    env = gym.make(f"cua.bench.local.kicad@{task_id}", max_steps=200)   # boots trycua/cua-xfce + installs KiCad
    agent = agents.make("gpt-5.5", env=env)
    try:
        return (await agent.sample(env)).episode_return
    finally:
        await env.close()

async def main():
    tasks = gym.registry.task_ids("cua.bench.local.kicad")["eval"]   # 25 KiCad tasks (needs install.sh)
    rewards = [await run(t) for t in tasks]
    solved = sum(1 for r in rewards if (r or 0) >= 0.5)
    print(f"{solved}/{len(tasks)} solved, mean {sum(r or 0 for r in rewards)/len(tasks):.3f}")

asyncio.run(main())   # → ~6/25 solved, mean ~0.23 (matches official 6/25, 0.240)
```

> **Concurrency.** Native containers run concurrently fine for datasets with **no per-task install**
> (basic/webtop — e.g. `--concurrency 8`). **KiCad is the exception: run it at `--concurrency 1`** — its
> per-task `apt install` streams through cua's `run_command`, which truncates under concurrent load
> (verified: `--concurrency 4` scored the canary `154d0750` at 0.0 + reset errors, vs 0.667 sequential).

## Notes & caveats

- **Auth**: only Cua **cloud** needs `CUA_API_KEY`. The default local mode (`cua.sandbox`, `cua.bench.local`)
  needs just Docker.
- **Drag-and-drop**: `DragAction` is a synthetic mouse drag (mousedown→move→up); it doesn't reliably
  trigger **HTML5 native drag-and-drop**, so drag-drop-style tasks score 0 on both webtop and local
  (verified). Plain mouse drags (sliders) work fine — this is a synthetic-drag limitation, not a bug.
- **Search / CAPTCHA**: headless cloud browsers from datacenter IPs get CAPTCHA'd by Google/Bing — prefer
  a search API or DuckDuckGo/Brave in your instruction.
