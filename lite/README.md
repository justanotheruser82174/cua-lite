# `lite/`

`lite/` is the runtime library for CUA-Lite. It contains the shared data
contracts, agent runtimes, environment runtimes, rollout/eval code, and training
utilities used by the top-level scripts.

This README is a user-facing map. It explains where concepts live and which
files are intended entry points. It is not a work log, status tracker, or place
for dated developer notes.

## Quick Start

Most users enter the package through scripts:

- rollout/eval: [`scripts/rollout.py`](/scripts/rollout.py), which calls
  [`lite/infer/rollout.py`](/lite/infer/rollout.py);
- env-server: [`scripts/serve_env.py`](/scripts/serve_env.py), backed by
  [`lite/gym/remote/server.py`](/lite/gym/remote/server.py);
- model serving: [`scripts/serve_sglang.py`](/scripts/serve_sglang.py);
- training: [`scripts/train/run_*.sh`](/scripts/train).

Python callers usually touch the public facades:

```python
from lite.agents.factory import make as make_agent
from lite.gym import make as make_env
```

Use package leaf modules when working on implementation details. For example,
message contracts live under [`lite/core/messages`](/lite/core/messages), while
env-server lifecycle code lives under [`lite/gym/remote`](/lite/gym/remote).

## Mental Model

The repo is organized around one boundary: an agent decides what to do, an env
executes it, and the result becomes a durable sample or rollout step.

```text
lite.core + lite.utils
  shared contracts and low-semantic infrastructure

lite.agents      lite.gym
model runtime    env runtime

lite.data        lite.infer        lite.train
datasets         rollout/eval      training/export
```

Rules of thumb:

- Put facts that cross layers in [`lite/core`](/lite/core).
- Put model/provider behavior in [`lite/agents`](/lite/agents).
- Put env behavior and env-server code in [`lite/gym`](/lite/gym).
- Put dataset storage, staging, and validation in [`lite/data`](/lite/data).
- Put rollout/eval entry logic in [`lite/infer`](/lite/infer).
- Put training algorithms, export, and train-time rollout code in
  [`lite/train`](/lite/train).
- Keep [`lite/utils`](/lite/utils) boring: path, config, logging, registry,
  image IO, parquet IO, timers, and generic serving retries.

## Package Map

- [`lite/core`](/lite/core)
  Owns cross-layer contracts: messages, metadata, samples, tool schemas,
  calls/results, and action vocabulary.
  Start with [`lite/core/__init__.py`](/lite/core/__init__.py),
  [`lite/core/tools`](/lite/core/tools), and
  [`lite/core/messages`](/lite/core/messages).

- [`lite/agents`](/lite/agents)
  Owns agent registries, sampling loops, adapters, protocols, and model-family
  runtimes.
  Start with [`lite/agents/factory.py`](/lite/agents/factory.py),
  [`lite/agents/core`](/lite/agents/core), and
  [`lite/agents/models/README.md`](/lite/agents/models/README.md).

- [`lite/gym`](/lite/gym)
  Owns the env API, concrete envs, sandbox execution, env-server, and
  lifecycle/reaping.
  Start with [`lite/gym/__init__.py`](/lite/gym/__init__.py),
  [`lite/gym/base.py`](/lite/gym/base.py), and
  [`lite/gym/remote`](/lite/gym/remote).

- [`lite/data`](/lite/data)
  Owns dataset rows, staging, HF round-trip, preprocessing, and row validation.
  Start with [`lite/data/staging.py`](/lite/data/staging.py),
  [`lite/data/utils/rows.py`](/lite/data/utils/rows.py), and
  [`lite/data/preproc`](/lite/data/preproc).

- [`lite/infer`](/lite/infer)
  Owns rollout/eval orchestration and debug validators.
  Start with [`lite/infer/rollout.py`](/lite/infer/rollout.py) and
  [`lite/infer/debug`](/lite/infer/debug).

- [`lite/train`](/lite/train)
  Owns SFT/RL rollout builders, export/tokenization, and train-time utilities.
  Start with [`lite/train/rollout`](/lite/train/rollout) and
  [`lite/train/export`](/lite/train/export).

- [`lite/utils`](/lite/utils)
  Owns low-semantic helpers used across packages.
  Start with [`lite/utils/path.py`](/lite/utils/path.py) and
  [`lite/utils/registry.py`](/lite/utils/registry.py).

## Core Contracts

`lite.core` exists so the agent side and the env side do not privately invent
the same protocol. These names are the main shared contracts:

- [`LiteBaseMetadata`](/lite/core/metadata.py), `LiteCUAMetadata`, and
  `LiteGenericMetadata`: tagged task metadata with routing `dims`, extra tool
  schemas, and env-specific `others`.
- [`LiteSample`](/lite/core/samples.py), `LiteRLStep`, `LiteRLSample`: durable
  sample and training row shapes.
- [`LiteToolSchema`](/lite/core/tools/schemas.py): how a tool is declared.
- [`LiteToolCall`](/lite/core/tools/calls.py): how a tool is invoked.
- [`LiteToolResult`](/lite/core/tools/results.py): how a tool returns feedback.
- [`lite.core.messages`](/lite/core/messages): message roles, content parts,
  image references, final messages, and turn grouping.
- [`lite.core.tools.action_space`](/lite/core/tools/action_space): canonical
  action names, action-batch tools, coordinate/key helpers, and action-set
  schemas.

If a value crosses the agent/env boundary or is stored in canonical data, prefer
putting the definition in `lite.core` and importing it from both sides.

## Agent and Env Boundary

The most important data flow is:

1. An env exposes [`LiteBaseMetadata`](/lite/core/metadata.py) and reset/step
   observations through [`lite.gym`](/lite/gym).
2. An agent uses that metadata to select prompts, tool schemas, and action
   spaces through [`lite.agents`](/lite/agents).
3. The model emits [`LiteToolCall`](/lite/core/tools/calls.py) objects.
4. The env executes them and returns [`LiteToolResult`](/lite/core/tools/results.py)
   objects.
5. The agent stores canonical messages/images in [`LiteSample`](/lite/core/samples.py)
   or training rollout structures.

Keep this boundary simple. A shared protocol fact should have one owner, not one
copy in `lite.agents` and another in `lite.gym`.

## Naming Rules

Use words that tell the reader which layer a name belongs to:

- `action`: a model-selectable verb such as `click`, `type`, `tap`, or `point`.
- `action-batch tool`: a tool that carries canonical actions, such as
  `computer` or `mobile`.
- `extra tool`: a tool call selected by `extra_tool_schemas`, such as
  `response`, `terminate`, or `bash`.
- `key vocabulary`: canonical Lite key tokens: lowercase named keys plus literal
  printable glyphs.
- `ingress`: classifying and validating an incoming env call.
- `Tools` subclass or tool set: what a concrete env exposes to the model.

Avoid reviving old broad words such as `surface`, `primitive`, or native GUI
for new code. They do not identify the layer clearly enough.

## Placement Rules

- `lite/core` must stay provider-free and env-free. It may define shared
  contracts; it should not import `lite.agents`, `lite.gym`, `lite.data`,
  `lite.train`, or `lite.infer`.
- `lite/agents/core` is for provider-independent agent runtime concepts:
  adapters, action-space dialects, agent loops, protocols, hooks, logging, and
  shared agent utilities.
- `lite/agents/models/<family>` is for model-family-specific behavior. See
  [`lite/agents/models/README.md`](/lite/agents/models/README.md).
- `lite/gym/remote` owns env-server HTTP/wire/lifecycle policy. Do not hide that
  policy in top-level scripts or generic utilities.
- `lite/gym/utils` is for env-owned helper families. It is not a second
  protocol owner.
- `lite/data/preproc/<source>` may contain source-specific parsing and cleanup.
  Do not promote a helper just because two raw datasets happen to encode the
  same source quirk.
- `lite/train` owns training-time rollout, export, tokenization, and image-slot
  mechanics.
- `lite/utils` is only for low-semantic cross-domain helpers. If a helper knows
  message shape, tool shape, final-turn policy, action vocabulary, row policy, or
  env lifecycle semantics, it belongs to the owning package instead.

## Adding Code

Before adding a new module or helper:

1. Decide which concept owns it.
2. Check whether a shared contract already exists in `lite.core`.
3. Prefer extending the owner over adding a parallel copy.
4. Keep public entry files readable: put primary abstractions at the package
   surface and move narrow helpers into named utility leaves only when that
   improves the first read.
5. Add focused tests when changing shared contracts, prompt/tool surfaces,
   env-server behavior, durable data rows, or training semantics.

## Documentation Style

Keep this README stable and user-facing:

- no dated status notes;
- no numeric counts unless they are essential and easy to reproduce;
- no temporary checklists;
- no long target trees for files that do not exist;
- no historical incident logs.

If a detail is only useful while refactoring, put it in a working plan or a test,
not in this package map.
