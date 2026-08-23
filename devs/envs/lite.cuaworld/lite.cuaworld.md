# lite.cuaworld

Dev-facing. For *using* a built env (run a rollout) see [README.md](/lite/gym/envs/lite/cuaworld/README.md).
**Keep this file current** when you change the lite.cuaworld engine or onboarding flow.

`lite.cuaworld.*` re-hosts gym-anything (CMU CUAWorld, MIT) software environments onto
cua-lite. The guiding principle: **the upstream assets hardcode the unix user `ga`
everywhere (not just shell — `su - ga`, `/home/ga` — but task prompts and verifiers),
so we match the CONTAINER to the assets rather than the reverse.** The base image is
the shared `sandbox` Dockerfile built with `--build-arg USER=ga`
(`cua-lite/lite.cuaworld.base`); fetched assets run without a ga->user transform.
The cached materials stay pristine, while uploaded hook bodies receive
deterministic CUA-Lite execution guards and narrow setup-only integration fixups.

## Two repos

- **cua-lite (this repo) = engine.** Builds images, registers tasks, runs
  setup/verify. No env content lives here (only `.cache/`, gitignored).
- **[cua-lite/lite.cuaworld-assets](https://huggingface.co/datasets/cua-lite/lite.cuaworld-assets) = content.**
  Public HF dataset. Per-env
  `env.json` + `scripts/` + `tasks/` + `data/` + `registered.json`
  [+ `post_build.sh`], maintained as a pinned fork for integration adaptations.
  Upstream task/verifier behavior changes require separate review and are not
  part of routine environment onboarding. `scripts/install.sh`
  fetches the needed env into `.cache/` on demand at the commit locked in
  `data/assets.lock.yaml`. Its
  [docs/WORKFLOW.md](https://huggingface.co/datasets/cua-lite/lite.cuaworld-assets/blob/main/docs/WORKFLOW.md)
  is the canonical import/edit/expand reference.

The HF dataset and everything owned by this repo use the `lite.cuaworld` /
`LITE_CUAWORLD` namespace. For local
or offline materials development, point
`LITE_CUAWORLD_MATERIALS_REPO=/path/to/lite.cuaworld-assets` at a checkout. Local
mode has two freshness scopes: staged image-context bytes participate in image
freshness, while the full selected subtree (including `tasks/` and
`registered.json`) owns materials-cache freshness. Local-materials mode refuses
pull/push.

## Image layers

```
ubuntu:22.04
 └ cua-lite/lite.cuaworld.base   lite/gym/sandbox/docker/Dockerfile.linux built with --build-arg USER=ga  (shared sandbox Dockerfile, desktop unix user `ga`)
    └ cua-lite/lite.cuaworld.<sw>  built by scripts/install.sh via docker/Dockerfile (FROM lite.cuaworld.base + the software's hooks, run by docker/run_hooks.sh)
```

**Why the `ga` desktop user (not `user`).** The upstream gym-anything assets
hardcode the unix user `ga` everywhere — not only shell (`su - ga -c`, `/home/ga`,
`chown ga:ga`) but task PROMPTS shown to the agent and VERIFIERS (e.g. "save the
final drawing to `/home/ga/Documents/LibreCAD/…`"). A static scan-and-replace
`ga`->`user` was never clean, so we match the CONTAINER to the assets: the base is
the shared `sandbox` Dockerfile built with `--build-arg USER=ga`
(`cua-lite/lite.cuaworld.base`), and the fetched assets stay `ga` at runtime.
lite.osworld/lite.demo keep the same Dockerfile's default `user`;
`gnupg`/`file` (which a few install hooks need) are installed by those hooks, so
nothing cuaworld-specific is baked into the base.

**Why the exec session runs as `root` (unlike lite.osworld).** The env sets
`EXEC_USER = "root"` (base `SandboxBaseEnv` default is `user`). The upstream hooks
launch GUI apps via `su - ga -c "…"` (GUI apps must not run as root) — and
`su - ga` is passwordless *only from root*; a `ga`→`su - ga` prompts for a
password and hangs the whole setup. So the exec-stdio session runs as root and the
hooks drop to `ga` themselves. The desktop/X (`:1`, no auth) is drivable by root,
so screenshots + xdotool input work unchanged.

Exec-stdio prepends `/opt/env/venv/bin` for environment-owned helpers, but
CUAWorld materials install their hook dependencies for the system interpreter.
Therefore build-time `pre_start` and runtime `post_start`, `pre_task`, and
`post_task` hooks all run with the normal desktop session PATH; a hook's bare
`python3` resolves to `/usr/bin/python3`. The cached hook files remain
unchanged, but the engine normalizes uploaded hook bodies for shell/session
guards; setup-only material fixups are excluded from export/post_task evidence
hooks.

## Code map (this dir)

| file | role |
|---|---|
| `main.py` | the onboarded-env list: one `register_cuaworld_software(...)` per software. The single place to see/toggle what's wired. Per-software `memory`/`cpu`/`gpus` overrides live here. |
| `src/software.py` | `register_cuaworld_software`: reads defaults from `configs/default.yaml` + `.cache/<sw>/<env>/registered.json`, builds `SandboxTaskConfig`s for **all splits**, registers tasks + legacy baked-shape metadata + env-server services. |
| `src/adapter.py` | runtime: run a hook from a **file** (`bash /tmp/cuaworld_*.sh`, so a hook's `pkill -f <app>` can't kill the setup shell) + host-side verifier bridge. |
| `src/vlm.py` | `gym_anything.vlm` shim so VLM-judged verifiers import + score. |
| `configs/default.yaml` | shared registration defaults (`env_var_prefix: LITE_CUAWORLD`, computer baseline, fallback task count, post-action delay, step timeout). Swap wholesale with `LITE_CUAWORLD_CONFIG=<name>`. |
| `data/assets.lock.yaml` | immutable HF materials repo/revision used by install and image freshness. |
| `scripts/install.sh` | verb-first `install.sh <verb> <sw>` or compatibility `install.sh <sw> [verb]`: build provisions locked materials + host verifier deps before local image work; pull checks the remote freshness label first, then provisions after adopting the image. Also `uninstall.sh`, `cleanup.sh`. |
| `docker/Dockerfile` | static per-software build recipe (FROM lite.cuaworld.base; `COPY` context; `RUN run_hooks.sh`). `docker/run_hooks.sh` runs the install-time `pre_start`; the env class runs `post_start` once after the desktop container boots. |
| `.cache/<sw>/<upstream_env>/` | fetched materials (gitignored). |

`src/software.py` / `src/adapter.py` / `src/vlm.py` are no-underscore runtime
modules; the public surface is `register_cuaworld_software` (called from
`main.py`). `scripts/` and `docker/` follow the spec's env layout
(see [/docs/envs.md](/docs/envs.md)) — lite.cuaworld is a Sandbox-family, browsergym-style
multi-variant env whose content lives in an external materials repo. Two shared
engine touchpoints: the opt-in `--gpus` passthrough in
[/lite/gym/sandbox/exec_stdio/client.py](/lite/gym/sandbox/exec_stdio/client.py)
(`DockerProvisioner`), used when a software sets `gpus>0`; and the
`EXEC_USER = "root"` override described above — cuaworld is the ONE env that
overrides the `SandboxBaseEnv` default, and both
[/lite/gym/sandbox/base.py](/lite/gym/sandbox/base.py) and `client.py::attach`
name it explicitly.

## Onboard a new software

Prereq: it must run under plain rootless Docker (no VM / privilege / dind).

1. **Scout & smoke.** Temporarily add a `register_cuaworld_software("<sw>","<env>")`
   line, `uv run --no-sync bash lite/gym/envs/lite/cuaworld/scripts/install.sh build <sw>`,
   and confirm the app launches (boot a
   container, run the task's `setup_task.sh` as root, screenshot). If it needs a
   VM/privilege/dind, stop — don't onboard.
2. **Import pristine → materials** (maintainer). Copy upstream
   `environments/<env>/` into the materials repo `<env>/` unmodified + add
   `SOURCE.json`, commit as the baseline. See materials `docs/WORKFLOW.md`.
3. **Adapt in materials.** Edit installation/startup scripts only where the app
   genuinely needs an integration adaptation; add `post_build.sh` if the app
   launches at boot (not per-task). Keep task and verifier behavior unchanged
   and record upstream failures instead of repairing them here. Write
   `registered.json` from the upstream split without filtering VLM-verifier
   tasks (derive it from upstream
   `splits/<env>_split.json`: `eval`=`test_tasks`, `train`=`train_tasks`,
   plus `additional_splits`). Push.
4. **Wire** in `main.py`: one `register_cuaworld_software("<sw>", "<env>", memory=…)`
   line (resource knobs live here).
5. **Build & test.**
   `uv run --no-sync bash lite/gym/envs/lite/cuaworld/scripts/install.sh build <sw>`;
   use `install.sh rebuild <sw>` after changing local materials, and
   `install.sh provision <sw>` when you only need materials + host verifier deps.
   run a 5-task rollout (`--head 5`) — see README.

`registered.json` carries **all** tasks/splits; we only *sample* a few per test
run via `--head`. Don't trim it to 5.

## Where to fix things (decision table)

| symptom | root cause | fix WHERE |
|---|---|---|
| app doesn't launch at runtime; per-task `setup_task.sh` only *waits* for it | env `post_start` failed after container boot | read the HOST log (env-server / rollout stderr) for `lite.cuaworld post_start hook failed (rc=…)` — the hook's merged stdout+stderr is captured by `run_command` and its tail is logged there, so there is no in-container log file to open. Adapt the startup hook in materials only when the container substrate requires it. Use `post_build.sh` only for a service that genuinely belongs in the image's boot supervisor. |
| package "installed" but missing | install hook needs a tool/lib/repo-key the base lacks | edit `install_<sw>.sh` in **materials** (gnupg/file already in sandbox; add e.g. GTK libs there) |
| `could not connect to display :1` | hook points `XAUTHORITY` at a missing file | edit the hook in **materials** to drop/unset it (cb's Xvnc is `-SecurityTypes None`) |
| hook dies at `pkill -f <app>` | inline exec put script text in argv | already handled — `adapter` runs hooks from a file. Don't inline. |
| VLM-judged verifier reports a judge error | `query_vlm` call failed (endpoint/model/creds/image read) | `src/vlm.py` raises `VLMProviderError`, which bypasses verifier-local `except Exception` blocks; the adapter also checks a provider-failure side channel after verifier return, so bare `except:` blocks cannot turn judge outages into reward 0. The judge default is the env config's `env_kwargs.judge.model` (currently `gpt-5.5`) and litellm resolves provider env from the evaluator process, so export `OPENAI_API_KEY` **in the env-server process** and set `OPENAI_BASE_URL` there only for a custom endpoint. For another provider set `VLM_MODEL` to a litellm-known or prefixed id (`openai/…`, `anthropic/…`, `gemini/…`) plus that provider's own standard key var; for your own server set both `VLM_MODEL` and `VLM_BASE_URL`. There is no backend switch. Fix the referenced image path when applicable. |
| verifier is cut off before its configured VLM retries finish | total `step_timeout` is below the 180s preparation budget plus the derived verifier budget | raise `make_kwargs.step_timeout` when increasing `VLM_TIMEOUT`, `VLM_MAX_RETRIES`, or `LITE_CUAWORLD_VERIFIER_TIMEOUT`. |
| a task silently missing from registry | malformed/absent `task.json` | `_build_configs` skips it (logs a warning); fix the `task.json` in **materials** |
| wrong mem/cpu/gpu/timeout | resource knob | the `register_cuaworld_software(...)` line in `main.py` |

## Migration defects (fixed)

Defects **we** introduced porting gym-anything onto cua-lite's native substrate — all now
fixed (full root-cause history is in git). The recurring shape: the port enumerated by hand
what upstream ships and silently dropped the rest, then `sandbox/base.py` relabelled the
permanent breakage *transient* → it surfaced as a meaningless "reset timeout".

- **`config/` + `assets/` never staged** → kstars baked a broken image (0/58), ugene tasks had
  no input files. `scripts/install.sh` now stages both into the build context.
- **docker-compose worldview vs native run** (openemr / wordpress / odoo / moodle): upstream
  reaches its DB via `docker exec <container> mysql|psql`; the native port has no docker. Fixed
  by rewriting those calls to native in the materials + deleting the `/usr/local/bin/docker`
  shims; odoo sets `pg_hba.conf` to `trust`. moodle keeps tasks byte-identical (upstream ships a
  native branch) and only forces `setup_moodle.sh` down the native path.
- **setsid re-homed to the engine** (`src/adapter.py::_run_hook` runs hooks under `setsid -w`) so a
  backgrounded GUI survives the hook returning, and mirrored `tasks/` stay byte-identical.
- **VLM single-frame cap** removed — the adapter builds real per-step `frames`/`steps`.
- **diagnostics** — `run_cuaworld_setup` logs stdout *and* stderr (the real failure was on stdout).

Historical validation rebuilt the full image set and ran a no-LLM sweep over all 2419 train
tasks, finding **zero** residual setup bugs at that revision. Current reachability is tracked in
[/takeover.md](/takeover.md): `eclipse` is presently blocked upstream, and `knime` has 0 live
tasks after excludes. Remaining task-content defects are recorded in
[/devs/envs/lite.cuaworld/UPSTREAM_ISSUES.md](/devs/envs/lite.cuaworld/UPSTREAM_ISSUES.md).

## Known upstream task limitations

Some upstream task definitions or verifiers are infeasible, incomplete, or
gameable. This integration does not change their setup, success criteria, or
fallback scoring, and it does not remove a task merely because its verifier
uses a VLM. Record upstream issues in materials metadata/maintainer notes.
Any deliberate split curation belongs in a separate, reviewed materials change,
not this integration PR. Runtime bridge defects, such as passing the wrong
trajectory or breaking the upstream VLM helper contract, remain CUA-Lite bugs
and must be fixed in this repository.

Known examples in the pinned upstream source include Sweet Home 3D verifiers
that call `copy_from_env` with one argument and expect a returned path, although
the upstream environment API documents `(container_src, host_dst)`. The bridge in
`src/adapter.py` therefore serves both upstream call signatures: `host_dst` is optional (a missing one
becomes a `tempfile.mkstemp` path), and the host path is returned
**unconditionally**. That is a deliberate CUA-Lite contract, not a mirror of the
upstream signature: 67 sweet_home_3d call sites open the returned path (the
two-argument-only shim raised `TypeError` and killed the verifier before scoring),
and 2 further sites — `sumo/evaluate_phased_evacuation` and
`astroimagej/extract_linear_shock_profile` — truthiness-test the return, so a
`None` would have guaranteed reward 0 on registered, non-excluded tasks. Every
other caller discards the return, to which the change is invisible. By contrast,
`sample_trajectory_frames(..., n=...)` is part of the pinned verifier contract
through both `vlm_utils` and `gym_anything.vlm`; CUA-Lite supports that keyword
on both compatibility import paths.
Pinned upstream adds both trajectory endpoints before applying the remaining
sample budget, so `n=1` on a multi-frame trajectory returns two frames. The
integration preserves that surprising behavior rather than changing verifier
inputs locally.

The pinned gvSIG split also includes `intersect_rivers_countries` with the
undeclared reward type `weighted`. Upstream accepts the task but its reward
mapping falls back to the current step reward, which is zero because the task
defines no reward shaping. CUA-Lite keeps the task registered and preserves that
zero-reward behavior instead of silently dropping it or inventing weighted
semantics.

If a mirrored hook or verifier references `user` or `/home/user`, treat that as
a materials provenance defect from an older `ga` to `user` rewrite. Correct it
only in a separately reviewed materials change that restores the pristine
upstream path; do not add a runtime rewrite in this repository or fold the
change into routine onboarding.

Rule of thumb: **installation/startup adaptations belong in the maintained
materials fork; upstream task/verifier defects are recorded rather than
repaired; per-software knobs live in `main.py`; run mechanics
(exec/launch/verify bridge) live in `adapter`/`software`; the desktop base is
`cua-lite/lite.cuaworld.base` (the shared `sandbox` Dockerfile built with
`USER=ga`).** Resist re-introducing a global `ga` to `user` transform.

## Testing rules (run for EVERY env before uncommenting its `main.py` line)

Hard-won from onboarding: a build that "succeeds" and a `reset()` that "returns
OK" both routinely lie. An env graduates only after **T0–T2 pass, T4 returns a
number, and T5 runs**. Record a row per env (template at the end).

- **T0 — build & install actually landed.** `install.sh` rc=0 is necessary, not
  sufficient. gym-anything install hooks swallow failures (`|| true`, tolerated
  RUN tails), so a download/key/repo failure yields a "successful" build with the
  app **missing**. Verify the binary/dir exists:
  `docker run --rm --entrypoint bash cua-lite/lite.cuaworld.<sw> -c 'which <bin> || ls /opt/<app>'`,
  and grep the build log for `E:|not found|command not found|Unable to locate|ERROR`.
- **T1 — boot.** Container boots, `/tmp/gnome-ready` within ~60s, and
  exec-as-root drives X: `docker exec -u root <c> bash -c 'DISPLAY=:1 xdotool getdisplaygeometry'`.
- **T2 — app launches (decisive, always eyeball a screenshot).** After `reset()`
  plus a wait **sized to the app** (light GUI ~10s; JVM/heavy IDE/sim ~45–60s):
  the app's OWN window is in `DISPLAY=:1 wmctrl -l`, its process is running, AND a
  **screenshot shows real app content**. A blank desktop is a FAIL even if
  `reset()` returned OK and the PNG is non-trivial in size — *open the image and
  look*. Common root causes + fixes are in the decision table above (boot-launch,
  missing lib, XAUTHORITY). First-run dialogs (welcome/updater) are acceptable
  (the agent dismisses them) but note them.
- **T3 — setup integrity.** The task's `setup_task.sh` ran to completion (seeded
  files present; no mid-setup `set -e` abort that skipped the launch) and the app
  opened the task's input.
- **T4 — verify bridge.** On a freshly-reset, un-acted env, `run_cuaworld_verify`
  returns a **numeric** reward (usually low/0) with no exception — proves the
  export hook + `verifier.py` + `env_info` bridge. VLM-judged tasks score via the
  `gym_anything.vlm` shim (`src/vlm.py`): `adapter` injects it into `sys.modules` and
  records the upstream-shaped trajectory, then feeds all step/final screenshots
  into `traj`, so the verifier loads + scores instead of crashing on the missing
  upstream package.
- **T5 — rollout.** A `gpt-5.5` rollout of ≥1 `eval` task completes (agent steps
  → `terminate` → score). The bar is "the episode runs and the verifier scores",
  NOT a high return (that's model capability). For sign-off sample 5 (`--head 5`).

> **Caveat — a passing rollout does NOT prove the GUI is agent-usable.** Because
> verifiers score only the **output file** (upstream principle: reward on result,
> not path), the agent is free to ignore the app and script the task in a
> **gnome-terminal** (e.g. `python3 - <<'PY' … astropy/csv …`). Several envs
> (kstars, libreoffice_calc, astroimagej, ardour, gvsig …) were solved this way —
> the app sat open, untouched. So do NOT read "rollout passed / files produced" as
> "the agent operated the software". Counting all `type`/`click` as app-operation
> is wrong — they may target the terminal. To judge the **environment** (which is
> our concern), verify the *software itself* is usable: launch the env and drive
> the app directly (menus/render/load the task file). To judge **GUI-operability**
> instead, that's a task/reward change (remove the desktop terminal, or require
> GUI-state evidence), out of scope for the env.

Procedure rules:
- ALWAYS open a screenshot at T2; non-blank byte count is not proof.
- Size the wait to the app — never conclude "blank" from an 8s shot on a JVM app.
- Test ≥2 tasks when launch logic differs per task.
- Heavy task setup: preserve or correct that task's upstream
  `hooks.pre_task_timeout` in the materials source; do not add a CUA-Lite-wide
  timeout floor.
- Clean up your OWN containers afterward (`docker rm -f` your
  `lite-env-local-lite.cuaworld.*`);
  never touch co-tenant containers.

Record (one row/env): `env | T0 install ✓/✗ | T2 screenshot verdict | T4 reward | T5 avg(n)`.

## Status

**40 software environments active in `main.py`** (T2-verified launching on
`cua-lite/lite.cuaworld.base`; per-env install/adaptation notes live in each `<env>/AGENTS.md` in
the materials repo).

- **Original 18**: pymol, gmat, openvsp, pycharm, vscode, dbeaver, eclipse,
  coppeliasim, sumo, imagej, freecad, qgis, webots, hec_ras, knime, slicer3d,
  blender3d, moodle.
- **+19 A-tier (direct-GUI, native installs)**: libreoffice_calc; vlc_media_player,
  ardour; gretl, ugene, jstock; librecad, solvespace, sweet_home_3d, openrocket;
  gvsig_desktop; diagrams_net; astroimagej, kstars_sim; geogebra, gcompris;
  gpredict, qblade, openlca.
- **+3 C-tier web (de-dockerized to native LAMP/Postgres + supervisor)**: wordpress
  (native MariaDB), openemr (native LAMP + headless `InstallerAuto`, `exit;`-guard
  removed), odoo (native PostgreSQL + Odoo 17 `.deb`). See each env's materials
  AGENTS.md for the de-dind recipe.

**Dropped (documented in materials AGENTS.md, unregistered):**
- **rstudio / jasp / jamovi** — browser-engine apps whose Chromium/bwrap sandbox
  needs `seccomp=unconfined` (a privilege cuaworld avoids); `--no-sandbox` doesn't help.
- **opentoonz** (no rootless install: snap/flatpak/deb-multimedia all fail),
  **gimp** (0 eval tasks upstream), **draw_desktop** (duplicate of diagrams_net).

**Pending (diagnosed, task-scoped — not env defects):**
- **libreoffice_writer / impress** — per-task `setup_task.sh` runs a headless
  soffice convert *then* a GUI launch; the two instances collide on the profile
  lock → "Fatal Error". Office is covered by calc.
- **zotero** — Firefox-engine; `libdbus-glib` + software-GL added, real-flow launch
  still flaky.

**Known env-data caveat (task scope):** Slicer3D has several task-scoped
data/export/verifier caveats; see
[/devs/envs/lite.cuaworld/UPSTREAM_ISSUES.md](/devs/envs/lite.cuaworld/UPSTREAM_ISSUES.md).
