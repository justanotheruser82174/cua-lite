# WebGym GPT Collection — Pipeline & Plan

Collect clean `gpt-5.5` demonstrations on the webgym **train** split to distill web-agent skills
into a Qwen student (**target 8B/32B**, so multi-step *research* is learnable — not only a 2B).
The dataset must cover **all tiers' skills**: easy/medium tasks **don't abuse goto** (read the page
you're on), and complex tasks that genuinely need it **do search/navigate** — the
"minimal-necessary goto" principle.

This doc is the converged, scale-ready pipeline. Exploration rollouts that produced it live in
`.logs/rollout/webgym_explore/` (transient); the real collection writes to `.data/` (curated).

## Pipeline (3 components, validated)

1. **Teacher prompt — balanced** ([/scripts/configs/gpt/recipes/collect/webgym.yaml](/scripts/configs/gpt/recipes/collect/webgym.yaml)):
   default to the open page; `goto`/search **only when the answer isn't reachable on the current
   site**; **search on DuckDuckGo** (Google/Bing return blank pages in the webgym browser) and
   **open+read a result** (don't answer from snippets); don't fabricate deep URLs. (Both extremes
   are bad — see §Evidence: a *strict* no-search prompt can't do research tasks, a *permissive* one
   flails search on easy tasks and tanks success.)
2. **Filter — the single filtering step** (`filter.py`):
   `--drop-failed --drop-loops --drop-serp-only --drop-captcha --drop-unsubmitted --drop-illposed-task`.
   **ALL filtering lives here** (`stage` no longer applies any default filter). `--drop-failed` is the
   success filter (`episode_return>=1.0`, the predicate that used to be `stage`'s hidden default).
   `serp_only` drops trajectories that **search but never click through to a real page**
   (snippet-scraping / SERP bouncing — degenerate for any model size) while **keeping productive
   research** (search → click a result → read, however many steps — the learnable complex-task skill).
   `--drop-captcha` drops bot-verification-wall trajectories; `--drop-unsubmitted` drops trajectories
   whose answer was never sent via a top-level non-empty `response` call; `--drop-illposed-task` drops
   degenerate underspecified instructions (≥3 unfilled "a specific …" slots — a data-gen bug; best
   applied pre-rollout). No-op stripping drops a turn but no picture, so the output row is
   **compacted**: `compact_row_images` ([devs/data/utils.py](/devs/data/utils.py)) drops the images
   nothing references any more and renumbers
   the survivors `0..N-1` (image list and every `{"type":"image","index": N}` part in one step),
   and when an old pre-`role:"tool"` leading observation is dropped, carried goal content preserves
   text, metadata, and earlier indexed reference image parts while dropping the stale observation image.
   For a **2B** target, swap to the stricter `--drop-search-flail` (also drops ≥2-engine /
   ≥3-search chains).
   *Search reality (validated):* from the cluster's datacenter egress IP the search **engines**
   (DuckDuckGo/Google/Bing) IP-block the search **request** (results error with "anomaly"/reCAPTCHA;
   the homepage still loads), so blocked search attempts become dead-ends. We still **keep productive
   search** and only drop `serp_only` — a blanket `--drop-search-goto` would kill the learnable skill
   and is NOT used for 8B/32B; the durable fix is infra (clean egress IP / proxy), not filtering. Most
   search/block/unsubmitted/ill-posed cases score 0 and are already removed by `--drop-failed`; the
   new flags are explicit guards + pre-rollout hygiene.
3. **Distribution — 25 / 60 / 15** (easy / medium / hard, site-start only), see §1.

## 1. Tier distribution — per-tier `--sample` 500 / 1000 / 2000 (easy / medium / hard)

The collection knob is a **flat per-tier `--sample`** (§3): easy d1–3 → 500 each, medium d4–6 →
1000 each, hard d7 → 2000. This weights attempts toward mid/high **because that's where
distillation actually adds value** — a base Qwen3-VL-8B already does easy well but cracks on
medium/hard (natural train split is 79 / 19 / 2 easy/medium/hard, so the raw pool is the opposite
shape). The realized clean mix (§3) is ~33 / 52 / 15 over the difficulty tiers — easy ran higher
and hard lower than an even target, because easy yield is ~2× hard's and the d7 pool capped at
~1,392 (below `--sample 2000`):

- **easy — low headroom, keep only a floor.** Base 8B on eval-easy ≈ 62% raw / ~81% reachable
  — comparable to the gpt teacher. easy = "read the visible page", a skill the base already has, so
  SFT on more easy mostly wastes budget. We keep a floor as a **grounding/format anchor** (eval is
  71% easy, so the student must stay fluent there) — not zero. (`--sample 500`/tier; realized 771
  clean, higher than intended because easy yield is highest.)
- **medium — highest value × largest pool → the bulk.** Paired eval on identical tasks
  (same seed/budget): **gpt 83% vs base 8B 33%** (preliminary) — a big gap. The base's navigation is
  broken (re-goto's its own homepage, loops, jumps to blank google), exactly what the teacher fixes.
  Medium's task pool is huge (51k site-start), so this is the **safe place to scale**.
- **hard — high headroom, sample the full d7 site-start pool.** Highest per-task gain. We set
  `--sample 2000` to take the whole d7 site-start pool, but it **caps at ~1,392** (the pool is
  smaller than 2000); d7 site-start clean-yield is **measured 25%** (this batch) → **350 clean** at
  n=1. NOTE: this is `difficulty==7` ONLY — d8+ not yet collected; to add the harder-than-7 skills,
  run a separate `difficulty>=8` pass (see "Within hard" below and §Open).
  - **Future (when a `difficulty>=8` pass exists): OVERSAMPLE d8+ before staging.** This batch is
    d7-only, so nothing to oversample yet. Once d8+ is collected: the >=7 pool decays hard with
    difficulty (~59/20/10/5/6% for d7/d8/d9/d10/d11+) and success-rate also falls (d7 ~40% → d10
    ~15%), so a random pick is ~⅔ d7. To keep the harder skills represented, choose the cleaned
    rows/roots explicitly: **take all available d8+ clean demos first, then fill with d7** before
    invoking `stage` (best-effort — d10+ is genuinely scarce, "cover what exists" not "force a
    quota"). Difficulty histogram of any tier:
    `uv run python devs/data/webgym/quality_check.py` (via metadata), or group clean
    `episode_return>=1.0` trajectories by `metadata.others.difficulty`.

**Collect site-start only, all tiers** (exclude search-engine start pages). Search-start tasks open
on google.com/bing.com whose **results page is blank in the headless browser** (§2) → ~0 clean demos
(the hardest browsecomp riddles, unsolvable here). There are **no ddg-start tasks** in the dataset,
so "exclude {google,bing}-start" = "site-start only". Composition (train): medium **92% site-start**
/ 8% google·bing-blank; hard **45% site-start (3,007)** / 55% google·bing-blank. Kept site-start
demos still search via **ddg** when a task needs it (ddg works; §2).

| band | difficulties (site-start only) | `--sample`/tier | realized clean | share of 3,251 |
|------|------|------:|---------------:|------:|
| easy    | d1–3 | 500  | 771   | 24% |
| medium  | d4–6 | 1000 | 1,202 | 37% |
| hard    | d7   | 2000 | 350   | 11% |
| popular | curated pool | full (~2,102) | 928 | 29% |
| **total** | | | **3,251** | |

(Realized this batch `a2cad60b`, per §3. Over the difficulty tiers only that's ~33 / 52 / 15
easy/medium/hard — hard came in under an even target because the d7 pool capped at ~1,392 and its
yield is the lowest (25%). Scale to 5,000 is additive — §8.)

## 2. Search & navigation policy

The dataset must teach **both** "stay on the page" (easy/medium) **and** "search/navigate when a
task truly needs it" (complex). The split between a *good* and a *bad* search demo is **not** how
many searches — it's whether the agent **clicks through to a real page** or just reads SERP snippets:

- **KEEP (productive research):** `search → click a result → read the page` (→ optionally refine &
  repeat). This is the complex-task skill; an 8B/32B learns it. Measured: 18/18 medium and 23/24
  hard search-trajectories click through.
- **DROP (`serp_only`):** searches but **never clicks through** — answers from snippets, or bounces
  the same query across engines (e.g. `126677`: "oriolesband.com" on Google→Bing→DuckDuckGo, 0
  clicks, then guesses — while ignoring the open page that had the answer). Degenerate for any size.
- **DROP (`loops`):** ≥3 consecutive identical actions (stalls).

Why one search engine is enough: **Google/Bing return a blank results page in the webgym browser**
(confirmed by screenshot) — so the agent was *forced* to bounce to DuckDuckGo. The prompt now tells
it to search on DuckDuckGo directly, which removes the bounce — validated: medium = **20/20
search demos single-engine ddg, 0 bounce** (the prior prompt bounced google→bing→ddg). `serp_only` still catches any
residual snippet-scraping.

The older blanket drops **F1** (any search goto) / **F2** (cross-domain goto) / **F6** (search-start
task) are **OFF** — they kill legitimate research. They remain in `filter.py` for a
single-site-grounded dataset. `--drop-search-flail` is the stricter 2B-target variant of the
search filter.

## 3. Attempt sizing — flat per-tier `--sample`

This batch used a **flat per-tier sample**, NOT a band-budget split: each **easy** tier (d1–3)
`--sample 500`, each **medium** tier (d4–6) `--sample 1000`, **hard** (d7) `--sample 2000`. The
curated popular pool runs the full ~2,102 tasks (no `--sample`). **Clean-yield** = success demos
surviving the §Pipeline filter (`--drop-failed --drop-loops --drop-serp-only` +
captcha/unsubmitted/illposed). Measured this batch (`a2cad60b`, seed 1, n=1):

| tier | `--sample` | ran | clean | yield |
|------|-----------:|----:|------:|------:|
| d1 | 500  | 500  | 316 | 63% |
| d2 | 500  | 499  | 255 | 51% |
| d3 | 500  | 477  | 200 | 42% |
| d4 | 1000 | 975  | 470 | 48% |
| d5 | 1000 | 988  | 356 | 36% |
| d6 | 1000 | 997  | 376 | 38% |
| d7 | 2000 | 1,392 | 350 | 25% |
| popular | full (~2,102) | 2,065 | 928 | 45% |
| **total** | | **7,893** | **3,251** | **41%** |

Notes:
- **Yield falls steadily with difficulty** (d1 63% → d7 25%), as expected — harder tasks fail or
  loop more, so the same `--sample` buys fewer clean demos up the ladder.
- **d7 `--sample 2000` only ran ~1,392** — that's the available d7 site-start pool (the pool cap
  binds before 2000), giving **350 clean** at n=1. d8+ remain uncollected (`difficulty==7` filter
  only); adding them needs a separate `difficulty>=8` pass (§Open).
- **Concurrency — keep GLOBAL ≈16** (throughput scales sub-linearly past ~16, §7; ≈16 also stays
  well within the browser pool — `WEBGYM_INSTANCES`, §4). Collect **one tier at a time** at
  `--concurrency 16` — never several tiers in parallel. Run the popular pool the same way (its own
  single phase at `--concurrency 16`), so the global concurrency is always ~16, one phase at a time.
- Re-confirm per-tier yields on the next scaled batch.

## 4. Execution

Real collection writes to `.data/` (curated). Loop `d` over each difficulty with its `--sample N_d`.

**Run through the env server (default).** Build the webgym image once (`install.sh build`), then
launch [`scripts/serve_env.py`](/scripts/serve_env.py) once and point the collector at it via
`CUA_LITE_ENV_SERVER_URL`. The env-server starts the shared OmniBoxes backend container on first
use and owns instance lease/release with a drift-reaper + TTL + session scoping, so a crashed or
stalled stream can't leak browser workers — steadier than the in-process **direct** path (which is
the fallback when `CUA_LITE_ENV_SERVER_URL` is unset).

```bash
# --- host ---

# step 0: build the webgym image once, then start a durable env-server.
#   - JUDGE CREDS (load-bearing): the VLM reward judge needs OpenAI creds IN THE
#     SERVER's environment. Without them webgym reset() fails 500
#     (EnvDepsMissingError) — and since --drop-failed below keys on
#     `episode_return` (which IS that judge's reward), no creds ⇒ no clean data.
#     OPENAI_BASE_URL is optional; set it only for a custom endpoint.
#   - WEBGYM_INSTANCES sizes the OmniBoxes browser pool: set >=64 for production
#     collection; omit to auto-size from host RAM/CPU (capped at 128). The server
#     starts the shared backend container on first use and sets WEBGYM_MASTER_URL
#     itself. Launch detached (setsid/nohup) so it outlives the shell. Any free
#     port works (30100 here); point CUA_LITE_ENV_SERVER_URL at it.
uv run --no-sync bash lite/gym/envs/webgym/scripts/install.sh build
export OPENAI_API_KEY=...                  # + export OPENAI_BASE_URL=... for a custom endpoint
# webgym's server imports install-added host judge deps, so serve_env is one of
# the explicit `--no-sync` exceptions.
setsid bash -c 'WEBGYM_INSTANCES=64 uv run --no-sync python scripts/serve_env.py \
  --port 30100 --env-ids webgym >> serve_env.out 2>&1' &
HOST_IP=$(hostname -I | awk '{print $1}')
export CUA_LITE_ENV_SERVER_URL=http://${HOST_IP}:30100

# VERSION this rollout batch by a cua-lite commit id, PINNED ONCE here at batch start — the
# commit whose recipe (collect.yaml + filter.py + sft config) this batch uses. Every log-root
# lives under .../gpt/$COMMIT/, so one batch = one dir. Set it as a FIXED literal and reuse it
# for the whole batch (incl. resumes) — do NOT re-derive from HEAD per command: a multi-day
# collection will see unrelated intermediate commits, and the batch must keep its original id.
# This same id is the HF tag at upload (`upload --tag $COMMIT`), so local dir == HF revision ==
# producing commit (`download --revision $COMMIT` pulls exactly this version).
COMMIT=a2cad60b   # ← pin to your batch's recipe commit (e.g. `git rev-parse --short HEAD` once, then freeze)

# step 1: collect (per difficulty d). Balanced prompt handles on-page vs search-when-needed.
# --filter is SITE-START ONLY for every tier (§1): difficulty==d AND the start site is not a
# search-engine root (their SERP is blank → ~0 yield). Only the *roots* are excluded — real
# product sites like accounts.google.com / translate.google.com stay in.
# Per-tier --sample (flat, §3): d1,d2,d3 → 500 each; d4,d5,d6 → 1000 each; d7 → 2000.
# Collect ONE tier at a time (loop d over the difficulties) at --concurrency 16 — never several
# tiers in parallel (that would push global concurrency past ~16, §3/§7).
uv run python scripts/rollout.py --model-id gpt-5.5 --env-id webgym \
  --splits train --sample <500|1000|2000 by tier> --seed 1 --concurrency 16 --max-attempts 2 \
  --filter "lambda m: m.others.get('difficulty',0)==<d> and m.others.get('website','').split('//')[-1].split('/')[0].removeprefix('www.') not in ('google.com','bing.com','duckduckgo.com')" \
  --config-path scripts/configs/gpt/recipes/collect/webgym.yaml \
  --log-root .data/rollout/webgym/gpt/$COMMIT/d<d>

# step 1.5: PRIORITY — the curated "popular" pool (high-value, do this first / weight it heavily).
# `webgym_popular_2102.parquet` is OpenWebRL's filtered+cleaned popular subset (2102 train tasks).
# It is already pre-filtered, so it is driven by --prompt-data (NOT --splits/--filter, which are
# mutually exclusive with it — the parquet IS the task list). Because these tasks matter more, give
# them a LARGER attempt budget: --max-attempts 5 (vs 2 for the bulk difficulty tiers) so a transient
# judge/nav hiccup gets retried instead of lost. Run the FULL pool (no --sample) — all 2102 tasks.
# Use --group-size 1 (same as the bulk tiers): one rollout per task — coverage over diversity, and
# consistent with the rest of the corpus. Resume re-runs only the unfinished tasks.
uv run python scripts/rollout.py --model-id gpt-5.5 --env-id webgym \
  --prompt-data lite/gym/envs/webgym/data/webgym_popular_2102.parquet \
  --seed 1 --concurrency 16 --max-attempts 5 --group-size 1 \
  --config-path scripts/configs/gpt/recipes/collect/webgym.yaml \
  --log-root .data/rollout/webgym/gpt/$COMMIT/popular
# then filter it exactly like a tier (step 2 with --log-root .../popular --out .../popular_clean).

# step 2: filter — success filter (--drop-failed) + strip no-ops + drop stalls + drop snippet-scrape
#   (keep productive research) + drop captcha-wall / unsubmitted / ill-posed-task trajectories.
#   This is the ONLY filtering step; stage (step 3) stages everything. No-op stripping leaves
#   surviving image indices untouched and preserves text, metadata, and earlier indexed reference
#   image parts when carrying a dropped leading goal forward.
uv run python devs/data/webgym/filter.py \
  --log-root .data/rollout/webgym/gpt/$COMMIT/d<d> --out .data/rollout/webgym/gpt/$COMMIT/d<d>_clean \
  --drop-failed --drop-loops --drop-serp-only --drop-captcha --drop-unsubmitted --drop-illposed-task
```

**Flat per-tier `--sample`** (§3): 500 each for d1–3, 1000 each for d4–6, 2000 for d7 — NOT a band
budget split. Medium is 92% site-start and hard 45% (3,007 tasks) — the SITE-START filter just drops
the google·bing-blank remainder (and d7 site-start runs out at ~1,392 < 2000).

**On failure (crash / stall / host-overload kill): just re-launch the SAME command — do NOT wipe
`--log-root`.** A fixed `--log-root` makes `api.py` *resume*: it re-runs only the missing samples and
skips already-succeeded tasks (`lite/infer/rollout.py` `get_pending`; see `--max-attempts` help).
Deleting the log-root throws away good trajectories and re-rolls the whole tier — only do that for a
genuinely corrupt run. A stall looks like: the tier log stops growing (`mtime` frozen) for minutes
with the process parked in `do_poll`, usually the aftermath of a step-timeout wave when the host is
oversubscribed (other users' jobs spiking load) — kill ONLY your own driver process group
(`kill -KILL -<pgid>`, never others' procs and never the shared OmniBoxes backend container) and
relaunch; resume continues from where it left off.

## 5. Package as a publishable HF dataset

```bash
# --- host ---
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"
READBACK_ROOT="$PWD/.data/huggingface-readback"
COMMIT=a2cad60b   # same pinned batch id as §4 (the log-root version)

# step 3: stage all cleaned roots → canonical cua-lite/WebGym (NO filter — filtering was done in
#   step 2; stage has no hidden success predicate; content-addresses images; writes repo.json).
#   Record the final seen/kept/dropped_by_filter line and per-config row lines:
#   this is the row-content validation gate.
#   --log-roots takes many roots → ONE
#   dataset, so growing the corpus = append more cleaned roots here (§8).
uv run python -m lite.data.hf.stage \
  --log-roots .data/rollout/webgym/gpt/$COMMIT/d1_clean .data/rollout/webgym/gpt/$COMMIT/d2_clean \
              .data/rollout/webgym/gpt/$COMMIT/d3_clean .data/rollout/webgym/gpt/$COMMIT/d4_clean \
              .data/rollout/webgym/gpt/$COMMIT/d5_clean .data/rollout/webgym/gpt/$COMMIT/d6_clean \
              .data/rollout/webgym/gpt/$COMMIT/d7_clean .data/rollout/webgym/gpt/$COMMIT/popular_clean \
  --name WebGym \
  --repo-dir devs/data/webgym

# step 4: upload to a private smoke repo. Upload is transport only: it packages,
#   pushes, and tags the staged tree; it does not replace stage validation.
#   --tag defaults to the current HEAD short commit; pass --tag $COMMIT explicitly
#   so the HF revision matches the pinned batch id. Use the release org only for
#   the approved final publish.
: "${HF_ORG:?set HF_ORG to your Hub user/org for the private smoke repo}"
uv run python -m lite.data.hf.upload WebGym --org "$HF_ORG" --private --tag "$COMMIT"

# step 5: read back the pinned revision and run a small conversion smoke.
uv run python -m lite.data.hf.download WebGym \
  --org "$HF_ORG" \
  --revision "$COMMIT" \
  --out "${READBACK_ROOT}/cua-lite/WebGym"

uv run python -m lite.train.export.export_sft \
  --config scripts/configs/qwen3_5/default/webgym.yaml \
  --model-id Qwen/Qwen3.5-9B \
  --data-paths "${READBACK_ROOT}/cua-lite/WebGym" \
  --image-root "${READBACK_ROOT}" \
  --head 10 \
  --num-proc 1 \
  -o .data/sft/qwen3_5/webgym-smoke/train.parquet
```

Then distill + eval per the **Consumer** flow in
[/docs/examples/rollout_to_hf.md](/docs/examples/rollout_to_hf.md#consumer--train-from-the-hub)
(`download → export_sft → run_sft`, then base-vs-SFT on the held-out webgym eval split).

## 6. Bookkeeping

- **Dedup:** persist consumed `task_id`s across runs; exclude the **eval split** to prevent
  leakage. Collection uses `seed 1` (smoke/exploration used `seed 0`); still dedup post-hoc.
- **n_samples/task = 1** for all tiers including the curated `popular` pool (`--group-size 1` —
  coverage over diversity).
- **Paths:** curated collection under `.data/rollout/webgym/gpt/<commit>/` — one dir per rollout
  BATCH, named by a cua-lite commit pinned once at batch start (= the HF tag at upload; see §4).
  Transient exploration under `.logs/rollout/webgym_explore/`. Keep `<commit>/d<d>_clean` until staged.

## 7. Cost / time

This batch ran **~5,830 difficulty-tier attempts** (`--sample` 500/1000/2000, §3) **plus ~2,065 for
the popular pool** (the ~2,102-task pool × `--group-size 1`, §4 step 1.5) ⇒ **~7,900 total** → 3,251
clean. `gpt-5.5` latency-bound. Measured throughput at concurrency 56 ≈ **~1.2 traj/min** (~26
turns/min) ⇒ on the order of **days** single-stream (the popular pool adds ~35% to the tier run).
That 56 figure is the prior batch; the **recommended global 16** (§3) is slower in wall-clock but
trades throughput for pool stability (gains past ~16 are sharply diminishing anyway — see below).

**Throughput root-cause (2026-06-17 investigation; full reproducible analysis + commands in
[/devs/envs/webgym.md](/devs/envs/webgym.md)):**
- Per-turn nameable cost ≈ **15–20 s**: LLM ~8–12 s (dominated by **input/context** processing
  of the chained multi-screenshot history — NOT output; ~94 out-tok/turn measured
  when `reasoning_effort` was `low`) + env ~2–5 s. Plus a **per-trajectory VLM reward judge ~30 s**. × 20+ turns
  on hard tiers ⇒ trajectories are inherently minutes long.
- **Throughput scales strongly sub-linearly** with concurrency (clean isolated-pool sweep:
  conc-16 per-slot 3.2× worse than conc-1; production conc-56 ≈ 126 s/turn/slot vs ~15–20 s of
  nameable work). The rollout process is **CPU-idle** (not loop/GIL-bound); ruled out by direct
  measurement: provider concurrency, every single env op, litellm, memory/GC, wedged instances. The
  residual is consistent with combined-load I/O contention but was not definitively pinned
  (py-spy blocked by ptrace).
- **Levers (in order):** reduce turns (`max_steps`) on failing tiers · cheaper/shared reward
  judge · **horizontal scale across hosts/independent pools** (NOT more processes on one host —
  multiprocessing won't help since it's not CPU-bound) · keep the pool stable (it crashes under
  load, which also drags throughput). Output length was **not** a lever at the
  `reasoning_effort: low` these numbers were taken at; the collect recipe now pins
  `medium` (`/scripts/configs/gpt/recipes/collect/webgym.yaml`), so re-measure
  out-tok/turn before reusing this cost model.
- Never exceed concurrency 64 (pool size); adding concurrency past ~16 gives sharply diminishing
  returns.

## 8. Scale to 5k — additive

The run writes per-tier `.data/rollout/webgym/gpt/<commit>/d<d>` dirs; the published dataset is
`stage --log-roots <all <commit>/d<d>_clean dirs>` (many roots → one dataset). To grow the corpus
WITHIN a batch, collect more tasks into the same `<commit>/` dirs and re-stage; a new recipe ⇒ pin a
new `<commit>/` at the next batch start + a new HF tag. **Dedup is collection-side, not staging-side**:
`stage` does NOT dedup — it keeps every trajectory row (so e.g. the tier↔popular overlap lands as
multiple rows; `hash_split(task_id)` only guarantees all rows of
one task share a split, no train/eval leakage). To avoid re-collecting a task, persist consumed
`task_id`s and exclude them at rollout time (§6). Extend toward 5,000 (e.g. easy 1,250 / medium 3,000 /
hard 750). **Hard at 750 is the hard part**: at the measured ~25% d7 yield that's ~3,000 attempts,
but the d7-only site-start pool is just ~1,392 (the full `difficulty>=7` pool is ~3,007 incl. d8+) →
needs a `difficulty>=8` pass and/or `n_samples=2` (duplicate tasks, lower diversity), or accept
fewer hard. Medium (3,000 of a 51k pool) scales freely. Re-confirm hard yield before committing to
750 — if it stays ~10–15%, cap hard lower and let medium absorb the rest.

## Evidence (prompt exploration, n=32/tier)

| prompt | easy succ | med succ | hard succ | goto behavior |
|--------|----------:|---------:|----------:|---------------|
| strict (no search) | ~42% | ~23% | ~12% | can't do research tasks |
| **permissive** (over-loose) | 13% | 14% | **0%** | 86–96% of attempts goto; flails search on easy |
| **balanced** (current) | **41%** | **34%** | **10%** | easy demos on-page; search only when needed |

- Per-task A/B: `191272` (arxiv) — permissive `goto google`; balanced stays on arxiv, uses its own
  search, succeeds.
- Click-through (productive research vs snippet-scrape): medium 18/18, hard 23/24 search-trajectories
  click through → the `serp_only` filter keeps them and drops only the snippet-scrapers.
- **Bing/Google blank-SERP** confirmed by screenshot → forced engine-bouncing. The ddg-prompt fix
  (the +ddg prompt) made search single-engine: medium 20/20 search demos use only DuckDuckGo (the prior prompt bounced).
- Post-filter clean demos (per 32): balanced easy 12 / med 11 / hard 3 → balanced+ddg
  easy 13 / med 15 / hard 2–3. ddg lifts easy+medium and removes the bounce; hard stays low/noisy.
- **Site-start hard demos come in two good shapes, both kept:** (a) solved **on-page** —
  e.g. `172888` Amazon search, `177424` Apple spec comparison (scroll), `251269` BBC search; and
  (b) solved by **ddg → click a result → read** (research) when the open site lacks the answer.
  `serp_only` keeps both and drops only snippet-scraping. (Search-*start* hard — opening on
  google.com — is excluded at collection: its SERP is blank, ~0 yield. §1/§3.)
- **End-to-end chain validated:** collect→clean→stage→`export_sft` on the +ddg demos → 30 success rows →
  valid Qwen SFT parquet (GPT `action_description` → Qwen `Action:`+`<tool_call>` targets).

## Current status

- The ddg prompt, site-start restriction, and clean→stage→export_sft chain have already been
  validated on small batches.
- Current collection should use the env-server path and the §1/§3 tier mix. Keep hard near the
  documented share unless a fresh scaled batch proves the site-start yield is materially higher.
- Canonical row validation now requires nested Lite tool calls (`id/type/function`) and hard-fails
  null or missing `function.arguments`; legacy flat/provider-envelope repair belongs under
  `devs/migration`, not `lite/data`.
- For any scaled batch, record per-tier clean yield before publishing. The small-batch hard yield is
  noisy enough that it should not by itself justify raising the hard share.
