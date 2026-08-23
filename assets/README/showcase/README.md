# Benchmark Showcase GIFs

The five GIFs in the top-level [README](/README.md) — one `gpt-5.5` trajectory per
benchmark, each picked to be visually representative of its environment family
(desktop / browser / mobile).

## Files

```
├── make_showcase.sh      # Reproduction script — runs the 5 rollouts, collects + packs the GIFs
├── lite_demo.gif         # desktop — lite.demo create_file
├── lite_osworld.gif      # desktop — lite.osworld LibreOffice Impress
├── webarena.gif          # browser — browsergym.webarena One Stop Market storefront
├── androidworld.gif     # mobile  — androidworld Create contact
└── mobilegym.gif         # mobile  — mobilegym Spotify queue + play
```

## Tasks

| GIF | Env | Task | What it shows |
|---|---|---|---|
| `lite_demo.gif`     | `lite.demo`           | `create_file`                          | Terminal: create a file on a GNOME desktop |
| `lite_osworld.gif`  | `lite.osworld`        | `osworld_libreoffice_impress_05dd4c1d` | LibreOffice Impress slide editing |
| `webarena.gif`      | `browsergym.webarena` | `21`                                   | Browsing the One Stop Market storefront |
| `androidworld.gif` | `androidworld`       | `ContactsAddContact`                   | Android Material "Create contact" form |
| `mobilegym.gif`     | `mobilegym`           | `spotify.AddToQueueAndPlay`            | Spotify: search a song, queue + play it |

Each GIF starts from rollout `--save-gif` output (no-op turns — `screenshot` /
`wait` / `noop` — dropped so it doesn't "freeze"), then post-processed by
`make_showcase.sh`: desktop/browser → 720px wide, mobile → 480px, a thin gray border
baked in (GitHub strips `<img>` CSS), and a 128-color palette to keep the files
small — none of which changes the README layout.

## Reproduce

```bash
export OPENAI_API_KEY=...        # gpt-5.5 (Azure or OpenAI)
export OPENAI_BASE_URL=...
uv run bash assets/README/showcase/make_showcase.sh
```

`lite.demo` / `lite.osworld` / `androidworld` / `mobilegym` run in-process (one
local container each). **WebArena** needs an env-server; `--warm-singleton`
prewarms its shopping/gitlab/forum/wikipedia app stack, which direct mode can't
start. The server process must also export `OPENAI_API_KEY` (the task
evaluator reads it at `env.reset()`):

```bash
OPENAI_API_KEY="$OPENAI_API_KEY" OPENAI_BASE_URL="$OPENAI_BASE_URL" \
  uv run python scripts/serve_env.py --port 30911 --token "$TOK" \
    --env-ids browsergym.webarena --warm-singleton &  # prewarms the stack (~2-3 min)
export CUA_LITE_ENV_SERVER_URL=http://localhost:30911 CUA_LITE_ENV_SERVER_TOKEN="$TOK"
uv run bash assets/README/showcase/make_showcase.sh       # WebArena step now runs too
```

See [docs/envs.md](/docs/envs.md) for per-env installation and the env-server guide.
