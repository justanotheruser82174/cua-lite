See [AGENTS.md](/devs/agents/api/AGENTS.md) and [shared guidelines](/devs/agents/AGENTS.md).

# Gemini Agent

**Goal:** Computer-use agent using Google's Gemini models over the **native**
`generateContent` REST surface.

**Shipped configs:** `lite.osworld`, `osworld`, `osworld_2` only — the pure-GUI
desktop rows. The code supports browser and mobile too (`gemini@browser@use`,
`gemini@mobile@use`, verified end-to-end on mobilegym at `episode_return=1.0`),
but no config ships for them: every row needing `extra_tools` fails against this
deployment (failure 1 below), the browser rows fail for an unrelated reason even
without extra tools (failure 2), and a row that runs only by dropping its tools
produces numbers not comparable to the gpt/claude twins. See the blocker below;
restore those rows from the twins once it is resolved.

**References** (priority order — measurement beats docs, docs beat the reference
impls; where they conflict, the earlier one wins):
- **official doc:** https://ai.google.dev/gemini-api/docs/computer-use
- **official reference (desktop/browser):** `${CUA_LITE_REFERENCES_ROOT}/gemini-computer-use-preview`
- **official reference (mobile):** `${CUA_LITE_REFERENCES_ROOT}/gemini-android-computer-use-quickstart`

**Agent slug:** `gemini`

**Files:**
- `lite/agents/models/gemini/action_space.py` — `GeminiDesktopActionSpace`, `GeminiMobileActionSpace`
- `lite/agents/models/gemini/agent.py` — `GeminiDesktopUseAgent(key=r"gemini@(desktop|browser)@use")`, `GeminiMobileUseAgent(key="gemini@mobile@use")`
- `lite/agents/models/gemini/utils/transport.py` — the ONLY module that knows the URL, headers, and httpx
- `lite/agents/models/gemini/utils/{parse,history,loop}.py` — response parsing and
  provenance, the provider/durable history split, and the terminal-tool hook

**Launch:**
```bash
uv run python scripts/rollout.py --model-id gemini-3.6-flash --env-id lite.osworld \
  --config-path scripts/configs/gemini/default/lite.osworld.yaml --head 1 \
  --agent-kwargs '{"api_base": "<proxy-root>", "api_key": "<key>"}'
```

`api_base` / `api_key` are **agent_kwargs, not env vars**. Unlike the Claude and
GPT rows, this family never touches litellm, so there is no `*_API_KEY` pickup —
`transport.py` raises when `api_base` is missing.

## Key Details

- **API:** `POST {api_base}/v1beta/models/{model}:generateContent` over httpx,
  with the key in the `x-goog-api-key` header. **Not** litellm: it has no
  representation for `computerUse` at all, so the OpenAI-compatible path drops
  the whole tool (`VertexToolName` knows only googleSearch /
  googleSearchRetrieval / enterpriseWebSearch / url_context / code_execution /
  googleMaps; anything else hits the `Invalid tool={}` branch). Losing the tool
  loses `environment`, which is what selects the desktop / browser / mobile verb
  set.
- **Coordinate system:** identity. Gemini emits flat `x`/`y` integers on the
  canonical 0-1000 grid, so there is no pixel domain to cross and `resolution` is
  accepted and ignored. The one asymmetry: canonical `MAX_NORM = 1000` is
  inclusive while the wire tops out at 999, so the render leg clamps that single
  value (1000 → 999).
- **Black-box tool:** the client declares no action schema.
  `get_tool_schemas()` returns `[]` on both spaces;
  `excludedPredefinedFunctions` only ever SUBTRACTS from the server's set. The
  exclusion list is intersected with the offered verbs, which makes "excluded a
  verb this environment does not have" — an opaque server-side rejection —
  unrepresentable.
- **Verb sets** (enumerated against the live API, not the docs): desktop 17,
  browser 20 (desktop + `navigate` / `go_back` / `go_forward`), mobile 10. The
  three browser-only verbs are always excluded — they have no canonical action,
  and browser navigation is an env extra tool (`goto` / `back` / `forward`).
- **`thoughtSignature`:** a Part-level sibling of `functionCall`, **not** an
  entry in `args`. It is replayed verbatim by echoing the response's raw parts;
  rebuilding a model Part field-by-field would drop it and earn a 400.
- **`intent`:** Gemini puts it INSIDE `functionCall.args`, alongside the real
  action parameters. It is stripped out with `safety_decision` before the
  remainder is treated as canonical action arguments, and lifted to the message
  layer as `action_description`. Several calls in one turn join into ONE
  description — `get_action_description` returns only the first non-empty part.
- **Reasoning depth:** `api_kwargs.thinking_level` →
  `generationConfig.thinkingConfig.thinkingLevel`, upper-cased on the wire.
  Legal: `minimal` / `low` / `medium` / `high`. `xhigh` and `max` are rejected;
  `high` is the ceiling.
- **extra_tools:** declared as a native `functionDeclarations` Tool alongside
  `computerUse`. A custom tool whose name collides with a predefined verb (e.g.
  `open_app`) MUST also appear in `excludedPredefinedFunctions`, or the request
  is rejected — which is why `open_app` is excluded (the builtin takes an
  Android package name, the env's tool takes display names: same name,
  incompatible domain). `list_apps` is excluded for an unrelated reason — no
  canonical action and no env handler, so offering it only wastes a turn.
- **Frames:** an action batch shows the provider only its FINAL frame
  (`pairs[-1:]`), while durable Lite storage keeps every frame the env returned.

## Why no mobile / browser / bash configs ship

Two distinct failures. Nothing in this family's code is known to be at fault,
but the origin of (1) is NOT proven — see the caveat there. Both are reasons the
configs were withdrawn rather than shipped in a degraded shape.

### 1. Custom tools + a computerUse call in history

Isolated by A/B — rows 1 and 2 carry an identical `tools` list and differ only
in the history; row 3 is the control that varies `tools` instead:

| history contains | tools | result |
|---|---|---|
| `bash(...)` — a CUSTOM call | `[computerUse, functionDeclarations]` | 200 |
| `take_screenshot(...)` — a computerUse verb | `[computerUse, functionDeclarations]` | **500** |
| `take_screenshot(...)` — a computerUse verb | `[computerUse]` | 200 |

So it is **not** "custom tools do not work". They work alone, and alongside
`computerUse`, as long as the history's calls are custom. What is refused is
declaring `functionDeclarations` while the history holds a **computerUse** call —
i.e. every GUI rollout from turn 2. This is what killed the mobile rows (whose
`open_app` / `response` / `terminate` are env extras) and `lite.osworld.bash`
(whose `bash` IS the feature, so it cannot be dropped the way mobile's extras
could).

A plausible mechanism: litellm has no representation for Gemini's `computerUse`
at all — `VertexToolName` (`litellm/types/llms/vertex_ai.py:217`) enumerates only
`googleSearch`, `googleSearchRetrieval`, `enterpriseWebSearch`, `url_context`,
`code_execution`, `googleMaps`. A history call naming a computerUse verb
therefore matches nothing litellm knows, and the mismatch only becomes reachable
once `functionDeclarations` turns its tool-translation path on.

**Origin is not proven**, because this deployment masks it: every upstream non-2xx
arrives as an opaque plain-text 500 — reproduce with `-d '{"contents":[]}'`, no
computer use needed. That is litellm issue
[#25466](https://github.com/BerriAI/litellm/issues/25466), still open as of
v1.83.5, so **upgrading does not fix it**.

To settle it, use a path that does not translate the body:

- Set `GEMINI_API_KEY` on the proxy and point `api_base` at `<proxy>/gemini`.
  That route is raw passthrough — it only rewrites the URL and appends the key as
  a query param (`llm_passthrough_endpoints.py:213-223`), never touching `tools`.
  The proxy already holds Google credentials (its router path works); this is a
  different lookup of the same secret, not a new key.
- Or replay against `https://generativelanguage.googleapis.com` with a native
  Google AI Studio key, which surfaces the real JSON error.

No code in this family changes for either route: `transport.py` builds the URL as
`api_base.rstrip("/") + "/v1beta/models/{model}:generateContent"`.

### 2. ENVIRONMENT_BROWSER fails on turn 2

The two `browsergym` rows never completed a rollout and have been withdrawn.
Probed directly (twice): under
`environment: ENVIRONMENT_BROWSER` the SECOND request of an episode returns an
opaque 500 with `computerUse` as the only declared tool and nothing excluded. The
identical two-turn probe returns 200 under `ENVIRONMENT_DESKTOP` and
`ENVIRONMENT_MOBILE`. So this is not the `functionDeclarations` problem — it is
specific to the browser environment, and it makes both browsergym rows
non-functional until investigated — a strictly separate problem from (1), since
no custom tool is involved.
