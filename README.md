# Relay: LLM chatbot with key failover and response caching

*Relay never drops the baton: when one API key fails, the next one picks up the request.*

React chat UI → FastAPI backend → LiteLLM gateway (key pool + response cache) → Gemini / Groq.
Session history lives in SQLite. Everything runs on free tiers.

```
┌──────────────┐  POST /chat   ┌──────────────────────── FastAPI ────────────────────────┐
│ React (Vite) │ ────────────▶ │ validate ─▶ load history (SQLite) ─▶ LLMGateway          │
│  data-testid │ ◀──────────── │                                     │                    │
└──────────────┘  reply+meta   │   for key in [key_1, key_2, ...]:   ▼                    │
                               │     simulated failure? → next key                       │
                               │     LiteLLM cache hit?  → LLM_CACHE                      │
                               │     litellm.acompletion → LIVE_INFERENCE (store in cache)│
                               │     429 / quota / 5xx / timeout → next key               │
                               │   all failed → 503 "all providers unavailable"           │
                               │ save user+assistant turn (SQLite, one transaction)       │
                               └──────────────────────────────────────────────────────────┘
```

## 1. Live URLs

| Tier | URL |
|---|---|
| Frontend | https://lyftr-llm-chatbot-pied.vercel.app |
| Backend | https://lyftr-llm-chatbot-api.onrender.com (try [`/health`](https://lyftr-llm-chatbot-api.onrender.com/health)) |
| Repository | https://github.com/Anshuman9016/lyftr-llm-chatbot |

> The backend runs on Render's free plan, which sleeps after ~15 min idle. The first request after
> a sleep can take ~30–60 s; open `/health` once before testing.

## 2. Run locally

### With Docker

```bash
cp .env.example .env          # then put 2+ real keys in LLM_API_KEYS
make up                       # backend → http://localhost:8000, frontend → http://localhost:3000
make logs                     # follow logs
make down                     # stop (SQLite volume is kept)
make clean                    # stop and delete the SQLite volume
```

`docker-compose.yml` runs both services and mounts the named volume `app-data` at `/data`, where
the backend keeps `app.db` (`DATABASE_URL=sqlite:////data/app.db`), so history survives restarts.
The frontend image reads `PUBLIC_API_URL` at container start (written into `config.js`), so the same
image works against any backend.

### Without Docker

```bash
# backend (Python 3.11+)
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
export LLM_API_KEYS="key-one,key-two" ENABLE_FAILURE_SIMULATION=true \
       DATABASE_URL="sqlite:///./app.db" ALLOWED_ORIGINS="http://localhost:3000"
uvicorn app.main:app --reload --port 8000
python -m pytest -q                                    # 38 tests, no network or real keys needed

# frontend (Node 18+), in another terminal
cd frontend
npm install
VITE_API_URL=http://localhost:8000 npm run dev         # http://localhost:3000
```

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `LLM_API_KEYS` | – | Comma-separated key pool (≥ 2). Ids are `key_1`, `key_2`, … in order |
| `LLM_MODEL` | `gemini/gemini-3.5-flash-lite` | LiteLLM model used by every key |
| `LLM_KEY_MODELS` | empty | Optional per-key model override, e.g. `gemini/gemini-3.5-flash-lite,groq/llama-3.1-8b-instant` |
| `DATABASE_URL` | `sqlite:///./app.db` | SQLite location (`sqlite:////data/app.db` in Docker) |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `ENABLE_FAILURE_SIMULATION` | `false` | `true` enables the `X-Simulate-Failure` header |
| `CACHE_ENABLED` / `CACHE_TTL_SECONDS` | `true` / `86400` | LiteLLM response cache |
| `LLM_TIMEOUT_SECONDS` | `30` | Per-key timeout before failing over |
| `LLM_TEMPERATURE` / `LLM_MAX_OUTPUT_TOKENS` | `0.3` / `1024` | Sampling settings (part of the cache key) |
| `MAX_HISTORY_MESSAGES` | `20` | Previous messages sent as context |
| `DEFAULT_SYSTEM_PROMPT` | built-in | Used when a session has no system prompt |
| `ENABLE_DOCS` | `false` | Expose `/docs`; off so only the three contract endpoints exist |
| `PUBLIC_API_URL` (compose) / `VITE_API_URL` (Vercel) | `http://localhost:8000` | Backend URL the browser calls |

## 3. Design decisions

### Key fallback
- Each key in `LLM_API_KEYS` becomes a deployment (`key_1`, `key_2`, …) that can carry its own
  model, so the pool can span providers (e.g. Gemini + Groq). This matters because **Gemini quotas
  are per Google Cloud project, not per key**: two keys from one project share one quota. Use keys
  from different projects or different providers for real redundancy.
- `LLMGateway.chat` walks the pool in order and calls `litellm.acompletion(..., api_key=…, num_retries=0)`.
  Any provider failure — `RateLimitError` (429 and quota/`RESOURCE_EXHAUSTED`), `InternalServerError`,
  `BadGatewayError`, `ServiceUnavailableError`, timeouts, connection errors, even an invalid key or
  an empty reply — moves to the next key. LiteLLM normalises every provider's errors into these
  exception types, so one code path handles Gemini, Groq, OpenAI, etc.
- I drive the loop myself instead of using LiteLLM's `Router` retries so the response can report
  exactly which key answered (`key_id`) and how many failed first (`fallback_attempts`), and so the
  test hook can fail a specific key for one request without touching shared router state
  (cooldowns would leak into other users' requests).
- `X-Simulate-Failure` raises the **same LiteLLM exception** a real provider error would
  (`RateLimitError` for 429, `InternalServerError` for 500, …) for the listed keys, inside that request
  only. It is ignored unless `ENABLE_FAILURE_SIMULATION=true`. A simulated key is never called.
- If every key fails, the API returns `503 {"detail": "all providers unavailable"}` and nothing is
  written to history. A catch-all exception handler guarantees JSON errors, never stack traces.
- API keys never leave the server: responses and logs only contain `key_1`, `key_2`, …

### Caching
- Uses LiteLLM's built-in cache (`litellm.caching.Cache(type="local")`, in-memory with a TTL).
  I call `async_get_cache` / `async_add_cache` explicitly with `model + messages + temperature +
  max_tokens`, so the cache key is exactly *model + full message list* (system prompt + history +
  new message) — and deliberately **not** the API key, so a reply generated with `key_1` can be served
  when the request lands on `key_2`.
- The cached value stores the reply text, model and the original token usage, so a hit returns the
  identical reply with the stored `input_tokens` / `output_tokens`, `key_id: null`,
  `fallback_attempts: 0`. Cache hits take ~1–5 ms versus hundreds of ms for a live call.
- Because history is part of the key, the same first message in a fresh session hits the cache, but
  the same text later in a conversation (different history) does not. A changed system prompt also
  changes the key.
- Simulated failures are evaluated before the cache lookup for that key, so `*=429` always yields 503,
  matching the contract even for a cached question.
- Trade-off: the in-memory cache is per-process and resets on restart. LiteLLM's disk cache was
  removed in recent versions, and Redis is not allowed by the brief; for one instance this is fine.

### SQLite session storage
- Schema is created on startup (`CREATE TABLE IF NOT EXISTS`): the required `messages` table, plus a
  `sessions` table (session id, system prompt, timestamps) and an index on `(session_id, id)`.
- WAL mode; one short-lived connection per operation, run in a thread pool so the event loop never
  blocks.
- A turn is written only after the LLM succeeds, with the user and assistant rows in one
  transaction, so failed requests (422/503) never appear in history.
- The client sends only the new message; the server rebuilds context from SQLite (last
  `MAX_HISTORY_MESSAGES` messages) and prepends the session's system prompt. A system prompt sent
  once is remembered for the session; the history endpoint never returns it.
- `latency_ms` is measured server-side from request start to just before the response is returned.

## 4. Verification

Set `B` to the backend URL (`http://localhost:8000` locally).

```bash
B=http://localhost:8000
curl -s $B/health
# {"status":"ok","keys_configured":2,"cache_enabled":true}

# Live inference, then a cache hit from a NEW session with the same first message
curl -s $B/chat -H 'Content-Type: application/json' -d '{"session_id":"s-a","message":"What is the capital of France?"}'
curl -s $B/chat -H 'Content-Type: application/json' -d '{"session_id":"s-b","message":"What is the capital of France?"}'
# second one: "resolution_source":"LLM_CACHE", same reply, "key_id":null, latency a few ms

# Failover
curl -s $B/chat -H 'Content-Type: application/json' -H 'X-Simulate-Failure: key_1=429' \
     -d '{"session_id":"s-c","message":"Say hi"}'          # 200, key_id=key_2, fallback_attempts=1
curl -s -i $B/chat -H 'Content-Type: application/json' -H 'X-Simulate-Failure: *=429' \
     -d '{"session_id":"s-d","message":"Say hi"}'          # 503 {"detail":"all providers unavailable"}
curl -s -i $B/sessions/s-d/messages                        # 404 - the failed turn was not stored

# Multi-turn memory and history
curl -s $B/chat -H 'Content-Type: application/json' -d '{"session_id":"s-e","message":"My name is Ada."}'
curl -s $B/chat -H 'Content-Type: application/json' -d '{"session_id":"s-e","message":"What is my name?"}'
curl -s $B/sessions/s-e/messages

# Validation
curl -s -i $B/chat -H 'Content-Type: application/json' -d '{"session_id":"s-f","message":"   "}'   # 422
```

**In the UI**: every assistant reply shows chips for source (`LIVE_INFERENCE` green / `LLM_CACHE`
amber), model, latency, input and output tokens for that request only, the key that answered and a
red "failover ×N" chip when keys failed first. Reload the page (new session) and repeat your first
message to see a cache hit. The **Failover demo** dropdown sends `X-Simulate-Failure` so failover and
the 503 path can be shown without curl. Errors show as a red message in the chat.

**Tests**: `make test` (or `cd backend && python -m pytest -q`) runs 38 tests with a fake provider
covering the contract: health, validation (422), cache hit/miss rules, per-request tokens,
multi-turn and session isolation, history/404, persistence across app restarts, simulated and real
failover, the 503 path, and that keys never appear in responses.

## 5. Setup Used

- Editor: VS Code *(edit to match yours)*
- AI tools: Claude (Anthropic) for code generation, review and documentation
- LLM providers: Google AI Studio (Gemini) free tier; Groq free tier optional as a second provider
- Hosting: Render (backend, free Docker web service), Vercel (frontend, free Hobby plan)

## 6. Bonus features

- **Animated UI with [React Bits](https://github.com/DavidHDev/react-bits)**: an `Aurora` WebGL
  background, `GradientText` / `ShinyText` branding, glowing `SpotlightCard` panels and `ClickSpark`
  click effects, in a vivid violet / cyan / pink palette. Each metadata chip has its own colour
  (green = live, gold = cache, pink = failover), so the source of a reply is readable at a glance.
  The components are copied, unmodified, into `frontend/src/components/reactbits/` and add no npm dependencies.
- **Suggested prompts and live counters** (replies, cache hits, failovers) to make the demo quick.
- **Failover demo control in the UI**: a dropdown that sends `X-Simulate-Failure`, so failover is
  visible in the app itself (handy for the demo video).
- **Cross-provider key pool** (`LLM_KEY_MODELS`): mix Gemini and Groq keys so one provider's outage
  or project-wide quota doesn't take the app down.
- **Extra metadata in the UI**: `key_id` and a failover badge next to each reply, plus a backend
  status panel driven by `/health`.
- **Per-session system prompt** persisted in SQLite.
- **Bounded context** (`MAX_HISTORY_MESSAGES`) to keep token usage and latency predictable.
- **Runtime-configurable frontend image** (API URL injected at container start).
- **Hardening**: non-root backend container, Docker `HEALTHCHECK`, `/docs` disabled by default,
  generic JSON 500 handler, keys never logged.
- **CI** (GitHub Actions): backend tests, frontend build and both Docker builds on every push.
