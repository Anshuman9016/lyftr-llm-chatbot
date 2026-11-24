"""LiteLLM gateway: key pool + ordered failover via litellm.Router, and LiteLLM's response cache.

Every key in LLM_API_KEYS becomes one Router deployment in its own model group
(key_1, key_2, ...), and the Router is configured with ordered fallbacks
key_1 -> key_2 -> ... . Any provider failure (429 / quota, 5xx, timeout, connection
error, invalid key) makes the Router fall back to the next key. If every key fails,
AllProvidersFailed is raised and the API answers 503.

X-Simulate-Failure is implemented as a LiteLLM callback (`async_pre_call_check`) that
raises the same LiteLLM exception a real provider error would, for the listed keys,
for the current request only (per-request state lives in a ContextVar).
"""

from __future__ import annotations

import contextvars
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import litellm
from litellm import Router
from litellm.caching.caching import Cache
from litellm.integrations.custom_logger import CustomLogger

from .config import Settings

log = logging.getLogger("gateway")

litellm.suppress_debug_info = True
litellm.drop_params = True  # ignore params a given provider does not support

SIMULATABLE_STATUSES = {429, 500, 502, 503}

LIVE_INFERENCE = "LIVE_INFERENCE"
LLM_CACHE = "LLM_CACHE"

# Keys that must fail for the request currently being handled: {"key_1": 429, ...}
_simulated_failures: contextvars.ContextVar[dict[str, int]] = contextvars.ContextVar(
    "simulated_failures", default={}
)


class AllProvidersFailed(Exception):
    def __init__(self, failures: list[dict[str, Any]] | None = None):
        super().__init__("all providers unavailable")
        self.failures = failures or []


@dataclass
class Deployment:
    key_id: str
    api_key: str
    model: str


@dataclass
class ChatResult:
    reply: str
    resolution_source: str
    model: str
    input_tokens: int
    output_tokens: int
    key_id: str | None
    fallback_attempts: int
    failures: list[dict[str, Any]] = field(default_factory=list)


def parse_simulation_header(value: str | None, key_ids: list[str]) -> dict[str, int]:
    """'key_1=429,key_2=500' or '*=429' -> {'key_1': 429, ...}. Invalid pairs are ignored."""
    result: dict[str, int] = {}
    if not value:
        return result
    for pair in value.split(","):
        if "=" not in pair:
            continue
        target, _, status_raw = pair.partition("=")
        target, status_raw = target.strip(), status_raw.strip()
        try:
            status = int(status_raw)
        except ValueError:
            continue
        if status not in SIMULATABLE_STATUSES:
            continue
        if target == "*":
            for key_id in key_ids:
                result[key_id] = status
        elif target in key_ids:
            result[target] = status
    return result


def simulated_error(status: int, model: str, key_id: str) -> Exception:
    """Build the same exception type LiteLLM raises for a real provider error."""
    provider = model.split("/", 1)[0] if "/" in model else "unknown"
    msg = f"Simulated HTTP {status} for {key_id} (X-Simulate-Failure)"
    if status == 429:
        return litellm.RateLimitError(msg, llm_provider=provider, model=model)
    if status == 500:
        return litellm.InternalServerError(msg, llm_provider=provider, model=model)
    if status == 502:
        return litellm.BadGatewayError(msg, llm_provider=provider, model=model)
    return litellm.ServiceUnavailableError(msg, llm_provider=provider, model=model)


class FailureSimulator(CustomLogger):
    """Router callback: makes the simulated keys fail just before they would be called."""

    async def async_pre_call_check(self, deployment: dict, parent_otel_span=None):
        key_id = (deployment.get("model_info") or {}).get("id")
        status = _simulated_failures.get().get(key_id)
        if status:
            model = (deployment.get("litellm_params") or {}).get("model", "unknown")
            log.warning("%s failed (simulated HTTP %s)", key_id, status)
            raise simulated_error(status, model, key_id)
        return None


class FailureLogger(CustomLogger):
    """Logs every failed key attempt (never the key itself)."""

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        info = (kwargs.get("litellm_params") or {}).get("model_info") or {}
        exc = kwargs.get("exception")
        log.warning(
            "%s failed (%s %s): %s",
            info.get("id", "?"),
            type(exc).__name__ if exc else "error",
            getattr(exc, "status_code", None),
            str(exc)[:200] if exc else "",
        )


_simulator = FailureSimulator()
_failure_logger = FailureLogger()


class LLMGateway:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.deployments: list[Deployment] = []
        for i, key in enumerate(settings.api_keys):
            model = (
                settings.key_models[i]
                if i < len(settings.key_models) and settings.key_models[i]
                else settings.default_model
            )
            self.deployments.append(Deployment(f"key_{i + 1}", key, model))

        for cb in (_simulator, _failure_logger):
            if cb not in litellm.callbacks:
                litellm.callbacks.append(cb)

        ids = self.key_ids
        self.router: Router | None = None
        if self.deployments:
            self.router = Router(
                model_list=[
                    {
                        "model_name": d.key_id,  # one model group per key
                        "litellm_params": {
                            "model": d.model,
                            "api_key": d.api_key,
                            "timeout": settings.llm_timeout_seconds,
                        },
                        "model_info": {"id": d.key_id},
                    }
                    for d in self.deployments
                ],
                # Ordered failover: key_1 -> key_2 -> ... -> key_N
                fallbacks=[{ids[i]: ids[i + 1:]} for i in range(len(ids) - 1)],
                num_retries=0,  # move to the next key straight away
                # Cooldowns would make one request's (possibly simulated) failure
                # affect other users' requests, so they are off.
                disable_cooldowns=True,
            )

        # LiteLLM's built-in in-memory cache. Called explicitly so the cache key is
        # exactly: model + full message list (+ fixed sampling params) - never the API key.
        self.cache: Cache | None = (
            Cache(type="local", ttl=settings.cache_ttl_seconds)
            if settings.cache_enabled
            else None
        )

    @property
    def key_ids(self) -> list[str]:
        return [d.key_id for d in self.deployments]

    def _cache_kwargs(self, model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "model": model,
            "messages": messages,
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_output_tokens,
        }

    async def _cache_get(self, model: str, messages: list[dict[str, str]]) -> dict | None:
        if not self.cache:
            return None
        try:
            hit = await self.cache.async_get_cache(**self._cache_kwargs(model, messages))
        except Exception:  # a cache problem must never break chat
            log.exception("cache read failed")
            return None
        if isinstance(hit, dict) and hit.get("reply"):
            return hit
        return None

    async def _cache_put(self, model: str, messages: list[dict[str, str]], value: dict) -> None:
        if not self.cache:
            return
        try:
            await self.cache.async_add_cache(value, **self._cache_kwargs(model, messages))
        except Exception:
            log.exception("cache write failed")

    async def chat(
        self, messages: list[dict[str, str]], simulate: dict[str, int] | None = None
    ) -> ChatResult:
        if not self.router:
            raise AllProvidersFailed()
        simulate = simulate or {}

        # Cache lookup for the first key that is not (simulated) down.
        first_up = next((d for d in self.deployments if d.key_id not in simulate), None)
        if first_up:
            cached = await self._cache_get(first_up.model, messages)
            if cached:
                return ChatResult(
                    reply=cached["reply"],
                    resolution_source=LLM_CACHE,
                    model=cached.get("model", first_up.model),
                    input_tokens=int(cached.get("input_tokens", 0)),
                    output_tokens=int(cached.get("output_tokens", 0)),
                    key_id=None,
                    fallback_attempts=0,
                )

        token = _simulated_failures.set(simulate)
        started = time.perf_counter()
        try:
            response = await self.router.acompletion(
                model=self.deployments[0].key_id,  # always start at key_1
                messages=messages,
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_output_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - every key failed
            raise AllProvidersFailed(
                [{"error": type(exc).__name__, "status": getattr(exc, "status_code", None)}]
            ) from exc
        finally:
            _simulated_failures.reset(token)

        hidden = getattr(response, "_hidden_params", {}) or {}
        key_id = hidden.get("model_id") or self.deployments[0].key_id
        dep = next((d for d in self.deployments if d.key_id == key_id), self.deployments[0])
        attempts = self.key_ids.index(dep.key_id)  # keys before it in the chain all failed

        reply = (response.choices[0].message.content or "").strip()
        if not reply:
            raise AllProvidersFailed([{"key_id": dep.key_id, "error": "empty reply"}])

        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        if not input_tokens:
            input_tokens = litellm.token_counter(model=dep.model, messages=messages)
        if not output_tokens:
            output_tokens = litellm.token_counter(model=dep.model, text=reply)

        await self._cache_put(
            dep.model,
            messages,
            {
                "reply": reply,
                "model": dep.model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )
        log.info(
            "live inference via %s (%s) in %.0f ms after %d failed key(s)",
            dep.key_id, dep.model, (time.perf_counter() - started) * 1000, attempts,
        )
        return ChatResult(
            reply=reply,
            resolution_source=LIVE_INFERENCE,
            model=dep.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            key_id=dep.key_id,
            fallback_attempts=attempts,
        )
