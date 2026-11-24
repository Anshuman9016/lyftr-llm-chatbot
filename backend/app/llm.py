"""LiteLLM gateway: key pool with ordered failover + LiteLLM's built-in response cache.

Every key in LLM_API_KEYS becomes one "deployment" (key_1, key_2, ...). A request walks
the pool in order; any provider error (429 / quota, 5xx, timeout, connection error,
invalid key) moves on to the next key. If every key fails, AllProvidersFailed is raised
and the API answers 503.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import litellm
from litellm.caching.caching import Cache

from .config import Settings

log = logging.getLogger("gateway")

litellm.suppress_debug_info = True
litellm.drop_params = True  # ignore params a given provider does not support

SIMULATABLE_STATUSES = {429, 500, 502, 503}

LIVE_INFERENCE = "LIVE_INFERENCE"
LLM_CACHE = "LLM_CACHE"


class AllProvidersFailed(Exception):
    def __init__(self, failures: list[dict[str, Any]]):
        super().__init__("all providers unavailable")
        self.failures = failures


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


CompletionFn = Callable[..., Awaitable[Any]]


class LLMGateway:
    def __init__(self, settings: Settings, completion_fn: CompletionFn | None = None):
        self.settings = settings
        self.completion_fn = completion_fn or litellm.acompletion
        self.deployments: list[Deployment] = []
        for i, key in enumerate(settings.api_keys):
            model = (
                settings.key_models[i]
                if i < len(settings.key_models) and settings.key_models[i]
                else settings.default_model
            )
            self.deployments.append(Deployment(f"key_{i + 1}", key, model))

        # LiteLLM's built-in in-memory cache. We call it explicitly so the cache key is
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
        simulate = simulate or {}
        failures: list[dict[str, Any]] = []

        for dep in self.deployments:
            try:
                if dep.key_id in simulate:
                    raise simulated_error(simulate[dep.key_id], dep.model, dep.key_id)

                cached = await self._cache_get(dep.model, messages)
                if cached:
                    return ChatResult(
                        reply=cached["reply"],
                        resolution_source=LLM_CACHE,
                        model=cached.get("model", dep.model),
                        input_tokens=int(cached.get("input_tokens", 0)),
                        output_tokens=int(cached.get("output_tokens", 0)),
                        key_id=None,
                        fallback_attempts=0,
                        failures=failures,
                    )

                started = time.perf_counter()
                response = await self.completion_fn(
                    model=dep.model,
                    messages=messages,
                    api_key=dep.api_key,
                    temperature=self.settings.llm_temperature,
                    max_tokens=self.settings.llm_max_output_tokens,
                    timeout=self.settings.llm_timeout_seconds,
                    num_retries=0,  # failover is handled here, across keys
                )
                reply = (response.choices[0].message.content or "").strip()
                if not reply:
                    raise ValueError("provider returned an empty reply")

                usage = getattr(response, "usage", None)
                input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                if not input_tokens:
                    input_tokens = litellm.token_counter(model=dep.model, messages=messages)
                if not output_tokens:
                    output_tokens = litellm.token_counter(model=dep.model, text=reply)

                result = ChatResult(
                    reply=reply,
                    resolution_source=LIVE_INFERENCE,
                    model=dep.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    key_id=dep.key_id,
                    fallback_attempts=len(failures),
                    failures=failures,
                )
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
                    dep.key_id, dep.model, (time.perf_counter() - started) * 1000, len(failures),
                )
                return result

            except Exception as exc:  # noqa: BLE001 - any provider failure => next key
                status = getattr(exc, "status_code", None)
                failures.append(
                    {"key_id": dep.key_id, "status": status, "error": type(exc).__name__}
                )
                # Never log the key itself; LiteLLM messages can be long, so trim.
                log.warning(
                    "%s failed (%s %s): %s",
                    dep.key_id, type(exc).__name__, status, str(exc)[:200],
                )

        raise AllProvidersFailed(failures)
