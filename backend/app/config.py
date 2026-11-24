"""All configuration comes from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _split(value: str | None) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def sqlite_path_from_url(url: str) -> str:
    """sqlite:////data/app.db -> /data/app.db ; sqlite:///./app.db -> ./app.db"""
    prefix = "sqlite:///"
    if url.startswith(prefix):
        return url[len(prefix):] or ":memory:"
    if url.startswith("sqlite://"):
        return url[len("sqlite://"):] or ":memory:"
    return url  # plain file path is accepted too


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Answer clearly and accurately."
)


@dataclass
class Settings:
    api_keys: list[str] = field(default_factory=list)
    key_models: list[str] = field(default_factory=list)
    default_model: str = "gemini/gemini-3.5-flash-lite"
    database_path: str = "./app.db"
    allowed_origins: list[str] = field(default_factory=lambda: ["*"])
    enable_failure_simulation: bool = False
    cache_enabled: bool = True
    cache_ttl_seconds: int = 86400
    llm_timeout_seconds: float = 30.0
    llm_temperature: float = 0.3
    llm_max_output_tokens: int = 1024
    max_history_messages: int = 20
    default_system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        return cls(
            api_keys=_split(env.get("LLM_API_KEYS")),
            key_models=_split(env.get("LLM_KEY_MODELS")),
            default_model=env.get("LLM_MODEL", "gemini/gemini-3.5-flash-lite").strip(),
            database_path=sqlite_path_from_url(
                env.get("DATABASE_URL", "sqlite:///./app.db")
            ),
            allowed_origins=_split(env.get("ALLOWED_ORIGINS")) or ["*"],
            enable_failure_simulation=_bool(env.get("ENABLE_FAILURE_SIMULATION")),
            cache_enabled=_bool(env.get("CACHE_ENABLED"), default=True),
            cache_ttl_seconds=int(env.get("CACHE_TTL_SECONDS", "86400")),
            llm_timeout_seconds=float(env.get("LLM_TIMEOUT_SECONDS", "30")),
            llm_temperature=float(env.get("LLM_TEMPERATURE", "0.3")),
            llm_max_output_tokens=int(env.get("LLM_MAX_OUTPUT_TOKENS", "1024")),
            max_history_messages=int(env.get("MAX_HISTORY_MESSAGES", "20")),
            default_system_prompt=env.get("DEFAULT_SYSTEM_PROMPT")
            or DEFAULT_SYSTEM_PROMPT,
        )
