"""FastAPI app: GET /health, POST /chat, GET /sessions/{session_id}/messages."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from .config import Settings
from .db import Database
from .llm import AllProvidersFailed, LLMGateway, parse_simulation_header

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("app")

MAX_MESSAGE_CHARS = 4000


# ---------------------------------------------------------------- schemas
class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str
    system_prompt: Optional[str] = Field(default=None, max_length=8000)

    @field_validator("session_id")
    @classmethod
    def _session_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("session_id must not be empty")
        return v

    @field_validator("message")
    @classmethod
    def _message_rules(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be empty")
        if len(v) > MAX_MESSAGE_CHARS:
            raise ValueError(f"message must be at most {MAX_MESSAGE_CHARS} characters")
        return v

    @field_validator("system_prompt")
    @classmethod
    def _system_prompt_blank_is_none(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        return v or None


class ChatMetadata(BaseModel):
    resolution_source: str
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: Optional[str]
    fallback_attempts: int


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    metadata: ChatMetadata


# ---------------------------------------------------------------- app factory
def create_app(settings: Settings | None = None, gateway: LLMGateway | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    db = Database(settings.database_path)
    gateway = gateway or LLMGateway(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        db.init()
        log.info(
            "started: %d key(s) %s, cache=%s, failure_simulation=%s, db=%s",
            len(gateway.deployments),
            [f"{d.key_id}:{d.model}" for d in gateway.deployments],
            settings.cache_enabled,
            settings.enable_failure_simulation,
            settings.database_path,
        )
        yield

    show_docs = os.environ.get("ENABLE_DOCS", "false").lower() == "true"
    app = FastAPI(
        title="LLM Chatbot with Key Failover and Response Caching",
        lifespan=lifespan,
        docs_url="/docs" if show_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if show_docs else None,
    )
    app.state.settings, app.state.db, app.state.gateway = settings, db, gateway

    allow_all = "*" in settings.allowed_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if allow_all else settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],  # includes Content-Type and X-Simulate-Failure
    )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):  # never leak a stack trace
        log.exception("unhandled error: %s", exc)
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    # ------------------------------------------------------------ endpoints
    @app.get("/health")
    async def health():
        db_ok = await run_in_threadpool(db.ping)
        keys = len(gateway.deployments)
        body = {
            "status": "ok" if (db_ok and keys > 0) else "unavailable",
            "keys_configured": keys,
            "cache_enabled": gateway.cache is not None,
        }
        if not db_ok or keys == 0:
            body["detail"] = "database unreachable" if not db_ok else "no API keys configured"
            return JSONResponse(status_code=503, content=body)
        return body

    @app.post("/chat", response_model=ChatResponse)
    async def chat(
        req: ChatRequest,
        x_simulate_failure: Optional[str] = Header(default=None, alias="X-Simulate-Failure"),
    ):
        started = time.perf_counter()

        stored_prompt = await run_in_threadpool(db.get_system_prompt, req.session_id)
        system_prompt = req.system_prompt or stored_prompt or settings.default_system_prompt
        history = await run_in_threadpool(
            db.get_history, req.session_id, settings.max_history_messages
        )
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages += [{"role": m["role"], "content": m["content"]} for m in history]
        messages.append({"role": "user", "content": req.message})

        simulate = (
            parse_simulation_header(x_simulate_failure, gateway.key_ids)
            if settings.enable_failure_simulation
            else {}
        )

        try:
            result = await gateway.chat(messages, simulate)
        except AllProvidersFailed as exc:
            log.error("all providers failed: %s", exc.failures)
            raise HTTPException(status_code=503, detail="all providers unavailable")

        metadata = {
            "resolution_source": result.resolution_source,
            "model": result.model,
            "latency_ms": 0,  # filled in below
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "key_id": result.key_id,
            "fallback_attempts": result.fallback_attempts,
        }
        metadata["latency_ms"] = int(round((time.perf_counter() - started) * 1000))
        await run_in_threadpool(
            db.save_exchange,
            req.session_id,
            req.message,
            result.reply,
            metadata,
            req.system_prompt,
        )
        return {"session_id": req.session_id, "reply": result.reply, "metadata": metadata}

    @app.get("/sessions/{session_id}/messages")
    async def session_messages(session_id: str):
        exists = await run_in_threadpool(db.session_exists, session_id)
        if not exists:
            raise HTTPException(status_code=404, detail="session not found")
        messages = await run_in_threadpool(db.get_history, session_id)
        return {"session_id": session_id, "messages": messages}

    return app


app = create_app()
