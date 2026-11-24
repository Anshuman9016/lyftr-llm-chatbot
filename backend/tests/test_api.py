"""End-to-end API tests with a fake LLM provider (no network, no real keys)."""

from __future__ import annotations

from types import SimpleNamespace

import litellm
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm import LLMGateway, parse_simulation_header
from app.main import create_app


class FakeProvider:
    """Stands in for litellm.acompletion. Keys listed in `broken` raise real LiteLLM errors."""

    def __init__(self):
        self.calls: list[dict] = []
        self.broken: dict[str, Exception] = {}

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["api_key"] in self.broken:
            raise self.broken[kwargs["api_key"]]
        messages = kwargs["messages"]
        last = messages[-1]["content"]
        # Echo something that proves history was sent.
        user_turns = [m["content"] for m in messages if m["role"] == "user"]
        reply = f"echo: {last} | turns={len(user_turns)} | first={user_turns[0]}"
        prompt_tokens = sum(len(m["content"].split()) + 4 for m in messages)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
            usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=len(reply.split())),
        )


@pytest.fixture
def provider():
    return FakeProvider()


@pytest.fixture
def make_client(tmp_path, provider):
    def _make(**overrides):
        settings = Settings(
            api_keys=["sk-one", "sk-two"],
            default_model="gemini/gemini-test",
            database_path=str(tmp_path / "data" / "app.db"),
            enable_failure_simulation=True,
        )
        for k, v in overrides.items():
            setattr(settings, k, v)
        gateway = LLMGateway(settings, completion_fn=provider)
        return TestClient(create_app(settings, gateway))

    return _make


@pytest.fixture
def client(make_client):
    with make_client() as c:
        yield c


def chat(client, session_id, message, headers=None, **extra):
    return client.post(
        "/chat", json={"session_id": session_id, "message": message, **extra}, headers=headers or {}
    )


# ------------------------------------------------------------------ health
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "keys_configured": 2, "cache_enabled": True}


def test_health_without_keys(make_client):
    with make_client(api_keys=[]) as c:
        r = c.get("/health")
        assert r.status_code == 503
        assert "detail" in r.json()


# ------------------------------------------------------------------ chat basics
def test_chat_live_inference_contract(client):
    r = chat(client, "s-1", "What is the capital of France?")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == "s-1"
    assert body["reply"]
    meta = body["metadata"]
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert meta["model"] == "gemini/gemini-test"
    assert isinstance(meta["latency_ms"], int)
    assert meta["input_tokens"] > 0 and meta["output_tokens"] > 0
    assert meta["key_id"] == "key_1"
    assert meta["fallback_attempts"] == 0


def test_message_is_trimmed(client, provider):
    chat(client, "s-trim", "   hello   ")
    assert provider.calls[-1]["messages"][-1]["content"] == "hello"


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": "s", "message": ""},
        {"session_id": "s", "message": "    "},
        {"session_id": "", "message": "hi"},
        {"session_id": "x" * 129, "message": "hi"},
        {"message": "hi"},
        {"session_id": "s"},
        {"session_id": "s", "message": "a" * 4001},
    ],
)
def test_validation_422(client, payload, provider):
    r = client.post("/chat", json=payload)
    assert r.status_code == 422
    assert "detail" in r.json()
    assert provider.calls == []


def test_malformed_json_422(client):
    r = client.post("/chat", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert "detail" in r.json()


def test_4000_chars_after_trim_is_ok(client):
    r = chat(client, "s-long", "  " + "a" * 4000 + "  ")
    assert r.status_code == 200


# ------------------------------------------------------------------ caching
def test_same_first_message_new_session_hits_cache(client, provider):
    a = chat(client, "s-a", "What is the capital of France?").json()
    b = chat(client, "s-b", "What is the capital of France?").json()
    assert a["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert b["metadata"]["resolution_source"] == "LLM_CACHE"
    assert b["reply"] == a["reply"]
    assert b["metadata"]["key_id"] is None
    assert b["metadata"]["fallback_attempts"] == 0
    assert b["metadata"]["input_tokens"] == a["metadata"]["input_tokens"]
    assert b["metadata"]["output_tokens"] == a["metadata"]["output_tokens"]
    assert len(provider.calls) == 1


def test_different_message_never_from_cache(client):
    chat(client, "s-a", "What is the capital of France?")
    r = chat(client, "s-b", "What is the capital of Spain?").json()
    assert r["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_cache_key_includes_history(client):
    chat(client, "s-a", "hello")
    chat(client, "s-a", "tell me more")
    r = chat(client, "s-b", "tell me more").json()  # same text, different history
    assert r["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_cache_key_includes_system_prompt(client):
    chat(client, "s-a", "hi there")
    r = chat(client, "s-b", "hi there", system_prompt="Talk like a pirate").json()
    assert r["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_cache_disabled(make_client):
    with make_client(cache_enabled=False) as c:
        chat(c, "s-a", "same")
        r = chat(c, "s-b", "same").json()
        assert r["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert c.get("/health").json()["cache_enabled"] is False


# ------------------------------------------------------------------ tokens
def test_tokens_are_per_request_not_cumulative(client):
    a = chat(client, "s-a", "alpha bravo").json()["metadata"]
    b = chat(client, "s-b", "delta golf").json()["metadata"]  # equal length, new session
    assert a["input_tokens"] == b["input_tokens"]
    second = chat(client, "s-a", "charlie").json()["metadata"]
    # second turn includes history, so input grows, but output is just this reply
    assert second["input_tokens"] > a["input_tokens"]
    assert second["output_tokens"] < a["output_tokens"] + second["output_tokens"]


# ------------------------------------------------------------------ multi-turn + history
def test_multi_turn_and_session_isolation(client, provider):
    chat(client, "s-x", "my name is Ada")
    r = chat(client, "s-x", "what is my name?").json()
    assert "turns=2" in r["reply"] and "first=my name is Ada" in r["reply"]
    other = chat(client, "s-y", "what is my name?").json()
    assert "turns=1" in other["reply"]


def test_history_endpoint(client):
    chat(client, "s-h", "first question")
    chat(client, "s-h", "second question", system_prompt="be brief")
    r = client.get("/sessions/s-h/messages")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == "s-h"
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert body["messages"][0]["content"] == "first question"
    assert "metadata" not in body["messages"][0]
    meta = body["messages"][1]["metadata"]
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert "input_tokens" in meta and "output_tokens" in meta
    assert all(m["role"] != "system" for m in body["messages"])


def test_history_unknown_session_404(client):
    r = client.get("/sessions/nope/messages")
    assert r.status_code == 404
    assert r.json() == {"detail": "session not found"}


def test_system_prompt_persists_for_session(client, provider):
    chat(client, "s-p", "hi", system_prompt="You are a pirate")
    chat(client, "s-p", "again")
    assert provider.calls[-1]["messages"][0] == {"role": "system", "content": "You are a pirate"}


def test_history_survives_restart(make_client):
    with make_client() as c:
        chat(c, "s-persist", "remember me")
    with make_client() as c2:  # new app, same DB file
        r = c2.get("/sessions/s-persist/messages")
        assert r.status_code == 200
        assert r.json()["messages"][0]["content"] == "remember me"


# ------------------------------------------------------------------ failover (simulated)
@pytest.mark.parametrize("header", ["key_1=429", "key_1=500", "key_1=502", "key_1=503"])
def test_simulated_failover_to_key_2(client, header, provider):
    r = chat(client, "s-f", f"failover {header}", headers={"X-Simulate-Failure": header})
    assert r.status_code == 200
    meta = r.json()["metadata"]
    assert meta["key_id"] == "key_2"
    assert meta["fallback_attempts"] == 1
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert not r.json()["reply"].lower().startswith("simulated")
    assert [c["api_key"] for c in provider.calls] == ["sk-two"]  # key_1 never really called


def test_simulate_all_keys_503_and_nothing_stored(client):
    r = chat(client, "s-dead", "hello", headers={"X-Simulate-Failure": "*=429"})
    assert r.status_code == 503
    assert r.json() == {"detail": "all providers unavailable"}
    assert client.get("/sessions/s-dead/messages").status_code == 404


def test_simulate_all_keys_503_even_if_cached(client):
    chat(client, "s-a", "cached question")
    r = chat(client, "s-b", "cached question", headers={"X-Simulate-Failure": "*=500"})
    assert r.status_code == 503


def test_failed_turn_not_added_to_existing_history(client):
    chat(client, "s-keep", "one")
    chat(client, "s-keep", "two", headers={"X-Simulate-Failure": "key_1=429,key_2=500"})
    msgs = client.get("/sessions/s-keep/messages").json()["messages"]
    assert len(msgs) == 2


def test_simulation_ignored_when_disabled(make_client):
    with make_client(enable_failure_simulation=False) as c:
        r = chat(c, "s", "hi", headers={"X-Simulate-Failure": "*=429"})
        assert r.status_code == 200
        assert r.json()["metadata"]["key_id"] == "key_1"


# ------------------------------------------------------------------ failover (real errors)
def test_real_rate_limit_fails_over(client, provider):
    provider.broken["sk-one"] = litellm.RateLimitError("quota", llm_provider="gemini", model="m")
    r = chat(client, "s-real", "hello").json()
    assert r["metadata"]["key_id"] == "key_2"
    assert r["metadata"]["fallback_attempts"] == 1


def test_real_errors_on_all_keys_503(client, provider):
    provider.broken["sk-one"] = litellm.InternalServerError("boom", llm_provider="gemini", model="m")
    provider.broken["sk-two"] = litellm.ServiceUnavailableError("down", llm_provider="gemini", model="m")
    r = chat(client, "s-real", "hello")
    assert r.status_code == 503
    assert r.json() == {"detail": "all providers unavailable"}


def test_key_never_exposed(client, provider):
    provider.broken["sk-one"] = litellm.RateLimitError("quota", llm_provider="gemini", model="m")
    text = chat(client, "s-sec", "hello").text + client.get("/sessions/s-sec/messages").text
    assert "sk-one" not in text and "sk-two" not in text


def test_per_key_models(make_client, provider):
    with make_client(key_models=["gemini/gemini-test", "groq/llama-test"]) as c:
        r = chat(c, "s-m", "hi", headers={"X-Simulate-Failure": "key_1=429"}).json()
        assert r["metadata"]["model"] == "groq/llama-test"


# ------------------------------------------------------------------ helpers
def test_parse_simulation_header():
    ids = ["key_1", "key_2"]
    assert parse_simulation_header("key_1=429", ids) == {"key_1": 429}
    assert parse_simulation_header("key_1=429,key_2=500", ids) == {"key_1": 429, "key_2": 500}
    assert parse_simulation_header("*=503", ids) == {"key_1": 503, "key_2": 503}
    assert parse_simulation_header("key_1=418,key_9=429,garbage", ids) == {}
    assert parse_simulation_header(None, ids) == {}


def test_unknown_route_is_json(client):
    r = client.get("/nope")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")
