import { useEffect, useMemo, useRef, useState } from "react";
import { API_URL, fetchHealth, newSessionId, sendChat } from "./api.js";

const SIMULATION_OPTIONS = [
  { value: "", label: "Off (normal)" },
  { value: "key_1=429", label: "key_1 → 429 rate limit" },
  { value: "key_1=500", label: "key_1 → 500 server error" },
  { value: "key_1=503", label: "key_1 → 503 unavailable" },
  { value: "*=429", label: "all keys → 429 (expect 503)" },
];

function AssistantMessage({ msg }) {
  const m = msg.metadata;
  const cached = m.resolution_source === "LLM_CACHE";
  return (
    <div className="msg assistant" data-testid="assistant-message">
      <div className="bubble">{msg.content}</div>
      <div className="meta">
        <span className={`chip source ${cached ? "cache" : "live"}`} title="Resolution source">
          <span data-testid="meta-source">{m.resolution_source}</span>
        </span>
        <span className="chip" title="Model">
          <span data-testid="meta-model">{m.model}</span>
        </span>
        <span className="chip" title="Server-measured latency">
          ⏱ <span data-testid="meta-latency">{m.latency_ms} ms</span>
        </span>
        <span className="chip" title="Input (prompt) tokens for this request">
          in <span data-testid="meta-input-tokens">{m.input_tokens}</span>
        </span>
        <span className="chip" title="Output (completion) tokens for this reply">
          out <span data-testid="meta-output-tokens">{m.output_tokens}</span>
        </span>
        <span className="chip subtle" title="Key that answered (null on cache hit)">
          key: {m.key_id ?? "—"}
        </span>
        {m.fallback_attempts > 0 && (
          <span className="chip warn" title="Keys that failed before this one succeeded">
            ↪ failover ×{m.fallback_attempts}
          </span>
        )}
      </div>
    </div>
  );
}

export default function App() {
  // A new session starts on each page load.
  const [sessionId, setSessionId] = useState(() => newSessionId());
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [simulate, setSimulate] = useState("");
  const [pending, setPending] = useState(false);
  const [health, setHealth] = useState(null);
  const endRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth({ ok: false, body: null }));
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, pending]);

  const totals = useMemo(() => {
    const a = messages.filter((m) => m.role === "assistant");
    return {
      replies: a.length,
      cacheHits: a.filter((m) => m.metadata.resolution_source === "LLM_CACHE").length,
    };
  }, [messages]);

  async function send() {
    const text = input.trim();
    if (!text || pending) return;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setPending(true);
    try {
      const data = await sendChat({
        sessionId,
        message: text,
        systemPrompt,
        simulateFailure: simulate,
      });
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.reply, metadata: data.metadata },
      ]);
    } catch (err) {
      const prefix = err.status ? `Error ${err.status}: ` : "";
      setMessages((prev) => [...prev, { role: "error", content: prefix + err.message }]);
    } finally {
      setPending(false);
      inputRef.current?.focus();
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  function resetSession() {
    setSessionId(newSessionId());
    setMessages([]);
  }

  const healthy = health?.ok;

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>
          Resilient <span>LLM</span> Chat
        </h1>
        <p className="tagline">Key failover · response caching · per-request tokens</p>

        <section className="panel">
          <div className="row">
            <span className={`dot ${healthy ? "ok" : health ? "bad" : ""}`} />
            <strong>
              {health ? (healthy ? "Backend online" : "Backend unavailable") : "Checking backend…"}
            </strong>
          </div>
          {health?.body && (
            <div className="small">
              {health.body.keys_configured} key(s) in pool · cache{" "}
              {health.body.cache_enabled ? "on" : "off"}
            </div>
          )}
          <div className="small mono wrap">{API_URL}</div>
        </section>

        <section className="panel">
          <label htmlFor="sp">System prompt (optional)</label>
          <textarea
            id="sp"
            data-testid="system-prompt-input"
            rows={4}
            placeholder="e.g. You are a concise travel guide."
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
          />
          <div className="small">Sent with each message; applies to this session.</div>
        </section>

        <section className="panel">
          <label htmlFor="sim">Failover demo (X-Simulate-Failure)</label>
          <select id="sim" value={simulate} onChange={(e) => setSimulate(e.target.value)}>
            {SIMULATION_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
          <div className="small">Only honored when the backend enables failure simulation.</div>
        </section>

        <section className="panel">
          <div className="small mono">session: {sessionId}</div>
          <div className="small">
            {totals.replies} replies · {totals.cacheHits} cache hit(s)
          </div>
          <button className="ghost" onClick={resetSession}>
            New session
          </button>
        </section>
      </aside>

      <main className="chat">
        <div className="messages">
          {messages.length === 0 && (
            <div className="empty">
              <h2>Ask anything</h2>
              <p>
                Tip: send the same first message in a new session to see an{" "}
                <code>LLM_CACHE</code> hit, or pick a failover option on the left.
              </p>
            </div>
          )}
          {messages.map((m, i) =>
            m.role === "assistant" ? (
              <AssistantMessage key={i} msg={m} />
            ) : (
              <div key={i} className={`msg ${m.role}`} role={m.role === "error" ? "alert" : undefined}>
                <div className="bubble">{m.content}</div>
              </div>
            )
          )}
          {pending && (
            <div className="msg assistant pending">
              <div className="bubble typing">
                <span />
                <span />
                <span />
              </div>
            </div>
          )}
          <div ref={endRef} />
        </div>

        <div className="composer">
          <textarea
            ref={inputRef}
            data-testid="chat-input"
            rows={1}
            placeholder="Type a message… (Enter to send, Shift+Enter for a new line)"
            value={input}
            maxLength={4000}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            autoFocus
          />
          <button data-testid="send-button" onClick={send} disabled={pending || !input.trim()}>
            {pending ? "…" : "Send"}
          </button>
        </div>
      </main>
    </div>
  );
}
