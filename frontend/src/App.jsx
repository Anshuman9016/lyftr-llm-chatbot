import { useEffect, useMemo, useRef, useState } from "react";
import { API_URL, fetchHealth, newSessionId, sendChat } from "./api.js";
import Aurora from "./components/reactbits/Aurora.jsx";
import ClickSpark from "./components/reactbits/ClickSpark.jsx";
import GradientText from "./components/reactbits/GradientText.jsx";
import ShinyText from "./components/reactbits/ShinyText.jsx";
import SpotlightCard from "./components/reactbits/SpotlightCard.jsx";

const APP_NAME = "Relay";

const SIMULATION_OPTIONS = [
  { value: "", label: "Off (normal)" },
  { value: "key_1=429", label: "key_1 → 429 rate limit" },
  { value: "key_1=500", label: "key_1 → 500 server error" },
  { value: "key_1=503", label: "key_1 → 503 unavailable" },
  { value: "*=429", label: "all keys → 429 (expect 503)" },
];

const SUGGESTIONS = [
  "What is the capital of France?",
  "My name is Anshuman. Remember it.",
  "Explain caching in one sentence.",
];

const BRAND = ["#a78bfa", "#22d3ee", "#f472b6", "#facc15", "#a78bfa"];

const ACCENTS = {
  green: "rgba(52, 211, 153, 0.45)",
  cyan: "rgba(34, 211, 238, 0.45)",
  pink: "rgba(244, 114, 182, 0.5)",
  amber: "rgba(250, 204, 21, 0.45)",
};

function Panel({ children, accent = "cyan" }) {
  return (
    <SpotlightCard
      className={`panel panel--${accent}`}
      spotlightColor={ACCENTS[accent]}
      intensity={0.35}
      spotlightSize={240}
    >
      {children}
    </SpotlightCard>
  );
}

function AssistantMessage({ msg }) {
  const m = msg.metadata;
  const cached = m.resolution_source === "LLM_CACHE";
  return (
    <div className="msg assistant" data-testid="assistant-message">
      <div className={`bubble ${cached ? "cached" : ""}`}>{msg.content}</div>
      <div className="meta">
        <span className={`chip source ${cached ? "cache" : "live"}`} title="Resolution source">
          {cached ? "⚡ " : "✦ "}
          <span data-testid="meta-source">{m.resolution_source}</span>
        </span>
        <span className="chip model" title="Model">
          <span data-testid="meta-model">{m.model}</span>
        </span>
        <span className="chip latency" title="Server-measured latency">
          ⏱ <span data-testid="meta-latency">{m.latency_ms} ms</span>
        </span>
        <span className="chip tin" title="Input (prompt) tokens for this request">
          in <span data-testid="meta-input-tokens">{m.input_tokens}</span>
        </span>
        <span className="chip tout" title="Output (completion) tokens for this reply">
          out <span data-testid="meta-output-tokens">{m.output_tokens}</span>
        </span>
        <span className="chip key" title="Key that answered (null on cache hit)">
          🔑 {m.key_id ?? "—"}
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
    document.title = `${APP_NAME} · resilient LLM chat`;
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
      failovers: a.filter((m) => m.metadata.fallback_attempts > 0).length,
    };
  }, [messages]);

  async function send(textOverride) {
    const text = (textOverride ?? input).trim();
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
    <>
      <div className="bg" aria-hidden="true">
        <Aurora
          colorStops={["#8b5cf6", "#22d3ee", "#f472b6"]}
          amplitude={1.3}
          blend={0.6}
          speed={0.7}
          height={0.75}
          layers={2}
          glow={0.9}
          brightness={1.25}
          opacity={1}
          mouseInteraction={false}
        />
      </div>

      <ClickSpark sparkColor="#f0abfc" sparkSize={11} sparkRadius={22} sparkCount={10} duration={450}>
        <div className="layout">
          <aside className="sidebar">
            <div className="brand">
              <div className="logo" aria-hidden="true">
                <span />
                <span />
                <span />
              </div>
              <div>
                <h1>
                  <GradientText colors={BRAND} animationSpeed={6}>
                    {APP_NAME}
                  </GradientText>
                </h1>
                <ShinyText
                  className="tagline"
                  text="never drops the baton"
                  color="#c4b5fd"
                  shineColor="#ffffff"
                  speed={3}
                />
              </div>
            </div>

            <Panel accent="green">
              <div className="row">
                <span className={`dot ${healthy ? "ok" : health ? "bad" : ""}`} />
                <strong>
                  {health
                    ? healthy
                      ? "Backend online"
                      : "Backend unavailable"
                    : "Waking backend…"}
                </strong>
              </div>
              {health?.body && (
                <div className="pills">
                  <span className="pill violet">{health.body.keys_configured} keys in pool</span>
                  <span className={`pill ${health.body.cache_enabled ? "amber" : ""}`}>
                    cache {health.body.cache_enabled ? "on" : "off"}
                  </span>
                </div>
              )}
              <div className="small mono wrap">{API_URL}</div>
            </Panel>

            <Panel accent="cyan">
              <label htmlFor="sp">System prompt</label>
              <textarea
                id="sp"
                data-testid="system-prompt-input"
                rows={3}
                placeholder="e.g. You are a concise travel guide."
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
              />
              <div className="small">Optional · applies to this session.</div>
            </Panel>

            <Panel accent="pink">
              <label htmlFor="sim">Failover demo</label>
              <select id="sim" value={simulate} onChange={(e) => setSimulate(e.target.value)}>
                {SIMULATION_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
              <div className="small">
                Sends <code>X-Simulate-Failure</code> with each message.
              </div>
            </Panel>

            <Panel accent="amber">
              <div className="stats">
                <div>
                  <b>{totals.replies}</b>
                  <span>replies</span>
                </div>
                <div>
                  <b className="amber-t">{totals.cacheHits}</b>
                  <span>cache hits</span>
                </div>
                <div>
                  <b className="pink-t">{totals.failovers}</b>
                  <span>failovers</span>
                </div>
              </div>
              <div className="small mono">session {sessionId}</div>
              <button className="ghost" onClick={resetSession}>
                ✨ New session
              </button>
            </Panel>
          </aside>

          <main className="chat">
            <div className="messages">
              {messages.length === 0 && (
                <div className="empty">
                  <h2>
                    <GradientText colors={BRAND} animationSpeed={5}>
                      Ask anything.
                    </GradientText>
                  </h2>
                  <p>
                    Every reply shows where it came from, which key answered, latency and tokens.
                    Repeat a first message in a new session to see an <b className="amber-t">LLM_CACHE</b>{" "}
                    hit, or break a key with the failover demo.
                  </p>
                  <div className="suggestions">
                    {SUGGESTIONS.map((s) => (
                      <button key={s} className="suggestion" onClick={() => send(s)}>
                        {s}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {messages.map((m, i) =>
                m.role === "assistant" ? (
                  <AssistantMessage key={i} msg={m} />
                ) : (
                  <div
                    key={i}
                    className={`msg ${m.role}`}
                    role={m.role === "error" ? "alert" : undefined}
                  >
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
                placeholder="Message Relay…  (Enter to send · Shift+Enter for a new line)"
                value={input}
                maxLength={4000}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={onKeyDown}
                autoFocus
              />
              <button
                data-testid="send-button"
                className="send"
                onClick={() => send()}
                disabled={pending || !input.trim()}
              >
                {pending ? "…" : "Send ➤"}
              </button>
            </div>
          </main>
        </div>
      </ClickSpark>
    </>
  );
}
