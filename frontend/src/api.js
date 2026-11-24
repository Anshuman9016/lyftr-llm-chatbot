const runtimeUrl = window.__APP_CONFIG__?.API_URL;
export const API_URL = (
  runtimeUrl || import.meta.env.VITE_API_URL || "http://localhost:8000"
).replace(/\/+$/, "");

export function newSessionId() {
  const rand =
    globalThis.crypto?.randomUUID?.().replace(/-/g, "").slice(0, 12) ||
    Math.random().toString(36).slice(2, 14);
  return `s-${rand}`;
}

function describeError(status, body) {
  const detail = body?.detail;
  if (Array.isArray(detail)) {
    // FastAPI 422 shape: [{loc, msg, ...}]
    return detail
      .map((d) => {
        const field = Array.isArray(d.loc) ? d.loc.filter((p) => p !== "body").join(".") : "";
        return field ? `${field}: ${d.msg}` : d.msg;
      })
      .join("; ");
  }
  if (typeof detail === "string") return detail;
  return `Request failed with HTTP ${status}`;
}

export async function sendChat({ sessionId, message, systemPrompt, simulateFailure }) {
  const headers = { "Content-Type": "application/json" };
  if (simulateFailure) headers["X-Simulate-Failure"] = simulateFailure;
  const payload = { session_id: sessionId, message };
  if (systemPrompt && systemPrompt.trim()) payload.system_prompt = systemPrompt;

  let res;
  try {
    res = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });
  } catch {
    throw new Error(
      "Could not reach the backend. It may be waking up (free hosting sleeps when idle) - try again in a few seconds."
    );
  }
  let body = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON body */
  }
  if (!res.ok) {
    const err = new Error(describeError(res.status, body));
    err.status = res.status;
    throw err;
  }
  return body;
}

export async function fetchHealth() {
  const res = await fetch(`${API_URL}/health`);
  return { ok: res.ok, body: await res.json().catch(() => null) };
}
