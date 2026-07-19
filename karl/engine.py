"""The engine — how an agent reaches the model, and why KARL feels fast.

One interface, ``chat(messages, tools, on_token) -> ChatResult``, three engines:

  HTTPEngine    any OpenAI-compatible ``/chat/completions`` endpoint (vLLM,
                llama.cpp, Ollama, a hosted API). Stdlib urllib only. It
                *streams*: tokens are handed to ``on_token`` the moment they
                come off the wire, so the crew speaks live instead of freezing
                for a whole completion. Servers that can't stream fall back to
                a plain request behind a short retry ladder — never a crash.
  MockEngine    the offline stand-in: one seeded in-character line per turn, so
                the crew visibly moves with no model attached.
  ScriptEngine  a scripted engine for tests, to drive the tool loop.

Small local models sometimes drop into a degenerate repeat loop; ``cut_loops``
trims that at the source before it poisons the transcript everyone reads next.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass, field

from karl.config import endpoint

_LOOP_MARK = " …(repeat loop trimmed)"
_SENT = re.compile(r"(?<=[.!?…])\s+")


def cut_loops(text: str | None) -> str:
    """Keep a runaway repetition once. Two shapes: the same sentence (24+ chars)
    said three or more times, and the punctuation-free twin — a 12–80 char unit
    tiling the tail. Ordinary text and honest repetition pass untouched."""
    if not text:
        return text or ""
    counts: dict = {}
    for part in _SENT.split(text):
        norm = " ".join(part.lower().split())
        if len(norm) < 24:
            continue
        counts[norm] = counts.get(norm, 0) + 1
        if counts[norm] == 3:
            first = text.find(part)
            if first >= 0:
                return text[: first + len(part)].rstrip() + _LOOP_MARK
    body = text.rstrip()
    n = len(body)
    for u in range(12, 81):
        if n < 3 * u:
            break
        unit = body[n - u:]
        if body[n - 3 * u:] == unit * 3:
            start = n
            while start - u >= 0 and body[start - u:start] == unit:
                start -= u
            return body[:start + u].rstrip() + _LOOP_MARK
    return text


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON string, as the wire delivers it


@dataclass
class ChatResult:
    content: str | None = None
    tool_calls: list = field(default_factory=list)


class Engine:
    def chat(self, messages: list, tools: list | None = None,
             on_token=None) -> ChatResult:
        raise NotImplementedError


# --------------------------------------------------------------------------
# SSE parsing, kept free of the socket so it can be tested cold
# --------------------------------------------------------------------------
def parse_sse(lines, on_token=None) -> ChatResult:
    """Fold a stream of ``data: {json}`` lines into one ChatResult, handing each
    content piece to ``on_token`` as it arrives. ``lines`` is any iterable of
    bytes or str. Tool-call deltas arrive sliced across chunks keyed by index;
    they are reassembled here."""
    content: list = []
    calls: dict = {}
    for raw in lines:
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            delta = json.loads(payload)["choices"][0].get("delta") or {}
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            continue
        piece = delta.get("content")
        if piece:
            content.append(piece)
            if on_token:
                on_token(piece)
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            slot = calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["name"] += fn["name"]
            if fn.get("arguments"):
                slot["arguments"] += fn["arguments"]
    out = [ToolCall(c["id"] or f"c{i}", c["name"], c["arguments"] or "{}")
           for i, c in sorted(calls.items()) if c["name"]]
    text = "".join(content)
    return ChatResult(content=cut_loops(text) if text else None, tool_calls=out)


# --------------------------------------------------------------------------
class HTTPEngine(Engine):
    RETRY_DELAYS = (1, 3, 8)
    STALL_S = 15   # a stream that died after this long was a stall, not "can't stream"

    def __init__(self, cfg: dict):
        self.base_url = cfg["base_url"].rstrip("/")
        self.model = cfg.get("model", "local")
        self.api_key = cfg.get("api_key", "-")
        self.temperature = cfg.get("temperature", 0.6)
        self.max_tokens = int(cfg.get("max_tokens", 2048))
        self.timeout = float(cfg.get("timeout", 300))
        self.stream = bool(cfg.get("stream", True))

    def _open(self, body: dict):
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=data,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        return urllib.request.urlopen(req, timeout=self.timeout)

    def _body(self, messages: list, tools: list | None) -> dict:
        body = {"model": self.model, "messages": messages,
                "temperature": self.temperature, "max_tokens": self.max_tokens}
        if tools:
            body["tools"] = tools
        return body

    def chat(self, messages: list, tools: list | None = None,
             on_token=None) -> ChatResult:
        if self.stream and on_token is not None:
            t0 = time.time()
            try:
                with self._open({**self._body(messages, tools),
                                 "stream": True}) as resp:
                    out = parse_sse(resp, on_token)
                if out.content or out.tool_calls:
                    return out
                # nothing SSE-shaped came back — a server that ignores
                # stream=true and answers plain JSON lands here; ask plainly
            except Exception as e:  # noqa: BLE001 — sort stall from can't-stream
                waited = time.time() - t0
                if waited > self.STALL_S:
                    # The server took the request and produced nothing for a
                    # long time. Falling back would multiply the wait by the
                    # whole retry ladder and look like a freeze — report the
                    # stall honestly instead.
                    return ChatResult(content=(
                        f"(the model stalled — no output for {waited:.0f}s, then "
                        f"{type(e).__name__}). The server is overloaded, still "
                        "loading, or running the model partly on CPU (VRAM too "
                        "small — check `~/karl.log` on the box for 'offloaded "
                        "X/Y layers'). Try a smaller model (`karl gpu model "
                        "<key>`), or raise `karl config --timeout`."))
                # failed fast → the server likely can't stream; ask plainly
        return self._complete(messages, tools)

    def _complete(self, messages: list, tools: list | None) -> ChatResult:
        body = self._body(messages, tools)
        last = None
        for delay in (0,) + self.RETRY_DELAYS:
            if delay:
                time.sleep(delay)
            try:
                with self._open(body) as resp:
                    payload = json.loads(resp.read().decode("utf-8", "replace"))
                msg = payload["choices"][0]["message"]
                calls = [ToolCall(tc.get("id", f"c{i}"),
                                  tc["function"]["name"],
                                  tc["function"].get("arguments") or "{}")
                         for i, tc in enumerate(msg.get("tool_calls") or [])]
                return ChatResult(content=cut_loops(msg.get("content")),
                                  tool_calls=calls)
            except Exception as e:  # noqa: BLE001 — a flaky server must not crash a run
                last = e
        return ChatResult(content=f"(the model endpoint didn't respond — "
                                  f"{type(last).__name__}: {str(last)[:120]}). "
                                  "The server or its tunnel may be down: try "
                                  "`karl ping`, and if it's a GPU box, `karl gpu "
                                  "status` then `karl gpu reconnect`.")


# --------------------------------------------------------------------------
class ScriptEngine(Engine):
    """A scripted engine for tests. Each item is one reply:
    ``{'text': ...}`` or ``{'tool': name, 'args': {...}}`` or
    ``{'tools': [{'tool':.., 'args':..}, ...], 'say': ...}``."""

    def __init__(self, script: list):
        self.script = list(script)
        self._n = 0

    def chat(self, messages: list, tools: list | None = None,
             on_token=None) -> ChatResult:
        if not self.script:
            return ChatResult(content="(script exhausted)")
        item = self.script.pop(0)
        if "text" in item:
            if on_token and item["text"]:
                on_token(item["text"])
            return ChatResult(content=item["text"])
        calls = item.get("tools") or [{"tool": item["tool"], "args": item.get("args", {})}]
        out = []
        for c in calls:
            self._n += 1
            out.append(ToolCall(f"s{self._n}", c["tool"], json.dumps(c.get("args", {}))))
        if on_token and item.get("say"):
            on_token(item["say"])
        return ChatResult(content=item.get("say"), tool_calls=out)


# --------------------------------------------------------------------------
class MockEngine(Engine):
    """The offline stand-in: one seeded in-character line, no tool calls, so the
    crew visibly moves with no model attached. The session seeds each line."""

    def __init__(self):
        self._pending = None

    def seed(self, text: str) -> None:
        self._pending = text

    def chat(self, messages: list, tools: list | None = None,
             on_token=None) -> ChatResult:
        if self._pending is not None:
            text, self._pending = self._pending, None
        else:
            tail = messages[-1]["content"] if messages else ""
            text = f"(offline: heard '{str(tail)[-80:]}')"
        if on_token and text:
            on_token(text)
        return ChatResult(content=text)


def make_engine():
    """The real model is the default — never theater. Returns (engine, mode):

      'model'    an endpoint is configured → HTTPEngine
      'offline'  KARL_OFFLINE=1 → the canned MockEngine, explicitly asked for
      'none'     nothing attached → (None, 'none'); the session refuses to run
                 a fake round and says how to attach a model instead
    """
    import os
    cfg = endpoint()
    if cfg["base_url"]:
        return HTTPEngine(cfg), "model"
    if os.environ.get("KARL_OFFLINE"):
        return MockEngine(), "offline"
    return None, "none"
