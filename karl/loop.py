"""The think→act loop — one agent's turn, for real.

Given an agent's system prompt, task, and tools, run the model until it ends
its turn. Two ways a turn ends, in order of preference:

  * the ``handoff`` tool — structured control flow: the agent names who gets
    the floor (a teammate or the operator) and says its line, as arguments,
    not prose. No regex, no ambiguity. The loop injects this tool itself.
  * a plain-text line — the fallback for models that won't call tools; the
    session then routes it by the last name the line mentions.

Turns are long enough for real work (default 40 steps, configurable) and the
message list is kept in bounds by *compaction*: old tool output shrinks to a
stub once the turn's context grows past budget — the agent keeps its recent
evidence verbatim and its old evidence in summary, instead of drowning or
being cut off.

Guards, kept from every honest agent loop and hardened by field fire:
the reflect reflex (acting repeatedly without a word of reasoning → pushed to
think), the stuck reflex (the exact same tool call over and over → pushed,
then stopped), the empty-answer nudge, and a budget warning near the end so
the agent consolidates instead of being cut off mid-reach.

The offline MockEngine runs the same loop; it seeds one line and never calls
a tool, so the machinery stays exercised with no model attached.
"""

from __future__ import annotations

import json

from karl.engine import MockEngine
from karl.tools import execute

_REFLECT_AFTER = 3   # act-only steps before we push to think
_MAX_STEPS = 40

_CONTEXT_BUDGET = 28_000   # chars of message content before old evidence shrinks
_KEEP_RECENT = 8           # last N messages are never compacted
_COMPACT_STUB = 240        # what an old tool result shrinks to

_REFLECT_NUDGE = (
    "You have acted several times without reasoning out loud. Stop. In plain "
    "English, say what you have found so far and what it means, then decide your "
    "next move. Do not call another tool until you have thought.")

_BUDGET_NUDGE = (
    "Two steps remain. Stop exploring; consolidate what you have and end your "
    "turn with the handoff tool.")

_EMPTY_NUDGE = (
    "You said nothing and handed off to no one. End your turn now: call the "
    "handoff tool with your line and who should speak next.")

_STUCK_NUDGE = (
    "You have made the exact same tool call several times — the result will not "
    "change. Do not call it again. Act on what you already know, or end your "
    "turn with the handoff tool.")


def handoff_spec(targets: list) -> dict:
    """The turn-ending tool, injected by the loop itself. ``targets`` are the
    names the agent may hand to (teammates + 'operator')."""
    return {"type": "function", "function": {
        "name": "handoff",
        "description": "End your turn. Hand the floor to a teammate or the "
                       "operator, with your message. This is how a turn ends — "
                       "prefer it over just naming someone in prose.",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string", "enum": targets,
                   "description": "who speaks next"},
            "message": {"type": "string",
                        "description": "your plain-English line to them"}},
            "required": ["to", "message"]}}}


def _compact(messages: list) -> None:
    """Shrink old tool results once the turn outgrows its context budget. The
    newest _KEEP_RECENT messages and everything that isn't tool output stay
    verbatim; older tool observations become stubs. Idempotent and in-place."""
    total = sum(len(str(m.get("content") or "")) for m in messages)
    if total <= _CONTEXT_BUDGET:
        return
    for m in messages[:-_KEEP_RECENT]:
        if total <= _CONTEXT_BUDGET:
            return
        content = str(m.get("content") or "")
        if m.get("role") == "tool" and len(content) > _COMPACT_STUB:
            m["content"] = (content[:_COMPACT_STUB]
                            + " …[old result compacted — re-run the tool if needed]")
            total -= len(content) - len(str(m["content"]))


def think_and_act(engine, *, system: str, user: str, tools: list, ctx,
                  seed: str | None = None, on_token=None,
                  on_tool=lambda *_: None, on_tool_start=lambda *_: None,
                  handoff_to: list | None = None,
                  max_steps: int = _MAX_STEPS):
    """Run one agent turn. Returns (spoken_line, tainted, to) — ``to`` is the
    structured handoff target, or None when the agent ended in prose and the
    session must route by name-mention."""
    if isinstance(engine, MockEngine) and seed is not None:
        engine.seed(seed)

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    openai_tools = [t.openai() for t in tools]
    if handoff_to:
        openai_tools.append(handoff_spec(list(handoff_to)))
    openai_tools = openai_tools or None
    act_streak = 0
    last_text = ""
    warned = False
    empty_nudged = False
    budget = max_steps
    last_calls = None    # the previous step's exact tool calls
    same_calls = 0       # …and how often they've repeated verbatim

    def _handoff_in(calls):
        for c in calls:
            if c.name == "handoff":
                try:
                    args = json.loads(c.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                to = str(args.get("to") or "").strip().lower()
                msg = str(args.get("message") or "").strip()
                if handoff_to and to in handoff_to:
                    return to, msg
                return None, msg   # bad target → keep the words, route by prose
        return None, None

    step = 0
    while step < budget:
        if not warned and step == budget - 2 and budget > 2:
            messages.append({"role": "user", "content": _BUDGET_NUDGE})
            warned = True
        _compact(messages)

        res = engine.chat(messages, openai_tools, on_token=on_token)

        if res.tool_calls:
            to, msg = _handoff_in(res.tool_calls)
            if msg is not None:
                line = (msg or res.content or last_text or "(said nothing)").strip()
                return line, bool(ctx.tainted), to

        if not res.tool_calls:
            if (not (res.content or "").strip() and not empty_nudged
                    and step < budget - 1):
                empty_nudged = True
                messages.append({"role": "user", "content": _EMPTY_NUDGE})
                step += 1
                continue
            line = (res.content or last_text or "(said nothing)").strip()
            return line, bool(ctx.tainted), None

        # it's acting — record the assistant turn (with its calls) verbatim
        if res.content:
            last_text = res.content
        messages.append({"role": "assistant", "content": res.content or "",
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.name,
                                                      "arguments": c.arguments}}
                                        for c in res.tool_calls]})
        for c in res.tool_calls:
            on_tool_start(c.name)
            obs = execute(tools, c, ctx)
            on_tool(c.name, obs.splitlines()[0][:80] if obs else "")
            messages.append({"role": "tool", "tool_call_id": c.id, "content": obs})

        # the stuck reflex: rewording the same sentence while making the same
        # tool call is still a loop — the reflect nudge below never sees it
        # because every step "says something". Same exact calls three times →
        # push once; a repeat after the push → stop the turn.
        calls = tuple((c.name, c.arguments) for c in res.tool_calls)
        same_calls = same_calls + 1 if calls == last_calls else 0
        last_calls = calls
        if same_calls == 2:
            messages.append({"role": "user", "content": _STUCK_NUDGE})
        elif same_calls >= 3:
            break

        act_streak = act_streak + 1 if not (res.content or "").strip() else 0
        if act_streak >= _REFLECT_AFTER:
            messages.append({"role": "user", "content": _REFLECT_NUDGE})
            act_streak = 0
        step += 1

    messages.append({"role": "user",
                     "content": "Enough acting. Say your line now, in plain English."})
    res = engine.chat(messages, None, on_token=on_token)
    line = (res.content or last_text or "(ran out of steps)").strip()
    return line, bool(ctx.tainted), None
