"""The think→act loop and its guards."""

from __future__ import annotations

from karl.engine import MockEngine, ScriptEngine
from karl.loop import _REFLECT_NUDGE, think_and_act
from karl.tools import ToolContext, build_tools


def _ctx(project, can_egress=False):
    ws = project.workspace
    ws.mkdir(parents=True, exist_ok=True)
    return ToolContext(workspace=ws, project=project, can_egress=can_egress)


def test_loop_executes_a_tool_and_grounds_the_answer(project):
    ctx = _ctx(project)
    (ctx.workspace / "note.txt").write_text("the east ford is out")
    engine = ScriptEngine([
        {"tool": "read_file", "args": {"path": "note.txt"}},
        {"text": "The east ford is out; route north."},
    ])
    seen = []
    line, tainted, _ = think_and_act(
        engine, system="s", user="read note.txt",
        tools=build_tools(["read_file"], ctx), ctx=ctx,
        on_tool=lambda name, first: seen.append(name))
    assert "north" in line
    assert tainted is False
    assert seen == ["read_file"]


def test_offline_mock_runs_the_loop_and_speaks(project):
    ctx = _ctx(project)
    line, _, _ = think_and_act(
        MockEngine(), system="s", user="u", tools=build_tools(["read_file"], ctx),
        ctx=ctx, seed="I am ready.")
    assert line == "I am ready."


def test_tokens_stream_out_through_on_token(project):
    ctx = _ctx(project)
    got = []
    line, _, _ = think_and_act(
        MockEngine(), system="s", user="u", tools=[], ctx=ctx,
        seed="Live words.", on_token=got.append)
    assert line == "Live words."
    assert got == ["Live words."]


class _Recorder(ScriptEngine):
    """A ScriptEngine that keeps the message list it was last shown."""

    def __init__(self, script):
        super().__init__(script)
        self.seen = []

    def chat(self, messages, tools=None, on_token=None):
        self.seen = list(messages)
        return super().chat(messages, tools, on_token=on_token)


def test_reflect_reflex_pushes_to_think(project):
    ctx = _ctx(project)
    (ctx.workspace / "f").write_text("x")
    engine = _Recorder([
        {"tool": "read_file", "args": {"path": "f", "offset": 0}},  # silent act 1
        {"tool": "read_file", "args": {"path": "f", "offset": 1}},  # silent act 2
        {"tool": "read_file", "args": {"path": "f", "offset": 2}},  # act 3 → nudge
        {"text": "I looked thrice; it says x."},
    ])
    line, _, _ = think_and_act(engine, system="s", user="u",
                            tools=build_tools(["read_file"], ctx), ctx=ctx)
    assert "x" in line
    assert any(m.get("content") == _REFLECT_NUDGE for m in engine.seen)


def test_step_budget_always_terminates(project):
    ctx = _ctx(project)
    (ctx.workspace / "f").write_text("x")
    endless = [{"tool": "read_file", "args": {"path": "f"}}] * 30
    engine = ScriptEngine(endless + [{"text": "closing line"}])
    line, _, _ = think_and_act(engine, system="s", user="u",
                            tools=build_tools(["read_file"], ctx), ctx=ctx,
                            max_steps=4)
    assert line  # spoke something rather than spinning forever


def test_identical_tool_calls_trip_the_stuck_reflex(project):
    # the field bug: rewording the same sentence while making the same tool
    # call forever — every step "says something", so the reflect nudge never
    # fires. The stuck reflex must push, then end the turn.
    from karl.loop import _STUCK_NUDGE
    ctx = _ctx(project)
    same = {"tool": "list_dir", "args": {}, "say": "I'll have wrench create it."}
    engine = _Recorder([dict(same) for _ in range(12)])
    line, _, _ = think_and_act(engine, system="s", user="u",
                            tools=build_tools(["list_dir"], ctx), ctx=ctx,
                            max_steps=12)
    assert line   # spoke and terminated
    assert any(m.get("content") == _STUCK_NUDGE for m in engine.seen)
    assert engine.script            # bailed long before burning the whole script


def test_tool_start_fires_before_the_result(project):
    # the tach relies on this ordering to say "running <tool>" during a slow
    # tool instead of impersonating a silent model
    ctx = _ctx(project)
    (ctx.workspace / "f").write_text("x")
    events = []
    engine = ScriptEngine([{"tool": "read_file", "args": {"path": "f"}},
                           {"text": "done"}])
    think_and_act(engine, system="s", user="u",
                  tools=build_tools(["read_file"], ctx), ctx=ctx,
                  on_tool_start=lambda n: events.append(("start", n)),
                  on_tool=lambda n, f: events.append(("end", n)))
    assert events == [("start", "read_file"), ("end", "read_file")]


def test_empty_answer_gets_one_nudge(project):
    ctx = _ctx(project)
    engine = _Recorder([{"text": ""}, {"text": "Here is my line."}])
    line, _, _ = think_and_act(engine, system="s", user="u", tools=[], ctx=ctx)
    assert line == "Here is my line."


# --- v10: structured handoff and context compaction -------------------------
def test_handoff_tool_ends_the_turn_with_structured_routing(project):
    ctx = _ctx(project)
    engine = ScriptEngine([
        {"tool": "read_file", "args": {"path": "nope"}},
        {"tool": "handoff", "args": {"to": "wrench",
                                     "message": "Nothing there. Build it."}},
    ])
    line, _, to = think_and_act(engine, system="s", user="u",
                                tools=build_tools(["read_file"], ctx), ctx=ctx,
                                handoff_to=["wrench", "operator"])
    assert to == "wrench"
    assert line == "Nothing there. Build it."


def test_bad_handoff_target_keeps_words_falls_back_to_prose(project):
    ctx = _ctx(project)
    engine = ScriptEngine([{"tool": "handoff",
                            "args": {"to": "nobody", "message": "hm. wrench, go."}}])
    line, _, to = think_and_act(engine, system="s", user="u", tools=[], ctx=ctx,
                                handoff_to=["wrench", "operator"])
    assert to is None                       # session routes by prose instead
    assert "wrench" in line


def test_old_tool_output_compacts_within_budget(project):
    import karl.loop as loop_mod
    ctx = _ctx(project)
    (ctx.workspace / "big.txt").write_text("z" * 3000)
    reads = [{"tool": "read_file", "args": {"path": "big.txt"}}] * 3
    # distinct args each read so the stuck reflex stays out of the way
    for i, r in enumerate(reads):
        r["args"] = {"path": "big.txt", "offset": i}
    engine = _Recorder(reads + [{"text": "done"}])
    old_budget = loop_mod._CONTEXT_BUDGET
    loop_mod._CONTEXT_BUDGET = 4000
    loop_mod_keep = loop_mod._KEEP_RECENT
    loop_mod._KEEP_RECENT = 2
    try:
        line, _, _ = think_and_act(engine, system="s", user="u",
                                   tools=build_tools(["read_file"], ctx), ctx=ctx)
    finally:
        loop_mod._CONTEXT_BUDGET = old_budget
        loop_mod._KEEP_RECENT = loop_mod_keep
    assert line == "done"
    assert any("compacted" in str(m.get("content"))
               for m in engine.seen if m.get("role") == "tool")
