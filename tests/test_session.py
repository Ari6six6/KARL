"""The name-driven scheduler and whole sessions end to end."""

from __future__ import annotations

from karl.engine import ScriptEngine
from karl.session import Session, _plain, next_speaker

NAMES = ["karl", "scout", "wrench"]


# --- the name-mention scheduler --------------------------------------------
def test_named_teammate_speaks_next():
    assert next_speaker("karl", "scout, look into this.", NAMES, "karl") == "scout"


def test_chief_addressing_operator_closes_the_round():
    assert next_speaker("karl", "operator, here is the result.", NAMES, "karl") == "operator"


def test_teammate_addressing_operator_is_routed_to_chief():
    # only the chief speaks with the operator
    assert next_speaker("wrench", "operator, it is done.", NAMES, "karl") == "karl"


def test_a_line_naming_no_one_falls_to_the_chief():
    assert next_speaker("wrench", "I finished the task.", NAMES, "karl") == "karl"


def test_speaker_does_not_address_itself():
    assert next_speaker("karl", "I, karl, will ask scout next.", NAMES, "karl") == "scout"


def test_names_match_whole_words_only():
    # "scoutmaster" must not summon scout
    assert next_speaker("karl", "the scoutmaster said no. wrench, check it.",
                        NAMES, "karl") == "wrench"


def test_last_name_wins_mid_sentence_mentions_do_not_steal_the_handoff():
    # the field bug: a question FOR the operator that mentions a teammate on
    # the way must close the round, not summon the teammate
    line = ("Once you provide these details, I'll have wrench set up the "
            "appropriate structure for you. operator")
    assert next_speaker("karl", line, NAMES, "karl") == "operator"
    # and the mirror image: quoting the operator then delegating still delegates
    line2 = "The operator asked for a check. wrench, run the tests."
    assert next_speaker("karl", line2, NAMES, "karl") == "wrench"


# --- transcript hygiene ----------------------------------------------------
def test_small_code_crosses_the_transcript_intact():
    line = _plain("done. ```python\nif x:\n    y = 1\n``` see the file")
    assert "if x:\n    y = 1" in line       # indentation survives
    assert "code omitted" not in line


def test_huge_code_blocks_become_a_pointer():
    line = _plain("done. ```\n" + "x = 1\n" * 500 + "``` see the file")
    assert "x = 1" not in line
    assert "large code omitted" in line
    assert "done." in line and "see the file" in line


# --- whole sessions --------------------------------------------------------
def test_no_model_means_no_round(project, capsys):
    s = Session(project, echo=True)    # nothing attached, no KARL_OFFLINE
    assert s.run_task("whats in the report?") is False
    assert s.transcript.entries() == []          # not one fake word
    out = capsys.readouterr().out
    assert "no model attached" in out and "gpu ssh" in out


def test_no_model_auto_reattaches_a_saved_box(project, monkeypatch):
    # a box on file from an earlier `gpu ssh` → the session tries reconnect
    from karl import gpu
    gpu._save({"ssh_conn": ["-p", "1", "root@h"], "local_port": 8080,
               "remote_port": 8080})
    calls = []
    monkeypatch.setattr(gpu, "handle",
                        lambda rest, out=print: calls.append(rest))
    s = Session(project, echo=False)
    assert s.run_task("go") is False             # fake reconnect set no config
    assert calls == ["reconnect"]


def test_offline_session_runs_and_terminates(project, monkeypatch):
    monkeypatch.setenv("KARL_OFFLINE", "1")      # the stand-in, explicitly
    s = Session(project, echo=False)
    assert s.run_task("take stock of the workspace") is True
    entries = s.transcript.entries()
    assert entries[0]["speaker"] == "operator"
    assert entries[-1]["speaker"] == "karl"
    assert entries[-1]["addressee"] == "operator"


def test_scripted_session_hands_off_by_name(project):
    engine = ScriptEngine([
        {"text": "wrench, list what is in the workspace."},   # karl delegates
        {"tools": [{"tool": "list_dir", "args": {}}],
         "say": None},
        {"text": "It is empty. karl, over to you."},          # wrench reports
        {"text": "operator, the workspace is empty."},        # karl closes
    ])
    s = Session(project, echo=False, engine=engine)
    s.run_task("what's in the workspace?")
    speakers = [e["speaker"] for e in s.transcript.entries()]
    assert speakers == ["operator", "karl", "wrench", "karl"]


def test_round_survives_a_wandering_crew(project):
    # a crew that never addresses the operator still terminates via the cap
    engine = ScriptEngine([{"text": "scout, keep looking."},
                           {"text": "karl, still looking."}] * 20
                          + [{"text": "operator, we ran long; here is where we stand."}])
    s = Session(project, echo=False, engine=engine)
    s.run_task("dig forever")
    entries = s.transcript.entries()
    assert entries[-1]["addressee"] == "operator"


def test_taint_flag_reaches_the_operator(project):
    engine = ScriptEngine([{"text": "operator, all done."}])
    s = Session(project, echo=False, engine=engine)
    s.tainted.append("example.com")

    # taint recorded before the round doesn't flag it…
    s.run_task("quick check")
    assert "⚠" not in s.transcript.entries()[-1]["text"]

    # …but taint picked up during the round does
    engine2 = ScriptEngine([{"text": "operator, summarized from the web."}])
    s2 = Session(project, echo=False, engine=engine2)

    real_speak = s2._speak

    def tainted_speak(*a, **kw):
        s2.tainted.append("example.com")
        return real_speak(*a, **kw)

    s2._speak = tainted_speak
    s2.run_task("fetch and summarize")
    last = s2.transcript.entries()[-1]["text"]
    assert "⚠" in last and "example.com" in last


def test_offline_rounds_announce_the_stand_in(project, capsys, monkeypatch):
    monkeypatch.setenv("KARL_OFFLINE", "1")
    s = Session(project, echo=True)
    s.run_task("whats in the report?")
    out = capsys.readouterr().out
    assert "offline stand-in" in out and "canned" in out


def test_engine_refreshes_between_tasks(project, monkeypatch):
    # session born pointing at a model…
    monkeypatch.setenv("KARL_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("KARL_OFFLINE", "1")
    s = Session(project, echo=False)
    assert s.mode == "model"
    # …endpoint dropped (gpu off / config change) → next task must notice
    monkeypatch.delenv("KARL_BASE_URL")
    s.run_task("still there?")
    assert s.mode == "offline"
    assert s.transcript.entries()[-1]["addressee"] == "operator"


def test_injected_test_engine_is_never_refreshed(project, monkeypatch):
    monkeypatch.setenv("KARL_BASE_URL", "http://127.0.0.1:1/v1")
    engine = ScriptEngine([{"text": "operator, done."}])
    s = Session(project, echo=False, engine=engine)
    s.run_task("go")
    assert s.mode == "test"
    assert s.transcript.entries()[-1]["text"] == "operator, done."


# --- v10: memory, structured handoffs, the board ---------------------------
class _Recorder(ScriptEngine):
    def __init__(self, script):
        super().__init__(script)
        self.all_prompts = []

    def chat(self, messages, tools=None, on_token=None):
        self.all_prompts.append("\n".join(str(m.get("content")) for m in messages))
        return super().chat(messages, tools, on_token=on_token)


def test_a_follow_up_round_remembers_the_conversation(project):
    engine = _Recorder([
        {"text": "operator, what type of project do you want?"},   # round 1
        {"text": "operator, a Node.js project named blitz it is."},  # round 2
    ])
    s = Session(project, echo=False, engine=engine)
    s.run_task("set up a new project")
    s.run_task("node.js, call it blitz")
    # the round-2 prompt must carry round 1 verbatim — question AND answer
    final = engine.all_prompts[-1]
    assert "what type of project" in final
    assert "set up a new project" in final
    assert "node.js, call it blitz" in final


def test_memory_folds_but_operator_words_outlive_chatter():
    from karl.session import Memory
    m = Memory()
    m.add("operator→karl", "the secret port is 4242")
    for i in range(60):
        m.add("scout→karl", f"finding {i}: " + "x" * 400)
    rendered = m.render()
    assert "EARLIER IN THIS SESSION" in rendered          # chatter condensed
    assert "4242" in rendered                             # the instruction survives
    assert "still binding" in rendered                    # …in the ledger
    assert len(rendered) < (Memory.RECENT_BUDGET + Memory.LEDGER_BUDGET
                            + Memory.DIGEST_BUDGET + 800)


def test_structured_handoff_routes_without_prose_names(project):
    engine = ScriptEngine([
        {"tool": "handoff", "args": {"to": "wrench", "message": "Take a look."}},
        {"tool": "handoff", "args": {"to": "karl", "message": "All clear here."}},
        {"tool": "handoff", "args": {"to": "operator", "message": "Done: all clear."}},
    ])
    s = Session(project, echo=False, engine=engine)
    s.run_task("check the workspace")
    speakers = [(e["speaker"], e["addressee"]) for e in s.transcript.entries()]
    assert speakers == [("operator", "karl"), ("karl", "wrench"),
                        ("wrench", "karl"), ("karl", "operator")]


def test_teammate_handoff_to_operator_is_routed_through_the_chief(project):
    engine = ScriptEngine([
        {"tool": "handoff", "args": {"to": "wrench", "message": "Go."}},
        {"tool": "handoff", "args": {"to": "operator", "message": "It is done."}},
        {"tool": "handoff", "args": {"to": "operator", "message": "Done."}},
    ])
    s = Session(project, echo=False, engine=engine)
    s.run_task("do the thing")
    entries = s.transcript.entries()
    assert entries[2]["speaker"] == "wrench"
    assert entries[2]["addressee"] == "karl"     # not straight to the operator


def test_the_board_reaches_every_agent(project):
    project.board_path.write_text("# Goal: ship v2\n- [ ] write tests\n")
    engine = _Recorder([{"text": "operator, on it."}])
    s = Session(project, echo=False, engine=engine)
    s.run_task("continue")
    assert "ship v2" in engine.all_prompts[0]     # in the system prompt


def test_update_board_tool_writes_the_board(project):
    import json as _json
    from karl.engine import ToolCall
    from karl.tools import ToolContext, build_tools, execute
    ctx = ToolContext(workspace=project.workspace, project=project)
    tools = build_tools(["update_board"], ctx)
    out = execute(tools, ToolCall("t1", "update_board", _json.dumps(
        {"markdown": "# Goal: fix the brakes\n- [x] diagnose\n- [ ] bleed lines"})), ctx)
    assert "board updated" in out
    assert "bleed lines" in project.board()


def test_session_transcript_lands_on_disk(project, monkeypatch):
    monkeypatch.setenv("KARL_OFFLINE", "1")
    s = Session(project, echo=False)
    s.run_task("hello crew")
    assert s.transcript.path.exists()
    assert len(s.transcript.path.read_text().strip().splitlines()) == len(
        s.transcript.entries())
