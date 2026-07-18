"""The toolbox and its rails."""

from __future__ import annotations

import json

from karl.engine import ToolCall
from karl.tools import ToolContext, build_tools, execute


def _ctx(project, **kw):
    ws = project.workspace
    ws.mkdir(parents=True, exist_ok=True)
    return ToolContext(workspace=ws, project=project, **kw)


def _run(ctx, names, tool, args):
    tools = build_tools(names, ctx)
    return execute(tools, ToolCall("t1", tool, json.dumps(args)), ctx)


# --- the sandbox rail ------------------------------------------------------
def test_path_escaping_the_workspace_is_refused(project):
    ctx = _ctx(project)
    out = _run(ctx, ["read_file"], "read_file", {"path": "../../etc/passwd"})
    assert "escapes the workspace" in out


def test_write_then_read_roundtrip(project):
    ctx = _ctx(project)
    _run(ctx, ["write_file"], "write_file", {"path": "a/b.txt", "content": "hello"})
    assert ctx.changed == ["a/b.txt"]
    out = _run(ctx, ["read_file"], "read_file", {"path": "a/b.txt"})
    assert out == "hello"


def test_long_files_page_instead_of_truncating(project):
    ctx = _ctx(project)
    (ctx.workspace / "big.txt").write_text("x" * 9000)
    first = _run(ctx, ["read_file"], "read_file", {"path": "big.txt"})
    assert "TRUNCATED" in first and "offset=8000" in first
    rest = _run(ctx, ["read_file"], "read_file", {"path": "big.txt", "offset": 8000})
    assert rest == "x" * 1000


# --- edit_file: surgical, refuses ambiguity --------------------------------
def test_edit_file_replaces_one_exact_match(project):
    ctx = _ctx(project)
    (ctx.workspace / "f.py").write_text("x = 1\ny = 2\n")
    out = _run(ctx, ["edit_file"], "edit_file",
               {"path": "f.py", "find": "y = 2", "replace": "y = 3"})
    assert "1 replacement" in out
    assert (ctx.workspace / "f.py").read_text() == "x = 1\ny = 3\n"


def test_edit_file_refuses_zero_and_many_matches(project):
    ctx = _ctx(project)
    (ctx.workspace / "f.py").write_text("a = 1\na = 1\n")
    missing = _run(ctx, ["edit_file"], "edit_file",
                   {"path": "f.py", "find": "z = 9", "replace": ""})
    assert "no match" in missing
    many = _run(ctx, ["edit_file"], "edit_file",
                {"path": "f.py", "find": "a = 1", "replace": "a = 2"})
    assert "matches 2 times" in many
    assert (ctx.workspace / "f.py").read_text() == "a = 1\na = 1\n"  # untouched


# --- search ----------------------------------------------------------------
def test_search_finds_and_caps(project):
    ctx = _ctx(project)
    (ctx.workspace / "a.py").write_text("def alpha(): pass\n")
    (ctx.workspace / "b.py").write_text("alpha()\n" * 60)
    out = _run(ctx, ["search"], "search", {"pattern": r"alpha"})
    assert "a.py:1:" in out
    assert "capped at 50" in out


def test_search_rejects_a_bad_regex(project):
    out = _run(_ctx(project), ["search"], "search", {"pattern": "("})
    assert "bad regex" in out


# --- shell rail ------------------------------------------------------------
def test_shell_off_is_denied(project):
    out = _run(_ctx(project), ["run_shell"], "run_shell", {"command": "id"})
    assert out.startswith("DENIED")


def test_no_runtime_asks_the_operator_and_obeys_the_answer(project, monkeypatch):
    import karl.shell
    monkeypatch.setattr(karl.shell, "probe_runtime", lambda: "")

    # headless / no consent hook → denied with the ways out
    ctx = _ctx(project, shell_mode="container")
    out = _run(ctx, ["run_shell"], "run_shell", {"command": "echo hi"})
    assert "DENIED" in out and "Docker" in out

    # operator says yes → the command runs on the host, this call
    asked = []
    ctx = _ctx(project, shell_mode="container",
               ask=lambda q: asked.append(q) or True)
    out = _run(ctx, ["run_shell"], "run_shell", {"command": "echo hi"})
    assert "[exit 0]" in out and "hi" in out
    assert "HOST" in asked[0]

    # operator says no → denied
    ctx = _ctx(project, shell_mode="container", ask=lambda q: False)
    out = _run(ctx, ["run_shell"], "run_shell", {"command": "echo hi"})
    assert "DENIED" in out


def test_session_shell_grant_survives_refresh(project, monkeypatch):
    monkeypatch.setenv("KARL_OFFLINE", "1")
    from karl.session import Session
    s = Session(project, echo=False)
    assert s.shell_mode == "container"    # the new default
    s._shell_grant = "host"               # what saying yes at the prompt sets
    s._refresh()
    assert s.shell_mode == "host"         # config re-read does not revoke it


def test_host_shell_runs_in_the_workspace(project):
    ctx = _ctx(project, shell_mode="host")
    (ctx.workspace / "hello.txt").write_text("hi")
    out = _run(ctx, ["run_shell"], "run_shell", {"command": "ls"})
    assert "[exit 0]" in out and "hello.txt" in out


# --- web rail --------------------------------------------------------------
def test_web_fetch_not_built_without_egress(project):
    ctx = _ctx(project, can_egress=False)
    assert build_tools(["web_fetch"], ctx) == []


def test_web_fetch_refuses_non_http_schemes(project):
    ctx = _ctx(project, can_egress=True)
    out = _run(ctx, ["web_fetch"], "web_fetch", {"url": "file:///etc/passwd"})
    assert out.startswith("DENIED")


def test_web_fetch_refuses_loopback(project):
    ctx = _ctx(project, can_egress=True)
    out = _run(ctx, ["web_fetch"], "web_fetch", {"url": "http://127.0.0.1:8080/"})
    assert "DENIED" in out and "non-public" in out


def test_gated_web_consults_the_allowlist(project):
    ctx = _ctx(project, can_egress=True, web_open=False)
    out = _run(ctx, ["web_fetch"], "web_fetch", {"url": "https://example.com/x"})
    assert "DENIED" in out and "allowlist" in out


# --- remember --------------------------------------------------------------
def test_remember_appends_a_durable_note(project):
    ctx = _ctx(project)
    out = _run(ctx, ["remember"], "remember", {"note": "the API needs a token"})
    assert "remembered" in out
    assert "the API needs a token" in project.notes()


# --- execute plumbing ------------------------------------------------------
def test_execute_survives_bad_json_and_unknown_tools(project):
    ctx = _ctx(project)
    tools = build_tools(["read_file"], ctx)
    assert "not valid JSON" in execute(tools, ToolCall("1", "read_file", "{oops"), ctx)
    assert "no such tool" in execute(tools, ToolCall("2", "nope", "{}"), ctx)
