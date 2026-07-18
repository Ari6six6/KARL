"""The cockpit from the outside: argv in, exit codes out."""

from __future__ import annotations

from karl.cli import config_from_args, main
from karl.config import endpoint


def test_version_flag(project, capsys):
    assert main(["--version"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("karl ")


def test_help_flag(project, capsys):
    assert main(["--help"]) == 0
    assert "/dash" in capsys.readouterr().out


def test_config_roundtrip(project, capsys):
    rc = config_from_args(["--base-url", "http://localhost:1234/v1",
                           "--model", "tiny", "--shell", "container",
                           "--timeout", "60"])
    assert rc == 0
    cfg = endpoint()
    assert cfg["base_url"] == "http://localhost:1234/v1"
    assert cfg["model"] == "tiny"
    assert cfg["shell"] == "container"
    assert cfg["timeout"] == 60.0


def test_config_rejects_unknown_flags_and_bad_values(project, capsys):
    assert config_from_args(["--wat", "x"]) == 2
    assert config_from_args(["--shell", "yolo"]) == 2
    assert config_from_args(["--web", "sideways"]) == 2


def test_ping_without_endpoint_fails_cleanly(project, capsys):
    assert main(["ping"]) == 1
    assert "no endpoint" in capsys.readouterr().out


def test_run_without_task_is_a_usage_error(project, capsys):
    assert main(["run"]) == 2


def test_headless_run_refuses_without_a_model(project, capsys):
    assert main(["run", "look around"]) == 1
    out = capsys.readouterr().out
    assert "no model attached" in out


def test_headless_run_with_the_opt_in_stand_in(project, capsys, monkeypatch):
    monkeypatch.setenv("KARL_OFFLINE", "1")
    assert main(["run", "look around"]) == 0
    out = capsys.readouterr().out
    assert "operator" in out       # the transcript printed
    assert "canned theater" in out # and it said what it was


def test_dash_command(project, capsys):
    assert main(["dash"]) == 0
    out = capsys.readouterr().out
    assert "project" in out and "engine" in out


def test_bare_exit_words_leave_the_cockpit(project, monkeypatch):
    monkeypatch.setenv("KARL_OFFLINE", "1")
    from karl.cli import _dispatch
    from karl.session import Session
    s = Session(project, echo=False)
    for word in ("quit", "exit", "q", ":q", "QUIT"):
        assert _dispatch(s, word) is False
    assert s.transcript.entries() == []    # none of them became a task


def test_ctrl_c_on_a_headless_run_exits_clean(project, monkeypatch, capsys):
    import karl.cli as cli
    monkeypatch.setattr(cli, "_main",
                        lambda argv=None: (_ for _ in ()).throw(KeyboardInterrupt))
    assert main(["run", "x"]) == 130
    assert "Traceback" not in capsys.readouterr().out


def test_workspace_flag_sets_the_override(project, capsys, tmp_path, monkeypatch):
    import os
    monkeypatch.delenv("KARL_WORKSPACE", raising=False)
    target = tmp_path / "ws"
    assert main(["-C", str(target), "dash"]) == 0
    assert os.environ["KARL_WORKSPACE"] == str(target.resolve())
