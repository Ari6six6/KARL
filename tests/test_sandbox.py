"""apt_install and the project sandbox image — validation, consent, plumbing.
No Docker needed: builds are monkeypatched; the rails are what's under test."""

from __future__ import annotations

import json

import karl.shell as shell
from karl.engine import ToolCall
from karl.shell import image_for, sandbox_dockerfile, sandbox_record
from karl.tools import ToolContext, build_tools, execute


def _ctx(project, **kw):
    ws = project.workspace
    ws.mkdir(parents=True, exist_ok=True)
    kw.setdefault("shell_mode", "container")
    return ToolContext(workspace=ws, project=project, **kw)


def _install(ctx, args):
    tools = build_tools(["apt_install"], ctx)
    return execute(tools, ToolCall("t1", "apt_install", json.dumps(args)), ctx)


# --- the injection rail ----------------------------------------------------
def test_package_names_cannot_smuggle_shell(project):
    ctx = _ctx(project)
    out = _install(ctx, {"packages": ["jq; rm -rf /"]})
    assert out.startswith("ERROR") and "invalid" in out
    out = _install(ctx, {"pip": ["requests && curl evil"]})
    assert out.startswith("ERROR") and "invalid" in out
    out = _install(ctx, {"packages": ["$(reboot)"]})
    assert out.startswith("ERROR")


def test_empty_and_oversized_requests_are_refused(project):
    ctx = _ctx(project)
    assert "nothing to install" in _install(ctx, {})
    out = _install(ctx, {"packages": [f"pkg{i}" for i in range(25)]})
    assert "more than 20" in out


# --- consent and mode rails ------------------------------------------------
def test_headless_or_declined_stays_denied(project, monkeypatch):
    monkeypatch.setattr(shell, "probe_runtime", lambda: "docker")
    ctx = _ctx(project)                       # ask hook absent → headless
    assert "DENIED" in _install(ctx, {"packages": ["jq"]})
    ctx = _ctx(project, ask=lambda q: False)  # operator says no
    assert "declined" in _install(ctx, {"packages": ["jq"]})


def test_installs_off_and_host_mode_short_circuit(project):
    ctx = _ctx(project, installs="off")
    assert "disabled" in _install(ctx, {"packages": ["jq"]})
    ctx = _ctx(project, shell_mode="host")
    assert "host" in _install(ctx, {"packages": ["jq"]})
    ctx = _ctx(project, shell_mode="off")
    assert "DENIED" in _install(ctx, {"packages": ["jq"]})


def test_consented_install_builds_and_reports(project, monkeypatch):
    monkeypatch.setattr(shell, "probe_runtime", lambda: "docker")
    built = []
    monkeypatch.setattr(shell, "build_sandbox",
                        lambda proj, rt, apt, pip, timeout=900:
                        built.append((apt, pip)) or (0, ""))
    asked = []
    ctx = _ctx(project, ask=lambda q: asked.append(q) or True)
    out = _install(ctx, {"packages": ["jq", "gcc"], "pip": ["requests"]})
    assert "installed into the sandbox" in out
    assert built == [(["jq", "gcc"], ["requests"])]
    assert "jq, gcc, requests" in asked[0]


def test_open_mode_skips_the_question(project, monkeypatch):
    monkeypatch.setattr(shell, "probe_runtime", lambda: "docker")
    monkeypatch.setattr(shell, "build_sandbox",
                        lambda *a, **k: (0, ""))
    ctx = _ctx(project, installs="open")      # no ask hook at all
    assert "installed" in _install(ctx, {"packages": ["jq"]})


# --- image plumbing --------------------------------------------------------
def test_dockerfile_is_deterministic_and_clean():
    df = sandbox_dockerfile("python:3.12-slim", ["gcc", "jq"], ["requests"])
    assert df.startswith("FROM python:3.12-slim\n")
    assert "apt-get install -y --no-install-recommends gcc jq" in df
    assert "pip install --no-cache-dir requests" in df
    assert sandbox_dockerfile("base", [], []) == "FROM base\n"


def test_image_for_switches_once_packages_exist(project):
    assert image_for(project) == shell.shell_image()
    assert image_for(None) == shell.shell_image()
    from karl.config import save_json
    save_json(project.root / "sandbox.json", {"apt": ["jq"], "pip": []})
    assert image_for(project) == f"karl-sandbox-{project.name}"
    assert sandbox_record(project)["apt"] == ["jq"]


def test_shell_hints_at_apt_install_on_missing_command(project, monkeypatch):
    monkeypatch.setattr(shell, "probe_runtime", lambda: "docker")
    monkeypatch.setattr(shell, "run_in_container",
                        lambda *a, **k: (127, "", "sh: jq: not found"))
    ctx = _ctx(project)
    tools = build_tools(["run_shell"], ctx)
    out = execute(tools, ToolCall("t1", "run_shell",
                                  json.dumps({"command": "jq ."})), ctx)
    assert "apt_install" in out
