"""The chassis: config, projects, and the allowlist."""

from __future__ import annotations

from karl.config import Project, endpoint, set_config, valid_project_name, web_open


def test_endpoint_defaults_offline(project):
    cfg = endpoint()
    assert cfg["base_url"] == ""
    assert cfg["shell"] == "off"
    assert cfg["stream"] is True


def test_env_beats_the_config_file(project, monkeypatch):
    set_config(base_url="http://file:1/v1", model="from-file")
    monkeypatch.setenv("KARL_BASE_URL", "http://env:2/v1/")
    monkeypatch.setenv("KARL_MODEL", "from-env")
    cfg = endpoint()
    assert cfg["base_url"] == "http://env:2/v1"   # trailing slash trimmed too
    assert cfg["model"] == "from-env"


def test_bogus_shell_mode_falls_back_to_off(project):
    set_config(shell="yolo")
    assert endpoint()["shell"] == "off"


def test_stream_can_be_switched_off(project):
    set_config(stream="off")
    assert endpoint()["stream"] is False


def test_web_open_by_default_and_gateable(project, monkeypatch):
    assert web_open() is True
    set_config(web="gated")
    assert web_open() is False
    monkeypatch.setenv("KARL_WEB", "open")
    assert web_open() is True


def test_project_names_are_validated():
    assert valid_project_name("my-project.2")
    assert not valid_project_name("")
    assert not valid_project_name("../evil")
    assert not valid_project_name(".hidden")


def test_allowlist_roundtrip_and_normalization(project):
    assert project.allow("HTTPS://Docs.Python.org/3/") == "docs.python.org"
    assert project.egress_allowed("docs.python.org")
    assert not project.egress_allowed("evil.example")
    assert project.allow("not a host!") == ""       # junk never lands
    assert project.disallow("docs.python.org")
    assert not project.egress_allowed("docs.python.org")


def test_wildcard_opens_everything(project):
    project.allow("*")
    assert project.egress_allowed("anything.example")


def test_workspace_override(project, monkeypatch, tmp_path):
    other = tmp_path / "elsewhere"
    monkeypatch.setenv("KARL_WORKSPACE", str(other))
    assert project.workspace == other.resolve()
