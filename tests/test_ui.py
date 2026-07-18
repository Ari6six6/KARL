"""The cockpit rendering (off a TTY everything must be inert plain text)."""

from __future__ import annotations

from karl import ui
from karl.shell import probe_runtime


def test_bar_renders_bounds():
    assert ui.bar(0.0).startswith("[") and "  0%" in ui.bar(0.0)
    assert "100%" in ui.bar(1.0)
    assert "100%" in ui.bar(7.5)      # clamped


def test_line_shows_operator_as_you():
    out = ui.line("karl", "operator", "all done.")
    assert "you" in out and "all done." in out


def test_dash_aligns_keys():
    out = ui.dash([("a", "1"), ("longer", "2")])
    lines = out.splitlines()
    assert len(lines) == 2
    assert lines[0].index("1") == lines[1].index("2")


def test_stream_is_inert_off_a_tty(capsys):
    s = ui.Stream("scout", enabled=True)   # still gated by isatty
    s.token("hello")
    s.end()
    assert s.spoke is False
    assert capsys.readouterr().out == ""


def test_tach_is_inert_off_a_tty():
    t = ui.Tach("thinking")
    t.start()
    assert t.active is False
    t.stop()      # must not raise


def test_probe_runtime_returns_string():
    # docker may or may not exist here; the contract is a str, never a crash
    assert isinstance(probe_runtime(), str)
