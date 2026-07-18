"""The engine: SSE stream parsing, loop trimming, and the stand-ins."""

from __future__ import annotations

import json

from karl.engine import (MockEngine, ScriptEngine, cut_loops, make_engine,
                         parse_sse)


# --- SSE parsing -----------------------------------------------------------
def _chunk(delta):
    return "data: " + json.dumps({"choices": [{"delta": delta}]})


def test_sse_content_streams_and_folds():
    got = []
    res = parse_sse([_chunk({"content": "The "}), _chunk({"content": "east "}),
                     _chunk({"content": "ford is out."}), "data: [DONE]"],
                    on_token=got.append)
    assert res.content == "The east ford is out."
    assert got == ["The ", "east ", "ford is out."]
    assert res.tool_calls == []


def test_sse_tool_call_reassembled_from_slices():
    res = parse_sse([
        _chunk({"tool_calls": [{"index": 0, "id": "c1",
                                "function": {"name": "read_file", "arguments": ""}}]}),
        _chunk({"tool_calls": [{"index": 0,
                                "function": {"arguments": "{\"path\": "}}]}),
        _chunk({"tool_calls": [{"index": 0,
                                "function": {"arguments": "\"a.txt\"}"}}]}),
        "data: [DONE]",
    ])
    assert len(res.tool_calls) == 1
    call = res.tool_calls[0]
    assert call.id == "c1"
    assert call.name == "read_file"
    assert json.loads(call.arguments) == {"path": "a.txt"}


def test_sse_survives_noise_and_bytes():
    res = parse_sse([b": keepalive", b"", _chunk({"content": "ok"}).encode(),
                     b"data: not json", b"data: [DONE]"])
    assert res.content == "ok"


def test_sse_parallel_tool_calls_keep_order():
    res = parse_sse([
        _chunk({"tool_calls": [{"index": 0, "id": "a",
                                "function": {"name": "list_dir", "arguments": "{}"}},
                               {"index": 1, "id": "b",
                                "function": {"name": "search",
                                             "arguments": "{\"pattern\": \"x\"}"}}]}),
        "data: [DONE]",
    ])
    assert [c.name for c in res.tool_calls] == ["list_dir", "search"]


# --- loop trimming ---------------------------------------------------------
def test_cut_loops_trims_a_repeated_sentence():
    s = "The gate is locked and will not open today. "
    out = cut_loops(s * 5)
    assert out.count("The gate is locked") == 1
    assert out.endswith("…(repeat loop trimmed)")


def test_cut_loops_trims_a_tiling_tail():
    out = cut_loops("Here is the plan: " + "go north, " * 12)
    assert out.endswith("…(repeat loop trimmed)")
    # the detector's smallest unit is 12 chars, so the kept unit may hold the
    # 10-char phrase twice — the point is 12 repetitions collapsed to ≤2
    assert out.count("go north") <= 2


def test_cut_loops_leaves_ordinary_text_alone():
    text = "First check the file. Then run the tests. Then report back."
    assert cut_loops(text) == text
    assert cut_loops("") == ""
    assert cut_loops(None) == ""


# --- the stand-ins ---------------------------------------------------------
def test_mock_engine_speaks_its_seed_and_streams_it():
    e = MockEngine()
    e.seed("I am ready.")
    got = []
    res = e.chat([{"role": "user", "content": "go"}], on_token=got.append)
    assert res.content == "I am ready."
    assert got == ["I am ready."]


def test_script_engine_drives_tools_then_text():
    e = ScriptEngine([{"tool": "read_file", "args": {"path": "x"}},
                      {"text": "done"}])
    first = e.chat([])
    assert first.tool_calls[0].name == "read_file"
    assert e.chat([]).content == "done"
    assert e.chat([]).content == "(script exhausted)"


def test_make_engine_offline_without_endpoint(project):
    engine, mode = make_engine()
    assert mode == "offline"
    assert isinstance(engine, MockEngine)
