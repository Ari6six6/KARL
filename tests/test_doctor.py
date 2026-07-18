"""`karl doctor` — the diagnostics must diagnose, fast, without a real box."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from karl.config import set_config
from karl.doctor import run_doctor


def test_doctor_without_endpoint(project, capsys):
    assert run_doctor() == 1
    out = capsys.readouterr().out
    assert "no endpoint set" in out and "gpu ssh" in out


def test_doctor_flags_a_dead_endpoint(project, capsys):
    set_config(base_url="http://127.0.0.1:9/v1", model="m")  # nothing listens
    assert run_doctor() == 1
    out = capsys.readouterr().out
    assert "no answer" in out
    assert "reconnect" in out


class _Live(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"data": []}')

    def do_POST(self):
        body = json.dumps({"choices": [{"message": {"role": "assistant",
                                                    "content": "OK"}}]}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)


def test_doctor_all_clear_with_a_live_endpoint(project, capsys):
    srv = HTTPServer(("127.0.0.1", 0), _Live)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        set_config(base_url=f"http://127.0.0.1:{srv.server_port}/v1", model="m")
        assert run_doctor() == 0
        out = capsys.readouterr().out
        assert "all clear" in out
        assert "replied in" in out
    finally:
        srv.shutdown()


def test_doctor_unmasks_a_dead_gpu(project, capsys, monkeypatch):
    from karl import gpu
    set_config(base_url="http://127.0.0.1:9/v1", model="m")
    gpu._save({"ssh_conn": ["-p", "1", "root@h"], "served": True,
               "local_port": 8080, "remote_port": 8080, "tunnel_pid": None})

    def fake_run(cargs, cmd, timeout=120):
        if "nvidia-smi" in cmd:
            return 1, "", "Unable to determine the device handle for GPU"
        return 0, "some log line", ""

    monkeypatch.setattr(gpu, "run", fake_run)
    monkeypatch.setattr(gpu, "server_running", lambda cargs: True)
    assert run_doctor() == 1
    out = capsys.readouterr().out
    assert "dead or fell off the bus" in out
    assert "destroy this instance" in out


def test_doctor_from_the_cli(project, capsys):
    from karl.cli import main
    assert main(["doctor"]) == 1          # no endpoint in a fresh home
    assert "no endpoint" in capsys.readouterr().out
