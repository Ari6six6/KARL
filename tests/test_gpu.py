"""The turbo: ssh arg surgery, planning, and connection triage."""

from __future__ import annotations

import pytest

from karl import gpu
from karl.models import GLM, HERMES


# --- ssh arg surgery -------------------------------------------------------
def test_parse_forward_reads_the_L_flag():
    assert gpu.parse_forward(["-p", "24439", "root@1.2.3.4",
                              "-L", "8080:localhost:8080"]) == (8080, "localhost", 8080)
    assert gpu.parse_forward(["-L8081:10.0.0.5:8000", "host"]) == (8081, "10.0.0.5", 8000)
    assert gpu.parse_forward(["-L", "9000:9001", "host"]) == (9000, "127.0.0.1", 9001)


def test_parse_forward_rejects_junk():
    assert gpu.parse_forward(["root@host"]) is None
    assert gpu.parse_forward(["-L", "nope:x", "host"]) is None
    assert gpu.parse_forward(["-L", "99999:h:80", "host"]) is None


def test_conn_args_drops_the_tunnel_parts():
    args = ["-p", "24439", "root@1.2.3.4", "-L", "8080:localhost:8080", "-N"]
    assert gpu.conn_args(args) == ["-p", "24439", "root@1.2.3.4"]


def test_replace_forward_slides_only_the_box_side():
    out = gpu.replace_forward(["-L", "8080:localhost:8080", "h"], 18080)
    assert out == ["-L", "8080:localhost:18080", "h"]
    out = gpu.replace_forward(["-L8080:localhost:8080", "h"], 18080)
    assert out == ["-L8080:localhost:18080", "h"]


# --- planning --------------------------------------------------------------
def test_plan_picks_a_context_tier_by_vram():
    tp, max_len, util, total = gpu.plan([("A6000", 49140), ("A6000", 49140)], HERMES)
    assert tp == 2
    assert total == 95
    assert max_len == 65536      # first tier above 95GB is (96, 65536)
    assert util == 0.92


def test_plan_refuses_a_too_small_box():
    with pytest.raises(gpu.ProvisionError, match="VRAM"):
        gpu.plan([("RTX 3060", 12288)], GLM)
    with pytest.raises(gpu.ProvisionError, match="no GPUs"):
        gpu.plan([], GLM)


# --- connection triage -----------------------------------------------------
def _fake_run(rc, out="", err=""):
    return lambda cargs, cmd, timeout=120: (rc, out, err)


def test_check_connection_ok(monkeypatch):
    monkeypatch.setattr(gpu, "run", _fake_run(0, out="KARL_OK\n"))
    ok, why, transient = gpu.check_connection(["h"])
    assert ok and not transient


def test_check_connection_flags_a_booting_box_as_transient(monkeypatch):
    monkeypatch.setattr(gpu, "run", _fake_run(
        255, err="kex_exchange_identification: read: Connection reset by peer"))
    ok, why, transient = gpu.check_connection(["h"])
    assert not ok and transient


def test_check_connection_flags_bad_auth_as_permanent(monkeypatch):
    monkeypatch.setattr(gpu, "run", _fake_run(255, err="Permission denied (publickey)"))
    ok, why, transient = gpu.check_connection(["h"])
    assert not ok and not transient


def test_detect_gpus_parses_nvidia_smi(monkeypatch):
    monkeypatch.setattr(gpu, "run", _fake_run(
        0, out="NVIDIA A100-SXM4-80GB, 81920\nNVIDIA A100-SXM4-80GB, 81920\n"))
    gpus = gpu.detect_gpus(["h"])
    assert gpus == [("NVIDIA A100-SXM4-80GB", 81920)] * 2


def test_gpu_ssh_forgives_a_pasted_leading_ssh(monkeypatch, project):
    """`gpu ssh ssh -p … host -L …` — the redundant word must not become the
    hostname. The usage error must NOT trigger (a forward is present)."""
    said = []
    monkeypatch.setattr(gpu, "check_connection",
                        lambda cargs: (False, "stop here", False))
    gpu.handle("ssh ssh -p 1 root@h -L 8080:localhost:8080", out=said.append)
    text = "\n".join(said)
    assert "usage:" not in text
    assert "stop here" in text
