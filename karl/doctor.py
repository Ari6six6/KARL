"""``karl doctor`` — onboard diagnostics for the stuck moment.

One command, a few seconds, and it answers the chain in order: is an endpoint
configured → is the tunnel process alive → does the endpoint answer at all →
does the model actually produce words → and, when a GPU box is on file, what
the server on the box says about itself (running? crashed? spilled to CPU?).
It ends with a verdict and the exact next command, because "he's stuck" is
always one of these and never a mystery.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request

from karl import ui
from karl.config import endpoint

_QUICK_S = 25          # the completion probe's whole budget
_OFFLOAD = re.compile(r"offloaded (\d+)\s*/\s*(\d+) layers")


def _ok(out, label, detail=""):
    out("  " + ui.green("✓") + f" {label:11}" + ui.dim(str(detail)))


def _bad(out, label, detail=""):
    out("  " + ui.red("✗") + f" {label:11}" + str(detail))


def run_doctor(out=print) -> int:
    from karl import gpu
    cfg = endpoint()
    trouble: list = []

    # 1 · is anything configured at all?
    if not cfg["base_url"]:
        _bad(out, "config", "no endpoint set")
        out(ui.dim("     → karl gpu ssh <ssh… -L port:host:port>   or   "
                   "karl config --base-url <url>"))
        return 1
    _ok(out, "config", f"{cfg['model']} @ {cfg['base_url']}")

    # 2 · the tunnel process, when a box is on file
    state = gpu._load()
    boxed = bool(state.get("ssh_conn"))
    if boxed and state.get("served"):
        if gpu._alive(state.get("tunnel_pid")):
            _ok(out, "tunnel", f"pid {state.get('tunnel_pid')} alive")
        else:
            _bad(out, "tunnel", "process gone")
            trouble.append("the tunnel is down — `karl gpu reconnect`")

    # 3 · does the endpoint answer at all?
    t0 = time.time()
    up = False
    try:
        with urllib.request.urlopen(cfg["base_url"] + "/models", timeout=6):
            pass
        _ok(out, "endpoint", f"/models answered in {time.time() - t0:.1f}s")
        up = True
    except Exception as e:  # noqa: BLE001 — every failure here is a finding
        _bad(out, "endpoint", f"no answer ({type(e).__name__})")
        trouble.append("nothing listens at the endpoint — tunnel down or server "
                       "dead: `karl gpu reconnect`, then `karl doctor` again")

    # 4 · does the model produce words? (single shot, tight budget, no retries)
    if up:
        from karl.engine import HTTPEngine
        probe = dict(cfg, timeout=_QUICK_S, max_tokens=8)
        eng = HTTPEngine(probe)
        t0 = time.time()
        try:
            with eng._open(eng._body(
                    [{"role": "user", "content": "Reply with exactly: OK"}],
                    None)) as resp:
                payload = json.loads(resp.read().decode("utf-8", "replace"))
            word = (payload["choices"][0]["message"].get("content") or "").strip()
            _ok(out, "completion", f"replied in {time.time() - t0:.1f}s: "
                                   f"{word[:40] or '(empty)'}")
        except Exception as e:  # noqa: BLE001
            _bad(out, "completion", f"no words in {_QUICK_S}s ({type(e).__name__})")
            trouble.append("the server accepts requests but produces nothing — "
                           "still loading, overloaded, or running the model on "
                           "CPU (see the box check below)")

    # 5 · what the box itself says, over ssh
    if boxed:
        cargs = state["ssh_conn"]
        out(ui.dim("  checking the box over ssh…"))

        # the GPU itself first — a dead GPU explains everything downstream,
        # and it's the one failure that impersonates "slow model" perfectly
        rc, smi, _ = gpu.run(
            cargs, "nvidia-smi --query-gpu=name,memory.used,memory.total,"
                   "utilization.gpu --format=csv,noheader,nounits", timeout=25)
        if rc != 0 or not smi.strip() or "ERR" in smi.upper():
            _bad(out, "gpu", "nvidia-smi is not answering sanely — the GPU "
                             "is dead or fell off the bus")
            trouble.append("the box's GPU is broken — destroy this instance "
                           "in the Vast console and rent another; no setting "
                           "fixes dead silicon")
        else:
            used = total = 0
            names = []
            for ln in smi.strip().splitlines():
                bits = [b.strip() for b in ln.split(",")]
                if len(bits) >= 4:
                    names.append(f"{bits[0]} {bits[1]}/{bits[2]}MB util {bits[3]}%")
                    try:
                        used += int(float(bits[1]))
                        total += int(float(bits[2]))
                    except ValueError:
                        pass
            _ok(out, "gpu", " · ".join(names) or smi.strip()[:100])
            if gpu.server_running(cargs) and total and used < 1024:
                _bad(out, "vram", f"server is up but only {used}MB VRAM in use "
                                  "— the model is not on the GPU")
                trouble.append("the model loaded outside the GPU (CPU fallback) "
                               "— check `~/karl.log` for CUDA errors; if the "
                               "GPU looks healthy, re-run `karl gpu ssh …`")

        if gpu.server_running(cargs):
            _ok(out, "server", "process alive on the box")
        else:
            _bad(out, "server", "not running on the box")
            trouble.append("the model server died or was never started — "
                           "re-run `karl gpu ssh <your ssh line>`")
        rc, log, _ = gpu.run(cargs, f"tail -n 80 {gpu.LOG_FILE} 2>/dev/null",
                             timeout=25)
        if rc == 0 and log.strip():
            hits = _OFFLOAD.findall(log)
            if hits:
                on_gpu, total = map(int, hits[-1])
                if on_gpu < total:
                    _bad(out, "gpu fit", f"offloaded {on_gpu}/{total} layers — "
                                         "the rest run on CPU")
                    trouble.append("the model does not fit this box's VRAM and "
                                   "crawls on CPU — pick a smaller one: "
                                   "`karl gpu model qwen`, then re-run "
                                   "`karl gpu ssh …`")
                else:
                    _ok(out, "gpu fit", f"all {total} layers on GPU")
            for ln in log.strip().splitlines()[-3:]:
                out(ui.dim("     | " + ln[:150]))

    # verdict
    out("")
    if trouble:
        out("  " + ui.yellow("verdict: ") + trouble[0])
        for t in trouble[1:]:
            out(ui.dim("           also: " + t))
        return 1
    out("  " + ui.green("verdict: all clear.")
        + ui.dim(" If it still feels slow, that's the model itself — watch the "
                 "tach hint, or pick a smaller one."))
    return 0
