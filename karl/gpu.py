"""`karl gpu …` — the turbo: one command from a rented GPU box to a live model.

    karl gpu ssh -p 24439 root@1.2.3.4 -L 8080:localhost:8080

does the whole thing: reach the box (waiting out a booting sshd), detect the
GPUs, pick a VRAM/context tier, install vLLM (or build llama.cpp for GGUF
models), launch the server with tool-calling enabled, open a detached SSH
tunnel, and poll — with a download bar — until the model answers. Then it
points the harness at ``http://localhost:<port>/v1`` by writing KARL's config,
so ``karl``, ``karl ping``, and every session use it automatically.

Stdlib only: remote commands run over the ``ssh`` binary, readiness is polled
over urllib. The box keeps ``~/karl.pid`` and ``~/karl.log`` so status and
teardown stay runtime-agnostic. The tunnel is a detached background process
tracked by PID — close the shell and the served model stays up; ``karl gpu
off`` drops it.

This file carries the scars of real rented boxes, on purpose: apt and network
retries for half-provisioned images, CUDA library paths exported (and
registered with ldconfig) so llama-server survives exec, a squatted box port
that slides to a free one with the tunnel following, and a download bar that
notices when the server died rather than waiting out the deadline on a corpse.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

from karl import ui
from karl.config import karl_home, load_json, save_json, set_config
from karl.models import CATALOG, DEFAULT_KEY, ModelSpec, get_spec

VENV_DIR = "~/.karl-venv"
VLLM_BIN = f"{VENV_DIR}/bin/vllm"
LLAMA_DIR = "~/.karl-llama"
LLAMA_BIN = f"{LLAMA_DIR}/llama-server"
LLAMA_REPO = "https://github.com/ggml-org/llama.cpp"
PID_FILE = "~/karl.pid"
LOG_FILE = "~/karl.log"


class ProvisionError(Exception):
    pass


# --------------------------------------------------------------------------
# SSH arg surgery: reuse the operator's own pasted ssh command
# --------------------------------------------------------------------------
def conn_args(ssh_args: list) -> list:
    """The connection part of a pasted ssh command: drop ``-N`` and the ``-L``
    forward (those belong to the tunnel), keep host/port/user/-i/etc."""
    out, i = [], 0
    while i < len(ssh_args):
        a = ssh_args[i]
        if a == "-N":
            i += 1
            continue
        if a == "-L":
            i += 2
            continue
        if a.startswith("-L") and len(a) > 2:
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def parse_forward(ssh_args: list):
    """(local_port, remote_host, remote_port) from a ``-L lp:host:rp`` forward.

    Returns None for anything malformed — a pasted ssh line must never take the
    shell down with it."""
    val = ""
    for i, a in enumerate(ssh_args):
        if a == "-L" and i + 1 < len(ssh_args):
            val = ssh_args[i + 1]
            break
        if a.startswith("-L") and len(a) > 2:
            val = a[2:]
            break
    if not val:
        return None
    bits = val.split(":")
    try:
        if len(bits) == 3:
            lp, host, rp = int(bits[0]), bits[1], int(bits[2])
        elif len(bits) == 2:  # lp:rp -> localhost
            lp, host, rp = int(bits[0]), "127.0.0.1", int(bits[1])
        else:
            return None
    except ValueError:
        return None
    if not host or not (1 <= lp <= 65535) or not (1 <= rp <= 65535):
        return None
    return lp, host, rp


def replace_forward(ssh_args: list, new_remote_port: int) -> list:
    """The same ssh args with the ``-L`` forward's box-side port swapped — when
    the launch slides the server to a free port, the tunnel must follow."""
    out = list(ssh_args)
    for i, a in enumerate(out):
        if a == "-L" and i + 1 < len(out):
            bits = out[i + 1].split(":")
            if len(bits) in (2, 3):
                bits[-1] = str(new_remote_port)
                out[i + 1] = ":".join(bits)
            return out
        if a.startswith("-L") and len(a) > 2:
            bits = a[2:].split(":")
            if len(bits) in (2, 3):
                bits[-1] = str(new_remote_port)
                out[i] = "-L" + ":".join(bits)
            return out
    return out


_SSH_OPTS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
             "-o", "ConnectTimeout=15"]


def run(cargs: list, command: str, timeout: int = 120):
    """Run a remote command; returns (rc, stdout, stderr)."""
    try:
        p = subprocess.run(["ssh", *_SSH_OPTS, *cargs, command],
                           capture_output=True, text=True, errors="replace",
                           timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"
    except FileNotFoundError:
        return 127, "", "ssh binary not found"


def check_connection(cargs: list):
    """Try to reach the box. Returns (ok, reason, transient). ``transient``
    marks a failure worth *waiting on*: a freshly-rented box resets or refuses
    the first handshakes while sshd comes up, then answers — the caller retries
    those instead of giving up on the first reset. Auth failures and a missing
    ssh binary are permanent — waiting never fixes them."""
    rc, out, err = run(cargs, "echo KARL_OK", timeout=30)
    if rc == 0 and "KARL_OK" in out:
        return True, "ok", False
    low = ((err or "") + (out or "")).lower()
    if rc == 127:
        return False, "ssh binary not found on this machine", False
    if "permission denied" in low or "no such identity" in low:
        return False, ("auth denied — the box isn't accepting your SSH key "
                       "(add it / ssh-agent)"), False
    # The freshly-booted-box signature: TCP connected, but sshd reset the
    # handshake before its banner. Almost always still coming up, or fail2ban
    # briefly rate-limiting — worth a wait, not a wrong key.
    if ("kex_exchange_identification" in low or "reset by peer" in low
            or "connection reset" in low):
        return False, ("the box reset the SSH handshake — almost always still "
                       "booting, or briefly rate-limiting new connections"), True
    if "connection refused" in low:
        return False, ("connection refused — sshd isn't up yet; the box may "
                       "still be booting"), True
    if rc == 124:
        return False, "no answer in 30s — wrong host/port, or the box is still booting", True
    return False, f"ssh failed: {(err or out).strip()[:160] or f'exit {rc}'}", False


# --------------------------------------------------------------------------
# detection and planning
# --------------------------------------------------------------------------
def detect_gpus(cargs: list):
    rc, out, err = run(cargs, "nvidia-smi --query-gpu=name,memory.total "
                              "--format=csv,noheader,nounits", timeout=30)
    if rc != 0:
        raise ProvisionError(f"nvidia-smi failed: {(err or out).strip()[:200]}")
    gpus = []
    for line in out.strip().splitlines():
        try:
            name, mem = line.rsplit(",", 1)
            gpus.append((name.strip(), int(float(mem.strip()))))
        except ValueError:
            continue
    return gpus


def plan(gpus: list, spec: ModelSpec):
    """-> (tensor_parallel, max_model_len, gpu_mem_util, total_gb). Raises when
    the box is too small for the chosen model."""
    if not gpus:
        raise ProvisionError("no GPUs detected on the box (nvidia-smi empty)")
    total_gb = sum(mb for _, mb in gpus) // 1024
    if total_gb < spec.min_total_gb:
        raise ProvisionError(
            f"only {total_gb}GB total VRAM — {spec.label} needs ~{spec.min_total_gb}GB+. "
            f"Pick a smaller model with `gpu model <key>`, or rent a bigger box.")
    max_len = spec.context_beyond
    for threshold, length in spec.context_tiers:
        if total_gb < threshold:
            max_len = length
            break
    util = 0.95 if total_gb < 72 else 0.92
    return len(gpus), max_len, util, total_gb


# --------------------------------------------------------------------------
# install + launch (the scars: apt locks, flaky mirrors, CUDA link paths)
# --------------------------------------------------------------------------
_APT_WAIT = (
    "apt_wait() { for _i in $(seq 1 60); do apt-get \"$@\" 2>/tmp/.karl_apt && return 0; "
    "grep -q 'Could not get lock\\|is held by process' /tmp/.karl_apt || "
    "{ cat /tmp/.karl_apt >&2; return 1; }; sleep 5; done; cat /tmp/.karl_apt >&2; return 1; }; "
)
_NET_WAIT = (
    "net_wait() { for _i in $(seq 1 24); do \"$@\" 2>/tmp/.karl_net && return 0; "
    "grep -qiE 'could not resolve host|temporary failure in name resolution|network is unreachable|"
    "could not connect to|connection timed out' /tmp/.karl_net || "
    "{ cat /tmp/.karl_net >&2; return 1; }; sleep 5; done; cat /tmp/.karl_net >&2; return 1; }; "
)
# `ssh host "command"` runs non-interactively, so .bashrc's CUDA exports (PATH,
# LD_LIBRARY_PATH — where the box's real, versioned libcublas/libcudart live)
# never apply: cmake still finds nvcc, but the linker falls back to broken stubs
# and dies with "undefined reference to ...@libcublas.so.NN". Find the real
# toolkit dirs ourselves and export them, whatever the shell init did or didn't.
_CUDA_ENV = (
    "for _d in /usr/local/cuda*/lib64 /usr/local/cuda*/targets/*/lib; do "
    "[ -d \"$_d\" ] && export LD_LIBRARY_PATH=\"$_d:$LD_LIBRARY_PATH\" "
    "LIBRARY_PATH=\"$_d:$LIBRARY_PATH\"; done; "
    "for _b in /usr/local/cuda*/bin; do [ -d \"$_b\" ] && export PATH=\"$_b:$PATH\"; done; "
)


def _vllm_cmd(spec: ModelSpec, tp: int, max_len: int, util: float, port: int) -> str:
    parts = [VLLM_BIN, "serve", spec.repo,
             f"--served-model-name {spec.served_name}",
             f"--quantization {spec.quantization}",
             f"--tensor-parallel-size {tp}",
             f"--max-model-len {max_len}",
             f"--gpu-memory-utilization {util}",
             f"--enable-auto-tool-choice --tool-call-parser {spec.tool_call_parser}",
             "--host 127.0.0.1", f"--port {port}"]
    if spec.tokenizer:
        parts.append(f"--tokenizer {spec.tokenizer}")
    return " ".join(parts)


def _llama_cmd(spec: ModelSpec, max_len: int, port: int) -> str:
    if spec.gguf_file:
        weights = [f"--hf-repo {spec.repo}", f"--hf-file {spec.gguf_file}"]
    else:
        weights = [f"-hf {spec.repo}:{spec.gguf_quant}"]
    return " ".join([LLAMA_BIN, *weights, f"--alias {spec.served_name}",
                     "--host 127.0.0.1", f"--port {port}",
                     f"--ctx-size {max_len}", "--n-gpu-layers 999", "--jinja"])


def _install_vllm(cargs: list, log) -> None:
    log(ui.dim("  installing vLLM (first time can take several minutes)…"))
    cmd = (_APT_WAIT +
           f"test -x {VLLM_BIN} && exit 0; "
           f"python3 -m venv --system-site-packages {VENV_DIR} 2>/dev/null || "
           f"{{ apt_wait update -qq && apt_wait install -y -qq python3-venv && "
           f"python3 -m venv --system-site-packages {VENV_DIR}; }} && "
           f"{VENV_DIR}/bin/pip install -q -U pip vllm hf_transfer")
    rc, _, err = run(cargs, cmd, timeout=1800)
    if rc != 0:
        raise ProvisionError(f"vLLM install failed: {err.strip()[-500:]}")


def _install_llama(cargs: list, log) -> None:
    log(ui.dim("  building llama.cpp with CUDA (first time can take several minutes)…"))
    cmd = (_APT_WAIT + _NET_WAIT +
           f"test -x {LLAMA_BIN} && exit 0; mkdir -p {LLAMA_DIR} && "
           "apt_wait update -qq && apt_wait install -y -qq git cmake build-essential "
           "libcurl4-openssl-dev && "
           f"rm -rf {LLAMA_DIR}/src && "
           f"net_wait git clone --depth 1 {LLAMA_REPO} {LLAMA_DIR}/src && "
           "CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader "
           "2>/dev/null | head -1 | tr -d '. '); " + _CUDA_ENV +
           f"cmake -S {LLAMA_DIR}/src -B {LLAMA_DIR}/src/build -DGGML_CUDA=ON "
           "-DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=Release "
           "${CUDA_ARCH:+-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH} && "
           f"cmake --build {LLAMA_DIR}/src/build --config Release -j --target llama-server && "
           f"cp {LLAMA_DIR}/src/build/bin/llama-server {LLAMA_BIN}")
    rc, _, err = run(cargs, cmd, timeout=3600)
    if rc != 0:
        raise ProvisionError(f"llama.cpp build failed: {err.strip()[-500:]}")


def _register_cuda_libs(cargs: list) -> None:
    """Teach the box's dynamic linker where the CUDA toolkit lives, once and
    for all. Exporting LD_LIBRARY_PATH only helps the one shell that launches —
    but a crash-on-exec ("libcudart.so.NN: cannot open shared object file")
    strikes any session that forgets to. Writing the toolkit dirs into
    ld.so.conf and running ldconfig fixes it system-wide. Idempotent, cheap,
    best-effort (needs root, which a rented box gives)."""
    run(cargs, "ls -d /usr/local/cuda*/lib64 /usr/local/cuda*/targets/*/lib "
               "2>/dev/null > /etc/ld.so.conf.d/cuda-karl.conf && ldconfig", timeout=30)


def server_running(cargs: list) -> bool:
    rc, out, _ = run(cargs, f"cat {PID_FILE} 2>/dev/null && kill -0 $(cat {PID_FILE}) "
                            "2>/dev/null && echo RUNNING")
    return "RUNNING" in out


def _clear_stale_and_check_port(cargs: list, port: int) -> str:
    """Clear orphaned llama-servers and report if ``port`` is still held. We
    only get here with no tracked-alive server, so a leftover llama-server is a
    lost orphan from an earlier session — kill it. If the port is *still* taken
    afterwards it's another service on the box (vast.ai often squats on 8080):
    return what holds it so the caller can slide or fail clearly rather than
    bind-fail forever. Returns '' when the port is free."""
    run(cargs, "pkill -9 -f llama-server 2>/dev/null; sleep 1", timeout=20)
    rc, out, _ = run(cargs, f"ss -tln 2>/dev/null | grep ':{port} ' || true", timeout=20)
    return out.strip()


def launch(cargs: list, spec: ModelSpec, tp, max_len, util, port: int, log,
           auto_port: bool = False) -> int:
    """Install the runtime and launch the server. Returns the box-side port the
    server actually bound — with ``auto_port``, a held port slides to a free
    one (and the caller slides the tunnel forward with it) instead of failing."""
    if server_running(cargs):
        log(ui.yellow("  a model server is already running on the box "
                      "(gpu down to relaunch)."))
        return port
    if spec.server == "llama_cpp":
        _install_llama(cargs, log)
        _register_cuda_libs(cargs)  # so llama-server finds libcudart/libcublas on exec
        held = _clear_stale_and_check_port(cargs, port)
        if held and auto_port:
            # don't make the operator re-type the forward — slide the box-side
            # port ourselves and report where the server landed
            alt = port + 10000 if port < 55535 else 8000
            if _clear_stale_and_check_port(cargs, alt):
                raise ProvisionError(
                    f"ports {port} and {alt} are both held on the box — pick a "
                    "free remote port for your -L forward by hand.")
            log(ui.yellow(f"  port {port} is held on the box — sliding the "
                          f"server to {alt} (your local side stays as is)."))
            port = alt
        elif held:
            alt = port + 10000 if port < 55535 else 8000
            raise ProvisionError(
                f"port {port} on the box is already held by another service — "
                f"the server can never bind it. Re-run with a different remote "
                f"port in your -L forward, e.g. -L {port}:localhost:{alt}  "
                f"(your local {port} still works; only the box-side port moves).")
        cmd = _llama_cmd(spec, max_len, port)
    else:
        _install_vllm(cargs, log)
        cmd = _vllm_cmd(spec, tp, max_len, util, port)
    log(ui.dim(f"  launching: {cmd[:120]}…"))
    # Belt-and-suspenders with the ldconfig registration above: the build's
    # CUDA env lived only in that ssh session — this is a separate one. A
    # dynamically-linked llama-server that can't find libcublas dies on exec,
    # silently, before it opens a socket — which looks exactly like a hung
    # download bar.
    env_setup = _CUDA_ENV if spec.server == "llama_cpp" else ""
    rc, _, err = run(cargs, env_setup + "HF_HUB_ENABLE_HF_TRANSFER=1 nohup " + cmd +
                     f" > {LOG_FILE} 2>&1 & echo $! > {PID_FILE}", timeout=60)
    if rc != 0:
        raise ProvisionError(f"launch failed: {err.strip()[-400:]}")
    return port


def stop(cargs: list) -> None:
    run(cargs, f"kill $(cat {PID_FILE}) 2>/dev/null; rm -f {PID_FILE}", timeout=30)


# --------------------------------------------------------------------------
# readiness: poll until the weights load and the endpoint answers
# --------------------------------------------------------------------------
def _cache_bytes(cargs: list):
    rc, out, _ = run(cargs, "du -sb ~/.cache/llama.cpp ~/.cache/huggingface 2>/dev/null "
                            "| awk '{s+=$1} END{print s+0}'", timeout=20)
    try:
        return int(out.strip()) if rc == 0 else None
    except ValueError:
        return None


def _weights_total(spec: ModelSpec):
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*GB", spec.weights_note or "")
    return int(float(m.group(1)) * 1_000_000_000) if m else None


def _fmt(n: int) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def endpoint_up(local_port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{local_port}/v1/models",
                                    timeout=5) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def wait_ready(cargs: list, local_port: int, spec: ModelSpec, log,
               deadline_s: int = 2400) -> bool:
    """Poll until the endpoint answers, painting a download bar while weights
    land. A crash on exec kills the process in under a second but looks, from
    here, identical to a slow download — the bar just sits at 0%. Rather than
    silently burning rental time on a corpse, check the process is still alive
    once warm-up has had a moment, and bail with the crash reason the instant
    it isn't."""
    total = _weights_total(spec)
    start = time.time()
    baseline = None
    tty = sys.stdout.isatty()
    log(ui.dim(f"  waiting for the model to wake — {spec.weights_note}"))
    while time.time() - start < deadline_s:
        if endpoint_up(local_port):
            _wipe(tty)
            return True
        if time.time() - start > 8 and not server_running(cargs):
            _wipe(tty)
            rc, out, _ = run(cargs, f"tail -n 15 {LOG_FILE} 2>/dev/null", timeout=20)
            log(ui.red("  ✗ the server process died before it came up."))
            for ln in (out.strip().splitlines() if rc == 0 and out.strip() else []):
                log(ui.dim("  | " + ln[:200]))
            return False
        cur = _cache_bytes(cargs)
        if cur is not None and baseline is None:
            baseline = cur
        if cur is not None and total and tty:
            got = max(0, cur - (baseline or 0))
            frac = min(0.99, got / total)
            sys.stdout.write("\r  " + ui.cyan(ui.bar(frac, label=f"weights {_fmt(got)}/{_fmt(total)}")))
            sys.stdout.flush()
        else:
            # no total or piped: stream a few fresh log lines so warm-up shows
            rc, out, _ = run(cargs, f"tail -n 3 {LOG_FILE} 2>/dev/null", timeout=20)
            if rc == 0 and out.strip():
                for ln in out.strip().splitlines()[-3:]:
                    log(ui.dim("  | " + ln[:150]))
        time.sleep(4)
    _wipe(tty)
    return False


def _wipe(tty: bool) -> None:
    if tty:
        sys.stdout.write("\r" + " " * 72 + "\r")
        sys.stdout.flush()


# --------------------------------------------------------------------------
# the tunnel (detached, PID-tracked) and box state
# --------------------------------------------------------------------------
def _state_path():
    return karl_home() / "gpu.json"


def _load() -> dict:
    return load_json(_state_path(), {})


def _save(state: dict) -> None:
    save_json(_state_path(), state)


def _alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def _kill_tunnel(state: dict) -> None:
    pid = state.get("tunnel_pid")
    if _alive(pid):
        try:
            os.kill(int(pid), signal.SIGTERM)
        except OSError:
            pass
    state["tunnel_pid"] = None


def _open_tunnel(ssh_args: list, out) -> int | None:
    """Launch a detached ``ssh -N …`` tunnel. Returns its PID, or None."""
    logf = karl_home() / "gpu-tunnel.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ssh", "-N",
           "-o", "StrictHostKeyChecking=accept-new",
           "-o", "BatchMode=yes",
           "-o", "ServerAliveInterval=30",
           # give up after ~90s of a dead link so the tunnel exits cleanly (its
           # PID dies) instead of zombie-ing — `gpu status` then reads it as down
           "-o", "ServerAliveCountMax=3",
           "-o", "ExitOnForwardFailure=yes",
           "-o", "ConnectTimeout=15"] + ssh_args
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL,
                                stderr=logf.open("w"), start_new_session=True)
    except FileNotFoundError:
        out(ui.red("  ssh not found on PATH."))
        return None
    # give it a moment to fail fast (bad forward, auth) before we trust it
    for _ in range(25):
        if proc.poll() is not None:
            tail = ""
            try:
                tail = logf.read_text().strip().splitlines()[-1][:200]
            except (OSError, IndexError):
                pass
            out(ui.red("  ⛓  tunnel failed to come up.") + (ui.dim("  " + tail) if tail else ""))
            return None
        time.sleep(0.1)
    return proc.pid


# --------------------------------------------------------------------------
# the command
# --------------------------------------------------------------------------
def handle(rest: str, out=print) -> None:
    parts = rest.split()
    sub = parts[0].lower() if parts else "status"
    state = _load()

    if sub == "ssh":
        ssh_args = parts[1:]
        # forgive a pasted full ssh line: `gpu ssh ssh -p … host -L …` — the
        # leading `ssh` is redundant and would otherwise read as the hostname
        while ssh_args and ssh_args[0] == "ssh":
            ssh_args = ssh_args[1:]
        fwd = parse_forward(ssh_args)
        if not fwd:
            out(ui.yellow("usage: karl gpu ssh <ssh args including -L localport:host:remoteport>"))
            out(ui.dim("  e.g. karl gpu ssh -p 24439 root@1.2.3.4 -L 8080:localhost:8080"))
            return
        local_port, _rhost, rport = fwd
        cargs = conn_args(ssh_args)
        spec = get_spec(state.get("model_id"))

        # 1. reach the box, waiting out the transient resets of a booting box
        out(ui.dim("  reaching the box…"))
        ok, why, transient = check_connection(cargs)
        tries = 0
        while not ok and transient and tries < 4:
            tries += 1
            out(ui.dim(f"  {why} — still coming up; retry {tries}/4 in 12s (Ctrl-C to stop)"))
            time.sleep(12)
            ok, why, transient = check_connection(cargs)
        if not ok:
            out(ui.red("  can't reach the box: ") + ui.dim(why))
            return

        # 2. detect GPUs + plan the tier
        try:
            gpus = detect_gpus(cargs)
            tp, max_len, util, total_gb = plan(gpus, spec)
        except ProvisionError as e:
            out(ui.red("  " + str(e)))
            return
        out(ui.green(f"  {len(gpus)}× GPU · {total_gb}GB VRAM")
            + ui.dim(f"  ({', '.join(n for n, _ in gpus)})"))
        out(ui.dim(f"  serving {spec.label} · context {max_len} · port {rport}"))

        # 3. install runtime + launch server (slides off a squatted box port)
        try:
            new_rport = launch(cargs, spec, tp, max_len, util, rport, out,
                               auto_port=True)
        except ProvisionError as e:
            out(ui.red("  " + str(e)))
            return
        if new_rport != rport:
            ssh_args = replace_forward(ssh_args, new_rport)
            out(ui.dim(f"  tunnel follows the slide → -L {local_port}:localhost:{new_rport}"))

        # 4. open the tunnel (detached, survives this command)
        _kill_tunnel(state)
        pid = _open_tunnel(ssh_args, out)
        if pid is None:
            return

        # 5. wait for the weights to load and the endpoint to answer
        ready = wait_ready(cargs, local_port, spec, out)
        base_url = f"http://localhost:{local_port}/v1"
        state.update(base_url=base_url, served=True, model=spec.served_name,
                     model_id=spec.key, ssh_conn=cargs, local_port=local_port,
                     remote_port=new_rport, tunnel_pid=pid)
        _save(state)
        # point the whole harness at it
        set_config(base_url=base_url, model=spec.served_name)
        if ready:
            out(ui.green(f"  ⛓  the model is up at {base_url}")
                + ui.dim(f"  (model: {spec.served_name}). Try `karl ping`, then `karl`."))
        else:
            out(ui.yellow("  tunnel up and the server is launching, but it didn't "
                          "answer in time — weights may still be loading."))
            out(ui.dim("     check with `karl gpu status` or `karl ping` in a few minutes."))

    elif sub in ("model", "models"):
        if len(parts) < 2:
            cur = state.get("model_id", DEFAULT_KEY)
            for k, s in CATALOG.items():
                mark = ui.green("→") if k == cur else " "
                out(f"  {mark} {ui.cyan(k):16} {ui.dim(s.label)}")
            out(ui.dim("  pick:  karl gpu model <key>   (served on the next `karl gpu ssh …`)"))
            return
        key = parts[1]
        if key not in CATALOG:
            out(ui.yellow(f"unknown model '{key}' — one of: {', '.join(CATALOG)}"))
            return
        spec = get_spec(key)
        state.update(model_id=key, model=spec.served_name)
        _save(state)
        if state.get("served"):
            set_config(model=spec.served_name)
        out(ui.green(f"  model → {spec.label}"))
        out(ui.dim(f"  needs ~{spec.min_total_gb}GB VRAM · {spec.weights_note}"))

    elif sub == "reconnect":
        cargs = state.get("ssh_conn")
        lp, rp = state.get("local_port"), state.get("remote_port")
        if not cargs or not lp or not rp:
            out(ui.yellow("  nothing to reconnect to — run `karl gpu ssh <ssh… -L "
                          "port:host:port>` first."))
            return
        spec = get_spec(state.get("model_id"))
        _kill_tunnel(state)
        out(ui.dim("  reopening the tunnel…"))
        pid = _open_tunnel(list(cargs) + ["-L", f"{lp}:localhost:{rp}"], out)
        if pid is None:
            return
        ready = wait_ready(cargs, lp, spec, out)
        base_url = f"http://localhost:{lp}/v1"
        state.update(tunnel_pid=pid, served=True, base_url=base_url)
        _save(state)
        set_config(base_url=base_url, model=spec.served_name)
        out(ui.green(f"  ⛓  reconnected at {base_url}.") if ready else
            ui.yellow("  tunnel reopened, but the server didn't answer yet — it "
                      "may still be waking, or the box is gone. Try `karl ping`, "
                      "or re-run `karl gpu ssh …`."))

    elif sub in ("test", "ping"):
        from karl.cli import cmd_ping
        cmd_ping()

    elif sub == "serve":  # point at an already-reachable url by hand
        if len(parts) < 2:
            out(ui.yellow("usage: karl gpu serve <base_url> [model]"))
            return
        base_url = parts[1].rstrip("/")
        model = parts[2] if len(parts) > 2 else state.get("model", "local")
        state.update(base_url=base_url, served=True, model=model)
        _save(state)
        set_config(base_url=base_url, model=model)
        out(ui.green(f"  pointed at {base_url} (model: {model})."))

    elif sub == "down":  # stop the server on the box AND drop the tunnel
        cargs = state.get("ssh_conn")
        if cargs:
            out(ui.dim("  stopping the model server on the box…"))
            stop(cargs)
        _kill_tunnel(state)
        state["served"] = False
        _save(state)
        set_config(base_url="")
        out(ui.dim("  server stopped, tunnel down — the harness falls back offline."))

    elif sub in ("off", "detach"):  # drop the tunnel; leave the server running
        _kill_tunnel(state)
        state["served"] = False
        _save(state)
        set_config(base_url="")
        out(ui.dim("  tunnel down — offline. (server left running; `karl gpu down` stops it.)"))

    else:  # status
        if state.get("served"):
            if _alive(state.get("tunnel_pid")):
                live = ui.green("tunnel live")
            else:
                live = ui.yellow("tunnel down — `karl gpu reconnect`")
            out(ui.dim(f"  served: {state.get('base_url')} (model: {state.get('model')}) — ") + live)
        else:
            out(ui.dim("  no GPU attached. `karl gpu ssh <ssh… -L port:host:port>` to serve a model."))
