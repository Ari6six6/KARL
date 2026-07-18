"""Container isolation for the shell tool.

With Docker or Podman available, ``run_shell`` executes in a disposable
container with only the workspace mounted and no network by default — a model
driving the shell can't touch the host's files, keys, or network. With no
runtime present the shell stays off unless the operator explicitly opts into an
unsandboxed host shell; this module never pretends isolation is there when it
isn't.

Deliberately small: one probe, one run, ``docker run --rm`` each time. No image
builds, no long-lived containers, no bookkeeping.
"""

from __future__ import annotations

import os
import subprocess

DEFAULT_IMAGE = "python:3.12-slim"


def shell_image() -> str:
    return os.environ.get("KARL_SHELL_IMAGE", DEFAULT_IMAGE)


def probe_runtime() -> str:
    """'docker' or 'podman' if one is present AND its daemon answers, else ''.

    ``<rt> version`` exits non-zero when the client exists but the daemon is
    unreachable, so this reports a *usable* runtime, not merely an installed
    binary."""
    for rt in ("docker", "podman"):
        try:
            p = subprocess.run([rt, "version"], capture_output=True, timeout=15)
            if p.returncode == 0:
                return rt
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
    return ""


def run_in_container(workspace, command: str, *, runtime: str, network: str = "none",
                     image: str | None = None, timeout: int = 120):
    """Run ``command`` in a throwaway container with ``workspace`` mounted at
    /work. Returns (rc, stdout, stderr). Network is off by default; a build
    that needs the internet takes the operator's leave (``--shell-net bridge``).
    """
    ws = str(workspace.resolve())
    image = image or shell_image()
    cmd = [runtime, "run", "--rm",
           "-v", f"{ws}:/work", "-w", "/work",
           "--network", network,
           "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
           "--memory", "1g", "--cpus", "2", "--pids-limit", "512",
           image, "sh", "-lc", command]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                           timeout=timeout)
    except FileNotFoundError:
        return 127, "", f"{runtime} not found on PATH"
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"
    return p.returncode, p.stdout, p.stderr
