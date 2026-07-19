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


# --------------------------------------------------------------------------
# the project sandbox image — how the crew's toolbox grows
#
# `apt_install` bakes Debian (and pip) packages into a per-project image the
# shell then uses. Build-time gets the network (that's what installing means);
# run-time stays network-off. The recorded package set is the source of truth:
# every build starts FROM the base image with the full union, so the image is
# reproducible and `sandbox reset` returns to a clean slate.
# --------------------------------------------------------------------------
def sandbox_image_name(project) -> str:
    return f"karl-sandbox-{project.name}"


def sandbox_record(project) -> dict:
    from karl.config import load_json
    return load_json(project.root / "sandbox.json", {"apt": [], "pip": []})


def image_for(project) -> str:
    """The image run_shell should use: the project's baked sandbox when
    packages have been installed, the stock base otherwise."""
    if project is None:
        return shell_image()
    rec = sandbox_record(project)
    if rec.get("apt") or rec.get("pip"):
        return sandbox_image_name(project)
    return shell_image()


def sandbox_dockerfile(base: str, apt_pkgs: list, pip_pkgs: list) -> str:
    lines = [f"FROM {base}"]
    if apt_pkgs:
        lines.append("RUN apt-get update && apt-get install -y "
                     "--no-install-recommends " + " ".join(apt_pkgs)
                     + " && rm -rf /var/lib/apt/lists/*")
    if pip_pkgs:
        lines.append("RUN pip install --no-cache-dir " + " ".join(pip_pkgs))
    return "\n".join(lines) + "\n"


def build_sandbox(project, runtime: str, apt_pkgs: list, pip_pkgs: list,
                  timeout: int = 900):
    """Bake the union of everything ever installed into the project image.
    Returns (rc, err_tail). On success the record is updated on disk."""
    from karl.config import save_json
    rec = sandbox_record(project)
    all_apt = sorted(set(rec.get("apt", [])) | set(apt_pkgs))
    all_pip = sorted(set(rec.get("pip", [])) | set(pip_pkgs))
    df = sandbox_dockerfile(shell_image(), all_apt, all_pip)
    try:
        p = subprocess.run([runtime, "build", "-t", sandbox_image_name(project), "-"],
                           input=df, capture_output=True, text=True,
                           errors="replace", timeout=timeout)
    except FileNotFoundError:
        return 127, f"{runtime} not found on PATH"
    except subprocess.TimeoutExpired:
        return 124, f"build timed out after {timeout}s"
    if p.returncode != 0:
        return p.returncode, (p.stderr or p.stdout or "").strip()[-1500:]
    save_json(project.root / "sandbox.json", {"apt": all_apt, "pip": all_pip})
    return 0, ""


def remove_sandbox(project, runtime: str) -> None:
    """Drop the baked image and the record — back to the stock base."""
    subprocess.run([runtime, "rmi", "-f", sandbox_image_name(project)],
                   capture_output=True, timeout=60)
    p = project.root / "sandbox.json"
    if p.exists():
        p.unlink()


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
