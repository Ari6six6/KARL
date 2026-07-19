"""The chassis — paths, project state, the engine endpoint, the web allowlist.

KARL keeps its home under ``$KARL_HOME`` (default ``~/.karl``). A *project* is
one working world on disk: a shared workspace, saved transcripts, durable notes,
and its own web allowlist. All of it is plain files and JSON — inspect it, edit
it, delete it.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse


def karl_home() -> Path:
    return Path(os.environ.get("KARL_HOME", str(Path.home() / ".karl")))


def config_path() -> Path:
    return karl_home() / "config.json"


def projects_root() -> Path:
    return karl_home() / "projects"


# --------------------------------------------------------------------------
# JSON helpers (atomic write — a crash never leaves a torn file)
# --------------------------------------------------------------------------
def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# the engine endpoint
# --------------------------------------------------------------------------
def endpoint() -> dict:
    """How to reach the model. Environment beats the config file, so one run can
    point at a different server without editing anything:

        KARL_BASE_URL   e.g. http://localhost:8080/v1  (vLLM, llama.cpp, Ollama…)
        KARL_MODEL      the model name the server expects
        KARL_API_KEY    if the server wants one (default "-")

    No base URL anywhere → KARL idles on the built-in offline stand-in, so the
    crew still moves on a fresh clone.
    """
    cfg = load_json(config_path(), {})
    base = os.environ.get("KARL_BASE_URL") or cfg.get("base_url") or ""
    # the shell is ON by default, in its sandboxed form — a crew that cannot
    # build or test is a crew that stalls. "off" remains one config away.
    shell = os.environ.get("KARL_SHELL") or cfg.get("shell") or "container"
    if shell not in ("off", "container", "host"):
        shell = "container"
    installs = cfg.get("installs", "ask")
    if installs not in ("ask", "open", "off"):
        installs = "ask"

    def _clamped(key, default, lo, hi):
        try:
            return max(lo, min(hi, int(cfg.get(key, default))))
        except (TypeError, ValueError):
            return default

    return {
        "max_steps": _clamped("max_steps", 40, 4, 200),
        "max_turns": _clamped("max_turns", 24, 2, 64),
        "base_url": base.rstrip("/"),
        "model": os.environ.get("KARL_MODEL") or cfg.get("model") or "local",
        "api_key": os.environ.get("KARL_API_KEY") or cfg.get("api_key") or "-",
        "temperature": float(cfg.get("temperature", 0.6)),
        "max_tokens": int(cfg.get("max_tokens", 2048)),
        "timeout": float(cfg.get("timeout", 300)),
        "stream": cfg.get("stream", "on") != "off",
        "shell": shell,
        "shell_net": cfg.get("shell_net", "none"),
        "installs": installs,
    }


def set_config(**kwargs) -> dict:
    cfg = load_json(config_path(), {})
    cfg.update({k: v for k, v in kwargs.items() if v is not None})
    save_json(config_path(), cfg)
    return cfg


def web_open() -> bool:
    """True (default): the crew may fetch any public page, no per-domain leave.
    ``web = "gated"`` (or KARL_WEB=gated) falls back to the allowlist. The SSRF
    guard runs either way — loopback, LAN, and metadata addresses never open."""
    cfg = load_json(config_path(), {})
    mode = os.environ.get("KARL_WEB") or cfg.get("web") or "open"
    return mode != "gated"


# --------------------------------------------------------------------------
# projects
# --------------------------------------------------------------------------
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def valid_project_name(name: str) -> bool:
    return bool(_NAME.match(name or ""))


def current_project_file() -> Path:
    return karl_home() / "current_project"


def current_project_name() -> str:
    f = current_project_file()
    if f.exists():
        name = f.read_text().strip()
        if name:
            return name
    return "default"


def use_project(name: str) -> None:
    karl_home().mkdir(parents=True, exist_ok=True)
    current_project_file().write_text(name.strip() + "\n")


class Project:
    """One working world on disk."""

    def __init__(self, name: str):
        self.name = name
        self.root = projects_root() / name

    def ensure(self) -> "Project":
        for sub in ("workspace", "sessions"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        return self

    @property
    def workspace(self) -> Path:
        # ``KARL_WORKSPACE`` (or ``karl -C DIR``) points the crew at a real
        # directory instead of the managed one; metadata (sessions, notes,
        # allowlist) stays under the project root either way.
        override = os.environ.get("KARL_WORKSPACE")
        if override:
            return Path(override).expanduser().resolve()
        return self.root / "workspace"

    @property
    def notes_path(self) -> Path:
        # plain markdown the crew appends to and reads back next session
        return self.root / "notes.md"

    def notes(self) -> str:
        p = self.notes_path
        return p.read_text().strip() if p.exists() else ""

    @property
    def board_path(self) -> Path:
        # the pit board: the crew's live plan — goal, tasks, status. Rewritten
        # by the chief as work progresses; shown to every agent every turn.
        return self.root / "board.md"

    def board(self) -> str:
        p = self.board_path
        return p.read_text().strip() if p.exists() else ""

    def session_path(self, stamp: str) -> Path:
        return self.root / "sessions" / f"{stamp}.jsonl"

    # --- web allowlist (consulted only in gated mode) --------------------
    @property
    def allow_path(self) -> Path:
        return self.root / "allow.json"

    def allowlist(self) -> list:
        return load_json(self.allow_path, {"domains": []}).get("domains", [])

    def egress_allowed(self, domain: str) -> bool:
        al = self.allowlist()
        return "*" in al or domain in al

    _HOSTNAME = re.compile(
        r"^(?=.{1,253}$)([a-z0-9]([a-z0-9-]*[a-z0-9])?\.)*[a-z0-9]([a-z0-9-]*[a-z0-9])?$")

    @classmethod
    def normalize_domain(cls, domain: str) -> str:
        """Whatever the operator typed, stored as a bare lowercase hostname the
        fetcher can match — no scheme, no path, no port. Returns '' for input
        naming no host, so the allowlist never fills with junk."""
        d = (domain or "").strip().lower()
        if not d or d == "*":
            return d
        if "://" not in d:
            d = "//" + d
        host = (urlparse(d).hostname or "").rstrip(".")
        return host if cls._HOSTNAME.match(host) else ""

    def allow(self, domain: str) -> str:
        domain = self.normalize_domain(domain)
        if not domain:
            return ""
        data = load_json(self.allow_path, {"domains": []})
        domains = data.setdefault("domains", [])
        if domain not in domains:
            domains.append(domain)
        save_json(self.allow_path, data)
        return domain

    def disallow(self, domain: str) -> bool:
        domain = self.normalize_domain(domain)
        data = load_json(self.allow_path, {"domains": []})
        domains = data.get("domains", [])
        if domain in domains:
            domains.remove(domain)
            save_json(self.allow_path, data)
            return True
        return False


def load_project() -> Project:
    return Project(current_project_name()).ensure()
