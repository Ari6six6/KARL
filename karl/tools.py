"""The toolbox — what an agent can do beyond talking, and the rails around it.

Every tool returns a plain observation string the agent reads on its next step.
Two rails are enforced here, not merely requested:

  * File tools are sandboxed to the project workspace — a path that escapes it
    is refused. Long files page instead of truncating silently, and
    ``edit_file`` makes surgical replacements instead of rewriting whole files.
  * ``web_fetch`` is the only way out. Only agents marked ``can_egress`` carry
    it, it is SSRF-guarded (public web only — never the host's loopback, LAN,
    or a cloud metadata endpoint), and it takes a single hop: redirects are
    reported, not followed. Everything it returns is flagged ``tainted`` so a
    round that leaned on outside data says so when it reports back.

``run_shell`` is off unless the operator turns it on; ``container`` mode runs
each command in a disposable Docker/Podman container with only the workspace
mounted and no network, and refuses rather than silently touching the host when
no runtime is up.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


@dataclass
class ToolContext:
    workspace: Path
    project: object = None        # carries the allowlist (egress_allowed)
    can_egress: bool = False      # only agents marked so may reach the web
    web_open: bool = True         # True = any public site; False = allowlist only
    shell_mode: str = "off"       # "off" | "container" (sandboxed) | "host" (unsafe)
    shell_net: str = "none"       # container network: "none" (default) or "bridge"
    tainted: list = field(default_factory=list)   # domains fetched from outside
    changed: list = field(default_factory=list)   # files written this run
    ask: object = None            # operator consent hook: (question) -> bool, TTY only


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: object  # (args: dict, ctx: ToolContext) -> str

    def openai(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": self.parameters}}


# --------------------------------------------------------------------------
# file tools (sandboxed to the workspace)
# --------------------------------------------------------------------------
def _safe(ctx: ToolContext, rel: str) -> Path:
    ctx.workspace.mkdir(parents=True, exist_ok=True)
    p = (ctx.workspace / (rel or ".")).resolve()
    root = ctx.workspace.resolve()
    if root not in p.parents and p != root:
        raise ValueError(
            f"path escapes the workspace (which is {root}) — if the operator's "
            "files live elsewhere, the operator can re-point it: "
            "/workspace <dir>")
    return p


_READ_WINDOW = 8000        # chars per read; the rest is paged, never dropped
_READ_MAX_BYTES = 32_000_000


def _read_file(args, ctx):
    p = _safe(ctx, args.get("path", ""))
    if not p.exists():
        return f"ERROR: no such file: {args.get('path')}"
    if p.is_dir():
        return f"ERROR: {args.get('path')} is a directory — use list_dir"
    if p.stat().st_size > _READ_MAX_BYTES:
        return (f"ERROR: {args.get('path')} is {p.stat().st_size:,} bytes — too big "
                "to read whole. Narrow it with search, or page it with run_shell.")
    try:
        offset = max(0, int(args.get("offset", 0)))
    except (TypeError, ValueError):
        offset = 0
    text = p.read_text("utf-8", "replace")
    chunk = text[offset:offset + _READ_WINDOW]
    remaining = len(text) - (offset + len(chunk))
    if remaining > 0:
        chunk += (f"\n\n[TRUNCATED: {remaining:,} characters remain. Call read_file "
                  f"with offset={offset + _READ_WINDOW} to continue.]")
    return chunk


def _write_file(args, ctx):
    p = _safe(ctx, args.get("path", ""))
    p.parent.mkdir(parents=True, exist_ok=True)
    content = args.get("content", "")
    p.write_text(content)
    ctx.changed.append(args.get("path", "?"))
    return f"wrote {len(content)} chars to {args.get('path')}"


def _edit_file(args, ctx):
    """Surgical replacement: swap one exact occurrence of ``find`` for
    ``replace``. Refuses ambiguity — zero or several matches return an error
    instead of guessing, so an edit lands exactly where intended or not at all."""
    p = _safe(ctx, args.get("path", ""))
    if not p.exists() or p.is_dir():
        return f"ERROR: no such file: {args.get('path')}"
    find = args.get("find") or ""
    if not find:
        return "ERROR: 'find' is empty — nothing to match"
    text = p.read_text("utf-8", "replace")
    hits = text.count(find)
    if hits == 0:
        return ("ERROR: no match — 'find' must appear in the file exactly as "
                "written (check whitespace). read_file to see the current text.")
    if hits > 1:
        return (f"ERROR: 'find' matches {hits} times — include enough "
                "surrounding text to make it unique.")
    p.write_text(text.replace(find, args.get("replace") or "", 1))
    ctx.changed.append(args.get("path", "?"))
    return f"edited {args.get('path')} (1 replacement)"


def _list_dir(args, ctx):
    p = _safe(ctx, args.get("path", "."))
    if not p.exists():
        return "(empty)"
    return "\n".join(sorted(
        x.name + ("/" if x.is_dir() else "") for x in p.iterdir())) or "(empty)"


def _looks_binary(raw: bytes) -> bool:
    return b"\x00" in raw[:1024]


def _search(args, ctx):
    """Grep the workspace for a regex — find who calls a function or where a
    name is set without opening every file. Returns ``path:line: text``."""
    pattern = args.get("pattern", "")
    if not pattern:
        return "ERROR: no pattern"
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        return f"ERROR: bad regex: {e}"
    root = _safe(ctx, args.get("path", "."))
    if not root.exists():
        return f"ERROR: no such path: {args.get('path')}"
    base = ctx.workspace.resolve()
    results, capped = [], False
    for path in sorted(root.rglob("*") if root.is_dir() else [root]):
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > 1_000_000:
                continue
            raw = path.read_bytes()
            if _looks_binary(raw):
                continue
            text = raw.decode("utf-8", "replace")
        except OSError:
            continue
        rel = path.relative_to(base) if base in path.parents or path == base else path.name
        for i, ln in enumerate(text.splitlines(), 1):
            if compiled.search(ln[:4000]):    # bounded: a huge single line can't hang
                results.append(f"{rel}:{i}: {ln.strip()[:200]}")
                if len(results) >= 50:
                    capped = True
                    break
        if capped:
            break
    if not results:
        return "no matches"
    tail = "\n[capped at 50 matches — narrow the pattern]" if capped else ""
    return "\n".join(results) + tail


# --------------------------------------------------------------------------
# shell (opt-in: container by default, host only by explicit choice)
# --------------------------------------------------------------------------
def _host_shell(cmd: str, ctx, timeout: int) -> str:
    try:
        p = subprocess.run(cmd, shell=True, cwd=str(ctx.workspace),
                           capture_output=True, text=True, errors="replace",
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return "[timed out]"
    tail = (p.stdout or "") + (("\n" + p.stderr) if p.stderr.strip() else "")
    return f"[exit {p.returncode}]\n{tail[:4000]}" if tail.strip() else f"[exit {p.returncode}] (no output)"


def _run_shell(args, ctx):
    if ctx.shell_mode == "off":
        return ("DENIED: the operator turned the shell off. "
                "`karl config --shell container` re-enables the sandboxed default.")
    cmd = (args.get("command") or "").strip()
    if not cmd:
        return "ERROR: no command"
    ctx.workspace.mkdir(parents=True, exist_ok=True)
    timeout = int(args.get("timeout", 120))

    if ctx.shell_mode == "container":
        from karl.shell import probe_runtime, run_in_container
        rt = probe_runtime()
        if not rt:
            # no runtime — put the choice to the operator right now, on a TTY,
            # instead of stalling the crew behind a wall the operator can't see
            if ctx.ask and ctx.ask("no container runtime is running — allow an "
                                   "unsandboxed HOST shell for this session?"):
                return _host_shell(cmd, ctx, timeout)
            return ("DENIED: the sandboxed shell needs Docker or Podman running, "
                    "and none answered. The operator can start one, allow the "
                    "host shell (`karl config --shell host`), or answer yes "
                    "when asked at the prompt.")
        rc, out, err = run_in_container(ctx.workspace, cmd, runtime=rt,
                                        network=ctx.shell_net, timeout=timeout)
        tail = (out or "") + (("\n" + err) if err and err.strip() else "")
        return f"[exit {rc}]\n{tail[:4000]}" if tail.strip() else f"[exit {rc}] (no output)"

    # host mode — explicit, unsandboxed opt-in
    return _host_shell(cmd, ctx, timeout)


# --------------------------------------------------------------------------
# web_fetch (the one egress: SSRF-guarded, one hop, taint-marked)
# --------------------------------------------------------------------------
def _resolve_public(host: str):
    """Resolve ``host`` and judge every address. Returns (ips, reason): reason
    is '' when all addresses are public unicast, else why the rail refuses (any
    private / loopback / link-local / reserved / multicast address — the
    cloud-metadata and LAN-pivot surface).

    Known limit: this is check-then-connect, so a resolver that rebinds between
    the two lookups can slip an address past. It stops honest infrastructure
    and mistakes, not a hostile DNS server."""
    import ipaddress
    import socket
    ips = []
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as e:
        return [], f"could not resolve {host} ({type(e).__name__})"
    for info in infos:
        addr = info[4][0].split("%")[0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        ips.append(addr)
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            return ips, f"{host} resolves to a non-public address ({addr})"
    return ips, ""


def _open_one_hop(req, timeout: float):
    """Open a request without following redirects — the gate takes one hop. A
    3xx surfaces as an HTTPError so the destination is reported, and crossing to
    it takes its own pass through the guard (the classic SSRF pivot is a
    redirect to a metadata endpoint the guard never got to check)."""
    import urllib.request

    class RefuseRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    return urllib.request.build_opener(RefuseRedirect).open(req, timeout=timeout)


def _web_fetch(args, ctx):
    url = (args.get("url") or "").strip()
    if not url:
        return "ERROR: no url"
    if "://" not in url:
        url = "https://" + url
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"DENIED: only http(s) may cross, not {parsed.scheme!r}."
    domain = (parsed.hostname or "").lower()
    if not ctx.can_egress:
        return "DENIED: this agent may not reach the web. Ask the one that can."
    # The public web is open by default. Only in gated mode is the allowlist
    # consulted. The SSRF guard below always runs and never asks.
    if not ctx.web_open and (ctx.project is None or not ctx.project.egress_allowed(domain)):
        return (f"DENIED: web is gated and {domain} isn't on the allowlist. Allow "
                f"it (`karl allow {domain}`) or open the web (`karl config --web open`).")
    ips, blocked = _resolve_public(domain)
    if blocked:
        return f"DENIED: {blocked} — the gate does not open onto private networks."
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "KARL/1.0"})
    try:
        with _open_one_hop(req, timeout=15) as r:
            return _deliver(r.status, domain, r.read(4000), ctx)
    except urllib.error.HTTPError as e:
        if 300 <= e.code < 400:
            where = e.headers.get("Location", "(no Location given)")
            return (f"DENIED: {domain} answered {e.code} redirecting to {where} — "
                    "the gate does not follow redirects. Fetch that destination "
                    "yourself and it passes the same guard.")
        try:
            raw = e.read(4000)
        except Exception:  # noqa: BLE001
            raw = b""
        return _deliver(e.code, domain, raw, ctx)
    except Exception as e:  # noqa: BLE001
        return f"ERROR reaching {domain}: {type(e).__name__}"


def _deliver(status: int, domain: str, raw: bytes, ctx) -> str:
    body = raw.decode("utf-8", "replace")
    ctx.tainted.append(domain)
    return f"[{status}] {domain} — {len(body)} bytes (TAINTED — from outside):\n{body[:2000]}"


# --------------------------------------------------------------------------
# remember (append a durable note to the project — long-term memory)
# --------------------------------------------------------------------------
def _remember(args, ctx):
    note = (args.get("note") or "").strip()
    if not note:
        return "ERROR: nothing to remember"
    if ctx.project is None:
        return "ERROR: no project to remember in"
    import time
    p = ctx.project.notes_path
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(f"- ({time.strftime('%Y-%m-%d')}) {note}\n")
    return f"remembered: {note}"


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------
TOOLBOX = {
    "read_file": Tool(
        "read_file", "Read a file in the workspace. Pass offset to page past a "
        "truncation notice and read further.",
        {"type": "object", "properties": {"path": {"type": "string"},
                                          "offset": {"type": "integer"}},
         "required": ["path"]}, _read_file),
    "write_file": Tool(
        "write_file", "Write a file in the workspace (overwrites). For a small "
        "change to an existing file, prefer edit_file.",
        {"type": "object", "properties": {"path": {"type": "string"},
                                          "content": {"type": "string"}},
         "required": ["path", "content"]}, _write_file),
    "edit_file": Tool(
        "edit_file", "Replace one exact occurrence of 'find' with 'replace' in a "
        "workspace file. 'find' must match exactly once — include surrounding "
        "text to disambiguate.",
        {"type": "object", "properties": {"path": {"type": "string"},
                                          "find": {"type": "string"},
                                          "replace": {"type": "string"}},
         "required": ["path", "find", "replace"]}, _edit_file),
    "list_dir": Tool(
        "list_dir", "List a directory in the workspace.",
        {"type": "object", "properties": {"path": {"type": "string"}}}, _list_dir),
    "search": Tool(
        "search", "Search the workspace for a regex pattern (returns path:line: "
        "text). Find where something is defined or used without opening every file.",
        {"type": "object", "properties": {"pattern": {"type": "string"},
                                          "path": {"type": "string"}},
         "required": ["pattern"]}, _search),
    "run_shell": Tool(
        "run_shell", "Run a shell command against the workspace. Off unless the "
        "operator enabled it; when on, it runs in an isolated container (no host "
        "access, no network by default).",
        {"type": "object", "properties": {"command": {"type": "string"},
                                          "timeout": {"type": "integer"}},
         "required": ["command"]}, _run_shell),
    "web_fetch": Tool(
        "web_fetch", "Fetch a URL from the public web. Only for agents allowed "
        "onto the web; private addresses never open.",
        {"type": "object", "properties": {"url": {"type": "string"}},
         "required": ["url"]}, _web_fetch),
    "remember": Tool(
        "remember", "Save a durable one-line note to the project's memory (kept "
        "between sessions and shown to the crew next time). Use for hard-won "
        "facts worth not relearning.",
        {"type": "object", "properties": {"note": {"type": "string"}},
         "required": ["note"]}, _remember),
}


def build_tools(names: list, ctx: ToolContext) -> list:
    """The tools an agent actually gets: the named subset, minus web_fetch when
    it can't egress."""
    out = []
    for name in names:
        t = TOOLBOX.get(name)
        if t is None:
            continue
        if name == "web_fetch" and not ctx.can_egress:
            continue
        out.append(t)
    return out


def execute(tools: list, call, ctx: ToolContext) -> str:
    by_name = {t.name: t for t in tools}
    t = by_name.get(call.name)
    if t is None:
        return f"ERROR: no such tool '{call.name}'"
    try:
        args = json.loads(call.arguments or "{}")
    except json.JSONDecodeError:
        return "ERROR: arguments were not valid JSON"
    try:
        return t.fn(args, ctx)
    except Exception as e:  # noqa: BLE001
        return f"ERROR in {call.name}: {type(e).__name__}: {e}"
