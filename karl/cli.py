"""``karl`` — the cockpit.

Run ``karl`` with no arguments for the interactive shell: type a task and the
crew works on it, live; slash-commands configure things. Or run one task
headless:

    karl run "summarize the files in the workspace"

Configure the engine once (or via KARL_BASE_URL / KARL_MODEL env vars):

    karl config --base-url http://localhost:8080/v1 --model my-model
"""

from __future__ import annotations

import sys
from pathlib import Path

from karl import ui
from karl.config import (Project, current_project_name, endpoint, load_project,
                         projects_root, set_config, use_project,
                         valid_project_name, web_open)
from karl.crew import load_crew, write_default_crew
from karl.session import Session

BANNER = f"""{ui.bold(ui.cyan('  KARL'))} {ui.dim('— a small, fast multi-agent harness for local models')}
{ui.dim('  type a task and the crew works on it, live · /help for commands · /quit to leave')}
"""

HELP = f"""{ui.bold('Commands')}
  {ui.cyan('<text>')}            give the crew a task
  {ui.cyan('/dash')}             the dashboard — project, engine, rails, crew, at a glance
  {ui.cyan('/agents')}           list the crew and who can reach the web
  {ui.dim('(the web is open by default — the crew can browse any public site freely)')}
  {ui.cyan('/allow')} <domain>   (only if you ran `config --web gated`) allowlist a domain
  {ui.cyan('/deny')} <domain>    remove a domain from the allowlist
  {ui.cyan('/note')} <text>      add a durable project note (memory across sessions)
  {ui.cyan('/notes')}            show the project notes
  {ui.cyan('/model')}            show how KARL reaches the model
  {ui.cyan('/gpu')} ssh <ssh…>   provision + serve a model on a GPU box, in one command
  {ui.cyan('/gpu')} reconnect    reopen a dropped tunnel to the same box
  {ui.cyan('/gpu')} model/status/off/down   pick a model · check · drop tunnel · stop box
  {ui.cyan('/ping')}             check that the model endpoint actually answers
  {ui.cyan('/project')} [name]   show, switch, or create the current project
  {ui.cyan('/crew')} init        write an editable crew.json you can customize
  {ui.cyan('/help')}             this
  {ui.cyan('/quit')}             leave
"""


def _shell_label(cfg: dict) -> str:
    mode = cfg.get("shell", "off")
    if mode == "container":
        return f"sandboxed (net {cfg.get('shell_net', 'none')})"
    if mode == "host":
        return "HOST (unsandboxed)"
    return "off"


def _web_label() -> str:
    return "open (any public site)" if web_open() else "gated (allowlist only)"


def cmd_dash(project) -> None:
    """The dashboard: everything that matters, one glance."""
    import os
    cfg = endpoint()
    if cfg["base_url"]:
        engine = f"{cfg['model']} {ui.dim('@ ' + cfg['base_url'])}"
        if not cfg["stream"]:
            engine += ui.dim(" · stream off")
    elif os.environ.get("KARL_OFFLINE"):
        engine = ui.yellow("offline stand-in (KARL_OFFLINE) — canned theater")
    else:
        engine = ui.red("none") + ui.dim(" — attach one: karl gpu ssh <ssh…> · "
                                         "karl config --base-url <url>")
    crew = load_crew(project)
    names = " · ".join(
        ui.speaker_paint(a.name)(a.name) + (ui.dim("*") if a.can_egress else "")
        for a in crew)
    print(ui.dash([
        ("project", ui.bold(project.name)),
        ("engine", engine),
        ("web", _web_label()),
        ("shell", _shell_label(cfg)),
        ("crew", names + ui.dim("   (* can reach the web)")),
    ]))


def cmd_agents(project) -> None:
    for a in load_crew(project):
        web = ui.green(" web") if a.can_egress else ""
        print(f"  {ui.speaker_paint(a.name)(a.name):16}{web} {ui.dim(a.role)}")
        print(ui.dim(f"      tools: {', '.join(a.tools)}"))


def cmd_model() -> None:
    cfg = endpoint()
    if cfg["base_url"]:
        print(ui.dim(f"  model: {cfg['model']} @ {cfg['base_url']}"))
        print(ui.dim(f"  shell: {_shell_label(cfg)} · web: {_web_label()}"))
    else:
        print(ui.dim("  no model attached — nothing runs until there is one. "
                     "Reach a model with one of:"))
        print(ui.dim("    karl gpu ssh -p <port> root@<ip> -L 8080:localhost:8080   # rent+serve"))
        print(ui.dim("    karl config --base-url http://localhost:8080/v1 --model <name>"))
        print(ui.dim("  or export KARL_BASE_URL / KARL_MODEL for one run."))


def cmd_ping() -> int:
    """Verify the endpoint actually answers — the first thing to run after
    pointing KARL at a remote GPU box."""
    cfg = endpoint()
    if not cfg["base_url"]:
        print(ui.yellow("  no endpoint set — nothing to ping. `karl config --base-url …` first."))
        return 1
    import time
    from karl.engine import HTTPEngine
    print(ui.dim(f"  pinging {cfg['model']} @ {cfg['base_url']} …"))
    t0 = time.time()
    res = HTTPEngine(cfg).chat(
        [{"role": "system", "content": "Reply with exactly one word: pong."},
         {"role": "user", "content": "ping"}])
    dt = time.time() - t0
    reply = (res.content or "").strip()
    if reply.startswith("(the model endpoint didn't respond"):
        print(ui.red("  ✗ no answer.  ") + ui.dim(reply))
        return 1
    print(ui.green(f"  ✓ answered in {dt:.1f}s: ") + reply[:200])
    return 0


def cmd_allow(project, rest: str) -> None:
    if web_open():
        print(ui.dim("  the web is open — the crew can already fetch any public "
                     "site, no allow needed."))
        print(ui.dim("  (want a whitelist instead? `karl config --web gated`, then "
                     "`karl allow <domain>`.)"))
        if not rest:
            return
    if not rest:
        al = project.allowlist()
        print(ui.dim("  allowed: " + (", ".join(al) if al else "(nothing yet)")))
        return
    opened = project.allow(rest.split()[0])
    if opened == "*":
        print(ui.yellow("  ⚠ the whole public web is now open to the crew."))
    elif opened:
        print(ui.green(f"  web is open for {opened}."))
    else:
        print(ui.yellow(f"  '{rest}' names no host — nothing opened."))


def cmd_deny(project, rest: str) -> None:
    if not rest:
        print(ui.yellow("usage: /deny <domain>"))
        return
    if project.disallow(rest.split()[0]):
        print(ui.dim(f"  closed {rest.split()[0]}."))
    else:
        print(ui.dim(f"  {rest.split()[0]} was not open."))


def cmd_project(rest: str):
    parts = rest.split()
    if not parts:
        print(ui.dim(f"  current project: {ui.bold(current_project_name())}"))
        root = projects_root()
        if root.exists():
            names = sorted(p.name for p in root.iterdir() if p.is_dir())
            if names:
                print(ui.dim("  projects: " + ", ".join(names)))
        return None
    name = parts[0]
    if not valid_project_name(name):
        print(ui.yellow("  a project name is letters, digits, . _ - (start with a letter/digit)."))
        return None
    use_project(name)
    Project(name).ensure()
    print(ui.green(f"  project → {name}"))
    return load_project()


def cmd_notes(project) -> None:
    notes = project.notes()
    print(notes if notes else ui.dim("  no notes yet — add one with /note <text>"))


def cmd_note(project, rest: str) -> None:
    if not rest:
        print(ui.yellow("usage: /note <text>"))
        return
    import time
    p = project.notes_path
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(f"- ({time.strftime('%Y-%m-%d')}) {rest}\n")
    print(ui.dim("  noted."))


def config_from_args(argv: list) -> int:
    """`karl config [--base-url U] [--model M] [--shell off|container|host] …`"""
    if not argv:
        cmd_model()
        return 0
    updates, i = {}, 0
    flags = {"--base-url": "base_url", "--model": "model", "--api-key": "api_key",
             "--max-tokens": "max_tokens", "--temperature": "temperature",
             "--shell": "shell", "--shell-net": "shell_net", "--web": "web",
             "--stream": "stream"}
    while i < len(argv):
        a = argv[i]
        if a in flags and i + 1 < len(argv):
            val = argv[i + 1]
            if a == "--max-tokens":
                val = int(val)
            elif a == "--temperature":
                val = float(val)
            elif a == "--shell" and val not in ("off", "container", "host"):
                print(ui.yellow("  --shell must be off, container, or host"))
                return 2
            elif a == "--shell-net" and val not in ("none", "bridge"):
                print(ui.yellow("  --shell-net must be none or bridge"))
                return 2
            elif a == "--web" and val not in ("open", "gated"):
                print(ui.yellow("  --web must be open or gated"))
                return 2
            elif a == "--stream" and val not in ("on", "off"):
                print(ui.yellow("  --stream must be on or off"))
                return 2
            updates[flags[a]] = val
            i += 2
        else:
            print(ui.yellow(f"  unknown option: {a}"))
            return 2
    set_config(**updates)
    if updates.get("shell") == "host":
        print(ui.yellow("  ⚠ host shell runs the model's commands directly on "
                        "this machine, unsandboxed. Prefer `--shell container`."))
    cmd_model()
    return 0


def _dispatch(session: Session, raw: str) -> bool:
    """One line from the operator. Returns False to exit. Mutates the session
    in place when the project changes."""
    if not raw.startswith("/"):
        session.run_task(raw)
        return True
    cmd, _, rest = raw[1:].partition(" ")
    cmd, rest = cmd.lower(), rest.strip()
    project = session.project
    if cmd in ("quit", "exit", "q"):
        return False
    elif cmd in ("help", "h", "?"):
        print(HELP)
    elif cmd in ("dash", "status"):
        cmd_dash(project)
    elif cmd == "agents":
        cmd_agents(project)
    elif cmd == "allow":
        cmd_allow(project, rest)
    elif cmd == "deny":
        cmd_deny(project, rest)
    elif cmd == "model":
        cmd_model()
    elif cmd == "gpu":
        from karl.gpu import handle
        handle(rest)
    elif cmd in ("ping", "test"):
        cmd_ping()
    elif cmd == "notes":
        cmd_notes(project)
    elif cmd == "note":
        cmd_note(project, rest)
    elif cmd == "project":
        newp = cmd_project(rest)
        if newp is not None:
            session.__init__(newp, echo=True)
    elif cmd == "crew":
        if rest == "init":
            write_default_crew(project)
            print(ui.green(f"  wrote {project.root / 'crew.json'} — edit it and "
                           "it loads next task."))
        else:
            cmd_agents(project)
    else:
        print(ui.yellow(f"  unknown command: /{cmd}  (/help for the list)"))
    return True


def repl() -> None:
    session = Session(load_project())
    print(BANNER)
    cmd_dash(session.project)
    print()
    while True:
        try:
            raw = input(ui.magenta("you ▸ ") if sys.stdin.isatty() else "")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        raw = raw.strip()
        if not raw:
            continue
        try:
            if not _dispatch(session, raw):
                break
        except KeyboardInterrupt:
            print(ui.yellow("\n  (interrupted)"))
        except Exception as e:  # noqa: BLE001 — the cockpit must survive one bad turn
            print(ui.red(f"  error: {type(e).__name__}: {e}"))
    print(ui.dim("  — bye —"))


def _pop_workspace(argv: list) -> list:
    """Pull a leading ``-C DIR`` / ``--workspace DIR`` off the args and apply it
    as the workspace override for this process."""
    import os
    out, i = [], 0
    while i < len(argv):
        if argv[i] in ("-C", "--workspace") and i + 1 < len(argv):
            os.environ["KARL_WORKSPACE"] = str(Path(argv[i + 1]).expanduser().resolve())
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out


def main(argv=None) -> int:
    try:
        return _main(argv)
    except BrokenPipeError:
        # stdout was closed under us (`karl … | head`) — die quietly, the
        # Unix way, instead of vomiting a traceback over the operator
        import os
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 141


def _main(argv=None) -> int:
    from karl import __version__
    argv = _pop_workspace(list(sys.argv[1:] if argv is None else argv))
    if argv and argv[0] in ("-h", "--help"):
        print(BANNER + "\n" + HELP)
        return 0
    if argv and argv[0] in ("-V", "--version", "version"):
        print(f"karl {__version__}")
        return 0
    if argv and argv[0] == "config":
        return config_from_args(argv[1:])
    if argv and argv[0] == "gpu":
        from karl.gpu import handle
        handle(" ".join(argv[1:]))
        return 0
    if argv and argv[0] in ("ping", "test"):
        return cmd_ping()
    if argv and argv[0] == "run":
        task = " ".join(argv[1:]).strip()
        if not task:
            print(ui.yellow("usage: karl run \"<task>\""))
            return 2
        return 0 if Session(load_project()).run_task(task) else 1
    if argv and argv[0] in ("dash", "status"):
        cmd_dash(load_project())
        return 0
    if argv and argv[0] == "allow":
        cmd_allow(load_project(), " ".join(argv[1:]).strip())
        return 0
    if argv and argv[0] == "deny":
        cmd_deny(load_project(), " ".join(argv[1:]).strip())
        return 0
    if argv and argv[0] == "agents":
        cmd_agents(load_project())
        return 0
    repl()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
