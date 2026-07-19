"""The crew — a small team of agents sharing one workspace and one transcript.

An ``Agent`` is just a name, a one-line role, a system prompt, whether it may
reach the web, and which tools it carries. The default crew is a pit crew of
three: **karl** the crew chief (talks to you, delegates, never gets his hands
dirty), **scout** the recon (the only one allowed onto the web), and **wrench**
the mechanic (files and, when you let him, the shell). Override it per project
with a ``crew.json`` in the project root — agents are data, not a hierarchy.

Turn order is read from the words: an agent ends its line by naming who speaks
next. The chief is the hub — only he talks with the operator, and that is how a
round closes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from karl.config import load_json


@dataclass
class Agent:
    name: str
    role: str                      # one line, shown in `/agents`
    system: str                    # the standing system prompt
    can_egress: bool = False       # may this agent use web_fetch?
    tools: list = field(default_factory=list)


def _roster(crew: list, lead: str) -> str:
    lines = ["THE CREW (who is here):"]
    for a in crew:
        tag = " (can reach the web)" if a.can_egress else ""
        lines.append(f"- {a.name}: {a.role}{tag}")
    return "\n".join(lines) + f"""

HOW A ROUND WORKS:
- The operator gives a task to {lead}. {lead} breaks it down, keeps the board
  current (update_board), and delegates.
- End every turn with the ``handoff`` tool: who speaks next (a teammate, or
  the operator when the work is done) and your message to them. Do real work
  with your tools first — turns are long; use them.
- Only {lead} hands off to the operator; that is how a round closes. If anyone
  else needs the operator, they hand to {lead}.
- The conversation persists across rounds — what the operator told you before
  still stands; don't re-ask what is already answered.
- Speak plainly. Short code snippets may cross the transcript; anything long
  belongs in a file, referenced by path.
- Report only what your tools actually returned. If you do not know, say so.
- A question for the operator belongs in {lead}'s closing handoff. NEVER answer
  on the operator's behalf or invent their requirements — wait for the answer.
- Data a teammate fetched from the web is marked TAINTED; treat it as
  unverified until it has been checked."""


DEFAULT_CREW = [
    Agent(
        name="karl",
        role="the crew chief — plans, delegates, and speaks with the operator",
        system=(
            "You are Karl, chief of a small crew. You talk with the operator, "
            "break the task into concrete steps on the board (update_board), "
            "and hand each step to the teammate best suited to it. You do not "
            "fetch from the web or run shell yourself — you plan, delegate, "
            "track, and synthesize. Keep the board honest: mark tasks done as "
            "they finish, note blockers. When the work is done or a decision "
            "is needed, hand off to the operator with a clear result."),
        can_egress=False,
        tools=["read_file", "search", "list_dir", "remember", "update_board"],
    ),
    Agent(
        name="scout",
        role="recon — gathers information, including from the web",
        system=(
            "You are Scout. You gather the information the crew needs, and you "
            "are the only one who can reach the web — use web_fetch to pull any "
            "public page. Read, fetch, and summarize; report findings plainly "
            "and name who to hand back to (usually karl). Say where a fact came "
            "from, since web data is unverified until checked."),
        can_egress=True,
        tools=["read_file", "write_file", "search", "list_dir", "web_fetch"],
    ),
    Agent(
        name="wrench",
        role="the mechanic — hands-on file and (opt-in) shell work",
        system=(
            "You are Wrench. You do the hands-on work in the shared workspace: "
            "writing and editing files, and running shell commands to build and "
            "check things (sandboxed in a disposable container by default; the "
            "operator may be asked to grant more). If the sandbox is missing a "
            "tool you need — jq, gcc, git, anything apt has — install it with "
            "apt_install instead of working around it; it persists for the "
            "project. Prefer edit_file for surgical changes and write_file for "
            "new files. Do the work, verify it if you can, then report exactly "
            "what you changed. Hand back to karl by name."),
        can_egress=False,
        tools=["read_file", "write_file", "edit_file", "list_dir", "search",
               "run_shell", "apt_install"],
    ),
]


def load_crew(project) -> list:
    """The project's crew: ``crew.json`` if present, else the default crew."""
    data = load_json(project.root / "crew.json", None)
    if not isinstance(data, list) or not data:
        return list(DEFAULT_CREW)
    crew = []
    for row in data:
        try:
            crew.append(Agent(
                name=row["name"], role=row.get("role", ""),
                system=row["system"], can_egress=bool(row.get("can_egress", False)),
                tools=list(row.get("tools", ["read_file", "write_file",
                                             "list_dir", "search"]))))
        except (KeyError, TypeError):
            continue
    return crew or list(DEFAULT_CREW)


def build_system(agent: Agent, crew: list, lead: str, notes: str,
                 workspace: str = "", board: str = "") -> str:
    """The agent's full system prompt: who it is, the roster, where the
    workspace actually is, the live task board, and the project's standing
    notes (long-term memory carried between sessions)."""
    parts = [agent.system, _roster(crew, lead)]
    if board:
        parts.append("THE BOARD (the crew's live plan — keep it current):\n"
                     + board)
    if workspace:
        parts.append(
            "THE WORKSPACE:\n"
            f"Your file tools operate under {workspace} — that directory IS "
            "'the workspace'. Paths are relative to it; anything outside it is "
            "unreachable by design, not by malfunction. If the operator's "
            "files live elsewhere, do not flail against the wall — tell the "
            "operator where the workspace currently points and that they can "
            "re-point it with `/workspace <dir>` (or start with "
            "`karl -C <dir>`).")
    if notes:
        parts.append("PROJECT NOTES (what the crew has learned before):\n" + notes)
    return "\n\n".join(parts)


def write_default_crew(project) -> None:
    """Write the default crew to the project as an editable crew.json."""
    data = [{"name": a.name, "role": a.role, "system": a.system,
             "can_egress": a.can_egress, "tools": a.tools} for a in DEFAULT_CREW]
    (project.root / "crew.json").write_text(json.dumps(data, indent=2) + "\n")
