"""The session — a crew working across rounds, with a memory that persists.

Three layers of state, each with a job:

  * **Memory** — the conversation across rounds. What the operator asked, what
    the crew answered, what was decided: carried verbatim while it fits, folded
    into a condensed digest as it ages. A follow-up round genuinely follows.
  * **The board** — the crew's live plan (``board.md``), rewritten by the chief
    as work progresses, shown to every agent every turn.
  * **The transcript** — the full record on disk (jsonl), untouched by any of
    the compaction above.

Turn order is structured first, prose second: an agent ends its turn with the
``handoff`` tool naming who speaks next; a plain line falls back to
name-mention routing (last name wins). The chief is the hub — only he speaks
with the operator, and that is how a round closes. A hard turn cap guarantees
every round ends.

On a TTY the crew speaks live — streaming tokens, gear lines, the tach — and
each round closes with one dim telemetry line.
"""

from __future__ import annotations

import json
import re
import time

from karl import ui
from karl.config import endpoint, load_project, web_open
from karl.crew import build_system, load_crew
from karl.engine import make_engine
from karl.loop import think_and_act
from karl.tools import ToolContext, build_tools

_OPERATOR_ALIASES = ("operator", "user")
_MAX_TURNS = 24

_FENCE = re.compile(r"```.*?```", re.DOTALL)
_MAX_FENCE = 1200      # code blocks up to this size cross the transcript intact
_MAX_LINE = 1500
_MAX_LINE_CODE = 4200  # a line carrying code gets more room


def _plain(text: str) -> str:
    """Prose is collapsed to clean single-spaced text; fenced code up to
    _MAX_FENCE crosses verbatim (indentation and all) so the crew can actually
    show each other a diff. Oversized blocks become a pointer to the file."""
    text = text or ""
    parts, pos, fenced = [], 0, False
    for m in _FENCE.finditer(text):
        outside = " ".join(text[pos:m.start()].split())
        if outside:
            parts.append(outside)
        block = m.group(0)
        if len(block) <= _MAX_FENCE:
            parts.append(block)
            fenced = True
        else:
            parts.append("[large code omitted — see the file]")
        pos = m.end()
    tail = " ".join(text[pos:].split())
    if tail:
        parts.append(tail)
    return ("\n".join(parts) if fenced else " ".join(parts)).strip()


def _cap(text: str) -> str:
    limit = _MAX_LINE_CODE if "```" in text else _MAX_LINE
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0] or text[:limit]
    return cut + " …(trimmed)"


class Transcript:
    """The one shared channel: every line is plain English (plus bounded code),
    addressed to someone, appended to disk (jsonl) and echoed to the terminal.
    The full record stays on disk; prompts read from Memory, not from here."""

    def __init__(self, path, *, echo: bool = True):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self._entries: list = []

    def post(self, speaker: str, addressee, text: str, *, echo=None) -> dict:
        entry = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "speaker": speaker,
                 "addressee": addressee, "text": _cap(_plain(text))}
        self._entries.append(entry)
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        if self.echo if echo is None else echo:
            print(ui.line(speaker, addressee, entry["text"]))
        return entry

    def entries(self) -> list:
        return list(self._entries)


class Memory:
    """The conversation across rounds. Recent lines are carried verbatim under
    a character budget; overflow folds, oldest first, into two condensed
    stores with different loyalties:

      * the **ledger** — everything the operator said. Requirements outlive
        chatter: these lines are folded gently and evicted last.
      * the **digest** — the crew's own conclusions, folded hard.

    Everything is bounded, so a marathon session eventually forgets — but it
    forgets crew chatter long before it forgets an instruction."""

    RECENT_BUDGET = 9000
    LEDGER_BUDGET = 3000
    DIGEST_BUDGET = 3000
    _FOLD_LINE = 280

    def __init__(self):
        self.ledger = ""
        self.digest = ""
        self.recent: list = []   # (who, text)

    def add(self, who: str, text: str) -> None:
        self.recent.append((who, text))
        self._fold()

    def _size(self) -> int:
        return sum(len(w) + len(t) + 4 for w, t in self.recent)

    def _fold(self) -> None:
        while self.recent and self._size() > self.RECENT_BUDGET:
            who, text = self.recent.pop(0)
            line = f"{who}: {' '.join(text.split())[:self._FOLD_LINE]}"
            if who.startswith("operator"):
                self.ledger = (self.ledger + "\n" + line).strip()
                if len(self.ledger) > self.LEDGER_BUDGET:
                    self.ledger = "…" + self.ledger[-self.LEDGER_BUDGET:]
            else:
                self.digest = (self.digest + "\n" + line).strip()
                if len(self.digest) > self.DIGEST_BUDGET:
                    self.digest = "…" + self.digest[-self.DIGEST_BUDGET:]

    def render(self) -> str:
        parts = []
        if self.ledger:
            parts.append("WHAT THE OPERATOR HAS SAID (condensed, still binding):\n"
                         + self.ledger)
        if self.digest:
            parts.append("EARLIER IN THIS SESSION (condensed):\n" + self.digest)
        if self.recent:
            parts.append("THE CONVERSATION SO FAR (verbatim, most recent last):\n"
                         + "\n".join(f"{w}: {t}" for w, t in self.recent))
        return "\n\n".join(parts) if parts else "(nothing said yet)"

    def clear(self) -> None:
        self.ledger = ""
        self.digest = ""
        self.recent = []


def next_speaker(speaker: str, text: str, names: list, lead: str) -> str:
    """Prose-fallback routing: who speaks next after ``speaker`` said ``text``.

    Used only when a turn ended without a structured handoff. The convention
    is "END your line by naming who speaks next", so the *last* name the line
    calls (that isn't the speaker's own) wins. Naming the operator closes the
    round when the chief says it, and is routed to the chief when anyone else
    says it. A line naming no one falls to the chief — and from the chief, to
    the operator, which is how a round ends.
    """
    vocab = [n.lower() for n in names] + list(_OPERATOR_ALIASES)
    pattern = re.compile(r"\b(" + "|".join(re.escape(v) for v in vocab) + r")\b",
                         re.IGNORECASE)
    chosen = None
    for m in pattern.finditer(text or ""):
        name = m.group(1).lower()
        if name != speaker.lower():
            chosen = name
    if chosen is None or chosen in _OPERATOR_ALIASES:
        return "operator" if speaker == lead else lead
    return chosen


class Session:
    def __init__(self, project=None, *, echo: bool = True, engine=None):
        self.project = project or load_project()
        self.crew = load_crew(self.project)
        self.names = [a.name for a in self.crew]
        self.lead = self.crew[0].name
        self.by_name = {a.name: a for a in self.crew}
        if engine is not None:
            self.engine, self.mode = engine, "test"
        else:
            self.engine, self.mode = make_engine()
        cfg = endpoint()
        self.shell_mode = cfg.get("shell", "off")
        self.shell_net = cfg.get("shell_net", "none")
        self.installs = cfg.get("installs", "ask")
        self.max_steps = cfg.get("max_steps", 40)
        self.max_turns = cfg.get("max_turns", _MAX_TURNS)
        self.web_open = web_open()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.transcript = Transcript(self.project.session_path(stamp), echo=echo)
        self.memory = Memory()
        self.tainted: list = []
        self._tool_count = 0
        self._shell_grant = None    # "host" once the operator says yes, this session

    def _refresh(self) -> None:
        """Re-read the endpoint before each task, so `/gpu ssh …` or a `karl
        config` from another terminal takes effect mid-session instead of the
        cockpit riding a stale engine until restart. Injected test engines are
        left alone."""
        if self.mode == "test":
            return
        self.engine, self.mode = make_engine()
        cfg = endpoint()
        self.shell_mode = self._shell_grant or cfg.get("shell", "off")
        self.shell_net = cfg.get("shell_net", "none")
        self.installs = cfg.get("installs", "ask")
        self.max_steps = cfg.get("max_steps", 40)
        self.max_turns = cfg.get("max_turns", _MAX_TURNS)
        self.web_open = web_open()

    def _attach(self) -> None:
        """No endpoint, but a GPU box on file from an earlier ``gpu ssh``?
        Reattach to it — the default is the real model, not a shrug."""
        from karl import gpu
        state = gpu._load()
        if not (state.get("ssh_conn") and state.get("local_port")
                and state.get("remote_port")):
            return
        echo = self.transcript.echo
        if echo:
            print("  " + ui.dim("no endpoint set, but a GPU box is on file — "
                                "reattaching…"))
        gpu.handle("reconnect", out=print if echo else (lambda *_: None))
        self._refresh()

    def _record(self, speaker: str, addressee, text: str, *, echo=None) -> None:
        self.transcript.post(speaker, addressee, text, echo=echo)
        self.memory.add(f"{speaker}→{'you' if addressee == 'operator' else addressee}",
                        _cap(_plain(text)))

    # -- one task --------------------------------------------------------
    def run_task(self, text: str) -> bool:
        """Run one round. Returns True if a crew actually worked the task —
        False when no model is attached (KARL does not fake it)."""
        import sys
        self._refresh()
        if self.mode == "none":
            self._attach()
        if self.mode == "none":
            if self.transcript.echo:
                print("  " + ui.red("✗ no model attached — KARL doesn't fake it."))
                print(ui.dim("    karl gpu ssh -p <port> root@<host> -L "
                             "8080:localhost:8080   # serve one on your GPU box"))
                print(ui.dim("    karl config --base-url http://localhost:8080/v1"
                             "             # or point at any endpoint"))
                print(ui.dim("    (want the canned demo crew anyway?  "
                             "KARL_OFFLINE=1 karl)"))
            return False
        if self.mode == "offline" and self.transcript.echo:
            print("  " + ui.yellow("⚠ offline stand-in — the crew below is "
                                   "canned theater (KARL_OFFLINE is set).")
                  + ui.dim("  attach a real model: karl gpu ssh <ssh…>  ·  "
                           "karl config --base-url <url>"))
        t0 = time.time()
        tools0, taint0 = self._tool_count, len(self.tainted)
        # on a TTY the operator just typed the task — repeating it is clutter
        self._record("operator", self.lead, text,
                     echo=False if sys.stdout.isatty() else None)
        speaker, heard_from, heard = self.lead, "operator", text

        for turn in range(self.max_turns):
            spoken, live, to = self._speak(speaker, heard_from, heard,
                                           opening=(turn == 0))
            addressee = self._route(speaker, spoken, to)
            if speaker == self.lead and addressee == "operator":
                self._close(spoken, taint0, live)
                self._telemetry(turn + 1, tools0, t0)
                return True
            self._record(speaker, addressee, spoken,
                         echo=None if not live else False)
            heard_from, speaker, heard = speaker, addressee, spoken

        # cap hit: the chief closes honestly rather than wandering on
        spoken, live, _ = self._speak(self.lead, heard_from, heard,
                                      opening=False, close=True)
        self._close(spoken, taint0, live)
        self._telemetry(self.max_turns + 1, tools0, t0)
        return True

    def _route(self, speaker: str, spoken: str, to) -> str:
        """Structured handoff wins; prose routing is the fallback. Either way,
        only the chief may address the operator — anyone else's 'operator'
        goes through him."""
        if to is None:
            return next_speaker(speaker, spoken, self.names, self.lead)
        if to in _OPERATOR_ALIASES:
            return "operator" if speaker == self.lead else self.lead
        return to if to in self.names else next_speaker(
            speaker, spoken, self.names, self.lead)

    def _close(self, spoken: str, taint0: int, live: bool) -> None:
        flagged = self._flag(spoken, taint0)
        if live and flagged is not spoken and self.transcript.echo:
            # the line already streamed; surface the taint warning it carries
            print("  " + ui.yellow(flagged[:flagged.index(spoken)].strip()))
        self._record(self.lead, "operator", flagged,
                     echo=None if not live else False)

    def _telemetry(self, turns: int, tools0: int, t0: float) -> None:
        if self.transcript.echo:
            used = self._tool_count - tools0
            gear = f" · {used} tool call{'s' if used != 1 else ''}" if used else ""
            print("  " + ui.dim(f"— {turns} turn{'s' if turns != 1 else ''}"
                                f"{gear} · {time.time() - t0:.1f}s"))

    # -- one agent's turn ------------------------------------------------
    def _speak(self, name: str, heard_from: str, heard: str, *,
               opening: bool, close: bool = False):
        """Run one agent's think→act turn, rendered live on a TTY. Returns
        (spoken_line, live, handoff_to)."""
        agent = self.by_name[name]
        system = build_system(agent, self.crew, self.lead, self.project.notes(),
                              workspace=str(self.project.workspace),
                              board=self.project.board())
        ctx = ToolContext(workspace=self.project.workspace, project=self.project,
                          can_egress=agent.can_egress, web_open=self.web_open,
                          shell_mode=self.shell_mode, shell_net=self.shell_net,
                          installs=self.installs, tainted=self.tainted)
        tools = build_tools(agent.tools, ctx)
        user = self._task(name, heard_from, heard, opening=opening, close=close)
        seed = self._seed(name, opening=opening, close=close)
        targets = [n for n in self.names if n != name] + ["operator"]

        stream = ui.Stream(name, enabled=self.transcript.echo)
        tach = ui.Tach(f"{name} is thinking",
                       hint="no tokens yet · Ctrl-C aborts · try: karl doctor")

        def ask_operator(question, scope=""):
            """The consent lever, pulled mid-round. TTY only; a yes grants
            ONLY the scope that was asked about."""
            import sys as _sys
            if not (_sys.stdin.isatty() and _sys.stdout.isatty()):
                return False
            tach.stop()
            stream.end()
            try:
                answer = input("  " + ui.yellow(f"⚠ {name} asks: {question} [y/N] "))
            except (EOFError, KeyboardInterrupt):
                answer = ""
            granted = answer.strip().lower() in ("y", "yes")
            if granted and scope == "host_shell":
                self._shell_grant = "host"
                self.shell_mode = ctx.shell_mode = "host"
                print(ui.dim("  host shell granted for this session — persist or "
                             "revoke with `karl config --shell host|container|off`"))
            elif granted:
                print(ui.dim("  granted."))
            else:
                print(ui.dim("  declined."))
            tach.start()
            return granted

        ctx.ask = ask_operator

        def on_tool_start(tool_name):
            tach.stop()
            tach.set(f"{name} · running {tool_name}")
            tach.hint = f"{tool_name} still running · Ctrl-C aborts"
            tach.start()

        def on_token(piece):
            tach.stop()
            stream.token(piece)

        def on_tool(tool_name, first_line):
            self._tool_count += 1
            tach.stop()
            detail = f"{tool_name}  {first_line}".strip()
            if stream.enabled:
                stream.tool(detail)
            elif self.transcript.echo:
                print("    " + ui.dim("⚙ " + detail))
            tach.set(f"{name} · {tool_name}")
            tach.hint = "no tokens yet · Ctrl-C aborts · try: karl doctor"
            tach.start()

        tach.start()
        try:
            spoken, _, to = think_and_act(
                self.engine, system=system, user=user, tools=tools, ctx=ctx,
                seed=seed, on_token=on_token, on_tool=on_tool,
                on_tool_start=on_tool_start, handoff_to=targets,
                max_steps=self.max_steps)
        finally:
            tach.stop()
            live = stream.spoke
            stream.end()
        return spoken, live, to

    def _task(self, name: str, heard_from: str, heard: str, *,
              opening: bool, close: bool) -> str:
        context = self.memory.render() + "\n\n"
        if close:
            return (context + "The round has run long. Bring the operator a clear, "
                    "plain-English summary of where things stand, and hand off "
                    "to the operator.")
        if opening and name == self.lead:
            return (context + f"The operator asked: \"{heard}\". This begins the "
                    "round — and the conversation above still stands. Update the "
                    "board if the plan changed, delegate with the handoff tool, "
                    "or handle a read-only part yourself first.")
        return (context + f"It is your turn. {heard_from} said to you: \"{heard}\". "
                "Do your part with your tools — take the steps you need — then "
                "end your turn with the handoff tool.")

    def _seed(self, name: str, *, opening: bool, close: bool) -> str:
        """A deterministic in-character line for the offline MockEngine, so a
        session still moves and terminates with no model. Real models ignore it."""
        teammates = [n for n in self.names if n != self.lead]
        first = teammates[0] if teammates else self.lead
        if close or (name == self.lead and not opening):
            return ("Thanks — that covers it. Operator, here is where we stand: "
                    "the crew looked it over and reported back.")
        if name == self.lead and opening:
            return (f"Understood. {first}, take the first look and report back "
                    f"to {self.lead}.")
        return f"I've done my part and noted what I found. {self.lead}, over to you."

    def _flag(self, text: str, before: int) -> str:
        new = self.tainted[before:]
        if not new:
            return text
        return (f"⚠ this rests on data fetched from outside ({', '.join(sorted(set(new)))}) "
                f"— treat as unverified until checked. " + text)


def run_headless(task: str) -> bool:
    """One task, printed, then exit — for scripting (`karl run \"...\"`).
    Returns False when no model was attached and nothing ran."""
    return Session().run_task(task)
