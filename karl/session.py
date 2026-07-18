"""The session — a crew working one task at a time in one shared transcript.

Turn order is read from the words themselves: an agent ends its line by naming
who speaks next, and that agent goes next (``next_speaker``). The chief is the
hub — he opens each round and closes it by turning to the operator. A hard turn
cap guarantees every round ends.

On a TTY the crew speaks *live*: tokens stream onto the screen as they come off
the wire, tool calls tick past as gear lines, and the tachometer covers the
silent seconds. Off a TTY every line prints whole — same transcript, no
theatrics. Each round closes with one dim telemetry line: turns, tool calls,
elapsed.
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
_MAX_TURNS = 10

_FENCE = re.compile(r"```.*?```", re.DOTALL)
_MAX_LINE = 1500


def _plain(text: str) -> str:
    # the transcript carries prose and file references, not pasted code — that
    # keeps lines readable and stops a pasted log from poisoning later turns
    cleaned = _FENCE.sub("[code omitted — see the file]", text or "")
    return " ".join(cleaned.split()).strip()


def _cap(text: str) -> str:
    if len(text) <= _MAX_LINE:
        return text
    cut = text[:_MAX_LINE].rsplit(" ", 1)[0] or text[:_MAX_LINE]
    return cut + " …(trimmed)"


class Transcript:
    """The one shared channel: every line is plain English, addressed to
    someone, appended to disk (jsonl) and echoed to the terminal. A live turn
    reads only the bounded *tail*; the full record stays on disk."""

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

    def tail_text(self, n: int = 14) -> str:
        rows = self._entries[-n:]
        if not rows:
            return "(nothing said yet)"
        return "\n".join(
            e["speaker"] + (f"→{e['addressee']}" if e.get("addressee") else "")
            + f": {e['text']}" for e in rows)


def next_speaker(speaker: str, text: str, names: list, lead: str) -> str:
    """Who speaks next after ``speaker`` said ``text``.

    The first name the line calls (that isn't the speaker's own) wins. Naming
    the operator closes the round when the chief says it, and is routed to the
    chief when anyone else says it (only the chief speaks with the operator). A
    line that names no one falls to the chief — and the chief, with no one to
    hand to, turns to the operator, which is how a round ends.
    """
    vocab = [n.lower() for n in names] + list(_OPERATOR_ALIASES)
    pattern = re.compile(r"\b(" + "|".join(re.escape(v) for v in vocab) + r")\b",
                         re.IGNORECASE)
    seen = set()
    for m in pattern.finditer(text or ""):
        name = m.group(1).lower()
        if name in seen:
            continue
        seen.add(name)
        if name == speaker.lower():
            continue
        if name in _OPERATOR_ALIASES:
            return "operator" if speaker == lead else lead
        return name
    return "operator" if speaker == lead else lead


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
        self.web_open = web_open()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.transcript = Transcript(self.project.session_path(stamp), echo=echo)
        self.tainted: list = []
        self._tool_count = 0

    def _refresh(self) -> None:
        """Re-read the endpoint before each task, so `/gpu ssh …` or a `karl
        config` from another terminal takes effect mid-session instead of the
        cockpit riding a stale engine until restart. Injected test engines are
        left alone."""
        if self.mode == "test":
            return
        self.engine, self.mode = make_engine()
        cfg = endpoint()
        self.shell_mode = cfg.get("shell", "off")
        self.shell_net = cfg.get("shell_net", "none")
        self.web_open = web_open()

    # -- one task --------------------------------------------------------
    def run_task(self, text: str) -> None:
        import sys
        self._refresh()
        if self.mode == "offline" and self.transcript.echo:
            # the stand-in must never pass for the real crew
            print("  " + ui.yellow("⚠ offline stand-in — no model attached; the crew "
                                   "below is canned theater.")
                  + ui.dim("  attach one: karl gpu ssh <ssh…>  ·  "
                           "karl config --base-url <url>"))
        tr = self.transcript
        t0 = time.time()
        tools0, taint0 = self._tool_count, len(self.tainted)
        # on a TTY the operator just typed the task — repeating it is clutter
        tr.post("operator", self.lead, text,
                echo=False if sys.stdout.isatty() else None)
        speaker, heard_from, heard = self.lead, "operator", text

        for turn in range(_MAX_TURNS):
            spoken, live = self._speak(speaker, heard_from, heard,
                                       opening=(turn == 0))
            addressee = next_speaker(speaker, spoken, self.names, self.lead)
            if speaker == self.lead and addressee == "operator":
                self._close(spoken, taint0, live)
                self._telemetry(turn + 1, tools0, t0)
                return
            tr.post(speaker, addressee, spoken, echo=None if not live else False)
            heard_from, speaker, heard = speaker, addressee, spoken

        # cap hit: the chief closes honestly rather than wandering on
        spoken, live = self._speak(self.lead, heard_from, heard,
                                   opening=False, close=True)
        self._close(spoken, taint0, live)
        self._telemetry(_MAX_TURNS + 1, tools0, t0)

    def _close(self, spoken: str, taint0: int, live: bool) -> None:
        flagged = self._flag(spoken, taint0)
        if live and flagged is not spoken and self.transcript.echo:
            # the line already streamed; surface the taint warning it carries
            print("  " + ui.yellow(flagged[:flagged.index(spoken)].strip()))
        self.transcript.post(self.lead, "operator", flagged,
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
        (spoken_line, live) — ``live`` True when the line already streamed to
        the screen, so it must not be echoed twice."""
        agent = self.by_name[name]
        system = build_system(agent, self.crew, self.lead, self.project.notes())
        ctx = ToolContext(workspace=self.project.workspace, project=self.project,
                          can_egress=agent.can_egress, web_open=self.web_open,
                          shell_mode=self.shell_mode, shell_net=self.shell_net,
                          tainted=self.tainted)
        tools = build_tools(agent.tools, ctx)
        user = self._task(name, heard_from, heard, opening=opening, close=close)
        seed = self._seed(name, opening=opening, close=close)
        steps = 12 if (agent.can_egress or "run_shell" in agent.tools) else 8

        stream = ui.Stream(name, enabled=self.transcript.echo)
        tach = ui.Tach(f"{name} is thinking")

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
            tach.start()

        tach.start()
        try:
            spoken, _ = think_and_act(
                self.engine, system=system, user=user, tools=tools, ctx=ctx,
                seed=seed, on_token=on_token, on_tool=on_tool, max_steps=steps)
        finally:
            tach.stop()
            live = stream.spoke
            stream.end()
        return spoken, live

    def _task(self, name: str, heard_from: str, heard: str, *,
              opening: bool, close: bool) -> str:
        context = "Recent transcript:\n" + self.transcript.tail_text() + "\n\n"
        if close:
            return (context + "The round has run long. Bring the operator a clear, "
                    "plain-English summary of where things stand, and address the "
                    "operator.")
        if opening and name == self.lead:
            return (context + f"The operator asked: \"{heard}\". This begins the "
                    "round. Break it into steps and delegate to a teammate by name, "
                    "or handle a read-only part yourself. End by naming who speaks "
                    "next.")
        return (context + f"It is your turn. {heard_from} said to you: \"{heard}\". "
                "Do your part with your tools, then speak one plain-English line and "
                "name who should speak next.")

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


def run_headless(task: str) -> None:
    """One task, printed, then exit — for scripting (`karl run \"...\"`)."""
    Session().run_task(task)
