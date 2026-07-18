"""The cockpit — everything KARL paints on the terminal.

Colour, a live tachometer for the silent seconds, a streaming line renderer so
the crew's words land token by token, and a small dashboard. ANSI only, no
dependencies, and everything degrades to plain text off a TTY or under NO_COLOR.
"""

from __future__ import annotations

import os
import sys

_ENABLED = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

_PALETTE = ("33", "32", "35", "34", "91", "95", "96")  # cycled over crew names


def _c(code: str):
    def paint(text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if _ENABLED else text
    return paint


dim = _c("2")
bold = _c("1")
red = _c("31")
green = _c("32")
yellow = _c("33")
blue = _c("34")
magenta = _c("35")
cyan = _c("36")
grey = _c("90")


def speaker_paint(name: str):
    """Every voice gets a steady colour. The chief drives in cyan; the operator
    is magenta; the rest of the crew cycle the palette by name."""
    if name in ("operator", "you"):
        return magenta
    if name in ("karl", "lead"):
        return cyan
    if name in ("system", None):
        return grey
    return _c(_PALETTE[sum(ord(ch) for ch in name) % len(_PALETTE)])


def line(speaker: str, addressee, text: str) -> str:
    """One finished transcript line: ``scout ▸ karl: text``. The operator shows
    as ``you`` on screen — that's who they are."""
    shown = "you" if addressee == "operator" else addressee
    arrow = f" {dim('▸')} {shown}" if shown else ""
    return f"{speaker_paint(speaker)(speaker)}{arrow}{dim(':')} {text}"


def bar(fraction: float, width: int = 22, label: str = "") -> str:
    """A dependency-free progress bar: [█████·········]  45%  label."""
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    body = "█" * filled + "·" * (width - filled)
    tail = f"  {label}" if label else ""
    return f"[{body}] {int(fraction * 100):3d}%{tail}"


def dash(rows: list) -> str:
    """The dashboard: aligned key/value gauges behind a thin rule.

        │ engine   glm-4.7-flash @ http://localhost:8080/v1
        │ web      open · shell off
    """
    width = max((len(k) for k, _ in rows), default=0)
    return "\n".join(f"  {dim('│')} {dim(k.ljust(width))}  {v}" for k, v in rows)


class Tach:
    """The tachometer — a live ``⠋ scout is thinking… 8.4s`` line for the silent
    seconds before a model speaks.

    A daemon thread redraws one line; ``stop()`` wipes it so streamed tokens can
    take the same row. Restartable: the loop stops it when words or tool lines
    arrive and spins it back up while the model is quiet again. TTY-only — off a
    TTY it is inert and ``active`` stays False so callers fall back to plain
    prints.
    """

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    HINT_AFTER = 40   # seconds of silence before the tach admits it

    def __init__(self, label: str = "working", hint: str = ""):
        self.label = label
        self.hint = hint          # shown once the current silence runs long
        self.active = False
        self._stop = None
        self._thread = None
        self._t0 = None           # start of the whole turn (what the clock shows)
        self._since = None        # start of the current silence (what the hint reads)

    def set(self, label: str) -> None:
        self.label = label

    def start(self) -> "Tach":
        import threading
        import time
        if sys.stdout.isatty() and not self.active:
            self.active = True
            self._stop = threading.Event()
            if self._t0 is None:
                self._t0 = time.time()
            self._since = time.time()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self):
        import time
        i = 0
        while not self._stop.wait(0.1):
            i += 1
            frame = self.FRAMES[i % len(self.FRAMES)]
            line = f"{self.label}… {time.time() - self._t0:.1f}s"
            if self.hint and time.time() - self._since > self.HINT_AFTER:
                line += f" · {self.hint}"
            sys.stdout.write("\r  " + cyan(frame) + " " + dim(line) + "   ")
            sys.stdout.flush()

    def stop(self) -> None:
        was = self.active
        self.active = False
        if self._stop:
            self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.3)
            self._thread = None
        if was and sys.stdout.isatty():
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()

    # context-manager sugar for one-shot uses
    def __enter__(self):
        return self.start()

    def __exit__(self, *a):
        self.stop()


class Stream:
    """Live rendering of one agent's turn: the words land as they arrive.

        scout ▸ the workspace has three modules…
          ⚙ read_file cli.py

    ``token()`` lazily opens a ``name ▸`` line and appends pieces; ``tool()``
    closes it and prints a gear line; ``end()`` closes whatever is open.
    ``spoke`` reports whether the *last* utterance reached the screen, so the
    transcript knows not to print the same line twice. Disabled (all no-ops,
    ``spoke`` False) off a TTY.
    """

    _CAP = 4000  # a runaway stream stops painting; the engine trims the record

    def __init__(self, name: str, enabled: bool):
        self.name = name
        self.enabled = enabled and sys.stdout.isatty()
        self._open = False
        self._count = 0
        self.spoke = False

    def token(self, piece: str) -> None:
        if not self.enabled or not piece:
            return
        if not self._open:
            sys.stdout.write("  " + speaker_paint(self.name)(self.name) + dim(" ▸ "))
            self._open = True
            self._count = 0
            self.spoke = True
        if self._count < self._CAP:
            sys.stdout.write(piece[: self._CAP - self._count])
            sys.stdout.flush()
        self._count += len(piece)

    def tool(self, text: str) -> None:
        if not self.enabled:
            return
        self._newline()
        self.spoke = False
        print("    " + dim("⚙ " + text))

    def end(self) -> None:
        self._newline()

    def _newline(self) -> None:
        if self._open:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._open = False
