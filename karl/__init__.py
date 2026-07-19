"""KARL — a small, fast multi-agent CLI harness for local LLMs.

A crew of agents shares one workspace and one transcript, takes turns by
addressing each other by name, uses real sandboxed tools (files, an opt-in
container shell, a single SSRF-guarded web egress), and you steer from the top.
Pure Python standard library — nothing to install — against any
OpenAI-compatible endpoint, streaming tokens live as the crew speaks. The real
model is the default: with nothing attached KARL refuses to fake it (the
canned demo crew is opt-in via KARL_OFFLINE=1).

KARL replaces MoRE. Same road, fewer parts, faster line.
"""

__version__ = "9.7.0"

from karl.cli import main

__all__ = ["main"]
