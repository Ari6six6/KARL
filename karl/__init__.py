"""KARL — a small, fast multi-agent CLI harness for local LLMs.

A crew of agents shares one workspace and one transcript, takes turns by
addressing each other by name, uses real sandboxed tools (files, an opt-in
container shell, a single SSRF-guarded web egress), and you steer from the top.
Pure Python standard library — nothing to install — against any
OpenAI-compatible endpoint, streaming tokens live as the crew speaks, with a
built-in offline stand-in when no model is attached.

KARL replaces MoRE. Same road, fewer parts, faster line.
"""

__version__ = "9.1.1"

from karl.cli import main

__all__ = ["main"]
