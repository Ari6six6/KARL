# KARL

*A small, fast multi-agent CLI harness for local LLMs.*

KARL runs a **crew** of LLM agents against any OpenAI-compatible endpoint —
your own GPU box, a local server, a hosted API. The agents share one workspace
and one transcript, take turns by addressing each other by name, use a handful
of real, sandboxed tools, and **you** steer from the top. Pure Python standard
library — nothing to install — and the crew speaks **live**: tokens land on
your screen as they come off the wire, with a tachometer covering the silent
seconds and one telemetry line closing every round.

**The default is the real thing.** No model attached → KARL refuses to run and
tells you how to attach one; if a GPU box is on file from an earlier
`gpu ssh`, it reattaches automatically. (Want to watch the machinery move with
no model? The canned demo crew is one env var away: `KARL_OFFLINE=1 karl`.)

```sh
git clone https://github.com/Ari6six6/KARL.git
cd KARL
./karl-cli                # run in place, no install
```
```
you ▸ summarize the python files in the workspace
karl ▸ scout, list and read the .py files and report what each does.
    ⚙ list_dir  .
    ⚙ read_file  cli.py
scout ▸ Three files: cli.py (entry point), db.py (storage), api.py (routes).
        karl, over to you.
karl ▸ The workspace has three modules — a CLI entry point, a storage layer,
       and an HTTP routes file. Want a deeper look at any one?
  — 3 turns · 2 tool calls · 11.8s
```

KARL is not an acronym. It's a name — ask a 911 what it stands for.

**Live on a GPU box right now?** The [Field Manual](MANUAL.md) takes you from
a Vast.ai SSH string to a working crew in 90 seconds, with the troubleshooting
table for everything that happens after.

> **KARL replaces [MoRE](https://github.com/Ari6six6/MoRE).** MoRE is the
> BMW: sensible, roomy, well-documented, still parked where it always was.
> KARL is the Porsche — the same road with fewer parts and a faster line.
> What changed under the hood:
>
> | | MoRE | KARL |
> |---|---|---|
> | memory | none — every round starts blank | **conversation persists across rounds**; operator words outlive crew chatter |
> | control flow | regex over prose | **structured `handoff` tool**, prose fallback |
> | turn depth | 8–12 tool calls | **40 (configurable to 200)** with context compaction |
> | planning | none | **a live task board** every agent sees, the chief maintains |
> | transcript | code stripped | **bounded code crosses intact** |
> | output | whole completions, then print | **streams live, token by token** |
> | no model | silent canned stand-in | **refuses + auto-reattaches a saved box**; stand-in opt-in |
> | tools | read/write/search/shell/web | + **surgical `edit_file`**, + **`apt_install`** (persistent sandbox image) |
> | shell | off by default | **on, sandboxed**; scoped consent for more |
> | cockpit | banner + spinner | tachometer, gear lines, `/dash`, `/doctor`, telemetry |
> | crew | lead / researcher / worker | **karl / scout / wrench** |
> | rails | sandbox · SSRF guard · taint | same rails, hardened (scoped consent, timeout caps) |
> | GPU deploy | one command | same one command, + dead-GPU detection |

---

## Install

**You don't need to install anything.** Run it from inside the folder
(Python 3.10+):

```sh
cd KARL
./karl-cli
```

Everywhere below, `karl <thing>` is shorthand for `./karl-cli <thing>` run from
the `KARL` folder. They are the same.

<details>
<summary>Optional: a global <code>karl</code> command</summary>

Either add an alias:

```sh
echo "alias karl=\"$HOME/KARL/karl-cli\"" >> ~/.bashrc && source ~/.bashrc
```

…or install it (on modern Debian/Ubuntu use a venv or pipx — a bare
`pip install` is blocked by the OS):

```sh
python3 -m venv .venv && . .venv/bin/activate && pip install -e .   # venv
# or:  pipx install .                                               # isolated
```
</details>

Verify (optional):

```sh
python3 -m pytest -q            # no model or network needed
```

---

## Deploy on a Vast.ai GPU — one command

Rent a box on Vast.ai (it hands you an SSH command like
`ssh -p 24439 root@1.2.3.4`). Add a `-L` port forward and give the whole thing
to KARL:

```sh
karl gpu model qwen                                         # pick a model (optional)
karl gpu ssh -p 24439 root@1.2.3.4 -L 8080:localhost:8080   # do everything
```

That single command reaches the box (retrying while it's still booting),
detects the GPUs, picks a VRAM/context tier, installs vLLM (or builds
llama.cpp for GGUF models), launches the server **with tool-calling enabled**,
opens the SSH tunnel, and waits — with a download bar — until the model
answers. Then it points the harness at `http://localhost:8080/v1` for you.
Confirm and go:

```sh
karl ping        # ✓ answered in 12.3s: pong
karl             # start working
```

Manage it:

```sh
karl gpu model          # list the catalog · karl gpu model <key> to pick
karl gpu status         # is the tunnel live? what's served?
karl gpu reconnect      # reopen a dropped tunnel to the same box
karl gpu off            # drop the tunnel (leaves the server running)
karl gpu down           # stop the server AND drop the tunnel
```

The model catalog lives in `karl/models.py` — edit it to add your own (repo,
served name, VRAM floor, context tiers, vLLM vs llama.cpp). The bundled rows
are a starting set; confirm a repo resolves before you lean on it.

### Bring-your-own endpoint

Already have a server running (local Ollama, a hosted API, a box you
provisioned by hand)? Skip `gpu` entirely:

```sh
karl config --base-url http://localhost:8080/v1 --model your-model
karl ping
```

> If you serve a model yourself, enable tool-calling or the crew can talk but
> not act. For vLLM: `--enable-auto-tool-choice --tool-call-parser hermes`.
> (`karl gpu` already does this for you.) Streaming is on by default; a server
> that can't stream falls back to plain requests automatically, or force it
> with `karl config --stream off`.
>
> **Running KARL *on* the GPU box?** Cleanest way to let the crew use the
> shell freely — the box is disposable. `pip install -e .` on the box and
> `karl config --base-url http://localhost:8080/v1 --shell host`.

### Containerized

Run the whole harness in a container so the shell is isolated by construction:

```sh
docker build -t karl .
docker run --rm -it \
  -e KARL_BASE_URL=http://your-gpu-box:8080/v1 \
  -e KARL_MODEL=your-model \
  -e KARL_SHELL=host \
  -v "$PWD":/work -e KARL_WORKSPACE=/work \
  -v karl-state:/root/.karl \
  karl
```

---

## The drivetrain

What separates a harness from a chat loop:

- **Memory.** The conversation persists across rounds: recent exchanges ride
  verbatim under a budget; older ones fold into condensed stores — and what
  *you* said folds last and survives longest. A follow-up genuinely follows;
  the crew doesn't re-ask what you already answered. `/history` shows what
  they remember; `/reset` clears it.
- **Structured handoffs.** Agents end turns with a `handoff(to, message)`
  tool — routing is data, not regex. Prose routing stays as the fallback for
  models that won't call tools.
- **Deep turns, bounded context.** Default 40 tool calls per turn
  (`karl config --max-steps`, up to 200; `--max-turns` for the round). As a
  turn's context grows past budget, old tool output compacts to stubs — the
  agent keeps recent evidence verbatim, old evidence in summary, and never
  gets cut off mid-reach.
- **The board.** A live `board.md` — goal, tasks, blockers — that karl
  maintains with `update_board` and every agent sees every turn. `/board`
  shows it; it survives sessions.

## The crew

A pit crew of three by default:

| agent    | does                                              | web |
|----------|---------------------------------------------------|-----|
| `karl`   | the chief — plans, delegates, speaks with you     | no  |
| `scout`  | recon — gathers information, including the web    | yes |
| `wrench` | the mechanic — file and (opt-in) shell work       | no  |

**How a round works.** You give a task; it goes to `karl`. Each agent ends its
line by naming who should speak next, and that agent goes next. Only `karl`
speaks with you — when he turns to you, the round is done. A hard turn cap
means every round ends.

**Make it yours.** `/crew init` writes an editable `crew.json` in the project.
Change the names, roles, prompts, tools, and who may reach the web. Agents are
plain data, not fixed roles.

---

## The tools, and the rails

Each agent gets only the tools its definition lists. Every tool returns a plain
observation the agent reads on its next step.

- **`read_file`, `write_file`, `edit_file`, `list_dir`, `search`** — sandboxed
  to the workspace. A path that escapes it is refused; long files page instead
  of truncating silently. `edit_file` swaps one exact occurrence and refuses
  ambiguity — an edit lands exactly where intended or not at all.
- **`run_shell`** — **on by default, sandboxed**. Three modes
  (`karl config --shell …`):
  - `container` *(default)* — runs in a disposable Docker/Podman container
    with only the workspace mounted and **no network**. No runtime running?
    KARL asks you at the prompt whether to allow a host shell for the session
    — it never touches the host silently. Allow network for a build with
    `karl config --shell-net bridge`.
  - `host` — runs directly on the host in the workspace dir. Unsandboxed;
    sensible when KARL itself is already in a throwaway box/container.
  - `off` — refused; a talk-only crew.
- **`apt_install`** — the toolbox grows on demand. The crew asks for what it's
  missing (jq, gcc, git, pip packages…); you consent once at the prompt, and
  the packages are baked into a **persistent per-project sandbox image** the
  shell uses from then on. Network is allowed only during the install; the
  shell itself stays offline. Package names are strictly validated — nothing
  can smuggle shell syntax into the build. `/sandbox` shows what's baked,
  `/sandbox reset` starts clean, `karl config --installs ask|open|off` sets
  the policy.
- **`web_fetch`** — the crew's way onto the public web:
  - only agents marked `can_egress` (the `scout`) get it;
  - **open by default** — any public site, no per-domain permission. (Prefer a
    whitelist? `karl config --web gated`, then `karl allow <domain>`.)
  - always SSRF-guarded — public web only, never the host's loopback/LAN or a
    cloud-metadata address. This never asks you anything; it just refuses;
  - one hop — redirects are reported, not followed;
  - anything it returns is flagged **tainted**, and a round that leaned on
    outside data says so when it reports back.
- **`remember`** — appends a durable note to the project's memory, shown to
  the crew next session.

---

## Commands

Run `karl` with no arguments for the cockpit; inside it:

```
<text>            give the crew a task
/dash             the dashboard — project, engine, rails, crew, at a glance
/agents           list the crew and who can reach the web
/allow <domain>   open web access for a domain  (/allow with no arg shows the list)
/deny <domain>    close a domain again
/ping             check the model endpoint answers
/gpu ssh <ssh…>   provision + serve a model on a GPU box (see above)
/gpu model|status|reconnect|off|down   pick · check · re-tunnel · drop · stop
/note <text>      add a durable project note   ·  /notes  show them
/model            show how KARL reaches the model
/project [name]   show, switch, or create a project
/crew init        write an editable crew.json
/help  ·  /quit
```

From the shell (scriptable):

```sh
karl run "audit the workspace for secrets and report"   # one task, then exit
karl -C ~/code/my-repo run "find the failing test"      # work on a real directory
karl config --base-url URL --model M --shell container  # configure
karl ping                                               # test the endpoint
karl dash                                               # the dashboard
karl allow docs.python.org                              # open one domain (gated mode)
```

Environment overrides (handy for one-off runs and containers): `KARL_BASE_URL`,
`KARL_MODEL`, `KARL_API_KEY`, `KARL_SHELL`, `KARL_WEB`, `KARL_WORKSPACE`,
`KARL_HOME`, `KARL_OFFLINE` (opt into the canned no-model demo crew).

---

## Where things live

Everything is plain files under `$KARL_HOME` (default `~/.karl`):

```
~/.karl/
  config.json                     # endpoint + settings
  current_project
  gpu.json                        # GPU box + tunnel state
  projects/<name>/
    workspace/                    # the shared workspace (unless -C / KARL_WORKSPACE)
    sessions/<timestamp>.jsonl    # full transcript of each session
    notes.md                      # durable project memory
    allow.json                    # the web allowlist (gated mode)
    crew.json                     # (optional) your custom crew
```

Inspect it, edit it, delete it — it's just files.

---

*Version numbers start at 9.1.1, for obvious reasons.*
