# The KARL Field Manual

*For the operator standing at a live Vast.ai box. Everything here is the
short version of what the code actually does — when in doubt, `/help`.*

---

## 0 · You're live on Vast — the 90-second start

The Vast console hands you an SSH command like:

```
ssh -p 24439 root@ssh4.vast.ai
```

Add a `-L` port forward to it and give the whole thing to KARL:

```sh
cd KARL
./karl-cli gpu model                # see the catalog, → marks the pick
./karl-cli gpu model qwen           # (optional) pick by VRAM — table in §7
./karl-cli gpu ssh -p 24439 root@ssh4.vast.ai -L 8080:localhost:8080
```

Pasting the whole line including a leading `ssh` word is forgiven —
`gpu ssh ssh -p …` works.

Then:

```sh
./karl-cli ping                     # ✓ answered in 12.3s: pong
./karl-cli                          # the cockpit — start working
```

That's it. Everything below is detail.

---

## 1 · What that one command just did

`karl gpu ssh …` runs the whole pipeline, in order:

1. **Reaches the box.** A freshly-rented box resets the first SSH handshakes
   while sshd boots; KARL recognizes that and retries 4× at 12s intervals
   instead of giving up. Auth failures are reported immediately — waiting
   never fixes a wrong key.
2. **Detects the GPUs** (`nvidia-smi`) and picks a context tier from total
   VRAM. Too small for the chosen model → clean refusal with the floor it
   needed; pick a smaller model (`gpu model <key>`) or a bigger box.
3. **Installs the runtime** — vLLM in a venv for FP8 models, or a CUDA build
   of llama.cpp for GGUF models. First time takes minutes; after that it's
   cached on the box.
4. **Launches the server with tool-calling enabled** (this matters — without
   it the crew can talk but not act). It writes `~/karl.pid` and `~/karl.log`
   on the box.
   - **Vast squats port 8080** on many images. If the box-side port is held,
     KARL slides the server to a free port (usually `18080`) and moves the
     tunnel's box side to follow. *Your local port stays 8080.* You'll see
     the slide announced.
5. **Opens a detached SSH tunnel** (PID-tracked, survives the command — close
   your terminal and the model stays reachable).
6. **Waits until the model answers**, painting a download bar while the
   weights land. If the server process dies (usually a CUDA library issue —
   KARL pre-arms `ldconfig` against exactly that), it stops immediately and
   shows the last 15 log lines instead of burning your rental on a corpse.
7. **Writes the config** — `base_url` and `model` land in `~/.karl/config.json`,
   so `karl`, `karl ping`, and every session use the box automatically.

---

## 2 · Daily driving

Run `karl` (or `./karl-cli`) for the cockpit. The dashboard prints on entry;
type a task in plain words and watch the crew work — live, token by token.

```
you ▸ find every TODO in the workspace and rank them by risk
```

- **karl** (cyan) plans and delegates; only he speaks with you.
- **scout** does recon, including the public web (the only agent allowed out).
- **wrench** edits files and runs the shell — when you've enabled it (§3).
- `⚙` lines are tools firing. The closing dim line is round telemetry.

Cockpit commands worth knowing:

```
/dash              the dashboard, any time
/agents            the crew, their tools, who can reach the web
/note <text>       durable project memory (the crew also saves its own via `remember`)
/notes             read the memory back
/project <name>    switch worlds — separate workspace, transcript, notes
/crew init         write crew.json — rename agents, edit prompts, re-cut tools
/quit
```

Scriptable one-shots from your shell:

```sh
karl run "audit the workspace for secrets and report"
karl -C ~/code/my-repo run "find the failing test"     # crew works on a real dir
```

`-C` (or `KARL_WORKSPACE`) points the crew's sandbox at any directory —
transcripts and notes still go under `~/.karl`.

---

## 3 · The shell decision (make it once, deliberately)

`run_shell` is **off** by default — wrench will say so and work around it.
Three modes, via `karl config --shell …`:

| mode        | what runs where | when |
|-------------|-----------------|------|
| `off`       | nothing         | default |
| `container` | disposable Docker/Podman container, workspace-only mount, **no network** | recommended on your own machine |
| `host`      | directly on the machine KARL runs on | **only when KARL itself is in a disposable box** |

**The Vast play:** the rented box is disposable by definition, so the clean
setup is to run KARL *on the box* and open the host shell there:

```sh
# on the box:
git clone https://github.com/Ari6six6/KARL.git && cd KARL
./karl-cli config --base-url http://localhost:18080/v1 --model <served-name> --shell host
./karl-cli
```

(Use the port the launch actually bound — `karl gpu status` on your laptop
shows it, or check `~/karl.log` on the box.)

A build inside `container` mode that needs the internet:
`karl config --shell-net bridge`.

---

## 4 · The web, and taint

The public web is **open** to scout by default — no per-domain ceremony.
What is *never* open, and never asks: loopback, LAN, link-local, and
cloud-metadata addresses. Redirects are reported, not followed.

Anything fetched from outside is marked **TAINTED**, and a round that leaned
on it says so in karl's report: treat those facts as unverified until checked.

Prefer an allowlist? `karl config --web gated`, then `karl allow <domain>`,
`karl deny <domain>`, `/allow` to list.

---

## 5 · Tunnel & box management

```sh
karl gpu status      # served URL, model, tunnel live/down
karl gpu reconnect   # laptop slept? tunnel dropped? reopen it — state is saved
karl gpu off         # drop the tunnel, leave the server running on the box
karl gpu down        # stop the model server AND drop the tunnel
karl gpu serve <url> [model]   # point KARL at any endpoint by hand
```

The tunnel dies cleanly after ~90s of a dead link (no zombies), and
`gpu status` will read it as down — `reconnect` brings it back without
re-provisioning anything.

> ⚠ **Billing:** `karl gpu down` stops the *model server*, not the rental.
> The box bills until you **destroy the instance in the Vast console.**

---

## 6 · Troubleshooting, in the order it actually happens

| symptom | meaning | do |
|---|---|---|
| `the box reset the SSH handshake` | box still booting / fail2ban breathing | nothing — KARL retries 4× for you; re-run if it gives up |
| `auth denied` | wrong key on the box | add your key in the Vast console / `ssh-add` |
| `port 8080 is held — sliding to 18080` | Vast's proxy squats 8080 | nothing — the tunnel followed; local side unchanged |
| download bar sits, then `server process died` + log tail | crash on exec, OOM, bad repo | read the tail; smaller model (`gpu model <key>`) or bigger box |
| bar finished but "didn't answer in time" | big model still warming | wait, then `karl ping` / `karl gpu status` |
| `karl ping` fails after it all worked before | tunnel dropped (sleep, wifi) | `karl gpu reconnect` |
| `(the model endpoint didn't respond …)` mid-session | same as above, seen from inside | `karl gpu reconnect`, then re-ask |
| crew talks but never uses tools | server launched without tool-calling | use `karl gpu ssh` (it enables it), or add `--enable-auto-tool-choice --tool-call-parser hermes` to your own vLLM |
| garbled/looping output | small model in a repeat loop | KARL trims it automatically (`…repeat loop trimmed`); lower temperature: `karl config --temperature 0.3` |
| screen theatrics broken over a dumb pipe | not a TTY | expected — off-TTY KARL prints whole plain lines |

---

## 7 · The catalog (VRAM floors)

| key | model | needs | runtime |
|---|---|---|---|
| `glm` *(default)* | GLM-4.7-Flash uncensored FP16 GGUF | ~66 GB | llama.cpp |
| `hermes` | Hermes-4.3-36B FP8 | ~44 GB | vLLM |
| `qwen-official` | Qwen3.6-27B official FP8 | ~30 GB | vLLM |
| `qwen-40b` | Qwen3.6-40B uncensored Q5 GGUF | ~30 GB | llama.cpp |
| `qwen` | Qwen3.6-27B uncensored Q5 GGUF | ~22 GB | llama.cpp |

Context length scales automatically with the VRAM above the floor. Rows are a
starting set — confirm a repo resolves before renting a box for it, and edit
`karl/models.py` to add your own.

---

## 8 · Reference

**Config:** `karl config --base-url U --model M --api-key K --temperature T
--max-tokens N --shell off|container|host --shell-net none|bridge
--web open|gated --stream on|off`

**Env (beats config, per-run):** `KARL_BASE_URL`, `KARL_MODEL`,
`KARL_API_KEY`, `KARL_SHELL`, `KARL_WEB`, `KARL_WORKSPACE`, `KARL_HOME`.

**On disk** — everything is plain files:

```
~/.karl/
  config.json                   # endpoint + settings
  gpu.json                      # box + tunnel state (what reconnect reads)
  gpu-tunnel.log                # why a tunnel refused to open
  projects/<name>/
    workspace/                  # the crew's sandbox
    sessions/<stamp>.jsonl      # every word of every round
    notes.md                    # durable memory
    crew.json                   # your custom crew (after /crew init)
```

On the box: `~/karl.pid`, `~/karl.log`, `~/.karl-venv` (vLLM),
`~/.karl-llama` (llama.cpp), weights in `~/.cache`.

*Drive it like you rented it — because you did.* 🏁
