#!/usr/bin/env python3
"""score the v2 checkpoints against each other, from one cell.

same reason train_v2.py exists: driving colab cell by cell costs hours. this runs the
whole comparison and tees to MyDrive/wag/eval.log, which syncs home, so the numbers are
readable with tail instead of screenshots.

why compare at all instead of just shipping the last checkpoint: v2's held-out loss
bottoms out at the epoch-1 boundary (1.478) and then sits on a plateau around 1.52 for
the rest of the run. 0.04 is too small a gap to decide anything on, and voice adherence
isn't the same axis as loss anyway — a model can keep getting better at sounding like her
after it's stopped getting better at predicting held-out text. so score all three and let
the voice numbers pick.

    python drive_eval.py            # everything
    python drive_eval.py ep1 ep3    # just those
"""
import subprocess
import sys
import time
from pathlib import Path

DRIVE = Path("/content/drive/MyDrive/wag")
LOG = DRIVE / "eval.log"
BASE = "Qwen/Qwen3.5-4B"

# name -> weights. the keep-* dirs are bare lora adapters; eval.py spots the
# adapter_config.json and loads the base underneath by itself.
MODELS = {
    "base": BASE,
    "ep1": str(DRIVE / "keep-step500"),
    "ep2": str(DRIVE / "keep-step1000"),
    "ep3": str(DRIVE / "wag-final-v2"),
}

_logf = LOG.open("a", encoding="utf-8", buffering=1)


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    _logf.write(line + "\n")


def run(*args):
    """one eval.py invocation, output teed. a failure is reported, not fatal — one dead
    checkpoint shouldn't cost you the numbers for the other three."""
    cmd = [sys.executable, "-u", str(DRIVE / "eval.py"), *args]
    say("\n$ " + " ".join(str(c) for c in cmd[2:]))
    p = subprocess.run(cmd, cwd=str(DRIVE), capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    for line in out.splitlines():
        say("  " + line)
    if p.returncode:
        say(f"  !! exit {p.returncode}")
    return p.returncode == 0


wanted = [a for a in sys.argv[1:] if not a.startswith("-")] or list(MODELS)

say("=" * 70)
say(f"wag v2 eval   {time.strftime('%H:%M:%S')}   models: {' '.join(wanted)}")

for name in wanted:
    path = MODELS.get(name)
    if path is None:
        say(f"\n-- {name}: no such model, skipping")
        continue
    if path != BASE and not Path(path).exists():
        say(f"\n-- {name}: {path} isn't there yet, skipping")
        continue

    say("\n" + "=" * 70)
    say(f"-- {name}  ({path})")

    single = f"out/v2_{name}.jsonl"
    multi = f"out/v2_{name}_multi.jsonl"

    if run("gen", "--backend", "hf", "--model", path, "--out", single):
        run("voice", single)
    if run("gen", "--backend", "hf", "--model", path, "--multi", "--out", multi):
        run("convo", multi)

    # the real test of whether the voice is in the weights rather than in the prompt.
    # only worth the gpu time on the tuned ones — the base has no voice to bake in.
    if name != "base":
        nosys = f"out/v2_{name}_nosys.jsonl"
        if run("gen", "--backend", "hf", "--model", path, "--no-system", "--out", nosys):
            run("voice", nosys)

say("\n" + "=" * 70)
say(f"{time.strftime('%H:%M:%S')} done. gen files are in {DRIVE / 'out'}")
