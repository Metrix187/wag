#!/usr/bin/env python3
"""gguf export for wag v2, in one cell.

same shape as train_v2.py and drive_eval.py: everything tees to MyDrive/wag/gguf.log so
progress is readable with tail instead of screenshots.

this is v1's recipe (train.ipynb cells 20-21) with two things re-checked, because v2 isn't
the same checkpoint shape as v1:

  architectures   v1 saved a VLM wrapper (Qwen3_5ForConditionalGeneration) because it
                  loaded through the image-text class. train_v2.py used
                  AutoModelForCausalLM, so v2's config says Qwen3_5ForCausalLM. that's a
                  different registration in the converter, so it gets grepped before the
                  build rather than discovered after it.
  block_count     still broken the same way. qwen3.5 normally ships a multi-token-
                  prediction head, so the converter writes block_count =
                  num_hidden_layers + mtp_num_hidden_layers = 33. our checkpoint has no
                  mtp head (save_pretrained never wrote one) while config.json still says
                  mtp_num_hidden_layers: 1, so the header promises 33 blocks over 32
                  blocks of tensors and the loader dies looking for blk.32.attn_norm.weight.
                  fix_gguf_blocks.py re-points the u32s; run it on the f16 and the quants
                  inherit the fix.

f16 and the quants are built on /content (local disk) and only copied to Drive at the end.
quantizing straight onto the Drive mount means pushing 8GB up and pulling it back down for
every quant, which is slow for no reason.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

DRIVE = Path("/content/drive/MyDrive/wag")
SRC = DRIVE / "wag-final-v2"
OUT = DRIVE / "gguf-v2"
WORK = Path("/content/gguf-build")
LLAMA = Path("/content/llama.cpp")
LOG = DRIVE / "gguf.log"

# v0.2.0, their first semver-stable tag. same pin v1 shipped on.
LLAMA_SHA = "5a32f7b66ef6cfb3e60deea26e3454cc6ad3438c"
QUANTS = ["q8_0", "q4_k_m"]

_logf = LOG.open("a", encoding="utf-8", buffering=1)


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    _logf.write(line + "\n")


def sh(cmd, cwd=None, quiet=False):
    say(f"\n$ {cmd}")
    p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    lines = out.splitlines()
    # builds are thousands of lines of nothing; keep the tail so a failure is still readable
    for line in (lines[-25:] if quiet and len(lines) > 25 else lines):
        say("  " + line)
    if p.returncode:
        say(f"  !! exit {p.returncode}")
    return p.returncode == 0


say("=" * 70)
say(f"wag v2 gguf   {time.strftime('%H:%M:%S')}")

cfg = json.loads((SRC / "config.json").read_text(encoding="utf-8"))
arch = cfg["architectures"][0]
layers, mtp = cfg["num_hidden_layers"], cfg.get("mtp_num_hidden_layers", 0)
say(f"source: {SRC}")
say(f"  arch {arch}   layers {layers}   mtp {mtp}  -> header will claim {layers + mtp}")

WORK.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- llama.cpp
if not LLAMA.exists():
    if not sh(f"git clone --quiet https://github.com/ggml-org/llama.cpp {LLAMA}"):
        sys.exit("clone failed")
if not sh(f"git checkout --quiet {LLAMA_SHA}", cwd=LLAMA):
    sys.exit("checkout failed — is the pin still there?")

# check the converter knows this architecture BEFORE spending build time on it
say("\n-- does the converter register this arch?")
sh(f"grep -rn '{arch}' conversion/ | head -5", cwd=LLAMA)

# heads up: this downgrades transformers to 4.x and numpy to 1.26. nothing after it in this
# script touches them, but don't run training in the same session afterwards.
sh("pip install -q -r requirements.txt", cwd=LLAMA, quiet=True)
sh("cmake -B build -DGGML_CUDA=OFF", cwd=LLAMA, quiet=True)
sh("cmake --build build --config Release -j --target llama-quantize", cwd=LLAMA, quiet=True)

quant_bin = LLAMA / "build" / "bin" / "llama-quantize"
if not quant_bin.exists():
    sys.exit("llama-quantize didn't build — see the tail above")
say(f"\nbuilt {quant_bin}")

# ---------------------------------------------------------------- convert
f16 = WORK / "wag-v2-f16.gguf"
if not f16.exists():
    if not sh(f"python convert_hf_to_gguf.py {SRC} --outfile {f16} --outtype f16",
              cwd=LLAMA):
        sys.exit("conversion failed")

say("\n-- patching block_count before quantizing, so the quants inherit it")
sh(f"python {DRIVE}/fix_gguf_blocks.py {f16}")

# ---------------------------------------------------------------- quantize
made = [f16]
for q in QUANTS:
    dest = WORK / f"wag-v2-{q}.gguf"
    if dest.exists():
        say(f"\n{dest.name} already here, skipping")
        made.append(dest)
        continue
    if sh(f"{quant_bin} {f16} {dest} {q}", quiet=True):
        made.append(dest)
    else:
        say(f"  !! {q} failed, carrying on with the rest")

# llama-quantize copies the kv block across, so this should say "already fine".
# cheap enough to check rather than assume.
say("\n-- verifying the quants kept the fix")
sh(f"python {DRIVE}/fix_gguf_blocks.py " + " ".join(str(m) for m in made if m != f16))

# ---------------------------------------------------------------- ship
say("\n-- copying to drive")
for m in made:
    dest = OUT / m.name
    if dest.exists() and dest.stat().st_size == m.stat().st_size:
        say(f"  {m.name} already on drive, same size")
        continue
    say(f"  {m.name} -> {dest}  ({m.stat().st_size / 1e9:.2f} GB)")
    shutil.copy2(m, dest)

sh(f"ls -lh {OUT}")
say(f"\n{time.strftime('%H:%M:%S')} done")
