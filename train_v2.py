#!/usr/bin/env python3
"""wag v2 SFT — the training notebook as one script, for the same reason generation is.

driving colab cell by cell cost hours last night. this runs from a single cell and tees
everything to MyDrive/wag/train.log, which syncs home, so progress is readable with tail
instead of screenshots.

the logic is train.ipynb's, unchanged except where v2 differs:
  base      Qwen3.5-4B, not 2B
  seq len   3072, because encode() drops over-length rows rather than truncating and
            1024 would quietly bin 171 of them — most of the long-input slice
  lora      decided by vram, since a 4B full fine-tune needs ~48GB before activations
            and colab hands out both 40GB and 80GB A100s without asking

    python train_v2.py            # train
    python train_v2.py --smoke    # load, tokenise, report, and stop before training
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch

DRIVE = Path("/content/drive/MyDrive/wag")
OUTDIR = DRIVE / "ckpt-v2"
FINAL = DRIVE / "wag-final-v2"
LOG = DRIVE / "train.log"

BASE = os.environ.get("WAG_BASE", "Qwen/Qwen3.5-4B")
EPOCHS = int(os.environ.get("WAG_EPOCHS", "3"))
SEQ_LEN = int(os.environ.get("WAG_SEQ", "3072"))
BATCH = int(os.environ.get("WAG_BATCH", "2"))
ACCUM = int(os.environ.get("WAG_ACCUM", "8"))
SAVE_STEPS = int(os.environ.get("WAG_SAVE", "100"))
GC = os.environ.get("WAG_GC", "") == "1"
SMOKE = "--smoke" in sys.argv

IGNORE = -100
_logf = LOG.open("a", encoding="utf-8", buffering=1)


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    _logf.write(line + "\n")


say("=" * 70)
say(f"wag v2 sft   base={BASE}   {time.strftime('%H:%M:%S')}   smoke={SMOKE}")

gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
# a 4B full fine-tune is ~8GB bf16 weights + 8GB grads + 32GB of fp32 adam moments before
# a single activation. that fits the 80GB card and nothing else, and which one you get is
# not up to you — so it's measured rather than assumed.
USE_LORA = vram < 60
BF16 = "T4" not in gpu
LR = 2e-4 if USE_LORA else 2e-5
say(f"gpu: {gpu} ({vram:.0f} GB)  ->  lora={USE_LORA}  bf16={BF16}  lr={LR}")

# ---------------------------------------------------------------- data
rows = [json.loads(l) for l in (DRIVE / "train.jsonl").open(encoding="utf-8")]
assert len(rows) > 5000, f"that looks like v1's data, not v2 ({len(rows)} rows)"
multi = sum(1 for r in rows if sum(m["role"] == "assistant" for m in r["messages"]) > 1)
say(f"{len(rows)} rows   multi-turn {multi} ({100*multi/len(rows):.0f}%)")
say("  slices: " + "  ".join(f"{k}:{v}" for k, v in
                             Counter(r.get("category", "?") for r in rows).most_common()))

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)


DROPS = Counter()


def encode(rec):
    """mask the user's turns, supervise every one of wag's.

    the v1 notebook supervised only the LAST assistant turn and masked everything before
    it. fine when 14% of rows were multi-turn; v2 is 54%, so it threw away five of wag's
    six replies in a six-exchange conversation — the smoke test put supervision at 31%.

    spans are found by character offset rather than by tokenising prefixes and trusting
    the lengths to line up. that trust was misplaced: qwen3.5's template injects an empty
    <think></think> into assistant turns that have no reasoning_content, so the rendered
    prefix is NOT a prefix of the rendered whole, and an exact-match check threw away
    7,811 of 8,242 rows. offsets don't care.
    """
    msgs = rec["messages"]
    full = tok.apply_chat_template(msgs, tokenize=False)
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    f_ids, offs = enc["input_ids"], enc["offset_mapping"]
    if len(f_ids) > SEQ_LEN:
        DROPS["over SEQ_LEN"] += 1
        return None

    labels = [IGNORE] * len(f_ids)
    cursor = 0
    for m in msgs:
        if m["role"] != "assistant":
            # step the cursor past it so a later turn can't match earlier text
            at = full.find(m["content"], cursor) if m["content"] else -1
            if at >= 0:
                cursor = at + len(m["content"])
            continue
        at = full.find(m["content"], cursor)
        if at < 0:
            DROPS["assistant text not found in render"] += 1
            return None
        end = at + len(m["content"])
        # take the closing <|im_end|> with it — that's what teaches it to stop
        tail = full.find("<|im_end|>", end)
        if 0 <= tail <= end + 4:
            end = tail + len("<|im_end|>")
        for j, (s0, s1) in enumerate(offs):
            if s1 > at and s0 < end:
                labels[j] = f_ids[j]
        cursor = end

    if all(x == IGNORE for x in labels):
        DROPS["nothing supervised"] += 1
        return None
    return {"input_ids": f_ids, "labels": labels, "attention_mask": [1] * len(f_ids)}


encoded = [e for e in (encode(r) for r in rows) if e]
dropped = len(rows) - len(encoded)
lens = sorted(len(e["input_ids"]) for e in encoded)
sup = sum(sum(1 for x in e["labels"] if x != IGNORE) for e in encoded)
tot = sum(len(e["labels"]) for e in encoded)
say(f"encoded {len(encoded)}   dropped {dropped} over {SEQ_LEN} tokens")
say(f"  length: median {lens[len(lens)//2]}  p95 {lens[int(len(lens)*0.95)]}  max {lens[-1]}")
say(f"  supervised tokens: {sup}/{tot} ({100*sup/tot:.0f}%)  <- want roughly 45-70%")
if DROPS:
    say("  drops: " + "  ".join(f"{k}={v}" for k, v in DROPS.most_common()))
if dropped > len(rows) * 0.02:
    say(f"!! {dropped} rows dropped — that's more than 2%, check SEQ_LEN")

from dataclasses import dataclass  # noqa: E402

from datasets import Dataset  # noqa: E402

ds = Dataset.from_list(encoded).train_test_split(test_size=0.02, seed=20260823)
say(f"  train {len(ds['train'])}  eval {len(ds['test'])}")


@dataclass
class PadCollator:
    pad_id: int

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        out = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in feats:
            gap = n - len(f["input_ids"])
            out["input_ids"].append(f["input_ids"] + [self.pad_id] * gap)
            out["labels"].append(f["labels"] + [IGNORE] * gap)
            out["attention_mask"].append(f["attention_mask"] + [0] * gap)
        return {k: torch.tensor(v) for k, v in out.items()}


collator = PadCollator(tok.pad_token_id or tok.eos_token_id)

if SMOKE:
    say("\nsmoke only — stopping before the model loads")
    say(tok.apply_chat_template(rows[0]["messages"], tokenize=False)[:600])
    raise SystemExit(0)

# ---------------------------------------------------------------- model
from transformers import AutoModelForCausalLM  # noqa: E402

dtype = torch.bfloat16 if BF16 else torch.float16
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=dtype, trust_remote_code=True)

frozen = 0
for name, p in model.named_parameters():
    if re.search(r"vision|visual|image_|video_|patch_embed", name, re.I):
        p.requires_grad = False
        frozen += p.numel()
if frozen:
    say(f"froze {frozen/1e6:.0f}M vision params")

model.config.use_cache = False
# checkpointing was off for v1 because qwen3.5's linear-attention layers raised
# CheckpointError on recompute. that was on the 2B, with reentrant checkpointing — the
# variant that trips over a recomputed pass seeing a different shape than the one it
# saved. non-reentrant doesn't work that way, so it's worth another look here: without
# it a 4B at batch 1 is sitting on ~29GB of activations, because the delta-rule path
# runs unfused (no causal_conv1d wheel) and keeps every intermediate for backward.

if USE_LORA:
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

if GC and USE_LORA:
    # the backbone is frozen, so the first checkpointed block gets an input with no grad
    # path and quietly produces nothing to backprop. this puts the path back.
    model.enable_input_require_grads()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
say(f"trainable: {trainable/1e9:.3f}B   gradient_checkpointing={GC}")

from transformers import Trainer, TrainingArguments  # noqa: E402

args = TrainingArguments(
    output_dir=str(OUTDIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_steps=0.05,
    logging_steps=10,
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps=SAVE_STEPS,
    bf16=BF16,
    fp16=not BF16,
    optim="adamw_torch_fused" if not USE_LORA else "adamw_torch",
    gradient_checkpointing=GC,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
    seed=20260823,
)

trainer = Trainer(model=model, args=args, train_dataset=ds["train"],
                  eval_dataset=ds["test"], data_collator=collator)

ckpts = sorted(OUTDIR.glob("checkpoint-*"),
               key=lambda p: int(p.name.split("-")[1])) if OUTDIR.exists() else []
say(f"resuming from {ckpts[-1].name}" if ckpts else "starting fresh")
say(f"steps/epoch ~= {len(ds['train']) // (BATCH * ACCUM)}")

trainer.train(resume_from_checkpoint=bool(ckpts))

if USE_LORA:
    model = model.merge_and_unload()     # bake the adapter in so gguf export is simple
model.save_pretrained(FINAL, safe_serialization=True)
tok.save_pretrained(FINAL)
say(f"\nsaved -> {FINAL}")
