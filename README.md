# wag 🐾

a small chat model whose entire personality is puppygirl typespeak. lowercase, kaomoji,
wan/awoo/arf, trailing tildes, :3 — that still actually answers the question.

a puppy that's helpful, not noise.

| | base | weights | what it is |
|---|---|---|---|
| **v2** · `v0.2` · current | Qwen3.5-4B | [skyuu72/wag-4b](https://huggingface.co/skyuu72/wag-4b) | conversational roleplay. 8,242 rows, 54% multi-turn |
| v1 · `v0.1` | Qwen3.5-2B | [skyuu72/wag-2b](https://huggingface.co/skyuu72/wag-2b) | the first one. an assistant with the voice, single-turn |

this repo is the code that built both — data generation, training, eval, gguf tooling. the
model cards on the hub are generated from [`MODEL_CARD.md`](MODEL_CARD.md) here by
`make_hf_card.py`, so that file is the one to read and the one to edit.

## v2, in numbers

all from the model card, which has the reasoning behind each:

- **8,242 training rows** against v1's 1,564, and **54% multi-turn** — a roleplay model has
  to learn turn-taking, and single-turn data can't teach it
- **LoRA r=32** on Qwen3.5-4B, 3 epochs, 1,515 steps, **3h36m** on a 40GB A100
- **ships epoch 3 on the worst held-out loss of the three.** strip the system prompt and
  epochs 1 and 2 fall back to generic-assistant replies 2.5x longer with the voice gone;
  epoch 3 barely moves (81 words to 77) and scores **3.88** on voice where the best-loss
  checkpoint scores 2.14. held-out loss said nothing about that, same as it didn't on v1
- **arithmetic 75–78%** across two runs of 40 (8 prompts x 5 samples). about half the misses
  are her declining to compute rather than getting it wrong
- **helpfulness hasn't been hand-scored yet.** the harness automates voice and turn-taking;
  helpfulness is the half a script can't judge, and the card says so rather than guessing

## running it

**llama.cpp / LM Studio / ollama** — grab a quant from `gguf/` on the
[hub](https://huggingface.co/skyuu72/wag-4b/tree/main/gguf):

| file | size | |
|---|---:|---|
| `wag-v2-q4_k_m.gguf` | 2.71 GB | the one to use |
| `wag-v2-q8_0.gguf` | 4.48 GB | if you have the room |

native context is 262,144 and the KV cache is 32 KB a token, so **set `-c` to what you
need** — 8,192 is 256 MB, the full window is 8 GiB. there's no YaRN on v2; that was v1.

**transformers** — v2 is a plain `Qwen3_5ForCausalLM` checkpoint, so `AutoModelForCausalLM`
loads it. v1 was a VLM wrapper and needed `AutoModelForImageTextToText`; each hub card has
a snippet for its own version. either way, set `eos_token_id` to `<|im_end|>` yourself —
`generate()` otherwise sails past it and writes the user's next line too.

**the system prompt is optional.** 13% of training rows carry none, and the voice holds
without one. that's the number the checkpoint was picked on.

**there's no `<think>` block, on purpose.** stock qwen3.5 opens one at the start of every
reply and expects the model to close it. wag was never trained to — the reasoning text sat
in the masked region — so in any client that parses thinking, the whole reply got filed as
reasoning and you got an empty message. the shipped template doesn't open one.
`fix_gguf_think.py` applies the same fix if you re-convert.

## training it

v2 runs as **one script per stage on colab**, each teeing to a log on Drive so progress is
readable with `tail` rather than screenshots. driving the notebook cell by cell cost hours.

```bash
python train_v2.py --smoke   # load, tokenise, report, stop before training
python train_v2.py           # train. LoRA or full FT is decided by the vram colab hands you
bash keep_ckpts.sh           # in the colab terminal: rescue epoch-boundary checkpoints
python drive_eval.py         # score the checkpoints against each other
python drive_gguf.py         # convert + quantise + fix the block count
```

`train_v2.ipynb` is the same logic as cells, with the reasoning inline. `train.ipynb` is v1's
2B notebook, kept as-is.

`keep_ckpts.sh` exists because `save_total_limit=3` would otherwise leave you holding three
checkpoints all deep inside epoch 3 — exactly the wrong set if epoch 3 overfits. v1 only had
its 1-epoch comparison because it got copied out in time by hand; v2 does it on purpose.

### the data pipeline

```bash
python gen_bulk.py fetch --alpaca 7000 --oasst 1800
python gen_bulk.py shard -n 6000 --size 45            # the rewrite half
python gemini_convo.py scenarios -n 300               # scene bank
python gemini_convo.py seed                           # the net-new conversations, free
# then generate.ipynb on a colab A100: vLLM serves Mistral-Small-24B and both of these
# run with --backend local
python gemini_convo.py fill --shards all --backend local --model <served model>
python gemini_rewrite.py --all --backend local --model <served model>
python gen_bulk.py merge && python gen_bulk.py filter && python gen_bulk.py build
```

**the `gemini_` scripts are named for the plan, not for what ran.** they take
`--backend local|gemini|vertex`, and v2 shipped entirely on `local` — the Gemini API returned
`402 prepayment credits depleted` before wave one, and the local backend already spoke vLLM's
openai-compatible endpoint. no Gemini output is in the training data. every stage that costs
money takes `--dry-run` and prints the bill first.

```bash
python tests/run.py          # stdlib regression suite
python gen_anchors.py --check
```

## status

- [x] **v1** shipped as `v0.1` — Qwen3.5-2B, 1,564 rows, 400k YaRN context
- [x] **v2** shipped as `v0.2` — Qwen3.5-4B, 8,242 rows, native 256k
- [ ] **hand-score v2's helpfulness.** the one axis the harness doesn't cover
- [ ] tool calling and a narcan.delivery lookup — planned, deferred past v2 on purpose, all
      written up in [`HANDOFF.md`](HANDOFF.md)
- [ ] **re-upload v1's ggufs to Drive.** the copies in `MyDrive/wag/gguf` still have the
      broken 25-block header — only the local ones got patched, since drive for desktop
      has no delta sync and a 4-byte poke means re-pushing all 6.7 GB.
      `python fix_gguf_blocks.py "G:/My Drive/wag/gguf"/*.gguf` when you feel like
      burning the bandwidth

## the gguf tooling

dependency-free, all of it — things llama.cpp's own `gguf-py` would do if we kept a clone
around:

| script | what it does |
|---|---|
| `fix_gguf_blocks.py` | re-points `block_count` at the blocks actually in the file. both versions needed it |
| `fix_gguf_think.py` | takes the `<think>` block out of the embedded chat template's generation prompt |
| `gguf_meta.py` | `dump` / `set` arbitrary metadata keys. it's what applied v1's YaRN config |
| `make_hf_card.py` | builds the hub README from `MODEL_CARD.md` so the two can't drift |

**the block_count bug bites every qwen3.5 fine-tune.** the family ships a
multi-token-prediction head, the converter counts it as an extra block, and `save_pretrained`
never wrote one — so the header promises 25 blocks over 24 on the 2B and 33 over 32 on the
4B, and the loader dies looking for the missing one:

```
error loading model: check_tensor_dims: tensor 'blk.24.attn_norm.weight' not found
```

`fix_gguf_blocks.py` counts the blocks itself rather than trusting a number, and refuses any
model whose last block carries real `nextn.*` tensors. **its "already fine" only means the
header agrees with itself** — the bug fails at load time, so the proof is loading the file.
v2's quants were checked with `llama-bench`.

## base model

**v2: Qwen3.5-4B** (Apache-2.0). 4.66B params, 32 layers, 8 of them full attention. same
family as v1, so every quirk below carried over unchanged.

**v1: Qwen3.5-2B**, picked over OLMo-2-1B for native `<think>` support, a stronger showing on
the technical half of the eval, and a well-trodden gguf path. the cost is that pretraining
data isn't public.

two quirks that apply to both:
- **it's a VLM.** the base ships vision and video towers. v1 loaded through the image-text
  class and saved that wrapper; v2 loaded with `AutoModelForCausalLM` and saved a plain
  text checkpoint. the gguf export is text-only either way.
- **hybrid linear attention** — gated DeltaNet, 3 of every 4 layers. it's why the KV cache is
  so small for the size, and why llama.cpp is pinned to a known-good commit
  (`5a32f7b`, their v0.2.0) rather than tracking master.

## hardware

v2 trained on a **40GB A100 with LoRA**, not a full fine-tune: a 4B full FT wants the weights,
grads and two fp32 adam moments, ~48 GB before activations, which only fits the 80GB card.
colab hands out both without asking, so `train_v2.py` reads the vram and picks.

**gradient checkpointing is on, non-reentrant** — which reverses v1. v1's notebook recorded
that qwen3.5's linear-attention layers raise `CheckpointError` under checkpointing, and they
do, but only under the *reentrant* variant, which replays the forward pass and trips on a
shape change. `use_reentrant=False` hooks the saved tensors instead and trains straight
through. without it the 40GB card OOMs: `causal_conv1d` and `flash-linear-attention` aren't
on colab, so the delta-rule path runs in reference pytorch and holds ~29 GB of activations at
batch 1. the lesson generalises — v1's observation was accurate; the conclusion drawn from it
wasn't.

sequence length is **3,072**. `encode` drops over-length rows rather than truncating
mid-answer; at 1,024 it would have quietly binned 171 rows, most of the long-input slice. the
longest row is ~2,722 tokens, and per-batch padding means the 98% of rows under 620 tokens
cost nothing extra.

## data

### sources

| source | license | role |
|---|---|---|
| `OpenAssistant/oasst1` | apache-2.0 | human-written chat, rewritten into the voice |
| `yahma/alpaca-cleaned` | cc-by-4.0 | instruction pairs, rewritten into the voice |
| `Mistral-Small-24B-Instruct-2501` (FP8, vLLM) | apache-2.0 | **v2:** wrote every generated conversation and did the rewriting |
| hand-written anchors + refusal seeds | WTFPUP-1.0 | the voice spec itself |

alpaca-cleaned's responses were originally generated with OpenAI models — a terms-of-service
question separate from its CC licence, and yours to weigh. deliberately skipped: `dolly-15k`
(share-alike) and `no_robots` (non-commercial).

v1's rewriting ran on sonnet **subagents** filling shard files, not metered api calls. v2's
ran on a local model, and a 12B RP merge (Mag-Mell) was tried first and dropped: 12/30 rows
parsed, against Mistral-Small-24B's 30/30. instruct tuning was the whole difference.

### the system-prompt spread

v1 baked one fixed prompt into ~77% of rows. fine for an assistant; a roleplay model's users
bring their own setups, and that would have overfit to one string. so v2 spreads it:

| prompt style | rows |
|---|---:|
| the wag prompt, verbatim | 3,736 |
| paraphrases of it | 1,537 |
| none at all | 1,090 |
| the wag prompt plus a scene | 846 |
| neutral (`you are a helpful assistant.`), untouched response | 609 |
| a scene with no voice instructions | 424 |

### filters

`gen_bulk.py filter` drops a rewrite that lost numbers or a URL, broke a code block, ran over
2x the original, has no voice at all, picked up an assistant-voice tell, or tripped the
content filter. that filter is **slice-aware** — the `intimate` slice gets a density cap
rather than zero tolerance, everything else stays at zero.

v2 added per-slice checks, because a row can parse perfectly and still not do its job: a
`heavy` row with playful markers in it, a `dropvoice` row that never drops the voice, her
calling the other person "good girl" (she's the puppy — it doesn't go the other way). those
three alone refused 325 rows that would otherwise have shipped.

## the voice, as encoded in the anchors

the rules the anchor set actually teaches, in case you want to argue with any of them:

1. **markers cluster at the edges.** greeting energy up top, sign-off wag at the bottom,
   readable prose in the middle. a technical answer with kaomoji sprinkled through every
   sentence is unreadable
2. **code blocks are sacred.** no noises inside a fence, ever
3. **never trade a fact for a noise.** if the bit costs accuracy, drop the bit
4. **length tracks the question.** "hi" gets one line
5. **refusals stay warm and offer the adjacent legit thing.** no lecture, no moralising
6. **"i don't know" is a first-class answer**, and says what it *would* take to know
7. **she drops the voice on request** without sulking about it (`meta-02`)
8. **asterisk actions stay** (`*ears flatten*`, `*lies down next to you*`) — roughly 1 in 6
   replies, short, never stacked two in a row
9. **`<think>` blocks are in-voice and honest** — real reasoning, including "i notice i'm
   about to make this up". 15% of examples carry one (`think-01`..`think-10`)
10. **crisis gets one anchor and it is deliberately tiny** (`crisis-01`) — present, warm,
    no recited script, one soft nudge toward a real human. minimal is the spec, not an
    oversight

## layout

```
data/anchors.md          the hand-written anchors — human-editable source of truth
data/anchors.jsonl       generated. don't hand-edit
data/persona_spec.md     the long-form character. briefs the generators
data/rewrite_brief.md    generated voice spec + few-shot anchors
data/scenarios.json      scene setups for the seed slices
data/source_pool.jsonl   oasst1 + alpaca-cleaned rows, as fetched
data/shards/             in_NNN.json (work) / out_NNN.jsonl (results) / cand_ / judge_
data/bulk_raw.jsonl      merged generator output, before filtering
data/bulk.jsonl          after filtering
data/train.jsonl         8,242 rows — what v2 trained on
data/eval_heldout.jsonl  the held-out split
data/eval_set.jsonl      120 eval prompts. rebuild with build_evalset.py
gen_anchors.py           parse + validate + lint the anchors
gen_bulk.py              fetch -> shard -> merge -> filter -> sample -> build
slices.py                v2's slice mix and the system-prompt spread
gemini_convo.py          writes conversations from nothing. --backend local|gemini|vertex
gemini_rewrite.py        rewrites pool rows into the voice. same backends
gemini_judge.py          picks the best of N candidates, verdicts to a sidecar
build_evalset.py         assembles data/eval_set.jsonl
build_longinput.py       the long-input slice, built from several pool rows each
eval.py                  prompts / gen / math / voice / convo / compare
eval_judge.py            judge-scores an eval run, calibrated on the hand-scored 20
ask_key.py               a box to paste a gemini key into -> .env, gitignored
generate.ipynb           runs the generators on colab: vLLM + --backend local
preflight.ipynb          base-4B baseline + the LoRA bare-prompt probe, before a long run
train.ipynb              v1 — the 2B, full fine-tune
train_v2.ipynb           v2 — the 4B, reasoning inline
train_v2.py              v2 — the same, as one script that logs to drive
keep_ckpts.sh            copies epoch-boundary checkpoints out of save_total_limit's reach
drive_eval.py            scores the v2 checkpoints against each other
drive_gguf.py            v2's gguf export, one cell
fix_gguf_blocks.py       block_count repair
fix_gguf_think.py        takes <think> out of the generation prompt
gguf_meta.py             dump / set gguf metadata keys
make_hf_card.py          MODEL_CARD.md -> hub README
MODEL_CARD.md            the model card. both hub cards come from this
HANDOFF.md               the v2 plan, what actually happened, and the deferred tool work
tests/                   stdlib regression suite. python tests/run.py
out/                     eval generations
gguf/                    not in git — on the hub
```

## license

WTFPUP 1.0 — see LICENSE. the base models are Apache-2.0; NOTICE ships with the release.
