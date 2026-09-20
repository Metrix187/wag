# wag 🐾

a small chat model whose entire personality is puppygirl typespeak. lowercase, kaomoji,
wan/awoo/arf, trailing tildes, :3 — that still actually answers the question.

a puppy that's helpful, not noise.

**v0.1** — weights and quants live on the hub at
[skyuu72/wag-2b](https://huggingface.co/skyuu72/wag-2b). this repo is the code that
built them: data generation, training notebook, eval harness, gguf tooling.

## status

- [x] base model picked
- [x] 67 anchors written and approved (10 with `<think>`, 15%)
- [x] bulk set — 1,774 rewrites across 40 shards, 1,617 kept after filtering
- [x] marker top-up pass — `awoo` went from 1 row to 25, `mrrp`/`hmf`/`;;` from zero
- [x] `data/train.jsonl` (1,564) + `data/eval_heldout.jsonl` (120)
- [x] llama.cpp pinned to v0.2.0 (`5a32f7b`), arch registration verified
- [x] model card drafted — eval table still stubbed
- [x] training run — 288 steps, ~33 min on an A100, ~4 compute units
- [x] eval — base 1.60 / wag 3ep **3.95** / wag 1ep 3.60 on helpfulness (0-5, 20 prompts)
- [x] gguf exports — `gguf/wag-q4_k_m.gguf` (1.19 GB), `gguf/wag-q8_0.gguf` (1.87 GB)
- [x] model card with the full eval table and the failure cases
- [x] gguf actually loads — the converter claimed 25 blocks over 24, see below
- [x] context extended to **400k** with YaRN (native is 256k), shipped enabled
- [x] v0.1 published — [hub](https://huggingface.co/skyuu72/wag-2b) + this repo
- [ ] **v2 in planning** — conversational roleplay puppygirl on a Qwen3.5-**4B** base.
      plan, budget and the deferred tool-calling work all live in
      [`HANDOFF.md`](HANDOFF.md); the character itself is
      [`data/persona_spec.md`](data/persona_spec.md)
- [ ] **re-upload the ggufs to Drive.** the copies in `MyDrive/wag/gguf` still have the
      broken 25-block header — only the local ones got patched, since drive for desktop
      has no delta sync and a 4-byte poke means re-pushing all 6.7 GB.
      `python fix_gguf_blocks.py "G:/My Drive/wag/gguf"/*.gguf` when you feel like
      burning the bandwidth

### running it

1. drop `data/train.jsonl` into Drive at `MyDrive/wag/train.jsonl`
2. open `train.ipynb` in colab, runtime → change runtime type → **A100**
3. Run All. ~33 min of actual training on top of the model download
4. checkpoints land in `MyDrive/wag/ckpt` every 50 steps and resume by themselves,
   so a dead session costs you nothing. they're ~12 GB each though (weights 4.4 +
   optimizer 7.5), and 3 are kept — budget ~40 GB of drive for a run

### running the gguf

point LM Studio (or anything else on llama.cpp) at `gguf/wag-q4_k_m.gguf` and go. tested on
LM Studio's 2.29.1 CUDA 12 runtime, both quants.

**pass `-c` if you're on bare llama.cpp.** the trained context is 400,000 after the YaRN
pass, and `-c` defaults to exactly that, so `llama-server -m wag-q4_k_m.gguf` reaches for
4.6 GiB of KV cache before printing anything. `-c 8192` for normal chat. LM Studio picks
its own smaller default and is fine.

if you ever reconvert, run `fix_gguf_blocks.py` over the output first. qwen3.5 normally has
a multi-token-prediction head and the converter counts it as a 25th block; our checkpoint
doesn't have one, so the header ends up promising a `blk.24` that isn't in the file:

```
error loading model: check_tensor_dims: tensor 'blk.24.attn_norm.weight' not found
```

```bash
python fix_gguf_blocks.py gguf/*.gguf
```

two u32s in the kv block, tensor data untouched. the shipped files already have it applied.
it's wired into `train.ipynb` too, so a fresh run comes out loadable.


### the gguf tooling

two small scripts, both dependency-free, both doing things llama.cpp's own `gguf-py` would
do if we kept a clone around:

| script | what it does |
|---|---|
| `fix_gguf_blocks.py` | re-points `block_count` at the blocks that are actually in the file |
| `gguf_meta.py` | `dump` / `set` arbitrary metadata keys — this is what applied the YaRN config |
| `make_hf_card.py` | builds the hub README from `MODEL_CARD.md` so the two can't drift |

```bash
python gguf_meta.py dump gguf/wag-q4_k_m.gguf "rope|context"
python gguf_meta.py set gguf/wag-q4_k_m.gguf qwen35.context_length:u32=400000
```

`set` rewrites the file when it has to add a key, since that grows the header — tensor data
gets copied over untouched and the directory offsets stay valid because they're relative to
the start of the data section.


## base model

**Qwen3.5-2B** (Apache 2.0). 2.27B params, 4.55 GB bf16.

why it over OLMo-2-1B:
- native `<think>` mode, which the chat template already plumbs via `reasoning_content`.
  a puppy that half-monologues in think tags is only free if the base supports it
- meaningfully smarter at the technical/code half of the eval set, and "still correct and
  helpful" is one of our two scoring axes
- gguf conversion is a well-trodden path — bartowski and unsloth both ship q4_k_m and q8_0
  of this exact model, so the export deliverable isn't a research project

what we give up: pretraining data isn't public, so no "fully inspectable stack" story.
if that matters more than raw capability, OLMo-2-0425-1B-Instruct is the swap and only
`train.ipynb` changes.

two quirks worth knowing:
- it's a **VLM** (`Qwen3_5ForConditionalGeneration`, vision + video towers). we train text-only
  and leave the vision tower frozen. gguf text export is unaffected
- **hybrid linear attention** (gated DeltaNet, `layer_types` alternates linear/full). llama.cpp
  supports it but has open bugs on the bigger variants — pin a llama.cpp commit known to
  convert the 2B cleanly rather than tracking master

## hardware

the brief's "full fine-tune fits a T4" doesn't survive contact with the numbers. T4 is Turing:
**no bf16 at all** (fp16 only), no flash-attn-2, 16 GB. full FT of 2.27B needs bf16 weights
4.5 + grads 4.5 + 8-bit adam states 4.5 + fp32 master 9.1 ≈ 22.8 GB before activations.

with colab pro:

| gpu | vram | plan | why |
|---|---|---|---|
| **A100** | 40 GB | **full fine-tune** ← default | 22.8 GB fits with real headroom |
| L4 | 24 GB | LoRA r=32, or full FT w/ adafactor | full FT + adam is ~1 GB over |
| T4 | 16 GB | LoRA only | no bf16, not enough room |

for a *style* transfer, LoRA is not a downgrade — it's less likely to nuke the base model's
helpfulness, which is exactly the failure mode we're watching for. full FT is the default
because the A100 makes it free, not because it's obviously better.

expected runtime, ~2000 examples x 3 epochs, packed seq 1024:
**~5 min of compute, 15–30 min wall clock** on A100 including setup and model pull.
roughly 6 compute units of the monthly 100.

## data

### sources (licenses checked 2026-08-23)

| dataset | license | role |
|---|---|---|
| `OpenAssistant/oasst1` | apache-2.0 | human-written, primary |
| `yahma/alpaca-cleaned` | cc-by-4.0 | attribution only — commercial use fine |
| hand-written refusal seeds | ours | oasst/alpaca have almost no refusals |

deliberately skipped: **dolly-15k** (cc-by-sa-3.0 — share-alike would infect our dataset
license) and **no_robots** (cc-by-nc-4.0 — non-commercial). one caveat to put in the model
card: alpaca-cleaned's responses were originally generated with openai models, which is a
ToS question separate from its CC licence. oasst1 is human-written and has no such asterisk.

### the system-prompt split

the brief says "bake one system prompt into every example" *and* "~10% plain examples so it
doesn't collapse into noises". those two fight each other — a plain answer sitting under a
"speak in puppyspeak" system prompt teaches the model to ignore its own system prompt.

so it's a three-way split instead, which gets both properties without the contradiction:

| kind | share | system prompt | response |
|---|---|---|---|
| voiced | 80% | the wag prompt | in-voice |
| **bare** | 10% | **none at all** | in-voice |
| plain | 10% | `you are a helpful assistant.` | untouched original |

the **bare** slice is what actually delivers "always-on with an empty system prompt" — the
model learns the voice is its identity, not an instruction it's following. the plain slice
keeps raw answer quality alive and gives an escape hatch that's consistent rather than
contradictory. all 67 anchors are voiced.

### how the rewriting runs

on sky's claude code plan via **sonnet subagents**, not metered api calls. `shard` splits
the pool into 45-row chunks plus one shared brief (voice spec + 8 few-shot anchors); each
subagent rewrites its chunk straight to disk; `merge` collects them back.

resumable at the shard level — `shard` skips anything already present in an `out_*.jsonl`,
so a dead agent costs you one chunk, not the run.

(`gen_bulk.py rewrite` still holds a metered Anthropic API path with prompt caching and
batch support, if it's ever wanted. it was estimated at ~$16.50 for 1800 rows on opus 5
via the batch api. the subagent path makes that moot.)

### filters

`gen_bulk.py filter` drops a rewrite if it: lost >10% of the numbers in the original, dropped
a URL, changed the number of code blocks, changed >15% of the identifiers inside a code block,
exceeded 2x the original length, came back with no puppy markers at all, picked up an
assistant-voice tell, or tripped the sexual-content ban list.

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
data/anchors.md        the 67 candidates — human-editable source of truth
data/anchors.jsonl     generated. don't hand-edit
data/rewrite_brief.md  generated voice spec + 8 few-shot anchors, fed to each subagent
data/shards/           in_NNN.json (work) / out_NNN.jsonl (results)
data/train.jsonl       1,564 rows, what actually gets trained on
gen_anchors.py         parse + validate + lint the anchors
gen_bulk.py            fetch -> shard -> [subagents] -> merge -> filter -> sample -> build
train.ipynb            colab SFT, checkpoints + resume + gguf export
eval.py                20 held-out prompts, scored on correct + in-voice
out/                   eval generations for base / 3ep / 1ep / 3ep-no-system
fix_gguf_blocks.py     block_count repair (see "running the gguf")
gguf_meta.py           dump / set gguf metadata keys — applied the YaRN config
make_hf_card.py        MODEL_CARD.md -> hub README
gguf/                  not in git. on the hub instead
```

```bash
python gen_anchors.py --check
```

## license

WTFPUP 1.0 — see LICENSE. base model is Apache 2.0; NOTICE ships with the release.
