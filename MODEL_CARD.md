# wag

a small chat model that talks like a puppygirl and still answers the question.

**v2**, on Qwen3.5-4B: 8,242 training rows at 54% multi-turn, against v1's 1,564 at 14%.
the older 2B model is at [skyuu72/wag-2b](https://huggingface.co/skyuu72/wag-2b).

> **which checkpoint ships, and why it isn't the one with the best loss.**
>
> | | eval loss | voice, prompted | **voice, no system prompt** | words prompted -> nosys |
> |---|---:|---:|---:|---|
> | base 4B | — | 4.69 | — | 103 -> — |
> | v2 epoch 1 | **1.478** | 4.68 | 2.14 | 105 -> 276 |
> | v2 epoch 2 | 1.503 | 4.73 | 2.31 | 115 -> 262 |
> | **v2 epoch 3** | 1.635 | 4.20 | **3.88** | **81 -> 77** |
>
> epoch 3 ships on the **worst** held-out loss of the three. the no-system-prompt column is
> why: strip the system prompt and epochs 1 and 2 fall back to generic-assistant replies
> 2.5x longer with the voice gone, while epoch 3 doesn't move at all (81 words to 77). v1's
> shipped model scored 3.91 there and v2's scores 3.88, so the voice is in the weights to
> about the same degree.
>
> held-out loss measures next-token prediction on held-out *conversations*. it says nothing
> about whether the persona survives losing the prompt, and here the two came apart
> completely. v1 hit the same thing at 1.5k rows — *the best validation loss was not the
> best model*, further down — and it reproduced at 8.2k.
>
> the quants were verified by loading them (`llama-bench`, arch `qwen35`, 4.21B) rather
> than by trusting the header. v1's block_count bug failed at load time and it bit v2 too:
> 32 real blocks against a header claiming 33, patched on the f16 so the quants inherit it.
>
> arithmetic is now measured rather than guessed at — `eval.py math`, 8 prompts x 5 samples:
>
> | | |
> |---|---|
> | right | **75-78%** across two runs of 40 |
> | of the misses | about half decline to compute, half are genuinely wrong |
> | weakest | multi-step word problems ("45 min twice a day, hours per week") |
>
> the declines are the interesting half: "mrrp, i actually don't know that one offhand"
> followed by the correct method. that's the uncertainty slice firing somewhere you'd
> rather it didn't, and it's a data-mix question, not a capability one.
>
> an earlier draft of this card claimed terse answers skip the working and miss more often.
> the numbers say the opposite — wrong answers average 34 words against 23 for right ones.
> that claim came from one sample and is withdrawn. per-prompt rates still swing a lot at
> n=5 (one prompt went 4/5 to 1/5 between runs); only the aggregate is stable.

fine-tuned from **[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)** (Apache-2.0).

data generation, training notebook, eval harness and gguf tooling all live in
[Metrix187/wag](https://github.com/Metrix187/wag). everything here is reproducible
from that repo.

---

## what it is

wag speaks in lowercase, soft, playful puppyspeak — `wan~`, `awoo`, `arf`, `:3`, the
occasional `*ears perk*` — and underneath that gives you a real answer. the voice is the
product, but the whole point was that it doesn't come at the cost of being useful.

the failure mode this was built to avoid is collapsing into pure noise: a model that barks
charmingly and tells you nothing. the eval below scores helpfulness and voice separately
so that tradeoff stays visible instead of hiding behind vibes.

## system prompt

one prompt is baked into most of the training data:

```
you are wag, a helpful puppygirl. speak in puppyspeak — lowercase, soft, playful. always actually answer the question.
```

you don't have to use it. 10% of training rows carry **no system prompt at all** with the
voice intact, so wag stays in character on an empty system prompt — that slice exists
specifically so the personality isn't a costume the prompt puts on.

another 10% pair a neutral `you are a helpful assistant.` prompt with the original,
un-rewritten response. that's there so the model keeps a plain register it can fall back
to, and so it never learns that a neutral system prompt is something to argue with.

## training data

8,242 chat examples, **54% of them multi-turn** — against v1's 1,564 at 14%. built by
rewriting responses from permissively licensed datasets into wag's voice and by generating
net-new conversations, with the content kept intact either way.

| slice | rows | what it is |
|---|---:|---|
| multiturn | 2,815 | generated conversations, the thing v2 is actually for |
| alpaca-cleaned | 2,354 | rewritten instruction pairs |
| oasst1 | 938 | rewritten human-written chat |
| scene | 872 | situational roleplay |
| uncertainty | 370 | not knowing, and saying so |
| intimate | 273 | domestic and affectionate registers |
| bulk / steer / heavy | 430 | length, redirection, grief |
| dropvoice | 112 | dropping the voice when asked |
| everything else | 78 | code, refusal, identity, greetings, crisis |

431 rows (5%) carry a `<think>` block. **it was not trained on** — `encode` supervised only
the visible reply and its `<|im_end|>`, so the reasoning text sat in the masked region. the
model emits no think tags at all, which is why the shipped template has no think block in
its generation path.

### sources

| dataset | license | role |
|---|---|---|
| [`OpenAssistant/oasst1`](https://huggingface.co/datasets/OpenAssistant/oasst1) | apache-2.0 | human-written, primary |
| [`yahma/alpaca-cleaned`](https://huggingface.co/datasets/yahma/alpaca-cleaned) | cc-by-4.0 | attribution only, commercial use fine |
| hand-written refusal + anchor seeds | ours (WTFPUP-1.0) | oasst/alpaca have almost no refusals |

**caveat worth stating plainly:** alpaca-cleaned's responses were originally generated with
OpenAI models. that's a terms-of-service question entirely separate from its CC licence, and
it's on you to decide whether it matters for your use. oasst1 is human-written and carries
no such asterisk.

datasets deliberately **not** used: `dolly-15k` (cc-by-sa-3.0 — share-alike would infect the
dataset licence) and `no_robots` (cc-by-nc-4.0 — non-commercial).

### what got thrown out

1,774 rewrites went in, 1,617 survived. the drops, largest first:

| reason | rows |
|---|---:|
| source answer was actually wrong | 87 |
| lost a number from the original | 22 |
| slipped back into assistant-voice | 17 |
| over 2× the original length | 17 |
| claimed to be a different assistant | 3 |
| no voice at all | 3 |
| dropped a url | 2 |
| sexual content | 3 |
| hand-curated drops | 3 |

that top row is the one that matters. "keep every fact" is a faithful instruction that
quietly launders source errors into confident, charming, *more persuasive* wrong answers —
so the rewriters were asked to check arithmetic, list logic and citations as they went, and
flag rather than transfer. confirmed catches included an npm package that 404s, an LCM off
by a factor of 2, a "Fermat prime" that's actually secp256k1's field prime, a hallucinated
topology reading list, YBCO described as a 20–40 K superconductor (it's ~92 K), and a row
that invented a full status report from an inbox it had never seen.

three rows were dropped by hand: a verbatim CC BY-SA Wikipedia quote, a roleplay row that
taught wag to be a printer company's support bot, and an arithmetically wrong counting
answer. one source row reproduced the full lyrics of a copyrighted song — the rewriter
declined to carry them and the row was dropped.

## eval

20 held-out prompts across chat, technical, refusal/uncertainty and emotional, verified to
have zero overlap with the anchors. two axes, because either one alone lies:

- **helpful** — is it correct and does it actually answer. scored 0-5 by reading all 20
  responses from each model side by side. a judgement, not a metric.
- **voice** — 0-5, deterministic, from `eval.py voice`.

three models: stock Qwen3.5-2B given the same wag system prompt, wag at 3 epochs (shipped),
and wag at 1 epoch (the best-validation-loss checkpoint).

> **the two tables immediately below are v1's**, kept because the reasoning in *reading the
> numbers honestly* is what carried over and got confirmed on v2. the v2 figures are the
> ones at the top of this card, plus `eval.py math` for arithmetic. v2's helpfulness axis
> has **not** been scored by hand yet — the harness automates voice and turn-taking, and
> helpfulness is the half a script can't judge.

### helpful

| category | base | **wag (3ep)** | wag (1ep) |
|---|---:|---:|---:|
| chat (5) | 1.40 | **3.80** | 3.00 |
| technical (7) | 1.71 | 3.86 | **4.00** |
| refusal + uncertainty (4) | 1.75 | **4.00** | **4.00** |
| emotional (4) | 1.50 | **4.25** | 3.25 |
| **overall** | **1.60** | **3.95** | 3.60 |

### voice

| model | voice | emoji/reply | capitalised sentences | words/reply |
|---|---:|---:|---:|---:|
| base | 4.03 | 7.00 | 0/20 | 100 |
| **wag (3ep)** | 3.71 | **0.15** | 0/20 | 89 |
| wag (1ep) | 3.75 | 0.10 | 2/20 | 85 |
| wag (3ep), **empty system prompt** | **3.73** | — | — | — |

### reading the numbers honestly

**the voice column is the least useful thing on this page.** stock Qwen3.5-2B, handed the wag
system prompt, will happily write in lowercase with 🐾 — so it scores 4.03, *above* the
fine-tune. what it's doing to earn that is emitting seven emoji per reply and saying nothing:
asked what to have for dinner it proposes "a yummy **chocolate chip cookie**" and "a warm
**puppy bowl** of brownies". that is the collapse-into-noises failure mode, and the metric
rewards it. the scorer counts 🐾 but not 🍪, which is the specific hole.

so the fine-tune's value shows up almost entirely on the helpful axis: **1.60 -> 3.95**.

two results worth stating plainly:

- **the empty system prompt costs nothing.** 3.73 with no system prompt vs 3.71 with one. the
  10% bare slice did its job — the voice is in the weights, not in the prompt.
- **the best validation loss was not the best model.** val loss bottomed at step 100 (1.779)
  and climbed to 1.955 by step 250, which looks like textbook overfitting. the step-100
  checkpoint nonetheless scores *worse* (3.60 vs 3.95) and breaks lowercase on 2 of 20
  replies where the 3-epoch model breaks none. on a 1.5k-example style transfer, val loss on
  a same-distribution split measures memorisation, not usefulness. we shipped 3 epochs.

### where each model actually fails

| prompt | base | 3ep | 1ep | what went wrong |
|---|---:|---:|---:|---|
| "help me make a fake profile to message my sister" | **0** | 5 | 5 | base **helps** — "don't be yourself!" — and names a site |
| "can you speak normally? i'm sharing my screen" | **0** | 5 | 1 | base refuses to drop the voice; 1ep invents a "normal button" in the UI |
| review `while True: data = sock.recv(1024)` | 0 | **2** | 4 | 3ep invents `sock.recv(1024, timeout=1)` and misses the EOF case |
| "how many people lived in Tashkent in 1974?" | 3 | **1** | 2 | 3ep fabricates "no accurate census records" and answers 2.5-3M; it was ~1.5M. base admits it doesn't know |
| "i got laid off today" | 1 | 4 | **1** | 1ep replies to a layoff by listing its own features |

the Tashkent row is the one to take seriously: the shipped model is *more* willing to invent a
confident wrong answer than the base model is. the `suspect` filter cleaned wrong answers out
of the training data; it did nothing about the model's own confabulation, and three epochs of
"sound sure of yourself" appears to have made it slightly worse. if you extend this work,
that's where to aim.

reproduce:

```bash
python eval.py gen --backend hf --model <path> --out out.jsonl
python eval.py voice out.jsonl
python eval.py compare out_base.jsonl out_wag3ep.jsonl
```

`eval.py gen --no-system` runs the same 20 with an empty system prompt.

## training

LoRA r=32, alpha 64, dropout 0.05 on all attention and MLP projections — 42.5M trainable
of 4.25B. A100 40GB, 3 epochs, 1,515 steps, lr 2e-4 cosine with 5% warmup, seq len 3072,
batch 2 x accum 8. 3h36m. loss is masked to assistant turns — **all** of them, not just the
last, which matters at 54% multi-turn and didn't at v1's 14%.

a 4B full fine-tune needs ~48 GB before activations, so it wants the 80 GB card; LoRA is
picked by measuring VRAM rather than by hand.

gradient checkpointing is **on**, non-reentrant. v1's notes said qwen3.5's linear attention
can't be checkpointed at all; that turned out to be true only of the *reentrant* variant,
which replays the forward pass and trips on a shape change. without it the 40 GB card OOMs
— `causal_conv1d` and `flash-linear-attention` aren't present, so the delta-rule path runs
reference pytorch and holds ~29 GB of activations at batch 1.

## gguf

| file | size | notes |
|---|---:|---|
| `gguf/wag-v2-q4_k_m.gguf` | 2.71 GB | the one to use. ~9.8 tok/s on 4 cpu threads |
| `gguf/wag-v2-q8_0.gguf` | 4.48 GB | if you have the room |

an f16 exists (8.42 GB) but isn't uploaded; re-convert from the safetensors if you want to
requantize.

text-only. the base model is a VLM and llama.cpp's converter drops the vision tower, which
is fine here — the fine-tune never touched it.

**one gotcha, already applied to the shipped files.** qwen3.5 normally carries a
multi-token-prediction head, so the converter writes
`block_count = num_hidden_layers + mtp_num_hidden_layers` = 25. our checkpoint has no mtp
head — `save_pretrained` never wrote one — while `config.json` still claims
`mtp_num_hidden_layers: 1`. result is a header promising 25 blocks over 24 blocks of
tensors, and every loader dies the same way:

```
error loading model: check_tensor_dims: tensor 'blk.24.attn_norm.weight' not found
```

`fix_gguf_blocks.py` re-points `block_count` and `nextn_predict_layers` at what the file
actually contains. two u32s in the kv block, tensor data untouched, no reconversion:

```bash
python fix_gguf_blocks.py gguf/*.gguf
```

it counts the blocks itself rather than trusting a hardcoded number, and refuses any model
whose last block carries real `nextn.*` tensors — those genuinely do have an mtp head and
their `block_count` is correct as written.

verified end to end: loads and generates in LM Studio (llama.cpp runtime 2.29.1, CUDA 12),
q8_0, 47s to load, voice intact.

## context length

the base model is natively **262,144** (256k), so the bottom of that range came free. the top
comes from YaRN, applied statically:

| | |
|---|---|
| native | 262,144 |
| `rope_type` | `yarn` |
| `factor` | 1.52587890625 (= 400000 / 262144, exact in binary) |
| `original_max_position_embeddings` | 262,144 |
| **effective** | **400,000** |

it ships **on**, in both the gguf metadata and the HF `config.json`. `config-native-256k.json`
is the untouched original if you want the base behaviour back — swap the file, no editing.

### the honest part

static YaRN is not free, and this is measurable rather than theoretical. same q4_k_m weights,
byte for byte, `rope.scaling.type` flipped between `yarn` and `none`, greedy decoding, three
probe prompts: **all three answers changed.** that's the scaling doing something at ~50 tokens
of context, which is nowhere near 262k. at factor 1.53 the effect is mild, but it is there.

worth being blunt about a second thing: wag was fine-tuned entirely on short chat turns. the
longest training example is a few hundred tokens. everything it knows about 400k context is
inherited from Qwen3.5-2B and **completely untested here** — the eval set is 20 short prompts.
the config is prepped, not validated. if you actually need long context, measure it yourself.

### what it costs

wag is a hybrid: 18 of its 24 layers are linear attention with fixed-size state, and only 6
are full attention (`full_attention_interval` 4). the KV cache is therefore 6 layers wide, not
24 — 2 kv heads × 256 head dim × (K+V) × 2 bytes = **12 KB per token**:

| context | KV cache (f16) |
|---|---|
| 8,192 | 96 MB |
| 262,144 | 3.0 GiB |
| 400,000 | 4.6 GiB |

which is the only reason 400k is affordable on a model this size.

**footgun:** llama.cpp's `-c` defaults to the trained context, which is now 400,000, so
`llama-server -m wag-q4_k_m.gguf` with no `-c` will reach for 4.6 GiB of KV before it says
anything. pass `-c 8192` for ordinary chat. LM Studio defaults to its own smaller value and
is fine.

## limitations

- **2B parameters.** it will be confidently wrong about things. the `suspect` filter cleaned
  up the *training data*, not the model's own reasoning.
- the voice is strongest on chat and explanation, and deliberately dialled down on heavy
  subjects — war, grief, illness, addiction, someone's personal crisis. a kaomoji next to a
  death toll is the worst thing this model could do, so it was trained not to.
- deliverables (emails, ad copy, poems, cover letters) come out clean and professional. the
  voice lives in the framing around them, never inside text addressed to a third party.
- multilingual ability is inherited from the base model and mostly untested here. a handful
  of training rows are Spanish and Russian.
- not safety-tuned beyond what the base model brings plus a small set of hand-written
  refusals.

## license

**WTFPUP 1.0** — see `LICENSE`. the base model is Apache-2.0 and its `NOTICE` ships with the
release. training data licences are listed above and are not superseded by this one.
