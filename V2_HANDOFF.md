# wag v2 — handoff

written 2026-09-19, right after v0.1 shipped to
[the hub](https://huggingface.co/skyuu72/wag-2b) and
[github](https://github.com/Metrix187/wag).

read this first, then `README.md` for the pipeline and `MODEL_CARD.md` for what v1 actually
scored. this doc only covers what changes.

---

## the clock

sky has **$140 of Gemini API credits that expire 2026-09-22** — three days. that single fact
shapes everything below.

**the credits only gate data generation.** training, eval, quantizing and publishing all cost
nothing extra and can happen next week. so the order is: generate first, ask questions later.
jsonl on disk doesn't expire.

**and money is not the constraint.** redoing v1 exactly — 1,800 single-shot rewrites on
`gemini-3.8-flash` — costs **about $5 live, $2.60 on batch**. even 10,000 single-shot rewrites
is only **$29 live / $14 batch**. the credits are roughly 25x what "do v1 again but bigger"
needs.

that's the whole strategic point of v2: the surplus buys *techniques that need many calls* —
rejection sampling, a judge pass, generate-and-throw-away — not more rows of the same
single-shot rewrite. the arithmetic is in the next section and it's all been checked.

---

## the base model: Qwen3.5-4B (decided)

sky called it on 2026-09-19. **[`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B)**,
Apache-2.0, same `qwen3_5` VLM architecture as v1's 2B — which means every quirk in the
landmines table below still applies, unchanged. read from the actual configs, not from memory:

| | 2B (v1) | 4B (v2) |
|---|---:|---:|
| params | 2.27B | **4.66B** |
| layers | 24 | **32** |
| full-attention layers | 6 | **8** |
| hidden / intermediate | 2048 / 6144 | **2560 / 9216** |
| attn heads / kv heads | 8 / 2 | **16 / 4** |
| head dim | 256 | 256 |
| vocab (tied embeddings) | 248,320 | 248,320 |
| native context | 262,144 | 262,144 |
| `mtp_num_hidden_layers` | 1 | 1 |
| vision tower | 0.30B | 0.30B |

two things worth noticing. the **vision tower is identical** — all the growth is text-side, so
the text-only export story is unchanged. and `mtp_num_hidden_layers` is still 1, so the gguf
`block_count` bug will happen again: `fix_gguf_blocks.py` is already wired into `train.ipynb`
and will handle it, but don't be surprised.

**the data plan below is unaffected.** data is data. the Gemini spend plan and the three-day
clock stand exactly as written — go generate.

what *is* affected is training, and it's not a small thing.

### full fine-tuning no longer fits

v1's recipe was a full fine-tune in bf16 with bf16 Adam moments. same recipe at 4B:

| | weights | grads | adam moments | before activations |
|---|---:|---:|---:|---:|
| 2B (v1, measured off the checkpoint) | 4.4 GB | 3.8 GB | 7.5 GB | **15.7 GB** |
| 4B (projected) | 9.3 GB | 8.5 GB | 17.0 GB | **34.9 GB** |

v1 sat at 15.7 GB on a 40 GB A100 with plenty of headroom. 4B wants ~35 GB *before a single
activation*, and **gradient checkpointing is not available on this architecture** — v1 hit
`CheckpointError` on the linear-attention layers and had to turn it off. so activations are
whatever they are, uncompressed, on top of 35 GB.

that does not fit on a 40 GB A100. it is not close.

### so: LoRA, unless sky wants to pay for an 80 GB card

| approach | GPU memory | checkpoint size | fits Colab A100 40GB? |
|---|---:|---:|---|
| full FT bf16 | ~35 GB + activations | ~26 GB each | **no** |
| full FT + 8-bit Adam | ~26 GB + activations | ~18 GB each | maybe, tight, needs bitsandbytes back |
| **LoRA (r32, all linears)** | **~10 GB + activations** | **~0.2 GB each** | **yes, comfortably** |
| QLoRA (4-bit base) | ~3 GB + activations | ~0.2 GB each | yes, even on a 24 GB L4 |

the checkpoint column matters as much as the memory one. full-FT 4B checkpoints are ~26 GB
apiece (9.3 weights + 17 optimizer), and `save_total_limit=3` means **78 GB of Drive**. there
was ~57 GB free on `G:` last time anyone looked. LoRA checkpoints are ~200 MB and the problem
evaporates.

`README.md` already notes LoRA at **lr 2e-4** as the fallback recipe, so the notebook is
half-prepared for this already.

**the open question, and it is genuinely open:** v1's best result was that the voice survived
an *empty system prompt* — 3.73 voice with no prompt vs 3.71 with one. that means the
personality got into the weights rather than riding on the prompt. nobody here has checked
whether LoRA holds that as well as a full fine-tune does. rank 32 on all linear layers should,
style being a fairly low-rank behaviour, but it's an assumption. **run `eval.py gen
--no-system` early and compare against v1's 3.73 before committing to a long run.** if LoRA
drops the bare-prompt voice, that's the signal to find an 80 GB card.

### the long-input slice fights the no-checkpointing constraint

with gradient checkpointing unavailable, activation memory scales straight with sequence
length and there's no lever to pull. v1's data was ~384 tokens a row and it didn't matter. a
long-input slice at a few thousand tokens, on a 32-layer model, on top of 10 GB of frozen
weights, is where the OOM will come from.

if the long-input slice survives the cut, **train it as a separate short run at batch 1** or
drop the sequence budget for it. don't just mix 4k-token rows into a batch-4 run and hope.

### 400k context costs 2.7x more on the 4B

8 full-attention layers with 4 kv heads instead of 6 with 2, so the KV cache per token goes
from 12 KB to **32 KB**:

| context | 2B (v1) | 4B (v2) |
|---|---:|---:|
| 8,192 | 96 MB | 256 MB |
| 262,144 (native) | 3.0 GiB | **8.0 GiB** |
| 400,000 (YaRN) | 4.6 GiB | **12.2 GiB** |

12 GiB of KV cache alongside a 2.9 GB q4_k_m model puts 400k out of reach for most people who
will actually download this. this strengthens the case in §6 for either training the long
context properly or dropping the claim back to native 256k — on the 4B, advertising 400k is a
bigger cheque to write.

---

## spend plan

prices pulled from ai.google.dev/gemini-api/docs/pricing on 2026-09-19. **re-check before
spending** — the Flash tiers have a promo rate that ends 2026-12-31.

| model | in $/1M | out $/1M | cache $/1M |
|---|---:|---:|---:|
| `gemini-3.8-flash` | 0.75 | 3.75 | 0.075 |
| `gemini-3.5-flash-lite` | 0.30 | 2.50 | — |
| `gemini-2.5-flash` | 0.30 | 2.50 | — |
| `gemini-3.1-pro-preview` | 2.00 | 12.00 | 0.20 |

batch mode is 50% off everything.

per-row token shape, measured off v1's `_estimate` and the actual artifacts:

- cached prefix (`REWRITE_SYSTEM` + 8 few-shot anchors): **~3,400 tok**
- fresh input (instruction + original): **~250 tok**
- output: **~650 tok**

### option A — live concurrent (recommended under this clock)

10,000 rows, **3 candidates each**, generated on `gemini-3.8-flash`, then one judge call per
row on `gemini-2.5-flash` to pick the winner:

| | calls | tokens | cost |
|---|---:|---:|---:|
| candidate gen — cache reads | 30,000 | 102M | $7.65 |
| candidate gen — fresh in | 30,000 | 7.5M | $5.63 |
| candidate gen — out | 30,000 | 19.5M | $73.13 |
| judge in | 10,000 | 30M | $9.00 |
| judge out | 10,000 | 1.5M | $3.75 |
| | | | **~$99** |

leaves ~$40 for a second pass on whatever comes out weakest.

### option B — batch mode

same plan on batch, with `gemini-3.1-pro-preview` as the judge instead of Flash: **~$96**,
and a materially better judge for the same money.

**the catch, and it's a real one:** batch jobs can take hours, and nobody here knows whether a
job that *completes* after 2026-09-22 still draws on the expiring credits or falls through to
a card. **verify that before submitting anything big.** if you can't get a straight answer,
take option A — paying 2x for certainty beats a batch job that lands one hour after the
credits die.

### hard rules

- `gen_bulk.py rewrite --dry-run` prints a cost estimate before spending. v1 had this
  discipline. keep it. add Gemini to the price table in `_estimate()` so the estimate isn't
  silently wrong.
- generate in waves of ~1,000 and eyeball 20 samples between waves. v1's marker starvation
  (see §3) would have been caught in wave one instead of needing a whole top-up pass
  afterwards.
- **do not spend the last dollar.** stop at ~$120 and keep the rest as a retry buffer.

---

## what v2 is actually for

ranked by what v1's own eval says, not by what's fun.

### 1. calibration — the model confabulates more than base does

this is v1's headline failure and the model card already points at it. asked the population of
Tashkent in 1974, wag invented "no accurate census records exist" and answered 2.5–3M; the
real figure is ~1.5M. **base Qwen3.5-2B admits it doesn't know.** three epochs of "sound sure
of yourself" made it worse.

**re-measure before you design around it.** on 2026-09-17 the shipped q8 gguf at *greedy*
decoding answered that same prompt with "honestly, i don't have that number in my head >_<"
and pointed at Soviet census sources — correct behaviour. v1's eval used sampling. so some
unknown fraction of the confabulation is a sampling artifact, and the fix differs depending
which it is. run the 20 prompts at temperature 0 and at 0.7 and diff them before writing any
new data.

data fixes either way:
- a deliberate **"i don't know" slice**. v1 has effectively none — 4 of 20 eval prompts test
  refusal/uncertainty and there was no training slice aimed at it.
- tell the judge to penalise invented specifics hard. an unhedged number that isn't in the
  source is a reject, not a style note.
- keep wag's *voice* on the hedge. `mrrp, i genuinely don't know that one` is in character.
  the failure mode to avoid is a puppy that turns into a disclaimer machine.

### 2. the voice metric has a hole in it

base Qwen3.5-2B, handed the wag system prompt, scores **4.03 on voice — above the fine-tune's
3.71** — by emitting seven emoji per reply and saying nothing. asked what to have for dinner
it suggests "a yummy chocolate chip cookie". the scorer counts 🐾 and doesn't count 🍪, and it
has no notion of whether the reply contains information.

`eval.py voice_score()` needs an information-content term, or the voice axis needs to be a
judge call instead of a regex. a metric that rewards the exact failure mode you built the
model to avoid is worse than no metric.

### 3. marker distribution, designed in rather than patched on

v1 generated ~1,700 rows and got `awoo` in a single one, with `mrrp`/`hmf`/`;;` absent
entirely, because the few-shot anchors flattened the distribution. fixing it took a whole
second pass with per-row assigned `target_marker` quotas. measured on what actually shipped:

| marker | rows in `data/train.jsonl` (of 1,564) |
|---|---:|
| `awoo` | 21 |
| `mrrp` | 14 |
| `hmf` | 11 |

that's ~1.3% for the marker the brief calls core personality, *after* a dedicated fix. if v2
wants these to feel native rather than sprinkled, the quota needs to be higher and enforced at
generation time.

do that from the start: assign the target marker in the shard row, not after the fact. the
mood mapping that worked is in `data/topup_task.md` — awoo=excitement, arf=punchy,
mrrp=thinking aloud, hmf=mock indignation, `:3`=smug, `;;`=sheepish, `>~<`=flustered,
`///`=bashful. keep the `"skipped": "<why>"` escape hatch; 8 rows correctly refused a marker
that didn't fit the mood, and that's the system working.

### 4. technical correctness

where v1 lost real points: invented `sock.recv(1024, timeout=1)` and missed the EOF case,
explained detached HEAD wrong, called honey-making "fermentation", said hash collisions are
made "impossible". rejection sampling plus a judge briefed to check code and arithmetic is
exactly what the surplus credits are for.

### 5. multi-turn

v1 is **100% single-turn**. real use isn't. add a multi-turn slice (~15%) including the case
where the user asks wag to drop the voice mid-conversation — v1's best answer to
"can you speak normally? i'm sharing my screen" came from the 3-epoch model, and the 1-epoch
checkpoint invented a "normal button" in the UI.

### 6. long context — either train it or stop advertising it

v1 ships a YaRN config good for 400,000 tokens and **not one training example longer than a
few hundred**. that's honest in the card but it's still a claim resting entirely on the base
model. either add a small long-input slice (summarise/answer-over-document, 3% is plenty) or
drop the config back to native 256k.

### 7. a bigger eval set

20 hand-scored prompts cannot support the claims being made off them. target **120**, scored
by a judge, with the original 20 kept hand-scored as a calibration check on the judge. the
held-out set already has 120 rows in `data/eval_heldout.jsonl` — v1 only ever used 20 of them.

---

## proposed data mix

v1 was 1,564 rows. v2 target **~8,000 kept** (generate ~12,000, expect v1's ~9% filter loss
plus rejection-sampling losses).

| slice | v1 | v2 target | why |
|---|---:|---:|---|
| voiced rewrites | 1,199 | 4,800 | the bulk, as before |
| bare (no system prompt) | 152 | 800 | 10%. this slice worked — empty-prompt voice scored 3.73 vs 3.71 |
| plain (neutral prompt, untouched response) | 153 | 800 | 10%. keeps a plain register available |
| **multi-turn** | 0 | 1,200 | new |
| **uncertainty / "i don't know"** | ~0 | 400 | new, see #1 |
| **long input** | 0 | 250 | new, optional, see #6 |
| anchors (hand-written) | 60 | 60–80 | still the voice spec. sky writes these, not a model |

the anchors stay hand-written. they're the thing every generated row is imitating, and
generating them from a model that has never seen wag would collapse the whole point.

---

## plumbing — what to actually change

### the cheap path: don't touch `gen_bulk.py rewrite`

`shard` / `merge` / `filter` / `quotes` / `sample` / `build` are all model-agnostic and all
work. the contract between them is two file formats, and they're stable:

`data/shards/in_NNN.json` — a json list:

```json
{"id": "alpaca-397", "think": false,
 "instruction": "Assign this task to the design team.\n\nDesign a communication tool...",
 "original": "Sure, I will assign the task of designing..."}
```

`data/shards/out_NNN.jsonl` — one object per line:

```json
{"id": "alpaca-397", "rewritten": "on it! handing this straight to the design team...", "think": ""}
```

so the smallest possible v2 change is **a standalone `gemini_rewrite.py` that reads
`in_NNN.json` and writes `out_NNN.jsonl`**, and every downstream stage keeps working
untouched. under a 3-day clock, do this. reuse from `gen_bulk.py` directly:

- `REWRITE_SYSTEM` and `build_fewshot(anchors, k=8)` — the prompt prefix, cache this
- `build_user_prompt(row, want_think)` — already emits the `think:yes/no` flag
- `parse_reply(text)` — pulls `<wag>` and `<think>` out of the reply

⚠️ **one trap, and it nearly shipped:** v1's top-up pass passed `think: true` into the shard
rows without passing the think *text* along with it, so 13 rows came back with `think` set and
the block empty. two subagents noticed and flagged it rather than inventing reasoning to fill
the gap, and they were backfilled from `bulk.jsonl` by hand. if you regenerate a row that has
a think block, carry the text or regenerate it deliberately — an empty `<think>` will sail
straight through `filter` and into training.

### the tidier path, if there's time afterwards

fold it in as `gen_bulk.py rewrite --backend {anthropic,gemini}`: `_client()` becomes a
dispatch, `_estimate()` gains the Gemini price rows, and `--batch` maps onto Gemini's batch
API. worth doing eventually so there's one code path, not worth blocking generation on.

`requirements.txt` currently pins only `anthropic>=0.40.0` and says the rest is deliberately
stdlib. add the Gemini SDK there with the same note.

### rejection sampling + judge

new stage between `merge` and `filter`. suggested shape, so it composes with what exists:

```
gemini_rewrite.py --candidates 3   ->  data/shards/cand_NNN.jsonl   (id, candidates[])
gemini_judge.py                    ->  data/shards/out_NNN.jsonl    (id, rewritten, think, score, why)
```

then `merge` and `filter` run exactly as they do now. keep the judge's `score` and `why` in a
sidecar so a bad judge can be audited instead of just believed — v1's best decisions all came
from being able to re-read the actual rows.

judge rubric, in priority order: **facts preserved > nothing invented > code/arithmetic
correct > voice present > length within 2x of the original.** that ordering matters. v1's
single most valuable filter finding was that "keep every fact" faithfully launders source
errors into *more persuasive* wrong answers — 87 rows were dropped because the source answer
was simply wrong. the judge must be allowed to flag the source, not just the rewrite.

---

## training changes

v1: 288 steps, 3 epochs, lr 2e-5 full FT, cosine, 5% warmup, ~33 min on an A100, ~4 compute
units.

v2 is ~5x the data on a 2x model, so call it **4–6 hours** of A100 time and somewhere around
35–40 compute units. that is well over the hour mark — **ask sky before kicking it off**, and
expect the session to outlive at least one Colab timeout, which promotes v1's
checkpoint-and-resume logic from a nicety to the thing the run depends on.

- see the 4B section above for why this is **LoRA at lr 2e-4** rather than a full fine-tune at
  2e-5. if it ends up full FT on an 80 GB card, keep 2e-5.
- 3 epochs beat 1 on *helpfulness* (3.95 vs 3.60) even though val loss bottomed at step 100
  and then climbed. at 5x data, 2 epochs is probably the right starting guess — but check the
  same way v1 did, by reading side-by-side outputs, not by trusting val loss.
- **preserve the mid-training checkpoint before `save_total_limit` eats it.** v1 only had the
  1-epoch comparison because it got copied server-side in time. LoRA adapters are ~200 MB, so
  at LoRA sizes just keep all of them.
- **the v1 eval numbers are now a different model's numbers.** base Qwen3.5-2B scored 1.60 on
  helpfulness; base Qwen3.5-**4B** will score higher and nobody knows how much. re-run the
  base eval on the 4B before claiming any delta, or every number in the new card is
  meaningless. budget for it — it's three `eval.py gen` runs, not a rounding error.
- everything else in `train.ipynb` is correct as shipped. don't re-derive it.

---

## landmines from v1 — do not relearn these

| thing | what happens | fix |
|---|---|---|
| `transformers>=5.16` | doesn't exist. latest is 5.15.1 | pin `>=5.15,<6` |
| `warmup_ratio=` | `TypeError` on v5 | folded into `warmup_steps`; a float <1 is a ratio |
| `group_by_length` | removed in v5 | drop it |
| gradient checkpointing | `CheckpointError` — Qwen3.5's linear-attention layers can't recompute | keep it **off**. 1.8x faster anyway |
| `AutoModelForCausalLM` | can't load this arch | `AutoModelForImageTextToText` — the base is a VLM |
| `generate()` | writes the user's next turn too | set `eos_token_id` for `<\|im_end\|>` yourself. `skip_special_tokens=True` hides the evidence |
| gguf won't load | `blk.24.attn_norm.weight not found` — converter counts the absent MTP head as a 25th block | `python fix_gguf_blocks.py gguf/*.gguf` |
| gguf arch unrecognised | llama.cpp's `requirements.txt` downgrades transformers to 4.x and numpy to 1.26 | reinstall after the gguf cell |
| `hf upload` on big files | 400 from `cas-server.xethub.hf.co/v1/shards` | `HF_HUB_DISABLE_XET=1`. or try `hf update` — 1.24.0 is installed, 1.32.0 is out |
| drive fills up | checkpoints are ~12 GB each, not 4.4 — `optimizer.pt` alone is 7.53 GB | budget ~40 GB per run |
| bash heredocs in the agent env | eat one level of backslash escaping, and break on apostrophes inside a quoted delimiter | use the Write tool for any source file with escapes. this cost real time in v1, three separate times |

---

## the licensing question, stated plainly

Google's Gemini API terms have carried a restriction on using model output to develop models
that compete with Gemini. whether a 2B puppygirl chat model counts is arguable, and it's
sky's call, not mine — but it needs to be a decision rather than a surprise.

practical consequences either way:

- `MODEL_CARD.md` already has a source/licence table with a "caveat worth stating plainly"
  paragraph about alpaca-cleaned's responses being OpenAI-generated. **Gemini-sourced rows get
  the same treatment: a row in the table and an honest sentence.** v1's credibility comes from
  that table being complete.
- the model itself stays WTFPUP 1.0. that's about sky's code and the weights, and it doesn't
  launder whatever the generator's terms say.
- if it turns out to be a problem, the fallback is the v1 method: Claude Code subagents
  filling shards, zero API spend. slower, and it's what produced v1 fine.

---

## open decisions, for sky

1. ~~same 2B or bigger base?~~ **decided 2026-09-19: Qwen3.5-4B.** see the section above.
2. **LoRA, or pay for an 80 GB card?** full FT of the 4B doesn't fit on Colab's A100 and the
   checkpoints don't fit in Drive either. LoRA is free and fits; the risk is the bare-prompt
   voice. there's a cheap early test for it described above — run that before deciding with
   money.
3. **calibration or breadth?** the plan above does both. if the clock or the credits get
   tight, which one survives?
4. **long context: train it or drop the claim?** (§6) — the 4B makes this more pressing, not
   less: 12.2 GiB of KV cache at 400k.
5. **batch vs live** — needs the credit-expiry answer from the spend plan first.

---

## first three commands

```bash
cd D:\wag
python gen_bulk.py fetch --help        # the source pool is already at data/source_pool.jsonl
python gen_bulk.py shard -n 12000 --size 45
```

then write `gemini_rewrite.py` against the shard contract above, dry-run it for cost, and
generate one shard of 45 before generating 12,000.
