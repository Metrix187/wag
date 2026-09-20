# wag — handoff

*v2, plus the work deferred past it. one document.*

> ## status — 2026-09-20
>
> the plan below still stands. these are the bits of it that turned out to be wrong when
> someone actually ran them, kept here rather than edited in place so the reasoning stays
> readable.
>
> **the first three commands don't work as written.** `shard -n 12000` produced 43 rows,
> because `shard` skips anything that already has output and v1 had spent 1,774 of the
> pool's 1,817. the pool is now topped up to 8,813.
>
> **⚠️ the top-up then corrupted 105 ids and it nearly shipped.** ids are the hf scan
> offset, and resuming started the scan at `len(pool)` — but v1 scanned ~2,000 rows to
> keep ~1,000, so its ids run past the row count. 105 already-used ids came back meaning
> different source rows, and `merge` joins on id, so rewrites got paired with the wrong
> instructions. the fact checks caught 21; **16 reached `bulk.jsonl` looking perfectly
> fine.** fixed at the source, pool repaired, `fetch` now refuses to write duplicate ids
> at all. this is why `tests/` exists.
>
> **the shard contract only covers half of v2.** `instruction` + `original` ->
> `rewritten` fits ~4,050 rows. multi-turn, scenes, uncertainty, drop-the-voice,
> follows-the-steer and intimate are **5,100 rows with no source to rewrite** — they need
> generating from nothing, which is `gemini_convo.py`. `merge` and `filter` had to learn
> the conversation shape too; before that, every seed row vanished into a counter reading
> "N unusable lines skipped".
>
> **the cached prefix is ~1,986 tokens, not 3,400**, so option A lands cheaper than
> budgeted. measured: **~$35 for all 5,100 seed rows**, ~$9 for the rewrite half at one
> candidate each. money was never the constraint and is even less of one than this says.
>
> **🔴 blocked: the api returns `402 RESOURCE_EXHAUSTED — prepayment credits depleted`.**
> the key is valid (`models.list` works, `gemini-3.8-flash` is there) but the project
> can't spend. worth checking whether the $140 is google *cloud* credit rather than ai
> studio prepay — they're separate pots and cloud credit needs the vertex backend, not an
> api key.
>
> **still needed from sky:** the 3–5 hand-written boundary anchors for the intimate
> slice. nothing else can define "slight", and the generated rows will imitate whatever
> those anchors show.
>
> done since: slice-aware filter, gemini prices in the estimator, the 5-way prompt
> spread, multi-turn `build`, both generators, the judge, the voice metric's
> information-content hole (§2), multi-turn eval (§5), and `tests/`.

**v2 is a conversational roleplay puppygirl. that's the whole brief.** not an assistant with a
personality bolted on — a character worth talking to for its own sake, that still answers you
properly when you actually ask something.

written 2026-09-19, right after v0.1 shipped to
[the hub](https://huggingface.co/skyuu72/wag-2b) and
[github](https://github.com/Metrix187/wag).

read this first, then `README.md` for the pipeline and `MODEL_CARD.md` for what v1 actually
scored. this doc only covers what changes.

tool calling and the narcan.delivery lookup were **deferred to a later release**. all of that
work is still here, at the bottom under "deferred" — keeping it out of v2 is deliberate, since
the central risk there is that voice training degrades tool calling, and that is much easier to
measure against a finished v2 than against a moving one.

the only other file this needs is **`data/persona_spec.md`** — the long-form character, which
is a pipeline input the generator gets briefed with rather than something to read here. the
prompt section points at it.

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

the brief is roleplay, so §5 is the centre of gravity and everything else is in service of it.
the rest is still ranked by what v1's own eval says rather than by what's fun — v1 was a good
assistant with a voice, and most of its measured weaknesses are things a roleplay model needs
*more* badly, not less. a character who confabulates confidently is worse company than an
assistant who does, because you're supposed to trust her.

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

### 5. conversational roleplay — the actual brief

v1 is **100% single-turn**, and single-turn is the one thing roleplay never is. this is the
biggest single gap between what shipped and what v2 is supposed to be.

what roleplay needs that v1 never trained:

- **turn-taking that ends.** the model stops when it's the other person's move. v1 had the
  mechanical version of this bug — `generate()` sailing past `<|im_end|>` and writing the
  user's next line — and eval.py works around it with a stop-token fix. in a roleplay model
  it's a *behavioural* failure too, and no stop token saves you from a reply that narrates
  what the user does next. **never write the other person's lines, actions, or feelings.**
- **energy matching.** two words in should not get three paragraphs back. v1 averaged 89 words
  a reply regardless of input — fine for an assistant, wrong for a conversation.
- **continuity.** remembering what was established four turns ago: names, the scene, what she
  already said she wanted. this is what multi-turn data actually teaches; single-turn data
  cannot teach it at all.
- **wanting things.** the failure mode of most roleplay models isn't being offensive, it's
  being *passive* — pure reaction, no initiative. wag should have opinions, moods, boredom,
  curiosity, something she'd rather be doing. v1's anchors already have flickers of this; it
  needs to be the point.
- **user-authored scenarios.** people bring their own setups. see the prompt section below —
  this has a direct consequence for how the training data is built, and getting it wrong is
  the most likely way v2 disappoints.
- **dropping the voice on request, cleanly.** v1's best answer to "can you speak normally? i'm
  sharing my screen" came from the 3-epoch model; the 1-epoch checkpoint invented a "normal
  button" in the UI. keep that behaviour and test it — no arguing, no "but i'm a puppy~".
- **the heavy-subject register.** already in v1 and it worked: markers get dialled down on
  grief, illness, someone's actual crisis. a kaomoji next to someone's bad news is the worst
  thing this model can do. roleplay makes this *more* likely to come up, not less.

### 5b. asterisk actions, and not drowning in them

roleplay convention is `*ears perk*`, and v1 already tracks this: `*action*` appears in 17.6%
of rows, which was the target and reads well. the failure mode is escalation — every sentence
becoming choreography until there's no dialogue left.

hold roughly the v1 rate. actions are punctuation, not stage direction. one per message,
occasionally two, and they should carry mood the words aren't already carrying.

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

## the prompt

sky asked for one. there are really two, doing different jobs, and conflating them is the
mistake to avoid.

### the one that ships (baked into training rows)

```
you're wag: a puppygirl, not an assistant playing one. all lowercase, soft and playful, a
little bratty when it's earned. puppy noises where they land, not a kennel. you have your own
opinions and moods. when someone asks you something real you answer it properly — the voice is
how you talk, not a way out of being useful.
```

~60 tokens. compare v1's, which was one line and worked:

```
you are wag, a helpful puppygirl. speak in puppyspeak — lowercase, soft, playful. always
actually answer the question.
```

the v2 version adds the two things the roleplay brief needs and v1's didn't say: **she's a
character, not a service** ("not an assistant playing one", "your own opinions and moods"), and
the noises have a ceiling ("not a kennel"). it keeps v1's load-bearing clause verbatim in
spirit, because that clause is the entire product.

### the long one (briefs the generator, and users can paste it)

`data/persona_spec.md`, written alongside this doc. ~350 tokens: the full marker/mood mapping,
the turn-taking rules, the energy-matching rule, the heavy-subject register, the
third-party-text rule.

**why not bake the long one in?** two reasons, and the second is the real one:

1. 350 tokens × ~9,000 rows is sequence budget you don't have. system-prompt tokens are masked
   out of the loss (v1 masked to assistant turns only) so they teach nothing, but they still
   cost sequence length — and with gradient checkpointing unavailable on this architecture,
   sequence length is the constraint you can't buy your way out of.
2. v1 already proved the voice goes into the *weights*, not the prompt: empty system prompt
   scored **3.73** voice against **3.71** with one. so a long prompt is buying something v1
   demonstrated you get for free. use the long spec to brief the generator, exactly as
   `rewrite_brief.md` did, and keep the baked prompt short.

### the part that actually matters: vary the prompt across rows

⚠️ **this is the most likely way v2 disappoints, and it's cheap to avoid.**

roleplay users bring their own scenarios. they will paste *"you are wag, a puppygirl. we're at
a coffee shop and you've just spilled my drink"* — a prompt the model has never seen the shape
of. v1 baked **one** fixed prompt into ~77% of rows. do that again on a roleplay model and it
overfits to that exact string and handles user-authored setups badly.

so spread the system prompt across training rows:

| prompt style | share | what it looks like |
|---|---:|---|
| the short baked prompt, verbatim | 40% | the block above |
| paraphrases of it | 20% | same content, different wording, different order |
| baked prompt **+ a scenario clause** | 20% | `...you're wag. we're walking home and it's raining.` |
| scenario-only, no voice instructions | 10% | `you're a puppygirl at a coffee shop with a friend.` — tests whether the voice holds without being asked for |
| none at all | 10% | v1's bare slice. keep it, it worked |

the scenario clauses are free to generate — ask Gemini for 300 varied two-line setups and
sample them. the point isn't the scenarios themselves, it's that the model learns *"system
prompt = who i am plus where i am"* rather than memorising one string.

and **eval it the same way**: the held-out set needs prompts in shapes the training data never
contained, or you're measuring memorisation again — the same trap as v1's val loss, where the
best-val checkpoint scored worse than the one that shipped.

---

## the intimate slice

**decided 2026-09-19: slight NSFW is in scope.** sky's words: "we can do slight nsfw, worst
comes to worst i can use open source models for that." both halves of that have consequences,
and the second half is less trouble than it sounds.

### "slight" has to become a number

this is the part that will go wrong if it's left as a word. hand a generator "write something
slightly spicy" across 300 rows with no calibration and you get a bimodal mess — half of it
chaste enough to be pointless, half of it well past the line, and no consistent register
anywhere. the model then learns both.

**v1 already has the mechanism to make this concrete.** `gen_bulk.py` has
`CONTENT_BANS` (a ~30-term regex) and `SEXUAL_DENSITY_MIN = 2`, and the filter currently
rejects on a *single* hit:

```python
elif len(hits) >= SEXUAL_DENSITY_MIN:
    reason = "sexual content"
elif hits:
    reason = "sexual content (single term)"
```

so **"slight" = a density cap on tagged rows.** something like: rows tagged `intimate` allow up
to 2 hits from `CONTENT_BANS` and still reject at 3+; every untagged row keeps the current
zero-tolerance rule. that's a one-line change at the call site plus a tag on the row, and it
turns a vibe into a number you can audit.

⚠️ **as written, the filter will silently eat this entire slice.** generate 300 rows, run
`filter`, get 0 through, with the reason column reading "sexual content (single term)" 300
times. make the filter slice-aware *before* generating, not after.

### sky writes the boundary anchors

v1's whole architecture was: **anchors are the spec, generated rows imitate them.** 67
hand-written anchors defined the voice and every one of ~1,700 generated rows was few-shot
against 8 of them. that's why the voice is consistent.

the same thing applies here and there's no shortcut. **a model cannot infer where sky's line is
from the word "slight".** three to five hand-written anchors showing the exact register — what
it does, where it stops — will do more than any amount of prompt wording. they go in
`data/anchors.md` like the rest, and the marker/density rules follow from them.

### the generator splits, and that's cheap

Gemini will decline some of this, so those rows come from a local model. **this costs almost
nothing to build**, because v1's shard contract is generator-agnostic — `in_NNN.json` in,
`out_NNN.jsonl` out, and `merge` / `filter` / `build` don't care who filled them.

sky already has the whole stack installed:

- **LM Studio with an OpenAI-compatible server** on `http://localhost:1234/v1/chat/completions`
  — this session used it for the gguf verification, it works
- a model library at `D:\models-studio` with several obvious candidates:
  | model | why |
  |---|---|
  | `huihui-qwen3.6-35b-a3b-...-abliterated` (22.6 GB) | strongest of the three, and A3B so only 3B active — fastest despite the size |
  | `cydonia-24b-v4.3` (14.3 GB) | TheDrummer's, purpose-built for this register |
  | `mn-12b-mag-mell-r1` (7.1 GB) | RP-native merge, smallest and quickest to iterate with |

so `local_rewrite.py` is `gemini_rewrite.py` with a different `base_url` and no API key. same
prompts, same `parse_reply`, same everything downstream.

**and this slice is not on the three-day clock.** no credits are involved, so it can happen
next week. generate the Gemini-funded slices first; come back to this one whenever. budget a
few hours of local GPU time for ~300 rows — the A3B will be much quicker than the dense 24B.

### rules that do not get a slice exemption

- **nothing involving minors, and no age-ambiguous framing.** v1's filter has no check for this
  because no slice was ever intimate; v2 needs one, and it's a hard reject with no tag-based
  exemption. worth being deliberate given that "puppygirl" plus a pet register can read
  ambiguously if a generator is left to its own devices — pin the character as an adult in the
  spec and in the anchors, and keep the filter rule unconditional.
- **the heavy-subject register still wins.** v1's rule was that markers dial down on grief,
  illness and someone's actual crisis. the adjacent failure here is a model that slides into
  flirting when the user is upset, which is the single most off-putting thing a companion model
  does. the intimate register must lose to the heavy-subject register every time, and that
  needs its own rows, not just a line in the spec.
- **she follows the steer, immediately.** the roleplay twin of v1's "drop the voice on request"
  behaviour: if the user pulls the scene somewhere else, wag goes there without negotiating,
  sulking, or circling back to it. this makes the model pleasant to use and it's the same
  training pattern as the drop-the-voice slice, so it's nearly free to add.

### publishing consequences

- a public hub repo whose weights do this wants the **`not-for-all-audiences`** tag in the
  README frontmatter. that's the hub's own mechanism and it costs one line in
  `make_hf_card.py`.
- the model card's source table gains a row for **whichever local model generated the slice**,
  and its licence needs checking — the library above is a mix, and some RP finetunes carry
  terms that aren't the base model's Apache-2.0. v1's credibility rests on that table being
  complete and honest; don't break the streak over 300 rows.

---

## proposed data mix

v1 was 1,564 rows. v2 target **~8,000 kept** (generate ~12,000, expect v1's ~9% filter loss
plus rejection-sampling losses).

the shape is different from v1's, not just bigger: **multi-turn is now the largest single
slice.** that's what "roleplay model" means in data terms.

| slice | v1 | v2 target | why |
|---|---:|---:|---|
| **multi-turn conversation** | 0 | 3,000 | §5. the brief. 3–8 turns, continuity, turn-taking that ends |
| voiced single-turn | 1,199 | 2,400 | still the backbone of the voice |
| **scene / scenario-led** | 0 | 900 | user-authored setups, see the prompt section |
| bare (no system prompt) | 152 | 800 | 10%. empty-prompt voice scored 3.73 vs 3.71 — keep it |
| plain (neutral prompt, untouched response) | 153 | 600 | keeps a plain register available |
| **uncertainty / "i don't know"** | ~0 | 400 | §1, and it matters more for a character than an assistant |
| **drop-the-voice-on-request** | ~2 | 150 | §5. v1 got this right by accident; make it deliberate |
| **heavy-subject register** | some | 200 | grief/illness/crisis with the markers dialled down |
| **intimate (slight)** | 0 (filtered out) | 300 | see the intimate-slice section. local generator, density-capped |
| **follows the steer** | ~0 | 150 | user redirects the scene, wag goes there without negotiating |
| long input | 0 | 250 | optional, §6 — and it fights the memory constraint, see the 4B section |
| anchors (hand-written) | 60 | 80–100 | still the voice spec. sky writes these, not a model |

multi-turn rows are longer, so they cost more per row *and* eat more sequence budget at
training time — the one constraint the 4B genuinely tightened. budget generation at ~3x a
single-turn row and cap conversation length rather than letting the generator ramble.

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

## deferred: tool calling and the narcan.delivery lookup

**not v2.** this was scoped into v2 on 2026-09-19 and pulled back out the same day — v2 is a
roleplay model and this wants its own release. none of the thinking was wrong, it just doesn't
belong in a personality overhaul.

**build it on a finished v2, not beside it.** the central risk below is that voice training
degrades tool calling, and that is far easier to measure against shipped weights than against
a moving target. everything here assumes the same 4B base and the same pipeline, so the
landmines table, the spend-plan arithmetic and the shard contract above all still apply.

### tool calling — a regression risk, not a feature request

sky wants wag to be genuinely good at this. the important thing to understand first:

**v1's training data would have made tool calling worse.** the chat template already ships
full `tools` / `<tool_call>` / `<tool_response>` plumbing — 21 references to `tool_call` in
`chat_template.jinja` — so the base model can already do this. v1 then trained 1,564 rows of
*pure conversational puppyspeak with zero tool calls*. two epochs of "every input gets a
lowercase puppy answer" is active pressure away from emitting a JSON block.

so the tool slice is not there to teach a new skill. **it is there to stop the voice training
from eating one that already works.** which means:

- it needs its own eval, or the loss happens silently. nothing in v1's 20 prompts would have
  caught it.
- **measure the base 4B's tool calling before training anything.** that's the number this has
  to avoid regressing. if you don't capture it first you have no baseline and no way to tell
  whether a bad result is the fine-tune's fault.

### the rule: voice in the framing, never in the payload

v1 already established and validated exactly this rule for deliverables — emails, cover
letters, ad copy come out clean and professional, and "the voice lives in the framing around
them, never inside text addressed to a third party." it held up.

tool calls are the same rule with a harsher failure mode. `wan~` inside a JSON string value is
a bad query; a lowercased or kaomoji-mangled key is a parse error. so:

```
user:   whats naloxone access like in ohio?
wag:    lemme check the live data, one sec~ *ears perk*
        <tool_call>{"name": "narcan_lookup", "arguments": {"state": "OH"}}</tool_call>
tool:   <tool_response>{ ...the real record... }</tool_response>
wag:    okay! ohio's got a statewide standing order, so you can walk into a pharmacy
        without a prescription — arf~  [then the actual fields, quoted]
```

puppy before and after, machine-readable in the middle. **every generated tool-call row must
be mechanically validated:** the JSON parses, the tool name exists, the arguments match the
schema. that's a filter, not a judge, and it's cheap — do not let an unparseable `<tool_call>`
into training.

### the slice needs negatives too

a model trained only on "call the tool" will call tools for "hello" and "what's 2+2". include
rows where the correct behaviour is **not** calling anything, and rows where the tool returns
an error or an empty result and wag has to cope. v1's single most valuable lesson was that the
hard cases are the ones you have to deliberately put in the data — refusals barely existed in
oasst/alpaca and had to be hand-written.

---

### the narcan.delivery tool

sky wants a custom tool that answers from the live [narcan.delivery](https://narcan.delivery)
data rather than from the model's memory. the dataset is at `D:\narcan.delivery\data.json`
and it's in good shape for this:

- **50 states, one uniform schema**, all 7 top-level fields present on all 50
- `state`, `abbreviation`, `last_updated`, `legal_framework`, `access_channels`,
  `practical_guidance`, `sources` (5 urls per state)
- nested where it matters: `access_channels.pharmacies.{mechanism, typical_cost,
  medicaid_coverage_notes}`, `access_channels.{community_programs, mail_based_programs}[]`,
  `legal_framework.good_samaritan_overdose_immunity.{exists, scope}`,
  `practical_guidance.{how_to_get_naloxone_quickly, barriers_and_workarounds}`
- **licensed CC0 / public domain.** no complication putting it in training data, and it gets a
  clean row in the model card's source table
- **one state entry is ~600 tokens. all 50 states is ~19,700 tokens.**

#### two facts that make the no-hallucination requirement tractable

**1. a single state record is tiny.** 600 tokens. the tool can return the *whole* record and
wag's only job is routing the query and framing the answer. there's almost no summarising
pressure, which is where drift comes from. design the tool to return the full record, not a
"relevant excerpt" — resisting the urge to pre-summarise is the single best thing you can do
for accuracy here.

**2. you never have to let a model invent a tool result.** this is the important one. to build
the training rows, generate the *user question* and the *framing prose* with Gemini, but paste
the **real record from `data.json`** as the `<tool_response>`. the training data is then
hallucination-free by construction — wag learns to echo fields that genuinely exist, because
every example it ever saw had real ones. this costs nothing extra and it's much stronger than
trying to filter fabrications out afterwards.

#### where the hallucination actually happens

not at the call step — the call is a short JSON blob and it either parses or it doesn't. it
happens at the **tool-result-to-answer** step, where the model paraphrases and a cost becomes
"about $50" or a program name drifts or a phone number gains a digit. so the training target
is specific:

- **quote verbatim** for anything a person would act on: costs, program names, URLs, phone
  numbers, eligibility rules. wag can be playful *around* them and must not restate them in
  her own words.
- **never add a field the tool didn't return.** if `mail_based_programs` is empty for a state,
  the answer is "no mail program listed for that one", not a plausible-sounding invention.
  this is the same failure as v1's Tashkent row — wag inventing "no accurate census
  records exist" to cover a gap — so it's the same fix, aimed at a place where being wrong
  actually costs somebody something.
- **cite `last_updated` and the source url.** it's in the record, it's free, and it turns
  "trust me" into "check me".

#### the eval writes itself, and it's mechanical

this is the best-specified eval in the whole project, because the ground truth is a json file:

> for each of the 50 states, ask a question, capture the tool result, and **diff every number,
> URL, dollar figure and proper noun in wag's reply against the record.** anything in the reply
> that isn't in the tool output is a fabrication. count them.

no judge, no rubric, no 0-5 scale — a count that should be zero. **that's the acceptance bar
for this slice: zero fabricated fields across all 50 states.** not "scores well".

and if a 4B can't hit zero, the fallback is to stop asking it to: have the tool return a
pre-formatted block and train wag to pass it through with puppy framing top and bottom. less
impressive, still useful, and it can't be wrong. worth deciding that up front rather than
discovering it at eval time.

#### one call, one state — the state that was asked about

**sky's rule, decided 2026-09-19, and it is not negotiable design space.** the tool returns the
record for the state in the question. it does not return neighbours, it does not return the
region, it does not return all 50.

this isn't only tidiness — it's the main defence against the exact failure sky is trying to
avoid. reasons, in order of how much they matter:

1. **a 4B handed 50 records will answer about Ohio using Alabama's numbers.** that's the most
   likely hallucination in the whole feature, and it's the kind that looks completely
   plausible — right shape, right units, wrong state. one record in context means there is no
   other state's cost or program name available to grab.
2. **it keeps the mechanical eval strict.** the reply gets diffed against exactly one record,
   so a figure from a different state is trivially caught as a fabrication. with all 50 in
   context, a wrong-state number is technically "in the context" and much harder to score
   against.
3. **attribution stays unambiguous.** one record, one `last_updated`, one `sources` list. wag
   can cite precisely instead of gesturing at a dataset.
4. 600 tokens instead of 19,700, per turn.

consequences to build in:

- **no `state="all"`, no `region=`, no free-text search parameter.** every one of those is a
  door back to a 50-record context, and the search param additionally invites the model to
  pass a paraphrase instead of a state and turns fuzzy matching into a health-lookup bug.
- **comparisons are two calls, not one big one.** "how does ohio compare to michigan?" →
  `narcan_lookup("OH")`, then `narcan_lookup("MI")`. each record stays individually attributed
  and individually diffable.
- **no state in the question → ask, don't guess.** "where can i get naloxone?" gets "which
  state are you in? :3", not a call with a defaulted argument and not an answer from memory.
  put this case in the training data; it's the tool-calling twin of v2's uncertainty slice.
- **an unrecognised state gets a miss, not a near match.** no silently resolving "washington
  dc" to Washington. the dataset is 50 states; anything else is out of scope and should say so.

the full dataset being small (19,700 tokens) is still useful for one thing and one thing only:
it's a convenient **offline test fixture** — every one of the 50 records is right there to
assert against without a network call. it is not the shipping architecture.

#### plumbing sketch

keep the tool dead simple and offline-capable — a lookup over a local copy of `data.json`,
refreshed from the site, not a live HTTP call per query. `narcan.delivery` already has
`validate-data.mjs`, and the `narcan-data-refresh` skill owns keeping the data current, so
wag's tool should be a *consumer* of that pipeline and must never write to it.

```python
narcan_lookup(state: str)            # one state. name or 2-letter abbreviation.
                                     # returns that record in full, or a miss. nothing else.
narcan_lookup(state, section=...)    # optional narrowing: one of the 7 top-level fields
```

that is the whole API, and the whole API is the point — see the scoping rule above. one
required argument with a closed set of 50 valid values is about as small a hallucination
surface as a tool can have.

---

### the data slices this needs

on top of whatever v2 shipped with:

| slice | rows | why |
|---|---:|---|
| tool calling (general) | 600 | preserves a skill the base already has |
| `narcan_lookup` | 250 | tool results pasted from real `data.json` records |
| no-tool negatives | 200 | stops it calling tools at "hello" |

tool rows are slightly pricier per row than plain chat — the schema rides along in every
input — but it's a rounding error against any sane budget.

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
6. ~~how far does "roleplay" go?~~ **decided 2026-09-19: slight NSFW, local models for the
   rows Gemini won't write.** see the intimate-slice section. the one thing still needed from
   sky is **3–5 hand-written boundary anchors** — nothing else can define "slight", and the
   generated rows will imitate whatever those anchors show.
7. **does the persona spec belong in the repo as the canonical character?** it's at
   `data/persona_spec.md` now. if wag's personality is something you want to keep iterating on
   by hand, that file is the place, and it should probably outrank anything a model generates.

for the deferred tool work, whenever it happens:

8. **does it ride on the v2 roleplay weights, or a separate fine-tune from base?** a
   roleplay-tuned model may be *further* from clean JSON output than base is, which would make
   it harder rather than easier. measure before assuming it composes.
9. **if the 4B can't hit zero fabricated fields on the narcan slice, ship the pass-through
   fallback or drop the tool?** answer before building, not after.

---

## first three commands

```bash
cd D:\wag
python gen_bulk.py fetch --help        # the source pool is already at data/source_pool.jsonl
python gen_bulk.py shard -n 12000 --size 45
```

then write `gemini_rewrite.py` against the shard contract above, dry-run it for cost, and
generate one shard of 45 before generating 12,000.
