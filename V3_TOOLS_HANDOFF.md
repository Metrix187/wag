# wag v3 (or whenever) — tool calling + the narcan.delivery lookup

**deferred out of v2 on 2026-09-19.** v2 became a conversational roleplay model instead —
see `V2_HANDOFF.md`. none of the thinking below was wrong, it just wants its own release
rather than riding along with a personality overhaul.

everything here still assumes the v2 base model (**Qwen3.5-4B**) and the v1 pipeline. read
`V2_HANDOFF.md` first for the landmines table, the spend-plan arithmetic and the shard
contract — this doc only covers the tool-specific parts and doesn't repeat them.

the one piece of timing advice: **build this on top of a finished v2, not beside it.** the
central risk below is that voice training degrades tool calling, and that's far easier to
measure against a shipped v2 than against a moving target.

---

## tool calling — a regression risk, not a feature request

sky wants v2 to be genuinely good at this. the important thing to understand first:

**v1's training data would have made tool calling worse.** the chat template already ships
full `tools` / `<tool_call>` / `<tool_response>` plumbing — 21 references to `tool_call` in
`chat_template.jinja` — so the base model can already do this. v1 then trained 1,564 rows of
*pure conversational puppyspeak with zero tool calls*. two epochs of "every input gets a
lowercase puppy answer" is active pressure away from emitting a JSON block.

so the tool slice is not there to teach a new skill. **it is there to stop the voice training
from eating one that already works.** which means:

- it needs its own eval, or the loss happens silently. nothing in v1's 20 prompts would have
  caught it.
- **measure the base 4B's tool calling before training anything.** that's the number v2 has to
  avoid regressing. if you don't capture it first you have no baseline and no way to tell
  whether a bad result is the fine-tune's fault.

## the rule: voice in the framing, never in the payload

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

## the slice needs negatives too

a model trained only on "call the tool" will call tools for "hello" and "what's 2+2". include
rows where the correct behaviour is **not** calling anything, and rows where the tool returns
an error or an empty result and wag has to cope. v1's single most valuable lesson was that the
hard cases are the ones you have to deliberately put in the data — refusals barely existed in
oasst/alpaca and had to be hand-written.

---

## the narcan.delivery tool

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

### two facts that make the no-hallucination requirement tractable

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

### where the hallucination actually happens

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

### the eval writes itself, and it's mechanical

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

### one call, one state — the state that was asked about

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

### plumbing sketch

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

## the data slices this needs

on top of whatever v2 shipped with:

| slice | rows | why |
|---|---:|---|
| tool calling (general) | 600 | preserves a skill the base already has |
| `narcan_lookup` | 250 | tool results pasted from real `data.json` records |
| no-tool negatives | 200 | stops it calling tools at "hello" |

tool rows are slightly pricier per row than plain chat — the schema rides along in every
input — but it's a rounding error against any sane budget.

## open decisions

1. **does this ride on the v2 roleplay weights, or a separate fine-tune from base?** a
   roleplay-tuned model may be *further* from clean JSON output than base is, which would make
   this harder, not easier. worth measuring before assuming it composes.
2. **if the 4B can't hit zero fabricated fields on the narcan slice, ship the pass-through
   fallback or drop the tool?** answer before building, not after.
