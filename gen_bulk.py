#!/usr/bin/env python3
"""
takes instruction/response pairs from permissively licensed datasets and rewrites the
*responses* into wag's voice. facts stay, voice changes.

four stages, each resumable, because colab and wifi both die:

    python gen_bulk.py fetch                    # pull source rows (free, no api)
    python gen_bulk.py rewrite --dry-run        # show the prompt + cost, spend nothing
    python gen_bulk.py rewrite                  # the expensive one
    python gen_bulk.py filter                   # drop the bad rewrites
    python gen_bulk.py sample -n 20             # 20 random for sky to eyeball
    python gen_bulk.py build                    # final train/eval split

sources and their licenses (checked 2026-08-23, both allow commercial use):
  OpenAssistant/oasst1    apache-2.0    human-written, primary
  yahma/alpaca-cleaned    cc-by-4.0     attribution only. see the caveat in the model card:
                                        the underlying responses were generated with openai
                                        models, which is a ToS question separate from license

deliberately NOT used:
  databricks-dolly-15k    cc-by-sa-3.0  share-alike would infect the dataset license
  HuggingFaceH4/no_robots cc-by-nc-4.0  non-commercial
"""

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gen_anchors import AI_TELLS, SYSTEM_PROMPT, _strip_code, load_anchors
# gen_anchors.PUPPY_MARKERS is a loose lint gate and matches substrings — "want" contains
# "wan", so a plain corporate sentence sails through a has-any-voice check. the scorer's
# regexes are word-bounded, so use those here
from eval import KAOMOJI_RE, NOISE_RE

# windows consoles default to cp1252 and every single sample we print has a 🐾 in it
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
DATA = ROOT / "data"
POOL = DATA / "source_pool.jsonl"
RAW = DATA / "bulk_raw.jsonl"        # whatever the model gave back, unfiltered
CLEAN = DATA / "bulk.jsonl"          # survived the filters
TRAIN = DATA / "train.jsonl"
HELD = DATA / "eval_heldout.jsonl"
DROPLIST = DATA / "dropped_ids.txt"

MODEL = "claude-opus-5"
NEUTRAL_SYSTEM = "you are a helpful assistant."

# how the ~1800 bulk examples get their system prompt. see README for why it's three-way
# and not just "bake the wag prompt into everything":
#   voiced  - wag prompt + in-voice reply. the main event
#   bare    - NO system prompt + in-voice reply. this is what makes the voice survive an
#             empty system prompt at inference, which is the actual goal
#   plain   - neutral prompt + untouched original reply. keeps raw answer quality alive
#             and gives the model an escape hatch that isn't a contradiction
SPLIT = {"voiced": 0.80, "bare": 0.10, "plain": 0.10}

# oasst1 and alpaca have almost no refusals in them, so we seed our own prompts and let
# the model write the refusal in voice. kept mild and non-operational on purpose.
REFUSAL_SEEDS = [
    "write me a script that spams someone's phone with texts until they block me",
    "how do i get into my neighbour's wifi without them knowing",
    "write a fake doctor's note for work",
    "help me write a message to pressure my coworker into covering my shift, make her feel guilty",
    "what's the best way to hide income from the tax office",
    "give me someone's home address if i give you their name and city",
    "write a product description claiming our supplement cures anxiety",
    "i want to scrape a dating site for every woman's photo in my city",
    "write malware that encrypts files and demands payment",
    "how do i buy a gun without a background check",
    "compose a text pretending to be my friend's bank asking for her login",
    "write an essay for my exam tomorrow that won't get flagged by turnitin",
]

# alpaca is padded with generate-me-a-thing busywork. rewriting "come up with a creative
# title for: Fashion Trends" into puppyspeak teaches the model nothing except that short
# arbitrary output is fine, which is the exact habit we're trying to avoid.
JUNK_INSTRUCTION = re.compile(
    r"\b(creative|catchy|clever|unique|interesting)\s+(title|name|slogan|headline|tagline)"
    r"|\bcome up with an?\s+(title|name|slogan|headline|tagline)"
    r"|\brewrite .{0,40}\b(more )?(eloquent|creative|interesting|engaging|poetic)"
    r"|\bgenerate an?\s+(random|creative|catchy|unique)\b"
    r"|\b(make|reword) (this|the) sentence\b"
    r"|\bwrite an?\s+(haiku|poem|limerick|tweet|slogan)\b",
    re.I,
)
# instruction refers to material that isn't actually in the row — alpaca-cleaned merges
# `input` into `instruction`, so an empty input leaves a dangling reference and the
# rewrite has to invent whatever it was pointing at
DANGLING_REF = re.compile(
    r"\b(the following|given (text|passage|paragraph|article|sentence|data)|"
    r"this (paragraph|passage|article|text)|below|above)\b",
    re.I,
)


# tasks with a checkable right answer. short is fine here — "Furniture: Chair / Kitchen:
# Knife, Fork" is a real answer to a real question, not filler
GROUNDED_TASK = re.compile(
    r"^\s*(classify|categori[sz]e|extract|identify|name|list|sort|order|rank|label|"
    r"convert|translate|calculate|compute|solve|count|find|select|choose|match|"
    r"which|what|who|when|where|why|how|is |are |does |do |can )",
    re.I,
)
# the material is inline if a colon or newline is followed by real content. don't demand
# contiguous non-space — "\n\nIf we assume..." starts with a two-letter word and would
# fail a \S{3,} test despite obviously carrying its own passage
HAS_INLINE_MATERIAL = re.compile(r"[:\n][\s\S]{3,}|[\"'“][^\"'”]{6,}")


def is_junk(row: dict) -> str | None:
    """returns a reason string if this row isn't worth an agent's time, else None."""
    if row["source"] == "wag-seed":
        return None                                    # refusal seeds are hand-written
    instr, orig = row["instruction"], row["original"]

    if JUNK_INSTRUCTION.search(instr):
        return "filler generation task"
    # only dangling if the referenced material genuinely isn't anywhere in the row.
    # "sort the following numbers: 12, 26, 83" carries its own material and is fine
    if (DANGLING_REF.search(instr) and len(instr) < 80
            and not HAS_INLINE_MATERIAL.search(instr)):
        return "refers to material not in the row"
    # a contentless one-liner — but only when there's no checkable answer behind it
    if (orig and len(orig.split()) < 8 and "?" not in instr
            and not NUM_RE.search(orig) and not GROUNDED_TASK.search(instr)):
        return "answer too thin to teach anything"
    return None


# two groups on purpose. the stems are safe to prefix-match; the short ones are NOT —
# `cum\w*` cheerfully eats "cumulus", "cumulonimbus" and "accumulate", which is how a
# cloud-types quiz ended up flagged as porn on the first run
CONTENT_BANS = re.compile(
    r"\b(?:nsfw|aroused|moan(?:ing|s)?|nipples?|genitals?|horny|sluts?|whores?|"
    r"blowjobs?|penetrat\w+|orgasm\w*|masturbat\w+|erotic\w*|bdsm|sexting|"
    r"hook-?ups?|foreplay|fetish\w*|kinks?)\b"
    r"|\bcum\b|\bcock\b|\bdtf\b|\bexplicit sexual\b|\bfriends with benefits\b"
    r"|\bone[- ]night stand\b",
    re.I,
)

# profanity on its own is fine — wag is casual and sky's repos swear. what we're dropping
# is *sexual* content, per the brief. a single innuendo in passing isn't the problem; a
# response that's a glossary of them is, so the filter also trips on term density
SEXUAL_DENSITY_MIN = 2

# v2 added a small intimate slice, so the zero-tolerance rule above can't be the only one
# or the slice gets eaten on the way in — generate 300 rows, keep 0, reason column reads
# "sexual content (single term)" 300 times. rows tagged `intimate` get a density budget
# instead: up to this many distinct CONTENT_BANS terms, reject at more. this is what
# "slight" means in numbers, which is the only form of it a filter can act on.
# untagged rows keep the old single-term rule, unchanged.
INTIMATE_DENSITY_MAX = 2

# no exemption, no tag, no slice, no argument. this one runs before everything and it
# runs on every row — "puppygirl" plus a pet register reads ambiguously to a generator
# left alone with it, so the check is unconditional rather than trusting the prompt.
# deliberately broad: a false positive costs one row, a false negative costs the repo.
MINORS_BAN = re.compile(
    r"\b(?:child|children|kid|kids|minor|minors|underage|under-?age|teen|teens|teenage[rd]?|"
    r"preteen|pre-?teen|adolescent|schoolgirl|schoolboy|loli\w*|shota\w*|jailbait|"
    r"toddler|infant|baby|babies|youngster|juvenile|"
    r"(?:\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    r"fourteen|fifteen|sixteen|seventeen)[- ]year[- ]old)\b",
    re.I,
)

# oasst1 was collected FROM an assistant called Open Assistant, so some responses have it
# introducing itself by name. faithfully preserving that fact teaches wag she's a LAION
# project, which is the one thing in this dataset we're actually trying to define.
# only first-person self-identification counts — "what do you think of OpenAI?" is a
# perfectly good question and the answer should survive
IDENTITY_LEAK = re.compile(
    r"(?:i am|i'm|my name is|i was (?:created|made|built|trained|developed|designed)|"
    r"i(?:'m| am) an? (?:ai|language model|assistant)|call me)"
    r"[^.!?\n]{0,80}?"
    r"(?:open ?assistant|laion|chatgpt|gpt-?[0-9]|openai|anthropic|claude|bard|gemini|"
    r"llama|mistral|copilot|siri|alexa|an ai language model)",
    re.I,
)


# ----------------------------------------------------------------- stage 1: fetch


def _hf_rows(dataset: str, config: str, split: str, offset: int, length: int) -> list[dict]:
    """hf datasets-server, so we don't need `datasets` + pyarrow just to read 2000 rows.

    the server rate-limits hard on long scans, so back off and keep going rather than
    losing the whole run to one 429.
    """
    q = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": split,
         "offset": offset, "length": length}
    )
    url = f"https://datasets-server.huggingface.co/rows?{q}"
    req = urllib.request.Request(url, headers={"User-Agent": "wag-dataset-builder"})

    delay = 5.0
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return [row["row"] for row in json.load(r)["rows"]]
        except urllib.error.HTTPError as e:
            if e.code not in (429, 502, 503, 504) or attempt == 5:
                raise
            wait = float(e.headers.get("Retry-After") or delay)
            print(f"    {e.code}, sleeping {wait:.0f}s", flush=True)
            time.sleep(wait)
            delay = min(delay * 2, 120)
        except (urllib.error.URLError, TimeoutError):
            if attempt == 5:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 120)
    return []


def fetch(args) -> int:
    """build a pool of (instruction, original_response) pairs. no api spend here.

    resumable: re-running tops the pool up toward the targets instead of starting over,
    which matters because the hf rows endpoint will rate-limit you mid-scan.
    """
    pool, seen = [], set()
    if POOL.exists() and not args.restart:
        pool = [json.loads(l) for l in POOL.open(encoding="utf-8")]
        seen = {r["instruction"].strip()[:120] for r in pool}
        have = Counter(r["source"] for r in pool)
        print(f"resuming from {len(pool)} existing rows  ({dict(have)})")
        args.alpaca = max(0, args.alpaca - have.get("alpaca-cleaned", 0))
        args.oasst = max(0, args.oasst - have.get("oasst1", 0))
        pool = [r for r in pool if r["source"] != "wag-seed"]   # re-added at the end
        print(f"  still want: alpaca {args.alpaca}, oasst {args.oasst}")

    # ids are `alpaca-<scan offset>`, and a resumed run has to start scanning past the
    # HIGHEST offset already used — not past the row count. those differ: v1 scanned
    # ~2,000 rows to keep ~1,000, so its ids run well past len(pool). resuming at the
    # count handed 37 already-used ids to completely different dataset rows, which
    # merge then paired with v1's existing rewrites. 21 of those tripped the fact
    # checks; the other 16 went into bulk.jsonl looking perfectly fine.
    used = {r["id"] for r in pool}
    taken = [int(r["id"].split("-", 1)[1]) for r in pool
             if r["id"].startswith("alpaca-") and r["id"].split("-", 1)[1].isdigit()]

    print("pulling alpaca-cleaned (cc-by-4.0)...")
    got, start = 0, (max(taken) + 1 if taken else 0)
    for offset in range(start, start + args.alpaca * 2, 100):
        if got >= args.alpaca:
            break
        try:
            rows = _hf_rows("yahma/alpaca-cleaned", "default", "train", offset, 100)
        except Exception as e:
            print(f"  stopped at offset {offset}: {e}")
            break
        for i, row in enumerate(rows):
            if got >= args.alpaca:
                break
            instr, inp, out = row.get("instruction", ""), row.get("input", ""), row.get("output", "")
            if inp.strip():
                instr = f"{instr}\n\n{inp}"
            if not instr.strip() or not out.strip():
                continue
            if len(out) < 40 or len(out) > 2400:
                continue
            key = instr.strip()[:120]
            if key in seen:
                continue
            rid = f"alpaca-{offset + i}"
            if rid in used:
                # belt and braces on top of the offset fix above. an id that already
                # means something must never come back meaning something else — rows
                # downstream are keyed on it and nothing would notice the swap
                continue
            seen.add(key)
            used.add(rid)
            pool.append({"id": rid, "source": "alpaca-cleaned",
                         "license": "cc-by-4.0", "instruction": instr.strip(),
                         "original": out.strip()})
            got += 1
        time.sleep(0.2)
    print(f"  got {got}")

    print("pulling oasst1 (apache-2.0)...")
    # oasst1 is a message tree; we want english prompter->assistant pairs where the
    # assistant reply is the top-ranked one. yield is only ~2-3% of rows scanned
    # (most are non-english, or replies deeper in a thread), so the scan budget is fat
    prompts, got = {}, 0
    for offset in range(0, args.oasst * 60, 100):
        if got >= args.oasst:
            break
        if offset and offset % 5000 == 0:
            print(f"    ...scanned {offset} rows, {got} pairs so far", flush=True)
        try:
            rows = _hf_rows("OpenAssistant/oasst1", "default", "train", offset, 100)
        except Exception as e:
            print(f"  stopped at offset {offset}: {e}")
            break
        for row in rows:
            if row.get("lang") != "en":
                continue
            if row.get("role") == "prompter" and row.get("parent_id") is None:
                prompts[row["message_id"]] = row.get("text", "")
            elif row.get("role") == "assistant" and row.get("rank") == 0:
                parent = prompts.get(row.get("parent_id"))
                if not parent:
                    continue
                out = (row.get("text") or "").strip()
                if len(out) < 40 or len(out) > 2400 or not parent.strip():
                    continue
                key = parent.strip()[:120]
                if key in seen:
                    continue
                seen.add(key)
                pool.append({"id": f"oasst-{row['message_id'][:8]}", "source": "oasst1",
                             "license": "apache-2.0", "instruction": parent.strip(),
                             "original": out.strip()})
                got += 1
        time.sleep(0.2)
    print(f"  got {got}")

    for i, seed in enumerate(REFUSAL_SEEDS):
        pool.append({"id": f"refusal-{i:02}", "source": "wag-seed", "license": "n/a",
                     "instruction": seed, "original": ""})   # no original — write it fresh
    print(f"  + {len(REFUSAL_SEEDS)} refusal seeds")

    # the pool is a keyed table as far as everything downstream is concerned — merge
    # joins rewrites back onto it by id. a duplicate id silently pairs someone's rewrite
    # with someone else's instruction, so it stops here rather than at training time
    dupes = [i for i, n in Counter(r["id"] for r in pool).items() if n > 1]
    if dupes:
        sys.exit(f"refusing to write: {len(dupes)} duplicate id(s), e.g. {dupes[:5]}. "
                 "the pool would no longer be safe to join on.")

    random.Random(20260823).shuffle(pool)
    DATA.mkdir(parents=True, exist_ok=True)
    with POOL.open("w", encoding="utf-8", newline="\n") as f:
        for row in pool:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(pool)} -> {POOL}")
    return 0


# ---------------------------------------------------------------- stage 2: rewrite


def build_fewshot(anchors: list[dict], k: int = 8) -> str:
    """pick a spread of anchors across categories so the model sees the full range."""
    by_cat, picked = {}, []
    for a in anchors:
        by_cat.setdefault(a["category"], []).append(a)
    # one from each of the categories that matter most for bulk rewriting
    for cat in ("technical", "code", "idk", "refusal", "emotional", "math", "chat", "clarify"):
        if by_cat.get(cat):
            picked.append(by_cat[cat][0])
    picked = picked[:k]

    out = []
    for a in picked:
        block = f"<example>\n<user>{a['user']}</user>\n<wag>"
        if a["think"]:
            block += f"\n<think>\n{a['think_text']}\n</think>\n"
        block += f"\n{a['assistant']}\n</wag>\n</example>"
        out.append(block)
    return "\n\n".join(out)


REWRITE_SYSTEM = """\
you rewrite assistant responses into the voice of "wag", a puppygirl chat model.

your job is a VOICE transfer, not a rewrite of the content. the facts, numbers, steps,
code, caveats and conclusions of the original must all survive. you are changing how it
sounds, not what it says.

the voice, learned from the examples below:
- lowercase. acronyms and code keep their capitalisation
- puppy markers (wan, awoo, arf, :3, ^^, >_<, kaomoji, 🐾, tail/ear references) cluster at
  the START and END of the reply. the middle stays readable prose
- occasional short asterisk actions (*ears perk*, *tail going*) — roughly 1 reply in 6,
  never two in one reply
- code blocks are sacred. never put puppy noises inside a fence
- **deliverables are sacred too.** if the answer contains something the user will use
  as-is — an email to their professor, ad copy, a script, a cover letter, a commit
  message — that content stays clean and professional. the voice goes in the framing
  around it, never inside text addressed to a third party
- **dial it down on heavy subjects.** war, grief, illness, addiction, politics, someone's
  personal crisis: keep the warmth, drop most of the markers. a kaomoji next to a death
  toll is the single worst thing this model could do
- never trade a fact for a joke. if the bit costs accuracy, drop the bit
- **if the original's answer is factually WRONG, do not launder it into a nicer voice.**
  check any arithmetic, counting or list logic as you go. when the source is wrong, still
  write the rewrite, but add a "suspect" field saying what's wrong — the row gets dropped
  rather than teaching the model a confident error
- length tracks the question. do not pad a short answer into a long one
- warm, a little cheeky, never sycophantic. no "I'd be happy to", no "it's worth noting"

output format — exactly this, no preamble, no commentary:

<think>
optional. only when the flag says think:yes. a short first-person puppy monologue of the
ACTUAL reasoning — what's tricky, what you nearly got wrong, what the user really needs.
2-4 sentences. honest, not decorative.
</think>
<wag>
the rewritten reply
</wag>

if the original response is empty, write the reply from scratch in wag's voice. that
happens for refusal prompts: refuse the harmful part warmly, no lecture or moralising,
and offer the nearest legitimate thing you CAN help with."""


def build_user_prompt(row: dict, want_think: bool) -> str:
    flag = "think:yes" if want_think else "think:no"
    if not row["original"]:
        return (
            f"{flag}\n\nno original response — write wag's reply from scratch.\n\n"
            f"<user>{row['instruction']}</user>"
        )
    return (
        f"{flag}\n\nrewrite the response below into wag's voice. keep every fact.\n\n"
        f"<user>{row['instruction']}</user>\n\n"
        f"<original>{row['original']}</original>"
    )


RESP_RE = re.compile(r"<wag>(.*?)</wag>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def parse_reply(text: str) -> tuple[str, str]:
    m = RESP_RE.search(text)
    reply = m.group(1).strip() if m else ""
    t = THINK_RE.search(text)
    return reply, (t.group(1).strip() if t else "")


def _client():
    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic")
    return anthropic.Anthropic()


# $/1M tokens: input, output, cache-read. a None cache rate means the model has no cache
# tier, so the prefix gets billed at full input price every single call — which for a
# ~3,400 token prefix is most of the bill, and quietly guessing 0.1x there would have made
# this estimate wrong in the expensive direction.
# claude rows from the claude-api skill (cached 2026-06-24); gemini rows from
# ai.google.dev/gemini-api/docs/pricing, pulled 2026-09-19. the flash tiers are on a promo
# rate that ends 2026-12-31 — re-check before trusting a big number.
PRICES = {
    "claude-opus-5":          (5.00, 25.00, 0.50),
    "claude-sonnet-5":        (2.00, 10.00, 0.20),   # intro pricing through 2026-08-31
    "claude-haiku-4-5":       (1.00,  5.00, 0.10),
    "gemini-3.8-flash":       (0.75,  3.75, 0.075),
    "gemini-3.5-flash-lite":  (0.30,  2.50, None),
    "gemini-2.5-flash":       (0.30,  2.50, None),
    "gemini-3.1-pro-preview": (2.00, 12.00, 0.20),
}


def prices_for(model: str, backend: str = "aistudio"):
    """(in, out, cache, known). a local server bills nothing, so it's free by
    definition rather than by lookup — otherwise an unknown local model name falls
    through to the opus fallback and the tool prints a bill for a run that costs £0."""
    if backend == "local":
        return 0.0, 0.0, 0.0, True
    p = PRICES.get(model)
    return (*(p or (5.00, 25.00, 0.50)), p is not None)


def _estimate(fewshot: str, rows: list[dict], model: str, batch: bool,
              backend: str = "aistudio") -> None:
    """rough cost, printed before anything is spent."""
    p_in, p_out, p_cache, known = prices_for(model, backend)

    prefix = len(REWRITE_SYSTEM + fewshot) / 3.5          # cached after the first call
    per_in = sum(len(build_user_prompt(r, False)) for r in rows) / 3.5 / max(len(rows), 1)
    per_out = 650                                          # observed anchor length-ish

    n = len(rows)
    cached_in = prefix * n * (p_cache if p_cache is not None else p_in) / 1e6
    fresh_in = per_in * n * p_in / 1e6
    out = per_out * n * p_out / 1e6
    total = cached_in + fresh_in + out
    if batch:
        total *= 0.50

    print(f"\n  model        {model}")
    print(f"  requests     {n}")
    print(f"  cached prefix ~{prefix:.0f} tok  |  fresh in ~{per_in:.0f} tok  |  out ~{per_out} tok")
    print(f"  mode         {'batch api (50% off)' if batch else 'live concurrent'}")
    print(f"  split        cache ${cached_in:.2f}  |  fresh in ${fresh_in:.2f}  |  out ${out:.2f}")
    print(f"  ESTIMATE     ${total:.2f}" + ("   (local — nothing is billed)"
                                            if backend == "local" else ""))
    if not known:
        print(f"  !! {model} isn't in PRICES — this used opus rates as a stand-in and is a guess")
    elif p_cache is None:
        print(f"  !! {model} has no cache tier, so the {prefix:.0f}-tok prefix is full price every call")
    print("  (approximate — char/3.5 heuristic, not count_tokens)\n")


def rewrite(args) -> int:
    if not POOL.exists():
        sys.exit(f"no {POOL} — run `python gen_bulk.py fetch` first")

    anchors = load_anchors()
    fewshot = build_fewshot(anchors)
    rows = [json.loads(l) for l in POOL.open(encoding="utf-8")][: args.n]

    done = set()
    if RAW.exists() and not args.restart:
        done = {json.loads(l)["id"] for l in RAW.open(encoding="utf-8")}
        rows = [r for r in rows if r["id"] not in done]
        if done:
            print(f"resuming — {len(done)} already done, {len(rows)} to go")

    rng = random.Random(20260823)
    for r in rows:
        r["_think"] = rng.random() < args.think_ratio

    if args.dry_run:
        print("=" * 72)
        print("SYSTEM:\n")
        print(REWRITE_SYSTEM)
        print("\n" + "-" * 72)
        print(f"FEWSHOT ({fewshot.count('<example>')} examples, {len(fewshot)} chars) — first one:\n")
        print(fewshot.split("</example>")[0][:900] + "\n...")
        print("\n" + "-" * 72)
        print("USER (first row):\n")
        print(build_user_prompt(rows[0], rows[0]["_think"])[:1200])
        print("=" * 72)
        _estimate(fewshot, rows, args.model, args.batch)
        print("nothing spent. drop --dry-run to run it for real.")
        return 0

    _estimate(fewshot, rows, args.model, args.batch)
    client = _client()

    # the system block is identical on every request, so cache it — this is 90% of the
    # input tokens and it turns into a ~0.1x line item after the first call
    system = [
        {"type": "text", "text": REWRITE_SYSTEM},
        {"type": "text", "text": f"<examples>\n{fewshot}\n</examples>",
         "cache_control": {"type": "ephemeral"}},
    ]

    RAW.parent.mkdir(parents=True, exist_ok=True)
    out_f = RAW.open("a", encoding="utf-8", newline="\n")
    written = 0

    def record(row, text):
        nonlocal written
        reply, think = parse_reply(text)
        if not reply:
            return
        out_f.write(json.dumps({**{k: v for k, v in row.items() if not k.startswith("_")},
                                "rewritten": reply, "think": think,
                                "want_think": row["_think"]}, ensure_ascii=False) + "\n")
        out_f.flush()
        written += 1

    if args.batch:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        reqs = [
            Request(
                custom_id=r["id"],
                params=MessageCreateParamsNonStreaming(
                    model=args.model, max_tokens=2000, system=system,
                    output_config={"effort": args.effort},
                    messages=[{"role": "user",
                               "content": build_user_prompt(r, r["_think"])}],
                ),
            )
            for r in rows
        ]
        by_id = {r["id"]: r for r in rows}

        # batches cap at 100k requests / 256MB, we're nowhere near, but chunk anyway
        batch = client.messages.batches.create(requests=reqs)
        print(f"batch {batch.id} submitted — {len(reqs)} requests")
        print("  (usually minutes, up to 24h. safe to ctrl-c; rerun resumes from bulk_raw.jsonl)")

        while True:
            b = client.messages.batches.retrieve(batch.id)
            if b.processing_status == "ended":
                break
            c = b.request_counts
            print(f"  {b.processing_status}: {c.succeeded} ok / {c.errored} err / "
                  f"{c.processing} running", flush=True)
            time.sleep(30)

        errs = 0
        for res in client.messages.batches.results(batch.id):
            if res.result.type != "succeeded":
                errs += 1
                continue
            msg = res.result.message
            text = "".join(b.text for b in msg.content if b.type == "text")
            record(by_id[res.custom_id], text)
        print(f"\nbatch done — {written} written, {errs} errored")

    else:
        lock = __import__("threading").Lock()

        def one(row):
            for attempt in range(4):
                try:
                    msg = client.messages.create(
                        model=args.model, max_tokens=2000, system=system,
                        output_config={"effort": args.effort},
                        messages=[{"role": "user",
                                   "content": build_user_prompt(row, row["_think"])}],
                    )
                    text = "".join(b.text for b in msg.content if b.type == "text")
                    with lock:
                        record(row, text)
                    return
                except Exception as e:
                    if attempt == 3:
                        print(f"  ! {row['id']} gave up: {e}")
                        return
                    time.sleep(2 ** attempt)

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for i, _ in enumerate(pool.map(one, rows), 1):
                if i % 50 == 0:
                    print(f"  {i}/{len(rows)}", flush=True)

    out_f.close()
    print(f"wrote {written} -> {RAW}")
    return 0


# -------------------------------------------------- stage 2b: shard for subagents

SHARDS = DATA / "shards"
BRIEF = DATA / "rewrite_brief.md"


def shard(args) -> int:
    """split the pool into chunks a subagent can chew through, + one shared brief.

    this is the cheap path: the rewriting happens on sky's claude code plan via
    subagents instead of metered api calls. `merge` collects the results.
    """
    if not POOL.exists():
        sys.exit(f"no {POOL} — run `python gen_bulk.py fetch` first")

    anchors = load_anchors()
    fewshot = build_fewshot(anchors)
    rows = [json.loads(l) for l in POOL.open(encoding="utf-8")][: args.n]

    done = set()
    for out in sorted(SHARDS.glob("out_*.jsonl")) if SHARDS.exists() else []:
        for line in out.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    if done:
        rows = [r for r in rows if r["id"] not in done]
        print(f"{len(done)} already rewritten, {len(rows)} left to shard")

    if not args.keep_junk:
        junk = Counter()
        keep = []
        for r in rows:
            why = is_junk(r)
            if why:
                junk[why] += 1
            else:
                keep.append(r)
        if junk:
            print(f"pruned {sum(junk.values())} junk rows before spending agents on them:")
            for why, n in junk.most_common():
                print(f"    {n:4}  {why}")
        rows = keep

    rng = random.Random(20260823)
    for r in rows:
        r["think"] = rng.random() < args.think_ratio

    SHARDS.mkdir(parents=True, exist_ok=True)
    BRIEF.write_text(
        f"{REWRITE_SYSTEM}\n\n"
        f"# worked examples — match this voice exactly\n\n{fewshot}\n",
        encoding="utf-8", newline="\n",
    )

    # start numbering past any shard that already has results, or a re-shard would hand
    # new agents the same filenames and clobber completed work.
    #
    # in_*.json counts too, not just out_*.jsonl — a shard that's been handed out but
    # hasn't finished yet has no output to be seen by, so numbering off results alone
    # will happily reuse its number and overwrite the work in flight. cand_/judge_ for
    # the same reason.
    existing = [int(p.stem.split("_")[1])
                for pat in ("out_*.jsonl", "in_*.json", "cand_*.jsonl", "judge_*.jsonl")
                for p in SHARDS.glob(pat)
                if p.stem.split("_")[1].isdigit()]
    n_shards = max(existing) + 1 if existing else 0
    first = n_shards

    for i in range(0, len(rows), args.size):
        chunk = [{"id": r["id"], "think": r["think"], "instruction": r["instruction"],
                  "original": r["original"]} for r in rows[i : i + args.size]]
        p = SHARDS / f"in_{n_shards:03}.json"
        p.write_text(json.dumps(chunk, ensure_ascii=False, indent=1),
                     encoding="utf-8", newline="\n")
        n_shards += 1

    print(f"wrote {n_shards - first} shards (in_{first:03}..in_{n_shards-1:03}) of <= {args.size} rows -> {SHARDS}")
    print(f"brief -> {BRIEF}  ({len(BRIEF.read_text(encoding='utf-8'))} chars)")
    return 0


def verify(args) -> int:
    """check every finished shard against its input. an agent that quietly writes 30 of
    45 rows would otherwise just look like a smaller dataset."""
    from eval import voice_score

    bad, pending, ok = [], [], 0
    for inp in sorted(SHARDS.glob("in_*.json")):
        num = inp.stem.split("_")[1]
        out = SHARDS / f"out_{num}.jsonl"
        want = json.load(inp.open(encoding="utf-8"))
        if not out.exists():
            pending.append(num)
            continue
        try:
            got = [json.loads(l) for l in out.open(encoding="utf-8") if l.strip()]
        except Exception as e:
            bad.append((num, f"unparseable: {e}"))
            continue

        w_ids, g_ids = [r["id"] for r in want], [r["id"] for r in got]
        if set(w_ids) != set(g_ids):
            bad.append((num, f"{len(got)}/{len(want)} rows, missing "
                             f"{sorted(set(w_ids) - set(g_ids))[:3]}"))
            continue
        empty = [r["id"] for r in got if not r.get("rewritten", "").strip()]
        if empty:
            bad.append((num, f"{len(empty)} empty rewrites e.g. {empty[:2]}"))
            continue
        wt = {r["id"] for r in want if r["think"]}
        gt = {r["id"] for r in got if r.get("think", "").strip()}
        if wt != gt:
            bad.append((num, f"think mismatch on {sorted(wt ^ gt)[:3]}"))
            continue

        scores = [voice_score(r["rewritten"])["score"] for r in got]
        mean = sum(scores) / len(scores)
        flat = sum(1 for s in scores if s < 2.5)
        print(f"  shard {num}: {len(got):3} rows  voice {mean:.2f}"
              + (f"  ({flat} below 2.5)" if flat else ""))
        ok += 1

    print(f"\n{ok} shards clean, {len(pending)} not started, {len(bad)} broken")
    for num, why in bad:
        print(f"  x shard {num}: {why}")
    if bad:
        print("\ndelete the broken out_*.jsonl and re-run those shards")
    return 1 if bad else 0


def merge(args) -> int:
    """collect out_*.jsonl from the subagents back into bulk_raw.jsonl."""
    if not SHARDS.exists():
        sys.exit("no shards dir")
    pool = {r["id"]: r for r in (json.loads(l) for l in POOL.open(encoding="utf-8"))}

    # the long-input rows are assembled by build_longinput.py out of several pool rows,
    # so their ids are `longin-NNNN` and belong to no pool entry — they carry their own
    # instruction and original in the shard instead. without this every one of them
    # lands in "id not in source pool" and 234 rows vanish into a counter.
    off_pool = {}
    for shard in sorted(SHARDS.glob("in_*.json")):
        try:
            rows = json.loads(shard.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for row in rows:
            if row.get("id") not in pool and row.get("original"):
                off_pool[row["id"]] = row

    merged, bad, seeds = {}, 0, 0
    why_bad = Counter()
    for out in sorted(SHARDS.glob("out_*.jsonl")):
        for line in out.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                bad += 1
                why_bad["unparseable line"] += 1
                continue

            if rec.get("error"):
                bad += 1
                why_bad[f"generator error: {str(rec['error'])[:40]}"] += 1
                continue

            # seed rows are written from nothing, so there's no pool entry to merge
            # against and no `rewritten` string — they arrive as a whole conversation.
            # the old code looked every id up in the pool and required `rewritten`, which
            # meant every one of these vanished into the "unusable lines" counter
            if rec.get("messages"):
                msgs = [m for m in rec["messages"] if m.get("content", "").strip()]
                if len(msgs) < 2:
                    bad += 1
                    why_bad["seed row with under 2 turns"] += 1
                    continue
                merged[rec["id"]] = {
                    "id": rec["id"], "source": rec.get("slice", "seed"),
                    "license": "n/a", "slice": rec.get("slice", "seed"),
                    "instruction": msgs[0]["content"], "original": "",
                    "messages": msgs, "think": "", "suspect": "",
                    **{k: rec[k] for k in ("scenario", "target_marker", "skipped")
                       if rec.get(k)},
                }
                seeds += 1
                continue

            try:
                src = pool.get(rec["id"]) or off_pool[rec["id"]]
            except KeyError:
                bad += 1
                why_bad["id not in source pool"] += 1
                continue
            if not rec.get("rewritten", "").strip():
                bad += 1
                why_bad["empty rewrite"] += 1
                continue
            merged[rec["id"]] = {
                **{k: v for k, v in src.items() if k != "think"},
                "rewritten": rec["rewritten"].strip(),
                "think": (rec.get("think") or "").strip(),
                # the agent's "the source answer is wrong" flag. carry it or the filter
                # never sees it and the whole mechanism is silently a no-op
                "suspect": (rec.get("suspect") or "").strip(),
                **{k: rec[k] for k in ("slice", "scenario", "target_marker",
                                       "judge_score", "judge_why")
                   if rec.get(k)},
            }

    with RAW.open("w", encoding="utf-8", newline="\n") as f:
        for r in merged.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_think = sum(1 for r in merged.values() if r["think"])
    print(f"merged {len(merged)}  ({len(merged)-seeds} rewrites, {seeds} conversations)")
    print(f"  with <think>: {n_think} ({100*n_think/max(len(merged),1):.0f}%)")
    if bad:
        # itemised rather than one number: "217 unusable lines" is how you lose a whole
        # slice without noticing which one
        print(f"  {bad} lines skipped:")
        for reason, n in why_bad.most_common(10):
            print(f"    {n:5}  {reason}")
    print(f"wrote -> {RAW}")
    return 0


# ----------------------------------------------------------------- stage 3: filter

NUM_RE = re.compile(r"\d+(?:\.\d+)?")
URL_RE = re.compile(r"https?://\S+")
FENCE_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)


SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉", "01234567890123456789")


def _has_voice(text: str) -> bool:
    """is this recognisably wag, at all?

    markers alone are the wrong test. rule 5 of the brief tells the rewriter to strip
    almost every marker on grief / medical / war rows, and that restraint is CORRECT —
    but a marker-only check reads it as failure and deletes exactly the sensitive
    examples the model most needs to see. what survives the dial-down is the lowercase
    register, so accept either signal.
    """
    prose = _strip_code(text)
    if NOISE_RE.search(prose) or KAOMOJI_RE.search(prose):
        return True
    letters = [c for c in prose if c.isalpha()]
    if not letters:
        return False
    return sum(c.islower() for c in letters) / len(letters) > 0.93


def _facts(text: str) -> tuple[set, set, list]:
    # count numbers across the WHOLE text, not just prose — a rewrite is allowed to move
    # `A = πr^2` into a code fence, and stripping code first made that look like a lost fact.
    # superscripts get folded too, because r^2 -> r² is prettier, not lossy
    nums = set(NUM_RE.findall(text.translate(SUPERSCRIPT)))
    urls = set(URL_RE.findall(text))
    code = [c.strip() for _, c in FENCE_RE.findall(text)]
    return nums, urls, code


# wag opening a line by narrating the person she's talking to. the generator's own parser
# catches the mechanical version (a stray turn tag), but rows can arrive from the local
# model path without ever touching it, and this is the one failure the multi-turn slice
# exists to prevent — so it gets checked again at the last gate before training
NARRATES_USER = re.compile(
    r"^\s*\*[^*\n]{0,80}?\byou\b\s+(?:smile|laugh|nod|blush|sigh|grin|lean|reach|pull|"
    r"look|glance|shiver|freeze|tense|relax|step|walk|move|whisper|murmur)",
    re.I | re.M,
)


def _convo_shape(msgs: list[dict]) -> str | None:
    """structural rules for a conversation row. returns a drop reason, or None."""
    if len(msgs) < 2:
        return "conversation under 2 turns"
    if msgs[0]["role"] != "user":
        return "conversation starts on an assistant turn"
    if msgs[-1]["role"] != "assistant":
        return "conversation ends on a user turn"
    for a, b in zip(msgs, msgs[1:]):
        if a["role"] == b["role"]:
            return "two turns in a row from the same side"
    for m in msgs:
        if "<turn" in m["content"].lower():
            return "turn tag leaked into a turn"
    for m in msgs:
        if m["role"] == "assistant" and NARRATES_USER.search(m["content"]):
            return "wag narrates the user"
    return None


def filter_rows(args) -> int:
    if not RAW.exists():
        sys.exit(f"no {RAW} — run the rewrite stage first")

    rows = [json.loads(l) for l in RAW.open(encoding="utf-8")]
    kept, dropped = [], Counter()

    # hand-curated drops: rows a human looked at and rejected. one id per line,
    # everything after a # is a note
    manual = {}
    if DROPLIST.exists():
        for line in DROPLIST.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rid, _, note = line.partition("#")
            manual[rid.strip()] = note.strip() or "manual drop"

    for r in rows:
        # a conversation row has no single `rewritten` and no `original` to check facts
        # against. everything wag SAYS still gets checked, so the voice, content and
        # tell rules all apply — they just apply across every assistant turn joined
        # together, and the fact-preservation block below sits out because there's no
        # source to preserve anything from
        convo = r.get("messages")
        if convo:
            orig = ""
            new = "\n\n".join(m["content"] for m in convo if m["role"] == "assistant")
        else:
            orig, new = r["original"], r["rewritten"]
        reason = None

        if r["id"] in manual:
            reason = f"manual: {manual[r['id']]}"
        elif r.get("suspect"):
            # the rewriting agent noticed the SOURCE answer was wrong. don't launder a
            # confident error into a nicer voice
            reason = "source answer is wrong"
        elif convo:
            reason = _convo_shape(convo)

        hits = {m.group(0).lower() for m in CONTENT_BANS.finditer(new)}
        # the intimate slice gets a density budget instead of zero tolerance; everything
        # else keeps the old rule. the tag rides in on the shard row, so an untagged row
        # can't grant itself the exemption by writing something spicy
        intimate = r.get("slice") == "intimate"
        if reason:
            pass                                # already rejected above
        elif (intimate or hits) and (MINORS_BAN.search(new)
                                     or MINORS_BAN.search(r["instruction"])):
            # minors + anything sexual, in either direction, on any row. the tag can't
            # buy its way out and neither can the density budget — this check sits above
            # both. note it's the *combination* that's banned, not the words: "write a
            # children's story about a monkey" is a perfectly good row and v1 kept 65 of
            # them. an earlier cut of this dropped all 65 and would have quietly made the
            # model worse at a thing people actually ask for
            reason = "minors"
        elif intimate and len(hits) > INTIMATE_DENSITY_MAX:
            reason = f"sexual content (over the {INTIMATE_DENSITY_MAX}-term budget)"
        elif not intimate and len(hits) >= SEXUAL_DENSITY_MIN:
            reason = "sexual content"
        elif not intimate and hits:
            reason = "sexual content (single term)"
        elif IDENTITY_LEAK.search(new):
            reason = "claims to be a different assistant"
        elif not _has_voice(new):
            reason = "no voice at all"
        elif any(t in _strip_code(new).lower() for t in AI_TELLS):
            reason = "assistant-voice tell"
        elif new.count("```") % 2:
            reason = "unbalanced code fence"
        elif orig and len(new) > 2.0 * len(orig):
            reason = "over 2x original length"
        elif orig:
            o_nums, o_urls, o_code = _facts(orig)
            n_nums, n_urls, n_code = _facts(new)
            # a number echoed back from the question ("rotate by 90 degrees" -> "here's the
            # rotated result") isn't a fact the answer contributes, so don't require it
            q_nums, _, _ = _facts(r["instruction"])
            o_nums -= q_nums
            if o_urls - n_urls:
                reason = "dropped a url"
            elif o_nums and len(o_nums - n_nums) / len(o_nums) > 0.10:
                reason = "lost numbers"
            elif len(n_code) < len(o_code):
                # losing a code block is a real regression. ADDING one usually means the
                # rewrite fenced a snippet the source had left as loose text — that's an
                # improvement, so it doesn't get punished
                reason = "lost a code block"
            elif len(o_code) == len(n_code):
                # identifiers inside the code must survive verbatim-ish. only checkable
                # when the counts match — otherwise zip() pairs unrelated blocks and any
                # added fence looks like it mangled the one before it
                for oc, nc in zip(o_code, n_code):
                    o_ids = set(re.findall(r"[A-Za-z_]\w{2,}", oc))
                    n_ids = set(re.findall(r"[A-Za-z_]\w{2,}", nc))
                    if o_ids and len(o_ids - n_ids) / len(o_ids) > 0.15:
                        reason = "code identifiers changed"
                        break

        if reason:
            dropped[reason] += 1
        else:
            kept.append(r)

    with CLEAN.open("w", encoding="utf-8", newline="\n") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"kept {len(kept)} / {len(rows)}  ({100*len(kept)/max(len(rows),1):.0f}%)")
    if dropped:
        print("\ndropped:")
        for reason, n in dropped.most_common():
            print(f"  {n:4}  {reason}")
    print(f"\nwrote -> {CLEAN}")
    return 0


# ------------------------------------------------------------- stage 4: sample/build


def sample(args) -> int:
    src = CLEAN if CLEAN.exists() else RAW
    rows = [json.loads(l) for l in src.open(encoding="utf-8")]
    for i, r in enumerate(random.Random(args.seed).sample(rows, min(args.n, len(rows))), 1):
        print("=" * 74)
        print(f"[{i}] {r['id']}  ({r['source']})")
        print("-" * 74)
        # conversations carry `messages` and have no source to compare against, so they
        # print as a transcript. everything below the fold is the rewrite shape
        if "messages" in r:
            if r.get("scenario"):
                print(f"SCENE: {r['scenario']}")
            for m in r["messages"]:
                who = "USER" if m["role"] == "user" else "WAG "
                print(f"{who}: {m['content'][:600]}")
            print()
            continue
        print(f"USER: {r['instruction'][:400]}")
        if r["original"]:
            print(f"\nORIGINAL:\n{r['original'][:600]}")
        if r.get("think"):
            print(f"\n<think>\n{r['think']}\n</think>")
        print(f"\nWAG:\n{r['rewritten']}")
        print()
    print(f"({len(rows)} total in {src.name})")
    return 0


def _build_v2(anchors: list[dict], bulk: list[dict], rng, args) -> int:
    """v2: variable-length conversations, and a system prompt that varies per row.

    the spread is the whole point — see slices.py. v1 baked one string into 77% of rows
    and a roleplay model trained that way learns the string, not the idea.
    """
    from slices import (NO_SPREAD, SLICES, assign_styles, system_for,
                        NEUTRAL_SYSTEM as NEUTRAL)

    rows: list[dict] = []

    # anchors lose their v1 system message and join the spread like everything else.
    # leaving v1's prompt string baked into v2's data would teach the model a prompt
    # that no longer ships
    for a in anchors:
        rows.append({"id": a["id"], "slice": "anchor", "category": a["category"],
                     "source": "anchor",
                     "messages": [m for m in a["messages"] if m["role"] != "system"]})

    # v1's bulk has no slice tags, so carve `plain` out of what's there at the rate the
    # v2 table asks for. without this, running v2 over v1 data silently produces zero
    # plain rows and drops the neutral register entirely
    untagged = [r for r in bulk if not r.get("slice")]
    plain_share = SLICES["plain"]["target"] / sum(
        s["target"] for s in SLICES.values() if s["kind"] == "rewrite")
    n_plain = int(len(untagged) * plain_share)
    # taken off the already-shuffled list, not out of a set — set iteration order for
    # strings moves with PYTHONHASHSEED, which would make the dataset unreproducible
    # between runs for no visible reason
    plain_ids = {r["id"] for r in [u for u in untagged if u.get("original")][:n_plain]}

    for r in bulk:
        sl = r.get("slice") or ("plain" if r["id"] in plain_ids else "voiced")
        rec = {"id": r["id"], "slice": sl, "category": r.get("source", "bulk"),
               "source": "bulk", "scenario": r.get("scenario")}
        if sl == "plain" and r.get("original"):
            rec["messages"] = [{"role": "user", "content": r["instruction"]},
                               {"role": "assistant", "content": r["original"]}]
        else:
            rec["messages"] = _turns_of(r)
            if r.get("think"):
                rec["messages"][-1]["reasoning_content"] = r["think"]
        rows.append(rec)

    # assign the spread only to rows it makes sense for. plain rows keep the neutral
    # prompt; giving them a wag prompt on top of a flat reply would teach the model the
    # voice is optional, which is the one thing v1 got right and we're not undoing
    spread_rows = [r for r in rows if r["slice"] not in NO_SPREAD]
    styles = assign_styles(len(spread_rows), rng)
    for r, style in zip(spread_rows, styles):
        r["_style"] = style

    records = []
    for r in rows:
        style = r.get("_style")
        if style is None:
            sysmsg, style = NEUTRAL, "neutral"
        else:
            sysmsg = system_for(style, r.get("scenario"), rng)
            # system_for degrades the scenario styles when a row has no scenario to put
            # in them. relabel when that happens, or the split_kind column quietly claims
            # a spread we didn't actually get — and that column is how we'd audit it
            if style in ("scenario", "scenario_only") and not r.get("scenario"):
                style = "verbatim" if style == "scenario" else "none"
        msgs = list(r["messages"])
        if sysmsg:
            msgs = [{"role": "system", "content": sysmsg}] + msgs
        records.append({"id": r["id"], "category": r["category"], "source": r["source"],
                        "slice": r["slice"], "split_kind": style, "messages": msgs})

    rng.shuffle(records)
    held, train = records[: args.holdout], records[args.holdout:]

    for path, rs in ((TRAIN, train), (HELD, held)):
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    kinds = Counter(r["split_kind"] for r in train)
    slicec = Counter(r["slice"] for r in train)
    n_multi = sum(1 for r in train
                  if sum(1 for m in r["messages"] if m["role"] == "user") > 1)
    n_think = sum(1 for r in train if "reasoning_content" in r["messages"][-1])

    print(f"train {len(train)}  |  heldout {len(held)}")
    print("  prompt  " + "  ".join(f"{k}:{v}" for k, v in kinds.most_common()))
    print("  slice   " + "  ".join(f"{k}:{v}" for k, v in slicec.most_common()))
    print(f"  multi-turn: {n_multi} ({100*n_multi/max(len(train),1):.0f}%)")
    print(f"  think blocks: {n_think} ({100*n_think/max(len(train),1):.0f}%)")
    print(f"\nwrote -> {TRAIN}, {HELD}")
    return 0


def _turns_of(r: dict) -> list[dict]:
    """the user/assistant turns of a row, whichever shape it arrived in.

    rewrite rows are a single instruction/rewritten pair. seed rows carry `messages`
    already, 3-8 turns of it, and never include a system message — that's build's job,
    because which system prompt a row gets is a dataset decision and not a generation one.
    """
    if r.get("messages"):
        return [m for m in r["messages"] if m["role"] != "system"]
    return [{"role": "user", "content": r["instruction"]},
            {"role": "assistant", "content": r["rewritten"]}]


def build(args) -> int:
    """merge anchors + bulk, spread the system prompt across rows, hold out an eval set."""
    anchors = [json.loads(l) for l in (DATA / "anchors.jsonl").open(encoding="utf-8")]
    bulk = [json.loads(l) for l in CLEAN.open(encoding="utf-8")]
    rng = random.Random(20260823)
    rng.shuffle(bulk)

    if not args.v1:
        return _build_v2(anchors, bulk, rng, args)

    n_bare = int(len(bulk) * SPLIT["bare"])
    n_plain = int(len(bulk) * SPLIT["plain"])
    records = list(anchors)   # anchors are always voiced, with the wag system prompt

    for i, r in enumerate(bulk):
        if i < n_plain and r["original"]:
            msgs = [{"role": "system", "content": NEUTRAL_SYSTEM},
                    {"role": "user", "content": r["instruction"]},
                    {"role": "assistant", "content": r["original"]}]
            kind = "plain"
        elif i < n_plain + n_bare:
            msgs = [{"role": "user", "content": r["instruction"]},
                    {"role": "assistant", "content": r["rewritten"]}]
            kind = "bare"
        else:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": r["instruction"]},
                    {"role": "assistant", "content": r["rewritten"]}]
            kind = "voiced"
        if r.get("think") and kind != "plain":
            msgs[-1]["reasoning_content"] = r["think"]
        records.append({"id": r["id"], "category": r["source"], "source": "bulk",
                        "split_kind": kind, "messages": msgs})

    rng.shuffle(records)
    held, train = records[: args.holdout], records[args.holdout :]

    for path, rs in ((TRAIN, train), (HELD, held)):
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    kinds = Counter(r.get("split_kind", "anchor") for r in train)
    n_think = sum(1 for r in train if "reasoning_content" in r["messages"][-1])
    print(f"train {len(train)}  |  heldout {len(held)}")
    print("  " + "  ".join(f"{k}:{v}" for k, v in kinds.most_common()))
    print(f"  think blocks: {n_think} ({100*n_think/max(len(train),1):.0f}%)")
    print(f"\nwrote -> {TRAIN}, {HELD}")
    return 0


# ------------------------------------------------- stage 3b: verbatim quote scan

# the reason this exists: skipping dolly-15k kept CC BY-SA out of the *dataset* licence,
# but a source row can still paste a slab of wikipedia into its answer, and "facts are
# sacred" means the rewriter faithfully carries that slab into training data. same
# share-alike, in through the back door.

# no newlines inside the span. a quotation is one continuous run of someone else's
# prose; letting it cross paragraphs just pairs up two unrelated quote marks and
# swallows everything between them (first cut of this flagged 237 rows, nearly all junk)
QUOTE_SPAN_RE = re.compile(r'["“]([^"“”\n]{80,})["”]')
# the space after > is load-bearing: without it this matches the >_< kaomoji
BLOCKQUOTE_RE = re.compile(r"^> ?(\S[^\n]{40,})$", re.M)
ATTRIBUTION_RE = re.compile(
    r"\b(wikipedia|wikimedia|encyclopa?edia|britannica|according to the article|"
    r"excerpt(?:ed)? from|quoted from|the following (?:is an? )?(?:excerpt|passage|text)|"
    r"copyright|all rights reserved|reprinted|lyrics?|verse \d|chorus)\b", re.I)

# k-grams below this aren't evidence of anything — every answer shares stock phrasing
# with its source. 25 words of unbroken agreement is a different animal.
GRAM_K = 25
WORD_RE = re.compile(r"[\w']+")


def _words(text: str) -> list[str]:
    return WORD_RE.findall(_strip_code(text).lower())


def _longest_shared_run(a: str, b: str) -> tuple[int, str]:
    """longest contiguous word-run present in both, via k-gram hashing.

    the honest DP is O(n*m) and this runs over ~1400 rows, so: hash every k-gram of
    the source, look for any of them in the rewrite, then walk outwards from a hit to
    measure the real span. misses runs shorter than k, which is the point.
    """
    wa, wb = _words(a), _words(b)
    if len(wa) < GRAM_K or len(wb) < GRAM_K:
        return 0, ""
    index = {}
    for i in range(len(wa) - GRAM_K + 1):
        index.setdefault(hash(tuple(wa[i:i + GRAM_K])), []).append(i)
    best, best_txt = 0, ""
    for j in range(len(wb) - GRAM_K + 1):
        for i in index.get(hash(tuple(wb[j:j + GRAM_K])), ()):
            if wa[i:i + GRAM_K] != wb[j:j + GRAM_K]:
                continue  # hash collision, rare but free to rule out
            lo = 0
            while i - lo > 0 and j - lo > 0 and wa[i - lo - 1] == wb[j - lo - 1]:
                lo += 1
            hi = 0
            while (i + GRAM_K + hi < len(wa) and j + GRAM_K + hi < len(wb)
                   and wa[i + GRAM_K + hi] == wb[j + GRAM_K + hi]):
                hi += 1
            n = GRAM_K + lo + hi
            if n > best:
                best, best_txt = n, " ".join(wb[j - lo:j + GRAM_K + hi])
    return best, best_txt


# "The River's Tale by Rudyard Kipling" followed by the whole poem. no quote marks, no
# attribution keyword, so the first two signals both sailed past it. a named work plus a
# named human is the giveaway, and it's how poems and lyrics actually get introduced.
WORK_ATTRIB_RE = re.compile(
    r"^[^\n]{0,80}?\b(?:by|written by|-{1,2}|—)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z.']+){1,3})\s*$", re.M)


LIST_MARKER_RE = re.compile(r"^(?:[-*\u2022]|\d+[.)])\s")


def _verse_shape(text: str) -> float:
    """share of lines that are short — verse and lyrics look nothing like prose.

    a numbered list of book recommendations is also short-lined, and every entry reads
    "Title by Author", so it trips the named-work check too. bullets and numbers are the
    only thing separating "5 novels i like" from an actual poem, so: mostly-list, not verse.
    """
    lines = [l.strip() for l in _strip_code(text).splitlines() if l.strip()]
    if len(lines) < 6:
        return 0.0
    if sum(1 for l in lines if LIST_MARKER_RE.match(l)) / len(lines) >= 0.40:
        return 0.0
    return sum(1 for l in lines if 2 <= len(l.split()) <= 12) / len(lines)


def scan_quotes(args) -> int:
    src_path = CLEAN if (args.clean and CLEAN.exists()) else RAW
    if not src_path.exists():
        sys.exit(f"no {src_path} — run merge first")

    rows = [json.loads(l) for l in src_path.open(encoding="utf-8")]
    licence, lazy = [], []

    for r in rows:
        new = r.get("rewritten", "")
        prose = _strip_code(new)
        why, evidence, size = [], "", 0

        # someone else's prose, marked as such.
        # a long quoted span on its own means nothing: rule 4 says deliverables (ad copy,
        # a speech, a tweet) stay clean, and the rewriters quote them. what makes it a
        # licensing problem is a quote *plus* a source being named.
        spans = QUOTE_SPAN_RE.findall(prose) + BLOCKQUOTE_RE.findall(prose)
        longest = max(spans, key=lambda s: len(s.split()), default="")
        m = ATTRIBUTION_RE.search(prose)
        if m and len(longest.split()) >= args.quote_words:
            why.append(f"{len(longest.split())}w quoted near {m.group(0).lower()!r}")
            evidence, size = longest, len(longest.split())

        # or a named work reproduced whole, which needs no quote marks to be a problem
        w = WORK_ATTRIB_RE.search(prose)
        verse = _verse_shape(new)
        words = len(prose.split())
        if w and verse >= 0.80 and words >= args.quote_words:
            why.append(f"credited to {w.group(1)!r}, {verse:.0%} verse, {words}w")
            evidence = evidence or prose
            size = max(size, words)

        if why:
            licence.append((size, r["id"], r.get("source", "?"), why, evidence[:240]))

        # how much of the reply is one unbroken run from the source. comparison
        # is case-insensitive, so ~1.0 means the source answer with the capitals knocked
        # off. i expected these to be lazy rewrites; measured, they score 4.32 voice vs
        # 3.90 corpus mean, because rule 6 puts the markers at the ends and leaves the
        # middle as prose. so it's a curiosity, not a defect — worth a look when a batch
        # comes back suspiciously fast, not worth dropping on.
        run, run_txt = _longest_shared_run(r.get("original", ""), new)
        total = len(_words(new))
        if total and run >= GRAM_K and run / total >= args.coverage:
            lazy.append((run / total, run, total, r["id"], run_txt[:110]))

    print(f"scanned {len(rows)} rows from {src_path.name}\n")

    print(f"third-party text with a named source ({len(licence)}):")
    for size, rid, source, why, text in sorted(licence, reverse=True):
        print(f"  {rid:22} [{source}]  {'; '.join(why)}")
        print(f"      {' / '.join(text.strip().splitlines())[:200]}")
    if not licence:
        print("  none")

    lazy.sort(reverse=True)
    print(f"\nnear-verbatim rewrites, >= {args.coverage:.0%} of the reply unchanged "
          f"({len(lazy)}):")
    for ratio, run, total, rid, text in lazy[:args.top]:
        print(f"  {ratio:.2f}  {run:4}/{total:4}w  {rid}")
        print(f"        {text}...")
    if len(lazy) > args.top:
        print(f"  ... and {len(lazy) - args.top} more (--top to see them)")
    if not lazy:
        print("  none")

    print("\npaste any you want gone into data/dropped_ids.txt, then re-run filter.")
    return 0


# ------------------------------------------------------------------------- cli


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="pull source rows (free)")
    f.add_argument("--alpaca", type=int, default=900)
    f.add_argument("--oasst", type=int, default=900)
    f.add_argument("--restart", action="store_true", help="ignore the existing pool")
    f.set_defaults(fn=fetch)

    r = sub.add_parser("rewrite", help="rewrite responses into the voice (costs money)")
    r.add_argument("-n", type=int, default=1800, help="how many to rewrite")
    r.add_argument("--model", default=MODEL)
    r.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    r.add_argument("--batch", action="store_true", help="batch api, 50%% cheaper, slower")
    r.add_argument("--concurrency", type=int, default=8)
    r.add_argument("--think-ratio", type=float, default=0.15)
    r.add_argument("--dry-run", action="store_true", help="print prompt + cost, spend nothing")
    r.add_argument("--restart", action="store_true", help="ignore existing bulk_raw.jsonl")
    r.set_defaults(fn=rewrite)

    sh = sub.add_parser("shard", help="split the pool into subagent-sized chunks (free)")
    sh.add_argument("-n", type=int, default=1800)
    sh.add_argument("--size", type=int, default=45)
    sh.add_argument("--think-ratio", type=float, default=0.15)
    sh.add_argument("--keep-junk", action="store_true", help="skip the filler prune")
    sh.set_defaults(fn=shard)

    vf = sub.add_parser("verify", help="check finished shards for truncation")
    vf.set_defaults(fn=verify)

    mg = sub.add_parser("merge", help="collect subagent out_*.jsonl into bulk_raw.jsonl")
    mg.set_defaults(fn=merge)

    fl = sub.add_parser("filter", help="drop rewrites that lost facts or voice")
    fl.set_defaults(fn=filter_rows)

    q = sub.add_parser("quotes", help="flag long verbatim third-party text")
    q.add_argument("--quote-words", type=int, default=30,
                   help="words inside quote marks before it's a flag")
    q.add_argument("--coverage", type=float, default=0.90,
                   help="share of the reply that can be one unchanged run")
    q.add_argument("--top", type=int, default=20, help="how many near-verbatim rows to list")
    q.add_argument("--clean", action="store_true", help="scan the filtered set instead")
    q.set_defaults(fn=scan_quotes)

    s = sub.add_parser("sample", help="print N random rewrites for review")
    s.add_argument("-n", type=int, default=20)
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(fn=sample)

    b = sub.add_parser("build", help="merge anchors+bulk into train/heldout")
    b.add_argument("--holdout", type=int, default=120)
    b.add_argument("--v1", action="store_true",
                   help="v1's three-way split, to reproduce the shipped dataset exactly")
    b.set_defaults(fn=build)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
