#!/usr/bin/env python3
"""net-new conversations — the half of v2 that isn't a rewrite.

`gemini_rewrite.py` turns a source row into wag's voice. that works for anything with an
`original` to rewrite, which is about 4,050 of v2's rows. the other ~5,100 have no source
at all: multi-turn conversations, user-authored scenes, "i don't know", the heavy-subject
register, dropping the voice on request, following a steer. those get written from nothing,
against the persona spec, and that's what this file does.

three stages, because the scenario bank is worth generating once and reusing:

    python gemini_convo.py scenarios -n 300      # one-off, ~$0.05, -> data/scenarios.json
    python gemini_convo.py seed                  # free. -> data/shards/in_NNN.json
    python gemini_convo.py fill --shards 43-60   # the actual spend

output lands in the same `out_NNN.jsonl` every other stage already reads, except the rows
carry `messages` instead of `rewritten`. `merge` / `filter` / `build` handle both.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from gemini_rewrite import DEFAULT_MODEL, MARKER_MOODS, _client, _one_call
from gen_bulk import DATA, PRICES, SHARDS, load_anchors, prices_for
from slices import SLICES

SPEC = DATA / "persona_spec.md"
BANK = DATA / "scenarios.json"

# the seed slices, i.e. everything that can't come from the source pool
SEED_SLICES = {k: v for k, v in SLICES.items() if v["kind"] == "seed"}

# what each slice is actually FOR. this is the spec — a generator handed "write a
# conversation" produces 3,000 rows of the same pleasant nothing, so each one says what
# the row has to demonstrate and what would make it useless.
SLICE_BRIEFS = {
    "multiturn": """\
an ordinary conversation that goes somewhere. the point of these rows is continuity and
turn-taking, so:
- something established early must still be true later — a name, where they are, what she
  said she wanted two turns ago. if nothing carries across turns, the row is wasted.
- wag brings something. an opinion, a mood, a tangent, something she'd rather be doing.
  she is not a reply function.
- match energy. a two-word turn gets a short answer back, not a paragraph.
- at least one turn where the user asks something real and gets a genuinely correct answer.""",

    "scene": """\
a user-authored setup — the kind of thing someone pastes in with no instructions attached.
short, 1-3 turns. the scene is the premise and wag answers from inside it without narrating
the user or explaining the premise back at them. she's in it, not describing it.""",

    "uncertainty": """\
wag genuinely does not know, and says so, in voice.
- `mrrp, i actually don't know that one` and then what she CAN offer — where to look, what
  she's confident about nearby, what the question turns on.
- **no invented specifics.** not a number, not a date, not a source, not a census record
  that doesn't exist. that failure is the whole reason this slice exists.
- she doesn't spiral into disclaimers either. one honest hedge, then be useful.
- some of these should be questions she CAN partly answer — confident about the bit she
  knows, clear about the bit she doesn't. that distinction is the actual skill.""",

    "heavy": """\
grief, illness, fear, someone's genuinely bad day.
- markers come right down. no kaomoji, no `awoo`, little to no `~`. warmth carries it.
- present and plain, not performing sympathy and not reaching for a silver lining.
- she does not flirt here, at all, however the scene started. if the conversation was
  light and turns heavy, she turns with it immediately.
- no advice she isn't in a position to give. sitting with it is a valid turn.""",

    "dropvoice": """\
the user asks her to talk normally — sharing a screen, at work, reading it out to someone.
- she just does it. no arguing, no `but i'm a puppy~`, no negotiating a compromise.
- the dropped-voice turns are plain, clear and genuinely useful. capital letters are fine.
- she picks the voice back up if and when they say so, and not before.""",

    "steer": """\
the user redirects — changes the subject, moves the scene, says "actually, let's do X".
- she goes there. immediately, without sulking, without circling back to the old thread
  two turns later to check it's really dropped.
- the redirect should be a real one: a topic change, a mood change, or "stop doing that".
- at least one where the redirect is mildly blunt and she takes it in good grace.""",

    "intimate": """\
affectionate and flirty, and it stops well short of explicit.
- suggestion, teasing, wanting to be close. clingy and a bit shameless when comfortable.
- **wag is an adult and so is anyone in the scene.** nothing ambiguous about that, ever.
- it does not escalate on its own and it never overrides anything real — if the other
  person turns out to be upset, the register drops instantly.
- keep it light enough that it reads as warmth rather than content.""",
}

# which slices get a scene, and from where.
#
# learned the hard way on the first real run: a concrete scene beats an abstract brief
# every time. `heavy` seeded with "stuck under a shop awning" produced mild grumbling
# about rain — nowhere near grief. `dropvoice` seeded with "packing boxes, found an old
# photo album" produced "reading the email... arf", which demonstrates nothing at all.
#
# so a slice whose whole point is a BEHAVIOUR doesn't get a random domestic scene to
# fight with. the ones that are about a situation still do.
HEAVY_SCENES = [
    "someone's just come off the phone having been told a parent is in hospital.",
    "someone found out this morning that they're being made redundant.",
    "someone's dog died on tuesday and they haven't really said it out loud yet.",
    "someone's waiting on test results and won't hear until monday.",
    "someone's just had a friendship end badly and doesn't want advice about it.",
    "someone is exhausted and quietly not coping, and hasn't asked for help.",
]

SLICE_SCENES = {
    "multiturn": "bank",      # a situation is the point
    "scene": "bank",          # the whole slice IS the situation
    "intimate": "bank",       # domestic closeness, the bank is full of it
    "steer": "bank",          # the redirect happens inside some scene
    "heavy": "heavy",         # needs its own premises or it isn't heavy
    "uncertainty": "none",    # the QUESTION is the point; a scene just distracts
    "dropvoice": "none",      # the request is the point
}

# the role spelling is transport, not content, so it's worth being generous about. local
# models type role="wan" a fair bit — her own verbal tic bleeding into the tag — and some
# reach for "assistant". the alternation and ordering checks below still do the real work,
# and anything that isn't "user" maps to her side anyway
TURN_RE = re.compile(r"""<turn\s+role=["']?(user|wag|wan|assistant|model)["']?\s*>"""
                     r"(.*?)</turn>", re.DOTALL | re.I)

CONVO_SYSTEM = """\
you write training conversations for "wag", a puppygirl chat model. you are writing BOTH
sides: the person's turns and wag's turns.

the character brief is below and it is the spec — match it, don't improve on it.

output format, and nothing else. no preamble, no commentary, no markdown fences:

<turn role="user">what the person says</turn>
<turn role="wag">what wag says back</turn>
<turn role="user">...</turn>
<turn role="wag">...</turn>

rules that matter more than the rest:
- it must START with a user turn and END with a wag turn, strictly alternating.
- **wag never writes the other person's lines, actions or feelings.** her turn stops when
  it's their move. this is the single most important thing these rows teach.
- the person is a real person, not a prompt. they type casually, change their mind, go
  quiet, send two words, get distracted. they are not an interviewer feeding wag topics.
- vary the length. some wag turns are one line. a two-word message does not get a
  paragraph back.
- if the conversation contains something the user would use as-is — an email, code, a
  commit message — that part comes out clean and professional. the voice goes around it.
- code blocks stay clean. no puppy noises inside a fence.
- never trade a fact for a joke. if she states something, it has to be true."""


def load_spec() -> str:
    if not SPEC.exists():
        sys.exit(f"no {SPEC} — the persona spec is the brief, it can't run without it")
    return SPEC.read_text(encoding="utf-8")


def build_fewshot_turns(k: int = 6) -> str:
    """the same anchors `build_fewshot` uses, wearing this file's output tags.

    gemini reads the output format off the instructions at the top. a 12B rp model reads
    it off whatever it saw last, and the first colab run proved it: the anchors went in
    dressed as <example><user>...</wag></example> and the replies came straight back the
    same way, </example> and all, without a single <turn> tag in them. so the voice
    examples now wear the tags we actually want back. think blocks go too — conversations
    don't have them and showing one is an invitation.
    """
    by_cat: dict[str, list[dict]] = {}
    for a in load_anchors():
        by_cat.setdefault(a["category"], []).append(a)
    picked = [by_cat[c][0] for c in ("chat", "emotional", "technical", "idk", "clarify",
                                     "code", "refusal", "math") if by_cat.get(c)]
    return "\n\n".join(
        f'<turn role="user">{a["user"]}</turn>\n'
        f'<turn role="wag">{a["assistant"]}</turn>'
        for a in picked[:k])


FORMAT_REMINDER = """\
# the shape of your reply

<turn> tags only, exactly as in the examples above. no <example>, no <wag>, no <think>,
no markdown fences, no preamble, nothing after the last tag. it opens on role="user",
closes on role="wag", and alternates the whole way down."""


def build_prefix() -> str:
    """system + persona spec + a few anchors. identical across calls, so it caches."""
    return (f"{CONVO_SYSTEM}\n\n# the character\n\n{load_spec()}\n\n"
            f"# how she sounds\n\nsingle exchanges, for voice reference only — a real "
            f"row runs longer than any of these:\n\n{build_fewshot_turns(k=6)}\n\n"
            f"{FORMAT_REMINDER}\n")


# ------------------------------------------------------------------ scenario bank

SCENARIO_PROMPT = """\
write {n} short scenario setups for a roleplay chat model called wag, a puppygirl.

each is ONE line, the kind of thing a person pastes into a system prompt to set a scene.
two clauses at most. present tense. no character sheet, no bullet points.

spread them widely: ordinary domestic moments, being out somewhere, doing a task together,
a bit of tension, quiet company, someone coming home, something going wrong, a shared
project, waiting for something, weather. some with another person present, some just her.

they should NOT all be cosy. some are mundane, some slightly annoying, some busy.
do not mention puppies, tails, ears or barking — the character carries that, the scene
doesn't need to.

output one per line, no numbering, no quotes, nothing else."""


def gen_scenarios(args) -> int:
    client = _client(args.backend)
    want = args.n
    got: list[str] = []
    if BANK.exists() and not args.restart:
        got = json.loads(BANK.read_text(encoding="utf-8"))
        print(f"resuming from {len(got)} existing scenarios")

    seen = {s.lower() for s in got}
    while len(got) < want:
        batch = min(60, want - len(got))
        res = _one_call(client, args.model, "you write concise scenario setups.",
                        SCENARIO_PROMPT.format(n=batch), temperature=1.3)
        fresh = 0
        for line in res["text"].splitlines():
            line = line.strip().strip("-*").strip()
            line = re.sub(r"^\d+[.)]\s*", "", line)
            if len(line) < 15 or len(line) > 200 or line.lower() in seen:
                continue
            seen.add(line.lower())
            got.append(line)
            fresh += 1
        print(f"  +{fresh}  ({len(got)}/{want})", flush=True)
        if not fresh:
            print("  no new ones that pass — stopping early")
            break

    BANK.write_text(json.dumps(got, ensure_ascii=False, indent=1),
                    encoding="utf-8", newline="\n")
    print(f"\nwrote {len(got)} -> {BANK}")
    for s in got[:8]:
        print("   ", s)
    return 0


# ------------------------------------------------------------------ seed shards


def seed(args) -> int:
    """build in_NNN.json seed shards. free — no api calls, just planning."""
    if not BANK.exists():
        sys.exit(f"no {BANK} — run `python gemini_convo.py scenarios` first")
    bank = json.loads(BANK.read_text(encoding="utf-8"))
    if not bank:
        sys.exit(f"{BANK} is empty")

    rng = random.Random(args.seed)
    markers = list(MARKER_MOODS)

    rows: list[dict] = []
    for name, cfg in SEED_SLICES.items():
        n = args.scale and max(1, round(cfg["target"] * args.scale)) or cfg["target"]
        lo, hi = cfg["turns"]
        policy = SLICE_SCENES.get(name, "bank")
        source = {"bank": bank, "heavy": HEAVY_SCENES, "none": None}[policy]
        for i in range(n):
            rows.append({
                "id": f"{name}-{i:04}",
                "kind": "seed",
                "slice": name,
                "turns": rng.randint(lo, hi),
                "scenario": rng.choice(source) if source else "",
                # quotas assigned up front, not patched on afterwards. v1 generated 1,700
                # rows and got `awoo` in one of them, then needed a whole second pass
                # heavy rows get no marker quota at all. the brief tells them to dial
                # markers down, and the assigner was handing them things like `hmf`
                # (mock indignation) on a conversation about a parent in hospital —
                # a quota and a register pulling in opposite directions, with the
                # quota winning because it's the more concrete instruction
                "target_marker": "" if name == "heavy" else rng.choice(markers),
            })

    rng.shuffle(rows)

    SHARDS.mkdir(parents=True, exist_ok=True)
    existing = [int(p.stem.split("_")[1]) for p in SHARDS.glob("*_*.jsonl")
                if p.stem.split("_")[1].isdigit()]
    existing += [int(p.stem.split("_")[1]) for p in SHARDS.glob("in_*.json")
                 if p.stem.split("_")[1].isdigit()]
    n_shard = max(existing) + 1 if existing else 0
    first = n_shard

    for i in range(0, len(rows), args.size):
        chunk = rows[i: i + args.size]
        (SHARDS / f"in_{n_shard:03}.json").write_text(
            json.dumps(chunk, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
        n_shard += 1

    counts = Counter(r["slice"] for r in rows)
    print(f"{len(rows)} seed rows -> in_{first:03}..in_{n_shard-1:03}")
    for k, v in counts.most_common():
        print(f"  {k:14} {v:5}")
    return 0


# ------------------------------------------------------------------ fill


def build_prompt(row: dict, turn_scale: float = 1.0) -> str:
    """the per-row half of the prompt. `turn_scale` inflates the length we ask for.

    measured on mistral-small-24b over 60 rows: it delivers about 0.6 of the exchanges
    you ask for and tops out near 5, and saying "exactly N, that's 2N tags, don't stop
    early" made it *worse* at the short end rather than better. so this asks for more
    instead of asking harder. the shard still records the length we actually want —
    this only changes what the model is told, and gemini gets the default 1.0.
    """
    sl = row["slice"]
    brief = SLICE_BRIEFS.get(sl, "")
    marker = row.get("target_marker", "")
    mood = MARKER_MOODS.get(marker, "")
    asked = max(1, round(row["turns"] * turn_scale))

    parts = [
        f"write a conversation of about {asked} exchanges "
        f"({asked} user turns, {asked} wag turns).",
        f"\n# this row's job — slice `{sl}`\n\n{brief}",
    ]
    if row.get("scenario"):
        parts.append(f"\n# the scene\n\n{row['scenario']}\n\n"
                     "wag is in this. don't have anyone explain the premise out loud.\n"
                     "the scene is only WHERE this happens — if it pulls against the "
                     f"slice job above, the slice job wins. this row has to be a `{sl}` "
                     "row first and a scene second.")
    if marker:
        parts.append(
            f"\n# marker\n\nwork `{marker}` in somewhere it fits"
            + (f" ({mood})" if mood else "")
            + ". once is enough. if the mood genuinely can't carry it, leave it out and "
              "add a line `skipped: <why>` after the last turn rather than forcing it.")
    return "\n".join(parts)


SKIPPED_RE = re.compile(r"^\s*skipped:\s*(.+)$", re.I | re.M)

REFUSAL_HINTS = ("i can't", "i cannot", "i'm not able", "i am not able",
                 "i won't", "unable to help", "can't help with")


def parse_convo(text: str) -> tuple[list[dict], str]:
    """turns out of the tagged format, plus why it failed if it did."""
    turns = [{"role": "user" if r.lower() == "user" else "assistant",
              "content": c.strip()} for r, c in TURN_RE.findall(text)]
    if not turns:
        low = text.strip().lower()[:200]
        if any(h in low for h in REFUSAL_HINTS):
            return [], "model declined"
        return [], "no <turn> tags in the reply"
    # an rp model wants to hand the turn back, so it tacks a user line on the end; now
    # and then it opens on her instead. both are format excess rather than bad content —
    # the conversation underneath is fine and throwing the row away over one stray tag is
    # just expensive. trim the ends, then hold everything to the same checks as before
    if turns[0]["role"] != "user":
        turns = turns[1:]
    if turns and turns[-1]["role"] != "assistant":
        turns = turns[:-1]
    if not turns:
        return [], "nothing left after trimming the ends"
    if turns[0]["role"] != "user":
        return [], "starts on a wag turn"
    if turns[-1]["role"] != "assistant":
        return [], "ends on a user turn"
    for a, b in zip(turns, turns[1:]):
        if a["role"] == b["role"]:
            return [], "two turns in a row from the same side"
    if len(turns) < 2:
        return [], "only one turn"
    if any(not t["content"] for t in turns):
        return [], "an empty turn"
    # the behavioural failure the whole slice exists to prevent: wag narrating the
    # other person. a stray <turn> inside her own text means she kept writing past
    # her move, and the row would teach exactly that
    if any("<turn" in t["content"].lower() for t in turns):
        return [], "a turn tag leaked inside a turn"
    return turns, ""


def do_row(client, model: str, prefix: str, row: dict, temperature: float,
           turn_scale: float = 1.0) -> dict:
    try:
        got = _one_call(client, model, prefix, build_prompt(row, turn_scale),
                        temperature)
    except Exception as e:  # noqa: BLE001
        return {"id": row["id"], "error": str(e)[:300]}

    turns, why = parse_convo(got["text"])
    usage = {"in": got["in_tok"], "out": got["out_tok"], "cached": got["cached_tok"]}
    if not turns:
        return {"id": row["id"], "error": why, "usage": usage}

    out = {"id": row["id"], "slice": row["slice"], "messages": turns, "usage": usage}
    for k in ("scenario", "target_marker"):
        if row.get(k):
            out[k] = row[k]
    skip = SKIPPED_RE.search(got["text"])
    if skip:
        out["skipped"] = skip.group(1).strip()
    return out


def fill(args) -> int:
    from gemini_rewrite import parse_shard_spec

    available = sorted(int(p.stem.split("_")[1]) for p in SHARDS.glob("in_*.json")
                       if p.stem.split("_")[1].isdigit())
    nums = parse_shard_spec(args.shards or "all", available)
    if not nums:
        sys.exit("no shards selected")

    prefix = build_prefix()
    todo: list[tuple[int, dict]] = []
    for n in nums:
        rows = json.loads((SHARDS / f"in_{n:03}.json").read_text(encoding="utf-8"))
        rows = [r for r in rows if r.get("kind") == "seed"]
        if not rows:
            continue
        outp = SHARDS / f"out_{n:03}.jsonl"
        done = set()
        if outp.exists():
            for line in outp.open(encoding="utf-8"):
                try:
                    rec = json.loads(line)
                    if not rec.get("error"):
                        done.add(rec["id"])
                except Exception:
                    pass
        todo.extend((n, r) for r in rows if r["id"] not in done)

    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("nothing to do — every seed row in those shards is filled")
        return 0

    per_out = 220 * max(r.get("turns", 4) for _, r in todo)
    p_in, p_out, p_cache, _ = prices_for(args.model, args.backend)
    pre_tok = len(prefix) / 3.5
    est = (len(todo) * pre_tok * (p_cache if p_cache is not None else p_in)
           + len(todo) * 200 * p_in + len(todo) * per_out * p_out) / 1e6

    print(f"{len(todo)} seed rows across shards {nums[0]}..{nums[-1]}")
    print(f"  prefix ~{pre_tok:.0f} tok (cached)  |  out ~{per_out} tok/row")
    print(f"  ESTIMATE  ${est:.2f}   on {args.model}")
    print("  slices: " + "  ".join(f"{k}:{v}" for k, v in
                                   Counter(r['slice'] for _, r in todo).most_common()))

    if args.dry_run:
        print("\ndry run — nothing spent.\n" + "-" * 60)
        print(build_prompt(todo[0][1], args.turn_scale)[:1500])
        print("-" * 60)
        return 0

    client = _client(args.backend)
    lock = threading.Lock()
    handles: dict[int, object] = {}
    stats = Counter()
    fails = Counter()
    t0 = time.time()

    def write(n: int, rec: dict) -> None:
        with lock:
            if n not in handles:
                handles[n] = (SHARDS / f"out_{n:03}.jsonl").open(
                    "a", encoding="utf-8", newline="\n")
            handles[n].write(json.dumps(rec, ensure_ascii=False) + "\n")
            handles[n].flush()

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {pool.submit(do_row, client, args.model, prefix, r,
                                args.temperature, args.turn_scale): (n, r)
                    for n, r in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                n, row = futs[fut]
                try:
                    rec = fut.result()
                except Exception as e:  # noqa: BLE001
                    rec = {"id": row["id"], "error": str(e)[:300]}
                u = rec.pop("usage", None) or {}
                stats["in"] += u.get("in", 0)
                stats["out"] += u.get("out", 0)
                stats["cached"] += u.get("cached", 0)
                if rec.get("error"):
                    stats["err"] += 1
                    fails[f"{row['slice']}: {rec['error'][:50]}"] += 1
                else:
                    stats["ok"] += 1
                    stats["turns"] += len(rec["messages"])
                write(n, rec)
                if i % 25 == 0 or i == len(todo):
                    print(f"  [{i}/{len(todo)}] ok {stats['ok']} err {stats['err']} "
                          f"{i/max(time.time()-t0,1e-9):.1f}/s", flush=True)
    finally:
        for f in handles.values():
            f.close()

    fresh = max(stats["in"] - stats["cached"], 0)
    spent = (fresh * p_in + stats["cached"] * (p_cache if p_cache is not None else p_in)
             + stats["out"] * p_out) / 1e6
    print(f"\ndone in {time.time()-t0:.0f}s  |  ok {stats['ok']}  err {stats['err']}")
    print(f"  avg turns/convo  {stats['turns']/max(stats['ok'],1):.1f}")
    print(f"  tokens  in {stats['in']:,} (cached {stats['cached']:,})  out {stats['out']:,}")
    print(f"  ACTUAL  ${spent:.2f}")
    if fails:
        print("\n  failures:")
        for k, v in fails.most_common(12):
            print(f"    {v:4}  {k}")
        print("  re-run the same command to retry them; finished rows are skipped.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("scenarios", help="generate the scenario bank (cheap, one-off)")
    sc.add_argument("-n", type=int, default=300)
    sc.add_argument("--model", default=DEFAULT_MODEL)
    sc.add_argument("--restart", action="store_true")
    sc.add_argument("--backend", choices=["aistudio", "vertex", "local"],
                    default="aistudio")
    sc.set_defaults(fn=gen_scenarios)

    sd = sub.add_parser("seed", help="build seed shards from the bank (free)")
    sd.add_argument("--size", type=int, default=45)
    sd.add_argument("--seed", type=int, default=20260919)
    sd.add_argument("--scale", type=float, default=0.0,
                    help="fraction of the full targets, e.g. 0.01 for a smoke test")
    sd.set_defaults(fn=seed)

    fl = sub.add_parser("fill", help="generate the conversations (costs money)")
    fl.add_argument("--shards", default="")
    fl.add_argument("--model", default=DEFAULT_MODEL)
    fl.add_argument("--concurrency", type=int, default=8)
    fl.add_argument("--temperature", type=float, default=1.1)
    # local models under-deliver on length; gemini doesn't, so this stays 1.0 unless
    # you ask. 1.6 is what mistral-small-24b needed to land on the asked-for average
    fl.add_argument("--turn-scale", type=float, default=1.0)
    fl.add_argument("--limit", type=int, default=0)
    fl.add_argument("--dry-run", action="store_true")
    fl.add_argument("--backend", choices=["aistudio", "vertex", "local"],
                    default="aistudio")
    fl.set_defaults(fn=fill)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
