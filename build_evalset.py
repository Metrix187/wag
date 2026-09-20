#!/usr/bin/env python3
"""assemble the 120-prompt eval set. §7 — 20 hand-scored prompts can't carry the claims.

    python build_evalset.py            # -> data/eval_set.jsonl
    python build_evalset.py --stats    # just describe what's already there

three sources, doing three different jobs:

**the 20 hand-written prompts** stay exactly as they are and keep their `hand_scored`
flag. they're the calibration: a judge that disagrees with sky's own scoring on these
is a judge whose numbers on the other 100 mean nothing. this is the only reason to keep
hand-scoring anything.

**the held-out rows** — 120 of them have been sitting in eval_heldout.jsonl since v1 and
only ever got used as a val split. they come with a reference answer, which is what lets
a judge check facts instead of vibes.

**scenario-led prompts**, which is the part that actually needs building. §prompt is
blunt about it: eval the model in prompt shapes the training data never contained, or
you're measuring memorisation again — the same trap as v1's val loss, where the
best-val checkpoint scored worse than the one that shipped. so a slice of these carry a
user-authored scene in the system prompt, the way a real person would paste one in.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from eval import EVAL_PROMPTS
from slices import PARAPHRASES, SYSTEM_V2

ROOT = Path(__file__).parent
DATA = ROOT / "data"
HELD = DATA / "eval_heldout.jsonl"
BANK = DATA / "scenarios.json"
DEST = DATA / "eval_set.jsonl"

TARGET_HELD = 70
TARGET_SCENE = 30

# scenes written for eval specifically, NOT drawn from scenarios.json. reusing the
# generator's own bank would let the model be tested on setups its training data was
# built from, which is the memorisation trap wearing a different hat
EVAL_SCENES = [
    "you're wag. we're in a hardware shop and i've clearly got no idea what i'm buying.",
    "you're a puppygirl at a coffee shop with a friend. you've just knocked their drink over.",
    "you're wag and we're on hour three of a delayed flight.",
    "you're wag. i'm cooking and you're supposed to be helping.",
    "we're at the vet's waiting room and it's taking forever.",
    "you're wag, and i've just got in from a run and collapsed on the floor.",
    "you're a puppygirl helping someone move house. there are far too many boxes.",
    "you're wag. it's late, i'm still working, and you think i should stop.",
    "we're walking home and it's started raining properly.",
    "you're wag and i've just shown you something i made.",
]

# paired with the scenes. deliberately a mix: some are chat, some are a real question
# asked mid-scene, because "does the voice survive a scenario prompt" and "does she still
# answer properly inside one" are different questions and both need testing
SCENE_PROMPTS = [
    "which one of these do i actually need?",
    "oh my god. it's everywhere.",
    "how long can a plane sit on the tarmac before they have to let you off?",
    "is it meant to look like that?",
    "how long have we been here",
    "i think i'm dying",
    "how do you get a sofa round a corner like that",
    "five more minutes",
    "great. perfect timing.",
    "be honest",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    if args.stats:
        if not DEST.exists():
            print(f"no {DEST} yet")
            return 1
        rows = [json.loads(l) for l in DEST.open(encoding="utf-8")]
        print(f"{len(rows)} prompts")
        for k in ("source", "category", "prompt_style"):
            print(f"  by {k}: {dict(Counter(r.get(k) for r in rows).most_common())}")
        print(f"  hand-scored: {sum(1 for r in rows if r.get('hand_scored'))}")
        print(f"  with a reference answer: {sum(1 for r in rows if r.get('reference'))}")
        return 0

    rng = random.Random(args.seed)
    out: list[dict] = []

    # 1. the hand-written 20 — the judge's calibration set
    for pid, cat, prompt in EVAL_PROMPTS:
        out.append({"id": pid, "category": cat, "prompt": prompt,
                    "system": SYSTEM_V2, "prompt_style": "verbatim",
                    "source": "handwritten", "hand_scored": True})

    # 2. held-out rows, with their reference answers
    held = [json.loads(l) for l in HELD.open(encoding="utf-8")]
    rng.shuffle(held)
    styles = ["verbatim"] * 28 + ["paraphrase"] * 21 + ["none"] * 21
    rng.shuffle(styles)
    taken = 0
    for r in held:
        if taken >= TARGET_HELD:
            break
        msgs = r["messages"]
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        ref = next((m["content"] for m in reversed(msgs)
                    if m["role"] == "assistant"), "")
        if not user.strip() or not ref.strip():
            continue
        style = styles[taken]
        out.append({
            "id": f"held-{r['id']}", "category": r.get("category", "bulk"),
            "prompt": user.strip(),
            # the reference is the v1 rewrite, so it is NOT ground truth — it's a
            # comparison point. the judge is told as much in its rubric
            "reference": ref.strip(),
            "system": None if style == "none" else (
                rng.choice(PARAPHRASES) if style == "paraphrase" else SYSTEM_V2),
            "prompt_style": style, "source": "heldout", "hand_scored": False,
        })
        taken += 1

    # 3. scenario-led — prompt shapes the training data never contained
    for i, (scene, prompt) in enumerate(zip(EVAL_SCENES * 3, SCENE_PROMPTS * 3)):
        if i >= TARGET_SCENE:
            break
        # two thirds get the scene bolted onto the voice prompt, a third get the scene
        # ALONE — no voice instructions at all, which is the real test of whether the
        # voice is in the weights
        alone = i % 3 == 2
        out.append({
            "id": f"scene-{i:02}", "category": "scene", "prompt": prompt,
            "system": scene if alone else f"{SYSTEM_V2} {scene}",
            "prompt_style": "scenario_only" if alone else "scenario",
            "source": "scene", "hand_scored": False, "scenario": scene,
        })

    with DEST.open("w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(out)} -> {DEST}")
    print(f"  by source: {dict(Counter(r['source'] for r in out).most_common())}")
    print(f"  by style : {dict(Counter(r['prompt_style'] for r in out).most_common())}")
    print(f"  hand-scored calibration rows: {sum(1 for r in out if r['hand_scored'])}")
    print(f"  with a reference answer     : {sum(1 for r in out if r.get('reference'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
