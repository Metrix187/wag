#!/usr/bin/env python3
"""build the long-input slice (§6) out of the pool we already have.

    python build_longinput.py --dry-run
    python build_longinput.py -n 250 --size 25

§6 is a real decision and this file only covers one side of it: v1 ships a YaRN config
good for 400,000 tokens and not one training example longer than a few hundred. either
train the long context or drop the claim back to native 256k. on the 4B the claim is a
bigger cheque — 8 full-attention layers with 4 kv heads puts 400k at **12.2 GiB of KV
cache**, alongside a 2.9 GB q4_k_m, for whoever downloads this.

**nothing in the source pool is long.** median row is ~198 tokens, p95 is 580, and
exactly 3 of 8,813 rows clear 1,000. so the slice can't be found, only built.

the shape here is needle-in-a-document, chosen for one reason: **the answer exists by
construction.** the document is assembled from real pool rows, and the question targets
exactly one of them, so the correct answer is that row's own response — known, not
invented. that's the same trick the deferred narcan work relies on when it pastes real
records in as tool results: you never have to let a model make up the thing it's being
graded against.

the rows come out in the ordinary shard shape, so gemini_rewrite (or --backend local)
voices them like anything else and merge/filter/build need no changes.

⚠️ these rows fight the memory constraint. gradient checkpointing raises CheckpointError
on qwen3.5's linear-attention layers, so activation memory scales straight with sequence
length and there's no lever. train this slice as a separate short run at batch 1 — do
not mix 3k-token rows into a batch-4 run and hope.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
POOL = DATA / "source_pool.jsonl"
SHARDS = DATA / "shards"

# ~3.5 chars a token. aiming at 1,500-3,500 tokens of document: long enough that it's
# genuinely a retrieval task, short enough to survive a batch-1 run on a 32-layer model
# with no gradient checkpointing available
TARGET_TOK = (1500, 3500)

QUESTION_FRAMES = [
    "i've pasted a pile of notes below. somewhere in there is the answer to this:\n{q}",
    "long one, sorry. using only what's in the notes below, answer this:\n{q}",
    "here's a dump of reference material. i need this one thing out of it:\n{q}",
    "read through this lot and tell me:\n{q}",
]


def _clean(text: str) -> str:
    r"""alpaca-cleaned carries literal backslash-n in ~65 rows — the two characters, not
    a newline. 4 of them reached v1's train.jsonl, which teaches the model to type escape
    sequences at people. not fixed in the shared filter on purpose: dropping them would
    change v1's output and it currently reproduces byte-identical, which is worth more."""
    return text.replace("\\n", "\n").replace("\\t", "\t")


def _good_needle(r: dict) -> bool:
    """can this row's instruction work as the question?

    alpaca carries instructions that are just an input fragment — "France",
    "expressive" — and a one-word needle is not a retrieval target, it's a word search.
    gen_bulk's own junk filter catches most of that class already, so use it rather
    than inventing a second opinion.
    """
    from gen_bulk import is_junk

    q = r["instruction"].strip()
    return len(q) >= 30 and len(q.split()) >= 5 and not is_junk(r)


def build_rows(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    usable = [r for r in pool
              if r.get("original") and 200 < len(r["instruction"]) + len(r["original"]) < 2500]
    rng.shuffle(usable)
    rows, i = [], 0

    while len(rows) < n and i < len(usable) - 4:
        lo, hi = TARGET_TOK[0] * 3.5, TARGET_TOK[1] * 3.5
        chunk, size = [], 0
        while size < lo and i < len(usable):
            r = usable[i]; i += 1
            piece = f"## {_clean(r['instruction']).strip()}\n\n{_clean(r['original']).strip()}"
            if size + len(piece) > hi and chunk:
                break
            chunk.append(r)
            size += len(piece) + 2
        if len(chunk) < 3:
            continue

        # the needle is never first or last — those are the positions a model can get
        # right by pure recency or primacy, which would let a row pass without the
        # retrieval it's meant to be teaching
        # only the middle sections can be the needle, and only ones whose instruction
        # is a real question. if none of them qualify, throw the chunk away rather than
        # shipping a row whose question is the word "France"
        candidates = [c for c in chunk[1:-1] if _good_needle(c)]
        if not candidates:
            continue
        target = rng.choice(candidates)

        # half the rows hide the target's heading behind a generic one. with every
        # heading present the question is a verbatim copy of one of them, so the task
        # collapses into "find the identical string" — which a model can learn without
        # reading anything. blanking it forces the match to be about content.
        blind = rng.random() < 0.5
        parts = []
        for c in chunk:
            head = ("## notes" if (blind and c is target)
                    else f"## {_clean(c['instruction']).strip()}")
            parts.append(f"{head}\n\n{_clean(c['original']).strip()}")
        doc = "\n\n".join(parts)
        frame = rng.choice(QUESTION_FRAMES).format(q=_clean(target["instruction"]).strip())

        rows.append({
            "id": f"longin-{len(rows):04}",
            "slice": "longinput",
            "think": False,
            "instruction": f"{frame}\n\n---\n\n{doc}",
            # known by construction — it's the source row's own answer, so no model ever
            # has to invent the thing this row is graded against
            "original": _clean(target["original"]).strip(),
            "needle_id": target["id"],
            "needle_blind": blind,
            "doc_sections": len(chunk),
            "doc_chars": len(doc),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=250)
    ap.add_argument("--size", type=int, default=25,
                    help="rows per shard. smaller than usual — these rows are big")
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not POOL.exists():
        raise SystemExit(f"no {POOL}")
    pool = [json.loads(l) for l in POOL.open(encoding="utf-8")]
    rows = build_rows(pool, args.n, random.Random(args.seed))

    toks = sorted(r["doc_chars"] / 3.5 for r in rows)
    secs = sorted(r["doc_sections"] for r in rows)
    print(f"{len(rows)} long-input rows")
    print(f"  document tokens  median {toks[len(toks)//2]:.0f}  "
          f"min {toks[0]:.0f}  max {toks[-1]:.0f}")
    print(f"  sections per doc median {secs[len(secs)//2]}  "
          f"min {secs[0]}  max {secs[-1]}")
    print(f"  longest row would need SEQ_LEN >= {toks[-1] + 400:.0f}")

    if args.dry_run:
        r = rows[0]
        print(f"\ndry run — nothing written.\n{'-'*62}")
        print(r["instruction"][:700])
        print(f"...\n[{r['doc_sections']} sections, needle is {r['needle_id']}]")
        print(f"{'-'*62}\nanswer:\n{r['original'][:300]}")
        return 0

    SHARDS.mkdir(parents=True, exist_ok=True)
    existing = [int(p.stem.split("_")[1]) for p in SHARDS.glob("*_*.json*")
                if p.stem.split("_")[1].isdigit()]
    n_shard = max(existing) + 1 if existing else 0
    first = n_shard
    for i in range(0, len(rows), args.size):
        (SHARDS / f"in_{n_shard:03}.json").write_text(
            json.dumps(rows[i:i + args.size], ensure_ascii=False, indent=1),
            encoding="utf-8", newline="\n")
        n_shard += 1

    print(f"\nwrote in_{first:03}..in_{n_shard-1:03}")
    print("  voice them with: python gemini_rewrite.py --shards "
          f"{first}-{n_shard-1} --backend local")
    print("  and train them as a SEPARATE batch-1 run — see the note at the top.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
