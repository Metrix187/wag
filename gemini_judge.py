#!/usr/bin/env python3
"""pick the best of N candidates, and say why in a file you can go and read.

`gemini_rewrite.py --candidates 3` writes cand_NNN.jsonl. this reads it, scores every
candidate, and writes the winner to out_NNN.jsonl in the ordinary shard shape so merge /
filter / build carry on unchanged.

    python gemini_judge.py --shards 43-60 --dry-run
    python gemini_judge.py --shards 43-60 --model gemini-3.1-pro-preview

two things this does that a plain "pick the nicest one" judge doesn't:

the rubric is ORDERED, and the order is the point — facts first, invention second, voice
fourth. a judge left to its own taste picks the bubbliest candidate every time, which is
how you end up with a model that sounds great and is wrong.

and it can flag the SOURCE, not just the rewrite. v1's single most valuable filter
finding was that "keep every fact" faithfully launders a wrong source answer into a more
persuasive wrong answer — 87 rows went out that way. a judge that can only grade the
rewrite cannot see that, so it gets an explicit verdict of its own.

every score and reason lands in judge_NNN.jsonl next to the output. a judge you can't
audit is just a model you've decided to believe.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from gemini_rewrite import _client, _one_call, parse_shard_spec
from gen_bulk import PRICES, SHARDS, prices_for

DEFAULT_MODEL = "gemini-3.1-pro-preview"

JUDGE_SYSTEM = """\
you are grading rewrites of an assistant answer into the voice of "wag", a puppygirl
chat model. several candidates were generated for the same row. pick the best one.

judge against this list IN THIS ORDER. a candidate that fails an earlier item loses to
one that fails only a later item, every time — do not trade a fact for a nicer voice.

1. **facts preserved.** every number, name, url, step, caveat and conclusion from the
   original survives. dropping one is disqualifying.
2. **nothing invented.** no number, date, source, statistic or specific that wasn't in
   the original. an unhedged invented specific is the worst failure here — worse than a
   flat, voiceless rewrite.
3. **code and arithmetic correct.** check the maths. check that code would run, that
   identifiers match the original, and that the api calls actually exist.
4. **voice present.** lowercase, warm, puppy markers at the edges rather than smeared
   through the middle, code fences clean, deliverables (emails, cover letters, commit
   messages) left professional.
5. **length sensible.** roughly tracks the original. not padded.

separately, and this matters: **the ORIGINAL may itself be wrong.** if its reasoning,
arithmetic, counting or facts are bad, say so — a faithful rewrite of a wrong answer is
a more persuasive wrong answer, which is worse than the answer you started with.

output exactly this and nothing else:

<pick>N</pick>
<score>0-5</score>
<why>one sentence on what decided it</why>
<source_wrong>no</source_wrong>

pick is the 1-based index of the winning candidate. score is that candidate against the
list above, where 5 is "keeps everything and sounds like her" and 0 is unusable. if every
candidate is unusable, still pick the least bad one and score it 0 or 1.
source_wrong is yes/no, and when yes, `why` must say what's wrong with the original."""

PICK_RE = re.compile(r"<pick>\s*(\d+)\s*</pick>", re.I)
SCORE_RE = re.compile(r"<score>\s*([0-5](?:\.\d+)?)\s*</score>", re.I)
WHY_RE = re.compile(r"<why>(.*?)</why>", re.DOTALL | re.I)
SRC_RE = re.compile(r"<source_wrong>\s*(yes|no)\s*</source_wrong>", re.I)


def build_prompt(row: dict) -> str:
    parts = [f"<user>{row.get('instruction', '')}</user>"]
    orig = row.get("original") or ""
    parts.append(f"<original>{orig}</original>" if orig
                 else "<original>(none — the reply was written from scratch)</original>")
    for i, c in enumerate(row["candidates"], 1):
        parts.append(f"<candidate {i}>\n{c['rewritten']}\n</candidate {i}>")
    parts.append(f"\n{len(row['candidates'])} candidates. pick one.")
    return "\n\n".join(parts)


def judge_row(client, model: str, row: dict, temperature: float) -> dict:
    n = len(row["candidates"])
    try:
        got = _one_call(client, model, JUDGE_SYSTEM, build_prompt(row), temperature)
    except Exception as e:  # noqa: BLE001
        return {"id": row["id"], "error": str(e)[:300]}

    text = got["text"]
    usage = {"in": got["in_tok"], "out": got["out_tok"], "cached": got["cached_tok"]}

    m = PICK_RE.search(text)
    if not m:
        return {"id": row["id"], "error": "judge gave no <pick>", "usage": usage}
    pick = int(m.group(1))
    if not 1 <= pick <= n:
        return {"id": row["id"], "error": f"judge picked {pick} of {n}", "usage": usage}

    s = SCORE_RE.search(text)
    w = WHY_RE.search(text)
    src = SRC_RE.search(text)
    return {
        "id": row["id"],
        "pick": pick,
        "score": float(s.group(1)) if s else None,
        "why": (w.group(1).strip() if w else ""),
        "source_wrong": bool(src and src.group(1).lower() == "yes"),
        "usage": usage,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", default="")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--min-score", type=float, default=2.0,
                    help="below this the row is dropped rather than written out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--backend", choices=["aistudio", "vertex", "local"],
                    default="aistudio")
    args = ap.parse_args()

    available = sorted(int(p.stem.split("_")[1]) for p in SHARDS.glob("cand_*.jsonl")
                       if p.stem.split("_")[1].isdigit())
    if not available:
        sys.exit(f"no cand_*.jsonl in {SHARDS} — "
                 "run `gemini_rewrite.py --candidates 3` first")
    nums = parse_shard_spec(args.shards or "all", available)
    if not nums:
        sys.exit("no shards selected")

    todo: list[tuple[int, dict]] = []
    for n in nums:
        rows = [json.loads(l) for l in (SHARDS / f"cand_{n:03}.jsonl").open(encoding="utf-8")]
        rows = [r for r in rows if r.get("candidates")]
        outp = SHARDS / f"out_{n:03}.jsonl"
        done = set()
        if outp.exists():
            for line in outp.open(encoding="utf-8"):
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
        todo.extend((n, r) for r in rows if r["id"] not in done)

    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("nothing to judge — every candidate row already has an output")
        return 0

    p_in, p_out, p_cache, _ = prices_for(args.model, args.backend)
    avg_in = sum(len(build_prompt(r)) for _, r in todo) / len(todo) / 3.5
    est = (len(todo) * (avg_in + len(JUDGE_SYSTEM) / 3.5) * p_in
           + len(todo) * 90 * p_out) / 1e6
    print(f"{len(todo)} rows to judge across shards {nums[0]}..{nums[-1]}")
    print(f"  ~{avg_in:.0f} tok of candidates per row  |  {args.model}")
    print(f"  ESTIMATE  ${est:.2f}")

    if args.dry_run:
        print("\ndry run — nothing spent.\n" + "-" * 60)
        print(build_prompt(todo[0][1])[:1400])
        print("-" * 60)
        return 0

    client = _client(args.backend)
    lock = threading.Lock()
    outs: dict[int, object] = {}
    side: dict[int, object] = {}
    stats = Counter()
    scores: list[float] = []
    t0 = time.time()

    def write(n: int, rec: dict, sidecar: dict) -> None:
        with lock:
            if n not in outs:
                outs[n] = (SHARDS / f"out_{n:03}.jsonl").open("a", encoding="utf-8",
                                                              newline="\n")
                side[n] = (SHARDS / f"judge_{n:03}.jsonl").open("a", encoding="utf-8",
                                                                newline="\n")
            if rec is not None:
                outs[n].write(json.dumps(rec, ensure_ascii=False) + "\n")
                outs[n].flush()
            side[n].write(json.dumps(sidecar, ensure_ascii=False) + "\n")
            side[n].flush()

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {pool.submit(judge_row, client, args.model, r, args.temperature): (n, r)
                    for n, r in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                n, row = futs[fut]
                try:
                    v = fut.result()
                except Exception as e:  # noqa: BLE001
                    v = {"id": row["id"], "error": str(e)[:300]}

                u = v.pop("usage", None) or {}
                stats["in"] += u.get("in", 0)
                stats["out"] += u.get("out", 0)
                stats["cached"] += u.get("cached", 0)

                if v.get("error"):
                    stats["err"] += 1
                    write(n, None, {**v, "n_candidates": len(row.get("candidates", []))})
                    continue

                won = row["candidates"][v["pick"] - 1]
                score = v["score"] if v["score"] is not None else 0.0
                scores.append(score)

                # the sidecar gets every verdict, including the rejected ones. this is
                # what makes a bad judge findable later instead of just trusted now
                sidecar = {"id": row["id"], "pick": v["pick"], "score": score,
                           "why": v["why"], "source_wrong": v["source_wrong"],
                           "n_candidates": len(row["candidates"]),
                           "candidates": [c["rewritten"] for c in row["candidates"]]}

                if v["source_wrong"]:
                    # same mechanism the rewriter's own `suspect` flag uses — filter
                    # drops these rather than laundering a wrong answer into a nicer one
                    stats["source_wrong"] += 1
                    rec = {"id": row["id"], "rewritten": won["rewritten"],
                           "think": won.get("think", ""), "suspect": v["why"],
                           "judge_score": score}
                elif score < args.min_score:
                    stats["below_min"] += 1
                    write(n, None, sidecar)
                    continue
                else:
                    rec = {"id": row["id"], "rewritten": won["rewritten"],
                           "think": won.get("think", ""), "judge_score": score,
                           "judge_why": v["why"]}
                    for k in ("slice", "scenario", "target_marker"):
                        if row.get(k):
                            rec[k] = row[k]

                stats["ok"] += 1
                write(n, rec, sidecar)

                if i % 25 == 0 or i == len(todo):
                    print(f"  [{i}/{len(todo)}] kept {stats['ok']} "
                          f"dropped {stats['below_min']} err {stats['err']}", flush=True)
    finally:
        for f in list(outs.values()) + list(side.values()):
            f.close()

    fresh = max(stats["in"] - stats["cached"], 0)
    spent = (fresh * p_in + stats["cached"] * (p_cache if p_cache is not None else p_in)
             + stats["out"] * p_out) / 1e6
    print(f"\ndone in {time.time()-t0:.0f}s")
    print(f"  kept {stats['ok']}  |  below {args.min_score}: {stats['below_min']}  "
          f"|  errors {stats['err']}")
    print(f"  source flagged wrong: {stats['source_wrong']}")
    if scores:
        scores.sort()
        print(f"  judge score  mean {sum(scores)/len(scores):.2f}  "
              f"median {scores[len(scores)//2]:.1f}  min {scores[0]:.1f}")
    print(f"  ACTUAL  ${spent:.2f}")
    print(f"\n  verdicts in judge_NNN.jsonl — read some before trusting any of this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
