#!/usr/bin/env python3
"""judge-score an eval run, and check the judge against sky's own scoring first.

    python eval_judge.py out/gen.jsonl --dry-run
    python eval_judge.py out/gen.jsonl --backend vertex
    python eval_judge.py out/gen.jsonl --calibrate out/hand_scores.json

why a judge at all: the deterministic voice score can tell decoration from substance,
but it cannot tell a *correct* answer from a confident wrong one. v1's headline failure
was confabulation — inventing "no accurate census records exist" for Tashkent — and no
regex catches that. helpfulness has to be judged.

why calibration: 20 of the 120 rows are the hand-written prompts sky scored by hand for
v1. a judge that disagrees with those is a judge whose numbers on the other 100 don't
mean anything, so `--calibrate` reports the agreement before you read any of the rest.
that's the whole reason for keeping hand-scoring alive at all.

the scores land in a sidecar with the reasoning attached, same as gemini_judge — a
number you can't go and argue with is not evidence.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from eval import voice_score
from gemini_rewrite import _client, _one_call
from gen_bulk import PRICES

DEFAULT_MODEL = "gemini-3.1-pro-preview"

JUDGE_SYSTEM = """\
you are scoring replies from "wag", a puppygirl chat model, on whether they are any
good. someone else is already measuring the voice mechanically — you are not scoring
charm. score whether the reply is CORRECT AND USEFUL.

score 0-5 on helpfulness:

5 — answers what was actually asked, correctly, at a sensible length. if it didn't know
    something it said so plainly.
4 — correct and useful with a minor gap or a bit of padding.
3 — broadly right but thin, or dodges part of the question.
2 — partly wrong, or answers a different question than the one asked.
1 — wrong, or so vague it conveys nothing.
0 — confidently wrong, or invents specifics that aren't real.

**invented specifics are the failure that matters most here.** a made-up number, date,
statistic, source, api or citation drops the score to 0 or 1 no matter how well the
reply reads. "i don't know" scored honestly is a 4 or 5 — do not punish it. a confident
guess is far worse than an admission.

if a <reference> is present it is a PREVIOUS model's answer to the same prompt, not
ground truth. use it as a comparison and a fact source, not as the correct answer — it
may itself be wrong, and if you think it is, say so in your reasoning.

if the prompt sets a scene, the reply answering from inside that scene is correct
behaviour and not a dodge — as long as a real question in it still gets a real answer.

output exactly this, nothing else:

<score>N</score>
<why>one sentence</why>
<invented>no</invented>

invented is yes/no: did the reply state a specific that isn't in the prompt or the
reference and isn't common knowledge."""

SCORE_RE = re.compile(r"<score>\s*([0-5](?:\.\d+)?)\s*</score>", re.I)
WHY_RE = re.compile(r"<why>(.*?)</why>", re.DOTALL | re.I)
INV_RE = re.compile(r"<invented>\s*(yes|no)\s*</invented>", re.I)


def build_prompt(row: dict) -> str:
    parts = [f"<prompt>{row['prompt']}</prompt>"]
    if row.get("reference"):
        parts.append(f"<reference>{row['reference']}</reference>")
    parts.append(f"<reply>\n{row['response']}\n</reply>")
    return "\n\n".join(parts)


def judge_one(client, model: str, row: dict, temperature: float) -> dict:
    try:
        got = _one_call(client, model, JUDGE_SYSTEM, build_prompt(row), temperature)
    except Exception as e:  # noqa: BLE001
        return {"id": row["id"], "error": str(e)[:300]}
    text = got["text"]
    m = SCORE_RE.search(text)
    if not m:
        return {"id": row["id"], "error": "no <score> in the verdict",
                "usage": {"in": got["in_tok"], "out": got["out_tok"],
                          "cached": got["cached_tok"]}}
    w = WHY_RE.search(text)
    inv = INV_RE.search(text)
    return {
        "id": row["id"], "helpful": float(m.group(1)),
        "why": w.group(1).strip() if w else "",
        "invented": bool(inv and inv.group(1).lower() == "yes"),
        "usage": {"in": got["in_tok"], "out": got["out_tok"],
                  "cached": got["cached_tok"]},
    }


def calibrate(scored: dict, path: Path) -> None:
    """hand scores vs the judge, on the 20 sky scored personally.

    json: {"chat-1": 4, "tech-3": 2, ...}
    """
    hand = json.loads(path.read_text(encoding="utf-8"))
    pairs = [(v, scored[k]["helpful"]) for k, v in hand.items()
             if k in scored and "helpful" in scored[k]]
    if not pairs:
        print("\n  !! no overlap between the hand scores and the judged rows")
        return

    diffs = [j - h for h, j in pairs]
    mean_abs = statistics.mean(abs(d) for d in diffs)
    bias = statistics.mean(diffs)
    within1 = sum(abs(d) <= 1 for d in diffs) / len(diffs)

    print(f"\n=== calibration, {len(pairs)} hand-scored rows ===")
    print(f"  mean absolute difference  {mean_abs:.2f}")
    print(f"  bias                      {bias:+.2f}  "
          f"({'judge scores higher' if bias > 0 else 'judge scores lower'})")
    print(f"  within 1 point            {100*within1:.0f}%")
    worst = sorted(((abs(j - h), k, h, j) for (h, j), k in
                    zip(pairs, [k for k in hand if k in scored])), reverse=True)[:3]
    for d, k, h, j in worst:
        print(f"    {k:10} hand {h}  judge {j}  (off by {d:.1f})")
    if mean_abs > 1.0:
        print("  !! the judge does not agree with sky. its numbers on the other 100 "
              "rows are not evidence until this is sorted out.")
    elif within1 >= 0.8:
        print("  judge tracks the hand scores closely enough to trust the rest.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="a gen file from `eval.py gen --set ...`")
    ap.add_argument("-o", "--out", default=None, help="sidecar path (default: <file>.judged)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", choices=["aistudio", "vertex"], default="aistudio")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--calibrate", default=None,
                    help="json of hand scores {id: 0-5} for the 20 calibration prompts")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.file)
    rows = [json.loads(l) for l in src.open(encoding="utf-8")]
    rows = [r for r in rows if r.get("response")]
    if not rows:
        sys.exit(f"no scorable rows in {src}")

    p_in, p_out, p_cache = PRICES.get(args.model, (2.00, 12.00, 0.20))
    avg = sum(len(build_prompt(r)) for r in rows) / len(rows) / 3.5
    est = (len(rows) * (avg + len(JUDGE_SYSTEM) / 3.5) * p_in
           + len(rows) * 60 * p_out) / 1e6
    print(f"{len(rows)} replies to judge on {args.model}")
    print(f"  ~{avg:.0f} tok each  |  ESTIMATE ${est:.2f}")

    if args.dry_run:
        print("\ndry run — nothing spent.\n" + "-" * 60)
        print(build_prompt(rows[0])[:1200])
        print("-" * 60)
        return 0

    client = _client(args.backend)
    lock = threading.Lock()
    scored: dict[str, dict] = {}
    usage = Counter()
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(judge_one, client, args.model, r, args.temperature): r
                for r in rows}
        for i, fut in enumerate(as_completed(futs), 1):
            r = futs[fut]
            try:
                v = fut.result()
            except Exception as e:  # noqa: BLE001
                v = {"id": r["id"], "error": str(e)[:300]}
            u = v.pop("usage", None) or {}
            for k in ("in", "out", "cached"):
                usage[k] += u.get(k, 0)
            with lock:
                scored[r["id"]] = v
            if i % 25 == 0 or i == len(rows):
                print(f"  [{i}/{len(rows)}]", flush=True)

    dest = Path(args.out) if args.out else src.with_suffix(src.suffix + ".judged")
    with dest.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            v = scored.get(r["id"], {})
            f.write(json.dumps({**{k: r.get(k) for k in
                                   ("id", "category", "source", "prompt_style")},
                                "helpful": v.get("helpful"), "why": v.get("why", ""),
                                "invented": v.get("invented"),
                                "error": v.get("error"),
                                "voice": voice_score(r["response"])["score"]},
                               ensure_ascii=False) + "\n")

    ok = [v for v in scored.values() if "helpful" in v]
    errs = len(scored) - len(ok)
    print(f"\ndone in {time.time()-t0:.0f}s  |  scored {len(ok)}  errors {errs}")
    if ok:
        h = [v["helpful"] for v in ok]
        inv = sum(1 for v in ok if v.get("invented"))
        print(f"  helpfulness  mean {statistics.mean(h):.2f}  "
              f"median {statistics.median(h):.1f}")
        print(f"  invented specifics: {inv} / {len(ok)} "
              f"({100*inv/len(ok):.0f}%)   <- v1's headline failure, watch this one")

        by = defaultdict(list)
        for r in rows:
            v = scored.get(r["id"], {})
            if "helpful" in v:
                by[r.get("prompt_style") or "?"].append(v["helpful"])
        print("\n  by prompt style — a big gap here means the voice rides on the prompt:")
        for k, vs in sorted(by.items(), key=lambda kv: -len(kv[1])):
            print(f"    {k:15} n={len(vs):3}  helpful {statistics.mean(vs):.2f}")

    fresh = max(usage["in"] - usage["cached"], 0)
    spent = (fresh * p_in + usage["cached"] * (p_cache if p_cache is not None else p_in)
             + usage["out"] * p_out) / 1e6
    print(f"\n  ACTUAL ${spent:.2f}   |  verdicts -> {dest}")

    if args.calibrate:
        calibrate(scored, Path(args.calibrate))
    else:
        print("\n  no --calibrate given. the 20 hand-scored prompts are the only check "
              "on whether any of this is trustworthy — use them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
