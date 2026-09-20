"""the 120-prompt eval set and its judge.

§7's point is that 20 hand-scored prompts can't carry the claims made off them. the two
things that make the bigger set worth anything are that it contains prompt shapes the
training data never had, and that the judge is checked against the hand scores before
anyone reads its numbers.
"""
import json
import pathlib
import statistics
import sys
import tempfile
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import eval_judge as EJ
from eval import EVAL_PROMPTS

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> int:
    fails = 0
    rows = [json.loads(l) for l in (ROOT / "data" / "eval_set.jsonl").open(encoding="utf-8")]

    print(f"  eval set: {len(rows)} prompts")
    if len(rows) != 120:
        print(f"  FAIL — §7 asks for 120, got {len(rows)}")
        fails += 1

    hand = [r for r in rows if r.get("hand_scored")]
    if len(hand) != len(EVAL_PROMPTS):
        print(f"  FAIL — calibration set is {len(hand)}, should be all "
              f"{len(EVAL_PROMPTS)} hand-written prompts")
        fails += 1
    else:
        print(f"  calibration rows: {len(hand)}")

    # the whole point of the scene slice: shapes training never contained
    styles = {r["prompt_style"] for r in rows}
    for need in ("scenario", "scenario_only", "none", "paraphrase", "verbatim"):
        if need not in styles:
            print(f"  FAIL — no `{need}` prompts, so that shape goes untested")
            fails += 1
    print(f"  prompt styles present: {len(styles)}")

    # scenario_only rows must genuinely carry no voice instructions, or they're not
    # testing what they claim to
    bad = [r for r in rows if r["prompt_style"] == "scenario_only"
           and r["system"] and "lowercase" in r["system"]]
    if bad:
        print(f"  FAIL — {len(bad)} scenario_only rows leak voice instructions")
        fails += 1
    else:
        print("  scenario_only rows carry no voice instructions: True")

    # no duplicate ids, or the judge's results collide silently
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        print("  FAIL — duplicate ids in the eval set")
        fails += 1

    # references must not be treated as ground truth by the rubric. whitespace-normalised
    # because the rubric is hard-wrapped and the phrase straddles a line break
    rubric = " ".join(EJ.JUDGE_SYSTEM.split())
    if "not ground truth" not in rubric:
        print("  FAIL — judge rubric doesn't warn that the reference may be wrong")
        fails += 1
    else:
        print("  rubric warns the reference isn't ground truth: True")

    # --- judge parsing
    cases = [
        ("clean", "<score>4</score>\n<why>correct and useful</why>\n<invented>no</invented>",
         dict(helpful=4.0, invented=False)),
        ("invented flagged",
         "<score>0</score>\n<why>made up a census figure</why>\n<invented>yes</invented>",
         dict(helpful=0.0, invented=True)),
        ("no score tag", "it was pretty good honestly", "error"),
    ]
    for name, text, want in cases:
        EJ._one_call = lambda *a, **k: {"text": text, "in_tok": 5, "out_tok": 5,
                                        "cached_tok": 0}
        got = EJ.judge_one(None, "m", {"id": "x", "prompt": "p", "response": "r"}, 0.1)
        ok = bool(got.get("error")) if want == "error" else \
            all(got.get(k) == v for k, v in want.items())
        fails += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  judge {name}")

    # --- calibration maths
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="wagtest-")) / "hand.json"
    tmp.write_text(json.dumps({"chat-1": 4, "chat-2": 3, "tech-1": 5}), encoding="utf-8")
    scored = {"chat-1": {"helpful": 4.0}, "chat-2": {"helpful": 4.0},
              "tech-1": {"helpful": 5.0}}
    print()
    EJ.calibrate(scored, tmp)

    print("\nall good" if not fails else f"\n{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
