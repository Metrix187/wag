"""judge logic, with the model mocked out — billing is blocked and this doesn't need it.

what's actually being checked: that a verdict is parsed correctly, that a malformed or
out-of-range pick is refused rather than silently indexing into the wrong candidate, and
that a source-wrong verdict routes to `suspect` so filter drops the row.
"""
import json
import pathlib
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import gemini_judge as J

ROW = {
    "id": "alpaca-1",
    "instruction": "How many seconds in an hour?",
    "original": "There are 3600 seconds in an hour.",
    "candidates": [
        {"rewritten": "wan~ 3600 seconds in an hour ^^", "think": ""},
        {"rewritten": "awoo~ about 3000 seconds i think :3", "think": ""},
        {"rewritten": "3600!! arf~", "think": ""},
    ],
}

VERDICTS = [
    ("clean pick",
     "<pick>1</pick>\n<score>4.5</score>\n<why>keeps the number and sounds right</why>\n"
     "<source_wrong>no</source_wrong>",
     dict(pick=1, score=4.5, source_wrong=False)),
    ("source flagged wrong",
     "<pick>3</pick>\n<score>3</score>\n<why>original says 3000, it's 3600</why>\n"
     "<source_wrong>yes</source_wrong>",
     dict(pick=3, score=3.0, source_wrong=True)),
    ("no pick tag", "I think candidate 1 is best.", "error"),
    ("pick out of range",
     "<pick>7</pick>\n<score>4</score>\n<why>x</why>\n<source_wrong>no</source_wrong>",
     "error"),
    ("pick zero",
     "<pick>0</pick>\n<score>4</score>\n<why>x</why>\n<source_wrong>no</source_wrong>",
     "error"),
    ("missing score, still usable",
     "<pick>2</pick>\n<why>least bad</why>\n<source_wrong>no</source_wrong>",
     dict(pick=2, score=None, source_wrong=False)),
]


def main() -> int:
    fails = 0
    for name, text, want in VERDICTS:
        J._one_call = lambda *a, **k: {"text": text, "in_tok": 10, "out_tok": 5,
                                       "cached_tok": 0}
        got = J.judge_row(None, "m", ROW, 0.2)
        if want == "error":
            ok = bool(got.get("error"))
            detail = got.get("error", "NO ERROR RAISED")
        else:
            ok = all(got.get(k) == v for k, v in want.items())
            detail = {k: got.get(k) for k in want}
        fails += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:28} {detail}")

    # the prompt must actually show the judge every candidate and the original
    p = J.build_prompt(ROW)
    for need in ("<candidate 1>", "<candidate 2>", "<candidate 3>", "3600"):
        if need not in p:
            print(f"  FAIL  prompt missing {need}")
            fails += 1
    print(f"\n  prompt contains all candidates + original: "
          f"{all(x in p for x in ('<candidate 3>', '3600'))}")

    # a from-scratch row (no original) must not render an empty <original>
    scratch = dict(ROW, original="")
    if "(none" not in J.build_prompt(scratch):
        print("  FAIL  from-scratch row renders a blank original")
        fails += 1
    else:
        print("  from-scratch rows labelled, not left blank: True")

    print("\nall good" if not fails else f"\n{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
