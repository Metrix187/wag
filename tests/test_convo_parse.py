"""the convo parser is the gate between gemini and training data, so it gets tests.

the cases that matter are the malformed ones — a row that ends on a user turn, or has
wag narrating the other person, is exactly the behaviour these rows are meant to teach
AGAINST, so it must not sail through.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from gemini_convo import parse_convo

T = '<turn role="user">{}</turn>'
W = '<turn role="wag">{}</turn>'

CASES = [
    ("clean 2-turn",
     T.format("hey") + W.format("hi hi~"), True, 2),
    ("clean 6-turn",
     "".join(T.format(f"u{i}") + W.format(f"w{i}") for i in range(3)), True, 6),
    ("whitespace and newlines",
     f"\n\n{T.format('hey')}\n  \n{W.format('hi~')}\n", True, 2),
    ("preamble before the tags",
     "Here is the conversation:\n" + T.format("hey") + W.format("hi~"), True, 2),
    ("ends on a user turn",
     T.format("hey") + W.format("hi~") + T.format("still there?"), False, 0),
    ("starts on a wag turn",
     W.format("hi~") + T.format("hey"), False, 0),
    ("two wag turns in a row",
     T.format("hey") + W.format("hi~") + W.format("also hi"), False, 0),
    ("empty turn",
     T.format("hey") + W.format("   "), False, 0),
    ("single turn only",
     T.format("hey"), False, 0),
    ("no tags at all",
     "hey there, how are you?", False, 0),
    ("model declined",
     "I can't help with that request.", False, 0),
    ("wag narrating the user (nested tag)",
     T.format("hey") + W.format('hi~ <turn role="user">you smile</turn>'), False, 0),
]


def main() -> int:
    fails = 0
    for name, text, want_ok, want_n in CASES:
        turns, why = parse_convo(text)
        ok = bool(turns)
        good = ok == want_ok and (not want_ok or len(turns) == want_n)
        fails += not good
        detail = f"{len(turns)} turns" if ok else f"rejected: {why}"
        print(f"  {'ok  ' if good else 'FAIL'}  {name:34} {detail}")
    print("\nall good" if not fails else f"\n{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
