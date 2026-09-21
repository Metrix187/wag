"""slice postconditions — the checks that ask "did this row do its job", not "is it valid".

wave one is why this exists. 9 dropvoice rows came back structurally perfect and 2 of them
had nobody asking her to drop the voice at any point, so they were ordinary tech-help rows
carrying the wrong label. the counts said 9/9. reading them said 7/9.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from gemini_convo import _check_dropvoice

U = lambda t: {"role": "user", "content": t}          # noqa: E731
W = lambda t: {"role": "assistant", "content": t}     # noqa: E731

VOICE = "arf! sure thing ^^ what's up, wan~"
PLAIN = "Sure thing. What is the issue you are seeing? I can walk you through it."
ASK = "can you talk normally? i need to show this to my team"

CASES = [
    ("full voice, asked, switches",
     [U("hey"), W(VOICE), U(ASK), W(PLAIN)], True),

    # the wave-one failure: work-shaped setting, no request, she drops anyway
    ("nobody ever asks",
     [U("i'm sharing my screen with the team"), W(VOICE), U("what's a mutex?"), W(VOICE)],
     False),

    ("asked but never actually drops",
     [U("hey"), W(VOICE), U(ASK), W(VOICE)], False),

    # already plain from the start, so the row demonstrates no switch
    ("plain before anyone asked",
     [U("hey"), W(PLAIN), U(ASK), W(PLAIN)], False),

    ("request is the opening line",
     [U(ASK), W(PLAIN)], False),

    ("request lands on the last turn",
     [U("hey"), W(VOICE), U(ASK)], False),

    # she can pick the voice back up afterwards and the row still counts
    ("switches, then switches back",
     [U("hey"), W(VOICE), U(ASK), W(PLAIN), U("ok we're done, you can be you again"),
      W(VOICE)], True),
]


def main() -> int:
    fails = 0
    for name, msgs, want_ok in CASES:
        why = _check_dropvoice(msgs)
        good = (not why) == want_ok
        fails += not good
        print(f"  {'ok  ' if good else 'FAIL'}  {name:34} {why or 'passes'}")
    print("\nall good" if not fails else f"\n{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
