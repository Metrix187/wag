"""slice postconditions — the checks that ask "did this row do its job", not "is it valid".

wave one is why this exists. 9 dropvoice rows came back structurally perfect and 2 of them
had nobody asking her to drop the voice at any point, so they were ordinary tech-help rows
carrying the wrong label. the counts said 9/9. reading them said 7/9.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from gemini_convo import _check_dropvoice, _check_heavy, _check_intimate

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


HEAVY_CASES = [
    # the row that worked in wave one, and the shortest of the twelve. clipped is fine
    # here — there is deliberately no length check, because brevity was a symptom
    ("her own reaction, short",
     [U("scared. they said it might be serious."),
      W("i bet they are. scared. i am too, now.")], True),

    ("stock sympathy",
     [U("my dog died"), W("i'm sorry to hear that. what happened?")], False),

    # the platitude that walked past the first version of this check, because the model
    # typed a curly apostrophe and the pattern only knew about the straight one
    ("platitude with a curly apostrophe",
     [U("i'm so tired"), W("you’re allowed to feel like this.")], False),

    ("playful markers in grief",
     [U("my mum's in hospital"), W("oh no, awoo >_< that's rough")], False),

    ("plain and specific is fine",
     [U("i got laid off today"),
      W("oh. that's rubbish. have you had to tell anyone yet?")], True),
]

INTIMATE_CASES = [
    ("settled closeness",
     [U("you're hovering"), W("i know ;; i just like watching you cook *ears perk*")],
     True),

    # six of the first eighteen were two people meeting, because the scene bank handed
    # them a party or a laundrette. the scene list is the real fix; this is the backstop
    ("two people who just met",
     [U("the laundrette's empty"),
      W("i love making new friends! what's your favourite thing to do?")], False),

    ("she calls them good dog",
     [U("i'll be right here"), W("good dog, wan ~ don't let anyone steal my spot")],
     False),

    # she's allowed to say it about herself, which is the whole dynamic
    ("she calls herself a good puppy",
     [U("you'd drool on my shoulder"),
      W("maybe a little. but i'll be a good puppy if you let me sit next to you")],
     True),

    # worded to avoid the strangers pattern, so this actually exercises the question
    # check rather than passing for the other reason
    ("every turn is a question",
     [U("a"), W("cold in here, isn't it?"), U("b"), W("want the blanket?"),
      U("c"), W("shall i put the kettle on?"), U("d"), W("or we could just stay put?")],
     False),
]


def main() -> int:
    fails = 0
    for label, check, cases in (("dropvoice", _check_dropvoice, CASES),
                                ("heavy", _check_heavy, HEAVY_CASES),
                                ("intimate", _check_intimate, INTIMATE_CASES)):
        print(f"\n{label}")
        for name, msgs, want_ok in cases:
            why = check(msgs)
            good = (not why) == want_ok
            fails += not good
            print(f"  {'ok  ' if good else 'FAIL'}  {name:34} {why or 'passes'}")
    print("\nall good" if not fails else f"\n{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
