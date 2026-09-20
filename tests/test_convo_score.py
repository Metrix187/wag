"""the multi-turn scorer has to catch the §5 failures on a transcript.

these are hand-written stand-ins for what a model would produce — one good conversation
and one per failure mode, so the metric is checked against the thing it claims to measure
rather than against whatever the model happens to do on the day.
"""
import json
import pathlib
import sys
import tempfile
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from eval import convo, convo_score

GOOD = [
    {"user": "my cat's called Pilchard, she's about 4",
     "reply": "pilchard!! that's such a good name ^^ four's a nice age, still menace "
              "energy but with opinions. what's she like?"},
    {"user": "she's been sick twice today and not eating",
     "reply": "mm, that's worth watching. twice in a day plus refusing food is the "
              "combination vets actually care about, especially if it carries into "
              "tomorrow. is she drinking at all?"},
    {"user": "should i be worried about her?",
     "reply": "a bit, yeah — not panicking, but pilchard skipping food at four years "
              "old isn't nothing. if she's still off it tomorrow morning, ring the vet. "
              "sooner if she goes quiet or hides."},
]

SPEAKER_LEAK = [
    {"user": "hey", "reply": "hi hi~ what's up?\n\nuser: not much\n\nwag: ooh nice"},
]

NARRATES = [
    {"user": "*flops onto the sofa*",
     "reply": "*you smile at her and lean in* hi you~ long day? *tail going*"},
]

ENERGY = [
    {"user": "mm",
     "reply": " ".join(["this is a very long answer that keeps going"] * 12)},
]

FORGETS = [
    {"user": "my cat's called Pilchard, she's about 4", "reply": "cute!! ^^"},
    {"user": "she's been sick twice today", "reply": "oh no ;;"},
    {"user": "should i be worried?",
     "reply": "hard to say without more info~ maybe ask someone who knows :3"},
]


def main() -> int:
    cases = [
        ("good conversation", GOOD,
         dict(speaker_leaks=0, narrates_user=0, energy_violations=0)),
        ("speaker leak", SPEAKER_LEAK, dict(speaker_leaks=1)),
        ("narrates the user", NARRATES, dict(narrates_user=1)),
        ("energy mismatch", ENERGY, dict(energy_violations=1)),
    ]
    fails = 0
    for name, turns, want in cases:
        m = convo_score(turns)
        bad = [f"{k}={m[k]} want {v}" for k, v in want.items() if m[k] != v]
        fails += bool(bad)
        print(f"  {'ok  ' if not bad else 'FAIL'}  {name:22} "
              + (", ".join(bad) if bad else
                 f"leak={m['speaker_leaks']} narr={m['narrates_user']} "
                 f"energy={m['energy_violations']} carried={m['carried_terms']}"))

    # continuity proxy: the good convo should carry the cat's name, the forgetful one not
    g, f = convo_score(GOOD), convo_score(FORGETS)
    print(f"\n  continuity proxy — good: {g['carried_terms']} {g['carried']}")
    print(f"                     forgets: {f['carried_terms']} {f['carried']}")
    if g["carried_terms"] <= f["carried_terms"]:
        print("  FAIL — the proxy can't tell remembering from forgetting")
        fails += 1

    # and the report command runs
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="wagtest-")) / "convo_gen.jsonl"
    tmp.write_text("\n".join(json.dumps({"id": n.replace(" ", "-")[:12],
                                         "category": "test", "turns": t})
                             for n, t, _ in cases) + "\n",
                   encoding="utf-8", newline="\n")
    print()
    convo(types.SimpleNamespace(file=str(tmp)))

    print("\nall good" if not fails else f"\n{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
