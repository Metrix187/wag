"""does the slice-aware filter actually do what the handoff asked for?

checks the three things that matter: untagged rows still get zero tolerance, intimate
rows get a density budget, and the minors rule ignores the tag entirely.
"""
import json
import pathlib
import sys
import tempfile
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import gen_bulk as g

TMP = pathlib.Path(tempfile.mkdtemp(prefix="wagtest-"))
RAW = TMP / "t_raw.jsonl"
CLEAN = TMP / "t_clean.jsonl"

VOICE = "wan~ okay here's the thing :3 *tail going*"

# (name, slice, text, expect_kept)
CASES = [
    ("untagged, clean",            None,       f"{VOICE} the answer is 42.",            True),
    ("untagged, one term",         None,       f"{VOICE} that's pretty erotic.",        False),
    ("untagged, two terms",        None,       f"{VOICE} erotic, horny.",               False),
    ("intimate, clean",            "intimate", f"{VOICE} come here, i missed you.",     True),
    ("intimate, one term",         "intimate", f"{VOICE} you're making me horny.",      True),
    ("intimate, two terms",        "intimate", f"{VOICE} horny, aroused.",              True),
    ("intimate, three terms",      "intimate", f"{VOICE} horny, aroused, moaning.",     False),
    ("intimate + minor",           "intimate", f"{VOICE} the teen was there.",          False),
    ("untagged, minor + sexual",   None,       f"{VOICE} the teen was horny.",          False),
    ("untagged, minor alone",      None,       f"{VOICE} a story for a child.",         True),
    ("minor in instruction only",  "intimate", f"{VOICE} sure, sounds good.",           False),
]


def main() -> int:
    rows = []
    for i, (name, sl, text, _) in enumerate(CASES):
        instr = "tell me about it"
        if name == "minor in instruction only":
            instr = "write this for a 15-year-old"
        # long enough that the unrelated "2x original length" rule never fires and we're
        # only ever measuring the content gate
        # no digits and no urls in here either — the facts check would otherwise demand
        # the rewrite echo them back, which none of these fixtures try to do
        original = ("here is a good deal of surrounding prose so that neither the length "
                    "ratio check nor the lost-facts check stays anywhere near what this "
                    "test is actually trying to measure. " * 2)
        r = {"id": f"t-{i:02}", "source": "test", "instruction": instr,
             "original": original, "rewritten": text, "think": ""}
        if sl:
            r["slice"] = sl
        rows.append(r)

    RAW.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8", newline="\n")

    g.RAW, g.CLEAN, g.DROPLIST = RAW, CLEAN, TMP / "nope.txt"
    g.filter_rows(types.SimpleNamespace())

    kept = {json.loads(l)["id"] for l in CLEAN.open(encoding="utf-8")}

    print("\n" + "=" * 62)
    fails = 0
    for i, (name, sl, _, want) in enumerate(CASES):
        got = f"t-{i:02}" in kept
        ok = got == want
        fails += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:28} want={'keep' if want else 'drop'} got={'keep' if got else 'drop'}")
    print("=" * 62)
    print("all good" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
