"""does a conversation row actually survive merge -> filter -> build?

this is the bit that was broken: merge looked every id up in the source pool and required
a `rewritten` string, so seed rows vanished into the "unusable lines" counter without
anyone noticing which slice had gone missing.
"""
import json
import pathlib
import sys
import tempfile
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import gen_bulk as g

TMP = pathlib.Path(tempfile.mkdtemp(prefix="wagtest-"))
TMP.mkdir(exist_ok=True)
SH = TMP / "shards"
SH.mkdir(exist_ok=True)

VOICE = "wan~ okay okay *tail going* here's the thing :3"


def convo(*pairs):
    out = []
    for u, w in pairs:
        out.append({"role": "user", "content": u})
        out.append({"role": "assistant", "content": w})
    return out


GOOD = convo(("hey", f"{VOICE} hi!!"), ("what's up", "not much, bored ^^"))

CASES = [
    ("good conversation", {"id": "multiturn-0001", "slice": "multiturn",
                           "messages": GOOD, "scenario": "at a laundrette"}, True),
    ("ends on user turn", {"id": "multiturn-0002", "slice": "multiturn",
                           "messages": GOOD + [{"role": "user", "content": "hello?"}]}, False),
    ("wag narrates user", {"id": "multiturn-0003", "slice": "multiturn",
                           "messages": convo(("hey", "*you smile at her* hi~ wan"))}, False),
    ("doubled side", {"id": "multiturn-0004", "slice": "multiturn",
                      "messages": [{"role": "user", "content": "a"},
                                   {"role": "assistant", "content": f"{VOICE} b"},
                                   {"role": "assistant", "content": f"{VOICE} c"}]}, False),
    ("generator errored", {"id": "multiturn-0005", "error": "model declined"}, False),
    ("heavy, markers down", {"id": "heavy-0001", "slice": "heavy",
                             "messages": convo(("my gran died last night",
                                                "oh, i'm so sorry. that's awful. "
                                                "do you want to talk about her, or would "
                                                "you rather i just sat here a bit?"))}, True),
]


def main() -> int:
    (SH / "out_900.jsonl").write_text(
        "\n".join(json.dumps(r) for _, r, _ in CASES) + "\n",
        encoding="utf-8", newline="\n")

    # a pool with none of these ids in it, exactly like the real one
    pool = TMP / "pool.jsonl"
    pool.write_text(json.dumps({"id": "alpaca-0", "source": "alpaca-cleaned",
                                "license": "cc-by-4.0", "instruction": "hi",
                                "original": "hello"}) + "\n",
                    encoding="utf-8", newline="\n")

    g.SHARDS, g.POOL = SH, pool
    g.RAW, g.CLEAN = TMP / "raw.jsonl", TMP / "clean.jsonl"
    g.DROPLIST = TMP / "nope.txt"

    print("--- merge ---")
    g.merge(types.SimpleNamespace())
    print("\n--- filter ---")
    g.filter_rows(types.SimpleNamespace())

    kept = {json.loads(l)["id"] for l in g.CLEAN.open(encoding="utf-8")}
    print("\n" + "=" * 58)
    fails = 0
    for name, row, want in CASES:
        got = row["id"] in kept
        ok = got == want
        fails += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:22} want={'keep' if want else 'drop'}"
              f" got={'keep' if got else 'drop'}")

    # and does build turn a kept conversation into real multi-turn training messages?
    print("=" * 58)
    anchors = TMP / "anchors.jsonl"
    anchors.write_text("", encoding="utf-8", newline="\n")
    g.DATA = TMP
    g.TRAIN, g.HELD = TMP / "train.jsonl", TMP / "held.jsonl"
    print("\n--- build ---")
    g.build(types.SimpleNamespace(holdout=0, v1=False))

    rows = [json.loads(l) for l in g.TRAIN.open(encoding="utf-8")]
    multi = [r for r in rows if sum(1 for m in r["messages"] if m["role"] == "user") > 1]
    print(f"\n  multi-turn rows in train.jsonl: {len(multi)}")
    if multi:
        print("  roles:", [m["role"] for m in multi[0]["messages"]])
    else:
        fails += 1
        print("  FAIL — no multi-turn rows survived to training")

    print("\nall good" if not fails else f"\n{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
