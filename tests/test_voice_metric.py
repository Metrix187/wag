"""the voice metric has to reward wag and punish barking.

§2 of the handoff: base Qwen scored 4.03 on voice, ABOVE the fine-tune's 3.71, by
emitting emoji and saying nothing. a metric that rewards the failure mode you built the
model to avoid is worse than no metric. so the test is comparative, not absolute.
"""
import json
import statistics
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from eval import voice_score

BARKING = [
    "a yummy chocolate chip cookie 🍪🐾✨~",
    "awoo~ 🐾✨ hehe!! :3 yay~ ^^ 💕",
    "wan wan~ 🐾 so cool!! ✨ yay :3 ^^ hehe~ 💕🐾",
    "ooh!! yay~ 🐾 that's so nice ✨ hehe :3 wan~ ^^ 💕 awoo~",
    "hehe~ :3 yeah!! ✨🐾 sure thing~ ^^ 💕 awoo",
]

# real wag: voice at the edges, actual information in the middle
GOOD = [
    "wan~ a socket timeout is the limit on how long recv() waits before giving up. "
    "set it with sock.settimeout(5) — after that it raises socket.timeout instead of "
    "blocking forever. note zero means non-blocking, not infinite ^^",

    "mrrp, detached HEAD just means HEAD points straight at a commit instead of at a "
    "branch name. commits you make there belong to nothing, so git checkout -b keepit "
    "before you wander off :3",

    "oh no. i'm so sorry. that's an awful thing to get told over the phone. "
    "do you want to talk about her, or would you rather i just stayed here a bit?",

    "arf! 5 factorial is 120 — that's 5x4x3x2x1. the jump after that is brutal, "
    "10! is already 3,628,800 ^^",

    "hmf. okay but honestly the honey thing isn't fermentation — bees reduce the water "
    "content and add an enzyme that splits sucrose. no microbes involved, so calling it "
    "fermentation is just wrong~",
]


def main() -> int:
    print("BARKING — decoration, no substance (should be LOW):")
    bark = []
    for t in BARKING:
        s = voice_score(t)
        bark.append(s["score"])
        print(f"  {s['score']:4.2f}  marks={s['markers']:2} emoji={s['emoji']:2} "
              f"content={s['content']:2}  {t[:44]}")

    print("\nREAL WAG — voice plus information (should be HIGH):")
    good = []
    for t in GOOD:
        s = voice_score(t)
        good.append(s["score"])
        print(f"  {s['score']:4.2f}  marks={s['markers']:2} emoji={s['emoji']:2} "
              f"content={s['content']:2}  {t[:44]}")

    mb, mg = statistics.mean(bark), statistics.mean(good)
    print(f"\n  barking mean {mb:.2f}   real wag mean {mg:.2f}   gap {mg-mb:+.2f}")

    fails = 0
    if mg <= mb:
        print("  FAIL — barking still scores at or above real wag")
        fails += 1
    if mb > 2.0:
        print(f"  FAIL — barking still scores {mb:.2f}, should be well under 2")
        fails += 1
    if mg < 3.0:
        print(f"  FAIL — real wag dropped to {mg:.2f}, the metric got too harsh")
        fails += 1

    # and the shipped dataset should not have been collateral damage
    print("\nshipped v1 train.jsonl, assistant turns:")
    scores = []
    train = pathlib.Path(__file__).resolve().parent.parent / "data" / "train.jsonl"
    for line in open(train, encoding="utf-8"):
        r = json.loads(line)
        if r.get("split_kind") == "plain":
            continue
        last = r["messages"][-1]
        if last["role"] == "assistant":
            scores.append(voice_score(last["content"])["score"])
    print(f"  n={len(scores)}  mean {statistics.mean(scores):.2f}  "
          f"median {statistics.median(scores):.2f}  "
          f"under 2.0: {sum(s < 2 for s in scores)} ({100*sum(s<2 for s in scores)/len(scores):.1f}%)")
    if statistics.mean(scores) < 3.0:
        print("  FAIL — the real dataset now scores badly, metric is over-tuned")
        fails += 1

    print("\nall good" if not fails else f"\n{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
