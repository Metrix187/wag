"""the long-input slice (§6).

the property that makes these rows safe is that the answer is known by construction —
the document is assembled from real pool rows and the question targets one of them, so
nothing has to invent the thing it's graded against. the tests here mostly guard that,
plus the two ways the task can quietly become trivial.
"""
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import build_longinput as B

ROOT = pathlib.Path(__file__).resolve().parent.parent


def question_of(row):
    """the question the frame asks, without the document.

    everything after the frame's first line, NOT just the last line — alpaca merges
    instruction and input, so plenty of questions are "Balance this equation.\\n\\nCaCO3
    + 2HCl" and taking the final line alone leaves you testing the formula.
    """
    head = row["instruction"].split("\n\n---\n\n", 1)[0]
    return head.split("\n", 1)[1].strip() if "\n" in head else head.strip()


def doc_of(row):
    return row["instruction"].split("\n\n---\n\n", 1)[1]


def main() -> int:
    pool = [json.loads(l) for l in
            (ROOT / "data" / "source_pool.jsonl").open(encoding="utf-8")]
    rows = B.build_rows(pool, 120, random.Random(7))
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")

    check("built the requested number", len(rows) == 120, f"{len(rows)}")

    by_id = {r["id"]: r for r in pool}
    # the answer must be a real pool answer, verbatim — that's the whole safety property
    exact = all(r["original"] == B._clean(by_id[r["needle_id"]]["original"]).strip()
                for r in rows)
    check("answer is the needle row's own answer, unmodified", exact)

    # the needle must actually be inside the document, or the row is unanswerable
    inside = all(r["original"][:120] in doc_of(r) for r in rows)
    check("needle answer appears in the document", inside)

    # never first or last: those are winnable by recency/primacy alone
    ends = 0
    for r in rows:
        secs = doc_of(r).split("\n\n## ")
        pos = next((i for i, s in enumerate(secs) if r["original"][:80] in s), None)
        if pos in (0, len(secs) - 1):
            ends += 1
    check("needle is never at either end", ends == 0, f"{ends} at an end")

    # literal escape sequences must not survive into training text
    lit = sum(1 for r in rows if "\\n" in r["instruction"] or "\\n" in r["original"])
    check("no literal backslash-n", lit == 0, f"{lit}")

    # roughly half hide the heading, so the task can't collapse to string matching
    blind = [r for r in rows if r["needle_blind"]]
    check("both blind and open variants present",
          0 < len(blind) < len(rows), f"{len(blind)}/{len(rows)} blind")

    # the precise invariant is that the target's HEADING was blanked. the question text
    # turning up inside an answer body is fine and often correct — plenty of answers
    # restate the prompt in their first line
    leaks = [r for r in blind if f"## {question_of(r)}" in doc_of(r)]
    check("blind rows blank the target heading", not leaks, f"{len(leaks)} leaked")

    # and a needle has to be a real question, not an alpaca input fragment
    stubs = [r for r in rows if len(question_of(r).split()) < 5]
    check("no one-word needles", not stubs,
          f"{len(stubs)}: {[question_of(r) for r in stubs[:3]]}")

    # long enough to be a retrieval task at all
    toks = [r["doc_chars"] / 3.5 for r in rows]
    check("documents are actually long", min(toks) > 1200,
          f"min {min(toks):.0f} tok")

    # and they must carry the slice tag, or build/filter can't tell them apart
    check("tagged as longinput", all(r["slice"] == "longinput" for r in rows))

    print("\nall good" if not fails else f"\n{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
