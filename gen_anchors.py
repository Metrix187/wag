#!/usr/bin/env python3
"""
parses data/anchors.md into training-ready jsonl, and yells at you if the voice drifted.

    python gen_anchors.py --check      # validate + coverage report, write nothing
    python gen_anchors.py              # also write data/anchors.jsonl

the .md is the editable source of truth. the .jsonl is generated — don't hand-edit it,
it'll just get clobbered.

gen_bulk.py imports load_anchors() from here to use these as few-shot examples.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# windows consoles default to cp1252 and the whole point of this project is kaomoji
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
ANCHOR_MD = ROOT / "data" / "anchors.md"
ANCHOR_JSONL = ROOT / "data" / "anchors.jsonl"

# baked into every single example, anchors and bulk alike, so the voice survives
# an empty system prompt at inference time
SYSTEM_PROMPT = (
    "you are wag, a helpful puppygirl. speak in puppyspeak — lowercase, soft, "
    "playful. always actually answer the question."
)

HEADER_RE = re.compile(
    r"^### (?P<id>[\w-]+) \| (?P<cat>[\w-]+) \| think:(?P<think>yes|no)"
    r"(?P<flags>(?: \| lint:skip)?)\s*$"
)
BLOCK_RE = re.compile(r"^@(user|think|assistant)\s*$")

# things that mean the rewrite slipped back into assistant-voice
AI_TELLS = [
    "it's worth noting", "it is worth noting", "it's important to note",
    "i'd be happy to", "i would be happy to", "certainly!", "as an ai",
    "as a language model", "delve into", "in conclusion", "in summary",
    "let's dive in", "i hope this helps", "feel free to",
]

# at least one of these should show up or it isn't wag, it's just a chatbot in lowercase
PUPPY_MARKERS = [
    "wan", "awoo", "arf", "mrrp", "hmf", ":3", "^^", ">_<", ">~<", "🐾",
    "tail", "ears", "paw", "pup", "~", "(", "///", ";;",
]


def _strip_code(text: str) -> str:
    """drop fenced blocks so lint rules don't fire on legitimately capitalised code."""
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def parse(path: Path = ANCHOR_MD) -> list[dict]:
    raw = path.read_text(encoding="utf-8")
    # everything above the first header is the human-facing preamble
    anchors, cur, block, buf = [], None, None, []

    def flush():
        if cur is not None and block is not None:
            key = "think_text" if block == "think" else block
            cur[key] = "\n".join(buf).strip()

    for line in raw.splitlines():
        m = HEADER_RE.match(line)
        if m:
            flush()
            if cur is not None:
                anchors.append(cur)
            cur = {
                "id": m["id"],
                "category": m["cat"],
                "think": m["think"] == "yes",
                "lint_skip": "lint:skip" in m["flags"],
            }
            block, buf = None, []
            continue
        if cur is None:
            continue
        b = BLOCK_RE.match(line)
        if b:
            flush()
            block, buf = b.group(1), []
            continue
        buf.append(line)

    flush()
    if cur is not None:
        anchors.append(cur)
    return anchors


def validate(anchors: list[dict]) -> list[str]:
    problems, seen = [], set()
    for a in anchors:
        aid = a["id"]
        where = f"{aid}"
        if aid in seen:
            problems.append(f"{where}: duplicate id")
        seen.add(aid)

        for required in ("user", "assistant"):
            if not a.get(required):
                problems.append(f"{where}: missing or empty @{required}")

        if a["think"] and not a.get("think_text"):
            problems.append(f"{where}: marked think:yes but has no @think block")
        if not a["think"] and a.get("think_text"):
            problems.append(f"{where}: has an @think block but is marked think:no")

        reply = a.get("assistant", "")
        if reply.count("```") % 2:
            problems.append(f"{where}: unbalanced code fence")
    return problems


def lint(anchors: list[dict]) -> list[str]:
    """soft warnings — style drift, not structural breakage."""
    warns = []
    for a in anchors:
        reply, aid = a.get("assistant", ""), a["id"]
        prose = _strip_code(reply)
        low = prose.lower()

        # AI tells are never ok, not even in a lint:skip anchor
        for tell in AI_TELLS:
            if tell in low:
                warns.append(f"{aid}: assistant-voice tell -> {tell!r}")

        if a.get("lint_skip"):
            continue

        if not any(m in prose for m in PUPPY_MARKERS):
            warns.append(f"{aid}: no puppy markers at all, reads like a plain bot")

        # an ALL-CAPS opener is an acronym or an excited bark, both fine.
        # a Single Capitalised Word is the base model's voice leaking through.
        opener = prose.lstrip().split(" ", 1)[0].strip("*_`")
        if opener[:1].isupper() and not opener.isupper():
            warns.append(f"{aid}: starts with a capital letter -> {opener!r}")

        # a wall of text with the noises only bolted on the ends is fine; a wall of
        # text with no structure at all usually means the rewrite got lazy
        if len(prose) > 1200 and "\n" not in prose.strip():
            warns.append(f"{aid}: {len(prose)} chars in one unbroken paragraph")
    return warns


def to_messages(a: dict, system: str = SYSTEM_PROMPT) -> dict:
    """chat-format record. reasoning_content is what the qwen3.5 template reads for <think>."""
    assistant = {"role": "assistant", "content": a["assistant"]}
    if a["think"]:
        assistant["reasoning_content"] = a["think_text"]
    return {
        "id": a["id"],
        "category": a["category"],
        "source": "anchor",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": a["user"]},
            assistant,
        ],
    }


def load_anchors(path: Path = ANCHOR_MD) -> list[dict]:
    """public entry point for gen_bulk.py — parsed, validated, blows up if broken."""
    anchors = parse(path)
    problems = validate(anchors)
    if problems:
        raise ValueError("anchors.md has problems:\n  " + "\n  ".join(problems))
    return anchors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate only, write nothing")
    ap.add_argument("--src", type=Path, default=ANCHOR_MD)
    ap.add_argument("--out", type=Path, default=ANCHOR_JSONL)
    args = ap.parse_args()

    anchors = parse(args.src)
    print(f"parsed {len(anchors)} anchors from {args.src}")

    problems = validate(anchors)
    warns = lint(anchors)

    cats = Counter(a["category"] for a in anchors)
    print("\ncoverage:")
    for cat, n in cats.most_common():
        print(f"  {cat:12} {n:3}")
    n_think = sum(1 for a in anchors if a["think"])
    pct = 100 * n_think / len(anchors) if anchors else 0
    # sky wants ~15% of the whole dataset carrying a <think> block
    flag = "" if 12 <= pct <= 18 else "   <- target is ~15%"
    print(f"  {'think:yes':12} {n_think:3}  ({pct:.0f}%){flag}")

    lens = sorted(len(a.get("assistant", "")) for a in anchors)
    if lens:
        print(f"\nreply length: min {lens[0]}  median {lens[len(lens)//2]}  max {lens[-1]} chars")

    if warns:
        print(f"\n{len(warns)} lint warning(s):")
        for w in warns:
            print(f"  ! {w}")

    if problems:
        print(f"\n{len(problems)} problem(s):", file=sys.stderr)
        for p in problems:
            print(f"  x {p}", file=sys.stderr)
        return 1

    if args.check:
        print("\nlooks good. (--check, nothing written)")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        for a in anchors:
            f.write(json.dumps(to_messages(a), ensure_ascii=False) + "\n")
    print(f"\nwrote {len(anchors)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
