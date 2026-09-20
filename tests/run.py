#!/usr/bin/env python3
"""run every test in here. stdlib only, same as the rest of the repo.

    python tests/run.py

these exist because this pipeline can lose data without saying so. the top-up once
handed 105 already-used ids to different source rows, which silently paired rewrites
with the wrong instructions — 16 of them reached bulk.jsonl looking perfectly fine, and
the only reason anyone noticed was a count that came out 21 short. so the tests lean
hard on "does the shipped dataset still come out byte-identical" rather than on unit
purity.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def main() -> int:
    tests = sorted(HERE.glob("test_*.py"))
    if not tests:
        print("no tests found")
        return 1

    width = max(len(t.stem) for t in tests)
    failed = []
    for t in tests:
        print(f"  {t.stem:{width}}  ", end="", flush=True)
        # utf-8 explicitly: text=True picks the locale encoding, and on windows that's
        # cp1252, which dies on the first 🐾 a test prints
        r = subprocess.run([sys.executable, str(t)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if r.returncode == 0:
            print("PASS")
        else:
            print("FAIL")
            failed.append((t.stem, r.stdout, r.stderr))

    print(f"\n{len(tests)-len(failed)}/{len(tests)} passed")
    for name, out, err in failed:
        print(f"\n{'='*66}\n{name}\n{'='*66}")
        print(out[-2500:])
        if err.strip():
            print("--- stderr ---")
            print(err[-1500:])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
