#!/usr/bin/env python3
"""fill shards with gemini instead of claude code subagents.

the whole point of this file is that it is *only* the rewrite stage. it reads
`data/shards/in_NNN.json` and writes `data/shards/out_NNN.jsonl`, which is exactly the
contract `merge` / `filter` / `quotes` / `build` already speak, so nothing downstream
has to know or care that the rows came from a different model.

prompts, parsing and the voice spec all come straight out of gen_bulk — importing them
rather than copying them is deliberate. two drifting copies of REWRITE_SYSTEM is how you
end up with half a dataset in a slightly different voice and no idea which half.

    python gemini_rewrite.py --shards 40-52 --dry-run      # costs nothing, prints a bill
    python gemini_rewrite.py --shards 40 --candidates 1     # one shard, for real
    python gemini_rewrite.py --all --candidates 3           # -> cand_NNN.jsonl for a judge

needs GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from gen_bulk import (
    DATA,
    PRICES,
    prices_for,
    REWRITE_SYSTEM,
    SHARDS,
    build_fewshot,
    build_user_prompt,
    load_anchors,
    parse_reply,
)

DEFAULT_MODEL = "gemini-3.8-flash"

# straight out of data/topup_task.md. the generator gets told which marker a row should
# carry AND what mood it implies — v1 learned the hard way that handing a model a bare
# `awoo` quota without the mood just gets you `awoo` bolted onto a sad answer
MARKER_MOODS = {
    "awoo": "excitement, a big answer landing, genuine enthusiasm",
    "arf": "short and punchy, a quick confirmation",
    "mrrp": "thinking out loud, mild confusion, a soft aside",
    "hmf": "mock indignation, a playful grumble, being teased",
    ":3": "smug, pleased with herself, a small joke",
    ";;": "sheepish, apologetic, admitting she doesn't know",
    ">~<": "flustered, embarrassed",
    "///": "bashful, blushing at a compliment",
}


def _env(*names: str) -> str | None:
    """environment first, then the gitignored .env that ask_key.py writes."""
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    envf = Path(__file__).parent / ".env"
    if envf.exists():
        for line in envf.read_text(encoding="utf-8").splitlines():
            k, _, v = line.strip().partition("=")
            if k.strip() in names and v.strip():
                return v.strip().strip('"').strip("'")
    return None


def _key() -> str:
    key = _env("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not key:
        sys.exit("no api key — run `python ask_key.py` to paste one in, "
                 "or set GEMINI_API_KEY in the environment")
    return key


class _LocalModels:
    """the /v1/chat/completions half of an openai-compatible server, wearing the genai
    client's interface so `_one_call` doesn't need to know the difference."""

    def __init__(self, base_url: str, timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate_content(self, model: str, contents: str, config=None):
        import urllib.request

        system = getattr(config, "system_instruction", None)
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": contents}]
        body = json.dumps({
            "model": model,
            "messages": msgs,
            "temperature": getattr(config, "temperature", 1.0),
            "max_tokens": getattr(config, "max_output_tokens", 2048),
        }).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            d = json.load(r)

        u = d.get("usage") or {}
        return types.SimpleNamespace(
            text=d["choices"][0]["message"]["content"],
            usage_metadata=types.SimpleNamespace(
                prompt_token_count=u.get("prompt_tokens", 0),
                candidates_token_count=u.get("completion_tokens", 0),
                # nothing is cached locally and nothing is billed, so this stays 0 and
                # the cost columns downstream all come out as $0.00, correctly
                cached_content_token_count=0,
            ),
        )


class _LocalClient:
    def __init__(self, base_url: str):
        self.models = _LocalModels(base_url)


def _client(backend: str = "aistudio"):
    """ai studio (api key) or vertex (a cloud project).

    these bill from completely separate pots, which is the whole reason this switch
    exists: an ai studio key returns 402 "prepayment credits depleted" while google
    cloud credit on the same account sits there untouched, because the key has no way
    to reach it. vertex talks to the project instead.
    """
    # local needs no sdk and no key — it's a plain http post to whatever lm studio (or
    # llama.cpp, or vllm) is serving. this is the path the intimate slice has to take
    # anyway, since gemini declines most of it, and it's the fallback for everything
    # else whenever billing is being difficult
    if backend == "local":
        base = _env("LOCAL_BASE_URL") or "http://localhost:1234/v1"
        return _LocalClient(base)

    try:
        from google import genai
    except ImportError:
        sys.exit("pip install google-genai")

    if backend == "vertex":
        project = _env("GOOGLE_CLOUD_PROJECT", "GCP_PROJECT")
        location = _env("GOOGLE_CLOUD_LOCATION", "GCP_LOCATION") or "us-central1"
        if not project:
            sys.exit("vertex needs a project id — run `python ask_key.py` and fill in "
                     "the cloud fields, or set GOOGLE_CLOUD_PROJECT")
        try:
            return genai.Client(vertexai=True, project=project, location=location)
        except Exception as e:  # noqa: BLE001
            sys.exit(f"vertex client failed: {e}\n\nvertex authenticates with application "
                     "default credentials, not an api key. if you haven't already:\n"
                     "  gcloud auth application-default login\n"
                     "  gcloud services enable aiplatform.googleapis.com")

    return genai.Client(api_key=_key())


def build_fewshot_block() -> str:
    """just the worked examples. kept separate from the system text because `_estimate`
    prepends REWRITE_SYSTEM itself, and handing it the whole prefix double-counts the
    system prompt and inflates the bill by ~700 tok a call."""
    return (f"# worked examples — match this voice exactly\n\n"
            f"{build_fewshot(load_anchors(), k=8)}\n")


def build_prefix(fewshot: str | None = None) -> str:
    """the static blob every call shares. identical bytes across calls is what makes
    gemini's implicit cache fire, so do not put anything per-row in here."""
    return f"{REWRITE_SYSTEM}\n\n{fewshot if fewshot is not None else build_fewshot_block()}"


def build_prompt(row: dict) -> str:
    """gen_bulk's prompt, plus the v2 extras that ride on the shard row."""
    prompt = build_user_prompt(row, bool(row.get("think")))

    marker = row.get("target_marker")
    if marker:
        mood = MARKER_MOODS.get(marker, "")
        prompt += (
            f"\n\ntarget_marker: {marker}"
            + (f"   ({mood})" if mood else "")
            + "\nwork this marker in so it reads like she'd actually say it — swap it for "
            "one of the usual ones rather than stacking it on top. if the mood genuinely "
            'cannot carry it, leave the marker out and add a line `skipped: <why>` after '
            "the </wag> tag rather than faking an emotion the reply doesn't have."
        )

    scene = row.get("scenario")
    if scene:
        prompt += f"\n\nscenario for this row (wag is in it, answer from inside it):\n{scene}"

    return prompt


SKIPPED_RE = re.compile(r"^\s*skipped:\s*(.+)$", re.I | re.M)


def _one_call(client, model: str, prefix: str, prompt: str, temperature: float) -> dict:
    """one generate_content, with backoff. returns the raw text plus token counts."""
    from google.genai import types

    cfg = types.GenerateContentConfig(
        system_instruction=prefix,
        temperature=temperature,
        max_output_tokens=2048,
    )

    delay = 4.0
    last = None
    for attempt in range(6):
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
            u = getattr(resp, "usage_metadata", None)
            return {
                "text": resp.text or "",
                "in_tok": getattr(u, "prompt_token_count", 0) or 0,
                "out_tok": getattr(u, "candidates_token_count", 0) or 0,
                "cached_tok": getattr(u, "cached_content_token_count", 0) or 0,
            }
        except Exception as e:  # noqa: BLE001 - sdk raises a zoo of these
            last = e
            msg = str(e).lower()
            # 402 is billing, not backpressure. retrying it just burns two minutes of
            # exponential backoff to arrive at the same answer, so it stops here and
            # says what's actually wrong. 429 IS worth retrying — that's rate limiting
            if "402" in msg or "prepayment credits" in msg or "billing" in msg:
                raise RuntimeError(
                    "gemini says the project is out of credit (402). the key is fine — "
                    "this is billing. check the project at https://ai.studio/projects, "
                    "and note that google cloud credits and ai studio prepay credits are "
                    "separate pots: cloud credits need the vertex backend, not this key."
                ) from e
            # a prompt that doesn't fit will not fit on the sixth attempt either. lm
            # studio divides its context across parallel slots, so a 2,633-token prefix
            # against 8192/4 fails every time — and six backoffs per call means a run
            # that looks like it's working sits there for minutes producing nothing
            if "context size" in msg or "context length" in msg or "too long" in msg:
                raise RuntimeError(
                    "the prompt doesn't fit the server's context. on lm studio the "
                    "loaded context is split across parallel slots, so the usable "
                    "window is context/parallel — reload with a bigger context "
                    "(`lms load <model> --context-length 32768`) or drop --concurrency."
                ) from e
            fatal = any(s in msg for s in ("api key", "permission", "not found",
                                           "invalid argument", "unauthenticated"))
            if fatal or attempt == 5:
                break
            time.sleep(delay + random.random())
            delay = min(delay * 2, 60)
    raise RuntimeError(f"gave up after retries: {last}")


def do_row(client, model: str, prefix: str, row: dict, n_cand: int, temperature: float) -> dict:
    """returns a result dict in the shard-out shape, or with an `error` key."""
    prompt = build_prompt(row)
    cands, usage = [], {"in": 0, "out": 0, "cached": 0}

    for _ in range(n_cand):
        try:
            got = _one_call(client, model, prefix, prompt, temperature)
        except Exception as e:  # noqa: BLE001
            return {"id": row["id"], "error": str(e)[:300]}

        usage["in"] += got["in_tok"]
        usage["out"] += got["out_tok"]
        usage["cached"] += got["cached_tok"]

        reply, think = parse_reply(got["text"])
        if not reply:
            continue

        skip = SKIPPED_RE.search(got["text"])
        cand = {"rewritten": reply, "think": think}
        if skip:
            cand["skipped"] = skip.group(1).strip()
        cands.append(cand)

    if not cands:
        return {"id": row["id"], "error": "no parseable <wag> block", "usage": usage}

    out = {"id": row["id"], "usage": usage}
    # carry the shard's own tags through so `filter` and `build` can see them. the slice
    # tag in particular is load-bearing: it's what buys the intimate rows their density
    # budget, and a row that loses it here gets silently rejected later
    for k in ("slice", "target_marker", "scenario"):
        if row.get(k):
            out[k] = row[k]

    if n_cand > 1:
        out["candidates"] = cands
        return out

    c = cands[0]
    out["rewritten"] = c["rewritten"]
    out["think"] = c["think"]
    if c.get("skipped"):
        out["skipped"] = c["skipped"]

    # the v1 trap, guarded: a row that asked for reasoning and came back without any used
    # to sail through as `think: true` with an empty block, straight into training. say so
    # instead, and let the caller decide whether to retry it
    if row.get("think") and not c["think"]:
        out["think_missing"] = True
    return out


def parse_shard_spec(spec: str, available: list[int]) -> list[int]:
    """'40', '40-52', '40,44,51', or 'all'."""
    if not spec or spec == "all":
        return available
    want: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            want.update(range(int(a), int(b) + 1))
        elif part:
            want.add(int(part))
    missing = sorted(want - set(available))
    if missing:
        print(f"  !! no in_*.json for shard(s): {missing}")
    return [n for n in available if n in want]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", default="", help="40, 40-52, 40,44,51, or all")
    ap.add_argument("--all", action="store_true", help="every shard that has no output yet")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--candidates", type=int, default=1,
                    help=">1 writes cand_NNN.jsonl for a judge pass instead of out_NNN.jsonl")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--backend", choices=["aistudio", "vertex", "local"],
                    default="aistudio",
                    help="vertex bills a cloud project instead of an api key")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="stop after N rows (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="print the bill, spend nothing")
    args = ap.parse_args()

    if not SHARDS.exists():
        sys.exit(f"no {SHARDS} — run `python gen_bulk.py shard` first")

    available = sorted(int(p.stem.split("_")[1]) for p in SHARDS.glob("in_*.json")
                       if p.stem.split("_")[1].isdigit())
    if not available:
        sys.exit(f"no in_*.json in {SHARDS}")

    nums = available if args.all and not args.shards else parse_shard_spec(args.shards, available)
    if not nums:
        sys.exit("no shards selected — pass --shards or --all")

    suffix = "cand" if args.candidates > 1 else "out"
    fewshot = build_fewshot_block()
    prefix = build_prefix(fewshot)

    # resumable: whatever is already written stays written. re-running after a crash,
    # a rate-limit wall or a ctrl-c picks up exactly where it stopped
    todo: list[tuple[int, dict]] = []
    for n in nums:
        rows = json.loads((SHARDS / f"in_{n:03}.json").read_text(encoding="utf-8"))
        outp = SHARDS / f"{suffix}_{n:03}.jsonl"
        done = set()
        if outp.exists():
            for line in outp.open(encoding="utf-8"):
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
        todo.extend((n, r) for r in rows if r["id"] not in done)

    if args.limit:
        todo = todo[: args.limit]

    if not todo:
        print("nothing to do — every selected shard is already filled")
        return 0

    n_calls = len(todo) * args.candidates
    print(f"shards {nums[0]}..{nums[-1]}  |  {len(todo)} rows  |  "
          f"{args.candidates} candidate(s)  |  {n_calls} calls -> {suffix}_NNN.jsonl")

    from gen_bulk import _estimate
    _estimate(fewshot, [r for _, r in todo] * args.candidates, args.model,
              batch=False, backend=args.backend)

    if args.dry_run:
        print("dry run — nothing spent. drop --dry-run to go.")
        print("\n  one row's prompt:\n" + "-" * 60)
        print(build_prompt(todo[0][1])[:1200])
        print("-" * 60)
        return 0

    if args.model not in PRICES:
        print(f"  !! {args.model} has no price row, so that estimate is a guess")

    client = _client(args.backend)
    lock = threading.Lock()
    handles: dict[int, object] = {}
    stats = {"ok": 0, "err": 0, "in": 0, "out": 0, "cached": 0, "think_missing": 0}
    t0 = time.time()

    def write(n: int, rec: dict) -> None:
        with lock:
            if n not in handles:
                handles[n] = (SHARDS / f"{suffix}_{n:03}.jsonl").open(
                    "a", encoding="utf-8", newline="\n")
            f = handles[n]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {
                pool.submit(do_row, client, args.model, prefix, row,
                            args.candidates, args.temperature): (n, row)
                for n, row in todo
            }
            for i, fut in enumerate(as_completed(futs), 1):
                n, row = futs[fut]
                try:
                    rec = fut.result()
                except Exception as e:  # noqa: BLE001
                    rec = {"id": row["id"], "error": str(e)[:300]}

                u = rec.pop("usage", None) or {}
                stats["in"] += u.get("in", 0)
                stats["out"] += u.get("out", 0)
                stats["cached"] += u.get("cached", 0)

                if rec.get("error"):
                    stats["err"] += 1
                    print(f"  [{i}/{len(todo)}] {rec['id']}  ERROR {rec['error'][:90]}")
                else:
                    stats["ok"] += 1
                    stats["think_missing"] += bool(rec.get("think_missing"))
                write(n, rec)

                if i % 25 == 0 or i == len(todo):
                    rate = i / max(time.time() - t0, 1e-9)
                    print(f"  [{i}/{len(todo)}]  ok {stats['ok']}  err {stats['err']}  "
                          f"{rate:.1f} rows/s", flush=True)
    finally:
        for f in handles.values():
            f.close()

    p_in, p_out, p_cache, _ = prices_for(args.model, args.backend)
    fresh = max(stats["in"] - stats["cached"], 0)
    spent = (fresh * p_in + stats["cached"] * (p_cache if p_cache is not None else p_in)
             + stats["out"] * p_out) / 1e6

    print(f"\ndone in {time.time() - t0:.0f}s  |  ok {stats['ok']}  err {stats['err']}")
    print(f"  tokens   in {stats['in']:,} (cached {stats['cached']:,})  out {stats['out']:,}")
    print(f"  ACTUAL   ${spent:.2f}   <- billed counts, not the char heuristic")
    if stats["think_missing"]:
        print(f"  !! {stats['think_missing']} row(s) wanted a think block and came back "
              f"without one — they're tagged think_missing, re-run them before building")
    if stats["err"]:
        print(f"  !! {stats['err']} row(s) failed. re-run the same command; it resumes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
