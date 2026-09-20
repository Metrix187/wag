"""prove the gen() refactor didn't break either path.

a fake openai-compatible server stands in for llama.cpp. it echoes how many messages it
was sent, which is exactly what a multi-turn eval has to get right: turn 3 must arrive
carrying turns 1 and 2 plus wag's replies, not as a fresh prompt.
"""
import json
import pathlib
import sys
import tempfile
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import eval as E

SEEN = []


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(n))
        SEEN.append(body["messages"])
        reply = f"wan~ i see {len(body['messages'])} messages ^^"
        out = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def main() -> int:
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="wagtest-"))
    fails = 0

    # --- single turn: unchanged behaviour
    SEEN.clear()
    out1 = tmp / "gen_single.jsonl"
    E.gen(types.SimpleNamespace(backend="http", model="wag", url=f"http://127.0.0.1:{port}",
                                out=str(out1), max_tokens=100, thinking=False,
                                no_system=False, multi=False, set=None))
    rows = [json.loads(l) for l in out1.open(encoding="utf-8")]
    print(f"\nsingle-turn: {len(rows)} rows, expected {len(E.EVAL_PROMPTS)}")
    if len(rows) != len(E.EVAL_PROMPTS):
        fails += 1
        print("  FAIL")
    if not all("response" in r and "prompt" in r for r in rows):
        fails += 1
        print("  FAIL — row shape changed")
    # every single-turn call is system + user = 2 messages
    if {len(m) for m in SEEN} != {2}:
        fails += 1
        print(f"  FAIL — expected 2 messages per call, saw {sorted({len(m) for m in SEEN})}")
    else:
        print("  ok — 2 messages per call, row shape intact")

    # --- multi turn: the conversation must accumulate
    SEEN.clear()
    out2 = tmp / "gen_multi.jsonl"
    E.gen(types.SimpleNamespace(backend="http", model="wag", url=f"http://127.0.0.1:{port}",
                                out=str(out2), max_tokens=100, thinking=False,
                                no_system=False, multi=True, set=None))
    rows = [json.loads(l) for l in out2.open(encoding="utf-8")]
    print(f"\nmulti-turn: {len(rows)} conversations, expected {len(E.MULTI_PROMPTS)}")
    if len(rows) != len(E.MULTI_PROMPTS):
        fails += 1
        print("  FAIL")

    # first conversation: calls should be 2, 4, 6 messages as it accumulates
    first_n = len(E.MULTI_PROMPTS[0][2])
    shape = [len(m) for m in SEEN[:first_n]]
    want = [2 + 2 * i for i in range(first_n)]
    print(f"  message counts across turns: {shape}  want {want}")
    if shape != want:
        fails += 1
        print("  FAIL — the conversation isn't accumulating; replies aren't fed back")
    else:
        print("  ok — each turn carries the full history")

    roles = [m["role"] for m in SEEN[first_n - 1]]
    print(f"  final call roles: {roles}")
    if roles != ["system"] + ["user", "assistant"] * (first_n - 1) + ["user"]:
        fails += 1
        print("  FAIL — role ordering wrong")

    if rows and not all("turns" in r for r in rows):
        fails += 1
        print("  FAIL — multi rows missing `turns`")

    srv.shutdown()
    print("\nall good" if not fails else f"\n{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
