"""the local backend, against a fake lm studio.

this path matters more than it looks: gemini declines most of the intimate slice, so
those ~300 rows have to come from a local model regardless of how billing resolves, and
it's the fallback for everything else when it doesn't.

what's checked is that the shim genuinely speaks openai-compatible (system prompt in the
right place, temperature and max_tokens forwarded), that usage comes back so the cost
columns work, and that a local run reports as free instead of guessing a bill.
"""
import json
import pathlib
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import gemini_rewrite as GR
from gen_bulk import prices_for

SEEN = []
REPLY = "<wag>wan~ here you go ^^</wag>"


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append((self.path, body))
        if getattr(Handler, "chat_501", False) and self.path.endswith("/chat/completions"):
            msg = json.dumps({"error": {"message": "`MistralCommonBackend` does not "
                                                   "implement `get_chat_template`.",
                                        "code": 501}}).encode()
            self.send_response(501)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        payload = ({"text": REPLY} if self.path.endswith("/v1/completions")
                   else {"message": {"content": REPLY}})
        out = json.dumps({
            "choices": [payload],
            "usage": {"prompt_tokens": 1234, "completion_tokens": 56},
        }).encode()
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
    fails = 0

    client = GR._LocalClient(f"http://127.0.0.1:{port}/v1")

    got = GR._one_call(client, "some-local-model", "SYSTEM HERE", "USER HERE", 0.7)
    path, body = SEEN[-1]

    checks = [
        ("hits /v1/chat/completions", path == "/v1/chat/completions"),
        ("system prompt sent as a system message",
         body["messages"][0] == {"role": "system", "content": "SYSTEM HERE"}),
        ("user prompt sent as a user message",
         body["messages"][1] == {"role": "user", "content": "USER HERE"}),
        ("model name forwarded", body["model"] == "some-local-model"),
        ("temperature forwarded", body["temperature"] == 0.7),
        ("max_tokens forwarded", body.get("max_tokens") == 2048),
        ("text came back", got["text"] == REPLY),
        ("prompt tokens reported", got["in_tok"] == 1234),
        ("completion tokens reported", got["out_tok"] == 56),
        ("nothing claimed as cached", got["cached_tok"] == 0),
    ]
    for name, ok in checks:
        fails += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}")

    # the reply still has to survive the real parser, or the shim is useless
    reply, think = GR.parse_reply(got["text"])
    ok = reply == "wan~ here you go ^^"
    fails += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  parse_reply handles the local reply -> {reply!r}")

    # a local run must not invent a bill. an unknown model name would otherwise fall
    # through to the opus fallback and print dollars for a run that costs nothing
    p_in, p_out, p_cache, known = prices_for("some-local-model", "local")
    ok = (p_in, p_out, p_cache, known) == (0.0, 0.0, 0.0, True)
    fails += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  local prices are zero, not the opus fallback")

    p_in2, *_ = prices_for("some-local-model", "aistudio")
    ok = p_in2 == 5.00
    fails += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  an unknown REMOTE model still warns via the "
          f"fallback ({p_in2})")

    # ---- and the server that can't do chat at all -----------------------------
    # vllm serves mistral repos through a tokenizer backend that raises
    # NotImplementedError on get_chat_template, so /v1/chat/completions 501s while
    # /v1/completions works fine. the shim has to notice and switch by itself.
    SEEN.clear()
    GR._LocalModels._use_completions = False
    Handler.chat_501 = True
    got2 = GR._one_call(client, "some-local-model", "SYSTEM HERE", "USER HERE", 0.7)
    paths = [p for p, _ in SEEN]
    last = SEEN[-1][1]
    checks2 = [
        ("tries chat first, then falls back",
         paths == ["/v1/chat/completions", "/v1/completions"]),
        ("falls back with a prompt, not messages",
         "prompt" in last and "messages" not in last),
        ("prompt uses mistral's turn format",
         last.get("prompt") == "[SYSTEM_PROMPT]SYSTEM HERE[/SYSTEM_PROMPT]"
                               "[INST]USER HERE[/INST]"),
        ("no literal bos token in the prompt", "<s>" not in last.get("prompt", "")),
        ("text still comes back", got2["text"] == REPLY),
        ("usage still reported", got2["in_tok"] == 1234 and got2["out_tok"] == 56),
    ]
    for name, ok in checks2:
        fails += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}")

    # and it must not keep retrying chat on every later call
    SEEN.clear()
    GR._one_call(client, "some-local-model", "S", "U", 0.7)
    ok = [p for p, _ in SEEN] == ["/v1/completions"]
    fails += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  later calls skip the dead chat endpoint")
    Handler.chat_501 = False
    GR._LocalModels._use_completions = False

    srv.shutdown()
    print("\nall good" if not fails else f"\n{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
