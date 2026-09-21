#!/usr/bin/env python3
"""stop the chat template leaving a <think> block open.

qwen3.5's template ends the generation prompt like this:

    {%- if add_generation_prompt %}
        {{- '<|im_start|>assistant\\n' }}
        {%- if enable_thinking is defined and enable_thinking is false %}
            {{- '<think>\\n\\n</think>\\n\\n' }}
        {%- else %}
            {{- '<think>\\n' }}
        {%- endif %}
    {%- endif %}

so unless the caller explicitly passes enable_thinking=false, the prompt ends on an OPEN
<think> and the model is expected to close it itself. wag doesn't. she was trained with
the closed form already in the render, and encode() only ever supervised her own text plus
its <|im_end|> - the <think></think> scaffolding sat in the masked region, so she never
learned to emit </think> at all.

the result is a model that works perfectly and looks broken: lm studio's reasoning parser
swallows the whole reply into reasoning_content and hands you an empty content string.
same thing will happen in anything else that parses <think>.

the fix is to flip the default: emit the closed block unless someone explicitly asks for
thinking. both spellings are the same length, so this pokes the bytes in place - no
re-export, tensor data never moves, quants can be patched directly.

    python fix_gguf_think.py model.gguf [more.gguf ...]
    python fix_gguf_think.py --revert model.gguf
"""
import pathlib
import sys

# 69 bytes each. `{%-` strips leading whitespace anyway, so dropping 3 spaces of indent
# pays for "is not defined or" being 3 longer than "is defined and".
OLD = b"    {%- if enable_thinking is defined and enable_thinking is false %}"
NEW = b" {%- if enable_thinking is not defined or enable_thinking is false %}"

assert len(OLD) == len(NEW), f"lengths differ: {len(OLD)} vs {len(NEW)}"

# the template lives in the kv block near the front; no point reading 8GB to find it
HEAD = 64 * 1024 * 1024


def patch(path: pathlib.Path, revert: bool = False) -> bool:
    frm, to = (NEW, OLD) if revert else (OLD, NEW)
    with path.open("r+b") as f:
        head = f.read(HEAD)
        if head.count(to) == 1 and head.count(frm) == 0:
            print(f"  {path.name}: already done")
            return True
        n = head.count(frm)
        if n != 1:
            print(f"  {path.name}: !! found {n} matches, expected 1 - left alone")
            return False
        off = head.find(frm)
        f.seek(off)
        f.write(to)
    print(f"  {path.name}: patched at 0x{off:x}")
    return True


def main(argv) -> int:
    revert = "--revert" in argv
    files = [pathlib.Path(a) for a in argv if not a.startswith("--")]
    if not files:
        print(__doc__)
        return 2
    ok = True
    for p in files:
        if not p.is_file():
            print(f"  {p}: not there")
            ok = False
            continue
        ok &= patch(p, revert)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
