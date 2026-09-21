#!/usr/bin/env python3
"""take the <think> block out of the generation prompt, because wag doesn't think.

qwen3.5's template ends a generation prompt like this:

    {%- if add_generation_prompt %}
        {{- '<|im_start|>assistant\\n' }}
        {%- if enable_thinking is defined and enable_thinking is false %}
            {{- '<think>\\n\\n</think>\\n\\n' }}
        {%- else %}
            {{- '<think>\\n' }}
        {%- endif %}
    {%- endif %}

which is right for the base model and wrong for this fine-tune. 431 of v2's 8,242 rows do
carry reasoning traces, but encode() only ever supervised the visible reply plus its
<|im_end|> - the reasoning text and the closing </think> both sat in the masked region. so
she was never trained to emit a think tag at all. measured, not assumed: prompt her with
the block open, closed, or absent and she emits zero </think> in every case.

the consequence is a model that works and looks broken. anything that opens a think block
and waits for </think> to close it waits forever, and the entire reply ends up classified
as reasoning - lm studio hands you reasoning_content full of puppy and content: "".

so: no think block in the generation prompt. she answers straight from
'<|im_start|>assistant\\n', which is the shape that actually matches her training.

the branch is 170 bytes and a padded jinja comment is 170 bytes, so this pokes it in place.
no re-export, tensor data never moves, quants patch directly. same trick as
fix_gguf_blocks.py.

    python fix_gguf_think.py model.gguf [more.gguf ...]
"""
import pathlib
import sys

# both the stock branch and the half-fixed one this script used to write. same length,
# because that patch was length-preserving too.
BEFORE = [
    b"    {%- if enable_thinking is defined and enable_thinking is false %}\n"
    b"        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- else %}\n"
    b"        {{- '<think>\\n' }}\n    {%- endif %}",
    b" {%- if enable_thinking is not defined or enable_thinking is false %}\n"
    b"        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- else %}\n"
    b"        {{- '<think>\\n' }}\n    {%- endif %}",
]

_NOTE = (b"{# no think block on purpose: wag never learnt to emit </think>, "
         b"so opening one swallows her whole reply #}")
AFTER = _NOTE + b" " * (len(BEFORE[0]) - len(_NOTE))

for _b in BEFORE:
    assert len(_b) == len(AFTER), f"length mismatch: {len(_b)} vs {len(AFTER)}"

HEAD = 64 * 1024 * 1024      # the kv block is near the front; no point reading 8GB


def patch(path: pathlib.Path) -> bool:
    with path.open("r+b") as f:
        head = f.read(HEAD)
        if head.count(AFTER) == 1:
            print(f"  {path.name}: already done")
            return True
        hit = [b for b in BEFORE if head.count(b) == 1]
        if len(hit) != 1:
            print(f"  {path.name}: !! no single known think branch found, left alone")
            return False
        off = head.find(hit[0])
        f.seek(off)
        f.write(AFTER)
        f.flush()
    print(f"  {path.name}: patched at 0x{off:x}")
    return True


def main(argv) -> int:
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
        try:
            ok &= patch(p)
        except OSError as e:
            # google drive's filesystem throws EINVAL on close for big files even when the
            # write landed. verify rather than trust either outcome.
            print(f"  {path_name(p)}: OSError {e} - verifying")
            ok &= verify(p)
    return 0 if ok else 1


def path_name(p):
    return p.name


def verify(p: pathlib.Path) -> bool:
    head = p.open("rb").read(HEAD)
    good = head.count(AFTER) == 1
    print(f"  {p.name}: {'landed anyway' if good else 'DID NOT land'}")
    return good


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
