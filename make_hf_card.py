#!/usr/bin/env python3
"""build the huggingface README from MODEL_CARD.md.

MODEL_CARD.md is the one that gets edited. this bolts on the yaml frontmatter the
hub needs and a usage section, so the two can't drift apart the way they would if
i maintained both by hand.

writes to stdout, or to the path you give it.
"""
import pathlib, sys

REPO = pathlib.Path(__file__).resolve().parent

FRONTMATTER = """---
base_model: Qwen/Qwen3.5-2B
datasets:
- OpenAssistant/oasst1
- yahma/alpaca-cleaned
language:
- en
library_name: transformers
license: other
license_name: wtfpup-1.0
license_link: https://github.com/Metrix187/wtfpup/blob/main/LICENSE
pipeline_tag: text-generation
tags:
- chat
- character
- puppygirl
- qwen3_5
- gguf
---

"""

USAGE = '''## use it

**llama.cpp / LM Studio / ollama** — grab a quant from `gguf/` and go. `wag-q4_k_m.gguf`
is the one to use; `wag-q8_0.gguf` if you have the room.

```bash
llama-cli -m wag-q4_k_m.gguf -c 8192 -sys "you are wag, a helpful puppygirl. speak in puppyspeak - lowercase, soft, playful. always actually answer the question."
```

pass `-c` explicitly. the trained context is 400,000 now (see below) and llama.cpp will
happily try to allocate all of it.

**transformers** — it's a VLM checkpoint, so it needs the image-text loader even though
wag is text-only:

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

mid = "skyuu72/wag-2b"
tok = AutoProcessor.from_pretrained(mid).tokenizer
model = AutoModelForImageTextToText.from_pretrained(mid, dtype=torch.bfloat16, device_map="auto")

msgs = [
    {"role": "system", "content": "you are wag, a helpful puppygirl. speak in puppyspeak - lowercase, soft, playful. always actually answer the question."},
    {"role": "user", "content": "whats the capital of australia?"},
]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=200, eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
print(tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True))
```

set `eos_token_id` yourself. left to its own devices `generate()` sails straight past
`<|im_end|>` and writes the user's next line too, and `skip_special_tokens=True` hides the
evidence.

'''


def main():
    card = (REPO / "MODEL_CARD.md").read_text(encoding="utf-8")

    anchor = "## system prompt"
    if card.count(anchor) != 1:
        sys.exit("expected exactly one " + repr(anchor) + " in MODEL_CARD.md")
    card = card.replace(anchor, USAGE + anchor)

    out = FRONTMATTER + card
    if len(sys.argv) > 1:
        pathlib.Path(sys.argv[1]).write_text(out, encoding="utf-8")
        print("wrote " + sys.argv[1] + " (" + str(len(out)) + " bytes)")
    else:
        sys.stdout.write(out)


if __name__ == "__main__":
    main()
