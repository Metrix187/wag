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
base_model: Qwen/Qwen3.5-4B
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

**llama.cpp / LM Studio / ollama** — grab a quant from `gguf/` and go. `wag-v2-q4_k_m.gguf`
is the one to use; `wag-v2-q8_0.gguf` if you have the room.

```bash
llama-cli -m wag-v2-q4_k_m.gguf -c 8192 -sys "you are wag, a helpful puppygirl. speak in puppyspeak - lowercase, soft, playful. always actually answer the question."
```

pass `-c` explicitly rather than letting it default to the base model's full context.

**transformers** — v2 is a plain causal-LM checkpoint (`Qwen3_5ForCausalLM`), so unlike
v1 it does *not* need the image-text loader:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = "skyuu72/wag-4b"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="auto")

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

**the system prompt is optional.** 10% of the training rows carry none at all, and the
voice survives without one — that's the measured difference between this checkpoint and
the earlier ones, see the eval section.

### no `<think>` block, on purpose

stock qwen3.5 ends a generation prompt on an open `<think>` and expects the model to close
it. wag doesn't — the reasoning scaffolding sat in the masked region during training, so
she never learnt to emit `</think>` at all. left as shipped that makes her look broken in
any client that parses thinking: the whole reply gets classified as reasoning and you get
an empty message back.

so the template here has no think block in the generation path. if you re-convert from the
safetensors yourself, apply `fix_gguf_think.py` from
[the repo](https://github.com/Metrix187/wag) to the result.

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
