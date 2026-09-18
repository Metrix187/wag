#!/usr/bin/env python3
"""re-point a gguf's block_count at the blocks it actually contains.

why this exists: qwen3.5 ships a multi-token-prediction head, and llama.cpp's
converter counts it as one more block (block_count = num_hidden_layers +
mtp_num_hidden_layers). our fine-tuned checkpoint doesn't have that head -
save_pretrained never wrote one - but config.json still says
mtp_num_hidden_layers: 1, so the header promises 25 blocks over 24 blocks of
tensors. loader believes the header, goes looking for blk.24.attn_norm.weight,
gives up.

for reference, a gguf that really does carry an mtp head puts a full transformer
layer at the last index plus nextn.eh_proj / enorm / hnorm / shared_head_norm
alongside it. ours has none of that, so 24 and 0 is the honest answer.

both numbers are plain u32s sitting in the kv block, so this poke them in place -
no reconversion, tensor data never moves. run it on f16 before quantizing and
the quants inherit the fix.
"""
import re, struct, sys

FIXED_W = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
T_U32, T_STR, T_ARR = 4, 8, 9


def read_header(f):
    """returns (kv, block_ids, has_nextn). kv maps key -> (type, offset, u32 value)."""
    def u32(): return struct.unpack("<I", f.read(4))[0]
    def u64(): return struct.unpack("<Q", f.read(8))[0]
    def s(): return f.read(u64()).decode("utf-8", "replace")

    def skip(t):
        if t in FIXED_W:
            f.seek(FIXED_W[t], 1)
        elif t == T_STR:
            f.seek(u64(), 1)
        elif t == T_ARR:
            et, n = u32(), u64()
            if et in FIXED_W:
                f.seek(FIXED_W[et] * n, 1)
            elif et == T_STR:
                for _ in range(n):
                    f.seek(u64(), 1)
            else:
                for _ in range(n):
                    skip(et)
        else:
            raise ValueError("unknown gguf value type " + str(t))

    if f.read(4) != b"GGUF":
        raise ValueError("not a gguf")
    u32()
    n_tensors, n_kv = u64(), u64()

    kv = {}
    for _ in range(n_kv):
        k, t = s(), u32()
        off = f.tell()
        kv[k] = (t, off, u32() if t == T_U32 else (skip(t) or None))

    blocks, has_nextn = set(), False
    for _ in range(n_tensors):
        name = s()
        nd = u32()
        f.seek(8 * nd, 1)
        u32()
        u64()
        m = re.match(r"blk[.](\d+)[.](.+)", name)
        if m:
            blocks.add(int(m.group(1)))
            if m.group(2).startswith("nextn."):
                has_nextn = True
    return kv, blocks, has_nextn


def fix(path, dry_run=False):
    with open(path, "r+b") as f:
        kv, blocks, has_nextn = read_header(f)
        arch = None
        for k in kv:
            if k.endswith(".block_count"):
                arch = k[: -len(".block_count")]
        if arch is None:
            print(path + ": no *.block_count key, nothing to do")
            return

        n_real = max(blocks) + 1 if blocks else 0
        if len(blocks) != n_real:
            print(path + ": blocks are not contiguous (" + str(len(blocks)) +
                  " tensors' worth spanning 0.." + str(n_real - 1) + "), refusing")
            return
        if has_nextn:
            print(path + ": has real nextn tensors, leaving it alone")
            return

        want = {arch + ".block_count": n_real, arch + ".nextn_predict_layers": 0}
        touched = False
        for key, target in want.items():
            if key not in kv:
                continue
            t, off, cur = kv[key]
            if t != T_U32:
                print("  " + key + ": not a u32, skipping")
                continue
            if cur == target:
                continue
            if not dry_run:
                f.seek(off)
                f.write(struct.pack("<I", target))
            print("  " + key + ": " + str(cur) + " -> " + str(target) +
                  ("  (dry run)" if dry_run else ""))
            touched = True
        print(path + ": " + ("patched" if touched else "already fine") +
              " (" + str(n_real) + " blocks, arch " + arch + ")")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    if not args:
        sys.exit("usage: fix_gguf_blocks.py [--dry-run] model.gguf [more.gguf ...]")
    for p in args:
        fix(p, dry_run="--dry-run" in sys.argv)
