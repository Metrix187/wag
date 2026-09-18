#!/usr/bin/env python3
"""read and rewrite a gguf's metadata block.

`dump` prints the kv pairs. `set` changes or adds them - and since adding a key
grows the header, that one writes a whole new file: header rebuilt, tensor blob
copied over byte for byte. tensor offsets in the directory are relative to the
start of the data section, so they survive the header moving as long as the
padding lands on the same alignment.

exists because llama.cpp ships gguf-new-metadata.py inside its own repo and we
don't keep a clone around. if you have one, theirs is the better-tested path.
"""
import os, re, struct, sys, tempfile

T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32 = 0, 1, 2, 3, 4, 5
T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 6, 7, 8, 9, 10, 11, 12
FIXED_W = {T_UINT8: 1, T_INT8: 1, T_UINT16: 2, T_INT16: 2, T_UINT32: 4, T_INT32: 4,
           T_FLOAT32: 4, T_BOOL: 1, T_UINT64: 8, T_INT64: 8, T_FLOAT64: 8}
PACK = {T_UINT8: "B", T_INT8: "b", T_UINT16: "H", T_INT16: "h", T_UINT32: "I",
        T_INT32: "i", T_FLOAT32: "f", T_BOOL: "?", T_UINT64: "Q", T_INT64: "q",
        T_FLOAT64: "d"}
TYPE_NAMES = {"u8": T_UINT8, "i8": T_INT8, "u16": T_UINT16, "i16": T_INT16,
              "u32": T_UINT32, "i32": T_INT32, "f32": T_FLOAT32, "bool": T_BOOL,
              "str": T_STRING, "u64": T_UINT64, "i64": T_INT64, "f64": T_FLOAT64}


class Reader:
    def __init__(self, f):
        self.f = f

    def u32(self):
        return struct.unpack("<I", self.f.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.f.read(8))[0]

    def string(self):
        return self.f.read(self.u64()).decode("utf-8", "replace")

    def value(self, t):
        """python value, or a short placeholder for the bulky array types."""
        if t in FIXED_W:
            return struct.unpack("<" + PACK[t], self.f.read(FIXED_W[t]))[0]
        if t == T_STRING:
            return self.string()
        if t == T_ARRAY:
            et, n = self.u32(), self.u64()
            if et in FIXED_W:
                self.f.seek(FIXED_W[et] * n, 1)
            elif et == T_STRING:
                for _ in range(n):
                    self.f.seek(self.u64(), 1)
            else:
                for _ in range(n):
                    self.value(et)
            return "<array of " + str(n) + ", elem type " + str(et) + ">"
        raise ValueError("unknown gguf value type " + str(t))


def parse(f):
    """kv comes back as mutable rows [key, type, display value, raw bytes] so a
    key we aren't touching gets copied out exactly as it came in."""
    r = Reader(f)
    if f.read(4) != b"GGUF":
        raise ValueError("not a gguf")
    version = r.u32()
    n_tensors, n_kv = r.u64(), r.u64()

    kv = []
    for _ in range(n_kv):
        key = r.string()
        t = r.u32()
        start = f.tell()
        val = r.value(t)
        end = f.tell()
        f.seek(start)
        kv.append([key, t, val, f.read(end - start)])

    tstart = f.tell()
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        tensors.append((name, dims, r.u32(), r.u64()))
    tend = f.tell()
    f.seek(tstart)
    tensor_raw = f.read(tend - tstart)

    align = 32
    for key, t, val, _ in kv:
        if key == "general.alignment":
            align = val
    return version, kv, tensors, tensor_raw, align, tend + (-tend % align)


def encode(t, text):
    if t == T_STRING:
        b = text.encode("utf-8")
        return struct.pack("<Q", len(b)) + b
    if t == T_BOOL:
        return struct.pack("<?", text.lower() in ("1", "true", "yes"))
    if t in (T_FLOAT32, T_FLOAT64):
        return struct.pack("<" + PACK[t], float(text))
    return struct.pack("<" + PACK[t], int(text))


def cmd_dump(path, pattern=None):
    with open(path, "rb") as f:
        version, kv, tensors, _, align, data_start = parse(f)
    print(path)
    print("  gguf v" + str(version) + "  tensors=" + str(len(tensors)) + "  kv=" +
          str(len(kv)) + "  align=" + str(align) + "  data@" + str(data_start))
    for key, t, val, _ in kv:
        if pattern and not re.search(pattern, key, re.I):
            continue
        s = str(val)
        print("    " + key + " (" + str(t) + ") = " + (s[:110] + "..." if len(s) > 110 else s))


def cmd_set(path, assignments, dry_run=False):
    with open(path, "rb") as f:
        version, kv, tensors, tensor_raw, align, data_start = parse(f)
        index = {row[0]: row for row in kv}
        changed, added = [], []
        for key, tname, text in assignments:
            t = TYPE_NAMES[tname]
            raw = encode(t, text)
            row = index.get(key)
            if row is None:
                row = [key, t, text, raw]
                kv.append(row)
                index[key] = row
                added.append(key + " = " + text)
            elif row[1] != t or row[3] != raw:
                old = str(row[2])
                row[1], row[2], row[3] = t, text, raw
                changed.append(key + ": " + old + " -> " + text)

        if not changed and not added:
            print(path + ": already set, nothing to do")
            return
        for line in changed:
            print("    changed " + line)
        for line in added:
            print("    added   " + line)
        if dry_run:
            print(path + ": dry run, not writing")
            return

        head = bytearray(b"GGUF")
        head += struct.pack("<I", version)
        head += struct.pack("<Q", len(tensors))
        head += struct.pack("<Q", len(kv))
        for key, t, _, raw in kv:
            kb = key.encode("utf-8")
            head += struct.pack("<Q", len(kb)) + kb + struct.pack("<I", t) + raw
        head += tensor_raw
        head += bytes(-len(head) % align)

        tmp = tempfile.NamedTemporaryFile(dir=os.path.dirname(os.path.abspath(path)),
                                          prefix=".ggufmeta-", delete=False)
        try:
            tmp.write(head)
            f.seek(data_start)
            while True:
                chunk = f.read(32 << 20)
                if not chunk:
                    break
                tmp.write(chunk)
            tmp.close()
        except BaseException:
            tmp.close()
            os.unlink(tmp.name)
            raise
    os.replace(tmp.name, path)
    print(path + ": rewritten, header now " + str(len(head)) + " bytes")


USAGE = ("usage:\n"
         "  gguf_meta.py dump FILE [key-regex]\n"
         "  gguf_meta.py set [--dry-run] FILE key:type=value [key:type=value ...]\n"
         "types: ")


def main(argv):
    if len(argv) < 3:
        sys.exit(USAGE + " ".join(sorted(TYPE_NAMES)))
    if argv[1] == "dump":
        cmd_dump(argv[2], argv[3] if len(argv) > 3 else None)
        return
    if argv[1] == "set":
        args = [a for a in argv[2:] if a != "--dry-run"]
        assignments = []
        for a in args[1:]:
            spec, _, value = a.partition("=")
            key, _, tname = spec.partition(":")
            if tname not in TYPE_NAMES:
                sys.exit("bad type " + repr(tname) + " in " + a)
            assignments.append((key, tname, value))
        cmd_set(args[0], assignments, dry_run="--dry-run" in argv)
        return
    sys.exit("unknown command " + argv[1])


if __name__ == "__main__":
    main(sys.argv)
