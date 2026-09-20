#!/usr/bin/env python3
"""a little box to paste the gemini api key into, so it never goes through a terminal.

writes `.env` next to this file, which is gitignored. the key stays on this machine:
nothing here prints it, logs it, or sends it anywhere except google's own endpoint if
you press Test. delete `.env` to revoke it.

    python ask_key.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
ENV = ROOT / ".env"
VAR = "GEMINI_API_KEY"
PROJ = "GOOGLE_CLOUD_PROJECT"
LOC = "GOOGLE_CLOUD_LOCATION"


def read_env() -> dict[str, str]:
    """existing .env as a dict. values are never printed, only counted."""
    out: dict[str, str] = {}
    if ENV.exists():
        for line in ENV.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def write_env(values: dict[str, str]) -> None:
    body = "# local secrets. gitignored. delete this file to revoke.\n"
    body += "".join(f"{k}={v}\n" for k, v in values.items())
    ENV.write_text(body, encoding="utf-8", newline="\n")
    try:
        os.chmod(ENV, 0o600)   # best effort; windows mostly shrugs at this
    except OSError:
        pass


def check_key(key: str) -> tuple[bool, str]:
    """models.list is free, so this proves the key works without spending anything."""
    try:
        from google import genai
    except ImportError:
        return False, "google-genai isn't installed (pip install google-genai)"
    try:
        client = genai.Client(api_key=key)
        names = [m.name for m in client.models.list()]
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        # never echo the key back, even if the sdk pasted it into the error
        msg = msg.replace(key, "<key>")
        return False, msg[:200]
    flash = [n for n in names if "flash" in n]
    return True, (f"key works — {len(names)} models, {len(flash)} flash. note this only "
                  "proves auth; credit is a separate question")


def check_vertex(project: str, location: str) -> tuple[bool, str]:
    """vertex bills the cloud project, which is a different pot from ai studio prepay.

    auth here is application default credentials, not a key — so a failure usually means
    `gcloud auth application-default login` hasn't been run, not that anything's wrong
    with the project.
    """
    try:
        from google import genai
    except ImportError:
        return False, "google-genai isn't installed"
    try:
        client = genai.Client(vertexai=True, project=project, location=location)
        names = [m.name for m in client.models.list()]
    except Exception as e:  # noqa: BLE001
        msg = str(e)[:220]
        if "default credentials" in msg.lower() or "could not automatically" in msg.lower():
            return False, "no application default credentials — run: " \
                          "gcloud auth application-default login"
        return False, msg
    return True, f"vertex reachable — {len(names)} models in {location}"


def main() -> int:
    import tkinter as tk
    from tkinter import ttk

    existing = read_env()
    has_key = bool(existing.get(VAR))

    root = tk.Tk()
    root.title("wag — gemini api key")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    frm = ttk.Frame(root, padding=16)
    frm.grid()

    ttk.Label(frm, text="paste your gemini api key",
              font=("Segoe UI", 11, "bold")).grid(column=0, row=0, columnspan=3, sticky="w")
    ttk.Label(frm, foreground="#666", wraplength=430, justify="left",
              text=(f"saved to {ENV} (gitignored, stays on this machine). "
                    "delete that file to revoke.")
              ).grid(column=0, row=1, columnspan=3, sticky="w", pady=(2, 10))

    var = tk.StringVar()
    entry = ttk.Entry(frm, textvariable=var, width=52, show="•")
    entry.grid(column=0, row=2, columnspan=2, sticky="we")
    entry.focus()

    show = tk.BooleanVar(value=False)
    ttk.Checkbutton(frm, text="show", variable=show,
                    command=lambda: entry.configure(show="" if show.get() else "•")
                    ).grid(column=2, row=2, padx=(8, 0))

    ttk.Separator(frm, orient="horizontal").grid(column=0, row=3, columnspan=3,
                                                 sticky="we", pady=(14, 10))
    ttk.Label(frm, text="google cloud (optional)",
              font=("Segoe UI", 10, "bold")).grid(column=0, row=4, columnspan=3, sticky="w")
    ttk.Label(frm, foreground="#666", wraplength=430, justify="left",
              text=("only if the credit sits on a cloud project rather than ai studio — "
                    "they're separate pots. then use --backend vertex.")
              ).grid(column=0, row=5, columnspan=3, sticky="w", pady=(2, 8))

    ttk.Label(frm, text="project id").grid(column=0, row=6, sticky="w")
    proj = tk.StringVar(value=existing.get(PROJ, ""))
    ttk.Entry(frm, textvariable=proj, width=34).grid(column=1, row=6, columnspan=2,
                                                     sticky="we", pady=2)

    ttk.Label(frm, text="location").grid(column=0, row=7, sticky="w")
    loc = tk.StringVar(value=existing.get(LOC, "") or "us-central1")
    ttk.Entry(frm, textvariable=loc, width=34).grid(column=1, row=7, columnspan=2,
                                                    sticky="we", pady=2)

    status = ttk.Label(frm, text=("a key is already saved — paste a new one to replace it"
                                  if has_key else ""),
                       foreground="#666", wraplength=430, justify="left")
    status.grid(column=0, row=8, columnspan=3, sticky="w", pady=(10, 0))

    result = {"saved": False}

    def say(msg: str, colour: str = "#666") -> None:
        status.configure(text=msg, foreground=colour)
        root.update_idletasks()

    def on_test() -> None:
        key = var.get().strip()
        if not key:
            say("nothing pasted yet", "#b00")
            return
        say("checking...")
        ok, msg = check_key(key)
        say(msg, "#070" if ok else "#b00")

    def on_test_vertex() -> None:
        p = proj.get().strip()
        if not p:
            say("no project id", "#b00")
            return
        say("checking vertex...")
        ok, msg = check_vertex(p, loc.get().strip() or "us-central1")
        say(msg, "#070" if ok else "#b00")

    def on_save() -> None:
        key = var.get().strip()
        p, l = proj.get().strip(), loc.get().strip()
        if not key and not p:
            say("nothing to save", "#b00")
            return
        vals = read_env()
        if key:
            vals[VAR] = key
        if p:
            vals[PROJ] = p
            vals[LOC] = l or "us-central1"
        write_env(vals)
        result["saved"] = True
        say(f"saved {', '.join(k for k in (VAR if key else None, PROJ if p else None) if k)}"
            f" to {ENV.name}", "#070")
        root.after(900, root.destroy)

    btns = ttk.Frame(frm)
    btns.grid(column=0, row=9, columnspan=3, sticky="e", pady=(14, 0))
    ttk.Button(btns, text="Test key", command=on_test).grid(column=0, row=0, padx=(0, 6))
    ttk.Button(btns, text="Test vertex", command=on_test_vertex).grid(column=1, row=0,
                                                                      padx=(0, 6))
    ttk.Button(btns, text="Save", command=on_save).grid(column=2, row=0, padx=(0, 6))
    ttk.Button(btns, text="Cancel", command=root.destroy).grid(column=3, row=0)

    root.bind("<Return>", lambda _e: on_save())
    root.bind("<Escape>", lambda _e: root.destroy())

    # roughly centre it so it doesn't land off-screen on a multi-monitor setup
    root.update_idletasks()
    w, h = root.winfo_width(), root.winfo_height()
    x = (root.winfo_screenwidth() - w) // 2
    y = (root.winfo_screenheight() - h) // 3
    root.geometry(f"+{x}+{y}")

    root.mainloop()

    if result["saved"]:
        print(f"key saved to {ENV}")
        return 0
    print("cancelled — nothing written")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
