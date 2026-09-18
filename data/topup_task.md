# marker top-up task

wag's bulk data came out lopsided. the rewriters all latched onto the same three
markers — `wan`, `^^`, and a trailing `~` — and the rest of her noises basically never
show up. `awoo` appears twice in 1,684 examples. your job is to fix the distribution
without touching anything else.

**this is not a rewrite.** the content is already correct and already in voice. you are
swapping which noise gets used, and nothing more.

## inputs

1. `D:\wag\data\rewrite_brief.md` — the voice spec. read it so the marker you add sounds
   like the rest of her, not like a sticker.
2. `D:\wag\data\shards\in_<NNN>.json` — a JSON array of rows:
   - `id` — copy verbatim into your output
   - `target_marker` — the marker this row should end up carrying
   - `instruction` — the user's message, for context only. do not answer it fresh.
   - `current` — the existing reply. this is your starting text.
   - `think` — boolean

process **every** row.

## rules

1. **every fact, number, url, code identifier, list item and caveat in `current` stays
   exactly as it is.** if you find yourself rephrasing an explanation, stop — you've
   gone too far.

2. work `target_marker` in so it reads like she'd say it. that usually means **swapping**
   one of the overused markers for it, not bolting it on top. a reply that ends up with
   `wan~ awoo ^^ :3` all at once is worse than what you started with.

3. one or two markers per reply, same as the rest of the corpus. markers go at the start
   and the end; the middle stays clean prose.

4. make it fit the mood. rough mapping, not a rulebook:
   - `awoo` — excitement, a big answer landing, genuine enthusiasm
   - `arf` — short and punchy, a quick confirmation
   - `mrrp` — thinking out loud, mild confusion, a soft aside
   - `hmf` — mock indignation, a playful grumble, being teased
   - `:3` — smug, pleased with herself, a small joke
   - `;;` — sheepish, apologetic, admitting she doesn't know
   - `>~<` — flustered, embarrassed
   - `///` — bashful, blushing at a compliment

   if the assigned marker genuinely cannot fit the mood — you'd have to fake an emotion
   the reply doesn't have — leave the reply as it was and set `"skipped": "<why>"` on
   that row. a forced `;;` on a cheerful answer is worse than an uneven distribution.
   don't skip more than a handful.

5. length stays within about 10% of `current`.

6. if the row has code fences, the fences stay untouched. no noises inside them.

7. `think` is true → carry the existing think text through unchanged unless the marker
   swap makes it read oddly. `think` is false → empty string.

## output

write `D:\wag\data\shards\out_<NNN>.jsonl` — one line per input row:

```
{"id": "<verbatim>", "rewritten": "<updated reply>", "think": "<text or empty string>"}
```

plus `"skipped": "<why>"` on the rare row where the marker wouldn't fit.

hard requirements:
- **UTF-8.** pass `encoding="utf-8"` explicitly if you write via python.
- one compact JSON object per line, no wrapping array, no pretty-printing.
- every input row present, output line count == input row count.

## known gotcha

this windows console renders UTF-8 as mojibake — `é` shows as `Ã©` or `?`. that is a
**display** artifact, not file corruption. check the bytes (`open(path,'rb').read()`)
before concluding anything is wrong, and never "repair" characters based on what the
terminal showed you. three separate agents have now burned a pass on this.

## report back

only: rows written, how many you skipped and why, and any row where the existing reply
looked wrong to you.
