"""Incremental detokenization — streaming text from streamed token ids.

Detokenizing a stream is NOT "decode each id and concatenate", for two reasons:

  1. Byte-level BPE (Qwen/GPT-2 style) tokens are byte sequences, and a token can
     end mid-UTF-8-character (an emoji is 4 bytes, often split across tokens).
     Decoding at that boundary yields U+FFFD (the replacement char "�") for
     the dangling bytes — text we must HOLD BACK until the rest arrives, or the
     client sees garbage that later "changes".
  2. A token's rendered text can depend on its neighbors (byte-level "Ġ" prefixes,
     SentencePiece "▁" markers, multi-token specials), so a token decoded alone
     may not equal its slice of the full decode.

The fix is the sliding-window algorithm every serving engine uses (vLLM/TGI):
keep the ids, and on each push decode a window that starts at the PREVIOUS
emission boundary — so new ids are always decoded together with the chunk before
them, never at position 0 — and emit only the text past what that window had
already produced. If the window's decode ends in "�" the tail bytes are
incomplete: emit nothing and wait for more ids (flush() at stream end releases
whatever remains, so a genuinely malformed tail is still delivered once).

The invariant the gate checks: the concatenation of every emitted delta equals
the one-shot decode of all ids, and no delta ever contains a held-back "�".

Only `tokenizer.decode(ids)` is used (HF's sanctioned job — the golden rule).
`clean_up_tokenization_spaces` is pinned off: that post-pass rewrites text
non-locally (" ." -> "."), which would make window decodes disagree with the
full decode at chunk boundaries.
"""

from __future__ import annotations

REPLACEMENT = "�"


class IncrementalDetokenizer:
    """Feed generated ids with push(); it returns the newly-stable text (possibly
    "" while a multi-byte character is still arriving). Call flush() once the
    stream ends to release any held tail."""

    def __init__(self, tokenizer, skip_special_tokens: bool = True):
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.ids: list[int] = []
        # Window bookkeeping: ids[:read_offset] have had their text emitted;
        # ids[prefix_offset:read_offset] is the context chunk re-decoded with the
        # new ids so they never render at window position 0.
        self.prefix_offset = 0
        self.read_offset = 0

    def _decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        return self.tokenizer.decode(
            ids, skip_special_tokens=self.skip_special_tokens,
            clean_up_tokenization_spaces=False)

    def push(self, new_ids: list[int]) -> str:
        self.ids.extend(new_ids)
        prefix = self._decode(self.ids[self.prefix_offset:self.read_offset])
        full = self._decode(self.ids[self.prefix_offset:])
        if full.endswith(REPLACEMENT):
            return ""  # dangling multi-byte tail: hold until more ids arrive
        delta = full[len(prefix):]
        self.prefix_offset = self.read_offset
        self.read_offset = len(self.ids)
        return delta

    def flush(self) -> str:
        """Stream over: emit whatever is still held (even a malformed tail —
        better a visible U+FFFD than silently dropped text)."""
        prefix = self._decode(self.ids[self.prefix_offset:self.read_offset])
        full = self._decode(self.ids[self.prefix_offset:])
        self.prefix_offset = self.read_offset = len(self.ids)
        return full[len(prefix):]

    def text(self) -> str:
        """The one-shot decode of everything so far (the non-streaming answer;
        also what the emitted deltas + flush() must concatenate to)."""
        return self._decode(self.ids)
