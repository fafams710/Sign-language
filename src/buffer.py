"""Frame-agnostic recognized-gloss token buffer.

Provides :class:`TokenBuffer`, a small streaming buffer that sits between a
(future) real-time sign predictor and the rule-based grammar layer
(:mod:`src.grammar`). It ingests one already-recognized gloss string per
call, collapses a run of consecutive identical calls (a sign held across many
frames) into a single buffered token, and lets the caller inspect the
in-progress token sequence or finalize/emit a completed utterance.

This module performs no model inference and imports only the standard
library: it never imports torch, MediaPipe, or OpenCV, and it consumes only
gloss strings that have already been recognized upstream.
"""


class TokenBuffer:
    """Accumulate a debounced sequence of recognized gloss tokens.

    A run of consecutive :meth:`add` calls carrying the identical gloss
    string collapses into exactly one buffered token (a held sign does not
    repeat). A different gloss starts a new token, and the same gloss seen
    again later - after an intervening different gloss - is recorded as a
    new token rather than being merged with the earlier run.
    """

    def __init__(self):
        self._tokens = []
        self._last_seen = None

    def add(self, gloss) -> None:
        """Ingest one recognized gloss string.

        ``None``, non-string, and blank/whitespace-only input is ignored
        without raising. A gloss identical to the immediately preceding one
        is treated as part of the same held sign and does not add a new
        token.
        """
        if gloss is None:
            return
        text = str(gloss).strip()
        if not text:
            return
        if text == self._last_seen:
            return
        self._tokens.append(text)
        self._last_seen = text

    @property
    def tokens(self) -> list:
        """Return a copy of the current buffered token sequence.

        Does not mutate or clear the buffer.
        """
        return list(self._tokens)

    def emit(self) -> list:
        """Finalize the current utterance.

        Returns the buffered, de-duplicated token list (directly consumable
        by :func:`src.grammar.glosses_to_sentence`) and clears the buffer.
        Returns ``[]`` when the buffer is empty, including on a second
        consecutive call.
        """
        finalized = list(self._tokens)
        self.reset()
        return finalized

    def reset(self) -> None:
        """Clear the buffer without returning its contents."""
        self._tokens = []
        self._last_seen = None
