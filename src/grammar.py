"""Rule-based sentence generation.

Converts a finalized sequence of recognized gloss tokens (already-recognized
sign names, e.g. from :class:`src.buffer.TokenBuffer`) into a readable,
punctuated English sentence using deterministic, data-driven templates. This
module never re-implements or re-hardcodes the project's sign vocabulary; it
imports :data:`src.utils.VOCABULARY` and treats it as the single source of
truth for which gloss tokens are in-vocabulary.

Scope: this generates text for a fixed vocabulary of predefined signs only.
It is not, and does not claim to be, a general ASL translator. (An optional
LLM-based grammar layer is future work and, if ever added, would be used
only for sentence phrasing, never for sign recognition.)

Public entry point
-------------------
``glosses_to_sentence(tokens)`` is the stable, module-level API that later
phases (real-time prediction, the UI) call with the buffer's emitted token
list. It never raises: empty input, blank/``None``-like tokens, and
out-of-vocabulary tokens are all handled gracefully rather than causing an
exception.
"""

from src.utils import VOCABULARY

# Case-insensitive membership set used only to decide which tokens are
# eligible for the templates below (question intent, the NEED+HELP request
# clause). Rendering of the resulting words is the same humanization step
# regardless of vocabulary membership (see _humanize); VOCABULARY is never
# duplicated as a list of signs, only consulted for membership.
_VOCAB_SET = {sign.upper() for sign in VOCABULARY}

# Gloss tokens whose presence anywhere in the sequence forces the sentence to
# be rendered as a question.
_QUESTION_SIGNS = {"WHERE", "WHAT", "WHY"}

# Terminal punctuation used for non-question output.
_STATEMENT_PUNCTUATION = "."
_QUESTION_PUNCTUATION = "?"


def _is_known(token_upper: str) -> bool:
    """Return True if ``token_upper`` matches a VOCABULARY entry."""
    return token_upper in _VOCAB_SET


def _humanize(token: str) -> str:
    """Render a single gloss token as a lowercase, space-separated word.

    Applies to both in-vocabulary tokens without a dedicated combination
    template and out-of-vocabulary tokens (the shared fallback rendering).
    """
    word = token.replace("_", " ").replace("-", " ").lower()
    return " ".join(word.split())


def _capitalize_first(text: str) -> str:
    """Uppercase the first alphabetic character of ``text``, if any."""
    for index, char in enumerate(text):
        if char.isalpha():
            return text[:index] + char.upper() + text[index + 1:]
    return text


def glosses_to_sentence(tokens: list) -> str:
    """Convert a finalized list of gloss tokens into one English sentence.

    Parameters
    ----------
    tokens:
        A sequence of recognized gloss strings, typically the list returned
        by :meth:`src.buffer.TokenBuffer.emit`. ``None`` entries, blank
        strings, and out-of-vocabulary tokens are all tolerated.

    Returns
    -------
    str
        ``""`` for an empty (or entirely blank) input. Otherwise a single
        sentence whose first alphabetic character is uppercase and which
        ends in ``"?"`` if any of the question signs WHERE/WHAT/WHY is
        present, or ``"."`` otherwise. Underscored gloss tokens (e.g.
        ``THANK_YOU``) are rendered as humanized, space-separated words. The
        request combination NEED followed by HELP is rendered as a single
        "I need help"-style clause rather than two disconnected words. This
        function never raises.
    """
    cleaned = []
    for token in tokens or []:
        if token is None:
            continue
        text = str(token).strip()
        if not text:
            continue
        cleaned.append(text)

    if not cleaned:
        return ""

    canonical = [text.upper() for text in cleaned]

    question = any(
        _is_known(sign) and sign in _QUESTION_SIGNS for sign in canonical
    )

    pieces = []
    index = 0
    total = len(cleaned)
    while index < total:
        current = canonical[index]
        if (
            current == "NEED"
            and _is_known("NEED")
            and index + 1 < total
            and canonical[index + 1] == "HELP"
            and _is_known("HELP")
        ):
            pieces.append("I need help")
            index += 2
            continue
        word = _humanize(cleaned[index])
        if word:
            pieces.append(word)
        index += 1

    if not pieces:
        return ""

    body = " ".join(pieces)
    sentence = _capitalize_first(body)
    punctuation = _QUESTION_PUNCTUATION if question else _STATEMENT_PUNCTUATION
    return f"{sentence}{punctuation}"
