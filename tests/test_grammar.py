"""Unit tests for src.grammar and src.buffer.

All tests run on plain Python strings/lists only. They require no webcam, no
MediaPipe / OpenCV install, no torch, and no ``models/hand_landmarker.task``
binary. Run with: ``pytest tests/``.
"""

import importlib
import sys

from src import buffer, grammar
from src.buffer import TokenBuffer
from src.grammar import glosses_to_sentence
from src.utils import VOCABULARY

_QUESTION_SIGNS = ("WHERE", "WHAT", "WHY")


# --------------------------------------------------------------------------- #
# Import safety / purity
# --------------------------------------------------------------------------- #
def test_grammar_imports_cleanly_without_heavy_deps():
    importlib.import_module("src.grammar")
    assert "torch" not in sys.modules
    for banned in ("mediapipe", "cv2"):
        assert banned not in sys.modules, (
            f"src.grammar (or its imports) pulled in {banned}; it must not."
        )


def test_buffer_imports_cleanly_without_heavy_deps():
    importlib.import_module("src.buffer")
    assert "torch" not in sys.modules
    for banned in ("mediapipe", "cv2"):
        assert banned not in sys.modules, (
            f"src.buffer (or its imports) pulled in {banned}; it must not."
        )


def test_grammar_uses_vocabulary_not_a_re_hardcoded_copy():
    # The module must import the shared VOCABULARY rather than defining its
    # own duplicate sign list.
    assert grammar.VOCABULARY is VOCABULARY


# --------------------------------------------------------------------------- #
# AC2 : empty input
# --------------------------------------------------------------------------- #
def test_empty_list_returns_empty_string():
    assert glosses_to_sentence([]) == ""


def test_blank_and_none_only_tokens_return_empty_string():
    assert glosses_to_sentence([None, "", "   "]) == ""


# --------------------------------------------------------------------------- #
# AC3 : single in-vocabulary token
# --------------------------------------------------------------------------- #
def test_single_known_token_is_capitalized_and_punctuated():
    assert "HELLO" in VOCABULARY
    sentence = glosses_to_sentence(["HELLO"])
    assert sentence != ""
    assert sentence[0].isupper()
    assert sentence[-1] in ".!?"


# --------------------------------------------------------------------------- #
# AC4 : multi-gloss statement -> period
# --------------------------------------------------------------------------- #
def test_multi_gloss_statement_ends_with_period():
    tokens = ["TEACHER", "CLASS", "TODAY"]
    for tok in tokens:
        assert tok in VOCABULARY
    sentence = glosses_to_sentence(tokens)
    assert sentence[0].isupper()
    assert sentence.endswith(".")


# --------------------------------------------------------------------------- #
# AC5 : question signs -> question mark
# --------------------------------------------------------------------------- #
def test_each_question_sign_forces_question_mark():
    for sign in _QUESTION_SIGNS:
        assert sign in VOCABULARY
        sentence = glosses_to_sentence([sign])
        assert sentence.endswith("?"), sentence


def test_multiple_question_signs_still_single_question_sentence():
    sentence = glosses_to_sentence(["WHERE", "WHAT"])
    assert sentence.endswith("?")
    assert sentence.count("?") == 1


def test_question_sign_combined_with_statement_signs_wins():
    sentence = glosses_to_sentence(["TEACHER", "WHERE", "CLASS"])
    assert sentence.endswith("?")


# --------------------------------------------------------------------------- #
# AC6 : underscore humanization
# --------------------------------------------------------------------------- #
def test_underscore_token_is_humanized_with_space():
    assert "THANK_YOU" in VOCABULARY
    sentence = glosses_to_sentence(["THANK_YOU"])
    assert "_" not in sentence
    assert "thank you" in sentence.lower()


# --------------------------------------------------------------------------- #
# AC7 : OOV robustness
# --------------------------------------------------------------------------- #
def test_single_oov_token_does_not_raise_and_returns_string():
    assert "NOT_A_REAL_SIGN" not in VOCABULARY
    sentence = glosses_to_sentence(["NOT_A_REAL_SIGN"])
    assert isinstance(sentence, str)


def test_all_oov_sequence_does_not_raise():
    tokens = ["FOOBAR", "NOT_A_SIGN", "ZZZ"]
    for tok in tokens:
        assert tok not in VOCABULARY
    sentence = glosses_to_sentence(tokens)
    assert isinstance(sentence, str)
    assert sentence != ""
    assert sentence[-1] in ".!?"


def test_mixed_known_and_unknown_tokens_still_coherent():
    sentence = glosses_to_sentence(["HELLO", "NOT_A_REAL_SIGN", "TODAY"])
    assert isinstance(sentence, str)
    assert sentence != ""
    assert "hello" in sentence.lower()
    assert "today" in sentence.lower()


# --------------------------------------------------------------------------- #
# AC8 : NEED + HELP single request clause
# --------------------------------------------------------------------------- #
def test_need_help_combination_is_one_clause():
    assert "NEED" in VOCABULARY and "HELP" in VOCABULARY
    sentence = glosses_to_sentence(["NEED", "HELP"])
    assert sentence.count(".") + sentence.count("!") + sentence.count("?") == 1
    assert "need" in sentence.lower()
    assert "help" in sentence.lower()


# --------------------------------------------------------------------------- #
# Determinism / long sequence robustness
# --------------------------------------------------------------------------- #
def test_output_is_deterministic():
    tokens = ["HELLO", "TEACHER", "WHERE"]
    first = glosses_to_sentence(tokens)
    second = glosses_to_sentence(tokens)
    assert first == second


def test_very_long_sequence_does_not_crash():
    tokens = [VOCABULARY[i % len(VOCABULARY)] for i in range(60)]
    sentence = glosses_to_sentence(tokens)
    assert isinstance(sentence, str)
    assert sentence != ""


# --------------------------------------------------------------------------- #
# Buffer: AC10 / AC11 / AC12
# --------------------------------------------------------------------------- #
def test_buffer_starts_empty():
    buf = TokenBuffer()
    assert buf.tokens == []


def test_buffer_consecutive_dedup_collapses_run():
    buf = TokenBuffer()
    for _ in range(5):
        buf.add("HELLO")
    assert buf.tokens == ["HELLO"]


def test_buffer_interleaved_repeat_allows_new_token_after_gap():
    buf = TokenBuffer()
    for gloss in ("A", "A", "B", "A"):
        buf.add(gloss)
    assert buf.tokens == ["A", "B", "A"]


def test_buffer_current_tokens_read_is_non_destructive():
    buf = TokenBuffer()
    buf.add("HELLO")
    buf.add("YES")
    first_read = buf.tokens
    second_read = buf.tokens
    assert first_read == second_read == ["HELLO", "YES"]
    # Mutating the returned list must not affect internal state.
    first_read.append("SHOULD_NOT_PERSIST")
    assert buf.tokens == ["HELLO", "YES"]


def test_buffer_emit_returns_tokens_and_clears():
    buf = TokenBuffer()
    buf.add("HELLO")
    buf.add("HELLO")
    buf.add("TODAY")
    finalized = buf.emit()
    assert finalized == ["HELLO", "TODAY"]
    assert buf.tokens == []


def test_buffer_emit_twice_second_call_is_empty():
    buf = TokenBuffer()
    buf.add("HELLO")
    first = buf.emit()
    second = buf.emit()
    assert first == ["HELLO"]
    assert second == []


def test_buffer_emit_on_empty_buffer_returns_empty_list():
    buf = TokenBuffer()
    assert buf.emit() == []


def test_buffer_reset_clears_state():
    buf = TokenBuffer()
    buf.add("HELLO")
    buf.reset()
    assert buf.tokens == []
    # A subsequent add of the same gloss is not treated as a continued run
    # from before the reset.
    buf.add("HELLO")
    assert buf.tokens == ["HELLO"]


def test_buffer_ignores_none_and_blank_input():
    buf = TokenBuffer()
    buf.add(None)
    buf.add("")
    buf.add("   ")
    assert buf.tokens == []
    buf.add("HELLO")
    assert buf.tokens == ["HELLO"]


def test_buffer_no_inference_module_attributes():
    # The buffer module must expose no torch/model-related surface.
    assert not hasattr(buffer, "torch")
