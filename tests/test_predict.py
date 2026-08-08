"""Unit tests for src.predict.

All tests run on synthetic data only: synthetic NumPy arrays, plain dicts,
small local fake-landmark objects, and ``tmp_path``. They require no webcam,
no MediaPipe / OpenCV install, no torch, no ``models/hand_landmarker.task``
binary and no trained checkpoint. Run with: ``pytest tests/``.
"""

import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src import landmarks, predict
from src.buffer import TokenBuffer
from src.grammar import glosses_to_sentence
from src.utils import SEQUENCE_LENGTH, VOCABULARY

N_PER_HAND = landmarks.N_FEATURES  # 63
N_FRAME = predict.N_FEATURES_PER_FRAME  # 126

_REPO_ROOT = Path(__file__).resolve().parent.parent

# A handful of tests below need a genuine torch checkpoint built via
# src.train, which transitively imports torch, pandas and scikit-learn. That
# check is done with importlib.util.find_spec (which locates a module
# WITHOUT importing it), never a bare ``import torch`` / ``pytest.
# importorskip("torch")`` at this point: pytest imports every test module
# during collection before running any test, and this file is collected
# before tests/test_preprocess.py, which asserts (by checking global
# sys.modules) that importing src.preprocess alone never pulls in torch. An
# in-process ``import torch`` here would leak into that later, unrelated
# test purely due to file collection order (torch's C extension state also
# cannot be safely dropped from sys.modules and re-imported later in the
# same process, so "import then clean up" is not a safe alternative). The
# actual torch-dependent work below therefore always runs in a fresh child
# process via subprocess.run, whose sys.modules is discarded when it exits.
_TRAIN_STACK_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "pandas", "sklearn")
)
_skip_without_train_stack = pytest.mark.skipif(
    not _TRAIN_STACK_AVAILABLE,
    reason="torch/pandas/scikit-learn are not installed",
)


def _run_in_subprocess(script: str) -> subprocess.CompletedProcess:
    """Run ``script`` in a fresh child interpreter, cwd'd at the repo root.

    Isolates torch/pandas/scikit-learn imports from this test process (see
    the module-level comment above) and gives ``script`` access to the
    ``src`` package the same way ``pytest tests/`` run from the repo root
    does.
    """
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"subprocess failed (exit {result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    return result


# --------------------------------------------------------------------------- #
# Synthetic data helpers (mirrors tests/test_preprocess.py)
# --------------------------------------------------------------------------- #
def _hand_features(seed):
    """Deterministic nonzero 63-vector with a nonzero wrist so normalization
    yields a clearly nonzero result."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 0.9, size=N_PER_HAND).astype(np.float32)


class _FakeLandmark:
    """Minimal stand-in for a MediaPipe landmark object, exposing .x/.y/.z."""

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def _feats_to_fake_landmarks(feats):
    return [
        _FakeLandmark(float(feats[i * 3]), float(feats[i * 3 + 1]), float(feats[i * 3 + 2]))
        for i in range(landmarks.N_LANDMARKS)
    ]


def _row(frame, hand_index, handedness, feats):
    return {
        "meta": {
            "timestamp": 0.0,
            "frame": frame,
            "hand_index": hand_index,
            "handedness": handedness,
        },
        "feats": feats,
    }


def _to_meta_and_feats(rows):
    meta = [r["meta"] for r in rows]
    feats = np.stack([r["feats"] for r in rows], axis=0).astype(np.float32)
    return meta, feats


# --------------------------------------------------------------------------- #
# AC3 : import purity
# --------------------------------------------------------------------------- #
def test_import_purity_no_heavy_deps():
    importlib.import_module("src.predict")
    for banned in ("torch", "mediapipe", "cv2"):
        assert banned not in sys.modules, (
            f"src.predict (or its imports) pulled in {banned}; it must not."
        )


# --------------------------------------------------------------------------- #
# AC4 : constant identity
# --------------------------------------------------------------------------- #
def test_vocabulary_identity():
    assert predict.VOCABULARY is VOCABULARY


def test_sequence_length_default_resolves():
    assert predict.SEQUENCE_LENGTH == SEQUENCE_LENGTH
    fb = predict.FrameBuffer()
    assert fb.length == SEQUENCE_LENGTH


# --------------------------------------------------------------------------- #
# AC5 : normalization delegation
# --------------------------------------------------------------------------- #
def test_frame_buffer_delegates_to_normalize_landmarks():
    feats = _hand_features(7)
    fb = predict.FrameBuffer(length=1)
    fb.add_frame([("Right", _feats_to_fake_landmarks(feats))])
    window = fb.window()
    expected = landmarks.normalize_landmarks(feats.reshape(1, N_PER_HAND))[0]
    np.testing.assert_array_equal(window[0, :N_PER_HAND], expected)
    assert np.all(window[0, N_PER_HAND:] == 0)


# --------------------------------------------------------------------------- #
# AC7 : public API surface + docstrings
# --------------------------------------------------------------------------- #
_REQUIRED_PUBLIC_NAMES = {
    "FrameBuffer",
    "PredictionSmoother",
    "MockPredictor",
    "RealPredictor",
    "SignRecognizer",
    "RecognitionResult",
    "create_recognizer",
    "N_FEATURES_PER_FRAME",
    "DEFAULT_CHECKPOINT_PATH",
    "DEFAULT_MIN_CONFIDENCE",
    "DEFAULT_STABILITY_FRAMES",
    "DEFAULT_COOLDOWN_FRAMES",
}


def test_public_api_surface_contains_required_names():
    public_names = {n for n in dir(predict) if not n.startswith("_")}
    missing = _REQUIRED_PUBLIC_NAMES - public_names
    assert not missing, f"missing required public names: {missing}"


def test_public_callables_and_classes_have_docstrings():
    for name in _REQUIRED_PUBLIC_NAMES:
        obj = getattr(predict, name)
        if callable(obj) or isinstance(obj, type):
            assert obj.__doc__ and obj.__doc__.strip(), f"{name} has no docstring"


# --------------------------------------------------------------------------- #
# AC8 : N_FEATURES_PER_FRAME derivation + equality with preprocess
# --------------------------------------------------------------------------- #
def test_n_features_per_frame_derived_and_126():
    assert predict.N_FEATURES_PER_FRAME == 2 * landmarks.N_FEATURES
    assert predict.N_FEATURES_PER_FRAME == 126


def test_n_features_per_frame_matches_preprocess():
    from src import preprocess  # test-side only

    assert predict.N_FEATURES_PER_FRAME == preprocess.N_FEATURES_PER_FRAME


# --------------------------------------------------------------------------- #
# AC9-AC13 : window construction, capacity, readiness, copy semantics, reset
# --------------------------------------------------------------------------- #
def test_frame_buffer_length_validation():
    with pytest.raises(ValueError):
        predict.FrameBuffer(length=0)


def test_frame_buffer_not_ready_until_full():
    fb = predict.FrameBuffer(length=3)
    assert not fb.is_ready
    assert fb.window() is None
    fb.add_frame(None)
    fb.add_frame(None)
    assert not fb.is_ready
    fb.add_frame(None)
    assert fb.is_ready
    assert fb.window() is not None


def test_frame_buffer_capacity_and_eviction_order():
    length = 5
    fb = predict.FrameBuffer(length=length)
    rows = []
    for i in range(length + 10):
        feats = _hand_features(100 + i)
        fb.add_frame([("Right", _feats_to_fake_landmarks(feats))])
        rows.append(feats)

    assert len(fb) == length
    window = fb.window()
    assert window.shape == (length, N_FRAME)

    expected_rows = rows[-length:]
    for i, feats in enumerate(expected_rows):
        expected = landmarks.normalize_landmarks(feats.reshape(1, N_PER_HAND))[0]
        np.testing.assert_array_equal(window[i, :N_PER_HAND], expected)


def test_frame_buffer_window_is_a_copy():
    fb = predict.FrameBuffer(length=2)
    fb.add_frame(None)
    fb.add_frame(None)
    window = fb.window()
    window[:] = 999.0
    fresh = fb.window()
    assert np.all(fresh == 0.0)


def test_frame_buffer_window_dtype_is_float32():
    fb = predict.FrameBuffer(length=1)
    fb.add_frame(None)
    assert fb.window().dtype == np.float32


def test_frame_buffer_reset():
    fb = predict.FrameBuffer(length=2)
    fb.add_frame(None)
    fb.add_frame(None)
    assert fb.is_ready
    fb.reset()
    assert len(fb) == 0
    assert not fb.is_ready
    assert fb.window() is None


def test_frame_buffer_none_and_empty_are_valid_no_hands_frames():
    fb = predict.FrameBuffer(length=2)
    fb.add_frame(None)
    fb.add_frame([])
    window = fb.window()
    assert np.all(window == 0.0)


# --------------------------------------------------------------------------- #
# AC14 : bitwise parity against src.preprocess
# --------------------------------------------------------------------------- #
def _parity_case(handedness_tuple):
    from src import preprocess  # test-side only

    # Exactly SEQUENCE_LENGTH frames: fix_length is then a no-op, matching
    # FrameBuffer's natural rolling-window behaviour. Feeding a surplus would
    # NOT be expected to match: fix_length center-trims a too-long recording
    # while FrameBuffer (a live buffer) always keeps the most recent frames;
    # those are deliberately different policies, not a parity requirement.
    n_frames = SEQUENCE_LENGTH
    rows = []
    fb = predict.FrameBuffer(length=SEQUENCE_LENGTH)
    for f in range(n_frames):
        hands = []
        for hi, hand in enumerate(handedness_tuple):
            feats = _hand_features(f * 10 + hi)
            rows.append(_row(f, hi, hand, feats))
            hands.append((hand, _feats_to_fake_landmarks(feats)))
        fb.add_frame(hands)

    fb_window = fb.window()

    meta, feats = _to_meta_and_feats(rows)
    if rows:
        seq = preprocess.frames_to_sequence(meta, feats)
        expected = preprocess.fix_length(seq, SEQUENCE_LENGTH)
    else:
        expected = np.zeros((SEQUENCE_LENGTH, N_FRAME), dtype=np.float32)

    return fb_window, expected


def test_parity_both_hands():
    fb_window, expected = _parity_case(("Right", "Left"))
    assert fb_window.dtype == np.float32 == expected.dtype
    assert np.array_equal(fb_window, expected)


def test_parity_right_only():
    fb_window, expected = _parity_case(("Right",))
    assert fb_window.dtype == np.float32 == expected.dtype
    assert np.array_equal(fb_window, expected)


def test_parity_left_only():
    fb_window, expected = _parity_case(("Left",))
    assert fb_window.dtype == np.float32 == expected.dtype
    assert np.array_equal(fb_window, expected)


def test_parity_no_hands():
    fb = predict.FrameBuffer(length=SEQUENCE_LENGTH)
    for _ in range(SEQUENCE_LENGTH):
        fb.add_frame(None)
    fb_window = fb.window()
    expected = np.zeros((SEQUENCE_LENGTH, N_FRAME), dtype=np.float32)
    assert fb_window.dtype == np.float32
    assert np.array_equal(fb_window, expected)


# --------------------------------------------------------------------------- #
# AC15 : live-robustness divergences
# --------------------------------------------------------------------------- #
def test_duplicate_handedness_first_wins_no_raise():
    feats_a = _hand_features(1)
    feats_b = _hand_features(2)
    fb = predict.FrameBuffer(length=1)
    fb.add_frame([
        ("Right", _feats_to_fake_landmarks(feats_a)),
        ("Right", _feats_to_fake_landmarks(feats_b)),
    ])
    window = fb.window()
    expected = landmarks.normalize_landmarks(feats_a.reshape(1, N_PER_HAND))[0]
    np.testing.assert_array_equal(window[0, :N_PER_HAND], expected)


def test_unknown_handedness_raises_value_error():
    feats = _hand_features(3)
    fb = predict.FrameBuffer(length=1)
    with pytest.raises(ValueError, match="right"):
        fb.add_frame([("right", _feats_to_fake_landmarks(feats))])


def test_wrong_length_landmark_payload_raises_value_error():
    fb = predict.FrameBuffer(length=1)
    with pytest.raises(ValueError, match="60"):
        fb.add_frame([("Right", list(_hand_features(4)[:60]))])


# --------------------------------------------------------------------------- #
# Smoothing state machine: AC17-AC24
# --------------------------------------------------------------------------- #
def test_smoothing_stability_three_sequence():
    smoother = predict.PredictionSmoother(stability_frames=3, cooldown_frames=0)
    results = [smoother.update("HELLO", 0.9) for _ in range(3)]
    assert results == [None, None, "HELLO"]


def test_smoothing_differing_gloss_restarts_run():
    smoother = predict.PredictionSmoother(stability_frames=3, cooldown_frames=0)
    seq = ["HELLO", "HELLO", "YES", "YES", "YES"]
    results = [smoother.update(g, 0.9) for g in seq]
    assert results == [None, None, None, None, "YES"]


def test_smoothing_cooldown_sequence_and_token_buffer_collapse():
    smoother = predict.PredictionSmoother(stability_frames=1, cooldown_frames=2)
    results = [smoother.update("HELLO", 0.9) for _ in range(4)]
    assert results == ["HELLO", None, None, "HELLO"]

    buf = TokenBuffer()
    for gloss in results:
        buf.add(gloss)
    assert buf.tokens == ["HELLO"]


def test_smoothing_reset_clears_state():
    smoother = predict.PredictionSmoother(stability_frames=2, cooldown_frames=5)
    smoother.update("HELLO", 0.9)
    smoother.reset()
    # After reset, a fresh run must start from 1, not continue at 2.
    assert smoother.update("HELLO", 0.9) is None
    assert smoother.update("HELLO", 0.9) == "HELLO"


def test_smoothing_threshold_equality_passes():
    smoother = predict.PredictionSmoother(
        min_confidence=0.5, stability_frames=1, cooldown_frames=0
    )
    assert smoother.update("HELLO", 0.5) == "HELLO"


def test_smoothing_below_threshold_resets_run():
    smoother = predict.PredictionSmoother(
        min_confidence=0.5, stability_frames=2, cooldown_frames=0
    )
    assert smoother.update("HELLO", 0.9) is None
    assert smoother.update("HELLO", 0.1) is None  # resets run
    assert smoother.update("HELLO", 0.9) is None  # run restarts at 1
    assert smoother.update("HELLO", 0.9) == "HELLO"


def test_smoothing_none_and_nan_confidence_tolerated():
    smoother = predict.PredictionSmoother(stability_frames=1, cooldown_frames=0)
    assert smoother.update("HELLO", None) is None
    assert smoother.update("HELLO", float("nan")) is None
    assert smoother.update(None, 0.9) is None
    assert smoother.update("HELLO", 0.9) == "HELLO"


def test_smoothing_confidence_above_one_treated_as_passing():
    smoother = predict.PredictionSmoother(stability_frames=1, cooldown_frames=0)
    assert smoother.update("HELLO", 1.5) == "HELLO"


def test_smoothing_constructor_validation():
    with pytest.raises(ValueError, match="min_confidence"):
        predict.PredictionSmoother(min_confidence=-0.1)
    with pytest.raises(ValueError, match="min_confidence"):
        predict.PredictionSmoother(min_confidence=1.1)
    with pytest.raises(ValueError, match="stability_frames"):
        predict.PredictionSmoother(stability_frames=0)
    with pytest.raises(ValueError, match="cooldown_frames"):
        predict.PredictionSmoother(cooldown_frames=-1)


def test_smoothing_boundary_configurations_are_valid():
    smoother = predict.PredictionSmoother(stability_frames=1, cooldown_frames=0)
    assert smoother.update("HELLO", 0.9) == "HELLO"
    # cooldown_frames=0 means no refractory period: immediately eligible again.
    assert smoother.update("HELLO", 0.9) == "HELLO"


def test_smoothing_alternating_glosses_never_emit():
    smoother = predict.PredictionSmoother(stability_frames=3, cooldown_frames=0)
    results = [smoother.update(g, 0.9) for g in ["HELLO", "YES", "HELLO", "YES"] * 3]
    assert all(r is None for r in results)


# --------------------------------------------------------------------------- #
# Predictors: shape validation, mock determinism/purity
# --------------------------------------------------------------------------- #
def _valid_window(seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, size=(SEQUENCE_LENGTH, N_FRAME)).astype(np.float32)


def test_predict_raises_on_wrong_shape():
    mock = predict.MockPredictor()
    with pytest.raises(ValueError, match="shape"):
        mock.predict(np.zeros((SEQUENCE_LENGTH, N_FRAME - 1), dtype=np.float32))


def test_mock_predictor_deterministic_repeat_calls():
    mock = predict.MockPredictor()
    window = _valid_window(1)
    first = mock.predict(window)
    second = mock.predict(window)
    assert first == second


def test_mock_predictor_pure_function_of_content():
    mock = predict.MockPredictor()
    w1 = _valid_window(1)
    w2 = _valid_window(2)
    r1 = mock.predict(w1)
    r2 = mock.predict(w1.copy())
    assert r1 == r2
    r3 = mock.predict(w2)
    assert isinstance(r3, tuple)


def test_mock_predictor_glosses_in_vocabulary():
    mock = predict.MockPredictor()
    for seed in range(10):
        gloss, confidence = mock.predict(_valid_window(seed))
        assert gloss in VOCABULARY
        assert 0.0 <= confidence <= 1.0
        assert confidence >= predict.DEFAULT_MIN_CONFIDENCE


def test_mock_predictor_exposes_predictor_protocol():
    mock = predict.MockPredictor()
    assert hasattr(mock, "predict")
    assert hasattr(mock, "labels")
    assert mock.last_latency_ms is None
    mock.predict(_valid_window(0))
    assert isinstance(mock.last_latency_ms, float)
    assert mock.last_latency_ms >= 0.0
    labels = mock.labels
    labels.append("SHOULD_NOT_PERSIST")
    assert "SHOULD_NOT_PERSIST" not in mock.labels


def test_real_predictor_exposes_predictor_protocol_names():
    assert hasattr(predict.RealPredictor, "predict")


# --------------------------------------------------------------------------- #
# RealPredictor: checkpoint-absent (torch absent-safe)
# --------------------------------------------------------------------------- #
def test_real_predictor_missing_checkpoint_raises_file_not_found(tmp_path):
    missing = tmp_path / "does_not_exist.pt"
    with pytest.raises(FileNotFoundError) as excinfo:
        predict.RealPredictor(checkpoint_path=missing)
    assert str(missing.resolve()) in str(excinfo.value)


def test_default_checkpoint_path_basename():
    assert predict.DEFAULT_CHECKPOINT_PATH.name == "sign_classifier.pt"
    assert "label_encoder" not in str(predict.DEFAULT_CHECKPOINT_PATH)


def test_real_predictor_directory_path_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        predict.RealPredictor(checkpoint_path=tmp_path)


# --------------------------------------------------------------------------- #
# Torch-free checkpoint validators
# --------------------------------------------------------------------------- #
def _valid_payload():
    return {
        "state_dict": {"weight": [1, 2, 3]},
        "architecture": "LSTM",
        "hyperparams": {"hidden_size": 64},
        "input_shape": (SEQUENCE_LENGTH, N_FRAME),
        "num_classes": len(VOCABULARY),
        "vocabulary": list(VOCABULARY),
    }


def test_validate_checkpoint_payload_accepts_valid():
    predict._validate_checkpoint_payload(_valid_payload())  # must not raise


def test_validate_checkpoint_payload_rejects_non_mapping():
    with pytest.raises(ValueError, match="mapping"):
        predict._validate_checkpoint_payload(["not", "a", "mapping"])


def test_validate_checkpoint_payload_rejects_missing_keys():
    payload = _valid_payload()
    del payload["state_dict"]
    del payload["num_classes"]
    with pytest.raises(ValueError) as excinfo:
        predict._validate_checkpoint_payload(payload)
    assert "state_dict" in str(excinfo.value)
    assert "num_classes" in str(excinfo.value)


def test_validate_checkpoint_semantics_class_count_mismatch():
    payload = _valid_payload()
    payload["num_classes"] = len(VOCABULARY) + 1
    with pytest.raises(ValueError, match="num_classes"):
        predict._validate_checkpoint_semantics(payload)


def test_validate_checkpoint_semantics_input_shape_mismatch():
    payload = _valid_payload()
    payload["input_shape"] = (SEQUENCE_LENGTH, N_FRAME + 1)
    with pytest.raises(ValueError, match="input_shape"):
        predict._validate_checkpoint_semantics(payload)


def test_validate_checkpoint_semantics_vocabulary_drift_warns():
    payload = _valid_payload()
    drifted = list(VOCABULARY)
    drifted[0], drifted[1] = drifted[1], drifted[0]
    payload["vocabulary"] = drifted
    with pytest.warns(UserWarning, match="vocabulary"):
        predict._validate_checkpoint_semantics(payload)


def test_validate_checkpoint_semantics_valid_no_warning():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        predict._validate_checkpoint_semantics(_valid_payload())  # must not warn/raise


# --------------------------------------------------------------------------- #
# SignRecognizer: statuses, latency, reset
# --------------------------------------------------------------------------- #
def _hands_frame(seed):
    feats = _hand_features(seed)
    return [("Right", _feats_to_fake_landmarks(feats))]


def test_recognizer_warming_up_then_predicted():
    recognizer = predict.SignRecognizer(predict.MockPredictor())
    for i in range(SEQUENCE_LENGTH - 1):
        result = recognizer.update(_hands_frame(i))
        assert result.status == "warming_up"
        assert result.gloss is None
        assert result.confidence is None
        assert result.latency_ms is None
        assert result.emitted is None
        assert result.window_ready is False
        assert result.frames_buffered == i + 1

    result = recognizer.update(_hands_frame(SEQUENCE_LENGTH))
    assert result.status == "predicted"
    assert result.window_ready is True
    assert isinstance(result.latency_ms, float)
    assert result.latency_ms >= 0.0
    assert recognizer.last_latency_ms == result.latency_ms


def test_recognizer_no_hands_status_and_no_emission():
    # Use stability_frames=1 so the mock would emit immediately if inference ran.
    recognizer = predict.SignRecognizer(
        predict.MockPredictor(), stability_frames=1, cooldown_frames=0
    )
    emissions = []
    for _ in range(SEQUENCE_LENGTH + 40):
        result = recognizer.update(None)
        emissions.append(result.emitted)
        if result.window_ready:
            assert result.status == "no_hands"
    assert all(e is None for e in emissions)


def test_recognizer_reset_returns_to_warming_up():
    recognizer = predict.SignRecognizer(
        predict.MockPredictor(), stability_frames=1, cooldown_frames=0
    )
    for i in range(SEQUENCE_LENGTH):
        recognizer.update(_hands_frame(i))
    recognizer.reset()
    result = recognizer.update(_hands_frame(0))
    assert result.status == "warming_up"
    assert result.frames_buffered == 1


# --------------------------------------------------------------------------- #
# create_recognizer
# --------------------------------------------------------------------------- #
def test_create_recognizer_mock_end_to_end_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recognizer = predict.create_recognizer(
        use_mock=True, stability_frames=1, cooldown_frames=0
    )
    emitted_any = False
    for i in range(SEQUENCE_LENGTH + 5):
        result = recognizer.update(_hands_frame(0))  # identical frame each time
        if result.emitted is not None:
            emitted_any = True
    assert emitted_any


def test_create_recognizer_real_missing_checkpoint_raises(tmp_path):
    missing = tmp_path / "no_such_checkpoint.pt"
    with pytest.raises(FileNotFoundError):
        predict.create_recognizer(use_mock=False, checkpoint_path=missing)


# --------------------------------------------------------------------------- #
# RealPredictor: real checkpoint round-trip (torch + src.train required)
# --------------------------------------------------------------------------- #
# Each test below builds a genuine checkpoint via src.train.make_model /
# src.train.save_checkpoint and loads it through predict.RealPredictor, all
# inside the child process spawned by _run_in_subprocess (see the
# module-level comment near _TRAIN_STACK_AVAILABLE for why). Shapes are
# derived from SEQUENCE_LENGTH, preprocess.N_FEATURES_PER_FRAME and
# VOCABULARY inside the child script, never hardcoded as 30 / 126 / 25.
_CHECKPOINT_PREAMBLE = """
import numpy as np

from src import predict, preprocess, train
from src.utils import SEQUENCE_LENGTH, VOCABULARY

seq_len = SEQUENCE_LENGTH
n_features = preprocess.N_FEATURES_PER_FRAME
n_classes = len(VOCABULARY)

model = train.make_model("mlp", seq_len, n_features, n_classes, hidden_size=4)
checkpoint_path = train.save_checkpoint(
    model, "mlp", seq_len, n_features, n_classes, {checkpoint_path!r},
)
window = np.random.default_rng(0).uniform(
    -1, 1, size=(seq_len, n_features)
).astype("float32")
"""


@_skip_without_train_stack
def test_real_predictor_round_trip_zero_config_model_builder(tmp_path):
    """Regression test for the model-factory defect: the zero-config path
    (no explicit model_builder) must resolve a working factory from
    src.train. Fails against the old code, which targeted a nonexistent
    src.train.build_model symbol and could never construct a RealPredictor.
    """
    checkpoint_path = str(tmp_path / "checkpoint.pt")
    script = _CHECKPOINT_PREAMBLE.format(checkpoint_path=checkpoint_path) + """
import json

real = predict.RealPredictor(checkpoint_path=checkpoint_path)
gloss, confidence = real.predict(window)

print(json.dumps({
    "labels": real.labels,
    "gloss": gloss,
    "confidence": confidence,
    "last_latency_ms": real.last_latency_ms,
}))
"""
    result = _run_in_subprocess(script)
    data = json.loads(result.stdout.strip().splitlines()[-1])

    assert data["labels"] == list(VOCABULARY)
    assert data["gloss"] in VOCABULARY
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["last_latency_ms"], float)
    assert data["last_latency_ms"] >= 0.0


@_skip_without_train_stack
def test_real_predictor_explicit_model_builder_is_honored(tmp_path):
    checkpoint_path = str(tmp_path / "checkpoint.pt")
    marker_path = str(tmp_path / "builder_called.marker")
    script = _CHECKPOINT_PREAMBLE.format(checkpoint_path=checkpoint_path) + f"""
import json
from pathlib import Path

marker_path = Path({marker_path!r})

def spy_builder(checkpoint):
    marker_path.write_text("called")
    return train.model_from_checkpoint(checkpoint)

real = predict.RealPredictor(checkpoint_path=checkpoint_path, model_builder=spy_builder)
gloss, confidence = real.predict(window)

print(json.dumps({{"gloss": gloss, "confidence": confidence}}))
"""
    result = _run_in_subprocess(script)
    data = json.loads(result.stdout.strip().splitlines()[-1])

    assert Path(marker_path).exists(), "explicit model_builder was never called"
    assert data["gloss"] in VOCABULARY
    assert 0.0 <= data["confidence"] <= 1.0


@_skip_without_train_stack
def test_create_recognizer_real_forwards_model_builder(tmp_path):
    checkpoint_path = str(tmp_path / "checkpoint.pt")
    marker_path = str(tmp_path / "builder_called.marker")
    script = _CHECKPOINT_PREAMBLE.format(checkpoint_path=checkpoint_path) + f"""
import json
from pathlib import Path

marker_path = Path({marker_path!r})

def spy_builder(checkpoint):
    marker_path.write_text("called")
    return train.model_from_checkpoint(checkpoint)

recognizer = predict.create_recognizer(
    use_mock=False, checkpoint_path=checkpoint_path, model_builder=spy_builder,
    stability_frames=1, cooldown_frames=0,
)

print(json.dumps({{"recognizer_type": type(recognizer).__name__}}))
"""
    result = _run_in_subprocess(script)
    data = json.loads(result.stdout.strip().splitlines()[-1])

    assert Path(marker_path).exists(), "explicit model_builder was never called"
    assert data["recognizer_type"] == "SignRecognizer"


def test_create_recognizer_mock_ignores_model_builder():
    def exploding_builder(checkpoint):
        raise AssertionError("model_builder must not be called in mock mode")

    recognizer = predict.create_recognizer(use_mock=True, model_builder=exploding_builder)
    assert isinstance(recognizer, predict.SignRecognizer)


# --------------------------------------------------------------------------- #
# End-to-end: mock -> TokenBuffer -> glosses_to_sentence
# --------------------------------------------------------------------------- #
def test_end_to_end_mock_to_buffer_to_grammar():
    recognizer = predict.create_recognizer(
        use_mock=True, stability_frames=1, cooldown_frames=0
    )
    buf = TokenBuffer()
    for i in range(SEQUENCE_LENGTH + 10):
        result = recognizer.update(_hands_frame(0))
        if result.emitted is not None:
            buf.add(result.emitted)

    sentence = glosses_to_sentence(buf.emit())
    assert isinstance(sentence, str)
    assert sentence != ""
