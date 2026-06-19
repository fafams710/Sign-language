"""Unit tests for src.preprocess.

All tests run on synthetic NumPy arrays and temporary directories only. They
require no webcam, no MediaPipe / OpenCV install, no torch, and no
``models/hand_landmarker.task`` binary. Run with: ``pytest tests/``.
"""

import csv
import importlib
import sys
import warnings

import numpy as np
import pytest

from src import landmarks, preprocess
from src.utils import SEQUENCE_LENGTH, VOCABULARY

N_PER_HAND = landmarks.N_FEATURES  # 63
N_FRAME = preprocess.N_FEATURES_PER_FRAME  # 126

CSV_HEADER = ["timestamp", "frame", "hand_index", "handedness"] + [
    f"{name}_{ax}" for name in landmarks.LANDMARK_NAMES for ax in ("x", "y", "z")
]


# --------------------------------------------------------------------------- #
# Synthetic data helpers
# --------------------------------------------------------------------------- #
def _hand_features(seed):
    """Deterministic nonzero 63-vector with a nonzero wrist so normalization
    yields a clearly nonzero result."""
    rng = np.random.default_rng(seed)
    feats = rng.uniform(0.1, 0.9, size=N_PER_HAND).astype(np.float32)
    return feats


def _row(frame, hand_index, handedness, feats):
    return {"meta": {"timestamp": 0.0, "frame": frame,
                     "hand_index": hand_index, "handedness": handedness},
            "feats": feats}


def _to_meta_and_feats(rows):
    meta = [r["meta"] for r in rows]
    feats = np.stack([r["feats"] for r in rows], axis=0).astype(np.float32)
    return meta, feats


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for r in rows:
            m = r["meta"]
            line = [m["timestamp"], m["frame"], m["hand_index"], m["handedness"]]
            line.extend(float(v) for v in r["feats"])
            w.writerow(line)


def _make_recording_rows(n_frames, handedness=("Right", "Left"), seed=0):
    """Build rows for a recording: each frame has the listed hands."""
    rows = []
    for f in range(n_frames):
        for hi, hand in enumerate(handedness):
            rows.append(_row(f, hi, hand, _hand_features(seed + f * 10 + hi)))
    return rows


# --------------------------------------------------------------------------- #
# AC2 / AC3 : import safety, torch absent
# --------------------------------------------------------------------------- #
def test_import_succeeds_and_returns_numpy():
    mod = importlib.import_module("src.preprocess")
    assert mod is not None
    assert isinstance(mod.N_FEATURES_PER_FRAME, int)
    assert mod.N_FEATURES_PER_FRAME == 126


def test_torch_not_imported_by_preprocess():
    # torch must not be pulled in transitively by importing src.preprocess.
    importlib.import_module("src.preprocess")
    assert "torch" not in sys.modules, (
        "src.preprocess (or its imports) imported torch; it must not."
    )


def test_mediapipe_and_cv2_not_required():
    # Importing the module must not require mediapipe / cv2 to be present.
    importlib.import_module("src.preprocess")
    # If they happen to be installed that's fine; the assertion is only that the
    # import above succeeded without raising, which pytest collection guarantees.


def test_n_features_per_frame_is_derived():
    assert preprocess.N_FEATURES_PER_FRAME == 2 * landmarks.N_FEATURES


# --------------------------------------------------------------------------- #
# AC4 : SEQUENCE_LENGTH single source
# --------------------------------------------------------------------------- #
def test_sequence_length_constant():
    assert SEQUENCE_LENGTH == 30


# --------------------------------------------------------------------------- #
# AC6 / AC7 : [Right|Left] collapse, normalization reuse, missing/zero hands
# --------------------------------------------------------------------------- #
def test_frames_to_sequence_both_hands_layout():
    rows = _make_recording_rows(1, handedness=("Right", "Left"))
    meta, feats = _to_meta_and_feats(rows)
    seq = preprocess.frames_to_sequence(meta, feats)
    assert seq.shape == (1, N_FRAME)
    assert seq.dtype == np.float32
    assert np.any(seq[0, :N_PER_HAND] != 0)       # Right slot populated
    assert np.any(seq[0, N_PER_HAND:] != 0)       # Left slot populated


def test_frames_to_sequence_right_only():
    rows = _make_recording_rows(1, handedness=("Right",))
    meta, feats = _to_meta_and_feats(rows)
    seq = preprocess.frames_to_sequence(meta, feats)
    assert np.any(seq[0, :N_PER_HAND] != 0)
    assert np.all(seq[0, N_PER_HAND:] == 0)


def test_frames_to_sequence_left_only():
    rows = _make_recording_rows(1, handedness=("Left",))
    meta, feats = _to_meta_and_feats(rows)
    seq = preprocess.frames_to_sequence(meta, feats)
    assert np.all(seq[0, :N_PER_HAND] == 0)
    assert np.any(seq[0, N_PER_HAND:] != 0)


def test_frames_to_sequence_normalization_reuse():
    feats = _hand_features(123)
    rows = [_row(0, 0, "Right", feats)]
    meta, feat_arr = _to_meta_and_feats(rows)
    seq = preprocess.frames_to_sequence(meta, feat_arr)
    expected = landmarks.normalize_landmarks(feats.reshape(1, N_PER_HAND))[0]
    np.testing.assert_array_equal(seq[0, :N_PER_HAND], expected)


def test_frames_to_sequence_zero_hand_frame():
    # Frame 0 has a hand, frame 1 has no rows -> only frame 0 appears.
    rows = [_row(0, 0, "Right", _hand_features(1))]
    meta, feats = _to_meta_and_feats(rows)
    seq = preprocess.frames_to_sequence(meta, feats)
    assert seq.shape == (1, N_FRAME)


def test_frames_to_sequence_unknown_handedness_errors():
    rows = [_row(0, 0, "RIGHT", _hand_features(1))]
    meta, feats = _to_meta_and_feats(rows)
    with pytest.raises(ValueError, match="handedness"):
        preprocess.frames_to_sequence(meta, feats)


def test_frames_to_sequence_duplicate_handedness_errors():
    rows = [_row(0, 0, "Right", _hand_features(1)),
            _row(0, 1, "Right", _hand_features(2))]
    meta, feats = _to_meta_and_feats(rows)
    with pytest.raises(ValueError, match="two rows"):
        preprocess.frames_to_sequence(meta, feats)


def test_frames_to_sequence_orders_by_frame():
    # Rows out of order; output must be ascending by frame.
    rows = [_row(2, 0, "Right", _hand_features(2)),
            _row(0, 0, "Right", _hand_features(0)),
            _row(1, 0, "Right", _hand_features(1))]
    meta, feats = _to_meta_and_feats(rows)
    seq = preprocess.frames_to_sequence(meta, feats)
    assert seq.shape == (3, N_FRAME)
    exp0 = landmarks.normalize_landmarks(
        _hand_features(0).reshape(1, N_PER_HAND))[0]
    np.testing.assert_array_equal(seq[0, :N_PER_HAND], exp0)


# --------------------------------------------------------------------------- #
# AC9 : fix_length center-trim / edge-repeat / passthrough
# --------------------------------------------------------------------------- #
def test_fix_length_passthrough():
    seq = np.ones((30, N_FRAME), dtype=np.float32)
    out = preprocess.fix_length(seq)
    assert out.shape == (30, N_FRAME)
    np.testing.assert_array_equal(out, seq)


def test_fix_length_edge_repeat_post_pad():
    seq = np.arange(10 * N_FRAME, dtype=np.float32).reshape(10, N_FRAME)
    out = preprocess.fix_length(seq)
    assert out.shape == (30, N_FRAME)
    # Original 10 frames preserved at the front.
    np.testing.assert_array_equal(out[:10], seq)
    # Remaining frames are the repeated LAST frame (post-pad).
    for i in range(10, 30):
        np.testing.assert_array_equal(out[i], seq[-1])


def test_fix_length_center_trim_even_surplus():
    seq = np.arange(50 * N_FRAME, dtype=np.float32).reshape(50, N_FRAME)
    out = preprocess.fix_length(seq)
    assert out.shape == (30, N_FRAME)
    # surplus = 20 (even) -> start = 10.
    np.testing.assert_array_equal(out, seq[10:40])


def test_fix_length_center_trim_odd_surplus_front_drop():
    seq = np.arange(33 * N_FRAME, dtype=np.float32).reshape(33, N_FRAME)
    out = preprocess.fix_length(seq)
    assert out.shape == (30, N_FRAME)
    # surplus = 3 (odd) -> start = (3+1)//2 = 2 (extra frame dropped from front).
    np.testing.assert_array_equal(out, seq[2:32])


def test_fix_length_empty_errors():
    with pytest.raises(ValueError):
        preprocess.fix_length(np.zeros((0, N_FRAME), dtype=np.float32))


# --------------------------------------------------------------------------- #
# AC5 : load_recording (CSV canonical) + empty-file error
# --------------------------------------------------------------------------- #
def test_load_recording_csv(tmp_path):
    rows = _make_recording_rows(3)
    p = tmp_path / "rec.csv"
    _write_csv(p, rows)
    meta, feats = preprocess.load_recording(p)
    assert feats.shape == (6, N_PER_HAND)  # 3 frames x 2 hands
    assert feats.dtype == np.float32
    assert len(meta) == 6


def test_load_recording_empty_csv_errors(tmp_path):
    p = tmp_path / "empty.csv"
    with open(p, "w", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)
    with pytest.raises(ValueError, match="no data rows"):
        preprocess.load_recording(p)


# --------------------------------------------------------------------------- #
# AC10 / AC11 : directory-per-label discovery, VOCABULARY validation + indexing
# --------------------------------------------------------------------------- #
def _make_dataset(root, spec, seed_base=0):
    """spec: dict label -> n_recordings. Each recording has 5 two-hand frames."""
    for i, (label, n_rec) in enumerate(spec.items()):
        d = root / label
        d.mkdir(parents=True, exist_ok=True)
        for j in range(n_rec):
            rows = _make_recording_rows(5, seed=seed_base + i * 100 + j * 10)
            _write_csv(d / f"rec_{j}.csv", rows)


def test_discover_recordings_basic(tmp_path):
    _make_dataset(tmp_path, {"HELLO": 2, "YES": 1})
    pairs = preprocess.discover_recordings(tmp_path)
    labels = [lbl for _, lbl in pairs]
    assert labels.count("HELLO") == 2
    assert labels.count("YES") == 1
    # Sorted deterministically by (label, path).
    assert labels == sorted(labels)


def test_discover_unknown_label_errors(tmp_path):
    _make_dataset(tmp_path, {"HELLO": 1})
    bad = tmp_path / "NOTASIGN"
    bad.mkdir()
    _write_csv(bad / "rec.csv", _make_recording_rows(5))
    with pytest.raises(ValueError, match="VOCABULARY"):
        preprocess.discover_recordings(tmp_path)


def test_discover_empty_root_errors(tmp_path):
    with pytest.raises(ValueError, match="no <SIGN> subdirectories"):
        preprocess.discover_recordings(tmp_path)


def test_discover_stray_root_file_errors(tmp_path):
    (tmp_path / "HELLO").mkdir()
    _write_csv(tmp_path / "HELLO" / "rec.csv", _make_recording_rows(5))
    _write_csv(tmp_path / "stray.csv", _make_recording_rows(5))
    with pytest.raises(ValueError, match="directly in the dataset root"):
        preprocess.discover_recordings(tmp_path)


def test_discover_missing_root_errors(tmp_path):
    with pytest.raises(NotADirectoryError):
        preprocess.discover_recordings(tmp_path / "does_not_exist")


def test_build_dataset_label_indexing(tmp_path):
    _make_dataset(tmp_path, {"HELLO": 1, "YES": 1})
    X, y = preprocess.build_dataset(tmp_path)
    assert set(y.tolist()) == {VOCABULARY.index("HELLO"),
                               VOCABULARY.index("YES")}


# --------------------------------------------------------------------------- #
# AC12 / AC13 : output shapes/dtypes + seed reproducibility
# --------------------------------------------------------------------------- #
def test_build_dataset_shapes_dtypes(tmp_path):
    _make_dataset(tmp_path, {"HELLO": 3, "YES": 2})
    X, y = preprocess.build_dataset(tmp_path)
    assert X.shape == (5, SEQUENCE_LENGTH, N_FRAME)
    assert X.dtype == np.float32
    assert y.shape == (5,)
    assert np.issubdtype(y.dtype, np.integer)


def test_split_dataset_shapes(tmp_path):
    _make_dataset(tmp_path, {"HELLO": 6, "YES": 6})
    parts = preprocess.split_dataset(tmp_path)
    X_tr, y_tr, X_val, y_val, X_te, y_te = parts
    assert X_tr.dtype == np.float32
    assert np.issubdtype(y_tr.dtype, np.integer)
    assert X_tr.shape[1:] == (SEQUENCE_LENGTH, N_FRAME)
    total = len(y_tr) + len(y_val) + len(y_te)
    assert total == 12


def test_split_dataset_reproducible(tmp_path):
    _make_dataset(tmp_path, {"HELLO": 6, "YES": 6})
    a = preprocess.split_dataset(tmp_path, seed=42)
    b = preprocess.split_dataset(tmp_path, seed=42)
    for arr_a, arr_b in zip(a, b):
        np.testing.assert_array_equal(arr_a, arr_b)


# --------------------------------------------------------------------------- #
# AC14 : stratify vs fallback branch + warning
# --------------------------------------------------------------------------- #
def test_split_dataset_stratified_no_warning(tmp_path):
    _make_dataset(tmp_path, {"HELLO": 4, "YES": 4})
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        preprocess.split_dataset(tmp_path)
    # No fallback warning expected when every class has >= 2 samples.
    msgs = [str(w.message) for w in record]
    assert not any("non-stratified" in m for m in msgs)


def test_split_dataset_fallback_warns(tmp_path):
    # One class with a single sample triggers the non-stratified fallback.
    _make_dataset(tmp_path, {"HELLO": 4, "YES": 1})
    with pytest.warns(UserWarning, match="non-stratified"):
        preprocess.split_dataset(tmp_path)
