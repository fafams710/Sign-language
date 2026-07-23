"""Unit tests for src.train.

Torch-guarded: this whole module is skipped when torch is not installed, so
the existing torch-free suite (including ``tests/test_preprocess.py``, which
is not modified) still passes with torch absent. All tests use tiny
synthetic contract-shaped arrays and temporary directories/paths only;
nothing is written into the repo tree permanently and nothing is committed
or pushed.

The torch-availability check uses ``importlib.util.find_spec`` (which
locates a module WITHOUT importing it) rather than a bare module-level
``pytest.importorskip("torch")`` / ``import torch``. This matters because
pytest's collection phase imports every test module (including this one)
before running any test in the session; a module-level ``import torch``
here would pollute ``sys.modules`` with ``torch`` before
``tests/test_preprocess.py``'s own ``test_torch_not_imported_by_preprocess``
gets a chance to run, breaking that unrelated, untouched test purely due to
import ordering. Deferring the actual ``import torch`` (and
``from src import train``, which itself imports torch) to inside fixtures
that only run when a test in this module actually executes avoids that
cross-file contamination while still skipping the whole module cleanly when
torch is unavailable -- an "equivalent" of ``pytest.importorskip`` per the
spec.
"""

import importlib.util

import numpy as np
import pytest

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

pytestmark = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="torch is not installed"
)


@pytest.fixture(scope="module")
def torch_mod():
    """Lazily import torch (only once a test in this module actually runs)."""
    import torch

    return torch


@pytest.fixture(scope="module")
def train_mod():
    """Lazily import src.train (pulls in torch transitively)."""
    from src import train

    return train


# --------------------------------------------------------------------------- #
# Synthetic data helpers
# --------------------------------------------------------------------------- #
def _synthetic_xy(train, n, n_classes, seed=0):
    """Tiny synthetic contract-shaped (X, y): float32 (n, 30, 126), int64
    VOCABULARY indices in [0, n_classes)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(
        -1.0, 1.0, size=(n, train.SEQUENCE_LENGTH, train.N_FEATURES_PER_FRAME)
    ).astype(np.float32)
    y = (np.arange(n) % max(n_classes, 1)).astype(np.int64)
    return X, y


# --------------------------------------------------------------------------- #
# (a) import safety
# --------------------------------------------------------------------------- #
def test_import_succeeds_with_no_dataset_or_task_binary(train_mod):
    assert train_mod is not None
    assert isinstance(train_mod.N_FEATURES_PER_FRAME, int)


def test_shape_constants_match_utils_and_preprocess(train_mod):
    # Shape logic derives from src.utils / src.preprocess, not bare literals.
    from src import preprocess
    from src.utils import SEQUENCE_LENGTH, VOCABULARY

    assert train_mod.SEQUENCE_LENGTH == SEQUENCE_LENGTH
    assert train_mod.N_FEATURES_PER_FRAME == preprocess.N_FEATURES_PER_FRAME
    assert len(VOCABULARY) == len(train_mod.VOCABULARY)


# --------------------------------------------------------------------------- #
# Model builders / registry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arch", ["lstm", "gru", "cnn", "mlp"])
def test_make_model_output_shape(torch_mod, train_mod, arch):
    n_classes = len(train_mod.VOCABULARY)
    model = train_mod.make_model(
        arch, train_mod.SEQUENCE_LENGTH, train_mod.N_FEATURES_PER_FRAME, n_classes
    )
    x = torch_mod.zeros(
        (3, train_mod.SEQUENCE_LENGTH, train_mod.N_FEATURES_PER_FRAME),
        dtype=torch_mod.float32,
    )
    logits = model(x)
    assert logits.shape == (3, n_classes)


def test_make_model_unknown_arch_raises(train_mod):
    n_classes = len(train_mod.VOCABULARY)
    with pytest.raises(ValueError):
        train_mod.make_model(
            "not-a-real-arch", train_mod.SEQUENCE_LENGTH,
            train_mod.N_FEATURES_PER_FRAME, n_classes,
        )


# --------------------------------------------------------------------------- #
# (b)/(c) train -> select -> save -> reload
# --------------------------------------------------------------------------- #
def test_train_select_save_reload_roundtrip(torch_mod, train_mod, tmp_path):
    n_classes = len(train_mod.VOCABULARY)
    seq_len = train_mod.SEQUENCE_LENGTH
    n_features = train_mod.N_FEATURES_PER_FRAME

    X_train, y_train = _synthetic_xy(train_mod, 8, n_classes, seed=1)
    X_val, y_val = _synthetic_xy(train_mod, 4, n_classes, seed=2)

    result = train_mod.train_and_select(
        ["lstm", "gru", "cnn", "mlp"], X_train, y_train, X_val, y_val,
        seq_len, n_features, n_classes,
        epochs=1, batch_size=4, lr=1e-3, device="cpu",
    )
    assert result["selected_arch"] in {"lstm", "gru", "cnn", "mlp"}
    assert result["val_skipped"] is False
    for name in ("lstm", "gru", "cnn", "mlp"):
        assert result["val_accuracies"][name] is not None

    out_path = tmp_path / "sign_classifier.pt"
    written = train_mod.save_checkpoint(
        result["selected_model"], result["selected_arch"], seq_len,
        n_features, n_classes, out_path,
    )
    assert written.exists()

    checkpoint = train_mod.load_checkpoint(written, map_location="cpu")
    for key in (
        "state_dict", "architecture", "hyperparams", "input_shape",
        "num_classes", "vocabulary",
    ):
        assert key in checkpoint
    assert checkpoint["input_shape"] == (seq_len, n_features)
    assert checkpoint["num_classes"] == n_classes
    assert checkpoint["vocabulary"] == list(train_mod.VOCABULARY)

    rebuilt = train_mod.model_from_checkpoint(checkpoint)
    logits = rebuilt(torch_mod.from_numpy(X_val))
    assert logits.shape == (X_val.shape[0], n_classes)


def test_label_encoder_pickle_is_plain_vocabulary_list(train_mod, tmp_path):
    out = tmp_path / "label_encoder.pkl"
    written = train_mod.save_label_encoder(out)
    assert written == out
    import pickle

    with open(out, "rb") as f:
        obj = pickle.load(f)
    assert obj == list(train_mod.VOCABULARY)
    assert isinstance(obj, list)
    assert not hasattr(obj, "classes_")  # never an sklearn LabelEncoder


# --------------------------------------------------------------------------- #
# (d) empty-val fallback
# --------------------------------------------------------------------------- #
def test_empty_val_fallback_reports_skipped(train_mod):
    n_classes = len(train_mod.VOCABULARY)
    X_train, y_train = _synthetic_xy(train_mod, 6, n_classes, seed=3)
    X_val = X_train[:0]
    y_val = y_train[:0]

    result = train_mod.train_and_select(
        ["lstm", "gru"], X_train, y_train, X_val, y_val,
        train_mod.SEQUENCE_LENGTH, train_mod.N_FEATURES_PER_FRAME, n_classes,
        epochs=1, batch_size=4, lr=1e-3, device="cpu",
    )
    assert result["val_skipped"] is True
    assert all(v is None for v in result["val_accuracies"].values())
    assert result["selected_arch"] in {"lstm", "gru"}


def test_evaluate_accuracy_empty_partition_is_none(train_mod):
    n_classes = len(train_mod.VOCABULARY)
    model = train_mod.make_model(
        "mlp", train_mod.SEQUENCE_LENGTH, train_mod.N_FEATURES_PER_FRAME, n_classes
    )
    X_empty = np.zeros(
        (0, train_mod.SEQUENCE_LENGTH, train_mod.N_FEATURES_PER_FRAME),
        dtype=np.float32,
    )
    y_empty = np.zeros((0,), dtype=np.int64)
    assert train_mod.evaluate_accuracy(model, X_empty, y_empty, "cpu") is None


# --------------------------------------------------------------------------- #
# (e) small-n robustness
# --------------------------------------------------------------------------- #
def test_n_equals_2_batch_larger_than_n(train_mod):
    X_train, y_train = _synthetic_xy(train_mod, 2, 2, seed=4)
    model = train_mod.make_model(
        "mlp", train_mod.SEQUENCE_LENGTH, train_mod.N_FEATURES_PER_FRAME, 2
    )
    # batch_size (16) > n (2): must not crash and must not drop the sole batch.
    train_mod.train_one_model(model, X_train, y_train, epochs=1, batch_size=16,
                              lr=1e-3, device="cpu")
    acc = train_mod.evaluate_accuracy(model, X_train, y_train, "cpu", batch_size=16)
    assert acc is not None


def test_single_class_train_partition_does_not_crash(train_mod):
    n_classes = len(train_mod.VOCABULARY)
    X_train, y_train = _synthetic_xy(train_mod, 4, n_classes, seed=5)
    y_train[:] = 0  # single class only
    model = train_mod.make_model(
        "lstm", train_mod.SEQUENCE_LENGTH, train_mod.N_FEATURES_PER_FRAME, n_classes
    )
    train_mod.train_one_model(model, X_train, y_train, epochs=1, batch_size=2,
                              lr=1e-3, device="cpu")
    acc = train_mod.evaluate_accuracy(model, X_train, y_train, "cpu")
    assert acc is not None


def test_single_arch_requested_selection_is_trivial(train_mod):
    n_classes = len(train_mod.VOCABULARY)
    X_train, y_train = _synthetic_xy(train_mod, 4, n_classes, seed=6)
    X_val = X_train[:0]
    y_val = y_train[:0]
    result = train_mod.train_and_select(
        ["cnn"], X_train, y_train, X_val, y_val,
        train_mod.SEQUENCE_LENGTH, train_mod.N_FEATURES_PER_FRAME, n_classes,
        epochs=1, batch_size=2, lr=1e-3, device="cpu",
    )
    assert result["selected_arch"] == "cnn"


def test_resolve_arch_names_all_expands_in_fixed_order(train_mod):
    assert train_mod._resolve_arch_names(["all"]) == list(train_mod.ARCH_ORDER)
    assert train_mod._resolve_arch_names(["mlp", "lstm"]) == ["lstm", "mlp"]


# --------------------------------------------------------------------------- #
# End-to-end run_training on a fake directory-per-label dataset
# --------------------------------------------------------------------------- #
def _write_csv_recording(path, rows, feats):
    import csv

    from src import landmarks

    header = ["timestamp", "frame", "hand_index", "handedness"] + [
        f"{name}_{ax}" for name in landmarks.LANDMARK_NAMES for ax in ("x", "y", "z")
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row, feat in zip(rows, feats):
            line = [row["timestamp"], row["frame"], row["hand_index"], row["handedness"]]
            line.extend(float(v) for v in feat)
            writer.writerow(line)


def _write_fake_dataset(root, signs, n_per_sign=3, seed=0):
    """Write tiny synthetic .csv recordings for a couple of VOCABULARY signs,
    matching the src.preprocess CSV contract (a single 'Right' hand per
    frame, CSV is canonical)."""
    from src import landmarks

    rng = np.random.default_rng(seed)
    n_frames = 5
    for sign in signs:
        sign_dir = root / sign
        sign_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_sign):
            rows = [
                {
                    "timestamp": float(frame), "frame": frame,
                    "hand_index": 0, "handedness": "Right",
                }
                for frame in range(n_frames)
            ]
            feats = rng.uniform(
                0.1, 0.9, size=(n_frames, landmarks.N_FEATURES)
            ).astype(np.float32)
            _write_csv_recording(sign_dir / f"rec{i}.csv", rows, feats)


def test_run_training_end_to_end_all_architectures(train_mod, tmp_path):
    signs = [train_mod.VOCABULARY[0], train_mod.VOCABULARY[1]]
    data_root = tmp_path / "data"
    _write_fake_dataset(data_root, signs, n_per_sign=4)

    out_path = tmp_path / "models" / "sign_classifier.pt"
    label_encoder_path = tmp_path / "models" / "label_encoder.pkl"

    summary = train_mod.run_training(
        data_root=data_root,
        arch_names=train_mod.ARCH_ORDER,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        out_path=out_path,
        seed=42,
        label_encoder_path=label_encoder_path,
    )

    assert out_path.exists()
    assert summary["selected_arch"] in set(train_mod.ARCH_ORDER)
    assert summary["device"] in {"cpu", "cuda"}

    checkpoint = train_mod.load_checkpoint(out_path, map_location="cpu")
    assert checkpoint["num_classes"] == len(train_mod.VOCABULARY)
    assert checkpoint["vocabulary"] == list(train_mod.VOCABULARY)

    report = train_mod.format_summary(summary)
    assert "classification accuracy" in report
    assert summary["selected_arch"] in report
