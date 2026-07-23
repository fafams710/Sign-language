"""Sign-classifier training (PyTorch).

Trains a landmark-sequence sign classifier using PyTorch. Four small model
architectures (LSTM, GRU, 1D CNN, MLP) are available; the training pipeline
trains each requested architecture, selects the best one by validation
accuracy (deterministic fallback when validation is empty), and writes a
single self-describing checkpoint to ``models/sign_classifier.pt`` (default).

This module consumes the Phase 2 data contract exclusively via
:func:`src.preprocess.split_dataset` -- it never re-implements recording
discovery, loading, normalization, sequence-building, or splitting. Shape
constants (``SEQUENCE_LENGTH``, ``N_FEATURES_PER_FRAME``, class count) are
always derived from :mod:`src.utils` / :mod:`src.preprocess`, never hardcoded.

Checkpoint contract (consumed by the future ``src/predict.py``)
-----------------------------------------------------------------
``torch.save`` writes a plain ``dict`` with at least the following keys:

* ``state_dict`` -- the selected model's ``state_dict()``.
* ``architecture`` -- the registry name of the selected architecture
  (one of ``"lstm"``, ``"gru"``, ``"cnn"``, ``"mlp"``).
* ``hyperparams`` -- the keyword arguments used to construct that model, so
  :func:`make_model` can rebuild an identical (untrained) module.
* ``input_shape`` -- ``(SEQUENCE_LENGTH, N_FEATURES_PER_FRAME)``.
* ``num_classes`` -- ``len(VOCABULARY)`` at training time.
* ``vocabulary`` -- the exact :data:`src.utils.VOCABULARY` list ordering used
  for label indices, so ``index -> sign`` is recoverable from the checkpoint
  alone, independent of any pickled label encoder.

Reload with ``torch.load(path, map_location=..., weights_only=False)`` (the
checkpoint contains a plain dict of Python/NumPy-safe values plus tensors, so
it reloads on a different device than it was saved on).

Label-encoder note
-------------------
The checkpoint above is the single authoritative, self-describing source of
label ordering. For compatibility with docs that mention
``models/label_encoder.pkl``, this module ALSO writes (best-effort) a thin
``models/label_encoder.pkl`` containing only a plain Python list (the
VOCABULARY order) -- never a pickled ``sklearn.preprocessing.LabelEncoder``.
``src/predict.py`` is never required to load this file; it exists purely as
an optional compatibility artifact.
"""

import argparse
import pickle
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Direct-script execution (``python src/train.py``) puts ``src/`` itself
    # on sys.path[0], not the repository root, so ``import src...`` would
    # fail. ``python -m src.train`` is the primary supported entry point and
    # never hits this branch; this is a minimal, documented compatibility
    # shim so the direct-script form also works.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.preprocess import N_FEATURES_PER_FRAME, split_dataset
from src.utils import SEQUENCE_LENGTH, VOCABULARY

DEFAULT_OUTPUT = "models/sign_classifier.pt"
DEFAULT_LABEL_ENCODER_OUTPUT = "models/label_encoder.pkl"
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-3

# Requested-architecture default order; also the deterministic tie-break /
# empty-val fallback order (first requested architecture wins).
ARCH_ORDER = ("lstm", "gru", "cnn", "mlp")


# --------------------------------------------------------------------------- #
# Model builders
# --------------------------------------------------------------------------- #
class LSTMClassifier(nn.Module):
    """Single-layer LSTM sequence classifier.

    Consumes ``(batch, seq_len, n_features)`` directly (LSTM's native
    batch-first layout) and classifies from the final hidden state.
    """

    def __init__(self, seq_len, n_features, n_classes, hidden_size=64):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size, num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


class GRUClassifier(nn.Module):
    """Single-layer GRU sequence classifier.

    Consumes ``(batch, seq_len, n_features)`` directly (GRU's native
    batch-first layout) and classifies from the final hidden state.
    """

    def __init__(self, seq_len, n_features, n_classes, hidden_size=64):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=n_features, hidden_size=hidden_size, num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])


class CNNClassifier(nn.Module):
    """Shallow 1D CNN sequence classifier.

    ``nn.Conv1d`` expects ``(batch, channels, length)``, whereas the data
    contract delivers ``(batch, seq_len, n_features)``. This model treats
    ``n_features`` as the channel dimension and ``seq_len`` as the length
    dimension, so the input is transposed (``.transpose(1, 2)``) before the
    convolution and the result is global-average-pooled over the length
    dimension before the final linear layer.
    """

    def __init__(self, seq_len, n_features, n_classes, hidden_channels=32):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_channels = hidden_channels
        kernel_size = min(3, seq_len)
        self.conv = nn.Conv1d(
            in_channels=n_features, out_channels=hidden_channels,
            kernel_size=kernel_size, padding=kernel_size // 2,
        )
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_channels, n_classes)

    def forward(self, x):
        # (batch, seq_len, n_features) -> (batch, n_features, seq_len).
        x = x.transpose(1, 2)
        x = self.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class MLPClassifier(nn.Module):
    """Single hidden-layer MLP sequence classifier.

    Flattens the ``(seq_len, n_features)`` sequence into one vector of length
    ``seq_len * n_features`` (time and feature axes concatenated, in that
    order) before the hidden linear layer.
    """

    def __init__(self, seq_len, n_features, n_classes, hidden_size=128):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(seq_len * n_features, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# Name -> (model class, default hyperparams beyond seq_len/n_features/n_classes).
ARCHITECTURES = {
    "lstm": (LSTMClassifier, {"hidden_size": 64}),
    "gru": (GRUClassifier, {"hidden_size": 64}),
    "cnn": (CNNClassifier, {"hidden_channels": 32}),
    "mlp": (MLPClassifier, {"hidden_size": 128}),
}


def make_model(arch, seq_len, n_features, n_classes, **hyperparams):
    """Construct a model by registry name.

    Parameters
    ----------
    arch : str
        One of the keys of :data:`ARCHITECTURES` (``"lstm"``, ``"gru"``,
        ``"cnn"``, ``"mlp"``).
    seq_len, n_features, n_classes : int
        Fixed input/output shape parameters.
    **hyperparams
        Extra constructor keyword arguments (e.g. ``hidden_size``); missing
        keys fall back to the architecture's small default.

    Returns
    -------
    torch.nn.Module
        A model accepting ``(batch, seq_len, n_features)`` and returning
        logits of shape ``(batch, n_classes)``.

    Raises
    ------
    ValueError
        If ``arch`` is not a known registry name.
    """
    if arch not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{arch}'. Expected one of "
            f"{sorted(ARCHITECTURES)}."
        )
    model_cls, defaults = ARCHITECTURES[arch]
    kwargs = dict(defaults)
    kwargs.update(hyperparams)
    return model_cls(seq_len, n_features, n_classes, **kwargs)


def get_hyperparams(model):
    """Extract the constructor hyperparameters recorded on a built model.

    Only the extra keyword arguments beyond ``seq_len``/``n_features``/
    ``n_classes`` are returned (those three are recorded separately in the
    checkpoint's ``input_shape`` / ``num_classes`` fields).
    """
    if isinstance(model, (LSTMClassifier, GRUClassifier)):
        return {"hidden_size": model.hidden_size}
    if isinstance(model, CNNClassifier):
        return {"hidden_channels": model.hidden_channels}
    if isinstance(model, MLPClassifier):
        return {"hidden_size": model.hidden_size}
    raise TypeError(f"Unrecognized model type {type(model)!r}.")


def arch_name_of(model):
    """Return the registry name for a built model instance."""
    for name, (model_cls, _) in ARCHITECTURES.items():
        if isinstance(model, model_cls):
            return name
    raise TypeError(f"Unrecognized model type {type(model)!r}.")


# --------------------------------------------------------------------------- #
# Reproducibility / device helpers
# --------------------------------------------------------------------------- #
def seed_everything(seed):
    """Seed Python's ``random``, NumPy, and torch (CPU + CUDA) from one seed.

    Exact bit-for-bit reproducibility across platforms/CUDA versions is not
    guaranteed (documented caveat); this only removes seed-level randomness.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Return ``"cuda"`` if available, else ``"cpu"``."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def _make_loader(X, y, batch_size, shuffle):
    """Build a small-n-safe DataLoader.

    Batch size is clamped to the number of available samples (never larger
    than ``len(X)``) and ``drop_last`` is always ``False``, so the sole batch
    at ``n_samples`` as small as 1 or 2 is never dropped.
    """
    n = X.shape[0]
    dataset = TensorDataset(
        torch.from_numpy(X), torch.from_numpy(y),
    )
    effective_batch = max(1, min(batch_size, n))
    return DataLoader(
        dataset, batch_size=effective_batch, shuffle=shuffle, drop_last=False,
    )


def train_one_model(model, X_train, y_train, epochs, batch_size, lr, device):
    """Train a single model in place with Adam + CrossEntropyLoss.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train (moved to ``device`` in place).
    X_train : numpy.ndarray
        ``float32`` array of shape ``(n, seq_len, n_features)``.
    y_train : numpy.ndarray
        ``int64`` array of shape ``(n,)``.
    epochs : int
        Number of full passes over ``X_train``.
    batch_size : int
        Requested batch size (clamped to ``len(X_train)`` internally).
    lr : float
        Adam learning rate.
    device : str
        ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    torch.nn.Module
        The same ``model`` instance, trained in place.
    """
    model.to(device)
    if X_train.shape[0] == 0:
        # Nothing to train on; leave the (randomly initialized) model as is.
        return model

    loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def evaluate_accuracy(model, X, y, device, batch_size=64):
    """Return classification accuracy of ``model`` on ``(X, y)``.

    Returns ``None`` when ``X`` has zero samples (accuracy is undefined, not
    zero) so callers can distinguish "empty partition" from "0% accuracy".
    """
    if X.shape[0] == 0:
        return None

    model.to(device)
    model.eval()
    loader = _make_loader(X, y, batch_size, shuffle=False)
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == yb).sum().item()
        total += yb.shape[0]
    return correct / total if total > 0 else None


# --------------------------------------------------------------------------- #
# Architecture training + selection
# --------------------------------------------------------------------------- #
def train_and_select(
    arch_names, X_train, y_train, X_val, y_val, seq_len, n_features,
    n_classes, epochs, batch_size, lr, device,
):
    """Train every requested architecture and select the best one.

    Selection rule (documented, deterministic):

    * If the validation partition is non-empty: select the architecture with
      the highest validation accuracy; ties are broken by ``arch_names``
      order (the first-listed architecture wins a tie).
    * If the validation partition is empty: validation-based selection and
      early stopping are skipped entirely. Fall back to the architecture
      with the highest TRAINING accuracy; if that is tied/degenerate, fall
      back further to the first architecture in ``arch_names``.

    Parameters
    ----------
    arch_names : Sequence[str]
        Requested architecture registry names, in priority (tie-break) order.
    X_train, y_train, X_val, y_val : numpy.ndarray
        Train / val partitions from :func:`src.preprocess.split_dataset`.
    seq_len, n_features, n_classes : int
        Fixed input/output shape parameters.
    epochs, batch_size, lr : int, int, float
        Training hyperparameters shared by all architectures.
    device : str
        ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    dict
        ``{"selected_arch": str, "selected_model": nn.Module,
        "val_accuracies": dict[str, float | None],
        "train_accuracies": dict[str, float | None],
        "val_skipped": bool}``. ``val_accuracies`` maps each requested
        architecture name to its validation accuracy, or ``None`` (reported
        as "skipped") when validation is empty.
    """
    val_is_empty = X_val.shape[0] == 0

    models = {}
    train_accuracies = {}
    val_accuracies = {}

    for name in arch_names:
        model = make_model(name, seq_len, n_features, n_classes)
        train_one_model(model, X_train, y_train, epochs, batch_size, lr, device)
        models[name] = model
        train_accuracies[name] = evaluate_accuracy(model, X_train, y_train, device)
        val_accuracies[name] = (
            None if val_is_empty
            else evaluate_accuracy(model, X_val, y_val, device)
        )

    if not val_is_empty:
        # Highest val accuracy; ties broken by arch_names order (stable sort,
        # first-listed wins because we iterate in that fixed order).
        best_name = max(
            arch_names,
            key=lambda n: (
                val_accuracies[n] if val_accuracies[n] is not None else -1.0
            ),
        )
    else:
        # Empty-val fallback: best training accuracy, else first requested.
        best_train = max(
            (train_accuracies[n] for n in arch_names if train_accuracies[n] is not None),
            default=None,
        )
        if best_train is None:
            best_name = arch_names[0]
        else:
            candidates = [
                n for n in arch_names
                if train_accuracies[n] is not None
                and train_accuracies[n] >= best_train
            ]
            best_name = candidates[0] if candidates else arch_names[0]

    return {
        "selected_arch": best_name,
        "selected_model": models[best_name],
        "val_accuracies": val_accuracies,
        "train_accuracies": train_accuracies,
        "val_skipped": val_is_empty,
    }


# --------------------------------------------------------------------------- #
# Checkpoint I/O
# --------------------------------------------------------------------------- #
def save_checkpoint(model, arch, seq_len, n_features, n_classes, out_path):
    """Write a self-describing checkpoint to ``out_path``.

    See the module docstring for the full field contract. Parent directories
    are created as needed.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "architecture": arch,
        "hyperparams": get_hyperparams(model),
        "input_shape": (seq_len, n_features),
        "num_classes": n_classes,
        "vocabulary": list(VOCABULARY),
    }
    torch.save(checkpoint, out_path)
    return out_path


def load_checkpoint(path, map_location=None):
    """Reload a checkpoint written by :func:`save_checkpoint`.

    Parameters
    ----------
    path : str or pathlib.Path
        Checkpoint path.
    map_location : str or torch.device, optional
        Forwarded to ``torch.load`` so a checkpoint saved on one device
        (e.g. cuda) reloads cleanly on another (e.g. cpu).

    Returns
    -------
    dict
        The checkpoint dict (see module docstring for keys).
    """
    return torch.load(path, map_location=map_location, weights_only=False)


def model_from_checkpoint(checkpoint):
    """Reconstruct an (untrained-weights-loaded) model from a checkpoint dict."""
    seq_len, n_features = checkpoint["input_shape"]
    model = make_model(
        checkpoint["architecture"], seq_len, n_features,
        checkpoint["num_classes"], **checkpoint["hyperparams"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model


def save_label_encoder(out_path=DEFAULT_LABEL_ENCODER_OUTPUT):
    """Best-effort write of a thin compatibility label-encoder pickle.

    Writes only a plain Python ``list`` (the VOCABULARY order) -- never an
    ``sklearn.preprocessing.LabelEncoder`` instance -- so there is no
    sklearn-version coupling. This artifact is optional; ``src/predict.py``
    must be able to work from the checkpoint alone and is never required to
    load this file. Failures here (e.g. a read-only models/ directory) are
    swallowed rather than aborting the training run.
    """
    try:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(list(VOCABULARY), f)
        return out_path
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _resolve_arch_names(arch_args):
    """Expand CLI ``--arch`` values (including ``"all"``) into a fixed order.

    The result preserves :data:`ARCH_ORDER` (the tie-break / fallback order)
    restricted to the requested architectures, with duplicates removed.
    """
    requested = set()
    for a in arch_args:
        if a == "all":
            requested.update(ARCH_ORDER)
        else:
            requested.add(a)
    return [name for name in ARCH_ORDER if name in requested]


def run_training(
    data_root, arch_names=ARCH_ORDER, epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE, lr=DEFAULT_LEARNING_RATE,
    out_path=DEFAULT_OUTPUT, seed=DEFAULT_SEED,
    label_encoder_path=DEFAULT_LABEL_ENCODER_OUTPUT,
):
    """Run the full train -> select -> save pipeline.

    Obtains data solely via :func:`src.preprocess.split_dataset`; trains and
    evaluates every architecture in ``arch_names``; selects the best one
    (see :func:`train_and_select`); saves the selected model's checkpoint
    (and a best-effort thin label-encoder pickle); returns a summary dict.

    Parameters
    ----------
    data_root : str or pathlib.Path
        Dataset root passed to :func:`src.preprocess.split_dataset`.
    arch_names : Sequence[str], optional
        Architecture registry names to train (default: all four, in
        :data:`ARCH_ORDER`).
    epochs, batch_size, lr : int, int, float
        Training hyperparameters.
    out_path : str or pathlib.Path, optional
        Checkpoint output path (default ``models/sign_classifier.pt``).
    seed : int, optional
        Seed for numpy/torch and for ``split_dataset`` (default 42).
    label_encoder_path : str or pathlib.Path, optional
        Optional thin compatibility pickle output path.

    Returns
    -------
    dict
        Summary containing per-architecture val/train accuracy (or ``None``
        for "skipped"), the selected architecture, final train/val/test
        accuracy for non-empty partitions, and the output paths written.
    """
    seed_everything(seed)
    device = get_device()

    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
        data_root, seed=seed,
    )

    n_classes = len(VOCABULARY)
    result = train_and_select(
        list(arch_names), X_train, y_train, X_val, y_val,
        SEQUENCE_LENGTH, N_FEATURES_PER_FRAME, n_classes,
        epochs, batch_size, lr, device,
    )

    selected_model = result["selected_model"]
    selected_arch = result["selected_arch"]

    checkpoint_path = save_checkpoint(
        selected_model, selected_arch, SEQUENCE_LENGTH, N_FEATURES_PER_FRAME,
        n_classes, out_path,
    )
    label_encoder_out = save_label_encoder(label_encoder_path)

    final_train_acc = evaluate_accuracy(selected_model, X_train, y_train, device)
    final_val_acc = evaluate_accuracy(selected_model, X_val, y_val, device)
    final_test_acc = evaluate_accuracy(selected_model, X_test, y_test, device)

    summary = {
        "device": device,
        "requested_archs": list(arch_names),
        "val_accuracies": result["val_accuracies"],
        "train_accuracies": result["train_accuracies"],
        "val_selection_skipped": result["val_skipped"],
        "selected_arch": selected_arch,
        "final_train_accuracy": final_train_acc,
        "final_val_accuracy": final_val_acc,
        "final_test_accuracy": final_test_acc,
        "test_reporting_skipped": X_test.shape[0] == 0,
        "checkpoint_path": str(checkpoint_path),
        "label_encoder_path": str(label_encoder_out) if label_encoder_out else None,
    }
    return summary


def format_summary(summary):
    """Render :func:`run_training`'s summary dict as a human-readable report.

    Accuracy wording follows the project's fixed-vocabulary reporting
    convention: results are reported as classification accuracy on a fixed
    vocabulary of predefined ASL signs under controlled webcam conditions; no
    general ASL translation claim is made.
    """
    lines = []
    lines.append(f"Device: {summary['device']}")
    lines.append("Per-architecture validation accuracy "
                  "(fixed vocabulary, controlled webcam conditions):")
    for name in summary["requested_archs"]:
        val_acc = summary["val_accuracies"].get(name)
        if summary["val_selection_skipped"] or val_acc is None:
            lines.append(f"  {name}: skipped (empty validation partition)")
        else:
            lines.append(f"  {name}: {val_acc:.4f}")
    if summary["val_selection_skipped"]:
        lines.append(
            "Validation-based selection and early stopping were SKIPPED "
            "(empty validation partition); the selected architecture was "
            "chosen by the documented training-accuracy fallback."
        )
    lines.append(f"Selected architecture: {summary['selected_arch']}")

    def _acc_line(label, value):
        if value is None:
            return f"{label} accuracy: skipped (empty partition)"
        return (
            f"{label} classification accuracy on a fixed vocabulary of "
            f"predefined ASL signs under controlled webcam conditions: "
            f"{value:.4f}"
        )

    lines.append(_acc_line("Train", summary["final_train_accuracy"]))
    lines.append(_acc_line("Validation", summary["final_val_accuracy"]))
    lines.append(_acc_line("Test", summary["final_test_accuracy"]))
    lines.append(f"Checkpoint written to: {summary['checkpoint_path']}")
    if summary["label_encoder_path"]:
        lines.append(
            f"Thin compatibility label encoder written to: "
            f"{summary['label_encoder_path']}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser():
    """Build the argparse parser shared by both CLI entry points."""
    parser = argparse.ArgumentParser(
        description="Train the landmark-sequence sign classifier.",
    )
    parser.add_argument(
        "--data-root", default="data/processed",
        help="Dataset root directory (directory-per-label recordings or "
             "preprocessed data), passed to src.preprocess.split_dataset.",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Training epochs per architecture (default {DEFAULT_EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Requested batch size, clamped to available samples "
             f"(default {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE,
        help=f"Adam learning rate (default {DEFAULT_LEARNING_RATE}).",
    )
    parser.add_argument(
        "--arch", action="append",
        choices=["lstm", "gru", "cnn", "mlp", "all"],
        help="Architecture(s) to train; repeat for multiple, or pass 'all' "
             "for all four. Default: all.",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Checkpoint output path (default {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed for numpy/torch/split (default {DEFAULT_SEED}).",
    )
    return parser


def main(argv=None):
    """CLI entry point shared by ``python -m src.train`` and
    ``python src/train.py`` (via the guarded ``__main__`` block below)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    arch_names = _resolve_arch_names(args.arch or ["all"])

    summary = run_training(
        data_root=args.data_root,
        arch_names=arch_names,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        out_path=args.output,
        seed=args.seed,
    )
    print(format_summary(summary))
    return summary


if __name__ == "__main__":
    main()
