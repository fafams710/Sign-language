"""Landmark preprocessing for model training.

Turns a directory-per-label dataset of recorded hand-landmark files into a
labeled, model-ready NumPy dataset, and produces a reproducible train / val /
test split. This module is the producer of the Phase 2 data contract that
``src/train.py`` and ``src/predict.py`` consume; everything below is documented
so those phases need not reverse-engineer the layout.

Data contract (binding for downstream phases)
---------------------------------------------
* One recording file == exactly one training sample (one sign performance). No
  in-file segmentation is performed.
* Each sample is a sequence of shape ``(SEQUENCE_LENGTH, N_FEATURES_PER_FRAME)``
  == ``(30, 126)`` of ``float32``.
* ``N_FEATURES_PER_FRAME`` is ``2 * src.landmarks.N_FEATURES`` (two hands x 63
  features per hand). Per-frame layout is a fixed two-slot ``[Right | Left]``
  vector: features ``0..62`` hold the Right hand, features ``63..125`` hold the
  Left hand. A hand that is absent in a frame contributes an all-zero 63-slot;
  a frame with no hands is an all-zero 126 vector. Zero-fill is applied AFTER
  per-hand normalization.
* Per-hand features are normalized with :func:`src.landmarks.normalize_landmarks`
  (wrist-relative translation + scale) so training data and live inference share
  one normalization. This module never re-implements normalization.
* Labels are the integer INDEX of the sign within :data:`src.utils.VOCABULARY`
  (e.g. ``"HELLO" -> 0``). The sign name is the parent directory of each
  recording and must match a ``VOCABULARY`` entry exactly (case-sensitive). No
  pickled ``LabelEncoder`` is produced or required.

CSV is the canonical recording format: if both ``<stem>.csv`` and ``<stem>.npy``
exist for the same recording, the CSV is loaded and the paired NPY is ignored.

Import safety: this module imports only numpy / pandas / scikit-learn (plus the
pure stdlib+numpy capture loaders) at module top. It does NOT import torch, and
it requires neither MediaPipe nor OpenCV nor the ``hand_landmarker.task`` binary
to import or to run. The capture column contract is reused from
``tools.capture_demo`` (``META_COLS``, ``load_csv``, ``load_npy``) rather than
re-derived here.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: F401  (declared dependency; available for downstream use)
from sklearn.model_selection import train_test_split

from src import landmarks
from src.utils import SEQUENCE_LENGTH, VOCABULARY
from tools.capture_demo import META_COLS, load_csv, load_npy  # noqa: F401

# Both hands concatenated, 63 features each -> 126 per frame. Derived from the
# landmark feature count, never hardcoded as a bare literal.
N_FEATURES_PER_FRAME = 2 * landmarks.N_FEATURES

# Fixed [Right | Left] slot layout. "Right" / "Left" are the MediaPipe
# display_name strings recorded by tools.capture_demo (case-sensitive).
_HANDEDNESS_SLOTS = {"Right": 0, "Left": 1}

_RANDOM_SEED = 42

# Recording file extensions in precedence order (CSV canonical).
_CSV_SUFFIX = ".csv"
_NPY_SUFFIX = ".npy"


def discover_recordings(root):
    """Discover directory-per-label recordings under a dataset root.

    The dataset root contains one subdirectory per sign; the subdirectory name
    is the label. Recording files (``*.csv`` preferred, else ``*.npy``) live
    directly inside those subdirectories. CSV is canonical: when a ``.csv`` and a
    ``.npy`` share a stem in the same directory, only the CSV is returned.

    Parameters
    ----------
    root : str or pathlib.Path
        Path to the dataset root directory.

    Returns
    -------
    list[tuple[pathlib.Path, str]]
        ``(recording_path, label)`` pairs, sorted deterministically by
        (label, path) so discovery order never depends on filesystem iteration
        order. ``label`` is the parent directory name.

    Raises
    ------
    NotADirectoryError
        If ``root`` does not exist or is not a directory.
    ValueError
        If the root has no sign subdirectories, contains no recordings, holds a
        stray recording file directly at the root (a recording with no ``<SIGN>``
        parent), or contains a subdirectory whose name is not an exact,
        case-sensitive entry in :data:`src.utils.VOCABULARY`.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(
            f"Dataset root '{root}' does not exist or is not a directory."
        )

    # A recording sitting directly in the root has no <SIGN> parent: error, do
    # not silently ignore mislabeled data.
    stray = [
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in (_CSV_SUFFIX, _NPY_SUFFIX)
    ]
    if stray:
        names = ", ".join(sorted(p.name for p in stray))
        raise ValueError(
            f"Recording file(s) found directly in the dataset root '{root}' "
            f"with no <SIGN> parent directory: {names}. Place each recording "
            "inside a <SIGN>/ subdirectory matching a VOCABULARY entry."
        )

    subdirs = sorted((p for p in root.iterdir() if p.is_dir()),
                     key=lambda p: p.name)
    if not subdirs:
        raise ValueError(
            f"Dataset root '{root}' has no <SIGN> subdirectories. Expected a "
            "directory-per-label layout (e.g. HELLO/, YES/, ...)."
        )

    pairs = []
    for sub in subdirs:
        label = sub.name
        if label not in VOCABULARY:
            raise ValueError(
                f"Subdirectory '{label}' under '{root}' is not a VOCABULARY "
                "entry (matching is exact and case-sensitive). Rename it to a "
                "known sign or remove it; recordings are never silently skipped."
            )
        recordings = _recordings_in(sub)
        for rec in recordings:
            pairs.append((rec, label))

    if not pairs:
        raise ValueError(
            f"Dataset root '{root}' contains sign subdirectories but no "
            f"recording files (*.csv / *.npy) inside them."
        )

    pairs.sort(key=lambda item: (item[1], str(item[0])))
    return pairs


def _recordings_in(directory):
    """Return the canonical recording files in a single sign directory.

    CSV is canonical: every ``*.csv`` is included; an ``*.npy`` is included only
    when no CSV with the same stem exists in the directory. Hidden / system files
    and non-recording files are ignored. Returns a list sorted by path.
    """
    directory = Path(directory)
    csv_stems = set()
    csvs = []
    npys = []
    for p in directory.iterdir():
        if not p.is_file() or p.name.startswith("."):
            continue
        suffix = p.suffix.lower()
        if suffix == _CSV_SUFFIX:
            csvs.append(p)
            csv_stems.add(p.stem)
        elif suffix == _NPY_SUFFIX:
            npys.append(p)
    canonical = list(csvs) + [p for p in npys if p.stem not in csv_stems]
    return sorted(canonical, key=lambda p: str(p))


def load_recording(path):
    """Load one recording's per-hand, per-frame rows.

    Reuses the capture column contract: ``tools.capture_demo.load_csv`` for CSV
    (canonical) and ``tools.capture_demo.load_npy`` for NPY. Both yield
    ``META_COLS`` metadata columns (timestamp, frame, hand_index, handedness)
    followed by ``src.landmarks.N_FEATURES`` (63) raw coordinate columns ordered
    by :data:`src.landmarks.LANDMARK_NAMES`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a ``.csv`` or ``.npy`` recording.

    Returns
    -------
    tuple[list[dict], numpy.ndarray]
        ``(meta_rows, feat_rows)`` where ``meta_rows`` is a list of dicts with
        keys ``timestamp``, ``frame``, ``hand_index``, ``handedness`` and
        ``feat_rows`` is a ``float32`` array of shape ``(n_rows, 63)``.

    Raises
    ------
    ValueError
        If the file has zero data rows (an empty recording is invalid as a
        training sample) or an unsupported extension.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == _CSV_SUFFIX:
        meta_rows, feat_rows = load_csv(str(path))
    elif suffix == _NPY_SUFFIX:
        meta_rows, feat_rows = load_npy(str(path))
    else:
        raise ValueError(
            f"Unsupported recording extension '{path.suffix}' for '{path}'. "
            "Expected .csv or .npy."
        )

    if feat_rows.size == 0 or len(meta_rows) == 0:
        raise ValueError(
            f"Recording '{path}' has no data rows; an empty recording is not a "
            "valid training sample."
        )
    if feat_rows.shape[1] != landmarks.N_FEATURES:
        raise ValueError(
            f"Recording '{path}' has {feat_rows.shape[1]} feature columns; "
            f"expected {landmarks.N_FEATURES} (21 landmarks x 3 coords)."
        )
    return meta_rows, feat_rows.astype(np.float32, copy=False)


def frames_to_sequence(meta_rows, feat_rows):
    """Collapse per-hand rows into a per-frame ``[Right | Left]`` sequence.

    Rows are grouped by their integer ``frame`` value (sorted ascending, never by
    row order). Within each frame, every present hand's 63 raw features are
    normalized via :func:`src.landmarks.normalize_landmarks` and then placed into
    its handedness slot: ``"Right" -> 0`` (features 0..62), ``"Left" -> 1``
    (features 63..125). A missing hand leaves its 63-slot all zeros; a frame with
    no hands yields an all-zero 126 vector. Zero-fill happens AFTER per-hand
    normalization.

    Parameters
    ----------
    meta_rows : list[dict]
        Per-row metadata (must include ``frame`` and ``handedness``), aligned
        positionally with ``feat_rows``.
    feat_rows : numpy.ndarray
        ``(n_rows, 63)`` raw per-hand feature rows.

    Returns
    -------
    numpy.ndarray
        ``float32`` array of shape ``(n_frames, 126)``, frames in ascending
        ``frame`` order.

    Raises
    ------
    ValueError
        If a row carries a handedness string outside the exact, case-sensitive
        set ``{"Right", "Left"}``, or if two rows of the SAME handedness appear
        in one frame (malformed data is an error, never silently overwritten).
    """
    feat_rows = np.asarray(feat_rows, dtype=np.float32)
    n_per_hand = landmarks.N_FEATURES

    # Preserve first-seen order of frame ids, then sort ascending.
    frame_order = {}
    for idx, meta in enumerate(meta_rows):
        frame_order.setdefault(int(meta["frame"]), []).append(idx)
    ordered_frames = sorted(frame_order)

    sequence = np.zeros((len(ordered_frames), N_FEATURES_PER_FRAME),
                        dtype=np.float32)

    for out_idx, frame_id in enumerate(ordered_frames):
        seen_slots = set()
        for row_idx in frame_order[frame_id]:
            handedness = meta_rows[row_idx]["handedness"]
            if handedness not in _HANDEDNESS_SLOTS:
                raise ValueError(
                    f"Frame {frame_id}: unrecognized handedness "
                    f"'{handedness}'. Expected exactly 'Right' or 'Left' "
                    "(case-sensitive)."
                )
            slot = _HANDEDNESS_SLOTS[handedness]
            if slot in seen_slots:
                raise ValueError(
                    f"Frame {frame_id}: two rows with handedness "
                    f"'{handedness}'. At most one Right and one Left hand are "
                    "expected per frame."
                )
            seen_slots.add(slot)

            raw = feat_rows[row_idx].reshape(1, n_per_hand)
            normalized = landmarks.normalize_landmarks(raw)[0]
            start = slot * n_per_hand
            sequence[out_idx, start:start + n_per_hand] = normalized

    return sequence


def fix_length(seq, length=SEQUENCE_LENGTH):
    """Force a per-frame sequence to exactly ``length`` frames.

    * ``len(seq) > length``: keep the CENTER window of ``length`` frames (trim).
      When the surplus is odd the EXTRA trimmed frame is removed from the FRONT
      (the start offset is ``ceil(surplus / 2)``), so the window is shifted one
      frame toward the end deterministically.
    * ``len(seq) < length``: edge-repeat the LAST frame and POST-pad until the
      sequence reaches ``length``.
    * ``len(seq) == length``: returned unchanged (as ``float32``).

    Trim/pad is applied AFTER per-frame normalization.

    Parameters
    ----------
    seq : numpy.ndarray
        ``(n_frames, 126)`` per-frame sequence.
    length : int, optional
        Target frame count (default :data:`src.utils.SEQUENCE_LENGTH`).

    Returns
    -------
    numpy.ndarray
        ``float32`` array of shape ``(length, 126)``.

    Raises
    ------
    ValueError
        If ``seq`` has zero frames (nothing to repeat for post-padding).
    """
    seq = np.asarray(seq, dtype=np.float32)
    n_frames = seq.shape[0]
    if n_frames == 0:
        raise ValueError("Cannot fix length of an empty (0-frame) sequence.")

    if n_frames == length:
        return seq.copy()

    if n_frames > length:
        surplus = n_frames - length
        # Odd surplus: drop the extra frame from the front (start = ceil(s/2)).
        start = (surplus + 1) // 2
        return seq[start:start + length].copy()

    # n_frames < length: post-pad by repeating the last frame.
    pad_count = length - n_frames
    last = seq[-1:]
    padding = np.repeat(last, pad_count, axis=0)
    return np.concatenate([seq, padding], axis=0)


def recording_to_sample(path):
    """Convert a single recording file into a fixed-length sample.

    Pipeline: :func:`load_recording` -> :func:`frames_to_sequence` ->
    :func:`fix_length`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a ``.csv`` or ``.npy`` recording.

    Returns
    -------
    numpy.ndarray
        ``float32`` array of shape ``(SEQUENCE_LENGTH, 126)``.
    """
    meta_rows, feat_rows = load_recording(path)
    sequence = frames_to_sequence(meta_rows, feat_rows)
    return fix_length(sequence, SEQUENCE_LENGTH)


def build_dataset(root):
    """Build the full labeled dataset from a directory-per-label root.

    Each recording becomes one sample; the parent directory name is mapped to its
    :data:`src.utils.VOCABULARY` index. Discovery order is deterministic (see
    :func:`discover_recordings`).

    Parameters
    ----------
    root : str or pathlib.Path
        Dataset root directory.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(X, y)`` where ``X`` is ``float32`` of shape
        ``(n_samples, SEQUENCE_LENGTH, 126)`` and ``y`` is ``int64`` of shape
        ``(n_samples,)`` holding VOCABULARY indices.

    Raises
    ------
    ValueError, NotADirectoryError
        Propagated from :func:`discover_recordings` / :func:`load_recording` for
        invalid roots, unknown labels, or empty recordings.
    """
    pairs = discover_recordings(root)
    samples = []
    labels = []
    for path, label in pairs:
        samples.append(recording_to_sample(path))
        labels.append(VOCABULARY.index(label))

    X = np.stack(samples, axis=0).astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=np.int64)
    return X, y


def split_dataset(root, seed=_RANDOM_SEED):
    """Build and split the dataset into 70/15/15 train/val/test partitions.

    Returns a 6-tuple ``(X_train, y_train, X_val, y_val, X_test, y_test)``. Each
    ``X_*`` is ``float32`` of shape ``(n_part, SEQUENCE_LENGTH, 126)`` and each
    ``y_*`` is ``int64`` of shape ``(n_part,)`` holding VOCABULARY indices.

    Splitting is two-step (train vs. temp at 70/30, then temp split evenly into
    val and test) and seeded with ``seed`` (default 42), so two calls on
    identical input produce identical partitions. It is STRATIFIED by label when
    every present class has at least 2 samples (and again, when feasible, for the
    val/test step); if any present class has fewer than 2 samples the function
    falls back to a non-stratified (still seeded) split and emits a
    :func:`warnings.warn`, rather than crashing.

    Rounding behavior on small datasets: scikit-learn's ``train_test_split``
    determines the temp/test size by rounding the requested fraction. With very
    small ``n`` the 15% val or test fraction can round down to zero, producing an
    empty val and/or test partition; this is deterministic and not an error.
    Downstream code that requires a non-empty validation set must check sizes.

    Parameters
    ----------
    root : str or pathlib.Path
        Dataset root directory.
    seed : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    tuple[numpy.ndarray, ...]
        ``(X_train, y_train, X_val, y_val, X_test, y_test)``.

    Raises
    ------
    ValueError, NotADirectoryError
        Propagated from :func:`build_dataset` for invalid datasets.
    """
    X, y = build_dataset(root)

    can_stratify = _can_stratify(y)
    if not can_stratify:
        warnings.warn(
            "split_dataset: at least one class has fewer than 2 samples; "
            "falling back to a non-stratified (still seeded) split. Class "
            "balance across partitions is not guaranteed.",
            stacklevel=2,
        )

    # Step 1: train (70%) vs temp (30%).
    strat1 = y if can_stratify else None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, shuffle=True, stratify=strat1,
    )

    # Step 2: split temp evenly into val (15%) and test (15%). Stratify only when
    # feasible for the temp subset; otherwise fall back without re-warning.
    if X_temp.shape[0] < 2:
        # Cannot split a single (or empty) temp sample into two partitions;
        # assign it deterministically to test, leaving val empty.
        X_val = X_temp[:0]
        y_val = y_temp[:0]
        X_test = X_temp
        y_test = y_temp
    else:
        strat2 = y_temp if (can_stratify and _can_stratify(y_temp)) else None
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=seed, shuffle=True,
            stratify=strat2,
        )

    return X_train, y_train, X_val, y_val, X_test, y_test


def _can_stratify(y):
    """Return True when every present class in ``y`` has at least 2 samples."""
    if len(y) == 0:
        return False
    _, counts = np.unique(y, return_counts=True)
    return bool(np.all(counts >= 2))
