"""Real-time inference and prediction smoothing.

Responsibility
--------------
Turns the per-frame hand landmarks produced by :func:`src.landmarks.extract`
into a stable, accepted gloss string that a caller feeds to
:meth:`src.buffer.TokenBuffer.add`. This module owns three concerns end to
end: (1) assembling a fixed-length, train-parity-preserving landmark window
(:class:`FrameBuffer`), (2) running a sign classifier over that window
(:class:`MockPredictor` / :class:`RealPredictor`), and (3) smoothing raw
per-window predictions into a debounced, confidence-gated emission
(:class:`PredictionSmoother`). :class:`SignRecognizer` composes all three, and
:func:`create_recognizer` is the single documented entry point an external UI
author binds to.

Downstream consumer: the ``emitted`` field of :class:`RecognitionResult`
(and the return value of :meth:`PredictionSmoother.update`) is a gloss
string meant to be passed, unmodified, to ``src.buffer.TokenBuffer.add``.
That buffer collapses a held sign into a single token; this module does not
attempt that deduplication itself (see :class:`PredictionSmoother`).

Import-purity guarantee
------------------------
Module-top imports are limited to the Python standard library, ``numpy``,
``src.landmarks`` and ``src.utils``. Importing ``src.predict`` never imports
(directly or transitively) ``torch``, ``mediapipe``, ``cv2``, ``pandas``,
``sklearn``, ``src.preprocess``, ``tools.capture_demo`` or ``streamlit``.
``torch`` is imported lazily, and only, inside :meth:`RealPredictor.__init__`
(never at module import time and never per frame), so this module and its
mock path run on a machine with none of those optional dependencies
installed. ``src.preprocess`` is deliberately never imported here: it pulls
in ``pandas`` and ``scikit-learn``, which would break this guarantee.

Per-frame window layout
------------------------
Each buffered frame is a ``float32`` vector of width ``N_FEATURES_PER_FRAME``
(126) laid out as two fixed 63-wide slots, ``[Right | Left]``: features
``0..62`` hold the Right hand, features ``63..125`` hold the Left hand. A
hand absent from a frame contributes an all-zero 63-slot; a frame with no
hands at all is an all-zero 126-vector. Each present hand's raw 63 features
are normalized with :func:`src.landmarks.normalize_landmarks` BEFORE being
written into its slot; the zero-fill for an absent hand happens AFTER
normalization of whichever hand(s) are present, never before.

Train/inference parity is guaranteed structurally, not by importing the
training pipeline, via five points shared with ``src.preprocess``:

1. ``SEQUENCE_LENGTH`` is imported from :mod:`src.utils`, never restated.
2. The per-frame width is derived from :data:`src.landmarks.N_FEATURES`
   (``N_FEATURES_PER_FRAME = 2 * src.landmarks.N_FEATURES``), never a bare
   literal.
3. Per-hand normalization reuses :func:`src.landmarks.normalize_landmarks`;
   this module never reimplements it.
4. The handedness-to-slot map is the same case-sensitive
   ``{"Right": 0, "Left": 1}`` used by ``src.preprocess`` (restated here,
   not imported, purely to avoid pulling ``pandas``/``scikit-learn`` into
   the live import graph; a bitwise parity test keeps the two in sync).
5. Normalization is always applied before zero-fill, matching
   ``src.preprocess.frames_to_sequence``.

Live-robustness divergences from ``src.preprocess`` (intentional; a mid-demo
crash is worse than one degraded frame): two hands sharing the same
handedness in one live frame keep the FIRST and silently ignore later
duplicates (``src.preprocess`` raises on this for training data); a
handedness string outside the exact set ``{"Right", "Left"}`` still raises
``ValueError``; a landmark payload that does not yield exactly
``src.landmarks.N_FEATURES`` floats still raises ``ValueError``. Neither
divergence changes the ``[Right | Left]`` layout.

Checkpoint contract
--------------------
:class:`RealPredictor` expects ``models/sign_classifier.pt`` to deserialize
(via ``torch.load``) to a mapping with exactly six keys: ``state_dict``,
``architecture``, ``hyperparams``, ``input_shape``, ``num_classes`` and
``vocabulary``. The CHECKPOINT's ``vocabulary`` ordering is authoritative for
``index -> gloss`` and becomes the predictor's ``labels``; if it differs from
:data:`src.utils.VOCABULARY` while remaining internally self-consistent, a
``warnings.warn`` is emitted rather than an error. ``models/label_encoder.pkl``
is never opened, referenced as required, or needed on any code path in this
module.

Threading
---------
This module assumes single-threaded use. No lock, queue or thread-safety
guarantee is provided; :class:`FrameBuffer`, :class:`PredictionSmoother` and
:class:`SignRecognizer` all hold plain mutable instance state.

Accuracy wording
-----------------
Any results-related text produced by or about this module must be phrased as
classification accuracy on a fixed vocabulary of predefined ASL signs under
controlled webcam conditions. Nothing in this module claims, or should be
read as claiming, general or unrestricted ASL translation.
"""

import hashlib
import math
import time
import warnings
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from src import landmarks
from src.utils import SEQUENCE_LENGTH, VOCABULARY

# --------------------------------------------------------------------------- #
# Module constants
# --------------------------------------------------------------------------- #

# Both hands concatenated, 63 features each -> 126 per frame. Derived from the
# landmark feature count, never hardcoded as a bare literal (mirrors
# src.preprocess.N_FEATURES_PER_FRAME, which this module does not import).
N_FEATURES_PER_FRAME = 2 * landmarks.N_FEATURES

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Trained checkpoint location. Gitignored and never committed; produced by
# training (see the RealPredictor docstring for the error raised when it is
# absent).
DEFAULT_CHECKPOINT_PATH = _REPO_ROOT / "models" / "sign_classifier.pt"

# Smoothing defaults. PROVISIONAL: chosen with no dataset in hand and
# expected to be retuned once real recordings and a trained model exist. See
# PredictionSmoother for the full rationale and the update-vs-wall-clock
# relationship.
DEFAULT_MIN_CONFIDENCE = 0.60
DEFAULT_STABILITY_FRAMES = 5
DEFAULT_COOLDOWN_FRAMES = 15

# Fixed [Right | Left] slot layout. "Right" / "Left" are the MediaPipe
# display_name strings src.landmarks.extract returns (case-sensitive).
# Restated (not imported) from src.preprocess's private copy to avoid pulling
# pandas/scikit-learn into the live import graph; parity point (d) in the
# module docstring, kept honest by a bitwise parity test.
_HANDEDNESS_SLOTS = {"Right": 0, "Left": 1}

# The closed set of RecognitionResult.status values.
_STATUSES = ("warming_up", "no_hands", "predicted")

# The six required checkpoint keys and their expected container types.
_CHECKPOINT_KEY_TYPES = {
    "state_dict": Mapping,
    "architecture": str,
    "hyperparams": Mapping,
    "input_shape": (list, tuple),
    "num_classes": int,
    "vocabulary": (list, tuple),
}


# --------------------------------------------------------------------------- #
# Window layer
# --------------------------------------------------------------------------- #
def _landmark_payload_to_array(payload):
    """Flatten one hand's landmark payload into a raw ``(63,)`` float32 array.

    Accepts either a sequence of 21 objects exposing ``.x``, ``.y``, ``.z``
    (detected by attribute presence on the first element) or an array-like
    that flattens to ``src.landmarks.N_FEATURES`` floats.

    Parameters
    ----------
    payload : object
        One hand's landmark payload, as carried by a
        ``(handedness, landmarks)`` pair from ``src.landmarks.extract``.

    Returns
    -------
    numpy.ndarray
        ``float32`` array of shape ``(src.landmarks.N_FEATURES,)``.

    Raises
    ------
    ValueError
        If the payload does not yield exactly ``src.landmarks.N_FEATURES``
        float values; the message names the received length.
    """
    items = list(payload)
    if items and hasattr(items[0], "x") and hasattr(items[0], "y") and hasattr(items[0], "z"):
        flat = []
        for point in items:
            flat.extend((float(point.x), float(point.y), float(point.z)))
        arr = np.asarray(flat, dtype=np.float32)
    else:
        arr = np.asarray(items, dtype=np.float32).reshape(-1)

    if arr.shape[0] != landmarks.N_FEATURES:
        raise ValueError(
            f"Landmark payload yielded {arr.shape[0]} float value(s); "
            f"expected exactly {landmarks.N_FEATURES} "
            "(src.landmarks.N_FEATURES: 21 landmarks x 3 coordinates)."
        )
    return arr


def _frame_to_vector(hands):
    """Convert one frame's hand pairs into a ``[Right | Left]`` feature row.

    Parameters
    ----------
    hands : iterable of (str, object) or None
        The value returned by ``src.landmarks.extract``: an iterable of
        ``(handedness, landmarks)`` pairs, or ``None`` / an empty iterable
        for a valid no-hands frame.

    Returns
    -------
    numpy.ndarray
        ``float32`` array of shape ``(N_FEATURES_PER_FRAME,)``.

    Raises
    ------
    ValueError
        If a ``handedness`` value is outside the exact, case-sensitive set
        ``{"Right", "Left"}`` (names the offending value), or if a hand's
        landmark payload does not yield exactly ``src.landmarks.N_FEATURES``
        floats (see :func:`_landmark_payload_to_array`). Two hands sharing
        the same handedness in one frame do NOT raise: the first is kept and
        later duplicates are ignored.
    """
    row = np.zeros(N_FEATURES_PER_FRAME, dtype=np.float32)
    if hands is None:
        return row

    seen_slots = set()
    for handedness, landmark_payload in hands:
        if handedness not in _HANDEDNESS_SLOTS:
            raise ValueError(
                f"Unrecognized handedness {handedness!r}; expected exactly "
                "'Right' or 'Left' (case-sensitive)."
            )
        slot = _HANDEDNESS_SLOTS[handedness]
        if slot in seen_slots:
            # First wins; a duplicate same-handed detection in one frame is
            # tolerated live rather than raised (see module docstring).
            continue
        seen_slots.add(slot)

        raw = _landmark_payload_to_array(landmark_payload)
        normalized = landmarks.normalize_landmarks(
            raw.reshape(1, landmarks.N_FEATURES)
        )[0]
        start = slot * landmarks.N_FEATURES
        row[start:start + landmarks.N_FEATURES] = normalized

    return row


class FrameBuffer:
    """Rolling fixed-length landmark window with train/inference parity.

    Accumulates one ``[Right | Left]`` feature row per call to
    :meth:`add_frame`, keeping at most ``length`` rows (oldest evicted first).
    See the module docstring for the exact per-frame layout and the five
    structural parity points shared with ``src.preprocess``.
    """

    def __init__(self, length: int = SEQUENCE_LENGTH):
        """Create an empty buffer.

        Parameters
        ----------
        length : int, optional
            Target window length in frames (default
            :data:`src.utils.SEQUENCE_LENGTH`).

        Raises
        ------
        ValueError
            If ``length`` is less than 1.
        """
        if length < 1:
            raise ValueError(f"length must be an integer >= 1, got {length!r}.")
        self._length = length
        self._frames = deque(maxlen=length)

    @property
    def length(self) -> int:
        """int: The configured (read-only) target window length."""
        return self._length

    def add_frame(self, hands) -> None:
        """Append one frame's hand data to the buffer.

        Parameters
        ----------
        hands : iterable of (str, object) or None
            The value returned by ``src.landmarks.extract``: an iterable of
            ``(handedness, landmarks)`` pairs, or ``None`` / an empty
            iterable for a valid no-hands frame. ``handedness`` must be
            exactly ``"Right"`` or ``"Left"``; ``landmarks`` is either a
            sequence of 21 objects exposing ``.x``/``.y``/``.z`` or an
            array-like of 63 floats.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            See :func:`_frame_to_vector`: an unrecognized handedness string
            or a landmark payload of the wrong length. Non-finite (NaN/inf)
            landmark values do not raise; they propagate through
            ``src.landmarks.normalize_landmarks``, which skips scaling when
            the point spread is not greater than ``1e-6`` (documented, not
            sanitized).
        """
        self._frames.append(_frame_to_vector(hands))

    def __len__(self) -> int:
        """int: The current number of buffered frames (never > ``length``)."""
        return len(self._frames)

    @property
    def is_ready(self) -> bool:
        """bool: ``True`` once exactly ``length`` frames have been added."""
        return len(self._frames) == self._length

    def window(self):
        """Return the current window as a fresh ``float32`` copy.

        Returns
        -------
        numpy.ndarray or None
            ``float32`` array of shape ``(length, N_FEATURES_PER_FRAME)``
            when :attr:`is_ready` is ``True``; ``None`` otherwise. The
            returned array is always a copy, so mutating it cannot corrupt
            buffer state.
        """
        if not self.is_ready:
            return None
        return np.array(self._frames, dtype=np.float32)

    def reset(self) -> None:
        """Empty the buffer.

        Returns
        -------
        None

        After this call ``len(buffer) == 0``, :attr:`is_ready` is ``False``
        and :meth:`window` returns ``None``.
        """
        self._frames.clear()


# --------------------------------------------------------------------------- #
# Decision layer
# --------------------------------------------------------------------------- #
def _is_passing_confidence(confidence, min_confidence: float) -> bool:
    """Return whether ``confidence`` is a real, finite number >= ``min_confidence``.

    ``None``, ``NaN`` and values that cannot be converted to ``float`` are
    treated as non-passing. A finite value above ``1.0`` from a misbehaving
    predictor is treated as passing (documented in
    :meth:`PredictionSmoother.update`).
    """
    if confidence is None:
        return False
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return False
    if math.isnan(value):
        return False
    return value >= min_confidence


def _is_blank_gloss(gloss) -> bool:
    """Return whether ``gloss`` counts as "no gloss" for smoothing purposes."""
    if gloss is None:
        return True
    if isinstance(gloss, str) and gloss.strip() == "":
        return True
    return False


class PredictionSmoother:
    """Update-counted confidence / stability / cooldown smoothing state machine.

    Turns a stream of raw per-window ``(gloss, confidence)`` predictions into
    a debounced, confidence-gated emission stream via three stages: a
    confidence floor, a consecutive-agreement run, and a post-emission
    refractory cooldown. All three tunables are counted in UPDATES (i.e. one
    unit per call to :meth:`update`, not wall-clock seconds), which keeps the
    state machine deterministic and unit-testable without a clock. At the
    25-30 fps a typical webcam loop delivers, ``stability_frames=5`` is
    roughly 0.17-0.20 s of held agreement and ``cooldown_frames=15`` is
    roughly 0.5-0.6 s of refractory period.

    The defaults (:data:`DEFAULT_MIN_CONFIDENCE`,
    :data:`DEFAULT_STABILITY_FRAMES`, :data:`DEFAULT_COOLDOWN_FRAMES`) are
    PROVISIONAL: they were chosen with no trained model or recorded dataset
    in hand and are expected to be retuned once both exist.

    The smoother deliberately does NOT deduplicate a held sign across
    cooldown boundaries: once a cooldown expires, holding the same sign will
    emit it again. This is correct, by design, because
    ``src.buffer.TokenBuffer.add`` already collapses a run of identical
    consecutive gloss emissions into a single token downstream; duplicating
    that logic here would be redundant.
    """

    def __init__(
        self,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        stability_frames: int = DEFAULT_STABILITY_FRAMES,
        cooldown_frames: int = DEFAULT_COOLDOWN_FRAMES,
    ):
        """Create a smoother with validated tunables.

        Parameters
        ----------
        min_confidence : float, optional
            Inclusive confidence floor in ``[0.0, 1.0]`` a window must meet
            to count toward an agreement run (default
            :data:`DEFAULT_MIN_CONFIDENCE`).
        stability_frames : int, optional
            Number of CONSECUTIVE passing updates carrying the same gloss
            required before it is accepted (default
            :data:`DEFAULT_STABILITY_FRAMES`). ``1`` means immediate accept.
        cooldown_frames : int, optional
            Number of updates, after an emission, during which every update
            returns ``None`` regardless of input (default
            :data:`DEFAULT_COOLDOWN_FRAMES`). ``0`` means no refractory
            period.

        Raises
        ------
        ValueError
            If ``min_confidence`` is outside ``[0.0, 1.0]``,
            ``stability_frames < 1``, or ``cooldown_frames < 0``. Each
            message names the offending parameter.
        """
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError(
                f"min_confidence must be within [0.0, 1.0], got {min_confidence!r}."
            )
        if stability_frames < 1:
            raise ValueError(
                f"stability_frames must be an integer >= 1, got {stability_frames!r}."
            )
        if cooldown_frames < 0:
            raise ValueError(
                f"cooldown_frames must be an integer >= 0, got {cooldown_frames!r}."
            )

        self.min_confidence = min_confidence
        self.stability_frames = stability_frames
        self.cooldown_frames = cooldown_frames

        self._candidate = None
        self._run = 0
        self._cooldown_remaining = 0

    def update(self, gloss: Optional[str], confidence: Optional[float]) -> Optional[str]:
        """Feed one raw prediction through the smoothing state machine.

        Evaluation order (exact, documented so behaviour is predictable from
        this docstring alone):

        1. If a cooldown is active (remaining > 0): decrement it, clear the
           candidate and run, and return ``None`` regardless of ``gloss`` /
           ``confidence``.
        2. Otherwise, if the window is non-passing -- ``gloss`` is ``None``
           or blank, or ``confidence`` is ``None``/``NaN``/not convertible to
           a real number, or ``confidence < min_confidence`` (the comparison
           is inclusive ``>=``, so a confidence exactly equal to
           ``min_confidence`` passes) -- clear the candidate and run and
           return ``None``. A confidence above ``1.0`` from a misbehaving
           predictor is treated as passing.
        3. Otherwise, if ``gloss`` equals the current candidate, increment
           the agreement run; a different gloss restarts the run at 1 for
           the new gloss.
        4. If the run has now reached ``stability_frames``, arm the cooldown
           (``cooldown_frames`` updates), clear the candidate and run, and
           return the accepted gloss. Otherwise return ``None``.

        Parameters
        ----------
        gloss : str or None
            The raw top-1 gloss for this window, or ``None``.
        confidence : float or None
            The raw top-1 confidence for this window, or ``None``.

        Returns
        -------
        str or None
            The accepted gloss at the moment of acceptance; ``None``
            otherwise.
        """
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self._candidate = None
            self._run = 0
            return None

        passing = (not _is_blank_gloss(gloss)) and _is_passing_confidence(
            confidence, self.min_confidence
        )
        if not passing:
            self._candidate = None
            self._run = 0
            return None

        if gloss == self._candidate:
            self._run += 1
        else:
            self._candidate = gloss
            self._run = 1

        if self._run >= self.stability_frames:
            accepted = gloss
            self._candidate = None
            self._run = 0
            self._cooldown_remaining = self.cooldown_frames
            return accepted
        return None

    def reset(self) -> None:
        """Clear the candidate gloss, the agreement run and any cooldown.

        Returns
        -------
        None

        The next passing :meth:`update` call starts a fresh agreement run.
        """
        self._candidate = None
        self._run = 0
        self._cooldown_remaining = 0


@dataclass(frozen=True)
class RecognitionResult:
    """Immutable, UI-facing outcome of one :meth:`SignRecognizer.update` call.

    Fields
    ------
    gloss : str or None
        The raw top-1 predicted gloss for this window (before smoothing);
        ``None`` when no inference ran this call. Feeds the "current sign"
        display.
    confidence : float or None
        The raw top-1 confidence in ``[0.0, 1.0]`` for ``gloss``; ``None``
        when no inference ran this call. Feeds the confidence meter.
    emitted : str or None
        The gloss accepted by :class:`PredictionSmoother` this call, or
        ``None`` when nothing was accepted. Pass this directly to
        ``src.buffer.TokenBuffer.add`` to feed the token stream.
    latency_ms : float or None
        Milliseconds elapsed in the single ``predictor.predict`` call made
        this update (excludes frame capture, landmark extraction and
        display); ``None`` when no inference ran this call. Feeds the
        latency readout.
    frames_buffered : int
        Number of frames currently held in the recognizer's window,
        ``0 <= frames_buffered <= length``. Feeds a warm-up progress
        indicator.
    window_ready : bool
        Whether the window was full (and therefore eligible for inference)
        this call. Feeds a warm-up indicator.
    status : str
        One of the closed set ``"warming_up"`` (window not yet full, no
        inference ran), ``"no_hands"`` (window full but entirely zero, no
        inference ran) or ``"predicted"`` (one inference ran this call). An
        external UI author can switch on this value exhaustively.
    """

    gloss: Optional[str]
    confidence: Optional[float]
    emitted: Optional[str]
    latency_ms: Optional[float]
    frames_buffered: int
    window_ready: bool
    status: str


# --------------------------------------------------------------------------- #
# Predictor layer
# --------------------------------------------------------------------------- #
def _validate_window(window):
    """Validate and return ``window`` as an array of the expected shape.

    Parameters
    ----------
    window : array-like
        Candidate prediction window.

    Returns
    -------
    numpy.ndarray
        ``window`` converted with :func:`numpy.asarray`.

    Raises
    ------
    ValueError
        If the array's shape is not ``(SEQUENCE_LENGTH, N_FEATURES_PER_FRAME)``;
        the message names both the expected and the received shape.
    """
    arr = np.asarray(window)
    expected_shape = (SEQUENCE_LENGTH, N_FEATURES_PER_FRAME)
    if arr.shape != expected_shape:
        raise ValueError(
            f"window has shape {arr.shape}; expected {expected_shape} "
            "(SEQUENCE_LENGTH, N_FEATURES_PER_FRAME)."
        )
    return arr


class MockPredictor:
    """Permanent, torch-free, deterministic test double and UI mock predictor.

    This is NOT scaffolding to be deleted once a real model exists: it is
    the toggle behind ``create_recognizer(use_mock=True)`` and the UI's mock
    mode, and it is what makes ``tests/test_predict.py`` hermetic. It
    requires no files and no optional dependency (no ``torch``).

    Determinism recipe (binding, stable within AND across processes):
    ``digest = hashlib.blake2b(numpy.ascontiguousarray(window,
    dtype=numpy.float32).tobytes(), digest_size=8).digest()``,
    ``n = int.from_bytes(digest, "big")``,
    ``gloss = labels[n % len(labels)]``,
    ``confidence = DEFAULT_MIN_CONFIDENCE + ((n >> 32) % 1000) / 1000.0 *
    (0.99 - DEFAULT_MIN_CONFIDENCE)``, which always lands in
    ``[0.60, 0.99]`` so an end-to-end mock run can actually emit. This never
    uses Python's string ``hash()``, unseeded ``random``, the wall clock, or
    object identity. Because the digest is taken over the exact float32
    bytes of the window, bit-level differences (for example ``-0.0`` versus
    ``0.0``, or distinct NaN payloads) can map to different answers.
    """

    def __init__(self, labels: Optional[Sequence[str]] = None):
        """Create a mock predictor.

        Parameters
        ----------
        labels : sequence of str, optional
            The gloss vocabulary to draw from (default
            :data:`src.utils.VOCABULARY`).

        Raises
        ------
        ValueError
            If ``labels`` is provided but empty.
        """
        self._labels = tuple(labels) if labels is not None else tuple(VOCABULARY)
        if not self._labels:
            raise ValueError("labels must be a non-empty sequence of gloss strings.")
        self._last_latency_ms = None

    @property
    def labels(self) -> list:
        """list[str]: A fresh copy of this predictor's gloss vocabulary."""
        return list(self._labels)

    @property
    def last_latency_ms(self) -> Optional[float]:
        """float or None: Milliseconds elapsed in the most recent :meth:`predict` call."""
        return self._last_latency_ms

    def predict(self, window):
        """Return a deterministic ``(gloss, confidence)`` pair for ``window``.

        Parameters
        ----------
        window : array-like
            Array of shape ``(SEQUENCE_LENGTH, N_FEATURES_PER_FRAME)``.

        Returns
        -------
        tuple[str, float]
            ``(gloss, confidence)`` with ``gloss`` in :attr:`labels` and
            ``confidence`` in ``[0.60, 0.99]``.

        Raises
        ------
        ValueError
            If ``window`` is not of shape
            ``(SEQUENCE_LENGTH, N_FEATURES_PER_FRAME)``.
        """
        arr = _validate_window(window)
        t0 = time.perf_counter()

        contiguous = np.ascontiguousarray(arr, dtype=np.float32)
        digest = hashlib.blake2b(contiguous.tobytes(), digest_size=8).digest()
        n = int.from_bytes(digest, "big")

        gloss = self._labels[n % len(self._labels)]
        span = 0.99 - DEFAULT_MIN_CONFIDENCE
        confidence = DEFAULT_MIN_CONFIDENCE + ((n >> 32) % 1000) / 1000.0 * span

        self._last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return gloss, float(confidence)


def _validate_checkpoint_payload(payload) -> None:
    """Validate the structural shape of a deserialized checkpoint mapping.

    Torch-free: operates on any ``Mapping`` (a plain ``dict`` in tests, or
    whatever ``torch.load`` returns in production), so it is directly
    unit-testable with synthetic dicts and no ``torch`` installed.

    Parameters
    ----------
    payload : object
        The deserialized checkpoint object.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``payload`` is not a ``Mapping``, or if any of the six required
        keys (``state_dict``, ``architecture``, ``hyperparams``,
        ``input_shape``, ``num_classes``, ``vocabulary``) is missing or
        holds a value of the wrong type. The message names every offending
        key, not just the first.
    """
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Checkpoint payload must be a mapping; got {type(payload).__name__}."
        )

    problems = []
    for key, expected_type in _CHECKPOINT_KEY_TYPES.items():
        if key not in payload:
            problems.append(f"missing key '{key}'")
        elif not isinstance(payload[key], expected_type):
            problems.append(
                f"key '{key}' has wrong type {type(payload[key]).__name__}"
            )

    if problems:
        raise ValueError(
            "Invalid checkpoint payload: " + "; ".join(problems) + ". Required "
            "keys: state_dict, architecture, hyperparams, input_shape, "
            "num_classes, vocabulary."
        )


def _validate_checkpoint_semantics(payload) -> None:
    """Validate the cross-field semantics of an already-structurally-valid payload.

    Torch-free: call only after :func:`_validate_checkpoint_payload` has
    succeeded.

    Parameters
    ----------
    payload : Mapping
        The deserialized checkpoint object.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``len(vocabulary) != num_classes`` (names both counts), or if
        ``tuple(input_shape) != (SEQUENCE_LENGTH, N_FEATURES_PER_FRAME)``
        (names both shapes; a train/inference layout mismatch is always
        fatal, never silently tolerated).

    Warns
    -----
    UserWarning
        If ``vocabulary`` differs from :data:`src.utils.VOCABULARY` while
        remaining internally self-consistent with ``num_classes``. The
        CHECKPOINT ordering remains authoritative for ``index -> gloss`` and
        becomes the predictor's ``labels``.
    """
    vocabulary = payload["vocabulary"]
    num_classes = payload["num_classes"]

    if len(vocabulary) != num_classes:
        raise ValueError(
            f"Checkpoint vocabulary has {len(vocabulary)} entries but "
            f"num_classes is {num_classes}; these must match."
        )

    expected_shape = (SEQUENCE_LENGTH, N_FEATURES_PER_FRAME)
    actual_shape = tuple(int(v) for v in payload["input_shape"])
    if actual_shape != expected_shape:
        raise ValueError(
            f"Checkpoint input_shape {actual_shape} does not match the "
            f"expected {expected_shape} (SEQUENCE_LENGTH, "
            "N_FEATURES_PER_FRAME); a train/inference layout mismatch is "
            "always treated as fatal."
        )

    if list(vocabulary) != list(VOCABULARY):
        warnings.warn(
            "Checkpoint vocabulary differs from src.utils.VOCABULARY; the "
            "checkpoint's ordering is authoritative for index -> gloss and "
            "is used as this predictor's labels.",
            stacklevel=2,
        )


def _resolve_model_builder():
    """Lazily resolve a checkpoint-to-model factory from the training module.

    Returns
    -------
    callable
        ``src.train.model_from_checkpoint``, with signature
        ``model_from_checkpoint(checkpoint) -> torch.nn.Module``. The
        checkpoint mapping is passed through unchanged (the same mapping
        deserialized by ``torch.load``), and the returned model already has
        its trained weights loaded via ``load_state_dict``.

    Raises
    ------
    RuntimeError
        If ``src.train`` cannot be imported (chained from the
        ``ImportError``), or if it does not expose a callable
        ``model_from_checkpoint``. The message names the remedy: pass an
        explicit ``model_builder`` to :class:`RealPredictor`, or ensure
        ``src.train`` provides the factory.
    """
    try:
        from src import train
    except ImportError as exc:
        raise RuntimeError(
            "Cannot resolve a model_builder: importing src.train failed "
            f"({exc}). Pass an explicit model_builder callable to "
            "RealPredictor, or ensure src.train defines "
            "model_from_checkpoint(checkpoint) -> torch.nn.Module."
        ) from exc

    builder = getattr(train, "model_from_checkpoint", None)
    if not callable(builder):
        raise RuntimeError(
            "src.train does not define a callable "
            "model_from_checkpoint(checkpoint) -> torch.nn.Module factory. "
            "Pass an explicit model_builder callable to RealPredictor "
            "instead."
        )
    return builder


class RealPredictor:
    """Loads ``models/sign_classifier.pt`` and runs the trained classifier.

    Loads EAGERLY in ``__init__`` (existence check, deserialize, validate,
    rebuild the architecture, load weights, ``eval()``), so no file I/O
    happens on the per-frame :meth:`predict` path and a failed load surfaces
    once, at construction, where a caller can show one clear error.

    ``torch`` is imported LAZILY, inside ``__init__`` only -- this is the
    only place this module ever imports ``torch`` -- and the module object
    is cached on the instance so :meth:`predict` never imports per call.
    Constructing / using :class:`MockPredictor` and importing this module
    both succeed on a machine with no ``torch`` installed.

    Trust boundary: ``torch.load`` can deserialize arbitrary Python objects.
    Only load checkpoints you produced yourself (e.g. via
    ``python -m src.train``); never load a checkpoint from an untrusted
    source.

    ``models/label_encoder.pkl`` is never opened, referenced as required, or
    needed anywhere in this class.

    No architecture is defined in this module (no ``torch.nn`` subclass):
    the architecture is rebuilt via an injectable ``model_builder`` callable
    (or a factory lazily resolved from ``src.train``), so ownership of model
    definitions stays with the training module. ``model_builder`` takes the
    whole checkpoint mapping and returns a model with its trained weights
    already loaded (``model_builder(checkpoint) -> torch.nn.Module``); this
    class does not call ``load_state_dict`` itself.

    VERIFICATION HONESTY: the ``torch``-dependent portion of this class (the
    lazy import, ``torch.load``, the architecture rebuild via a real
    ``src.train`` checkpoint, and the :meth:`predict` forward pass) IS
    exercised by ``tests/test_predict.py`` using a genuine checkpoint built
    with ``src.train.make_model`` / ``src.train.save_checkpoint`` on tiny
    synthetic tensors (skipped when ``torch`` is not installed). No trained
    or committed ``models/sign_classifier.pt`` is required or used.
    """

    def __init__(
        self,
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
        device: str = "cpu",
        model_builder=None,
    ):
        """Load and prepare the trained classifier.

        Parameters
        ----------
        checkpoint_path : str or pathlib.Path, optional
            Path to the ``.pt`` checkpoint (default
            :data:`DEFAULT_CHECKPOINT_PATH`).
        device : str, optional
            Torch device string used for ``map_location`` and the loaded
            model (default ``"cpu"``).
        model_builder : callable, optional
            ``model_builder(checkpoint) -> torch.nn.Module``, given the
            whole deserialized checkpoint mapping and returning a model with
            its trained weights already loaded. When ``None``, a factory is
            resolved lazily from ``src.train`` (see
            :func:`_resolve_model_builder`).

        Raises
        ------
        FileNotFoundError
            If ``checkpoint_path`` does not exist or is not a regular file
            (a directory takes this branch too). Raised BEFORE any
            ``torch`` import. The message names the missing absolute path,
            states that the file is produced by training and is
            gitignored/never committed, and lists three remedies: train the
            model, pass an explicit ``checkpoint_path``, or use
            :class:`MockPredictor` / ``create_recognizer(use_mock=True)``.
        RuntimeError
            If ``torch.load`` fails to deserialize the file (chained from
            the underlying exception, naming the path and the cause), or if
            no usable ``model_builder`` can be resolved (chained from
            ``ImportError`` when applicable).
        ValueError
            If the deserialized payload fails structural or semantic
            checkpoint validation (see :func:`_validate_checkpoint_payload`
            and :func:`_validate_checkpoint_semantics`).
        """
        path = Path(checkpoint_path)
        if not path.is_file():
            absolute = path.resolve()
            raise FileNotFoundError(
                f"Checkpoint not found at '{absolute}'. This file is "
                "produced by training (python -m src.train) and is "
                "gitignored / never committed. To proceed: (1) train the "
                "model so it is produced, (2) pass an explicit "
                "checkpoint_path to RealPredictor pointing at a checkpoint "
                "you already have, or (3) use MockPredictor / "
                "create_recognizer(use_mock=True) instead."
            )

        import torch  # Lazy: the only torch import in this module.

        self._torch = torch

        try:
            payload = torch.load(str(path), map_location=device, weights_only=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to deserialize checkpoint at '{path}': {exc}"
            ) from exc

        _validate_checkpoint_payload(payload)
        _validate_checkpoint_semantics(payload)

        builder = model_builder if model_builder is not None else _resolve_model_builder()
        model = builder(payload)
        model.to(device)
        model.eval()

        self._model = model
        self._device = device
        self._labels = tuple(payload["vocabulary"])
        self._last_latency_ms = None

    @property
    def labels(self) -> list:
        """list[str]: A fresh copy of the checkpoint's gloss vocabulary."""
        return list(self._labels)

    @property
    def last_latency_ms(self) -> Optional[float]:
        """float or None: Milliseconds elapsed in the most recent :meth:`predict` call."""
        return self._last_latency_ms

    def predict(self, window):
        """Run one forward pass and return the top-1 ``(gloss, confidence)``.

        Parameters
        ----------
        window : array-like
            Array of shape ``(SEQUENCE_LENGTH, N_FEATURES_PER_FRAME)``.

        Returns
        -------
        tuple[str, float]
            ``(gloss, confidence)`` with ``gloss`` in :attr:`labels` and
            ``0.0 <= confidence <= 1.0`` (softmax probability).

        Raises
        ------
        ValueError
            If ``window`` is not of shape
            ``(SEQUENCE_LENGTH, N_FEATURES_PER_FRAME)``.
        """
        arr = _validate_window(window)
        torch = self._torch

        t0 = time.perf_counter()
        with torch.no_grad():
            tensor = torch.as_tensor(
                arr, dtype=torch.float32, device=self._device
            ).unsqueeze(0)
            logits = self._model(tensor)
            probabilities = torch.softmax(logits, dim=-1)[0]
            top_index = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[top_index].item())
        self._last_latency_ms = (time.perf_counter() - t0) * 1000.0

        return self._labels[top_index], confidence


# --------------------------------------------------------------------------- #
# Composition layer
# --------------------------------------------------------------------------- #
class SignRecognizer:
    """Composes :class:`FrameBuffer`, a predictor and :class:`PredictionSmoother`.

    This is the per-frame object a live UI loop drives: call :meth:`update`
    once per landmark-extraction result and render the returned
    :class:`RecognitionResult`.
    """

    def __init__(
        self,
        predictor,
        length: int = SEQUENCE_LENGTH,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        stability_frames: int = DEFAULT_STABILITY_FRAMES,
        cooldown_frames: int = DEFAULT_COOLDOWN_FRAMES,
    ):
        """Create a recognizer around an already-constructed predictor.

        Parameters
        ----------
        predictor : object
            Any object implementing the predictor protocol (
            ``predict(window) -> (str, float)``, ``labels``,
            ``last_latency_ms``); typically a :class:`MockPredictor` or
            :class:`RealPredictor`.
        length : int, optional
            Window length passed to the internal :class:`FrameBuffer`
            (default :data:`src.utils.SEQUENCE_LENGTH`).
        min_confidence : float, optional
            Passed to the internal :class:`PredictionSmoother` (default
            :data:`DEFAULT_MIN_CONFIDENCE`).
        stability_frames : int, optional
            Passed to the internal :class:`PredictionSmoother` (default
            :data:`DEFAULT_STABILITY_FRAMES`).
        cooldown_frames : int, optional
            Passed to the internal :class:`PredictionSmoother` (default
            :data:`DEFAULT_COOLDOWN_FRAMES`).

        Raises
        ------
        ValueError
            Propagated unchanged from :class:`FrameBuffer` or
            :class:`PredictionSmoother` constructor validation.
        """
        self._predictor = predictor
        self._buffer = FrameBuffer(length=length)
        self._smoother = PredictionSmoother(
            min_confidence=min_confidence,
            stability_frames=stability_frames,
            cooldown_frames=cooldown_frames,
        )
        self._last_latency_ms = None

    @property
    def last_latency_ms(self) -> Optional[float]:
        """float or None: Milliseconds of the most recent inference.

        Measures ONLY the ``predictor.predict`` call made inside
        :meth:`update`; frame capture, landmark extraction and any display
        work are excluded. ``None`` before the first inference has run (and
        after :meth:`reset`, until the next inference runs).
        """
        return self._last_latency_ms

    def update(self, hands) -> RecognitionResult:
        """Append one frame and run at most one inference.

        Parameters
        ----------
        hands : iterable of (str, object) or None
            One frame's value from ``src.landmarks.extract``, passed
            through to :meth:`FrameBuffer.add_frame`.

        Returns
        -------
        RecognitionResult
            ``status="warming_up"`` while the window is not yet full (no
            inference runs; ``gloss``/``confidence``/``latency_ms``/
            ``emitted`` are all ``None``). ``status="no_hands"`` when the
            window is full but entirely zero (no inference runs; the
            smoother still receives one non-passing update so a stale
            agreement run cannot survive the hands leaving the frame and an
            active cooldown keeps ticking down; ``emitted`` is ``None``).
            ``status="predicted"`` otherwise, after exactly one
            ``predictor.predict`` call and one
            ``PredictionSmoother.update`` call.

        Raises
        ------
        ValueError
            Propagated from :meth:`FrameBuffer.add_frame` for malformed
            frame data.
        """
        self._buffer.add_frame(hands)

        if not self._buffer.is_ready:
            return RecognitionResult(
                gloss=None,
                confidence=None,
                emitted=None,
                latency_ms=None,
                frames_buffered=len(self._buffer),
                window_ready=False,
                status="warming_up",
            )

        window = self._buffer.window()

        if not window.any():
            self._smoother.update(None, None)
            return RecognitionResult(
                gloss=None,
                confidence=None,
                emitted=None,
                latency_ms=None,
                frames_buffered=len(self._buffer),
                window_ready=True,
                status="no_hands",
            )

        t0 = time.perf_counter()
        gloss, confidence = self._predictor.predict(window)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._last_latency_ms = latency_ms

        emitted = self._smoother.update(gloss, confidence)

        return RecognitionResult(
            gloss=gloss,
            confidence=confidence,
            emitted=emitted,
            latency_ms=latency_ms,
            frames_buffered=len(self._buffer),
            window_ready=True,
            status="predicted",
        )

    def reset(self) -> None:
        """Clear the frame buffer and all smoother state.

        Returns
        -------
        None

        The loaded predictor is left intact (no reload is triggered), so a
        UI "clear" control is cheap and does not risk re-raising a load
        error.
        """
        self._buffer.reset()
        self._smoother.reset()


def create_recognizer(
    use_mock: bool = False,
    checkpoint_path=DEFAULT_CHECKPOINT_PATH,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    stability_frames: int = DEFAULT_STABILITY_FRAMES,
    cooldown_frames: int = DEFAULT_COOLDOWN_FRAMES,
    device: str = "cpu",
    model_builder=None,
) -> SignRecognizer:
    """Build a ready :class:`SignRecognizer`.

    This is the single documented factory an external UI author should
    bind to. Typical usage, once per session::

        recognizer = create_recognizer(use_mock=True)  # or False, with a
                                                         # trained checkpoint
        # per frame:
        result = recognizer.update(hands)
        if result.emitted is not None:
            token_buffer.add(result.emitted)
        # on "finish sentence":
        sentence = glosses_to_sentence(token_buffer.emit())

    Parameters
    ----------
    use_mock : bool, optional
        If ``True``, builds a :class:`MockPredictor`: no checkpoint file and
        no ``torch`` import are touched. If ``False`` (default), builds a
        :class:`RealPredictor`, which loads ``checkpoint_path`` eagerly.
    checkpoint_path : str or pathlib.Path, optional
        Passed to :class:`RealPredictor` when ``use_mock=False`` (default
        :data:`DEFAULT_CHECKPOINT_PATH`). Ignored when ``use_mock=True``.
    min_confidence : float, optional
        PROVISIONAL tunable passed to the recognizer's smoother (default
        :data:`DEFAULT_MIN_CONFIDENCE`); retune once a real dataset exists.
    stability_frames : int, optional
        PROVISIONAL tunable passed to the recognizer's smoother (default
        :data:`DEFAULT_STABILITY_FRAMES`); counted in updates, not seconds.
    cooldown_frames : int, optional
        PROVISIONAL tunable passed to the recognizer's smoother (default
        :data:`DEFAULT_COOLDOWN_FRAMES`); counted in updates, not seconds.
    device : str, optional
        Passed to :class:`RealPredictor` when ``use_mock=False`` (default
        ``"cpu"``). Ignored when ``use_mock=True``.
    model_builder : callable, optional
        Passed to :class:`RealPredictor` when ``use_mock=False`` (default
        ``None``, meaning a factory is resolved lazily from ``src.train``;
        see :func:`_resolve_model_builder`). Ignored when ``use_mock=True``.

    Returns
    -------
    SignRecognizer
        A recognizer ready to receive :meth:`SignRecognizer.update` calls.

    Raises
    ------
    FileNotFoundError, RuntimeError, ValueError
        Propagated UNCHANGED from :class:`RealPredictor` when
        ``use_mock=False`` and the checkpoint is missing, unreadable, or
        fails validation. This factory never silently falls back to the
        mock predictor on a real-load failure.
    """
    if use_mock:
        predictor = MockPredictor()
    else:
        predictor = RealPredictor(
            checkpoint_path=checkpoint_path, device=device, model_builder=model_builder,
        )

    return SignRecognizer(
        predictor,
        min_confidence=min_confidence,
        stability_frames=stability_frames,
        cooldown_frames=cooldown_frames,
    )
