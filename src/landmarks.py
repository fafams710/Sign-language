"""Hand and pose landmark extraction.

Wraps MediaPipe's HandLandmarker to detect hand landmarks from each frame and
returns coordinate features for downstream sequence processing, plus the
verbatim feature ordering (:data:`LANDMARK_NAMES`) and the wrist-relative
:func:`normalize_landmarks` used across the pipeline.

Import safety: all MediaPipe imports are deferred into the detector factory and
the per-frame extraction call, and the model-path existence check happens only
at detector initialization. Importing this module therefore succeeds even when
MediaPipe or the model binary is absent. NumPy is a hard, already-present
dependency and is imported at module top.

Model file: the HandLandmarker ``.task`` binary is user-provided and is NEVER
committed (it is large and gitignored via ``*.task``). Place it at
:data:`MODEL_PATH` (``models/hand_landmarker.task`` relative to the repository
root) or pass an explicit path to :func:`create_detector`.
"""

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "hand_landmarker.task"

N_LANDMARKS = 21
COORDS_PER_LM = 3
N_FEATURES = N_LANDMARKS * COORDS_PER_LM

LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]


def normalize_landmarks(features: np.ndarray) -> np.ndarray:
    """Translate to wrist origin, then scale by max point distance."""
    out = features.copy()
    for i, row in enumerate(out):
        pts = row.reshape(N_LANDMARKS, COORDS_PER_LM)
        pts -= pts[0].copy()
        scale = np.max(np.linalg.norm(pts, axis=1))
        if scale > 1e-6:
            pts /= scale
        out[i] = pts.reshape(-1)
    return out


def create_detector(model_path=MODEL_PATH, num_hands=2,
                    det_conf=0.5, track_conf=0.5):
    """Build and return a MediaPipe HandLandmarker in VIDEO running mode.

    MediaPipe Tasks symbols are imported here (not at module level) so the
    module stays importable without MediaPipe installed. The model-path
    existence check is performed here and raises a clear, path-naming error if
    the user-provided ``.task`` binary is missing.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            "HandLandmarker model not found at "
            f"'{model_path}'. This file is user-provided and never committed; "
            "download 'hand_landmarker.task' and place it at that path, or "
            "pass an explicit model_path to create_detector()."
        )

    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        RunningMode,
    )

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=det_conf,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=track_conf,
    )
    return HandLandmarker.create_from_options(options)


def extract(detector, frame_bgr, ts_ms):
    """Run the detector on one BGR frame and return per-hand landmarks.

    Converts BGR -> RGB, wraps the frame in a MediaPipe ``Image``, and calls
    ``detect_for_video`` with the given strictly-increasing millisecond
    timestamp. Returns a list of ``(handedness, landmarks)`` tuples, where
    ``handedness`` is the display name string and ``landmarks`` is the list of
    21 landmark objects exposing ``.x``, ``.y``, ``.z``.
    """
    import cv2
    from mediapipe import Image, ImageFormat

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_img, ts_ms)

    hands = []
    if result.hand_landmarks:
        for hand_lms, hand_info in zip(result.hand_landmarks,
                                       result.handedness):
            handedness = hand_info[0].display_name
            hands.append((handedness, hand_lms))
    return hands
