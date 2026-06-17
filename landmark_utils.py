"""
Landmark Data Utilities  (mediapipe 0.10.30+, Python 3.13)
===========================================================
Load, inspect, and normalize saved CSV / NumPy landmark files.

Usage:
  python landmark_utils.py --csv captured_landmarks/landmarks_*.csv
  python landmark_utils.py --npy captured_landmarks/landmarks_*.npy --normalize
  python landmark_utils.py --npy captured_landmarks/landmarks_*.npy --normalize --save-normalized
"""

import argparse
import csv
import numpy as np
from pathlib import Path

META_COLS       = 4    # timestamp, frame, hand_index, handedness
N_LANDMARKS     = 21
COORDS_PER_LM   = 3   # x, y, z
N_FEATURES      = N_LANDMARKS * COORDS_PER_LM   # 63

LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC",  "THUMB_MCP",  "THUMB_IP",   "THUMB_TIP",
    "INDEX_MCP",  "INDEX_PIP",  "INDEX_DIP",  "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP",   "RING_PIP",   "RING_DIP",   "RING_TIP",
    "PINKY_MCP",  "PINKY_PIP",  "PINKY_DIP",  "PINKY_TIP",
]


def load_csv(path: str):
    meta_rows, feat_rows = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            meta_rows.append({
                "timestamp":  float(row["timestamp"]),
                "frame":      int(row["frame"]),
                "hand_index": int(row["hand_index"]),
                "handedness": row["handedness"],
            })
            feat_rows.append([
                float(row[f"{name}_{ax}"])
                for name in LANDMARK_NAMES for ax in ("x", "y", "z")
            ])
    return meta_rows, np.array(feat_rows, dtype=np.float32)


def load_npy(path: str):
    arr = np.load(path, allow_pickle=True)
    meta_rows = [
        {"timestamp": float(r[0]), "frame": int(r[1]),
         "hand_index": int(r[2]),  "handedness": str(r[3])}
        for r in arr
    ]
    return meta_rows, arr[:, META_COLS:].astype(np.float32)


def normalize_landmarks(features: np.ndarray) -> np.ndarray:
    """Translate to wrist origin, then scale by max point distance."""
    out = features.copy()
    for i, row in enumerate(out):
        pts   = row.reshape(N_LANDMARKS, COORDS_PER_LM)
        pts  -= pts[0].copy()
        scale = np.max(np.linalg.norm(pts, axis=1))
        if scale > 1e-6:
            pts /= scale
        out[i] = pts.reshape(-1)
    return out


def summarise(path, meta, features, normalised=False):
    print(f"\n{'='*60}")
    print(f"  File      : {path}")
    print(f"  Samples   : {len(meta)}")
    if meta:
        frames = [m["frame"]     for m in meta]
        times  = [m["timestamp"] for m in meta]
        left   = sum(1 for m in meta if m["handedness"] == "Left")
        right  = sum(1 for m in meta if m["handedness"] == "Right")
        dur    = times[-1] - times[0] if len(times) > 1 else 0
        print(f"  Frames    : {min(frames)} – {max(frames)}")
        print(f"  Duration  : {dur:.2f} s")
        print(f"  Hands     : {left} Left  |  {right} Right")
    tag = "(normalised)" if normalised else "(raw)"
    print(f"  Features  : shape={features.shape}  {tag}")
    if features.size:
        print(f"  x range   : [{features[:,0::3].min():.4f}, {features[:,0::3].max():.4f}]")
        print(f"  y range   : [{features[:,1::3].min():.4f}, {features[:,1::3].max():.4f}]")
        print(f"  z range   : [{features[:,2::3].min():.4f}, {features[:,2::3].max():.4f}]")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", nargs="+")
    group.add_argument("--npy", nargs="+")
    parser.add_argument("--normalize",       action="store_true")
    parser.add_argument("--save-normalized", action="store_true")
    args = parser.parse_args()

    paths  = args.csv or args.npy
    loader = load_csv if args.csv else load_npy

    for path in paths:
        try:
            meta, features = loader(path)
        except Exception as e:
            print(f"[ERROR] {path}: {e}")
            continue

        if args.normalize:
            features = normalize_landmarks(features)

        summarise(path, meta, features, normalised=args.normalize)

        if args.save_normalized and args.normalize:
            out_path = Path(path).with_suffix(".normalized.npy")
            arr_meta = np.array([[m["timestamp"], m["frame"],
                                  m["hand_index"], m["handedness"]]
                                 for m in meta], dtype=object)
            np.save(out_path, np.hstack([arr_meta, features]))
            print(f"  [SAVE] → {out_path}")


if __name__ == "__main__":
    main()
