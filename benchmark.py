"""
Frame Rate & Stability Benchmark  (mediapipe 0.10.30+, Python 3.13)
====================================================================
Headless test — no camera required. Uses synthetic frames.
Requires: hand_landmarker.task in the same folder.

Usage:
  python benchmark.py
  python benchmark.py --frames 300 --complexity 0
"""

import argparse
import csv
import time
import sys
from pathlib import Path

import cv2
import numpy as np

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from mediapipe import Image, ImageFormat

SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR / "hand_landmarker.task"
OUTPUT_DIR = SCRIPT_DIR / "captured_landmarks"
OUTPUT_DIR.mkdir(exist_ok=True)

if not MODEL_PATH.exists():
    print(f"[ERROR] Model file not found: {MODEL_PATH}")
    sys.exit(1)


def make_test_frame(width=640, height=480):
    """Synthetic BGR frame with a hand-like shape + noise."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.ellipse(frame, (width // 2, height // 2),
                (90, 120), 0, 0, 360, (120, 160, 200), -1)
    for i, x in enumerate(range(200, 440, 45)):
        cv2.rectangle(frame, (x, height // 2 - 130),
                      (x + 30, height // 2 - 10),
                      (100 + i * 20, 150, 180), -1)
    noise = np.random.randint(0, 40, (height, width, 3), dtype=np.uint8)
    return cv2.add(frame, noise)


def run_benchmark(n_frames=200, frame_width=640,
                  frame_height=480, max_hands=2):

    print(f"\n{'='*60}")
    print(f"  Sign Language Capture — Frame-Rate Benchmark")
    print(f"{'='*60}")
    print(f"  Frames     : {n_frames}")
    print(f"  Resolution : {frame_width}x{frame_height}")
    print(f"  Max hands  : {max_hands}")
    print(f"  Model      : {MODEL_PATH.name}")
    print(f"{'='*60}\n")

    opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=max_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    grab_times, infer_times, total_times, detections = [], [], [], []

    with HandLandmarker.create_from_options(opts) as detector:
        # Warm-up (not counted)
        dummy = make_test_frame(frame_width, frame_height)
        dummy_rgb = cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB)
        dummy_mp  = Image(image_format=ImageFormat.SRGB, data=dummy_rgb)
        for i in range(5):
            detector.detect_for_video(dummy_mp, i)

        print(f"  {'Frame':>6}  {'Grab ms':>8}  {'Infer ms':>9}  "
              f"{'Total ms':>9}  {'Hands':>5}  {'FPS':>7}")
        print(f"  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*5}  {'-'*7}")

        start_ms = int(time.time() * 1000)

        for i in range(n_frames):
            t0 = time.perf_counter()
            frame     = make_test_frame(frame_width, frame_height)
            t1 = time.perf_counter()

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
            ts_ms  = int(time.time() * 1000) - start_ms
            result = detector.detect_for_video(mp_img, ts_ms)
            t2 = time.perf_counter()

            grab_ms  = (t1 - t0) * 1000
            infer_ms = (t2 - t1) * 1000
            total_ms = (t2 - t0) * 1000
            n_hands  = len(result.hand_landmarks) if result.hand_landmarks else 0
            fps_inst = 1000.0 / total_ms if total_ms > 0 else 0

            grab_times.append(grab_ms)
            infer_times.append(infer_ms)
            total_times.append(total_ms)
            detections.append(n_hands)

            if i % 25 == 0 or i == n_frames - 1:
                print(f"  {i:>6}  {grab_ms:>8.2f}  {infer_ms:>9.2f}  "
                      f"{total_ms:>9.2f}  {n_hands:>5}  {fps_inst:>7.1f}")

    arr = np.array(total_times)
    stats = {
        "n_frames":      n_frames,
        "resolution":    f"{frame_width}x{frame_height}",
        "mean_fps":      round(1000.0 / arr.mean(), 2),
        "min_fps":       round(1000.0 / arr.max(),  2),
        "max_fps":       round(1000.0 / arr.min(),  2),
        "mean_ms":       round(arr.mean(), 3),
        "std_ms":        round(arr.std(),  3),
        "p50_ms":        round(float(np.percentile(arr, 50)), 3),
        "p95_ms":        round(float(np.percentile(arr, 95)), 3),
        "p99_ms":        round(float(np.percentile(arr, 99)), 3),
        "max_ms":        round(arr.max(), 3),
        "jitter_cv":     round(arr.std() / arr.mean(), 4),
        "total_detects": int(sum(detections)),
    }
    return stats, grab_times, infer_times, total_times, detections


def save_results(grab_times, infer_times, total_times, detections):
    csv_path = OUTPUT_DIR / "benchmark_results.csv"
    npy_path = OUTPUT_DIR / "benchmark_results.npy"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "grab_ms", "infer_ms", "total_ms", "n_hands"])
        for i, (g, p, t, d) in enumerate(
                zip(grab_times, infer_times, total_times, detections)):
            w.writerow([i, round(g,4), round(p,4), round(t,4), d])
    arr = np.array(list(zip(grab_times, infer_times,
                            total_times, detections)), dtype=np.float32)
    np.save(npy_path, arr)
    return str(csv_path), str(npy_path)


def print_summary(stats):
    jitter_label = ("LOW ✓"    if stats["jitter_cv"] < 0.15 else
                    "MEDIUM ⚠" if stats["jitter_cv"] < 0.30 else "HIGH ✗")
    print(f"\n{'='*60}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"  Resolution      : {stats['resolution']}")
    print(f"  Frames tested   : {stats['n_frames']}")
    print(f"")
    print(f"  ── Throughput ──────────────────────────")
    print(f"  Mean FPS        : {stats['mean_fps']:>8.1f}")
    print(f"  Min  FPS        : {stats['min_fps']:>8.1f}")
    print(f"  Max  FPS        : {stats['max_fps']:>8.1f}")
    print(f"")
    print(f"  ── Latency (ms/frame) ──────────────────")
    print(f"  Mean            : {stats['mean_ms']:>8.2f} ms")
    print(f"  Std dev         : {stats['std_ms']:>8.2f} ms")
    print(f"  p50 (median)    : {stats['p50_ms']:>8.2f} ms")
    print(f"  p95             : {stats['p95_ms']:>8.2f} ms")
    print(f"  p99             : {stats['p99_ms']:>8.2f} ms")
    print(f"  Max spike       : {stats['max_ms']:>8.2f} ms")
    print(f"")
    print(f"  ── Stability ───────────────────────────")
    print(f"  Jitter (CV)     : {stats['jitter_cv']:>8.4f}  → {jitter_label}")
    print(f"  Total detections: {stats['total_detects']:>8}")
    print(f"{'='*60}")
    target = 30
    if stats["mean_fps"] >= target:
        print(f"\n  ✅  Meets {target} FPS real-time target.")
    else:
        print(f"\n  ⚠️   {target - stats['mean_fps']:.1f} FPS below target — "
              f"try --complexity 0 or lower resolution.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--width",  type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hands",  type=int, default=2)
    args = parser.parse_args()

    stats, grab_t, infer_t, total_t, dets = run_benchmark(
        args.frames, args.width, args.height, args.hands)
    print_summary(stats)
    csv_p, npy_p = save_results(grab_t, infer_t, total_t, dets)
    print(f"\n  Raw data saved:")
    print(f"    CSV: {csv_p}")
    print(f"    NPY: {npy_p}\n")


if __name__ == "__main__":
    main()
