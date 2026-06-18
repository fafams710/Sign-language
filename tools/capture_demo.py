"""Interactive landmark capture demo.

Live OpenCV viewer that opens a webcam via :mod:`src.camera`, extracts hand
landmarks via :mod:`src.landmarks`, draws them with index labels and a HUD, and
lets the user save / record landmark buffers to ``captured_landmarks/`` at the
repository root.

Controls:
  S       - Save current frame landmarks to CSV + NumPy
  R       - Start / Stop session recording (auto-saves on stop)
  L       - Toggle landmark index labels (0-20)
  C       - Clear data buffer
  Q / ESC - Quit (auto-saves a non-empty buffer)

Run from the repository root so that ``src.*`` and ``tools.*`` resolve:
  python -m tools.capture_demo
"""

import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src import landmarks
from src.camera import Camera, CameraError

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "captured_landmarks"

META_COLS = 4  # timestamp, frame, hand_index, handedness

LANDMARK_NAMES = landmarks.LANDMARK_NAMES
CSV_HEADER = ["timestamp", "frame", "hand_index", "handedness"] + [
    f"{n}_{ax}" for n in LANDMARK_NAMES for ax in ("x", "y", "z")
]

CLR_LEFT = (255, 100, 50)
CLR_RIGHT = (50, 180, 255)
CLR_CONN = (220, 220, 220)
CLR_GREEN = (0, 255, 80)
CLR_RED = (0, 60, 220)
CLR_YELLOW = (0, 220, 220)


def draw_hand(frame, hand_lms, label):
    import cv2
    from mediapipe.tasks.python.vision import HandLandmarksConnections

    h, w = frame.shape[:2]
    color = CLR_LEFT if label == "Left" else CLR_RIGHT
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]

    for conn in HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, pts[conn.start], pts[conn.end], CLR_CONN, 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 6, color, -1)
        cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)

    wx, wy = pts[0]
    cv2.putText(frame, label, (wx, max(wy - 20, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)


def draw_labels(frame, hand_lms):
    import cv2

    h, w = frame.shape[:2]
    for i, lm in enumerate(hand_lms):
        cv2.putText(frame, str(i),
                    (int(lm.x * w) + 7, int(lm.y * h) - 5),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 0), 1)


def draw_hud(frame, fps, frame_idx, recording, buf_len, hands_found):
    import cv2

    h, w = frame.shape[:2]

    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 105), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.5, frame, 0.5, 0, frame)

    fps_color = CLR_GREEN if fps >= 20 else CLR_RED
    cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)

    hand_txt = f"Hands: {hands_found}" if hands_found else "No hands detected"
    hand_color = CLR_GREEN if hands_found else (100, 100, 100)
    cv2.putText(frame, hand_txt, (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, hand_color, 2)

    rec_color = CLR_RED if recording else (160, 160, 160)
    rec_txt = f"REC  {buf_len} rows" if recording else f"IDLE  buf={buf_len}"
    cv2.putText(frame, rec_txt, (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, rec_color, 2)

    cv2.rectangle(frame, (0, h - 28), (w, h), (0, 0, 0), -1)
    cv2.putText(frame,
                "S:save frame   R:record session   L:labels   C:clear   Q:quit",
                (8, h - 9), cv2.FONT_HERSHEY_PLAIN, 1.0, (180, 180, 180), 1)


def draw_startup_screen(frame, countdown):
    """Show a countdown overlay while the camera warms up."""
    import cv2

    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "SIGN LANGUAGE CAPTURE", (w // 2 - 220, h // 2 - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, CLR_GREEN, 2)
    cv2.putText(frame, "Camera warming up...", (w // 2 - 170, h // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    cv2.putText(frame, f"Starting in {countdown}s", (w // 2 - 130, h // 2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, CLR_YELLOW, 2)
    cv2.putText(frame, "Place your hand in front of the camera",
                (w // 2 - 270, h // 2 + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)


def landmarks_to_row(ts, fidx, hidx, handedness, hand_lms):
    row = [ts, fidx, hidx, handedness]
    for lm in hand_lms:
        row.extend([round(lm.x, 6), round(lm.y, 6), round(lm.z, 6)])
    return row


def save_buffer(buffer, tag=""):
    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"landmarks_{ts}" + (f"_{tag}" if tag else "")
    csv_p = OUTPUT_DIR / f"{stem}.csv"
    npy_p = OUTPUT_DIR / f"{stem}.npy"
    with open(csv_p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        w.writerows(buffer)
    np.save(npy_p, np.array(buffer, dtype=object))
    return str(csv_p), str(npy_p)


def load_csv(path: str):
    meta_rows, feat_rows = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            meta_rows.append({
                "timestamp": float(row["timestamp"]),
                "frame": int(row["frame"]),
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
         "hand_index": int(r[2]), "handedness": str(r[3])}
        for r in arr
    ]
    return meta_rows, arr[:, META_COLS:].astype(np.float32)


def run_capture(camera_index=0, det_conf=0.5, track_conf=0.5, max_hands=2):
    import cv2

    print(f"\n[INFO] Opening camera {camera_index}...")
    camera = Camera(camera_index=camera_index)
    try:
        opened_index = camera.open()
    except CameraError as exc:
        print(f"[ERROR] {exc}")
        return
    print(f"[INFO] Camera opened at index {opened_index}")
    print(f"[INFO] Model      : {landmarks.MODEL_PATH.name}")

    try:
        detector = landmarks.create_detector(
            num_hands=max_hands, det_conf=det_conf, track_conf=track_conf)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        camera.release()
        return

    data_buffer = []
    recording = False
    show_labels = False
    frame_idx = 0
    frame_times = []
    start_ms = int(time.time() * 1000)

    with detector:
        print("[INFO] Detector ready!")
        print("[INFO] Window opening - put your hand in front of the camera.\n")

        warmup_secs = 3
        warmup_end = time.time() + warmup_secs
        while time.time() < warmup_end:
            ret, frame = camera.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            countdown = max(1, int(warmup_end - time.time()) + 1)
            draw_startup_screen(frame, countdown)
            cv2.imshow("Sign Language Capture", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                camera.release()
                cv2.destroyAllWindows()
                return

        while True:
            t0 = time.perf_counter()
            ret, frame = camera.read()
            if not ret:
                print("[WARN] Frame grab failed - retrying...")
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)
            ts_ms = int(time.time() * 1000) - start_ms
            hands = landmarks.extract(detector, frame, ts_ms)

            ts_s = time.time()
            detected_rows = []
            hands_found = len(hands)

            for hi, (handedness, hand_lms) in enumerate(hands):
                draw_hand(frame, hand_lms, handedness)
                if show_labels:
                    draw_labels(frame, hand_lms)
                detected_rows.append(
                    landmarks_to_row(ts_s, frame_idx, hi,
                                     handedness, hand_lms))

            if recording:
                data_buffer.extend(detected_rows)

            frame_times.append(time.perf_counter() - t0)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps = 1.0 / (sum(frame_times) / len(frame_times))

            draw_hud(frame, fps, frame_idx, recording,
                     len(data_buffer), hands_found)
            cv2.imshow("Sign Language Capture", frame)
            frame_idx += 1

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                if detected_rows:
                    csv_p, _ = save_buffer(detected_rows, "frame")
                    print(f"[SAVE] Frame {frame_idx} -> {Path(csv_p).name}")
                else:
                    print("[SKIP] No hands in current frame.")
            elif key == ord("r"):
                recording = not recording
                if recording:
                    data_buffer.clear()
                    print("[REC ] Recording started - show your hand signs!")
                else:
                    if data_buffer:
                        csv_p, _ = save_buffer(data_buffer, "session")
                        print(f"[REC ] Saved {len(data_buffer)} rows -> "
                              f"{Path(csv_p).name}")
                    else:
                        print("[REC ] Stopped - no data was captured.")
            elif key == ord("l"):
                show_labels = not show_labels
                print(f"[INFO] Labels {'ON' if show_labels else 'OFF'}")
            elif key == ord("c"):
                data_buffer.clear()
                print("[INFO] Buffer cleared.")

    camera.release()
    cv2.destroyAllWindows()
    if data_buffer:
        csv_p, _ = save_buffer(data_buffer, "exit")
        print(f"[EXIT] Auto-saved {len(data_buffer)} rows -> "
              f"{Path(csv_p).name}")
    print(f"[EXIT] Done - {frame_idx} frames processed.")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--hands", type=int, default=2)
    parser.add_argument("--det-conf", type=float, default=0.5)
    parser.add_argument("--track-conf", type=float, default=0.5)
    args = parser.parse_args()
    run_capture(args.camera, args.det_conf, args.track_conf, args.hands)


if __name__ == "__main__":
    main()
