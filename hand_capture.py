
"""
Controls:
  S       - Save current frame landmarks to CSV + NumPy
  R       - Start / Stop session recording (auto-saves on stop)
  L       - Toggle landmark index labels (0-30)
  C       - Clear data buffer
  Q / ESC - Quit
"""
# tested all the controls working naman somehow??

import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import sys
from datetime import datetime
from pathlib import Path

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
)
from mediapipe import Image, ImageFormat


SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR / "hand_landmarker.task"
OUTPUT_DIR = SCRIPT_DIR / "captured_landmarks"
OUTPUT_DIR.mkdir(exist_ok=True)

if not MODEL_PATH.exists():
    print(f"[ERROR] hand_landmarker.task not found in {SCRIPT_DIR}")
    sys.exit(1)


LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC","THUMB_MCP","THUMB_IP","THUMB_TIP",
    "INDEX_MCP","INDEX_PIP","INDEX_DIP","INDEX_TIP",
    "MIDDLE_MCP","MIDDLE_PIP","MIDDLE_DIP","MIDDLE_TIP",
    "RING_MCP","RING_PIP","RING_DIP","RING_TIP",
    "PINKY_MCP","PINKY_PIP","PINKY_DIP","PINKY_TIP",
]
CSV_HEADER = ["timestamp","frame","hand_index","handedness"] + [
    f"{n}_{ax}" for n in LANDMARK_NAMES for ax in ("x","y","z")
]


CLR_LEFT   = (255, 100,  50)
CLR_RIGHT  = ( 50, 180, 255)
CLR_CONN   = (220, 220, 220)
CLR_GREEN  = (  0, 255,  80)
CLR_RED    = (  0,  60, 220)
CLR_YELLOW = (  0, 220, 220)



def draw_hand(frame, hand_lms, label):
    h, w = frame.shape[:2]
    color = CLR_LEFT if label == "Left" else CLR_RIGHT
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]

    for conn in HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, pts[conn.start], pts[conn.end], CLR_CONN, 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 6, color, -1)
        cv2.circle(frame, (x, y), 6, (255,255,255), 1)

    wx, wy = pts[0]
    cv2.putText(frame, label, (wx, max(wy - 20, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)


def draw_labels(frame, hand_lms):
    h, w = frame.shape[:2]
    for i, lm in enumerate(hand_lms):
        cv2.putText(frame, str(i),
                    (int(lm.x*w)+7, int(lm.y*h)-5),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, (255,255,0), 1)


def draw_hud(frame, fps, frame_idx, recording, buf_len, hands_found):
    h, w = frame.shape[:2]

    # Dark top bar
    bar = frame.copy()
    cv2.rectangle(bar, (0,0), (w, 105), (0,0,0), -1)
    cv2.addWeighted(bar, 0.5, frame, 0.5, 0, frame)

    # FPS — green if good, red if low (Current max 30fps this good? reduced to 680x480 for better performance)
    fps_color = CLR_GREEN if fps >= 20 else CLR_RED
    cv2.putText(frame, f"FPS: {fps:5.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)

    # Hand detection indicator
    hand_txt   = f"Hands: {hands_found}" if hands_found else "No hands detected"
    hand_color = CLR_GREEN if hands_found else (100,100,100)
    cv2.putText(frame, hand_txt, (10,58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, hand_color, 2)

    # Buffer / recording
    rec_color = CLR_RED if recording else (160,160,160)
    rec_txt   = f"REC  {buf_len} rows" if recording else f"IDLE  buf={buf_len}"
    cv2.putText(frame, rec_txt, (10,82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, rec_color, 2)

    # Bottom help bar
    cv2.rectangle(frame, (0, h-28), (w, h), (0,0,0), -1)
    cv2.putText(frame, "S:save frame   R:record session   L:labels   C:clear   Q:quit",
                (8, h-9), cv2.FONT_HERSHEY_PLAIN, 1.0, (180,180,180), 1)


def draw_startup_screen(frame, countdown):
    """Show a countdown overlay while camera warms up 3 seconds."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w,h), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "SIGN LANGUAGE CAPTURE", (w//2-220, h//2-60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, CLR_GREEN, 2)
    cv2.putText(frame, "Camera warming up...", (w//2-170, h//2-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
    cv2.putText(frame, f"Starting in {countdown}s", (w//2-130, h//2+40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, CLR_YELLOW, 2)
    cv2.putText(frame, "Place your hand in front of the camera",
                (w//2-270, h//2+90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180,180,180), 1)


def landmarks_to_row(ts, fidx, hidx, handedness, hand_lms):
    row = [ts, fidx, hidx, handedness]
    for lm in hand_lms:
        row.extend([round(lm.x,6), round(lm.y,6), round(lm.z,6)])
    return row


def save_buffer(buffer, tag=""):
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"landmarks_{ts}" + (f"_{tag}" if tag else "")
    csv_p = OUTPUT_DIR / f"{stem}.csv"
    npy_p = OUTPUT_DIR / f"{stem}.npy"
    with open(csv_p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        w.writerows(buffer)
    np.save(npy_p, np.array(buffer, dtype=object))
    return str(csv_p), str(npy_p)


def run_capture(camera_index=0, det_conf=0.5, track_conf=0.5, max_hands=2):

    # ── Try to open camera ────────────────────────────────────────────────────
    print(f"\n[INFO] Opening camera {camera_index}...")
    cap = None
    for idx in ([camera_index] if camera_index != 0 else [0, 1, 2]):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)   # CAP_DSHOW = faster on Windows claude recommended ts
        if cap.isOpened():
            print(f"[INFO] Camera opened at index {idx}")
            break
        cap.release()
    else:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("[ERROR] Cannot open any camera. Check it is connected and not in use.")
            sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # i dunno what this do im vibe codin

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Resolution : {w}x{h}")
    print(f"[INFO] Model      : {MODEL_PATH.name}")

    # Detector 
    opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=max_hands,
        min_hand_detection_confidence=det_conf,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=track_conf,
    )

    data_buffer  = []
    recording    = False
    show_labels  = False
    frame_idx    = 0
    frame_times  = []
    start_ms     = int(time.time() * 1000)

    with HandLandmarker.create_from_options(opts) as detector:
        print("[INFO] Detector ready!")
        print("[INFO] Window opening — put your hand in front of the camera.\n")

        warmup_secs = 3
        warmup_end  = time.time() + warmup_secs
        while time.time() < warmup_end:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            countdown = max(1, int(warmup_end - time.time()) + 1)
            draw_startup_screen(frame, countdown)
            cv2.imshow("Sign Language Capture", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                cap.release()
                cv2.destroyAllWindows()
                return

        # ── Main loop ─────────────────────────────────────────────────────────
        while True:
            t0  = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame grab failed — retrying…")
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)

            ts_ms  = int(time.time() * 1000) - start_ms
            result = detector.detect_for_video(mp_img, ts_ms)

            ts_s          = time.time()
            detected_rows = []
            hands_found   = 0

            if result.hand_landmarks:
                hands_found = len(result.hand_landmarks)
                for hi, (hand_lms, hand_info) in enumerate(
                    zip(result.hand_landmarks, result.handedness)
                ):
                    handedness = hand_info[0].display_name
                    draw_hand(frame, hand_lms, handedness)
                    if show_labels:
                        draw_labels(frame, hand_lms)
                    detected_rows.append(
                        landmarks_to_row(ts_s, frame_idx, hi,
                                         handedness, hand_lms))

            if recording:
                data_buffer.extend(detected_rows)

            # FPS
            frame_times.append(time.perf_counter() - t0)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps = 1.0 / (sum(frame_times) / len(frame_times))

            draw_hud(frame, fps, frame_idx, recording,
                     len(data_buffer), hands_found)
            cv2.imshow("Sign Language Capture", frame)
            frame_idx += 1

            # Keys
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                if detected_rows:
                    csv_p, _ = save_buffer(detected_rows, "frame")
                    print(f"[SAVE] Frame {frame_idx} → {Path(csv_p).name}")
                else:
                    print("[SKIP] No hands in current frame.")
            elif key == ord("r"):
                recording = not recording
                if recording:
                    data_buffer.clear()
                    print("[REC ] Recording started — show your hand signs!")
                else:
                    if data_buffer:
                        csv_p, _ = save_buffer(data_buffer, "session")
                        print(f"[REC ] Saved {len(data_buffer)} rows → {Path(csv_p).name}")
                    else:
                        print("[REC ] Stopped — no data was captured.")
            elif key == ord("l"):
                show_labels = not show_labels
                print(f"[INFO] Labels {'ON' if show_labels else 'OFF'}")
            elif key == ord("c"):
                data_buffer.clear()
                print("[INFO] Buffer cleared.")

    cap.release()
    cv2.destroyAllWindows()
    if data_buffer:
        csv_p, _ = save_buffer(data_buffer, "exit")
        print(f"[EXIT] Auto-saved {len(data_buffer)} rows → {Path(csv_p).name}")
    print(f"[EXIT] Done — {frame_idx} frames processed.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--camera",     type=int,   default=0)
    p.add_argument("--hands",      type=int,   default=2)
    p.add_argument("--det-conf",   type=float, default=0.5)
    p.add_argument("--track-conf", type=float, default=0.5)
    a = p.parse_args()
    run_capture(a.camera, a.det_conf, a.track_conf, a.hands)

