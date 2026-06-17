# Sign Language Capture System

A modular Python pipeline for capturing, detecting, and saving hand landmarks
for sign language recognition research and ML model training.

---

## Files

| File | Purpose |
|------|---------|
| `hand_capture.py` | **Main capture script** — live webcam + MediaPipe landmarks |
| `landmark_utils.py` | Load, inspect, and normalize saved CSV / NumPy data |
| `benchmark.py` | Headless FPS & latency benchmark (no camera needed) |
| `captured_landmarks/` | Auto-created output directory for all saved data |

---

## Quick Start

### 1. Install dependencies
```bash
pip install opencv-python mediapipe numpy
```

### 2. Run the benchmark first (no camera needed)
```bash
python benchmark.py
```
This tests your machine's throughput and writes results to
`captured_landmarks/benchmark_results.csv` and `.npy`.

### 3. Run live capture
```bash
python hand_capture.py
```

With options:
```bash
python hand_capture.py --camera 0 --hands 2 --det-conf 0.7 --track-conf 0.5
```

---

## Live Capture Controls

| Key | Action |
|-----|--------|
| **S** | Save current frame's landmarks (CSV + NumPy) |
| **R** | Toggle session recording — auto-saves when stopped |
| **L** | Toggle landmark index labels (0–20) |
| **C** | Clear the in-memory buffer |
| **Q / ESC** | Quit (unsaved buffer is flushed automatically) |

---

## Output Data Format

Every saved file has columns:

```
timestamp | frame | hand_index | handedness | WRIST_x | WRIST_y | WRIST_z | THUMB_CMC_x | ...
```

- **63 coordinate columns** (21 landmarks × x, y, z)
- Coordinates are in **normalized image space** (0.0 – 1.0) unless you run
  normalization via `landmark_utils.py`.

### Load from Python
```python
import numpy as np, csv

# NumPy
arr = np.load("captured_landmarks/landmarks_20240101_120000_session.npy",
              allow_pickle=True)
features = arr[:, 4:].astype(float)   # shape [N, 63]

# CSV
from landmark_utils import load_csv
meta, features = load_csv("captured_landmarks/landmarks_....csv")
```

---

## Normalize Landmarks

Wrist-origin + scale normalization makes features hand-size invariant:

```bash
python landmark_utils.py --npy captured_landmarks/landmarks_*.npy --normalize --save-normalized
```

This is the recommended pre-processing step before feeding data into a classifier.

---

## Landmark Index Map

```
                 8   12  16  20
                 |   |   |   |
              7  |  11  15  19
              |  |   |   |   |
           6  | 10  14  18
           |  |  |   |   |
     4  3  5  9  13  17
     |  | /
     2  |
     | /
     1
     |
     0  ← WRIST
```

| Index | Name       | Index | Name        |
|-------|------------|-------|-------------|
| 0     | WRIST      | 11    | MIDDLE_PIP  |
| 1     | THUMB_CMC  | 12    | MIDDLE_DIP  |
| 2     | THUMB_MCP  | 13    | MIDDLE_TIP  |
| 3     | THUMB_IP   | 14    | RING_MCP    |
| 4     | THUMB_TIP  | 15    | RING_PIP    |
| 5     | INDEX_MCP  | 16    | RING_DIP    |
| 6     | INDEX_PIP  | 17    | RING_TIP    |
| 7     | INDEX_DIP  | 18    | PINKY_MCP   |
| 8     | INDEX_TIP  | 19    | PINKY_PIP   |
| 9     | MIDDLE_MCP | 20    | PINKY_DIP   |
| 10    | MIDDLE_MCP | 21    | PINKY_TIP   |

---

## Next Steps for Sign Language Recognition

1. **Collect data** — record sessions for each sign/letter using `R` key
2. **Label** — add a label column to the CSV (e.g., letter A–Z)
3. **Normalize** — run `landmark_utils.py --normalize`
4. **Train** — feed the 63-feature vectors into a classifier
   (RandomForest, LSTM for sequences, or a small MLP)
5. **Inference** — integrate your trained model back into `hand_capture.py`
   and display predictions in the HUD

---

## Performance Tips

| Scenario | Recommendation |
|----------|---------------|
| Low FPS  | Set `--complexity 0` (faster, slightly less accurate) |
| High FPS needed | Lower resolution: `--width 640 --height 480` |
| Single hand only | Set `--hands 1` |
| GPU available | MediaPipe auto-detects and uses GPU acceleration |
