# Real-Time ASL-to-Text Recognition System

**A Real-Time Limited-Vocabulary ASL-to-Text Recognition System with Grammar-Assisted Sentence Generation**

A university capstone prototype that uses a standard webcam to recognize a limited
vocabulary of American Sign Language (ASL) signs in real time and convert the recognized
signs into readable English text.

> **Scope note:** This is a limited-vocabulary working prototype for academic demonstration.
> It is **not** an ASL interpreter replacement, does not translate unrestricted ASL, and is
> not production-ready. See [Non-Goals](#non-goals).

---

## Overview

In university settings, communication barriers can occur between hearing individuals and
people who use sign language, and interpreters are not always available for short or
spontaneous interactions. This project explores whether a lightweight software prototype can
recognize a small set of predefined ASL signs in real time and convert them into basic
English text for demonstration and assistive communication purposes.

The main contribution is the **integration** of computer vision, sign classification,
real-time inference, and sentence generation into a single usable prototype.

| | |
|---|---|
| **Primary goal** | Real-time ASL sign recognition from webcam input |
| **Output** | English text sentence generation |
| **Target users** | University students and academic communication contexts |
| **Vocabulary** | ~20-30 predefined academic/classroom signs |
| **Project type** | Working software prototype (one-semester capstone) |

---

## System Architecture

The system has two pipelines: an **offline build/train pipeline** that produces a trained
model and evaluation artifacts, and a **real-time inference pipeline** that runs the trained
model live against a webcam feed.

### Real-Time Inference Pipeline

```
Webcam Input
  -> Frame Capture            (OpenCV)
  -> Hand / Pose Landmarks    (MediaPipe)
  -> Feature Extraction       (normalize x, y, z coordinates)
  -> Sequence Buffer          (fixed-length window)
  -> Sign Classifier          (trained LSTM / GRU / 1D CNN / MLP)
  -> Prediction Smoothing     (confidence + stability)
  -> Token Buffer             (recognized sign tokens)
  -> Grammar Module           (rule-based templates)
  -> English Text Output
  -> Streamlit UI Display     (preview, predicted sign, confidence, sentence)
```

### Offline Build / Train Pipeline

```
Vocabulary & Scope Lock
  -> Record Videos            (5-10 signers, 10-20 samples/sign)
  -> Landmark Extraction      (MediaPipe)
  -> Preprocessing            (normalize, pad/trim, fixed-length sequences)
  -> Train / Val / Test Split (70 / 15 / 15)
  -> Model Training           (LSTM / GRU / 1D CNN / MLP)
  -> Evaluation               (accuracy, F1, confusion matrix, latency)
  -> Export Trained Model     (sign_classifier.pt, label_encoder.pkl)
```

---

## Recommended Vocabulary (MVP, 25 signs)

| Category | Signs |
|---|---|
| Basic communication | `HELLO` `YES` `NO` `PLEASE` `THANK_YOU` `SORRY` `HELP` |
| Clarification | `QUESTION` `REPEAT` `UNDERSTAND` `WHERE` `WHAT` `WHY` |
| Academic context | `TEACHER` `STUDENT` `CLASS` `ASSIGNMENT` `EXAM` `PROJECT` `PRESENTATION` |
| Time / action | `SUBMIT` `TODAY` `TOMORROW` `NEED` `FINISH` |

**Example end-to-end:** recognized tokens `NEED HELP ASSIGNMENT TOMORROW`
-> generated sentence *"I need help with my assignment tomorrow."*

---

## Technology Stack

| Component | Tool |
|---|---|
| Language | Python |
| Computer vision | MediaPipe |
| Video / data processing | OpenCV, NumPy, Pandas |
| ML framework | PyTorch |
| Classical baselines / metrics | scikit-learn |
| Visualization | Matplotlib, Seaborn |
| User interface | Streamlit |
| Grammar layer | Rule-based templates |

The demo uses a **rule-based** grammar module to convert recognized sign tokens into
readable English sentences. An optional free-LLM grammar layer (Gemini API / OpenRouter /
local small model) is noted as **future work**; if added, an LLM would be used **only** for
grammar correction and sentence generation, never for recognizing signs from video.

---

## Planned Repository Structure

```
Sign-language/
  README.md
  requirements.txt
  app.py                      # Streamlit entry point
  data/
    raw_videos/               # recorded sign clips
    landmarks/                # extracted landmark coordinates
    processed/                # normalized, fixed-length sequences
  notebooks/
    exploration.ipynb
    training.ipynb
    evaluation.ipynb
  src/
    camera.py                 # webcam capture / frame loop
    landmarks.py              # MediaPipe landmark extraction
    preprocess.py             # normalization, sequence preparation
    train.py                  # model training
    predict.py               # real-time inference, smoothing
    grammar.py                # rule-based sentence generation
    utils.py
  models/
    sign_classifier.pt
    label_encoder.pkl
  reports/
    figures/
    confusion_matrix.png
    accuracy_report.csv
  docs/
    proposal.tex
    final_report.tex
```

> This structure is the target layout. The repository currently contains the project README
> only; modules and folders are added as each development phase begins.

---

## Development Phases (One Semester)

| Week | Phase | Main output |
|---|---|---|
| 1 | Setup & scope lock | Final vocabulary, repository, architecture diagram, schedule |
| 1-2 | Webcam & landmark prototype | Real-time hand landmark detection |
| 2-4 | Dataset collection | Custom dataset + preprocessing pipeline |
| 4-6 | Baseline classifier | First trained sign-recognition model |
| 6-8 | Real-time prediction | Webcam-to-sign prediction integration |
| 8-9 | Text buffer + rule-based grammar | Recognized tokens + rule-based English sentences |
| 9-10 | Grammar refinement (LLM optional, future) | Improved sentence templates; LLM layer if time permits |
| 10-12 | UI & demo polish | Stable Streamlit demo interface |
| 12-14 | Evaluation | Accuracy, latency, confusion matrix, documentation |
| 15-16 | Final defense prep | Final report, slides, rehearsed demo |

---

## Evaluation Metrics

| Metric | Description | Target |
|---|---|---|
| Classification accuracy | Percentage of signs correctly classified | 80%+ |
| Latency | Sign input to displayed prediction | < 1 sec |
| Per-sign F1 | Precision/recall balance per sign | Reported |
| Confusion matrix | Which signs are confused with each other | Required |
| Demo stability | Runs without crashing during presentation | High |

**Accuracy wording (important):** report results as *"The system achieved X% classification
accuracy on a fixed vocabulary of Y predefined ASL signs under controlled webcam
conditions"* — do not claim general ASL translation accuracy.

---

## Getting Started

> Setup instructions are preliminary and will be finalized once the ML framework is locked
> and `requirements.txt` is committed.

```bash
# 1. Clone
git clone https://github.com/fafams710/Sign-language.git
cd Sign-language

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies (once requirements.txt is available)
pip install -r requirements.txt

# 4. Run the demo (once app.py is available)
streamlit run app.py
```

Planned `requirements.txt`: `opencv-python`, `mediapipe`, `numpy`, `pandas`,
`scikit-learn`, `matplotlib`, `seaborn`, `streamlit`, `torch`.

---

## Team Roles (Five-Person Team)

| Role | Responsibilities |
|---|---|
| Project Lead / Integrator | Schedule, integration, GitHub, final demo coordination |
| Computer Vision Developer | Webcam capture, MediaPipe landmarks, feature extraction |
| ML Developer | Classifier training, accuracy evaluation, model improvement |
| Backend Developer | Rule-based grammar module, text buffer, sentence templates |
| UI / Documentation Lead | Streamlit UI, report, diagrams, presentation materials |

---

## Non-Goals

This project will **not**:

- Replace professional ASL interpreters.
- Translate full, unrestricted ASL conversations.
- Support all ASL vocabulary.
- Build or train a custom large language model from scratch.
- Guarantee production-level accessibility performance.
- Perform legal, medical, or other high-risk interpretation.

---

## Data, Consent, and Ethics

If human participants are recorded for the custom dataset, the team documents: participant
consent, intended academic use, storage/sharing/deletion policy, whether faces are visible,
and the right to withdraw data. Recommended dataset minimum: 20-30 signs, 5-10 signers,
10-20 samples per sign per signer, 2-5 second clips under controlled lighting.

---

## License

To be determined.
