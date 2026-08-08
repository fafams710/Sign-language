# Project Roadmap

Real-Time Limited-Vocabulary ASL-to-Text Recognition System with grammar-assisted English sentence generation, delivered as a one-semester university capstone prototype.

## Locked Decisions

- **ML framework:** PyTorch.
- **Grammar layer:** rule-based templates for the demo (LLM-based grammar is future/optional, never used for sign recognition).
- **User interface:** Streamlit.
- **Recognition approach:** MediaPipe landmarks feeding a sequence classifier.
- **Vocabulary:** ~25 predefined academic/classroom signs.

## Phases (One Semester)

| Phase | Weeks | Main Output |
|---|---|---|
| Phase 0 - Setup & Scope Lock (completed) | 1 | Final vocabulary, repository scaffold, architecture, schedule |
| Phase 1 - Webcam & Landmark Prototype (completed) | 1-2 | Real-time hand landmark detection |
| Phase 2 - Dataset Collection (preprocessing pipeline complete; dataset collection in progress) | 2-4 | Custom dataset + preprocessing pipeline |
| Phase 3 - Baseline Classifier (training code complete; awaiting full dataset) | 4-6 | First trained sign-recognition model |
| Phase 4 - Real-Time Prediction (current) | 6-8 | Webcam-to-sign prediction integration |
| Phase 5 - Text Buffer + Rule-Based Grammar | 8-9 | Recognized tokens + rule-based English sentences |
| Phase 6 - Grammar Refinement (LLM optional, future) | 9-10 | Improved templates; optional LLM layer if time permits |
| Phase 7 - UI & Demo Polish | 10-12 | Stable Streamlit demo interface |
| Phase 8 - Evaluation | 12-14 | Accuracy, latency, confusion matrix, documentation |
| Final Defense Prep | 15-16 | Final report, slides, rehearsed demo |

## Folder Map

| Area | Location |
|---|---|
| Dataset | `data/` |
| Models | `models/` |
| UI | `app.py` + `src/` |
| Documentation | `docs/` |
| Evaluation | `reports/` |
