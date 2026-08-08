# Phase 7 Specification: Streamlit Demo Interface

This document is the requirements specification for the Streamlit demo
interface of the Real-Time Limited-Vocabulary ASL-to-Text Recognition System,
a university capstone prototype. It states WHAT the interface implemented in
`app.py` must do. It is addressed to the implementing engineer and is written
to be verified by operating the built application. The companion technical
design document, `Phase7ARCH.md`, states HOW the interface is built; this
document cites that document by file name and section number wherever a
mechanism is defined there rather than here.

## Table of Contents

1. Purpose, Deliverable and Out of Scope
2. Pipeline Overview
3. Current State of the System
4. Functional Requirements (FR-1..FR-n)
5. Control Semantics
6. Recognition Tunables and Defaults
7. Performance and Measurement Requirements
8. Degraded and Error Conditions
9. Constraints, Conventions and Sources of Truth
10. Manual Verification Checklist

## 1. Purpose, Deliverable and Out of Scope

The deliverable of this phase is the Streamlit demo interface implemented in
`app.py`: a browser-based application that drives the existing recognition
pipeline (webcam capture, hand-landmark extraction, sequence classification,
prediction smoothing, token buffering and rule-based sentence generation)
live against a standard webcam and displays the result as readable English
text. The system recognizes a small, fixed vocabulary of predefined
academic/classroom signs; it is a limited-vocabulary demonstration prototype,
not a general ASL interpreter, per the repository's locked project scope.

Out of scope for this phase:

- No new third-party dependency of any kind. `requirements.txt` is unchanged.
- `streamlit-webrtc` or any browser-side (client-captured) video pipeline.
  Capture is server-side only, as fixed by Phase7ARCH.md section 2.
- Any change to `src/camera.py`, `src/landmarks.py`, `src/predict.py`,
  `src/buffer.py`, `src/grammar.py`, `src/utils.py`, or any other existing
  module. The UI calls these modules; it does not modify them.
- Dataset collection, model training, or producing a trained checkpoint.
- General or unrestricted ASL translation of any kind.
- Production deployment, authentication, multi-user isolation, or
  persistence of recognized sentences beyond the running session.

## 2. Pipeline Overview

The interface drives the following pipeline, in order. Each stage names the
module that owns it; the interface calls these modules and performs no
recognition logic of its own (see section 9 and Phase7ARCH.md section 4.8).

1. Webcam frame - captured by `src/camera.py` (`Camera.read()`).
2. Hand landmarks - extracted by `src/landmarks.py` (`extract()`).
3. Per-frame feature vector - assembled by `src/predict.py`, internal to
   `FrameBuffer.add_frame()`.
4. Rolling fixed-length window - held by `src/predict.py`
   (`FrameBuffer`).
5. Classifier - run by `src/predict.py` (`MockPredictor` or
   `RealPredictor`, via `SignRecognizer.update()`).
6. Confidence/stability smoothing - performed by `src/predict.py`
   (`PredictionSmoother`, internal to `SignRecognizer.update()`).
7. Accepted gloss token - the `emitted` field of the `RecognitionResult`
   returned by `src/predict.py` (`SignRecognizer.update()`).
8. Token buffer - accumulated by `src/buffer.py` (`TokenBuffer`).
9. Rule-based sentence generation - performed by `src/grammar.py`
   (`glosses_to_sentence()`).
10. Displayed English text - rendered by the interface itself (`app.py`).

## 3. Current State of the System

No trained model file exists yet: dataset collection is still in progress,
per the repository's roadmap, and `models/sign_classifier.pt` does not
exist in this repository. The mock predictor (`src/predict.py`
`MockPredictor`, reached via `create_recognizer(use_mock=True)`) is
therefore the only working end-to-end recognition path available today. This
is the authoritative statement of the system's current mock-only status,
cited from Phase7ARCH.md sections 2 and 8.

The interface must be fully buildable and demonstrable against the mock
path: every functional requirement in section 4 must be exercisable with the
mock predictor selected. The real-predictor path
(`create_recognizer(use_mock=False)`) requires a trained checkpoint that
does not yet exist and, in this repository, cannot be exercised. The
required behavior when the real predictor is selected and the checkpoint is
absent, invalid, or otherwise fails to load is defined in section 8: the
failure is surfaced to the user as one clear message, and the application
never silently falls back to the mock predictor.

## 4. Functional Requirements

Each requirement below is phrased so that a reviewer can mark it pass or
fail by operating the built UI. Control semantics referenced here are
defined precisely in section 5.

- **FR-1 Live webcam preview.** While the capture loop is running, the
  interface displays a continuously updating preview of the webcam feed with
  correct colors (not color-inverted).
- **FR-2 Start control.** A start control begins the capture loop per the
  start semantics in section 5. Before the loop starts, the interface
  acquires the camera and the recognizer; if acquisition fails, the interface
  reports the failure per section 8 and does not start the loop.
- **FR-3 Stop control.** A stop control ends the capture loop per the stop
  semantics in section 5, leaving all recognition and token state intact.
- **FR-4 Reset control.** A reset control clears the recognizer's in-flight
  state and the token buffer per the reset semantics in section 5, is safe
  to press whether or not the loop is running, is safe to press twice in a
  row, and never raises an error visible to the user.
- **FR-5 Current recognized sign.** While the loop is running, the interface
  displays the current raw top-1 recognized gloss (`RecognitionResult.gloss`)
  for the most recent completed inference, and a defined non-error state
  (not a blank or an error) while warming up or while no hands are in view.
- **FR-6 Confidence for the current sign.** Alongside the current recognized
  sign, the interface displays its confidence score
  (`RecognitionResult.confidence`), and a defined non-error state when no
  inference has yet run.
- **FR-7 Running recognized-token list.** The interface displays the
  in-progress sequence of accepted gloss tokens
  (`TokenBuffer.tokens`), updated live as tokens are accepted, without
  destroying the buffer.
- **FR-8 Generated English sentence (live preview).** The interface displays
  a sentence generated from the current token sequence
  (`glosses_to_sentence(token_buffer.tokens)`), refreshed as tokens are
  accepted, without clearing the token buffer.
- **FR-9 Finish-sentence control.** A finish control finalizes the current
  utterance: it calls `TokenBuffer.emit()`, converts the returned list with
  `glosses_to_sentence()`, and displays the result per the finish semantics
  in section 5, including the behavior on a second consecutive press and on
  an empty buffer (section 8).
- **FR-10 Latency readout.** While the loop is running, the interface
  displays the most recent inference-only latency
  (`RecognitionResult.latency_ms`), labelled as inference-only, and a
  defined placeholder (not a blank or an error) when no inference has yet
  run, including immediately after stop is pressed during warm-up.
- **FR-11 Achieved sample-rate readout.** While the loop is running, the
  interface displays a measured, achieved sample rate: the number of frames
  actually pushed into the recognizer per second, labelled as measured. No
  nominal or assumed frame rate is ever displayed as if it were measured
  (see section 7).
- **FR-12 Confidence-threshold slider.** A slider lets the operator set
  `min_confidence` within its documented range and default (section 6);
  changing it takes effect per the reconstruction mechanism defined in
  Phase7ARCH.md section 6.
- **FR-13 Stability-window slider.** A slider lets the operator set
  `stability_frames` within its documented range and default (section 6);
  changing it takes effect per the reconstruction mechanism defined in
  Phase7ARCH.md section 6.
- **FR-14 Cooldown slider.** A slider lets the operator set
  `cooldown_frames` within its documented range and default (section 6);
  changing it takes effect per the reconstruction mechanism defined in
  Phase7ARCH.md section 6.
- **FR-15 Mock-predictor toggle.** A toggle lets the operator choose between
  the mock predictor and the real predictor. Selecting the real predictor
  without a usable checkpoint fails per section 8 (ERR-3/ERR-4 in
  Phase7ARCH.md section 8) rather than silently using the mock predictor.

## 5. Control Semantics

For each control, the following states precisely which pieces of state are
affected and which are preserved: the capture loop, the recognizer's frame
window, the smoother state, the token buffer, the last generated sentence,
and the camera handle. This is the authoritative statement of control
semantics; Phase7ARCH.md section 7 cites it rather than restating it.

**Start.** Sets the running flag to true, causing the capture loop to begin
on the next script pass. Preserves the recognizer's frame window, the
smoother state, the token buffer, and the last generated sentence exactly as
they were. Acquires the camera and the detector, opening them only if they
are not already open (cached); does not release or reopen an
already-acquired camera handle.

**Stop.** Sets the running flag to false, which ends the capture loop.
Preserves the recognizer's frame window, the smoother state, the token
buffer, and the last generated sentence exactly as they were. Does not
release the camera handle, does not clear any cached resource, and does not
rebuild the recognizer.

**Reset.** Clears the recognizer's frame window and all smoother state
(equivalent to `SignRecognizer.reset()`), clears the token buffer
(equivalent to `TokenBuffer.reset()`), and clears the displayed current
sign, confidence, and latency readouts to their defined empty states. Does
not change the running flag (the loop keeps running if it was running), does
not release or reopen the camera handle, and does not rebuild or reload the
recognizer or predictor. Reset is idempotent: pressing it while already in
the reset state, or pressing it twice in a row, produces the same observable
result and never raises an error visible to the user.

**Finish sentence.** Calls `TokenBuffer.emit()`, which returns the buffered
token list and clears the token buffer, and converts the returned list to a
sentence with `glosses_to_sentence()`. When the returned list is non-empty,
the newly generated sentence is displayed, replacing the previous one. When
the returned list is empty - because the buffer was already empty, or
because this is a second consecutive finish press - `glosses_to_sentence()`
returns an empty string and the previously displayed sentence remains
displayed unchanged; it is not overwritten with an empty result. Does not
change the running flag, does not clear the recognizer's frame window, and
does not clear the smoother state. The live sentence preview (FR-8) is
unaffected by finish because it is regenerated from the non-destructive
`tokens` property, not from `emit()`.

## 6. Recognition Tunables and Defaults

The confidence-threshold, stability-window and cooldown sliders (FR-12,
FR-13, FR-14) map onto the recognizer's constructor parameters as follows.
The constants and their defaults are transcribed authoritatively in
Phase7ARCH.md section 4.4; this section states only their UI-facing ranges,
counting semantics and provisional status.

| Slider                | Constructor parameter | Default | Valid range                  |
|------------------------|-----------------------|---------|-------------------------------|
| Confidence threshold  | `min_confidence`      | 0.60    | `[0.0, 1.0]` inclusive        |
| Stability window      | `stability_frames`    | 5       | integer, `>= 1`               |
| Cooldown              | `cooldown_frames`     | 15      | integer, `>= 0`               |

`stability_frames` and `cooldown_frames` are counted in recognizer UPDATES -
one unit per call to `SignRecognizer.update()` - not in wall-clock seconds.
The number of updates per second of wall-clock time depends on the achieved
capture rate, which is why section 7 requires the achieved sample rate to be
measured and displayed rather than assumed.

These defaults are provisional: they were chosen before any dataset or
trained model existed, and are expected to be retuned once real recordings
and a trained classifier are available. Nothing in this document or in the
interface should present them as final.

## 7. Performance and Measurement Requirements

This section is the authoritative home for the measured landmark-stage
benchmark figures. They were obtained from a headless benchmark run of 200
synthetic 640x480 frames against the landmark-extraction stage
(`src/landmarks.py`):

- Mean frame rate: 34.1 FPS.
- Mean per-frame time: 29.31 ms.
- p95 per-frame time: 40.54 ms.
- p99 per-frame time: 52.74 ms.
- Jitter, coefficient of variation: 0.32 (high).
- Hand detections: zero (the synthetic frames contain no hands).

Caveat, stated explicitly: because no hands were present in any of the 200
frames, the hand-landmark regression inside MediaPipe short-circuited before
running its full per-hand computation. These figures are therefore an
OPTIMISTIC CEILING on landmark-stage throughput under real conditions, not a
working figure for a live session with hands in view, and they exclude
classifier inference cost entirely. Phase7ARCH.md section 9 restates these
figures with a back-pointer to this section and derives a per-frame time
budget from them; it does not re-derive or contradict them.

The interface must display, while the loop is running:

- A measured, achieved sample rate: the number of frames actually pushed
  into the recognizer per second, computed from the running application
  itself and labelled as measured (FR-11).
- An inference latency readout taken from `RecognitionResult.latency_ms`,
  labelled as inference-only (FR-10).

No nominal or assumed frame rate (for example, the webcam's configured `FPS`
constant in `src/camera.py`, or the benchmark's mean FPS above) may ever be
displayed as if it were a measured, live figure. Any nominal figure shown
for reference must be labelled as nominal, not as measured.

## 8. Degraded and Error Conditions

The interface must handle the following conditions as described. The
detection point, exception type and matrix identifiers for each are defined
authoritatively in Phase7ARCH.md section 8; this section states only the
user-visible, functional requirement.

- **No webcam available.** The interface refuses to start the loop and
  shows one clear error message; the running flag is not set to true.
- **Missing MediaPipe `.task` asset** (`models/hand_landmarker.task`). The
  interface refuses to start the loop and shows one clear error message
  naming the missing, user-provided, gitignored asset.
- **Missing or invalid checkpoint on the real-predictor path.** The
  interface refuses to start the loop and shows one clear error message. The
  interface never silently falls back to the mock predictor when the real
  predictor was explicitly selected and fails to load.
- **Window still warming up.** The first `SEQUENCE_LENGTH` frames after a
  start or reset produce no inference. The interface renders a defined
  warm-up state (using the frames-buffered count), not a blank field and not
  an error.
- **No hands in view.** When the window is full but contains no hand data,
  no inference runs and nothing is emitted. The interface renders a defined
  "no hands" state, not a stall and not an error.
- **Malformed frame data.** If the recognizer rejects a frame as malformed,
  the interface surfaces the failure as one clear message and stops the
  loop rather than continuing to feed data the pipeline cannot use.
- **Empty token buffer at sentence-generation time.** Finishing a sentence
  with no accepted tokens (or a second consecutive finish press) produces an
  empty result; the interface renders a placeholder for the sentence rather
  than an error, per the finish semantics in section 5.

## 9. Constraints, Conventions and Sources of Truth

`src/utils.py` is the single source of truth for the 25-sign vocabulary
(`VOCABULARY`) and for the fixed sequence length (`SEQUENCE_LENGTH`). The
interface imports both rather than restating them, and this document does
not enumerate the vocabulary list.

Accuracy-wording rule: any results-related text, in this document or
rendered by the interface, is phrased as classification accuracy on a fixed
vocabulary of predefined ASL signs under controlled webcam conditions. No
text produced by or about this interface claims, or should be read as
claiming, general or unrestricted ASL translation.

Non-functional constraints binding on the implementation:

- Plain Streamlit only; no client-side capture library of any kind.
- `streamlit-webrtc` is explicitly out of scope.
- `requirements.txt` is unchanged; no new dependency is introduced.
- The target environment is Streamlit 1.58 (authoritative statement:
  Phase7ARCH.md section 1).
- Python code follows PEP 8.
- `app.py` carries a module-level docstring.
- Heavy imports (Streamlit, OpenCV, the inference pipeline) are deferred
  inside `main()`, as the existing `app.py` stub already establishes.

## 10. Manual Verification Checklist

This checklist is executed by a reviewer against the mock-predictor path,
since no trained checkpoint exists in this repository (section 3). Each
step corresponds to one or more functional requirements in section 4.

1. Launch the application. Confirm it loads with no error and the mock
   toggle is available (FR-15).
2. With the mock predictor selected, press start. Confirm the webcam
   preview appears and updates continuously with correct (non-inverted)
   colors (FR-1, FR-2).
3. Confirm the current recognized sign and its confidence display a defined
   warm-up state for the first `SEQUENCE_LENGTH` frames, then begin showing
   values (FR-5, FR-6; section 8).
4. Hold a hand in view until a sign is accepted. Confirm it appears in the
   running token list and the live sentence preview updates (FR-7, FR-8).
5. Press the finish-sentence control. Confirm a non-empty sentence is
   displayed, and confirm the token list used for the next utterance starts
   empty (FR-9).
6. Press the finish-sentence control a second time with no new tokens
   accepted. Confirm the previously displayed sentence remains unchanged
   rather than being blanked (FR-9; section 5).
7. Press stop. Confirm the preview stops updating and confirm the token
   list and last sentence remain exactly as they were (FR-3; section 5).
8. Press start again. Confirm capture resumes and the previously
   accumulated token list and sentence are still displayed (FR-2, FR-3).
9. Press reset. Confirm the token list, current sign, confidence and
   latency readouts return to their defined empty states, and confirm
   pressing reset a second time immediately afterward produces no error and
   no change (FR-4; section 5).
10. While the loop is running, confirm the latency readout shows a numeric,
    inference-only value, and confirm the achieved sample-rate readout shows
    a measured numeric value that is materially higher than the visible UI
    refresh rate of the preview image (FR-10, FR-11; section 7).
11. Move the confidence-threshold, stability-window and cooldown sliders one
    at a time while the loop is running. Confirm the loop stops and then
    resumes automatically, and confirm the token list and last sentence are
    preserved across each change (FR-12, FR-13, FR-14).
12. Flip the mock-predictor toggle while the loop is running. Confirm the
    loop stops and resumes, and confirm the token list and last sentence are
    preserved (FR-15).
13. Stop the application before any hand has ever been shown, so the loop
    ends during warm-up. Confirm the latency readout shows its defined
    placeholder rather than a blank or an error (FR-10; section 8).
