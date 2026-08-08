# Phase 7 Design: Streamlit Demo Interface

This document is the technical design for the Streamlit demo interface of
the Real-Time Limited-Vocabulary ASL-to-Text Recognition System. It states
HOW the interface described in `Phase7SPEC.md` is built: the locked
decisions, the public API of every module the interface calls, the state
model, the resource lifecycle, the control flow, the error-handling matrix,
the performance budget, and the invariants that keep the recognizer's
accuracy characteristics intact. It is addressed to the implementing
engineer and is written to be usable with no prior knowledge of this
repository beyond the six source modules it transcribes.

Every mandatory statement in this document ("must", "never", "always")
resolves to a numbered invariant (`INV-n`, section 10) or a labelled locked
decision (`LD-n`, section 2); prose elsewhere cites the identifier that
carries the reason rather than asserting new mandatory language on its own.

## Table of Contents

1. Purpose, Relationship to the Specification, Targeted Environment
2. Locked Decisions (LD-1..LD-n)
3. Streamlit Execution Model Primer
4. Dependency Interface Reference
   - 4.1 src/utils.py
   - 4.2 src/camera.py
   - 4.3 src/landmarks.py
   - 4.4 src/predict.py
   - 4.5 src/buffer.py
   - 4.6 src/grammar.py
   - 4.7 Exception Surface Summary
   - 4.8 Data Handoff Rules
5. State Model
6. Resource Lifecycle and Caching
7. Control Flow
8. Error Handling Matrix
9. Performance Budget and Measurement
10. Invariants (INV-1..INV-n)
11. Requirement Traceability
12. Notes on Warranted Code Changes

## 1. Purpose, Relationship to the Specification, Targeted Environment

This document designs the implementation of the requirements fixed in
`Phase7SPEC.md`. Where the specification states WHAT the interface must do
and how a reviewer verifies it, this document states HOW: the exact
Streamlit constructs, the exact module calls, the exact state keys, and the
reasoning behind every rule an implementing engineer might otherwise be
tempted to simplify away.

This document targets **Streamlit 1.58** as the environment the interface
is built and reviewed against. This is the single authoritative statement
of the targeted Streamlit version in either document; `Phase7SPEC.md`
section 9 cites this section rather than restating the version number. The
Streamlit API surface referenced throughout this document -
`st.session_state`, `st.cache_resource`, `st.empty`, `st.image`,
`st.sidebar`, `st.rerun`, and standard input widgets - is assumed available
and stable at that version. Recording the targeted version here lets a
future reader who observes different behavior diagnose version drift rather
than assume the reasoning in this document is wrong.

## 2. Locked Decisions

The following are decided, not open questions the implementer may revisit:

- **LD-1.** Plain Streamlit only. No client-side or browser-captured video
  library is used anywhere in the interface.
- **LD-2.** `streamlit-webrtc` is explicitly out of scope for this phase.
- **LD-3.** `requirements.txt` is unchanged by this phase; no new dependency
  is introduced.
- **LD-4.** The targeted environment is Streamlit 1.58 (section 1).
- **LD-5.** Capture is performed by a server-side `cv2.VideoCapture`-backed
  loop (via `src/camera.py`), driving a single `st.empty()` placeholder that
  the loop repeatedly overwrites with the latest preview frame.
- **LD-6.** The system's current status is mock-only: no trained checkpoint
  exists in this repository, so `create_recognizer(use_mock=True)` is the
  only recognition path that can be exercised end to end today
  (authoritative statement: `Phase7SPEC.md` section 3).

## 3. Streamlit Execution Model Primer

A Streamlit script is not a long-lived, event-driven program in the way a
typical desktop GUI is. The entire script re-executes top to bottom every
time the browser reports a new interaction (a button press, a slider drag
release, a toggle flip). Each such re-execution is called a "rerun."

Two consequences follow directly, and both are load-bearing for this
design:

- A widget interaction that occurs while a loop inside the currently
  running script is executing does not modify variables inside that running
  script. Instead, it schedules a rerun; Streamlit terminates the currently
  running script at its next opportunity (its next call into a Streamlit
  API) and starts a fresh top-to-bottom execution. There is no mechanism by
  which a widget "reaches into" a loop that is already running (see INV-7,
  INV-8).
- Because the running script is torn down and replaced rather than paused
  and resumed, any state that must survive a rerun cannot live in an
  ordinary local variable. It must live in `st.session_state` (survives
  reruns within one browser session) or `st.cache_resource` (survives
  reruns and is shared process-wide; section 6).

Consequently, start, stop and reset are not implemented as anything that
tries to keep one Python loop alive across interactions. They are
implemented as `st.session_state` boolean/state flags that are read at the
top of the script on every rerun; the top-of-script flow (section 7) decides
whether to enter the capture loop based on the current value of those flags,
not based on any control flow left over from a previous run.

## 4. Dependency Interface Reference

Every subsection below names its repository-relative source path first,
then lists constants, then signatures, then behavioral notes. Every
signature, constant and default is transcribed from the current source with
parameter names, order, defaults and return types unchanged; this is
verified by opening the named file, not recalled from memory.

### 4.1 src/utils.py

```
SEQUENCE_LENGTH = 30
```

The fixed per-sample frame count for the sequence classifier: roughly
1.0-1.2 seconds of motion at the 25-30 fps a standard webcam delivers. This
is the single source of truth for the fixed-length convention; the
interface imports it rather than restating it (INV-3).

```
VOCABULARY = [ ... ]  # a 25-entry list of gloss strings
```

`VOCABULARY` is the 25-sign MVP vocabulary for the academic/classroom
prototype. It is imported by the interface, never enumerated here or
anywhere in either document (see section 9's source-of-truth statement, and
`Phase7SPEC.md` section 9).

### 4.2 src/camera.py

```
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 60
BUFFER_SIZE = 1

class CameraError(RuntimeError):
    ...

class Camera:
    def __init__(self, camera_index=0, width=FRAME_WIDTH, height=FRAME_HEIGHT,
                 fps=FPS, buffer_size=BUFFER_SIZE):
        ...

    def open(self) -> int:
        ...

    def read(self):
        ...  # -> (ret, frame)

    def release(self) -> None:
        ...

    @property
    def is_open(self) -> bool:
        ...

    def __enter__(self):
        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...
```

Behavioral notes: `open()` returns the index that was actually opened. When
`camera_index == 0`, indices `0, 1, 2` are tried in order and the first that
reports `isOpened()` is kept; when `camera_index != 0`, only the requested
index is tried. `open()` raises `CameraError` if no candidate index can be
opened. `read()` returns `(ret, frame)`: `ret` is a bool and `frame` is a
BGR image (or `None` on failure); `read()` raises `CameraError` if called
before `open()`. `release()` releases the underlying device if open, and is
a no-op otherwise. `is_open` reports whether the capture device is
currently open. `Camera` supports the context-manager protocol
(`__enter__` calls `open()`, `__exit__` calls `release()`); the interface
does not use this protocol directly because it must not release the cached
camera on every script exit (section 6).

### 4.3 src/landmarks.py

```
N_LANDMARKS = 21
COORDS_PER_LM = 3
N_FEATURES = N_LANDMARKS * COORDS_PER_LM   # 63

MODEL_PATH = REPO_ROOT / "models" / "hand_landmarker.task"

def normalize_landmarks(features: np.ndarray) -> np.ndarray:
    ...

def create_detector(model_path=MODEL_PATH, num_hands=2,
                    det_conf=0.5, track_conf=0.5):
    ...  # -> HandLandmarker, running_mode=RunningMode.VIDEO

def extract(detector, frame_bgr, ts_ms):
    ...  # -> list of (handedness, landmarks)
```

`MODEL_PATH` (`models/hand_landmarker.task`, relative to the repository
root) is user-provided and gitignored; it is never committed. `create_detector`
raises `FileNotFoundError` naming the path when the asset is missing, and
otherwise returns a `HandLandmarker` running in MediaPipe's `VIDEO` running
mode. `extract(detector, frame_bgr, ts_ms)` converts the frame BGR to RGB
internally, calls `detect_for_video` with the given timestamp, and returns a
list of `(handedness, landmarks)` tuples: `handedness` is the MediaPipe
display-name string (case-sensitive; the exact set the interface must
tolerate downstream is `{"Right", "Left"}`, enforced inside `src/predict.py`,
not by the interface - section 4.8), and `landmarks` is the list of 21
landmark objects exposing `.x`, `.y`, `.z`. `normalize_landmarks(features)`
translates each hand's points to a wrist-relative origin, then scales by the
maximum point distance, skipping the scale when the point spread is not
greater than `1e-6`.

### 4.4 src/predict.py

```
N_FEATURES_PER_FRAME = 2 * src.landmarks.N_FEATURES   # 126

DEFAULT_CHECKPOINT_PATH = <repo root> / "models" / "sign_classifier.pt"
DEFAULT_MIN_CONFIDENCE = 0.60
DEFAULT_STABILITY_FRAMES = 5
DEFAULT_COOLDOWN_FRAMES = 15

def create_recognizer(
    use_mock: bool = False,
    checkpoint_path=DEFAULT_CHECKPOINT_PATH,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    stability_frames: int = DEFAULT_STABILITY_FRAMES,
    cooldown_frames: int = DEFAULT_COOLDOWN_FRAMES,
    device: str = "cpu",
    model_builder=None,
) -> SignRecognizer:
    ...
```

`N_FEATURES_PER_FRAME` (126) is laid out as two fixed 63-wide slots,
`[Right | Left]`: features `0..62` hold the Right hand, features `63..125`
hold the Left hand. A hand absent from a frame contributes an all-zero
63-slot; normalization (`src.landmarks.normalize_landmarks`) is always
applied to a present hand's raw features before the zero-fill of an absent
hand, never after. This layout is internal to `src/predict.py`; the
interface never constructs it directly (section 4.8).

`DEFAULT_MIN_CONFIDENCE`, `DEFAULT_STABILITY_FRAMES` and
`DEFAULT_COOLDOWN_FRAMES` are this document's single authoritative home for
these three constants and their defaults (0.60 / 5 / 15); `Phase7SPEC.md`
section 6 cites this subsection and owns only the UI-facing ranges,
update-counted semantics and provisional status. They are PROVISIONAL:
chosen with no dataset or trained model in hand, and expected to be
retuned.

```
class SignRecognizer:
    def __init__(
        self,
        predictor,
        length: int = SEQUENCE_LENGTH,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        stability_frames: int = DEFAULT_STABILITY_FRAMES,
        cooldown_frames: int = DEFAULT_COOLDOWN_FRAMES,
    ):
        ...

    @property
    def last_latency_ms(self) -> Optional[float]:
        ...

    def update(self, hands) -> RecognitionResult:
        ...

    def reset(self) -> None:
        ...
```

`update(hands)` appends one frame (`hands` is the unmodified return value of
`src.landmarks.extract`, section 4.8) and runs at most one inference. It
returns a `RecognitionResult`. `reset()` clears the internal frame window
and all smoother state; it does not reload or replace the predictor.
`last_latency_ms` reports the milliseconds of the most recent
`predictor.predict` call only (excludes capture, landmark extraction and
display); it is `None` before the first inference and after `reset()` until
the next inference runs.

```
@dataclass(frozen=True)
class RecognitionResult:
    gloss: Optional[str]
    confidence: Optional[float]
    emitted: Optional[str]
    latency_ms: Optional[float]
    frames_buffered: int
    window_ready: bool
    status: str
```

Field meanings: `gloss` is the raw top-1 predicted gloss for the most
recent completed window (before smoothing), or `None` if no inference ran
this call. `confidence` is the raw top-1 confidence in `[0.0, 1.0]` for
`gloss`, or `None` if no inference ran this call. `emitted` is the gloss
accepted by the internal `PredictionSmoother` this call, or `None` if
nothing was accepted this call; this is the ONLY field that may be passed to
`TokenBuffer.add` (INV-6). `latency_ms` is the milliseconds of the single
`predictor.predict` call made this update, or `None` if no inference ran.
`frames_buffered` is the number of frames currently held in the recognizer's
window (`0 <= frames_buffered <= length`). `window_ready` is whether the
window was full (and therefore eligible for inference) this call. `status`
is one of the closed set `{"warming_up", "no_hands", "predicted"}`:
`"warming_up"` means the window is not yet full and no inference ran (all of
`gloss`, `confidence`, `latency_ms`, `emitted` are `None`); `"no_hands"`
means the window is full but entirely zero, so no inference ran and nothing
was emitted, though the internal smoother still received one non-passing
update; `"predicted"` means exactly one inference ran this call. An
implementer switches on `status` exhaustively.

```
class MockPredictor:
    def __init__(self, labels: Optional[Sequence[str]] = None):
        ...

    @property
    def labels(self) -> list:
        ...

    @property
    def last_latency_ms(self) -> Optional[float]:
        ...

    def predict(self, window):
        ...  # -> (str, float)
```

`MockPredictor` is a permanent, deterministic component - not scaffolding
to be deleted once a real model exists - and requires no file and no
`torch` import. Its `predict(window)` returns a gloss drawn from `labels`
(default `src.utils.VOCABULARY`) and a confidence that always lands in
`[0.60, 0.99]`, so an end-to-end mock run can actually emit a token.

```
class RealPredictor:
    def __init__(
        self,
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
        device: str = "cpu",
        model_builder=None,
    ):
        ...

    @property
    def labels(self) -> list:
        ...

    @property
    def last_latency_ms(self) -> Optional[float]:
        ...

    def predict(self, window):
        ...  # -> (str, float)
```

`RealPredictor` loads eagerly in `__init__` (existence check, deserialize,
validate, rebuild the architecture, load weights, `eval()`), so a failed
load surfaces once, at construction. `torch` is imported lazily, only
inside `__init__`. `labels`, `last_latency_ms` and `predict(window)` have
the same shape and meaning as `MockPredictor`'s.

```
class FrameBuffer:
    def __init__(self, length: int = SEQUENCE_LENGTH):
        ...

    @property
    def length(self) -> int:
        ...

    def add_frame(self, hands) -> None:
        ...

    def __len__(self) -> int:
        ...

    @property
    def is_ready(self) -> bool:
        ...

    def window(self):
        ...  # -> numpy.ndarray or None

    def reset(self) -> None:
        ...
```

`FrameBuffer` is internal to `SignRecognizer`; the interface never
constructs one directly. It is documented here because its shape explains
`RecognitionResult.frames_buffered` and `window_ready`.

```
class PredictionSmoother:
    def __init__(
        self,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        stability_frames: int = DEFAULT_STABILITY_FRAMES,
        cooldown_frames: int = DEFAULT_COOLDOWN_FRAMES,
    ):
        ...

    def update(self, gloss: Optional[str], confidence: Optional[float]) -> Optional[str]:
        ...

    def reset(self) -> None:
        ...
```

`PredictionSmoother` is also internal to `SignRecognizer` (its instance is
a private attribute; INV-12 forbids reaching into it). It is documented
here because its four-step evaluation order explains why reconstruction,
not attribute mutation, is the sanctioned way to change a tunable (section
6). Evaluation order for `update(gloss, confidence)`, exact and in order:

1. If a cooldown is active (remaining updates > 0): decrement it, clear the
   candidate gloss and the agreement run, and return `None` regardless of
   the arguments.
2. Otherwise, if the window is non-passing - `gloss` is `None` or blank, or
   `confidence` is `None`/`NaN`/not a real number, or `confidence` is below
   `min_confidence` (the comparison is inclusive `>=`, so a confidence
   exactly equal to `min_confidence` passes) - clear the candidate and the
   run, and return `None`.
3. Otherwise, if `gloss` equals the current candidate, increment the
   agreement run; a different gloss restarts the run at 1 for the new
   gloss.
4. If the run has now reached `stability_frames`, arm the cooldown for
   `cooldown_frames` updates, clear the candidate and the run, and return
   the accepted gloss. Otherwise return `None`.

The smoother deliberately re-emits a held sign after each cooldown expires;
it does not itself deduplicate repeats, because `TokenBuffer.add` already
collapses a run of identical consecutive emissions into one token
downstream (section 4.5). Duplicating that deduplication here would be
redundant and is not done.

Checkpoint contract: `RealPredictor` expects `models/sign_classifier.pt` to
deserialize to a mapping with exactly six required keys - `state_dict`,
`architecture`, `hyperparams`, `input_shape`, `num_classes`, `vocabulary` -
each of the documented expected type. The checkpoint's `vocabulary` ordering
is authoritative for `index -> gloss` and becomes the predictor's `labels`;
if it differs from `src.utils.VOCABULARY` while remaining internally
self-consistent, a warning is emitted rather than an error (ERR-7, section
8). The interface must never take label ordering from a local copy of
`VOCABULARY` when a real predictor is in use; it reads `labels` from the
predictor.

### 4.5 src/buffer.py

```
class TokenBuffer:
    def __init__(self):
        ...

    def add(self, gloss) -> None:
        ...

    @property
    def tokens(self) -> list:
        ...

    def emit(self) -> list:
        ...

    def reset(self) -> None:
        ...
```

`add(gloss)` silently ignores `None`, non-string, and blank/whitespace-only
input without raising. A gloss identical to the immediately preceding one is
treated as part of the same held sign and does not add a new token
(consecutive-run collapse); the same gloss seen again later, after an
intervening different gloss, is recorded as a new token. `tokens` returns a
copy of the current buffered sequence without mutating or clearing the
buffer. `emit()` returns the buffered token list and clears the buffer;
it returns `[]` when the buffer is empty, including on a second consecutive
call. `reset()` clears the buffer without returning its contents.

### 4.6 src/grammar.py

```
def glosses_to_sentence(tokens: list) -> str:
    ...
```

Returns `""` for empty or entirely blank input. Capitalizes the first
alphabetic character of the resulting text. Terminates the sentence with
`"?"` when a question sign (one of the sign set that triggers question
intent) is present anywhere in the input, and with `"."` otherwise.
Underscored gloss tokens are humanized into lowercase, space-separated
words. Out-of-vocabulary tokens are tolerated and rendered with the same
humanization fallback. `glosses_to_sentence` never raises.

### 4.7 Exception Surface Summary

| Entry point                              | Exception          | When                                                         |
|-------------------------------------------|---------------------|----------------------------------------------------------------|
| `create_recognizer(use_mock=False)`       | `FileNotFoundError` | Checkpoint file absent.                                        |
| `create_recognizer(use_mock=False)`       | `RuntimeError`      | Checkpoint deserialization fails, or no model builder can be resolved. |
| `create_recognizer(use_mock=False)`       | `ValueError`        | Checkpoint fails structural or semantic validation.             |
| `SignRecognizer.update`                   | `ValueError`        | Malformed frame data (unrecognized handedness, or wrong-length landmark payload). |
| `Camera.open`                             | `CameraError`       | No candidate camera index can be opened.                       |
| `Camera.read`                             | `CameraError`       | Called before `open()`.                                        |
| `src.landmarks.create_detector`           | `FileNotFoundError` | The user-provided `.task` asset is missing.                    |
| `glosses_to_sentence`                     | (never raises)      | Not applicable.                                                 |
| `TokenBuffer.add`                         | (never raises)      | Not applicable.                                                 |

`create_recognizer(use_mock=False)` propagates `FileNotFoundError`,
`RuntimeError` and `ValueError` UNCHANGED and never falls back to the mock
predictor on a real-load failure (INV-9).

### 4.8 Data Handoff Rules

The value returned by `src.landmarks.extract(detector, frame_bgr, ts_ms)` is
passed UNMODIFIED to `SignRecognizer.update(hands)`. The interface performs
no normalization, no reshaping, no zero-filling and no handedness handling
of its own; all of that is internal to `src/predict.py` (section 4.4). Two
hands detected with the same handedness in one live frame are handled
inside `src/predict.py` (the first is kept, later duplicates are silently
ignored - a deliberate live-robustness divergence from the training
pipeline); the interface does nothing about this. Non-finite (NaN or
infinity) landmark values do not raise; they propagate through
`normalize_landmarks`, which skips scaling when the point spread is not
greater than `1e-6`. This is documented behavior, not something the
interface sanitizes.

## 5. State Model

A single table, columns: state | owner | key or accessor | type | initial
value | lifetime | mutator. The key names below are locked: sections 6, 7, 8
and 10 use exactly these names.

### st.cache_resource (process-global, keyed by argument values; section 6)

| State      | Accessor                                                                 | Type                          | Built                         | Lifetime                                   | Mutator                                              |
|------------|---------------------------------------------------------------------------|--------------------------------|--------------------------------|----------------------------------------------|--------------------------------------------------------|
| camera     | cached factory `get_camera()`                                             | `Camera`                       | on first call                  | process lifetime, or explicit cache clear    | the factory only                                        |
| detector   | cached factory `get_detector()`                                           | `HandLandmarker` (VIDEO mode)  | on first call                  | process lifetime, or explicit cache clear    | the factory only                                        |
| recognizer | cached factory `get_recognizer(use_mock, min_confidence, stability_frames, cooldown_frames)` | `SignRecognizer` | on first call per argument tuple | evicted when the argument tuple changes (`max_entries=1`) | the factory builds it; the loop calls `update()`; the reset control calls `reset()` |

### st.session_state (per browser session, survives reruns within that session)

| Key               | Type            | Initial value | Lifetime | Mutator                                                    |
|--------------------|-----------------|----------------|----------|--------------------------------------------------------------|
| `running`          | bool            | `False`        | session  | start control, stop control                                  |
| `token_buffer`     | `TokenBuffer`   | fresh instance | session  | the loop (`add`), the finish control (`emit`), the reset control (`reset`) |
| `last_sentence`    | str             | `""`           | session  | finish control, reset control, live-preview regeneration     |
| `last_gloss`       | str or None     | `None`         | session  | the loop                                                      |
| `last_confidence`  | float or None   | `None`         | session  | the loop                                                      |
| `last_status`      | str             | `"idle"`       | session  | the loop, reset control                                       |
| `last_latency_ms`  | float or None   | `None`         | session  | the loop                                                      |
| `frames_buffered`  | int             | `0`            | session  | the loop                                                      |
| `last_ts_ms`       | int             | `0`            | session  | the loop (timestamp discipline, section 6)                    |
| `sample_rate_hz`   | float or None   | `None`         | session  | the loop (measured achieved rate)                              |
| `loop_ms`          | float or None   | `None`         | session  | the loop (measured per-iteration wall time)                    |
| `error_message`    | str or None     | `None`         | session  | control handlers and the loop                                  |

Initialization rule: every `session_state` key above is created with its
initial value at the top of the script under an `if key not in
st.session_state` guard, before any control widget is constructed, so no
later code path reads a missing key.

## 6. Resource Lifecycle and Caching

`st.cache_resource` returns the SAME live object for the same argument
values; it is keyed by those argument values, and - unlike per-session
state - it is shared process-wide across every browser session connected to
the running server, not isolated per session.

**Camera.** Opened exactly once, inside `get_camera()`, decorated with
`max_entries=1`. `release()` is NEVER called on loop exit, on a rerun, or in
a `finally` block wrapped around the loop, because the handle is cached and
outlives any single script run (INV-4): releasing it there would leave the
cache holding a dead capture object that a later run would try to read
from, and the observable symptom is a preview that goes black and never
recovers until the process restarts. `release()` is called only by an
explicit teardown path that also clears that cache entry in the same
action; no such path is required by this phase.

**Detector.** Built exactly once, inside `get_detector()`, decorated with
`max_entries=1`. It holds a native `VIDEO`-running-mode session with a
strictly increasing timestamp requirement (INV-11), so rebuilding it on
every rerun would be both slow and would violate that requirement each time
capture resumed. It is not released on any per-run path.

**Recognizer.** `get_recognizer(use_mock, min_confidence, stability_frames,
cooldown_frames)`, decorated with `max_entries=1`. The three smoothing
tunables are constructor parameters and the internal `PredictionSmoother`
instance is a private attribute (section 4.4), so the ONLY sanctioned
mechanism for applying a slider change is RECONSTRUCTION: passing the new
argument values produces a new cache key, and with `max_entries=1` the
previous recognizer is evicted. Reaching into the private attributes of
`SignRecognizer` or its smoother is prohibited (INV-12): they are
implementation detail the inference module is free to change, and code
coupled to them breaks silently on an unrelated refactor there. What
reconstruction loses, stated plainly: the in-flight frame window and all
smoother state (candidate gloss, agreement run, cooldown remaining). What
survives, because it lives in `st.session_state` rather than inside the
recognizer: the token buffer and the last generated sentence.
`max_entries=1` is also what prevents one cached recognizer accumulating
per slider position, which on the real-predictor path would otherwise mean
repeatedly loading a model into memory.

**Multi-session consequence and policy.** Because the cache is
process-global, two browser tabs or two users connecting to the same server
process share one camera handle, one detector and one recognizer; their
frame windows interleave and neither session's recognition result is
meaningful. The documented policy is that this is a single-session demo
application; the design states the consequence rather than attempting
per-session isolation, which plain Streamlit with a single physical camera
cannot provide. If a browser tab is closed while the loop is running, the
server-side loop and the camera handle survive until the session is cleaned
up server-side; the documented guard is to press stop before closing the
tab, and the recommended recovery from an inconsistent shared state is
restarting the server process.

**Timestamp discipline.** `extract(detector, frame_bgr, ts_ms)` requires
`ts_ms` to be strictly increasing for the lifetime of the cached detector
(INV-11). The counter's key lives in `st.session_state` under
`last_ts_ms`, but each frame's timestamp is computed from the process-wide
monotonic clock (floored to milliseconds) and then forced to be strictly
greater than the previously stored value, so the value can never repeat or
move backward across a rerun, across a stop/start cycle, or across two
sessions sharing the cached, process-global detector. A naive per-run frame
counter (for example, one that restarts at zero on every script execution)
is rejected precisely because a rerun would reset it below values the
cached detector has already seen. If a timestamp does repeat or move
backward, MediaPipe's `VIDEO` running mode rejects or mis-tracks the frame,
and the observable failure is intermittent, unexplained detection dropouts
rather than a raised error.

## 7. Control Flow

**Top-of-script flow, in order:**

1. Module docstring; heavy imports (Streamlit, OpenCV, the inference
   pipeline) deferred inside `main()`.
2. Page configuration.
3. `st.session_state` initialization guards (section 5).
4. Sidebar construction: the mock-predictor toggle, the three tunable
   sliders, the display-throttle control.
5. Control row: start, stop, reset, finish sentence.
6. Control handling: handlers only mutate `st.session_state`; a handler
   never enters the capture loop itself (INV-8).
7. Resource acquisition via the cached factories (section 6), wrapped so
   that a `CameraError`, a missing `.task` asset, or a checkpoint failure
   surfaces as one error message and leaves `running` `False` (section 8).
8. Static layout construction: the `st.empty()` preview placeholder and the
   metric placeholders (current sign, confidence, latency, sample rate,
   token list, sentence).
9. Loop entry, if and only if `running` is `True`.

**Single loop iteration, in order and unambiguous:**

a. `ret, frame = camera.read()`. If `ret` is `False`, apply the
   read-failure policy (ERR-6, section 8) and `continue` without any
   recognizer call: a frame that was not captured is not a captured frame,
   and none is owed for it.
b. Compute `ts_ms` from the monotonic source (section 6) and store it back
   to `last_ts_ms`.
c. `hands = landmarks.extract(detector, frame, ts_ms)`.
d. `result = recognizer.update(hands)`. Exactly one call, unconditionally,
   on every captured frame (INV-1, INV-2).
e. If `result.emitted is not None`: `token_buffer.add(result.emitted)`
   (INV-6), and regenerate the live sentence preview from
   `token_buffer.tokens` (non-destructive).
f. Record `result.gloss`, `result.confidence`, `result.status`,
   `result.latency_ms` and `result.frames_buffered` into `st.session_state`.
g. Update the achieved-sample-rate and loop-time measurements (section 9).
h. Only when the iteration counter is a multiple of `DISPLAY_EVERY_N`:
   convert the BGR frame to RGB (INV-10) and write it to the preview
   placeholder, and refresh the metric placeholders. This is the only step
   that is throttled, and the only step that touches the Streamlit API
   (INV-1, INV-7).

`DISPLAY_EVERY_N` is a named module-level constant with a stated default of
`3`. Reasoning: at the measured landmark-stage ceiling of roughly 34 frames
per second (section 9), rendering every third frame yields roughly 11 UI
updates per second, which reads as smooth to a viewer while leaving the
recognizer fed at the full captured rate. The correct value depends on the
achieved capture rate of the demo machine, so it is configurable rather than
fixed to one immutable number. A value of `1` (render every frame) is a
valid but slower configuration: the render cost is then paid on every
iteration, and the achieved sample rate the interface measures and displays
drops accordingly - this is exactly the degradation the measured readout
exists to reveal, not a bug in the readout. A very high value makes the loop
touch the Streamlit API rarely, so stop-control latency grows (INV-7): stop
latency is bounded by `DISPLAY_EVERY_N` divided by the achieved capture
rate.

**Stop mechanism, stated exactly.** The loop does not poll
`st.session_state` to decide whether to stop (INV-8), because a widget
interaction is not visible to the currently running script - it schedules a
rerun (section 3). Pressing stop starts a new script run in which the
control handler sets `running` to `False`; the previously running loop is
terminated by Streamlit at its next Streamlit API call (bounded by the
display throttle, INV-7), and the new run's top-of-script flow does not
re-enter the loop because `running` is now `False`.

**Resume behavior after any widget interaction.** A rerun always terminates
whatever loop was running. The top-of-script flow (above) re-enters the
loop whenever `running` is still `True` at that point, so capture
auto-resumes after a slider move, a mock-toggle flip, or a reset press,
without a separate "resume" action from the operator. A slider move or a
toggle flip changes the argument tuple passed to `get_recognizer`, which
rebuilds the recognizer (section 6) and therefore discards the in-flight
frame window and smoother state while preserving the token buffer and the
displayed sentence, both of which live in `st.session_state`.

## 8. Error Handling Matrix

Columns: id | failure | detected where | user-visible outcome | loop
response.

| ID | Failure | Detected where | User-visible outcome | Loop response |
|----|---------|-----------------|------------------------|-----------------|
| ERR-1 | No webcam, or the device is held by another application. | `Camera.open()` raises `CameraError` after trying indices 0, 1, 2. Detected at resource acquisition, before the loop. | One error message naming the tried indices and the likely cause. | Refuse to start; `running` is not set to `True`. |
| ERR-2 | Missing `models/hand_landmarker.task`. | `create_detector` raises `FileNotFoundError` naming the path. Detected at resource acquisition. | One error message stating the asset is user-provided and never committed, and where to place it. | Refuse to start; recognition is impossible without it. |
| ERR-3 | Real predictor selected and `models/sign_classifier.pt` is absent (the situation in this repository today). | `create_recognizer(use_mock=False)` raises `FileNotFoundError`. | One clear message that also names the mock toggle as the available path. | Refuse to start. Never silently fall back to the mock (INV-9). The page does not crash. |
| ERR-4 | Checkpoint present but structurally or semantically invalid. | `ValueError` or `RuntimeError` from checkpoint validation or deserialization. | Same surfacing rule as ERR-3: one clear message. | Refuse to start; no fallback. |
| ERR-5 | Malformed frame data (an unrecognized handedness string, or a landmark payload of the wrong length). | `SignRecognizer.update` propagates `ValueError`. | The failure message is surfaced. | The loop stops. This indicates a systematic upstream mismatch, not a transient glitch: continuing would repeat the same failure on every subsequent frame and fill the window with data the model cannot use. |
| ERR-6 | Camera read returns `ret=False` mid-loop. | Detected inline in the loop, step (a) of section 7. | No message unless the threshold below is reached. | Treated as a transient device hiccup: the iteration is skipped with no recognizer call, and a consecutive-failure counter increments. At 30 consecutive failed reads (roughly one second at the measured rate), the loop stops with an error message. Any successful read resets the counter to zero. |
| ERR-7 | Checkpoint vocabulary differs from `src.utils.VOCABULARY` while remaining internally self-consistent. | A warning is emitted by the predictor at construction. | The warning is surfaced without blocking. | Continue; predictor ordering is authoritative. The interface takes its label ordering from the predictor, never from a local copy of `VOCABULARY`. |
| ERR-8 | Empty token buffer at sentence-generation time. | `glosses_to_sentence([])` returns `""`. | The interface renders a placeholder, not an error. | Continue; not treated as a failure. |

Non-error states that must not be rendered as failures: `status ==
"warming_up"` during the first `SEQUENCE_LENGTH` frames after a start or
reset, where `gloss`, `confidence`, `latency_ms` and `emitted` are all
`None` and the interface renders a warm-up progress state from
`frames_buffered` rather than blanks; and `status == "no_hands"` when the
window is full but entirely zero, where no inference runs and no emission
occurs, but the internal smoother still receives one non-passing update. If
stop is pressed during warm-up, `last_latency_ms` is still `None`; the
latency readout renders its defined placeholder rather than an empty cell.

## 9. Performance Budget and Measurement

The measured landmark-stage benchmark figures are recorded authoritatively
in `Phase7SPEC.md` section 7 (200 synthetic 640x480 frames; mean 34.1 FPS;
mean 29.31 ms/frame; p95 40.54 ms; p99 52.74 ms; jitter CV 0.32; zero hand
detections). This section restates them only to derive the per-frame time
budget below; it does not alter or re-derive the figures themselves. The
same caveat applies here as there: because no hands were present, the
landmark regression short-circuited, so these figures are an OPTIMISTIC
CEILING and exclude classifier inference cost.

Budget derivation: `SEQUENCE_LENGTH` (30 frames) is intended to span roughly
1.0 to 1.2 seconds (section 4.1), which implies a per-frame time budget of
roughly 33 to 40 milliseconds for capture plus landmark extraction plus
inference plus the amortised display cost. Against that budget, the
measured landmark-stage mean of 29.31 ms and p95 of 40.54 ms already consume
most or all of the available headroom before classifier cost or display
cost is even added - and that measurement excludes hands entirely. The
headroom is therefore thin, and the achieved rate must be measured on the
running system rather than assumed from any of these figures.

Measurement specification:

- The achieved sample rate (`sample_rate_hz`) is computed as recognizer
  updates divided by elapsed wall-clock time over a rolling window of
  recent loop iterations, and is labelled in the interface as measured.
- Two distinct latency figures are displayed, and they are labelled
  distinctly rather than merged into one number: the inference-only figure
  taken from `RecognitionResult.latency_ms` (excludes capture, landmark
  extraction and display), and the per-iteration loop time (`loop_ms`) the
  interface measures itself around the whole iteration body in section 7.
- No nominal frame rate (for example `src.camera.FPS`, or the benchmark's
  mean FPS above) is ever displayed as if it were measured (`Phase7SPEC.md`
  section 7).

## 10. Invariants

Every invariant below is presented in exactly three labelled parts - Rule,
Reason, Failure mode if violated - and none appears with fewer than all
three.

**INV-1 (the single most important rule in this document). Sample-rate /
display-rate decoupling.**
Rule: every captured frame is processed and fed to the recognition buffer;
only every Nth frame is rendered to the UI.
Reason: `SEQUENCE_LENGTH` is 30 frames, documented as roughly 1.0-1.2
seconds at 25-30 fps (section 4.1); training data is captured at
near-native webcam rate; feeding the buffer at the UI display rate instead
of the capture rate makes the same 30-frame window span two to three
seconds instead of one; the model then sees temporally stretched signs it
was never trained on; and recognition accuracy degrades in a way that looks
like a model fault but is actually a UI fault.
Failure mode if violated: an implementer who calls `recognizer.update()`
only on the throttled render steps (or only when displaying) silently
stretches every 30-frame window across two to three seconds of real time;
the resulting accuracy loss is misattributed to the model for the remainder
of the project, because nothing about the symptom points at the UI.

**INV-2. Exactly one update per captured frame.**
Rule: `SignRecognizer.update()` is called exactly once per captured frame -
never zero times, never more than once.
Reason: `RecognitionResult.frames_buffered`, the smoother's update-counted
tunables (section 4.4), and the achieved-sample-rate measurement (section
9) are all defined in terms of "one call per captured frame"; skipping or
duplicating calls breaks that unit definition.
Failure mode if violated: skipping calls reproduces the INV-1 failure by a
different route; duplicating calls on a single frame corrupts the
smoother's stability and cooldown counters and the achieved-rate
measurement, both silently.

**INV-3. `SEQUENCE_LENGTH` and `VOCABULARY` are imported, never restated.**
Rule: the interface imports `SEQUENCE_LENGTH` and `VOCABULARY` from
`src/utils.py` and never hardcodes either value.
Reason: `src/utils.py` is the project's single source of truth for both,
per the repository's locked conventions; restating either value anywhere else creates a
second place that can silently drift out of sync with training and
inference.
Failure mode if violated: a restated `SEQUENCE_LENGTH` or a restated
vocabulary list can diverge from the source of truth after a later change
to `src/utils.py`, producing a shape mismatch or an incorrect label set that
is invisible until a checkpoint trained against the updated values is
loaded.

**INV-4. Camera, detector and recognizer live under `st.cache_resource`.**
Rule: the camera handle, the landmark detector, and the recognizer are all
created under `st.cache_resource` factories (section 5, section 6).
Reason: a Streamlit script re-executes on every rerun (section 3); without
caching, each rerun would reopen the camera, reconstruct the detector, and
reload or rebuild the recognizer.
Failure mode if violated: a new camera handle or a reloaded model on every
rerun - at minimum a severe, visible stutter and slowdown on ordinary
interactions like a slider move, and at worst a device left open and never
released, or a fresh `VIDEO`-mode detector whose timestamp counter restarts
at zero (INV-11).

**INV-5. Token buffer and session flags live in `st.session_state`.**
Rule: the token buffer and the session control flags (`running`,
`last_sentence`, and the rest of the table in section 5) live in
`st.session_state`, never in an ordinary local variable of the script.
Reason: local variables do not survive a rerun (section 3); a widget
interaction always triggers one.
Failure mode if violated: every accumulated recognized token, and the
running/stopped state itself, is lost on the very next widget interaction -
the interface would appear to "forget" everything each time a slider is
touched.

**INV-6. Only `RecognitionResult.emitted` feeds the token buffer.**
Rule: only `RecognitionResult.emitted` is ever passed to
`TokenBuffer.add()`; `RecognitionResult.gloss` is never passed to it.
Reason: `gloss` is the raw, per-window top-1 prediction and changes on
nearly every inference; `emitted` is the smoothed, debounced, confidence-
gated acceptance that `PredictionSmoother` exists to produce (section 4.4).
Failure mode if violated: feeding raw `gloss` into the token buffer floods
it with noisy, rapidly-changing entries, defeating the entire purpose of
the stability/cooldown smoothing and producing a token stream and a
generated sentence that do not correspond to anything a user actually held.

**INV-7. The loop must be interruptible.**
Rule: the capture loop touches a Streamlit API (the throttled render step,
section 7 step h) often enough that a pending rerun can interrupt it.
Reason: Streamlit terminates a running script only at its next call into a
Streamlit API (section 3); a loop that never calls one cannot be
interrupted by any widget interaction, including stop. The interaction with
the display throttle is direct: stop latency is bounded by
`DISPLAY_EVERY_N` divided by the achieved capture rate.
Failure mode if violated: an uninterruptible loop and an unresponsive stop
control - the operator presses stop and nothing happens until the process
is killed externally, and a `DISPLAY_EVERY_N` set far too high reproduces a
milder version of the same symptom as a growing, user-visible delay.

**INV-8. The loop never polls session state to decide to stop.**
Rule: the loop does not read `st.session_state["running"]` inside its own
iteration to decide whether to keep going based on some other mechanism
guessing at a pending interaction; it stops because Streamlit tears the
script down on rerun (section 3, section 7).
Reason: a widget interaction is only visible to a script AFTER the rerun
that interaction triggers; the currently running script's local view of
`st.session_state` is a snapshot from when it started and does not update
mid-loop.
Failure mode if violated: code written to "poll and break" is solving a
problem the execution model already solves and gives a false sense that the
loop is checking something live, masking the real interruption mechanism
(INV-7) and making the actual stop latency harder to reason about.

**INV-9. No silent fallback from the real predictor to the mock.**
Rule: when the real predictor is selected and fails to load, the interface
never silently substitutes the mock predictor.
Reason: silently substituting a different predictor after an explicit
selection hides a load failure behind output that looks like normal
operation, which is indistinguishable to an operator from a working real
model (ERR-3, ERR-4).
Failure mode if violated: a demo or an evaluation session runs entirely on
the mock predictor while the operator believes the real, trained model is
active, invalidating any accuracy observation made during that session.

**INV-10. BGR to RGB conversion before display.**
Rule: frames are converted from BGR (as delivered by OpenCV) to RGB before
being written to the preview placeholder.
Reason: `st.image` and the browser's image rendering expect RGB channel
order; OpenCV delivers BGR.
Failure mode if violated: the preview displays with red and blue channels
swapped - visibly wrong skin tones and colors - without raising any error,
so the defect could ship unnoticed if not checked visually.

**INV-11. Landmark timestamps are strictly increasing.**
Rule: the `ts_ms` value passed to `extract(detector, frame_bgr, ts_ms)` is
strictly increasing for the entire lifetime of the cached, process-global
detector (section 6).
Reason: MediaPipe's `VIDEO` running mode requires monotonically increasing
timestamps to track state correctly across calls.
Failure mode if violated: a timestamp that repeats or moves backward
(easy to trigger with a naive per-run counter across a rerun, per section 6)
causes MediaPipe to reject or mis-track the frame, surfacing as
intermittent, unexplained hand-detection dropouts that do not correlate
with anything visible in the preview.

**INV-12. Private attributes of `SignRecognizer` or its smoother are never
read or mutated.**
Rule: the interface never reads or writes a private attribute of
`SignRecognizer` or its internal `PredictionSmoother`; tunable changes are
applied only by reconstructing the recognizer (section 6).
Reason: private attributes are implementation detail that `src/predict.py`
is free to change at any time without notice; the module's only documented,
stable contract is its public API (section 4.4).
Failure mode if violated: an interface coupled to a private attribute
compiles and runs today but breaks silently - or with a confusing
`AttributeError` - the next time `src/predict.py` is refactored, even
though its public behavior has not changed.

## 11. Requirement Traceability

Every functional requirement from `Phase7SPEC.md` section 4 is mapped below
to the design section that realizes it, or is marked as intentionally
unconstrained by this design.

| FR   | Design section(s)                                                        |
|------|------------------------------------------------------------------------------|
| FR-1  | Section 7 (control flow, step h); INV-1; INV-10.                            |
| FR-2  | Section 5 (running); section 6 (camera/detector acquisition); section 7 (top-of-script step 7); section 8 (ERR-1, ERR-2, ERR-3, ERR-4). |
| FR-3  | Section 3 (rerun termination); section 7 (stop mechanism); INV-7; INV-8.     |
| FR-4  | Section 5 (session_state reset targets); section 6 (recognizer/token_buffer reset semantics). |
| FR-5  | Section 4.4 (`RecognitionResult.gloss`, `status`); section 5 (`last_gloss`, `last_status`); section 8 (warming_up / no_hands). |
| FR-6  | Section 4.4 (`RecognitionResult.confidence`); section 5 (`last_confidence`). |
| FR-7  | Section 4.5 (`TokenBuffer.tokens`); section 5 (`token_buffer`); section 7 step e. |
| FR-8  | Section 4.5 (`TokenBuffer.tokens`); section 4.6 (`glosses_to_sentence`); section 7 step e. |
| FR-9  | Section 4.5 (`TokenBuffer.emit`); section 4.6 (`glosses_to_sentence`); section 5 (`last_sentence`). |
| FR-10 | Section 4.4 (`RecognitionResult.latency_ms`, `SignRecognizer.last_latency_ms`); section 5 (`last_latency_ms`); section 9 (two distinct latency figures); section 8 (warm-up placeholder). |
| FR-11 | Section 5 (`sample_rate_hz`); section 9 (measurement specification); INV-1. |
| FR-12 | Section 4.4 (`min_confidence`); section 6 (reconstruction mechanism); INV-12. |
| FR-13 | Section 4.4 (`stability_frames`); section 6 (reconstruction mechanism); INV-12. |
| FR-14 | Section 4.4 (`cooldown_frames`); section 6 (reconstruction mechanism); INV-12. |
| FR-15 | Section 5 (`recognizer` cache key includes `use_mock`); section 6 (reconstruction on toggle); section 8 (ERR-3, ERR-4); INV-9. |

No `FR` is intentionally unconstrained: every functional requirement in
`Phase7SPEC.md` section 4 has at least one design-section mapping above.

## 12. Notes on Warranted Code Changes

No code was changed under this task. This document and `Phase7SPEC.md` are
the only artifacts produced; no file under `src/`, `app.py`, `tests/`, or
any other existing path was created, modified, deleted or renamed. No
change to any existing module is believed warranted by the work of writing
these two documents: the transcription in section 4 matches the current
source in `src/utils.py`, `src/camera.py`, `src/landmarks.py`,
`src/predict.py`, `src/buffer.py` and `src/grammar.py` as read during
authoring, and no defect or inconsistency in that source was identified in
the course of this documentation task.
