"""Streamlit app shell for the Real-Time Limited-Vocabulary ASL-to-Text
Recognition System.

Phase 7 demo interface. This file currently contains the application shell
only: page layout, session-state initialization, the configuration sidebar,
the control row and the telemetry/sequence placeholder layout. The capture
loop and the recognition pipeline will be wired in future iterations; the
control callbacks are placeholders reserved for that work.

The displayed results will be classification outputs for a fixed 25-sign
academic/classroom vocabulary under controlled webcam conditions; nothing
here is, or should be read as, general or unrestricted ASL translation.
"""

import streamlit as st

from typing import Final

from src.buffer import TokenBuffer
from src.utils import VOCABULARY

PAGE_TITLE = "ASL-to-Text Classroom Demo"

SESSION_STATE_DEFAULT = {
    "running": False,
    "token_buffer": TokenBuffer(),
    "last_sentence": "",
    "last_gloss": None,
    "last_confidence": None,
    "last_status": "idle",
    "last_latency_ms": None,
    "frames_buffered": 0,
    "last_ts_ms": 0,
    "sample_rate_hz": None,
    "loop_ms": None,
    "error_message": None,
}

# Confidence thresholds
CONFIDENCE_MIN: Final[float] = 0.0
CONFIDENCE_MAX: Final[float] = 1.0
CONFIDENCE_DEFAULT: Final[float] = 0.60
CONFIDENCE_STEP: Final[float] = 0.01

# Stability window
STABILITY_WINDOW_MIN: Final[int] = 0
STABILITY_WINDOW_MAX: Final[int] = 30
STABILITY_WINDOW_DEFAULT: Final[int] = 5
STABILITY_WINDOW_STEP: Final[int] = 1

# Cooldown frames
COOLDOWN_FRAME_MIN: Final[int] = 0
COOLDOWN_FRAME_MAX: Final[int] = 100
COOLDOWN_FRAME_DEFAULT: Final[int] = 15
COOLDOWN_FRAME_STEP: Final[int] = 1

# Display interval
DISPLAY_EVERY_N_FRAME_MIN: Final[int] = 1
DISPLAY_EVERY_N_FRAME_MAX: Final[int] = 10
DISPLAY_EVERY_N_FRAME_DEFAULT: Final[int] = 3
DISPLAY_EVERY_N_FRAME_STEP: Final[int] = 1


def main():
    st.set_page_config(
        page_title=PAGE_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Limited-Vocabulary ASL-to-Text Recognition Prototype")
    st.markdown(
        """
        This university capstone prototype recognizes a small, fixed vocabulary of 25 academic signs 
        under controlled webcam conditions. It is not a general-purpose ASL translation tool.
        """
    )

    for key, default in SESSION_STATE_DEFAULT.items():
        if key not in st.session_state:
            st.session_state[key] = default

    st.sidebar.header("Configuration & Tunables")

    use_mock = st.sidebar.toggle(
        "Use Mock Predictor (Simulated)",
        value=True,
        help="Runs end-to-end without needing a trained classifier checkpoint.",
    )

    min_confidence = st.sidebar.slider(
        "Confidence Threshold",
        min_value=CONFIDENCE_MIN,
        max_value=CONFIDENCE_MAX,
        value=CONFIDENCE_DEFAULT,
        step=CONFIDENCE_STEP,
        help="Minimum confidence score needed to register a sign.",
    )

    stability_frames = st.sidebar.slider(
        "Stability Window",
        min_value=STABILITY_WINDOW_MIN,
        max_value=STABILITY_WINDOW_MAX,
        value=STABILITY_WINDOW_DEFAULT,
        step=STABILITY_WINDOW_STEP,
        help="Number of consecutive frames a sign must be held to accept it.",
    )

    cooldown_frames = st.sidebar.slider(
        "Cooldown Frames",
        min_value=COOLDOWN_FRAME_MIN,
        max_value=COOLDOWN_FRAME_MAX,
        value=COOLDOWN_FRAME_DEFAULT,
        step=COOLDOWN_FRAME_STEP,
        help="Frames to wait after accepting a sign before detecting another.",
    )

    display_every_n = st.sidebar.slider(
        "Display Throttle (Every N Frames)",
        min_value=DISPLAY_EVERY_N_FRAME_MIN,
        max_value=DISPLAY_EVERY_N_FRAME_MAX,
        value=DISPLAY_EVERY_N_FRAME_DEFAULT,
        step=DISPLAY_EVERY_N_FRAME_STEP,
        help="Decouples UI updates from loop execution to maintain high FPS.",
    )

    with st.sidebar.expander("Supported Vocabulary (25 Signs)"):
        st.write(", ".join(VOCABULARY))

    def cb_start():
        pass

    def cb_stop():
        pass

    def cb_reset():
        pass

    def cb_finish():
        pass

    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
    btn_col1.button(
        "Start Feed",
        on_click=cb_start,
        use_container_width=True,
    )
    btn_col2.button(
        "Stop Feed",
        on_click=cb_stop,
        use_container_width=True,
    )
    btn_col3.button(
        "Reset Buffer",
        on_click=cb_reset,
        use_container_width=True,
    )
    btn_col4.button(
        "Finish Sentence",
        on_click=cb_finish,
        use_container_width=True,
    )

    col_left, col_right = st.columns([2, 1])

    with col_left:
        preview_placeholder = st.empty()
        preview_placeholder.info(
            "Webcam feed is currently INACTIVE. Click 'Start Feed' above to begin."
        )

    with col_right:
        st.subheader("Telemetry & Diagnostics")
        gloss_placeholder = st.empty()
        confidence_placeholder = st.empty()
        status_placeholder = st.empty()
        latency_placeholder = st.empty()
        loop_time_placeholder = st.empty()
        sample_rate_placeholder = st.empty()

        gloss_placeholder.metric("Current Predicted Sign", "\u2014")
        confidence_placeholder.metric("Inference Confidence", "\u2014")
        status_placeholder.markdown("**Recognizer Status:** `idle`")
        latency_placeholder.markdown(
            "**Inference Latency:** `\u2014` (inference-only)"
        )
        loop_time_placeholder.markdown(
            "**Whole-Loop Step:** `\u2014` (measured)"
        )
        sample_rate_placeholder.markdown(
            "**Achieved Sample Rate:** `\u2014` (measured)"
        )

    st.subheader("Recognized Sequence")
    tokens_placeholder = st.empty()
    sentence_placeholder = st.empty()

    tokens_placeholder.markdown(
        "**Accepted Gloss Sequence:** *None yet*"
    )
    sentence_placeholder.markdown(
        "### Sentence Output:\n> *No sequence generated yet. Start feed and hold hand signs!*"
    )


if __name__ == "__main__":
    main()
