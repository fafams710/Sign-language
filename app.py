"""Streamlit UI entry point for the Real-Time ASL-to-Text prototype.

This is the demo front-end. When implemented, the UI will display:
  - a live webcam preview,
  - the current predicted sign and its confidence score,
  - the running list of recognized sign tokens,
  - the generated English sentence,
  - clear / reset controls,
  - a latency readout (sign input to displayed prediction).

Heavy imports (streamlit, OpenCV, the inference pipeline) are deferred into
``main()`` so this module stays importable even when those libraries are not
installed.
"""

# TODO (Phase 7 - UI & demo polish): build the Streamlit demo interface.


def main():
    """Launch the Streamlit demo application."""
    # TODO: import streamlit and wire up the inference pipeline here.
    pass


if __name__ == "__main__":
    main()
