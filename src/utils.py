"""Shared helpers and project-wide constants.

Holds the MVP vocabulary, the fixed sequence length, and common path helpers
used across the data, training, and inference modules.
"""

# TODO (Phase 0 - Setup): expand path helpers and configuration as later phases need them.

# Fixed per-sample frame count for the sequence classifier. 30 frames is roughly
# 1.0-1.2 seconds of motion at the 25-30 fps a standard webcam delivers, which is
# enough to cover one performance of a short academic sign while keeping the input
# tensor small. This is the single source of truth for the fixed-length convention;
# preprocess/train/predict import it from here rather than redefining it.
SEQUENCE_LENGTH = 30

# 25-sign MVP vocabulary for the academic/classroom prototype.
VOCABULARY = [
    "HELLO",
    "YES",
    "NO",
    "PLEASE",
    "THANK_YOU",
    "SORRY",
    "HELP",
    "QUESTION",
    "REPEAT",
    "UNDERSTAND",
    "TEACHER",
    "STUDENT",
    "CLASS",
    "ASSIGNMENT",
    "EXAM",
    "PROJECT",
    "PRESENTATION",
    "SUBMIT",
    "TODAY",
    "TOMORROW",
    "NEED",
    "FINISH",
    "WHERE",
    "WHAT",
    "WHY",
]
