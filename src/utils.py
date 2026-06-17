"""Shared helpers and project-wide constants.

Holds the MVP vocabulary, the fixed sequence length, and common path helpers
used across the data, training, and inference modules.
"""

# TODO (Phase 0 - Setup): expand path helpers and configuration as later phases need them.

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
