"""Project-wide data contracts."""

SUPERCLASSES: tuple[str, ...] = ("NORM", "MI", "STTC", "CD", "HYP")
TARGET_COLUMNS: tuple[str, ...] = tuple(f"label_{superclass}" for superclass in SUPERCLASSES)
LEADS: tuple[str, ...] = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)
PTBXL_VERSION = "1.0.3"
EXPECTED_RECORDS = 21_799
EXPECTED_PATIENTS = 18_869
