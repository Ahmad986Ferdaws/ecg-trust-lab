from ecg_trust.constants import (
    EXPECTED_PATIENTS,
    EXPECTED_RECORDS,
    LEADS,
    PTBXL_VERSION,
    SUPERCLASSES,
)


def test_ptbxl_contract() -> None:
    assert PTBXL_VERSION == "1.0.3"
    assert EXPECTED_RECORDS == 21_799
    assert EXPECTED_PATIENTS == 18_869
    assert SUPERCLASSES == ("NORM", "MI", "STTC", "CD", "HYP")
    assert len(LEADS) == 12
    assert len(set(LEADS)) == 12
