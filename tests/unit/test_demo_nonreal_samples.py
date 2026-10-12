from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ecg_trust import demo_backend
from ecg_trust.constants import LEADS
from ecg_trust.demo_backend import DemoInputError, validate_physical_signal


@pytest.mark.parametrize("kind", ["numpy", "torch", "mixed_list"])
def test_boolean_samples_cannot_be_reinterpreted_as_physical_millivolts(kind: str) -> None:
    if kind == "numpy":
        signal = np.ones((12, 1000), dtype=bool)
    elif kind == "torch":
        signal = torch.ones(12, 1000, dtype=torch.bool)
    else:
        signal = np.ones((12, 1000)).tolist()
        signal[0][0] = True
    with pytest.raises(DemoInputError):
        validate_physical_signal(signal)


@pytest.mark.parametrize("kind", ["bool_array", "bool_object", "complex_object", "mixed_list"])
def test_wfdb_reader_refuses_nonreal_samples_before_float_conversion(
    tmp_path, monkeypatch, kind: str,
) -> None:
    (tmp_path / "record.hea").touch()
    (tmp_path / "record.dat").touch()
    physical = np.ones((1000, 12), dtype=object)
    physical[0, 0] = np.complex128(1 + 9j) if kind == "complex_object" else True
    if kind == "bool_array":
        physical = physical.astype(bool)
    elif kind == "mixed_list":
        physical = physical.tolist()
    record = SimpleNamespace(fs=100, sig_name=LEADS, p_signal=physical, units=["mV"] * 12)
    monkeypatch.setattr(demo_backend.wfdb, "rdrecord", lambda _: record)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(DemoInputError):
            demo_backend.load_wfdb_physical_signal(tmp_path / "record")
