from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from ecg_trust import demo_backend
from ecg_trust.constants import LEADS
from ecg_trust.demo_backend import DemoInputError, validate_physical_signal


@pytest.mark.parametrize("kind", ["numpy", "torch", "list"])
def test_complex_signal_is_rejected_before_imaginary_values_can_be_discarded(kind: str) -> None:
    values = np.ones((12, 1000), dtype=np.complex64) * (1 + 9j)
    signal: Any = values
    if kind == "torch":
        signal = torch.from_numpy(values)
    elif kind == "list":
        signal = values.tolist()
    with pytest.raises(DemoInputError, match="real"):
        validate_physical_signal(signal)


@pytest.mark.parametrize("units", ["MV", "mv", "Mv", "uV", "V"])
def test_si_prefixes_are_case_sensitive(units: str) -> None:
    with pytest.raises(DemoInputError, match="units"):
        validate_physical_signal(torch.zeros(12, 1000), units=units)


@pytest.mark.parametrize("bad_value", ["complex", "megavolts"])
def test_wfdb_signal_contract_rejects_lossy_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_value: str,
) -> None:
    (tmp_path / "record.hea").touch()
    (tmp_path / "record.dat").touch()
    signal = np.ones((1000, 12), dtype=np.float32)
    units = ["mV"] * 12
    if bad_value == "complex":
        signal = signal.astype(np.complex64) * (1 + 2j)
    else:
        units[0] = "MV"
    record = SimpleNamespace(fs=100, sig_name=list(LEADS), p_signal=signal, units=units)
    monkeypatch.setattr(demo_backend.wfdb, "rdrecord", lambda _: record)
    with pytest.raises(DemoInputError):
        demo_backend.load_wfdb_physical_signal(tmp_path / "record")


def test_real_signal_keeps_values_and_float32_contract() -> None:
    signal = np.linspace(-1, 1, 12_000, dtype=np.float64).reshape(12, 1000)
    result = validate_physical_signal(signal, units="mV")
    assert result.dtype == torch.float32
    assert result.is_contiguous()
    np.testing.assert_array_equal(result.numpy(), signal.astype(np.float32))
