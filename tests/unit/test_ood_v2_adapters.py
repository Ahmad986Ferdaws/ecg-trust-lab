from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import wfdb  # type: ignore[import-untyped]
from scipy.signal import resample_poly

from ecg_trust.constants import LEADS
from ecg_trust.ood_v2 import adapters
from ecg_trust.ood_v2.adapters import (
    ADAPTER_VERSION,
    CanonicalExternalSignal,
    ExternalECGAdapterError,
    load_challenge_2011_signal,
    load_wfdb_12lead_signal,
    load_zzu_pediatric_signal,
)


def _raw_files(tmp_path: Path, name: str = "records/example") -> Path:
    base = tmp_path / name
    base.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{base}.hea").write_bytes(b"synthetic-header")
    Path(f"{base}.dat").write_bytes(b"synthetic-data")
    return base


def _header(
    *,
    fs: float = 500.0,
    sig_len: int = 6_000,
    names: tuple[str, ...] = LEADS,
    units: tuple[str, ...] | None = None,
    file_names: tuple[str, ...] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        fs=fs,
        sig_len=sig_len,
        n_sig=len(names),
        sig_name=list(names),
        units=list(units or (("mV",) * len(names))),
        file_name=list(file_names or (("example.dat",) * len(names))),
    )


def _install_valid_reader(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_order: tuple[str, ...] = tuple(reversed(LEADS)),
    physical: np.ndarray[Any, Any] | None = None,
    header: SimpleNamespace | None = None,
    loaded_file_names: tuple[str, ...] | None = None,
    loaded_units: tuple[str, ...] | None = None,
) -> np.ndarray[Any, Any]:
    canonical = np.stack(
        [np.arange(5_000, dtype=np.float64) / 5_000.0 + index for index in range(12)],
        axis=1,
    )
    canonical_source_order = adapters.canonicalize_source_lead_names(source_order)
    source_positions = [LEADS.index(name) for name in canonical_source_order]
    source = canonical[:, source_positions] if physical is None else physical

    def fake_rdheader(record_path: str) -> SimpleNamespace:
        return header or _header(
            names=source_order,
            file_names=(f"{Path(record_path).name}.dat",) * len(source_order),
        )

    monkeypatch.setattr(adapters.wfdb, "rdheader", fake_rdheader)

    def fake_rdrecord(
        record_path: str,
        *,
        sampfrom: int,
        sampto: int,
        physical: bool,
        return_res: int,
    ) -> SimpleNamespace:
        assert sampfrom == 0
        assert sampto == 5_000
        assert physical is True
        assert return_res == 64
        return SimpleNamespace(
            fs=500.0,
            sig_len=5_000,
            n_sig=12,
            sig_name=list(source_order),
            units=list(loaded_units or (("mV",) * 12)),
            file_name=list(
                loaded_file_names or ((f"{Path(record_path).name}.dat",) * 12)
            ),
            p_signal=source,
        )

    monkeypatch.setattr(adapters.wfdb, "rdrecord", fake_rdrecord)
    return canonical


def test_loads_first_ten_seconds_reorders_leads_and_resamples_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_base = _raw_files(tmp_path)
    canonical = _install_valid_reader(monkeypatch)

    loaded = load_wfdb_12lead_signal(record_base)
    expected = resample_poly(canonical.T, up=1, down=5, axis=1).astype(np.float32)

    assert isinstance(loaded, CanonicalExternalSignal)
    assert loaded.signal_mv.shape == (12, 1_000)
    assert loaded.signal_mv.dtype == np.float32
    assert loaded.signal_mv.flags.c_contiguous
    assert not loaded.signal_mv.flags.writeable
    np.testing.assert_allclose(loaded.signal_mv, expected, rtol=1e-6, atol=1e-6)
    assert loaded.source_frequency_hz == 500.0
    assert loaded.source_duration_seconds == 12.0
    assert loaded.source_lead_names == tuple(reversed(LEADS))
    assert loaded.canonical_leads == tuple(reversed(LEADS))
    assert loaded.output_leads == LEADS
    assert loaded.source_data_file_names == ("example.dat",) * 12
    assert loaded.raw_physical_units == ("mV",) * 12
    assert loaded.provenance.source_sample_count == 6_000
    assert loaded.adapter_version == ADAPTER_VERSION
    assert loaded.provenance.resample_up == 1
    assert loaded.provenance.resample_down == 5
    assert loaded.provenance.resample_window == ("kaiser", 5.0)
    assert loaded.provenance.resample_padtype == "constant"
    assert loaded.provenance.window_start_sample == 0
    assert loaded.provenance.window_source_samples == 5_000
    assert loaded.provenance_sha256.startswith("sha256:")
    assert len(loaded.provenance_sha256) == 71
    assert hash(loaded.provenance) == hash(loaded.provenance)
    with pytest.raises(ValueError):
        loaded.signal_mv[0, 0] = 99.0


def test_dataset_wrappers_share_the_frozen_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _raw_files(tmp_path, "challenge/a01")
    second = _raw_files(tmp_path, "zzu/p001")
    _install_valid_reader(monkeypatch, source_order=LEADS)

    challenge = load_challenge_2011_signal(first)
    pediatric = load_zzu_pediatric_signal(second)

    assert challenge.provenance.resample_up == pediatric.provenance.resample_up == 1
    assert challenge.provenance.resample_down == pediatric.provenance.resample_down == 5


def test_adapter_rejects_a_junction_in_record_ancestry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_base = _raw_files(tmp_path)
    junction = record_base.parent
    original = Path.is_junction
    monkeypatch.setattr(
        Path,
        "is_junction",
        lambda self: self == junction or original(self),
    )
    monkeypatch.setattr(
        adapters.wfdb,
        "rdheader",
        lambda *_args, **_kwargs: pytest.fail("junction must fail before WFDB access"),
    )
    with pytest.raises(ExternalECGAdapterError, match="indirect"):
        load_wfdb_12lead_signal(record_base)


def test_accepts_only_the_explicit_zzu_augmented_limb_lead_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_base = _raw_files(tmp_path)
    zzu_names = tuple(
        {"aVR": "AVR", "aVL": "AVL", "aVF": "AVF"}.get(lead, lead) for lead in LEADS
    )
    _install_valid_reader(monkeypatch, source_order=zzu_names)

    loaded = load_zzu_pediatric_signal(record_base)

    assert loaded.source_lead_names == zzu_names
    assert loaded.provenance.canonical_leads == LEADS
    assert loaded.output_leads == LEADS


def test_challenge_rejects_zzu_only_uppercase_aliases_before_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_base = _raw_files(tmp_path)
    zzu_names = tuple(
        {"aVR": "AVR", "aVL": "AVL", "aVF": "AVF"}.get(lead, lead) for lead in LEADS
    )
    monkeypatch.setattr(adapters.wfdb, "rdheader", lambda _: _header(names=zzu_names))
    monkeypatch.setattr(
        adapters.wfdb,
        "rdrecord",
        lambda *args, **kwargs: pytest.fail("Challenge aliases must fail before decode"),
    )

    with pytest.raises(ExternalECGAdapterError, match="unsupported"):
        load_challenge_2011_signal(record_base)


def test_permitted_alias_map_changes_names_only_never_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_base = _raw_files(tmp_path, "canonical/example")
    alias_base = _raw_files(tmp_path, "aliased/example")
    alias_names = tuple(
        {"aVR": "AVR", "aVL": "AVL", "aVF": "AVF"}.get(lead, lead) for lead in LEADS
    )

    _install_valid_reader(monkeypatch, source_order=LEADS)
    canonical = load_wfdb_12lead_signal(canonical_base)
    _install_valid_reader(monkeypatch, source_order=alias_names)
    aliased = load_wfdb_12lead_signal(alias_base)

    assert canonical.source_lead_names == LEADS
    assert aliased.source_lead_names == alias_names
    assert canonical.canonical_leads == aliased.canonical_leads == LEADS
    np.testing.assert_array_equal(canonical.signal_mv, aliased.signal_mv)


@pytest.mark.parametrize("unsupported", ["avr", "AvR", " aVR", "aVR ", "AVr"])
def test_rejects_casefold_and_whitespace_lead_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsupported: str,
) -> None:
    record_base = _raw_files(tmp_path)
    names = tuple(unsupported if lead == "aVR" else lead for lead in LEADS)
    monkeypatch.setattr(adapters.wfdb, "rdheader", lambda _: _header(names=names))
    monkeypatch.setattr(
        adapters.wfdb,
        "rdrecord",
        lambda *args, **kwargs: pytest.fail("unsupported aliases must fail before waveform read"),
    )

    with pytest.raises(ExternalECGAdapterError, match="unsupported"):
        load_wfdb_12lead_signal(record_base)


@pytest.mark.parametrize("bad_name", ["other.dat", "../example.dat", "sub/example.dat"])
def test_rejects_alternate_or_traversing_header_data_file_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_name: str,
) -> None:
    record_base = _raw_files(tmp_path)
    header = _header(file_names=(bad_name,) * 12)
    monkeypatch.setattr(adapters.wfdb, "rdheader", lambda _: header)
    monkeypatch.setattr(
        adapters.wfdb,
        "rdrecord",
        lambda *args, **kwargs: pytest.fail("bad file bindings must fail before waveform read"),
    )

    with pytest.raises(ExternalECGAdapterError, match="bind every lead"):
        load_wfdb_12lead_signal(record_base)


def test_rejects_loaded_record_data_file_binding_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_base = _raw_files(tmp_path)
    _install_valid_reader(
        monkeypatch,
        source_order=LEADS,
        loaded_file_names=("other.dat",) * 12,
    )

    with pytest.raises(ExternalECGAdapterError, match="loaded data-file bindings"):
        load_wfdb_12lead_signal(record_base)


@pytest.mark.parametrize("raw_unit", ["mv", "MV", "mV ", " mV"])
def test_rejects_nonexact_raw_physical_units_before_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_unit: str,
) -> None:
    record_base = _raw_files(tmp_path)
    monkeypatch.setattr(
        adapters.wfdb,
        "rdheader",
        lambda _: _header(units=(raw_unit,) * 12),
    )
    monkeypatch.setattr(
        adapters.wfdb,
        "rdrecord",
        lambda *args, **kwargs: pytest.fail("invalid units must fail before decode"),
    )

    with pytest.raises(ExternalECGAdapterError, match="exact physical unit"):
        load_wfdb_12lead_signal(record_base)


def test_rejects_loaded_record_raw_unit_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_base = _raw_files(tmp_path)
    _install_valid_reader(
        monkeypatch,
        source_order=LEADS,
        loaded_units=("MV",) * 12,
    )

    with pytest.raises(ExternalECGAdapterError, match="physical unit"):
        load_wfdb_12lead_signal(record_base)


@pytest.mark.parametrize(
    ("header", "message"),
    [
        (_header(fs=250.0), "expected"),
        (_header(sig_len=4_999), "shorter"),
        (_header(names=LEADS[:-1]), "12 leads"),
        (_header(names=(*LEADS[:-1], "X")), "unsupported"),
        (_header(names=(*LEADS[:-1], "I")), "duplicate"),
        (_header(units=("V",) * 12), "physical unit"),
    ],
)
def test_rejects_header_contract_violations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    header: SimpleNamespace,
    message: str,
) -> None:
    record_base = _raw_files(tmp_path)
    monkeypatch.setattr(adapters.wfdb, "rdheader", lambda _: header)
    monkeypatch.setattr(
        adapters.wfdb,
        "rdrecord",
        lambda *args, **kwargs: pytest.fail("invalid header must fail before waveform read"),
    )

    with pytest.raises(ExternalECGAdapterError, match=message):
        load_wfdb_12lead_signal(record_base)


@pytest.mark.parametrize("case", ["non_finite", "wrong_shape", "no_physical"])
def test_rejects_invalid_physical_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    record_base = _raw_files(tmp_path)
    source = np.zeros((5_000, 12), dtype=np.float64)
    if case == "non_finite":
        source[40, 2] = np.nan
    elif case == "wrong_shape":
        source = source[:-1]
    _install_valid_reader(monkeypatch, source_order=LEADS, physical=source)
    if case == "no_physical":
        original = adapters.wfdb.rdrecord

        def without_physical(*args: object, **kwargs: object) -> SimpleNamespace:
            result = original(*args, **kwargs)
            result.p_signal = None
            return result

        monkeypatch.setattr(adapters.wfdb, "rdrecord", without_physical)

    with pytest.raises(ExternalECGAdapterError):
        load_wfdb_12lead_signal(record_base)


def test_missing_suffix_and_raw_file_tamper_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_base = _raw_files(tmp_path)
    _install_valid_reader(monkeypatch, source_order=LEADS)

    with pytest.raises(ExternalECGAdapterError, match="suffix-free"):
        load_wfdb_12lead_signal(Path(f"{record_base}.hea"))
    Path(f"{record_base}.dat").unlink()
    with pytest.raises(ExternalECGAdapterError, match="missing"):
        load_wfdb_12lead_signal(record_base)


def test_real_wfdb_round_trip_obeys_the_adapter_contract(tmp_path: Path) -> None:
    samples = np.arange(6_000, dtype=np.float64) / 500.0
    physical = np.stack(
        [0.1 * np.sin((index + 1) * samples) + index * 0.01 for index in range(12)],
        axis=1,
    )
    wfdb.wrsamp(
        record_name="real-record",
        fs=500,
        units=["mV"] * 12,
        sig_name=list(LEADS),
        p_signal=physical,
        fmt=["16"] * 12,
        write_dir=str(tmp_path),
    )

    loaded = load_wfdb_12lead_signal(tmp_path / "real-record")

    assert loaded.signal_mv.shape == (12, 1_000)
    assert loaded.source_duration_seconds == 12.0
    assert loaded.source_lead_names == LEADS
    assert loaded.provenance.resample_window == ("kaiser", 5.0)
    assert np.isfinite(loaded.signal_mv).all()
