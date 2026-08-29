"""Fail-closed WFDB adapters for the preregistered external OOD v2 cohorts.

The adapters deliberately perform only the frozen, auditable transformation:
physical 12-lead millivolts are reordered into the project lead convention, the
first ten seconds are selected, and an exact rational polyphase resample maps
500 Hz to 100 Hz.  Leads are never synthesized and records are never padded.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import cast

import numpy as np
import numpy.typing as npt
import wfdb  # type: ignore[import-untyped]
from scipy.signal import resample_poly  # type: ignore[import-untyped]

from ecg_trust.constants import LEADS
from ecg_trust.data.manifest import sha256_file

ADAPTER_VERSION = "external-wfdb-12lead-first10s-polyphase-v2"
TARGET_FREQUENCY_HZ = 100
WINDOW_SECONDS = 10
TARGET_SAMPLES = TARGET_FREQUENCY_HZ * WINDOW_SECONDS
PHYSICAL_UNITS = "mV"
RESAMPLE_WINDOW: tuple[str, float] = ("kaiser", 5.0)
RESAMPLE_PADTYPE = "constant"
ADAPTER_PROVENANCE_DOMAIN = b"ecg_trust.ood_v2.adapter_provenance.v2\x00"
SOURCE_LEAD_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "AVR": "aVR",
        "AVL": "aVL",
        "AVF": "aVF",
    }
)


class ExternalECGAdapterError(ValueError):
    """Raised when an external waveform violates the frozen adapter contract."""


def _is_indirect(path: Path) -> bool:
    """Return whether ``path`` is a symlink or Windows junction/reparse link."""

    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(callable(is_junction) and is_junction())
    except OSError as error:
        raise ExternalECGAdapterError("path indirection could not be inspected") from error


def _assert_direct_ancestry(path: Path, *, field: str) -> None:
    current = path.absolute()
    while True:
        if _is_indirect(current):
            raise ExternalECGAdapterError(f"{field} traverses an indirect path")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _require_sha256(value: str, field: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ExternalECGAdapterError(f"{field} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class AdapterProvenance:
    """Hashable evidence for the exact bytes and transform used by an adapter."""

    adapter_version: str
    raw_header_sha256: str
    raw_header_size_bytes: int
    raw_data_sha256: str
    raw_data_size_bytes: int
    source_frequency_hz: float
    source_sample_count: int
    source_duration_seconds: float
    source_lead_names: tuple[str, ...]
    canonical_leads: tuple[str, ...]
    output_leads: tuple[str, ...]
    source_data_file_names: tuple[str, ...]
    raw_physical_units: tuple[str, ...]
    physical_units: str
    window_start_sample: int
    window_source_samples: int
    window_seconds: int
    resample_up: int
    resample_down: int
    resample_window: tuple[str, float]
    resample_padtype: str
    target_frequency_hz: int
    target_samples: int

    def __post_init__(self) -> None:
        if self.adapter_version != ADAPTER_VERSION:
            raise ExternalECGAdapterError("unsupported adapter provenance version")
        _require_sha256(self.raw_header_sha256, "raw_header_sha256")
        _require_sha256(self.raw_data_sha256, "raw_data_sha256")
        for field, value in (
            ("raw_header_size_bytes", self.raw_header_size_bytes),
            ("raw_data_size_bytes", self.raw_data_size_bytes),
            ("source_sample_count", self.source_sample_count),
            ("window_source_samples", self.window_source_samples),
            ("window_seconds", self.window_seconds),
            ("resample_up", self.resample_up),
            ("resample_down", self.resample_down),
            ("target_frequency_hz", self.target_frequency_hz),
            ("target_samples", self.target_samples),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ExternalECGAdapterError(f"{field} must be a positive integer")
        if self.window_start_sample != 0:
            raise ExternalECGAdapterError("only a first-window transform is permitted")
        if not math.isfinite(self.source_frequency_hz) or self.source_frequency_hz <= 0.0:
            raise ExternalECGAdapterError("source_frequency_hz must be finite and positive")
        if not math.isfinite(self.source_duration_seconds) or self.source_duration_seconds < 10.0:
            raise ExternalECGAdapterError("source_duration_seconds must be at least ten seconds")
        if not isinstance(self.source_lead_names, tuple):
            raise ExternalECGAdapterError("source_lead_names must be an immutable tuple")
        if not isinstance(self.canonical_leads, tuple):
            raise ExternalECGAdapterError("canonical_leads must be an immutable tuple")
        expected_canonical = canonicalize_source_lead_names(self.source_lead_names)
        if self.canonical_leads != expected_canonical:
            raise ExternalECGAdapterError(
                "canonical_leads must be the fixed name-only mapping of source_lead_names"
            )
        if self.output_leads != LEADS:
            raise ExternalECGAdapterError("output_leads do not match the project contract")
        if not isinstance(self.source_data_file_names, tuple):
            raise ExternalECGAdapterError(
                "source_data_file_names must be an immutable tuple"
            )
        data_file_names = _exact_text_sequence(
            self.source_data_file_names, "source data file names", len(LEADS)
        )
        if len(set(data_file_names)) != 1:
            raise ExternalECGAdapterError(
                "every source lead must bind the same WFDB data file"
            )
        data_leaf = data_file_names[0]
        if (
            "/" in data_leaf
            or "\\" in data_leaf
            or data_leaf in {".", ".."}
            or not data_leaf.endswith(".dat")
        ):
            raise ExternalECGAdapterError(
                "source data file name must be a local .dat leaf"
            )
        if not isinstance(self.raw_physical_units, tuple):
            raise ExternalECGAdapterError("raw_physical_units must be an immutable tuple")
        raw_units = _exact_text_sequence(
            self.raw_physical_units, "raw physical units", len(LEADS)
        )
        if raw_units != (PHYSICAL_UNITS,) * len(LEADS):
            raise ExternalECGAdapterError(
                "every raw source lead must use the exact physical unit 'mV'"
            )
        if self.physical_units != PHYSICAL_UNITS:
            raise ExternalECGAdapterError("adapter provenance must describe physical mV")
        if self.window_seconds != WINDOW_SECONDS:
            raise ExternalECGAdapterError("adapter window must be exactly ten seconds")
        if self.target_frequency_hz != TARGET_FREQUENCY_HZ:
            raise ExternalECGAdapterError("adapter target frequency must be 100 Hz")
        if self.target_samples != TARGET_SAMPLES:
            raise ExternalECGAdapterError("adapter target must contain exactly 1,000 samples")
        if self.resample_window != RESAMPLE_WINDOW:
            raise ExternalECGAdapterError("adapter must use the frozen Kaiser 5.0 window")
        if self.resample_padtype != RESAMPLE_PADTYPE:
            raise ExternalECGAdapterError("adapter must use constant polyphase padding")
        source_rate = Fraction(str(self.source_frequency_hz))
        target_rate = Fraction(self.target_frequency_hz, 1)
        ratio = target_rate / source_rate
        if (self.resample_up, self.resample_down) != (ratio.numerator, ratio.denominator):
            raise ExternalECGAdapterError("resample ratio does not match the exact source rate")
        if self.window_source_samples * self.resample_up != (
            self.target_samples * self.resample_down
        ):
            raise ExternalECGAdapterError("source and target window sizes are inconsistent")

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON-compatible provenance body."""

        return {
            "adapter_version": self.adapter_version,
            "canonical_leads": list(self.canonical_leads),
            "output_leads": list(self.output_leads),
            "physical_units": self.physical_units,
            "raw_data_sha256": self.raw_data_sha256,
            "raw_data_size_bytes": self.raw_data_size_bytes,
            "raw_header_sha256": self.raw_header_sha256,
            "raw_header_size_bytes": self.raw_header_size_bytes,
            "raw_physical_units": list(self.raw_physical_units),
            "resample_down": self.resample_down,
            "resample_padtype": self.resample_padtype,
            "resample_up": self.resample_up,
            "resample_window": list(self.resample_window),
            "source_duration_seconds": self.source_duration_seconds,
            "source_data_file_names": list(self.source_data_file_names),
            "source_frequency_hz": self.source_frequency_hz,
            "source_lead_names": list(self.source_lead_names),
            "source_sample_count": self.source_sample_count,
            "target_frequency_hz": self.target_frequency_hz,
            "target_samples": self.target_samples,
            "window_seconds": self.window_seconds,
            "window_source_samples": self.window_source_samples,
            "window_start_sample": self.window_start_sample,
        }

    @property
    def sha256(self) -> str:
        """Return a domain-separated identity for this immutable provenance."""

        digest = hashlib.sha256(
            ADAPTER_PROVENANCE_DOMAIN + _canonical_json_bytes(self.to_dict())
        ).hexdigest()
        return f"sha256:{digest}"


@dataclass(frozen=True, slots=True)
class CanonicalExternalSignal:
    """One finite, canonical ``[12, 1000]`` physical-mV external ECG."""

    signal_mv: npt.NDArray[np.float32]
    provenance: AdapterProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, AdapterProvenance):
            raise TypeError("provenance must be AdapterProvenance")
        signal = np.asarray(self.signal_mv)
        if signal.shape != (len(LEADS), TARGET_SAMPLES):
            raise ExternalECGAdapterError(
                f"canonical signal must have shape {(len(LEADS), TARGET_SAMPLES)!r}"
            )
        if signal.dtype != np.float32:
            raise ExternalECGAdapterError("canonical signal must use float32")
        if not np.isfinite(signal).all():
            raise ExternalECGAdapterError("canonical signal contains non-finite values")
        immutable = np.ascontiguousarray(signal).copy()
        immutable.flags.writeable = False
        object.__setattr__(self, "signal_mv", immutable)

    @property
    def source_frequency_hz(self) -> float:
        return self.provenance.source_frequency_hz

    @property
    def source_duration_seconds(self) -> float:
        return self.provenance.source_duration_seconds

    @property
    def source_lead_names(self) -> tuple[str, ...]:
        return self.provenance.source_lead_names

    @property
    def canonical_leads(self) -> tuple[str, ...]:
        """Canonicalized source names in unchanged source-column order."""

        return self.provenance.canonical_leads

    @property
    def output_leads(self) -> tuple[str, ...]:
        """Lead order of ``signal_mv`` after the explicit column permutation."""

        return self.provenance.output_leads

    @property
    def source_data_file_names(self) -> tuple[str, ...]:
        return self.provenance.source_data_file_names

    @property
    def raw_physical_units(self) -> tuple[str, ...]:
        return self.provenance.raw_physical_units

    @property
    def adapter_version(self) -> str:
        return self.provenance.adapter_version

    @property
    def provenance_sha256(self) -> str:
        return self.provenance.sha256


@dataclass(frozen=True, slots=True)
class _WFDBMetadata:
    frequency_hz: float
    sample_count: int
    duration_seconds: float
    source_lead_names: tuple[str, ...]
    canonical_leads: tuple[str, ...]
    source_data_file_names: tuple[str, ...]
    raw_physical_units: tuple[str, ...]


def _record_files(record_base: Path) -> tuple[Path, Path]:
    if record_base.suffix.casefold() in {".hea", ".dat"}:
        raise ExternalECGAdapterError("record_base must be suffix-free")
    header_path = Path(f"{record_base}.hea")
    data_path = Path(f"{record_base}.dat")
    _assert_direct_ancestry(record_base.parent, field="record directory")
    try:
        direct_parent = record_base.parent.resolve(strict=True)
    except OSError as error:
        raise ExternalECGAdapterError("record directory is unavailable") from error
    for kind, path in (("header", header_path), ("data", data_path)):
        if _is_indirect(path):
            raise ExternalECGAdapterError(
                f"raw {kind} file must not be a symlink or junction"
            )
        if not path.is_file():
            raise ExternalECGAdapterError(f"raw {kind} file is missing: {path!s}")
        try:
            path.resolve(strict=True).relative_to(direct_parent)
        except (OSError, ValueError) as error:
            raise ExternalECGAdapterError(
                f"raw {kind} file escapes its direct record directory"
            ) from error
    return header_path, data_path


def _positive_frequency(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ExternalECGAdapterError(f"{field} must be finite and positive")
    try:
        parsed = float(cast(float | int | str, value))
    except (TypeError, ValueError) as error:
        raise ExternalECGAdapterError(f"{field} must be finite and positive") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ExternalECGAdapterError(f"{field} must be finite and positive")
    return parsed


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ExternalECGAdapterError(f"{field} must be a positive integer")
    try:
        parsed = int(cast(int | str, value))
    except (TypeError, ValueError, OverflowError) as error:
        raise ExternalECGAdapterError(f"{field} must be a positive integer") from error
    if parsed < 1 or parsed != value:
        raise ExternalECGAdapterError(f"{field} must be a positive integer")
    return parsed


def _text_sequence(value: object, field: str, expected_length: int) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ExternalECGAdapterError(f"{field} must contain {expected_length} text values")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ExternalECGAdapterError(f"{field} must contain non-empty text values")
        result.append(item.strip())
    if len(result) != expected_length:
        raise ExternalECGAdapterError(f"{field} must contain {expected_length} values")
    return tuple(result)


def _exact_text_sequence(
    value: object, field: str, expected_length: int
) -> tuple[str, ...]:
    """Return WFDB text metadata without rewriting any source characters."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ExternalECGAdapterError(
            f"{field} must contain {expected_length} text values"
        )
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ExternalECGAdapterError(f"{field} must contain non-empty text values")
        if any(character in item for character in ("\x00", "\r", "\n")):
            raise ExternalECGAdapterError(f"{field} contains a forbidden character")
        result.append(item)
    if len(result) != expected_length:
        raise ExternalECGAdapterError(f"{field} must contain {expected_length} values")
    return tuple(result)


def canonicalize_source_lead_names(
    raw_names: Sequence[str],
    *,
    allowed_aliases: Mapping[str, str] = SOURCE_LEAD_ALIASES,
) -> tuple[str, ...]:
    """Apply an explicit subset of the fixed aliases, preserving source order."""

    if not isinstance(allowed_aliases, Mapping) or any(
        SOURCE_LEAD_ALIASES.get(source) != target
        for source, target in allowed_aliases.items()
    ):
        raise ExternalECGAdapterError("allowed_aliases exceeds the fixed alias contract")
    exact_names = _exact_text_sequence(raw_names, "lead names", len(LEADS))
    canonical: list[str] = []
    for raw_name in exact_names:
        canonical_name: str | None = (
            raw_name if raw_name in LEADS else allowed_aliases.get(raw_name)
        )
        if canonical_name is None:
            raise ExternalECGAdapterError(f"unsupported ECG lead name {raw_name!r}")
        canonical.append(canonical_name)
    if len(set(canonical)) != len(LEADS):
        duplicate = next(name for name in canonical if canonical.count(name) > 1)
        raise ExternalECGAdapterError(f"duplicate ECG lead {duplicate!r}")
    missing = [lead for lead in LEADS if lead not in canonical]
    if missing or len(canonical) != len(LEADS):
        raise ExternalECGAdapterError(
            f"record must contain exactly the canonical 12 leads; missing={missing!r}"
        )
    return tuple(canonical)


def _canonical_lead_positions(canonical_names: tuple[str, ...]) -> tuple[int, ...]:
    position_by_lead: dict[str, int] = {}
    for position, canonical_name in enumerate(canonical_names):
        if canonical_name in position_by_lead:
            raise ExternalECGAdapterError(f"duplicate ECG lead {canonical_name!r}")
        position_by_lead[canonical_name] = position
    missing = [lead for lead in LEADS if lead not in position_by_lead]
    if missing or len(position_by_lead) != len(LEADS):
        raise ExternalECGAdapterError(
            f"record must contain exactly the canonical 12 leads; missing={missing!r}"
        )
    return tuple(position_by_lead[lead] for lead in LEADS)


def _metadata_from_wfdb(
    value: object,
    *,
    context: str,
    expected_data_file_name: str,
    allowed_lead_aliases: Mapping[str, str],
) -> _WFDBMetadata:
    frequency_hz = _positive_frequency(getattr(value, "fs", None), f"{context} sampling rate")
    sample_count = _positive_integer(getattr(value, "sig_len", None), f"{context} sample count")
    lead_count = _positive_integer(getattr(value, "n_sig", None), f"{context} lead count")
    if lead_count != len(LEADS):
        raise ExternalECGAdapterError(f"{context} must contain exactly 12 leads")
    source_lead_names = _exact_text_sequence(
        getattr(value, "sig_name", None), "lead names", lead_count
    )
    canonical_leads = canonicalize_source_lead_names(
        source_lead_names,
        allowed_aliases=allowed_lead_aliases,
    )
    _canonical_lead_positions(canonical_leads)
    source_data_file_names = _exact_text_sequence(
        getattr(value, "file_name", None), "source data file names", lead_count
    )
    if any(name != expected_data_file_name for name in source_data_file_names):
        raise ExternalECGAdapterError(
            f"{context} must bind every lead to {expected_data_file_name!r}"
        )
    raw_physical_units = _exact_text_sequence(
        getattr(value, "units", None), "raw physical units", lead_count
    )
    if raw_physical_units != (PHYSICAL_UNITS,) * len(LEADS):
        raise ExternalECGAdapterError(
            "every source lead must use the exact physical unit 'mV'"
        )
    return _WFDBMetadata(
        frequency_hz=frequency_hz,
        sample_count=sample_count,
        duration_seconds=sample_count / frequency_hz,
        source_lead_names=source_lead_names,
        canonical_leads=canonical_leads,
        source_data_file_names=source_data_file_names,
        raw_physical_units=raw_physical_units,
    )


def _matching_read_metadata(
    record: object,
    header: _WFDBMetadata,
    *,
    expected_window_samples: int,
    allowed_lead_aliases: Mapping[str, str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
    read_rate = _positive_frequency(getattr(record, "fs", None), "loaded sampling rate")
    if not math.isclose(read_rate, header.frequency_hz, rel_tol=0.0, abs_tol=0.0):
        raise ExternalECGAdapterError("loaded sampling rate differs from the raw header")
    lead_count = _positive_integer(getattr(record, "n_sig", None), "loaded lead count")
    if lead_count != len(LEADS):
        raise ExternalECGAdapterError("loaded record must contain exactly 12 leads")
    read_names = _exact_text_sequence(
        getattr(record, "sig_name", None), "lead names", lead_count
    )
    if read_names != header.source_lead_names:
        raise ExternalECGAdapterError("loaded lead names differ from the raw header")
    read_canonical = canonicalize_source_lead_names(
        read_names,
        allowed_aliases=allowed_lead_aliases,
    )
    if read_canonical != header.canonical_leads:
        raise ExternalECGAdapterError(
            "loaded canonical lead names differ from the raw-header canonical mapping"
        )
    read_data_file_names = _exact_text_sequence(
        getattr(record, "file_name", None), "source data file names", lead_count
    )
    if read_data_file_names != header.source_data_file_names:
        raise ExternalECGAdapterError(
            "loaded data-file bindings differ from the raw header"
        )
    read_units = _exact_text_sequence(
        getattr(record, "units", None), "raw physical units", lead_count
    )
    if read_units != header.raw_physical_units:
        raise ExternalECGAdapterError("loaded physical units differ from the raw header")
    read_samples = _positive_integer(
        getattr(record, "sig_len", expected_window_samples), "loaded sample count"
    )
    if read_samples != expected_window_samples:
        raise ExternalECGAdapterError("WFDB did not return the exact first ten-second window")
    return read_names, read_canonical, _canonical_lead_positions(read_canonical)


def load_wfdb_12lead_signal(
    record_base: Path,
    expected_source_hz: float = 500.0,
    *,
    allowed_lead_aliases: Mapping[str, str] = SOURCE_LEAD_ALIASES,
) -> CanonicalExternalSignal:
    """Load and canonicalize one strict external WFDB record.

    ``record_base`` is the suffix-free path shared by the ``.hea`` and ``.dat``
    files.  Header duration is checked before the data file is opened.  WFDB is
    instructed to return physical values and exactly samples ``[0, 5000)`` for
    the frozen 500 Hz protocol.
    """

    if not isinstance(record_base, Path):
        raise TypeError("record_base must be a pathlib.Path")
    if not isinstance(allowed_lead_aliases, Mapping):
        raise TypeError("allowed_lead_aliases must be a mapping")
    expected_rate = _positive_frequency(expected_source_hz, "expected_source_hz")
    expected_fraction = Fraction(str(expected_rate))
    if expected_fraction * WINDOW_SECONDS != int(expected_fraction * WINDOW_SECONDS):
        raise ExternalECGAdapterError("ten seconds must contain an exact integer sample count")
    source_window_samples = int(expected_fraction * WINDOW_SECONDS)
    target_fraction = Fraction(TARGET_FREQUENCY_HZ, 1)
    ratio = target_fraction / expected_fraction
    if source_window_samples * ratio.numerator != TARGET_SAMPLES * ratio.denominator:
        raise ExternalECGAdapterError("source rate cannot produce the exact frozen target window")

    header_path, data_path = _record_files(record_base)
    before_hashes = (sha256_file(header_path), sha256_file(data_path))
    before_sizes = (header_path.stat().st_size, data_path.stat().st_size)
    try:
        raw_header = wfdb.rdheader(str(record_base))
    except Exception as error:
        raise ExternalECGAdapterError(
            f"could not read WFDB header {header_path!s}: {error}"
        ) from error
    header = _metadata_from_wfdb(
        raw_header,
        context="raw header",
        expected_data_file_name=data_path.name,
        allowed_lead_aliases=allowed_lead_aliases,
    )
    if not math.isclose(header.frequency_hz, expected_rate, rel_tol=0.0, abs_tol=0.0):
        raise ExternalECGAdapterError(
            f"source sampling rate is {header.frequency_hz!r} Hz; expected {expected_rate!r} Hz"
        )
    if header.sample_count < source_window_samples:
        raise ExternalECGAdapterError("source record is shorter than ten seconds")

    try:
        record = wfdb.rdrecord(
            str(record_base),
            sampfrom=0,
            sampto=source_window_samples,
            physical=True,
            return_res=64,
        )
    except Exception as error:
        raise ExternalECGAdapterError(
            f"could not read WFDB record {record_base!s}: {error}"
        ) from error
    _, _, order = _matching_read_metadata(
        record,
        header,
        expected_window_samples=source_window_samples,
        allowed_lead_aliases=allowed_lead_aliases,
    )

    physical = getattr(record, "p_signal", None)
    if physical is None:
        raise ExternalECGAdapterError("WFDB record does not expose a physical signal")
    raw_signal = np.asarray(physical)
    if raw_signal.ndim != 2 or raw_signal.shape != (source_window_samples, len(LEADS)):
        raise ExternalECGAdapterError(
            f"source window must have shape {(source_window_samples, len(LEADS))!r}"
        )
    if np.issubdtype(raw_signal.dtype, np.bool_) or not np.issubdtype(
        raw_signal.dtype, np.number
    ) or np.issubdtype(raw_signal.dtype, np.complexfloating):
        raise ExternalECGAdapterError("physical signal must contain real numeric values")
    try:
        native = np.asarray(raw_signal[:, list(order)].T, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ExternalECGAdapterError("physical signal could not be represented safely") from error
    if not np.isfinite(native).all():
        raise ExternalECGAdapterError("source physical signal contains non-finite values")

    resampled = resample_poly(
        native,
        up=ratio.numerator,
        down=ratio.denominator,
        axis=1,
        window=RESAMPLE_WINDOW,
        padtype=RESAMPLE_PADTYPE,
    )
    signal = np.ascontiguousarray(resampled, dtype=np.float32)
    if signal.shape != (len(LEADS), TARGET_SAMPLES):
        raise ExternalECGAdapterError("polyphase resampling produced an unexpected shape")
    if not np.isfinite(signal).all():
        raise ExternalECGAdapterError("signal became non-finite during polyphase resampling")

    after_hashes = (sha256_file(header_path), sha256_file(data_path))
    after_sizes = (header_path.stat().st_size, data_path.stat().st_size)
    rebound_header, rebound_data = _record_files(record_base)
    if (
        rebound_header != header_path
        or rebound_data != data_path
        or after_hashes != before_hashes
        or after_sizes != before_sizes
    ):
        raise ExternalECGAdapterError("raw WFDB files changed while the record was being loaded")

    provenance = AdapterProvenance(
        adapter_version=ADAPTER_VERSION,
        raw_header_sha256=before_hashes[0],
        raw_header_size_bytes=before_sizes[0],
        raw_data_sha256=before_hashes[1],
        raw_data_size_bytes=before_sizes[1],
        source_frequency_hz=header.frequency_hz,
        source_sample_count=header.sample_count,
        source_duration_seconds=header.duration_seconds,
        source_lead_names=header.source_lead_names,
        canonical_leads=header.canonical_leads,
        output_leads=LEADS,
        source_data_file_names=header.source_data_file_names,
        raw_physical_units=header.raw_physical_units,
        physical_units=PHYSICAL_UNITS,
        window_start_sample=0,
        window_source_samples=source_window_samples,
        window_seconds=WINDOW_SECONDS,
        resample_up=ratio.numerator,
        resample_down=ratio.denominator,
        resample_window=RESAMPLE_WINDOW,
        resample_padtype=RESAMPLE_PADTYPE,
        target_frequency_hz=TARGET_FREQUENCY_HZ,
        target_samples=TARGET_SAMPLES,
    )
    return CanonicalExternalSignal(signal_mv=signal, provenance=provenance)


def load_challenge_2011_signal(record_base: Path) -> CanonicalExternalSignal:
    """Load one PhysioNet Challenge 2011 Set A record under the frozen contract."""

    return load_wfdb_12lead_signal(
        record_base,
        expected_source_hz=500.0,
        allowed_lead_aliases={},
    )


def load_zzu_pediatric_signal(record_base: Path) -> CanonicalExternalSignal:
    """Load one ZZU pediatric WFDB record under the frozen contract."""

    return load_wfdb_12lead_signal(
        record_base,
        expected_source_hz=500.0,
        allowed_lead_aliases=SOURCE_LEAD_ALIASES,
    )


__all__ = [
    "ADAPTER_PROVENANCE_DOMAIN",
    "ADAPTER_VERSION",
    "PHYSICAL_UNITS",
    "RESAMPLE_PADTYPE",
    "RESAMPLE_WINDOW",
    "SOURCE_LEAD_ALIASES",
    "TARGET_FREQUENCY_HZ",
    "TARGET_SAMPLES",
    "WINDOW_SECONDS",
    "AdapterProvenance",
    "CanonicalExternalSignal",
    "ExternalECGAdapterError",
    "canonicalize_source_lead_names",
    "load_challenge_2011_signal",
    "load_wfdb_12lead_signal",
    "load_zzu_pediatric_signal",
]
