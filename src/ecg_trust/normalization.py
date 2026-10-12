"""Validated normalization artifacts independent of waveform loading and fitting.

Inspect or load existing training-derived statistics through this module without
importing the dataset, PyTorch or WFDB. Fitting remains in data.dataset; neither
loading nor validation authorizes access to a held-out cohort.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ecg_trust.constants import LEADS, SUPERCLASSES
from ecg_trust.protocol import TRAIN_FOLDS

_NORMALIZATION_SCHEMA_VERSION = 1
_HEX_DIGITS = frozenset("0123456789abcdef")


class NormalizationValidationError(ValueError):
    """Raised when normalization statistics are invalid or incompatible."""


def _require_non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NormalizationValidationError(f"{field} must be a non-empty string")
    return value


def _snapshot_sequence[Item](values: Sequence[Item]) -> tuple[Item, ...]:
    if isinstance(values, (str, bytes)):
        raise NormalizationValidationError("normalization fields must be sequences of values")
    try:
        return tuple(values)
    except TypeError as error:
        raise NormalizationValidationError("normalization fields must be sequences") from error


def _require_int(value: object, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise NormalizationValidationError(f"{field} must be an integer >= {minimum}")
    return value


def _require_float(value: object, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NormalizationValidationError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise NormalizationValidationError(f"{field} must be {qualifier}")
    return result


def _require_sequence(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise NormalizationValidationError(f"{field} must be a JSON array")
    return cast(list[object], value)


def _parse_string_tuple(value: object, field: str) -> tuple[str, ...]:
    parsed: list[str] = []
    for index, item in enumerate(_require_sequence(value, field)):
        parsed.append(_require_non_empty_string(item, f"{field}[{index}]"))
    return tuple(parsed)


def _parse_int_tuple(value: object, field: str) -> tuple[int, ...]:
    parsed: list[int] = []
    for index, item in enumerate(_require_sequence(value, field)):
        parsed.append(_require_int(item, f"{field}[{index}]"))
    return tuple(parsed)


def _parse_float_tuple(value: object, field: str) -> tuple[float, ...]:
    parsed: list[float] = []
    for index, item in enumerate(_require_sequence(value, field)):
        parsed.append(_require_float(item, f"{field}[{index}]"))
    return tuple(parsed)


def _mapping_value(payload: Mapping[str, object], key: str) -> object:
    if key not in payload:
        raise NormalizationValidationError(f"normalization file is missing {key!r}")
    return payload[key]


@dataclass(frozen=True, slots=True)
class NormalizationProvenance:
    """Evidence describing exactly which data produced normalization statistics."""

    dataset_version: str
    manifest_sha256: str
    training_folds: tuple[int, ...]
    record_count: int
    sample_count: int
    sampling_frequency_hz: float
    samples_per_record: int
    path_column: str
    fold_column: str
    target_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "training_folds", _snapshot_sequence(self.training_folds))
        object.__setattr__(self, "target_columns", _snapshot_sequence(self.target_columns))
        _require_non_empty_string(self.dataset_version, "dataset_version")
        digest = _require_non_empty_string(self.manifest_sha256, "manifest_sha256")
        if len(digest) != 64 or any(character not in _HEX_DIGITS for character in digest):
            raise NormalizationValidationError(
                "manifest_sha256 must be a lowercase hexadecimal SHA-256 digest"
            )
        if not self.training_folds:
            raise NormalizationValidationError("training_folds must not be empty")
        if tuple(sorted(set(self.training_folds))) != self.training_folds:
            raise NormalizationValidationError("training_folds must be unique and sorted")
        for fold in self.training_folds:
            _require_int(fold, "training_folds item")
        if not set(self.training_folds).issubset(TRAIN_FOLDS):
            raise NormalizationValidationError(
                "normalization must be fitted exclusively on training folds 1-7"
            )
        _require_int(self.record_count, "record_count")
        _require_int(self.sample_count, "sample_count")
        _require_float(self.sampling_frequency_hz, "sampling_frequency_hz", positive=True)
        _require_int(self.samples_per_record, "samples_per_record")
        if self.sample_count != self.record_count * self.samples_per_record:
            raise NormalizationValidationError(
                "sample_count must equal record_count * samples_per_record"
            )
        _require_non_empty_string(self.path_column, "path_column")
        _require_non_empty_string(self.fold_column, "fold_column")
        if len(self.target_columns) != len(SUPERCLASSES):
            raise NormalizationValidationError(
                f"target_columns must contain {len(SUPERCLASSES)} entries"
            )
        if len(set(self.target_columns)) != len(self.target_columns):
            raise NormalizationValidationError("target_columns must be unique")
        for target in self.target_columns:
            _require_non_empty_string(target, "target_columns item")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""

        return {
            "dataset_version": self.dataset_version,
            "manifest_sha256": self.manifest_sha256,
            "training_folds": list(self.training_folds),
            "record_count": self.record_count,
            "sample_count": self.sample_count,
            "sampling_frequency_hz": self.sampling_frequency_hz,
            "samples_per_record": self.samples_per_record,
            "path_column": self.path_column,
            "fold_column": self.fold_column,
            "target_columns": list(self.target_columns),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> NormalizationProvenance:
        """Validate and construct provenance loaded from JSON."""

        return cls(
            dataset_version=_require_non_empty_string(
                _mapping_value(payload, "dataset_version"), "dataset_version"
            ),
            manifest_sha256=_require_non_empty_string(
                _mapping_value(payload, "manifest_sha256"), "manifest_sha256"
            ),
            training_folds=_parse_int_tuple(
                _mapping_value(payload, "training_folds"), "training_folds"
            ),
            record_count=_require_int(_mapping_value(payload, "record_count"), "record_count"),
            sample_count=_require_int(_mapping_value(payload, "sample_count"), "sample_count"),
            sampling_frequency_hz=_require_float(
                _mapping_value(payload, "sampling_frequency_hz"),
                "sampling_frequency_hz",
                positive=True,
            ),
            samples_per_record=_require_int(
                _mapping_value(payload, "samples_per_record"), "samples_per_record"
            ),
            path_column=_require_non_empty_string(
                _mapping_value(payload, "path_column"), "path_column"
            ),
            fold_column=_require_non_empty_string(
                _mapping_value(payload, "fold_column"), "fold_column"
            ),
            target_columns=_parse_string_tuple(
                _mapping_value(payload, "target_columns"), "target_columns"
            ),
        )


@dataclass(frozen=True, slots=True)
class NormalizationStats:
    """Per-lead population statistics plus their training-data provenance."""

    mean: tuple[float, ...]
    std: tuple[float, ...]
    leads: tuple[str, ...]
    provenance: NormalizationProvenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", _snapshot_sequence(self.mean))
        object.__setattr__(self, "std", _snapshot_sequence(self.std))
        object.__setattr__(self, "leads", _snapshot_sequence(self.leads))
        if not isinstance(self.provenance, NormalizationProvenance):
            raise NormalizationValidationError("provenance must be NormalizationProvenance")
        if self.leads != LEADS:
            raise NormalizationValidationError(
                f"normalization leads must use canonical order {LEADS!r}"
            )
        if len(self.mean) != len(LEADS) or len(self.std) != len(LEADS):
            raise NormalizationValidationError(
                f"mean and std must each contain {len(LEADS)} values"
            )
        for index, value in enumerate(self.mean):
            _require_float(value, f"mean[{index}]")
        for index, value in enumerate(self.std):
            _require_float(value, f"std[{index}]", positive=True)

    def to_dict(self) -> dict[str, object]:
        """Return the versioned on-disk representation."""

        return {
            "schema_version": _NORMALIZATION_SCHEMA_VERSION,
            "leads": list(self.leads),
            "mean": list(self.mean),
            "std": list(self.std),
            "provenance": self.provenance.to_dict(),
        }

    def save(self, path: str | Path) -> None:
        """Save validated statistics as readable, versioned JSON."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> NormalizationStats:
        """Load statistics and reject malformed or unsupported files."""

        try:
            decoded: object = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise NormalizationValidationError(
                f"could not load normalization statistics from {path!s}: {error}"
            ) from error
        if not isinstance(decoded, dict):
            raise NormalizationValidationError("normalization file must contain a JSON object")
        payload = cast(dict[str, object], decoded)
        schema_version = _require_int(_mapping_value(payload, "schema_version"), "schema_version")
        if schema_version != _NORMALIZATION_SCHEMA_VERSION:
            raise NormalizationValidationError(
                f"unsupported normalization schema_version {schema_version}"
            )
        provenance_value = _mapping_value(payload, "provenance")
        if not isinstance(provenance_value, dict):
            raise NormalizationValidationError("provenance must be a JSON object")
        provenance_payload = cast(dict[str, object], provenance_value)
        return cls(
            mean=_parse_float_tuple(_mapping_value(payload, "mean"), "mean"),
            std=_parse_float_tuple(_mapping_value(payload, "std"), "std"),
            leads=_parse_string_tuple(_mapping_value(payload, "leads"), "leads"),
            provenance=NormalizationProvenance.from_dict(provenance_payload),
        )
