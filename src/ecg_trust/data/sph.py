"""Strict SPH adapter for a separately governed external-transport study.

This module is deliberately independent of the PTB-XL loader.  It implements
the public SPH source contract described by Liu et al. (2022): one ``ecg`` HDF5
dataset per record, native shape ``[12, L]``, 500 Hz sampling, physical values
in mV, and canonical lead order ``I, II, III, aVR, aVL, aVF, V1-V6``.

Only exact ten-second records with at least one directly mapped target belong
to the primary transport cohort.  Unmapped source statements never establish
an all-negative five-target reference.  Ambiguous primary codes may coexist
with direct targets, but they are preserved as provenance and never mapped.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import cast

import h5py  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
import torch
from scipy.signal import resample_poly  # type: ignore[import-untyped]
from torch.utils.data import Dataset

from ecg_trust.constants import LEADS, SUPERCLASSES, TARGET_COLUMNS

SPH_NATIVE_FREQUENCY_HZ = 500
SPH_TARGET_FREQUENCY_HZ = 100
SPH_NATIVE_SAMPLES = 5_000
SPH_TARGET_SAMPLES = 1_000
SPH_HDF5_DATASET_KEY = "ecg"
SPH_SIGNAL_UNIT = "mV"

SPH_SUPERCLASS_CODES: Mapping[str, frozenset[int]] = MappingProxyType(
    {
        "NORM": frozenset({1}),
        "MI": frozenset({160, 161, 165, 166}),
        "STTC": frozenset({145, 146, 147, 148}),
        "CD": frozenset({83, 84, 85, 86, 87, 88, 101, 102, 104, 105, 106, 108}),
        "HYP": frozenset({140, 142, 143}),
    }
)
SPH_AMBIGUOUS_PRIMARY_CODES = frozenset({80, 81, 82, 152, 153, 155})
SPH_DIRECT_PRIMARY_CODES = frozenset(
    code for codes in SPH_SUPERCLASS_CODES.values() for code in codes
)

_OFFICIAL_METADATA_COLUMNS = (
    "ECG_ID",
    "AHA_Code",
    "Patient_ID",
    "Age",
    "Sex",
    "N",
    "Date",
)
_OFFICIAL_CODE_COLUMNS = ("Category", "Code", "Description")
_CANONICAL_MANIFEST_COLUMNS = (
    "ecg_id",
    "patient_id",
    "record_path",
    "age",
    "sex",
    "native_samples",
    "acquisition_date",
    "raw_aha_codes",
    "primary_codes",
    "modifier_codes",
    "ambiguous_primary_codes",
    "has_ambiguous_primary",
    *TARGET_COLUMNS,
    "mapped_target_count",
    "mapping_status",
    "norm_abnormal_conflict",
)
_DECIMAL_PATTERN = re.compile(r"[0-9]+")
_ECG_ID_PATTERN = re.compile(r"A[0-9]{5}")
_PATIENT_ID_PATTERN = re.compile(r"S[0-9]{5}")


class SPHMetadataValidationError(ValueError):
    """Raised when official SPH metadata or AHA codes violate the contract."""


class SPHRecordValidationError(ValueError):
    """Raised when an SPH HDF5 waveform violates the signal contract."""


@dataclass(frozen=True, slots=True)
class AHACodeDefinition:
    """One row of the official SPH ``code.csv`` dictionary."""

    category: str
    code: int
    description: str


@dataclass(frozen=True, slots=True)
class AHAStatement:
    """One primary AHA statement and its optional, order-independent modifiers."""

    primary_code: int
    modifier_codes: tuple[int, ...]


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SPHMetadataValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _parse_decimal(value: object, field: str) -> int:
    text = _require_text(value, field)
    if _DECIMAL_PATTERN.fullmatch(text) is None:
        raise SPHMetadataValidationError(f"{field} must be a positive decimal integer")
    parsed = int(text)
    if parsed < 1:
        raise SPHMetadataValidationError(f"{field} must be a positive decimal integer")
    return parsed


def _read_strict_csv(path: str | Path, expected_columns: Sequence[str]) -> pd.DataFrame:
    source = Path(path)
    try:
        frame = pd.read_csv(
            source,
            dtype=str,
            encoding="utf-8-sig",
            keep_default_na=False,
        )
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise SPHMetadataValidationError(f"could not read {source!s}: {error}") from error
    actual_columns = tuple(str(column) for column in frame.columns)
    expected = tuple(expected_columns)
    if actual_columns != expected:
        raise SPHMetadataValidationError(
            f"{source.name} columns must be exactly {expected!r}; got {actual_columns!r}"
        )
    if frame.empty:
        raise SPHMetadataValidationError(f"{source.name} must contain at least one row")
    return frame


def read_sph_code_dictionary(path: str | Path) -> dict[int, AHACodeDefinition]:
    """Read and strictly validate the official SPH ``code.csv`` dictionary."""

    frame = _read_strict_csv(path, _OFFICIAL_CODE_COLUMNS)
    definitions: dict[int, AHACodeDefinition] = {}
    for row_number, values in enumerate(frame.itertuples(index=False, name=None), start=2):
        category = _require_text(values[0], f"code.csv row {row_number} Category")
        code = _parse_decimal(values[1], f"code.csv row {row_number} Code")
        description = _require_text(values[2], f"code.csv row {row_number} Description")
        if code in definitions:
            raise SPHMetadataValidationError(f"code.csv contains duplicate code {code}")
        is_modifier = 200 <= code < 500
        if is_modifier and category != "Modifier":
            raise SPHMetadataValidationError(f"code.csv code {code} must use category 'Modifier'")
        if not is_modifier and re.fullmatch(r"[A-N]", category) is None:
            raise SPHMetadataValidationError(
                f"code.csv primary code {code} has invalid category {category!r}"
            )
        definitions[code] = AHACodeDefinition(
            category=category,
            code=code,
            description=description,
        )

    missing_contract_codes = sorted(
        (SPH_DIRECT_PRIMARY_CODES | SPH_AMBIGUOUS_PRIMARY_CODES) - definitions.keys()
    )
    if missing_contract_codes:
        raise SPHMetadataValidationError(
            f"code.csv is missing conservative-map codes {missing_contract_codes!r}"
        )
    return definitions


def parse_sph_aha_codes(
    value: object,
    *,
    known_codes: Collection[int] | None = None,
) -> tuple[AHAStatement, ...]:
    """Parse the official semicolon/plus AHA encoding without substring matches.

    Semicolons separate diagnostic statements.  Plus signs join one primary
    code with zero or more modifiers.  The order within a composite statement
    is not assumed: official modifiers occupy the range ``[200, 500)``.
    """

    encoded = _require_text(value, "AHA_Code")
    raw_statements = encoded.split(";")
    if any(not statement.strip() for statement in raw_statements):
        raise SPHMetadataValidationError("AHA_Code contains an empty statement")

    parsed_statements: list[AHAStatement] = []
    for statement_index, raw_statement in enumerate(raw_statements):
        raw_tokens = raw_statement.split("+")
        if any(not token.strip() for token in raw_tokens):
            raise SPHMetadataValidationError(
                f"AHA_Code statement {statement_index} contains an empty code"
            )
        codes = tuple(
            _parse_decimal(token, f"AHA_Code statement {statement_index} code")
            for token in raw_tokens
        )
        if len(set(codes)) != len(codes):
            raise SPHMetadataValidationError(
                f"AHA_Code statement {statement_index} contains a duplicate code"
            )
        if known_codes is not None:
            unknown = sorted(set(codes) - set(known_codes))
            if unknown:
                raise SPHMetadataValidationError(
                    f"AHA_Code statement {statement_index} contains unknown codes {unknown!r}"
                )
        primary_codes = tuple(code for code in codes if not 200 <= code < 500)
        modifier_codes = tuple(code for code in codes if 200 <= code < 500)
        if len(primary_codes) != 1:
            raise SPHMetadataValidationError(
                f"AHA_Code statement {statement_index} must contain exactly one primary code"
            )
        parsed_statements.append(
            AHAStatement(
                primary_code=primary_codes[0],
                modifier_codes=modifier_codes,
            )
        )
    return tuple(parsed_statements)


def map_sph_superclasses(statements: Sequence[AHAStatement]) -> tuple[int, ...]:
    """Map exact primary AHA codes to the frozen PTB-XL superclass order."""

    if not statements:
        raise SPHMetadataValidationError("at least one AHA statement is required")
    primary_codes = {statement.primary_code for statement in statements}
    return tuple(
        int(bool(primary_codes & SPH_SUPERCLASS_CODES[superclass])) for superclass in SUPERCLASSES
    )


def _validated_iso_date(value: object, field: str) -> str:
    text = _require_text(value, field)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as error:
        raise SPHMetadataValidationError(f"{field} must be a valid ISO date") from error
    if parsed.isoformat() != text:
        raise SPHMetadataValidationError(f"{field} must use YYYY-MM-DD format")
    return text


def _mapping_status(targets: tuple[int, ...], ambiguous_codes: tuple[int, ...]) -> str:
    if any(targets):
        return "mapped_with_ambiguous" if ambiguous_codes else "mapped"
    return "ambiguous_only" if ambiguous_codes else "unmapped"


def build_sph_transport_manifest(
    metadata_path: str | Path,
    code_dictionary_path: str | Path,
) -> pd.DataFrame:
    """Build a validated canonical manifest from official SPH CSV files.

    The returned manifest contains every valid source row.  Use
    :func:`select_sph_transport_records` to apply the frozen primary-cohort
    rule instead of treating wholly unmapped rows as all-negative examples.
    """

    definitions = read_sph_code_dictionary(code_dictionary_path)
    frame = _read_strict_csv(metadata_path, _OFFICIAL_METADATA_COLUMNS)
    rows: list[dict[str, object]] = []
    seen_ecg_ids: set[str] = set()

    for row_number, values in enumerate(frame.itertuples(index=False, name=None), start=2):
        ecg_id = _require_text(values[0], f"metadata.csv row {row_number} ECG_ID")
        if _ECG_ID_PATTERN.fullmatch(ecg_id) is None:
            raise SPHMetadataValidationError(
                f"metadata.csv row {row_number} ECG_ID must match A#####"
            )
        if ecg_id in seen_ecg_ids:
            raise SPHMetadataValidationError(f"metadata.csv contains duplicate ECG_ID {ecg_id!r}")
        seen_ecg_ids.add(ecg_id)

        raw_aha_codes = _require_text(values[1], f"metadata.csv row {row_number} AHA_Code")
        statements = parse_sph_aha_codes(raw_aha_codes, known_codes=definitions.keys())
        patient_id = _require_text(values[2], f"metadata.csv row {row_number} Patient_ID")
        if _PATIENT_ID_PATTERN.fullmatch(patient_id) is None:
            raise SPHMetadataValidationError(
                f"metadata.csv row {row_number} Patient_ID must match S#####"
            )
        age = _parse_decimal(values[3], f"metadata.csv row {row_number} Age")
        if not 18 <= age <= 100:
            raise SPHMetadataValidationError(
                f"metadata.csv row {row_number} Age must be between 18 and 100"
            )
        sex = _require_text(values[4], f"metadata.csv row {row_number} Sex")
        if sex not in {"M", "F"}:
            raise SPHMetadataValidationError(
                f"metadata.csv row {row_number} Sex must be 'M' or 'F'"
            )
        native_samples = _parse_decimal(values[5], f"metadata.csv row {row_number} N")
        acquisition_date = _validated_iso_date(values[6], f"metadata.csv row {row_number} Date")

        primary_codes = tuple(statement.primary_code for statement in statements)
        modifier_codes = tuple(
            modifier for statement in statements for modifier in statement.modifier_codes
        )
        ambiguous_codes = tuple(sorted(set(primary_codes) & SPH_AMBIGUOUS_PRIMARY_CODES))
        targets = map_sph_superclasses(statements)
        mapped_target_count = sum(targets)
        norm_abnormal_conflict = bool(targets[0] and any(targets[1:]))
        row: dict[str, object] = {
            "ecg_id": ecg_id,
            "patient_id": patient_id,
            "record_path": f"{ecg_id}.h5",
            "age": age,
            "sex": sex,
            "native_samples": native_samples,
            "acquisition_date": acquisition_date,
            "raw_aha_codes": raw_aha_codes,
            "primary_codes": primary_codes,
            "modifier_codes": modifier_codes,
            "ambiguous_primary_codes": ambiguous_codes,
            "has_ambiguous_primary": bool(ambiguous_codes),
            "mapped_target_count": mapped_target_count,
            "mapping_status": _mapping_status(targets, ambiguous_codes),
            "norm_abnormal_conflict": norm_abnormal_conflict,
        }
        row.update(dict(zip(TARGET_COLUMNS, targets, strict=True)))
        rows.append(row)

    manifest = pd.DataFrame(rows, columns=list(_CANONICAL_MANIFEST_COLUMNS))
    manifest = manifest.sort_values("ecg_id", kind="stable").reset_index(drop=True)
    for target_column in TARGET_COLUMNS:
        manifest[target_column] = manifest[target_column].astype(np.int8)
    manifest["mapped_target_count"] = manifest["mapped_target_count"].astype(np.int8)
    return manifest


def _require_manifest_columns(manifest: pd.DataFrame) -> None:
    if not isinstance(manifest, pd.DataFrame):
        raise TypeError("manifest must be a pandas DataFrame")
    missing = set(_CANONICAL_MANIFEST_COLUMNS) - set(manifest.columns)
    if missing:
        raise SPHMetadataValidationError(
            f"SPH manifest is missing required columns {sorted(missing)!r}"
        )


def select_sph_transport_records(
    manifest: pd.DataFrame,
    *,
    exclude_ambiguous: bool = False,
    exclude_norm_conflicts: bool = False,
) -> pd.DataFrame:
    """Select the frozen primary SPH external-transport cohort.

    Primary eligibility is exact ``N=5000`` and at least one direct target.
    Ambiguous co-statements and exact-token NORM/abnormal conflicts are retained
    by default, with explicit opt-in filters for preregistered sensitivities.
    """

    _require_manifest_columns(manifest)
    if not isinstance(exclude_ambiguous, bool):
        raise TypeError("exclude_ambiguous must be a boolean")
    if not isinstance(exclude_norm_conflicts, bool):
        raise TypeError("exclude_norm_conflicts must be a boolean")
    selected = manifest.loc[
        (manifest["native_samples"] == SPH_NATIVE_SAMPLES) & (manifest["mapped_target_count"] >= 1)
    ].copy()
    if exclude_ambiguous:
        selected = selected.loc[~selected["has_ambiguous_primary"].astype(bool)].copy()
    if exclude_norm_conflicts:
        selected = selected.loc[~selected["norm_abnormal_conflict"].astype(bool)].copy()
    return selected.sort_values("ecg_id", kind="stable").reset_index(drop=True)


def select_sph_exact_10s_records(manifest: pd.DataFrame) -> pd.DataFrame:
    """Select all exact-ten-second rows for inference-only broad sensitivity.

    This cohort includes wholly unmapped rows.  Their all-zero vectors mean
    "no direct mapped positive" rather than verified absence of all five PTB-XL
    superclasses, so they must not be scored as primary reference negatives.
    """

    _require_manifest_columns(manifest)
    selected = manifest.loc[manifest["native_samples"] == SPH_NATIVE_SAMPLES].copy()
    return selected.sort_values("ecg_id", kind="stable").reset_index(drop=True)


def _validated_record_reference(value: object) -> str:
    reference = _require_text(value, "record_path").replace("\\", "/")
    posix = PurePosixPath(reference)
    windows = PureWindowsPath(reference)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise SPHMetadataValidationError("record_path must be relative")
    if any(part in {"", ".", ".."} for part in posix.parts):
        raise SPHMetadataValidationError("record_path must not contain traversal segments")
    if posix.suffix.lower() != ".h5":
        raise SPHMetadataValidationError("record_path must end in .h5")
    return posix.as_posix()


def load_sph_signal(path: str | Path) -> npt.NDArray[np.float32]:
    """Load official SPH mV data and resample 500 Hz to 100 Hz anti-aliased.

    No amplitude scaling or normalization is applied.  ``resample_poly(1, 5)``
    supplies the anti-aliasing low-pass filter before decimation.  The result is
    a finite, contiguous ``float32`` array in shape ``[12, 1000]``.
    """

    source = Path(path)
    if source.suffix.lower() != ".h5":
        raise SPHRecordValidationError(f"record {source!s} must use the .h5 suffix")
    try:
        with h5py.File(source, "r") as handle:
            if SPH_HDF5_DATASET_KEY not in handle:
                raise SPHRecordValidationError(
                    f"record {source!s} is missing root dataset {SPH_HDF5_DATASET_KEY!r}"
                )
            node = handle[SPH_HDF5_DATASET_KEY]
            if not isinstance(node, h5py.Dataset):
                raise SPHRecordValidationError(
                    f"record {source!s} key {SPH_HDF5_DATASET_KEY!r} must be a dataset"
                )
            if node.shape != (len(LEADS), SPH_NATIVE_SAMPLES):
                raise SPHRecordValidationError(
                    f"record {source!s} has shape {node.shape!r}; "
                    f"expected {(len(LEADS), SPH_NATIVE_SAMPLES)!r}"
                )
            dtype = np.dtype(node.dtype)
            if np.issubdtype(dtype, np.bool_) or not (
                np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)
            ):
                raise SPHRecordValidationError(
                    f"record {source!s} must contain real numeric mV values"
                )
            native = np.asarray(node[()])
    except SPHRecordValidationError:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise SPHRecordValidationError(f"could not read record {source!s}: {error}") from error

    if native.shape != (len(LEADS), SPH_NATIVE_SAMPLES):
        raise SPHRecordValidationError(f"record {source!s} changed shape while reading")
    if not np.isfinite(native).all():
        raise SPHRecordValidationError(f"record {source!s} contains non-finite values")
    native_float = native.astype(np.float64, copy=False)
    resampled = resample_poly(native_float, up=1, down=5, axis=1)
    signal = np.ascontiguousarray(resampled, dtype=np.float32)
    if signal.shape != (len(LEADS), SPH_TARGET_SAMPLES):
        raise SPHRecordValidationError(
            f"record {source!s} produced resampled shape {signal.shape!r}; "
            f"expected {(len(LEADS), SPH_TARGET_SAMPLES)!r}"
        )
    if not np.isfinite(signal).all():
        raise SPHRecordValidationError(f"record {source!s} became non-finite during resampling")
    return signal


class SPHExternalTransportDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Torch dataset over a selected SPH primary external-transport manifest.

    Signals remain in physical mV after 500-to-100 Hz resampling.  Frozen PTB-XL
    normalization, if required by an evaluator, must be applied downstream and
    is intentionally not recomputed or inferred here.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        records_dir: str | Path,
        *,
        allow_all_zero: bool = False,
    ) -> None:
        _require_manifest_columns(manifest)
        if not isinstance(allow_all_zero, bool):
            raise TypeError("allow_all_zero must be a boolean")
        if manifest.empty:
            raise SPHMetadataValidationError("selected SPH manifest must not be empty")
        if not (manifest["native_samples"] == SPH_NATIVE_SAMPLES).all():
            raise SPHMetadataValidationError(
                f"selected SPH manifest must contain only N={SPH_NATIVE_SAMPLES} records"
            )
        if not allow_all_zero and not (manifest["mapped_target_count"] >= 1).all():
            raise SPHMetadataValidationError(
                "selected SPH manifest must contain at least one direct target per row"
            )

        try:
            numeric_targets = (
                manifest.loc[:, list(TARGET_COLUMNS)]
                .apply(pd.to_numeric, errors="raise")
                .to_numpy(dtype=np.float64)
            )
        except (TypeError, ValueError) as error:
            raise SPHMetadataValidationError("SPH targets must be numeric") from error
        if not np.isfinite(numeric_targets).all() or not np.isin(numeric_targets, (0.0, 1.0)).all():
            raise SPHMetadataValidationError("SPH targets must be finite binary values")
        mapped_counts = pd.to_numeric(manifest["mapped_target_count"], errors="raise").to_numpy(
            dtype=np.int64
        )
        if not np.array_equal(numeric_targets.sum(axis=1).astype(np.int64), mapped_counts):
            raise SPHMetadataValidationError(
                "mapped_target_count must equal the sum of direct target labels"
            )

        self._manifest = manifest.copy().reset_index(drop=True)
        self._records_dir = Path(records_dir)
        self._record_references = tuple(
            _validated_record_reference(value) for value in self._manifest["record_path"]
        )
        self._targets = cast(
            npt.NDArray[np.float32], numeric_targets.astype(np.float32, copy=False)
        )

    @property
    def manifest(self) -> pd.DataFrame:
        """Return a defensive copy of the selected canonical manifest."""

        return self._manifest.copy()

    def __len__(self) -> int:
        return len(self._manifest)

    def _position(self, index: int) -> int:
        if isinstance(index, bool):
            raise TypeError("dataset index must be an integer, not a boolean")
        try:
            position = operator.index(index)
        except TypeError as error:
            raise TypeError("dataset index must be an integer") from error
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError("SPH dataset index out of range")
        return position

    def record_path(self, index: int) -> Path:
        """Return the validated HDF5 path for one selected record."""

        position = self._position(index)
        return self._records_dir.joinpath(*PurePosixPath(self._record_references[position]).parts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        position = self._position(index)
        signal = torch.from_numpy(load_sph_signal(self.record_path(position)))
        target = torch.from_numpy(self._targets[position].copy())
        return signal, target


__all__ = [
    "AHACodeDefinition",
    "AHAStatement",
    "SPH_AMBIGUOUS_PRIMARY_CODES",
    "SPH_DIRECT_PRIMARY_CODES",
    "SPH_HDF5_DATASET_KEY",
    "SPH_NATIVE_FREQUENCY_HZ",
    "SPH_NATIVE_SAMPLES",
    "SPH_SIGNAL_UNIT",
    "SPH_SUPERCLASS_CODES",
    "SPH_TARGET_FREQUENCY_HZ",
    "SPH_TARGET_SAMPLES",
    "SPHExternalTransportDataset",
    "SPHMetadataValidationError",
    "SPHRecordValidationError",
    "build_sph_transport_manifest",
    "load_sph_signal",
    "map_sph_superclasses",
    "parse_sph_aha_codes",
    "read_sph_code_dictionary",
    "select_sph_exact_10s_records",
    "select_sph_transport_records",
]
