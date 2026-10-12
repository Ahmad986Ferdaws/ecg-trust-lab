"""Strict PTB-XL waveform loading and training-only normalization.

The module intentionally depends only on ordinary manifest columns.  A manifest
builder may therefore evolve independently as long as it supplies a WFDB record
path, five binary target columns, and (when fold selection is requested) a fold
column.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import os
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
import torch
import wfdb  # type: ignore[import-untyped]
from torch.utils.data import Dataset

from ecg_trust.constants import LEADS, PTBXL_VERSION, SUPERCLASSES, TARGET_COLUMNS
from ecg_trust.data.manifest import ManifestError, validate_relative_path
from ecg_trust.normalization import (
    NormalizationProvenance as NormalizationProvenance,
)
from ecg_trust.normalization import (
    NormalizationStats as NormalizationStats,
)
from ecg_trust.normalization import (
    NormalizationValidationError as NormalizationValidationError,
)
from ecg_trust.protocol import (
    TRAIN_FOLDS,
    ExperimentProtocol,
    FinalTestAccessToken,
)


class ManifestValidationError(ValueError):
    """Raised when manifest rows do not satisfy the dataset contract."""


class RecordValidationError(ValueError):
    """Raised when a WFDB record does not satisfy the signal contract."""


def _normalize_requested_folds(folds: int | Collection[int] | None) -> tuple[int, ...] | None:
    if folds is None:
        return None
    if isinstance(folds, bool):
        raise ManifestValidationError("folds must contain integers, not booleans")
    raw_folds: Collection[int] = (folds,) if isinstance(folds, int) else folds
    normalized: set[int] = set()
    for fold in raw_folds:
        if isinstance(fold, bool):
            raise ManifestValidationError("folds must contain integers, not booleans")
        try:
            parsed = operator.index(fold)
        except TypeError as error:
            raise ManifestValidationError("folds must contain integers") from error
        if parsed < 1:
            raise ManifestValidationError("fold values must be positive integers")
        normalized.add(parsed)
    if not normalized:
        raise ManifestValidationError("folds must not be empty")
    return tuple(sorted(normalized))


def _validated_fold_values(series: pd.Series) -> npt.NDArray[np.int64]:
    try:
        numeric = pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ManifestValidationError("fold column must contain integers") from error
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ManifestValidationError("fold column must contain finite integers")
    if (numeric < 1).any():
        raise ManifestValidationError("fold column must contain positive integers")
    return cast(npt.NDArray[np.int64], numeric.astype(np.int64))


def _validate_column_names(target_columns: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(target_columns)
    if len(normalized) != len(SUPERCLASSES):
        raise ManifestValidationError(
            f"target_columns must contain exactly {len(SUPERCLASSES)} columns"
        )
    if len(set(normalized)) != len(normalized):
        raise ManifestValidationError("target_columns must be unique")
    if any(not isinstance(column, str) or not column for column in normalized):
        raise ManifestValidationError("target_columns must contain non-empty strings")
    return normalized


def _validate_targets(
    frame: pd.DataFrame,
    target_columns: tuple[str, ...],
    *,
    require_positive_target: bool,
) -> npt.NDArray[np.float32]:
    try:
        targets = (
            frame.loc[:, list(target_columns)]
            .apply(pd.to_numeric, errors="raise")
            .to_numpy(dtype=np.float64)
        )
    except (TypeError, ValueError) as error:
        raise ManifestValidationError(
            "target columns must contain numeric binary values"
        ) from error
    if not np.isfinite(targets).all():
        raise ManifestValidationError("target columns must contain finite values")
    if not np.isin(targets, (0.0, 1.0)).all():
        raise ManifestValidationError("target columns must contain only 0 or 1")
    if require_positive_target and (targets.sum(axis=1) == 0.0).any():
        raise ManifestValidationError("each record must have at least one positive target")
    return cast(npt.NDArray[np.float32], targets.astype(np.float32, copy=False))


def _record_reference(value: object) -> str:
    if not isinstance(value, (str, os.PathLike)):
        raise ManifestValidationError("record path values must be strings or path-like objects")
    reference = os.fspath(value)
    if isinstance(reference, bytes):
        reference = os.fsdecode(reference)
    if not reference.strip():
        raise ManifestValidationError("record path values must not be empty")
    try:
        return validate_relative_path(reference)
    except ManifestError as error:
        raise ManifestValidationError(f"invalid record path {reference!r}: {error}") from error


class PTBXLDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load strictly validated 100 Hz PTB-XL records from manifest rows.

    Args:
        manifest: One row per ECG.  The output target order is the order given by
            ``target_columns`` (the canonical five superclasses by default).
        root_dir: Base directory for relative WFDB record paths.
        split: Optional exact value to select from ``split_column``.
        folds: Optional fold or folds to select from ``fold_column``.  Supplying
            both ``split`` and ``folds`` uses their intersection.
        normalization: Optional training-derived per-lead statistics.

    WFDB paths may be stored with or without ``.hea``/``.dat`` suffixes.  No
    resampling, padding, truncation, unit conversion, or per-record normalization is performed.
    Such silent repairs would weaken the experimental data contract.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        root_dir: str | Path,
        *,
        path_column: str = "record_path",
        target_columns: Sequence[str] = TARGET_COLUMNS,
        split: str | None = None,
        split_column: str = "split",
        folds: int | Collection[int] | None = None,
        fold_column: str = "strat_fold",
        expected_samples: int = 1_000,
        expected_frequency_hz: float = 100.0,
        normalization: NormalizationStats | None = None,
        require_positive_target: bool = True,
        protocol: ExperimentProtocol | None = None,
        test_access: FinalTestAccessToken | None = None,
    ) -> None:
        if not isinstance(manifest, pd.DataFrame):
            raise TypeError("manifest must be a pandas DataFrame")
        if manifest.empty:
            raise ManifestValidationError("manifest must not be empty")
        if isinstance(expected_samples, bool) or not isinstance(expected_samples, int):
            raise ValueError("expected_samples must be a positive integer")
        if expected_samples < 1:
            raise ValueError("expected_samples must be a positive integer")
        if isinstance(expected_frequency_hz, bool) or not isinstance(
            expected_frequency_hz, (int, float)
        ):
            raise ValueError("expected_frequency_hz must be finite and positive")
        expected_frequency_hz = float(expected_frequency_hz)
        if not math.isfinite(expected_frequency_hz) or expected_frequency_hz <= 0.0:
            raise ValueError("expected_frequency_hz must be finite and positive")

        resolved_protocol = protocol or ExperimentProtocol.canonical()
        if not isinstance(resolved_protocol, ExperimentProtocol):
            raise TypeError("protocol must be an ExperimentProtocol")

        targets = _validate_column_names(target_columns)
        if targets != tuple(f"label_{label}" for label in resolved_protocol.label_order):
            raise ManifestValidationError(
                "target_columns must preserve the protocol label order "
                f"{resolved_protocol.label_order!r}"
            )
        required_columns = {path_column, fold_column, *targets}
        requested_folds = _normalize_requested_folds(folds)
        if split is not None:
            if not isinstance(split, str) or not split:
                raise ManifestValidationError("split must be a non-empty string")
            required_columns.add(split_column)
        missing = sorted(required_columns.difference(manifest.columns))
        if missing:
            raise ManifestValidationError(f"manifest is missing required columns: {missing}")

        fold_values = _validated_fold_values(manifest[fold_column])
        if requested_folds is not None:
            resolved_protocol.guard_fold_access(requested_folds, test_access=test_access)

        selection = np.ones(len(manifest), dtype=bool)
        if split is not None:
            selection &= manifest[split_column].eq(split).to_numpy(dtype=bool)
        if requested_folds is not None:
            selection &= np.isin(fold_values, requested_folds)
        selected = manifest.loc[selection].reset_index(drop=True).copy()
        if selected.empty:
            details = []
            if split is not None:
                details.append(f"split={split!r}")
            if requested_folds is not None:
                details.append(f"folds={requested_folds!r}")
            condition = ", ".join(details) if details else "selection"
            raise ManifestValidationError(f"no manifest rows match {condition}")

        selected_folds = tuple(sorted(int(value) for value in np.unique(fold_values[selection])))
        resolved_protocol.guard_fold_access(selected_folds, test_access=test_access)

        record_references = tuple(_record_reference(value) for value in selected[path_column])
        target_values = _validate_targets(
            selected, targets, require_positive_target=require_positive_target
        )

        self.root_dir = Path(root_dir).resolve()
        self.path_column = path_column
        self.target_columns = targets
        self.split = split
        self.split_column = split_column
        self.folds = requested_folds
        self.fold_column = fold_column
        self.expected_samples = expected_samples
        self.expected_frequency_hz = expected_frequency_hz
        self.normalization = normalization
        self.protocol = resolved_protocol
        self._manifest = selected
        self._record_references = record_references
        self._targets = target_values

        if normalization is not None:
            provenance = normalization.provenance
            if provenance.samples_per_record != expected_samples:
                raise NormalizationValidationError(
                    "normalization samples_per_record does not match dataset"
                )
            if not math.isclose(
                provenance.sampling_frequency_hz,
                expected_frequency_hz,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise NormalizationValidationError(
                    "normalization sampling_frequency_hz does not match dataset"
                )

    @property
    def manifest(self) -> pd.DataFrame:
        """Return a defensive copy of the selected manifest rows."""

        return self._manifest.copy()

    def __len__(self) -> int:
        return len(self._record_references)

    def _position(self, index: int) -> int:
        try:
            position = operator.index(index)
        except TypeError as error:
            raise TypeError("dataset index must be an integer") from error
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError("dataset index out of range")
        return position

    def record_path(self, index: int) -> Path:
        """Return the suffix-free WFDB path for a selected row."""

        position = self._position(index)
        path = Path(self._record_references[position])
        if path.suffix.casefold() in {".hea", ".dat"}:
            path = path.with_suffix("")
        return path if path.is_absolute() else self.root_dir / path

    def load_signal(self, index: int) -> torch.Tensor:
        """Load one signal as finite float32 ``[12, expected_samples]``."""

        path = self.record_path(index)
        try:
            record = wfdb.rdrecord(str(path))
        except Exception as error:
            raise RecordValidationError(f"could not read WFDB record {path!s}: {error}") from error

        frequency = cast(float | int | str, getattr(record, "fs", None))
        try:
            frequency_hz = float(frequency)
        except (TypeError, ValueError) as error:
            raise RecordValidationError(
                f"record {path!s} has an invalid sampling frequency"
            ) from error
        if not math.isfinite(frequency_hz) or not math.isclose(
            frequency_hz,
            self.expected_frequency_hz,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RecordValidationError(
                f"record {path!s} has sampling frequency {frequency_hz!r}; "
                f"expected {self.expected_frequency_hz!r} Hz"
            )

        physical_signal = getattr(record, "p_signal", None)
        if physical_signal is None:
            raise RecordValidationError(f"record {path!s} has no physical signal")
        signal = np.asarray(physical_signal, dtype=np.float32)
        if signal.ndim != 2:
            raise RecordValidationError(f"record {path!s} signal must be two-dimensional")
        if signal.shape[0] != self.expected_samples:
            raise RecordValidationError(
                f"record {path!s} has {signal.shape[0]} samples; expected {self.expected_samples}"
            )

        signal_names = getattr(record, "sig_name", None)
        if not isinstance(signal_names, Sequence) or isinstance(signal_names, (str, bytes)):
            raise RecordValidationError(f"record {path!s} has no valid lead names")
        if signal.shape[1] != len(signal_names):
            raise RecordValidationError(
                f"record {path!s} signal columns do not match its lead-name count"
            )

        canonical_by_key = {lead.casefold(): lead for lead in LEADS}
        positions: dict[str, int] = {}
        for column, raw_name in enumerate(signal_names):
            if not isinstance(raw_name, str):
                raise RecordValidationError(f"record {path!s} contains a non-string lead name")
            canonical = canonical_by_key.get(raw_name.strip().casefold())
            if canonical is None:
                raise RecordValidationError(
                    f"record {path!s} contains unexpected lead {raw_name!r}"
                )
            if canonical in positions:
                raise RecordValidationError(
                    f"record {path!s} contains duplicate lead {canonical!r}"
                )
            positions[canonical] = column
        missing_leads = [lead for lead in LEADS if lead not in positions]
        if missing_leads or len(positions) != len(LEADS):
            raise RecordValidationError(
                f"record {path!s} must contain exactly the canonical 12 leads; "
                f"missing={missing_leads!r}"
            )

        order = [positions[lead] for lead in LEADS]
        canonical_signal = np.ascontiguousarray(signal[:, order].T, dtype=np.float32)
        if canonical_signal.shape != (len(LEADS), self.expected_samples):
            raise RecordValidationError(
                f"record {path!s} produced invalid shape {canonical_signal.shape!r}"
            )
        if not np.isfinite(canonical_signal).all():
            raise RecordValidationError(f"record {path!s} contains non-finite signal values")

        units = getattr(record, "units", None)
        if (
            not isinstance(units, Sequence)
            or isinstance(units, (str, bytes))
            or len(units) != len(signal_names)
        ):
            raise RecordValidationError(
                f"record {path!s} must declare one physical unit per lead"
            )
        for lead in LEADS:
            unit = units[positions[lead]]
            if unit != "mV":
                raise RecordValidationError(
                    f"record {path!s} lead {lead!r} has physical unit {unit!r}; expected 'mV'"
                )

        tensor = torch.from_numpy(canonical_signal)
        if self.normalization is not None:
            mean = torch.tensor(self.normalization.mean, dtype=torch.float32).unsqueeze(1)
            std = torch.tensor(self.normalization.std, dtype=torch.float32).unsqueeze(1)
            tensor = (tensor - mean) / std
            if not torch.isfinite(tensor).all().item():
                raise RecordValidationError(
                    f"record {path!s} became non-finite after normalization"
                )
        return tensor.contiguous()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        position = self._position(index)
        signal = self.load_signal(position)
        target = torch.from_numpy(self._targets[position].copy())
        return signal, target


def _selected_manifest_sha256(dataset: PTBXLDataset, fold_column: str) -> str:
    """Hash stable, normalized row content used for statistics."""

    fold_values = _validated_fold_values(dataset._manifest[fold_column])
    serialized_rows: list[str] = []
    for position, reference in enumerate(dataset._record_references):
        row = {
            "fold": int(fold_values[position]),
            "path": reference.replace("\\", "/"),
            "targets": [int(value) for value in dataset._targets[position]],
        }
        serialized_rows.append(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
    content = "\n".join(sorted(serialized_rows)).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def compute_normalization_stats(
    manifest: pd.DataFrame,
    root_dir: str | Path,
    *,
    training_folds: int | Collection[int],
    path_column: str = "record_path",
    target_columns: Sequence[str] = TARGET_COLUMNS,
    fold_column: str = "strat_fold",
    expected_samples: int = 1_000,
    expected_frequency_hz: float = 100.0,
    dataset_version: str = PTBXL_VERSION,
) -> NormalizationStats:
    """Compute population mean/std per lead using only requested training folds.

    A mergeable Welford update processes one record at a time in float64, so the
    operation does not materialize the full PTB-XL signal tensor in memory.
    """

    normalized_folds = _normalize_requested_folds(training_folds)
    if normalized_folds is None:  # pragma: no cover - excluded by the public type contract
        raise ManifestValidationError("training_folds must not be None")
    if not set(normalized_folds).issubset(TRAIN_FOLDS):
        raise NormalizationValidationError(
            "normalization must be fitted exclusively on training folds 1-7"
        )
    dataset = PTBXLDataset(
        manifest,
        root_dir,
        path_column=path_column,
        target_columns=target_columns,
        folds=normalized_folds,
        fold_column=fold_column,
        expected_samples=expected_samples,
        expected_frequency_hz=expected_frequency_hz,
        normalization=None,
    )

    count = 0
    running_mean = np.zeros(len(LEADS), dtype=np.float64)
    running_m2 = np.zeros(len(LEADS), dtype=np.float64)
    for index in range(len(dataset)):
        values = dataset.load_signal(index).numpy().astype(np.float64, copy=False)
        batch_count = values.shape[1]
        batch_mean = values.mean(axis=1, dtype=np.float64)
        centered = values - batch_mean[:, np.newaxis]
        batch_m2 = np.square(centered).sum(axis=1, dtype=np.float64)

        combined_count = count + batch_count
        delta = batch_mean - running_mean
        running_mean += delta * (batch_count / combined_count)
        running_m2 += batch_m2 + np.square(delta) * (count * batch_count / combined_count)
        count = combined_count

    if count == 0:  # Defensive: PTBXLDataset already rejects an empty selection.
        raise NormalizationValidationError("cannot compute statistics from zero samples")
    variance = running_m2 / count
    std = np.sqrt(variance)
    if not np.isfinite(running_mean).all() or not np.isfinite(std).all():
        raise NormalizationValidationError("computed normalization values are non-finite")
    zero_variance_leads = [LEADS[index] for index, value in enumerate(std) if value <= 0.0]
    if zero_variance_leads:
        raise NormalizationValidationError(
            f"cannot normalize zero-variance leads: {zero_variance_leads!r}"
        )

    provenance = NormalizationProvenance(
        dataset_version=dataset_version,
        manifest_sha256=_selected_manifest_sha256(dataset, fold_column),
        training_folds=normalized_folds,
        record_count=len(dataset),
        sample_count=count,
        sampling_frequency_hz=expected_frequency_hz,
        samples_per_record=expected_samples,
        path_column=path_column,
        fold_column=fold_column,
        target_columns=tuple(target_columns),
    )
    return NormalizationStats(
        mean=tuple(float(value) for value in running_mean),
        std=tuple(float(value) for value in std),
        leads=LEADS,
        provenance=provenance,
    )
