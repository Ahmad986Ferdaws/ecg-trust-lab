"""Label-free, fold-10-safe cohort identities for OOD completion."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ecg_trust.ood_completion.embedding_artifact import EmbeddingRole
from ecg_trust.source_calibration import SourceRole, patient_split_role

COHORT_IDENTITY_ALGORITHM = "ordered_role_input_identity_v1"
COHORT_IDENTITY_DOMAIN = b"ecg_trust.ordered_role_input_identity.v1\x00"
COHORT_IDENTITY_COLUMNS = ("ecg_id", "patient_id", "strat_fold", "record_path")

_MAX_MANIFEST_ROWS = 100_000


class OODCohortError(ValueError):
    """Raised when label-free cohort inputs violate their contract."""


class OODCohortIntegrityError(OODCohortError):
    """Raised when cohort isolation, identity, or expected counts fail."""


@dataclass(frozen=True, slots=True)
class CohortRecord:
    """One validated label-free manifest identity."""

    ecg_id: int
    patient_id: int
    strat_fold: int
    record_path: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "ecg_id", _positive_integer(self.ecg_id, "ecg_id"))
        object.__setattr__(
            self,
            "patient_id",
            _positive_integer(self.patient_id, "patient_id"),
        )
        fold = _positive_integer(self.strat_fold, "strat_fold")
        if fold > 9:
            raise OODCohortError("strat_fold must remain within authorized folds 1-9")
        object.__setattr__(self, "strat_fold", fold)
        object.__setattr__(self, "record_path", normalize_record_path(self.record_path))

    def to_identity_dict(self) -> dict[str, int | str]:
        return {
            "ecg_id": self.ecg_id,
            "patient_id": self.patient_id,
            "record_path": self.record_path,
            "strat_fold": self.strat_fold,
        }


@dataclass(frozen=True, slots=True)
class CohortCounts:
    """Expected or observed record and patient counts."""

    records: int
    patients: int

    def __post_init__(self) -> None:
        records = _positive_integer(self.records, "records")
        patients = _positive_integer(self.patients, "patients")
        if patients > records:
            raise OODCohortError("patient count cannot exceed record count")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "patients", patients)


@dataclass(frozen=True, slots=True)
class OODExpectedCohortCounts:
    """Complete expected-count contract for a frozen cohort preparation."""

    reference: CohortCounts
    decision_fit: CohortCounts
    threshold_fit: CohortCounts
    source_validation: CohortCounts
    full_fold9: CohortCounts

    def __post_init__(self) -> None:
        for name in (
            "reference",
            "decision_fit",
            "threshold_fit",
            "source_validation",
            "full_fold9",
        ):
            if not isinstance(getattr(self, name), CohortCounts):
                raise TypeError(f"{name} must be CohortCounts")
        partition_records = (
            self.decision_fit.records
            + self.threshold_fit.records
            + self.source_validation.records
        )
        partition_patients = (
            self.decision_fit.patients
            + self.threshold_fit.patients
            + self.source_validation.patients
        )
        if (
            partition_records != self.full_fold9.records
            or partition_patients != self.full_fold9.patients
        ):
            raise OODCohortError("expected A/B/C counts must exhaust full_fold9")


@dataclass(frozen=True, slots=True)
class OrderedCohort:
    """One strictly ordered private cohort and its domain-separated identity."""

    role: EmbeddingRole
    records: tuple[CohortRecord, ...]
    identity_sha256: str

    @classmethod
    def create(
        cls,
        role: EmbeddingRole | str,
        records: Sequence[CohortRecord],
    ) -> OrderedCohort:
        try:
            normalized_role = EmbeddingRole(role)
        except (TypeError, ValueError) as error:
            raise OODCohortError("unsupported OOD cohort role") from error
        ordered = _strict_ordered_records(records)
        if not ordered:
            raise OODCohortError(f"{normalized_role.value} cohort must not be empty")
        return cls(
            role=normalized_role,
            records=ordered,
            identity_sha256=ordered_role_input_identity_sha256(ordered),
        )

    @property
    def record_count(self) -> int:
        return len(self.records)

    @property
    def patient_count(self) -> int:
        return len({record.patient_id for record in self.records})

    @property
    def folds(self) -> tuple[int, ...]:
        return tuple(sorted({record.strat_fold for record in self.records}))

    @property
    def counts(self) -> CohortCounts:
        return CohortCounts(records=self.record_count, patients=self.patient_count)


@dataclass(frozen=True, slots=True)
class OODCohorts:
    """Frozen R/B/C cohorts plus label-free fold-9 identity evidence."""

    reference: OrderedCohort
    threshold_fit: OrderedCohort
    source_validation: OrderedCohort
    full_fold9_records: tuple[CohortRecord, ...]
    full_fold9_sha256: str
    full_fold9_counts: CohortCounts
    decision_fit_counts: CohortCounts

    @property
    def reference_sha256(self) -> str:
        return self.reference.identity_sha256


def normalize_record_path(value: object) -> str:
    """Return one normalized, safe project-relative POSIX record path."""

    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise OODCohortError("record_path must be a non-empty project-relative POSIX path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    raw_parts = value.split("/")
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in raw_parts)
        or posix.as_posix() != value
    ):
        raise OODCohortError("record_path must be a normalized project-relative POSIX path")
    return posix.as_posix()


def ordered_role_input_identity_sha256(records: Sequence[CohortRecord]) -> str:
    """Hash exact sorted identities with the frozen v1 domain separator."""

    ordered = _strict_ordered_records(records)
    if not ordered:
        raise OODCohortError("cohort identity requires at least one record")
    payload = {
        "algorithm": COHORT_IDENTITY_ALGORITHM,
        "records": [record.to_identity_dict() for record in ordered],
        "schema_version": 1,
    }
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(COHORT_IDENTITY_DOMAIN + canonical).hexdigest()


def load_ood_cohorts(
    manifest_path: str | Path,
    *,
    patient_split_salt: str,
    expected_counts: OODExpectedCohortCounts | None = None,
) -> OODCohorts:
    """Load only label-free folds 1-9 and construct patient-isolated R/B/C."""

    if not isinstance(patient_split_salt, str) or not patient_split_salt:
        raise OODCohortError("patient_split_salt must be a non-empty string")
    if expected_counts is not None and not isinstance(
        expected_counts, OODExpectedCohortCounts
    ):
        raise TypeError("expected_counts must be OODExpectedCohortCounts")
    source = Path(manifest_path)
    if source.suffix.casefold() != ".parquet":
        raise OODCohortError("OOD cohort manifest must be a parquet file")
    if source.is_symlink() or not source.is_file():
        raise OODCohortError("OOD cohort manifest is missing or symbolic")
    try:
        frame = pd.read_parquet(
            source,
            columns=list(COHORT_IDENTITY_COLUMNS),
            filters=[("strat_fold", "<=", 9)],
        )
    except (OSError, TypeError, ValueError) as error:
        raise OODCohortError(f"could not load label-free OOD cohort manifest: {error}") from error
    records = _validated_manifest_records(cast(pd.DataFrame, frame))
    return _partition_cohorts(
        records,
        patient_split_salt=patient_split_salt,
        expected_counts=expected_counts,
    )


def _validated_manifest_records(frame: pd.DataFrame) -> tuple[CohortRecord, ...]:
    if not isinstance(frame, pd.DataFrame):
        raise OODCohortError("parquet reader did not return a DataFrame")
    if list(frame.columns) != list(COHORT_IDENTITY_COLUMNS):
        raise OODCohortError("manifest must return exactly the four label-free columns")
    if frame.empty or len(frame) > _MAX_MANIFEST_ROWS:
        raise OODCohortError("manifest row count is outside the supported bound")
    if frame.isna().any().any():
        raise OODCohortError("manifest identity fields must not be missing")
    records: list[CohortRecord] = []
    for ecg_id, patient_id, strat_fold, record_path in frame.itertuples(
        index=False, name=None
    ):
        records.append(
            CohortRecord(
                ecg_id=_strict_manifest_integer(ecg_id, "ecg_id"),
                patient_id=_strict_manifest_integer(patient_id, "patient_id"),
                strat_fold=_strict_manifest_integer(strat_fold, "strat_fold"),
                record_path=record_path,
            )
        )
    ordered = _strict_ordered_records(records)
    _assert_one_fold_per_patient(ordered)
    return ordered


def _partition_cohorts(
    records: Sequence[CohortRecord],
    *,
    patient_split_salt: str,
    expected_counts: OODExpectedCohortCounts | None,
) -> OODCohorts:
    reference_records = tuple(record for record in records if record.strat_fold <= 8)
    fold9_records = tuple(record for record in records if record.strat_fold == 9)
    if not reference_records or not fold9_records:
        raise OODCohortError("manifest must contain both reference and fold-9 records")

    role_by_patient = {
        patient_id: patient_split_role(patient_id=patient_id, salt=patient_split_salt)
        for patient_id in {record.patient_id for record in fold9_records}
    }
    decision_records = tuple(
        record
        for record in fold9_records
        if role_by_patient[record.patient_id] is SourceRole.DECISION_FIT
    )
    threshold_records = tuple(
        record
        for record in fold9_records
        if role_by_patient[record.patient_id]
        is SourceRole.CONFORMAL_AND_OOD_THRESHOLD_FIT
    )
    validation_records = tuple(
        record
        for record in fold9_records
        if role_by_patient[record.patient_id] is SourceRole.SOURCE_VALIDATION
    )
    if not decision_records or not threshold_records or not validation_records:
        raise OODCohortError("fold-9 patient partition must produce non-empty A/B/C roles")
    if len(decision_records) + len(threshold_records) + len(validation_records) != len(
        fold9_records
    ):
        raise OODCohortIntegrityError("fold-9 A/B/C partition is not exhaustive")

    reference = OrderedCohort.create(EmbeddingRole.REFERENCE, reference_records)
    threshold = OrderedCohort.create(EmbeddingRole.THRESHOLD_FIT, threshold_records)
    validation = OrderedCohort.create(EmbeddingRole.SOURCE_VALIDATION, validation_records)
    _assert_disjoint_patients(reference, threshold, validation)
    fold9_counts = _counts(fold9_records)
    decision_counts = _counts(decision_records)
    result = OODCohorts(
        reference=reference,
        threshold_fit=threshold,
        source_validation=validation,
        full_fold9_records=fold9_records,
        full_fold9_sha256=ordered_role_input_identity_sha256(fold9_records),
        full_fold9_counts=fold9_counts,
        decision_fit_counts=decision_counts,
    )
    if expected_counts is not None:
        observed = {
            "reference": reference.counts,
            "decision_fit": decision_counts,
            "threshold_fit": threshold.counts,
            "source_validation": validation.counts,
            "full_fold9": fold9_counts,
        }
        for name, counts in observed.items():
            if counts != getattr(expected_counts, name):
                raise OODCohortIntegrityError(f"observed {name} counts differ from expectation")
    return result


def _strict_ordered_records(records: Sequence[CohortRecord]) -> tuple[CohortRecord, ...]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence of CohortRecord values")
    values = tuple(records)
    if any(not isinstance(record, CohortRecord) for record in values):
        raise TypeError("records must contain only CohortRecord values")
    ordered = tuple(sorted(values, key=lambda record: record.ecg_id))
    ecg_ids = tuple(record.ecg_id for record in ordered)
    if len(ecg_ids) != len(set(ecg_ids)):
        raise OODCohortIntegrityError("cohort ecg_id values must be unique")
    return ordered


def _assert_one_fold_per_patient(records: Sequence[CohortRecord]) -> None:
    folds_by_patient: dict[int, int] = {}
    for record in records:
        prior = folds_by_patient.setdefault(record.patient_id, record.strat_fold)
        if prior != record.strat_fold:
            raise OODCohortIntegrityError("a patient occurs in multiple manifest folds")


def _assert_disjoint_patients(*cohorts: OrderedCohort) -> None:
    seen: set[int] = set()
    for cohort in cohorts:
        patients = {record.patient_id for record in cohort.records}
        if seen.intersection(patients):
            raise OODCohortIntegrityError("R/B/C cohorts contain overlapping patients")
        seen.update(patients)


def _counts(records: Sequence[CohortRecord]) -> CohortCounts:
    return CohortCounts(
        records=len(records),
        patients=len({record.patient_id for record in records}),
    )


def _strict_manifest_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise OODCohortError(f"manifest {context} must contain integers")
    return int(value)


def _positive_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OODCohortError(f"{context} must be a positive integer")
    return value


__all__ = [
    "COHORT_IDENTITY_ALGORITHM",
    "COHORT_IDENTITY_COLUMNS",
    "COHORT_IDENTITY_DOMAIN",
    "CohortCounts",
    "CohortRecord",
    "OODCohortError",
    "OODCohortIntegrityError",
    "OODCohorts",
    "OODExpectedCohortCounts",
    "OrderedCohort",
    "load_ood_cohorts",
    "normalize_record_path",
    "ordered_role_input_identity_sha256",
]
