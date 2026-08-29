"""Deterministic patient-disjoint, temporally ordered split artifacts."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from ecg_trust.longitudinal.contracts import (
    AGGREGATE_ONLY_LIMIT,
    RESEARCH_USE_LIMIT,
    LongitudinalError,
    LongitudinalIntegrityError,
    SourceRole,
    canonical_sha256,
    canonical_utc_timestamp,
    parse_utc,
)
from ecg_trust.longitudinal.timeline import LongitudinalCohort

SPLIT_SCHEMA_VERSION = 1
SPLIT_ARTIFACT_TYPE = "ecg_trust.longitudinal_temporal_split_manifest"
_HMAC_TOKEN = re.compile(r"^hmac-sha256:[0-9a-f]{64}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class TemporalPartition(StrEnum):
    """Chronologically ordered patient-level research partitions."""

    DEVELOPMENT = "development"
    CALIBRATION = "calibration"
    TEMPORAL_EVALUATION = "temporal_evaluation"


_PARTITION_ORDER = {
    TemporalPartition.DEVELOPMENT: 0,
    TemporalPartition.CALIBRATION: 1,
    TemporalPartition.TEMPORAL_EVALUATION: 2,
}


@dataclass(frozen=True, slots=True)
class TemporalSplitAssignment:
    """One pseudonymized patient assignment; raw linkage keys are never stored."""

    patient_token: str
    source_scope_tokens: tuple[str, ...]
    assignment_time_utc: str
    partition: TemporalPartition
    source_role: SourceRole

    def __post_init__(self) -> None:
        if _HMAC_TOKEN.fullmatch(self.patient_token) is None:
            raise LongitudinalError("patient_token must be a keyed HMAC-SHA256")
        if not self.source_scope_tokens:
            raise LongitudinalError("source_scope_tokens must not be empty")
        if any(_HMAC_TOKEN.fullmatch(item) is None for item in self.source_scope_tokens):
            raise LongitudinalError("source_scope_tokens must contain keyed HMAC-SHA256 values")
        if len(set(self.source_scope_tokens)) != len(self.source_scope_tokens):
            raise LongitudinalError("source_scope_tokens must be unique")
        if self.source_scope_tokens != tuple(sorted(self.source_scope_tokens)):
            raise LongitudinalError("source_scope_tokens must be canonically sorted")
        canonical_utc_timestamp(self.assignment_time_utc, "assignment_time_utc")

    def to_dict(self) -> dict[str, object]:
        return {
            "patient_token": self.patient_token,
            "source_scope_tokens": list(self.source_scope_tokens),
            "assignment_time_utc": self.assignment_time_utc,
            "partition": self.partition.value,
            "source_role": self.source_role.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TemporalSplitAssignment:
        expected = {
            "patient_token",
            "source_scope_tokens",
            "assignment_time_utc",
            "partition",
            "source_role",
        }
        if set(payload) != expected:
            raise LongitudinalIntegrityError("temporal assignment has missing or unknown fields")
        try:
            partition = TemporalPartition(_string(payload["partition"], "partition"))
            source_role = SourceRole(_string(payload["source_role"], "source_role"))
        except ValueError as exc:
            raise LongitudinalIntegrityError(
                "temporal assignment contains an invalid role"
            ) from exc
        return cls(
            patient_token=_string(payload["patient_token"], "patient_token"),
            source_scope_tokens=_string_sequence(
                payload["source_scope_tokens"], "source_scope_tokens"
            ),
            assignment_time_utc=_string(payload["assignment_time_utc"], "assignment_time_utc"),
            partition=partition,
            source_role=source_role,
        )


@dataclass(frozen=True, slots=True, init=False)
class TemporalSplitManifest:
    """Canonical, self-hashed temporal split with pseudonymous assignments."""

    timeline_config_sha256: str
    development_end_utc: str
    calibration_end_utc: str
    assignments: tuple[TemporalSplitAssignment, ...]
    manifest_sha256: str

    def _body(self) -> dict[str, object]:
        return {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "artifact_type": SPLIT_ARTIFACT_TYPE,
            "timeline_config_sha256": self.timeline_config_sha256,
            "development_end_utc": self.development_end_utc,
            "calibration_end_utc": self.calibration_end_utc,
            "assignments": [item.to_dict() for item in self.assignments],
            "privacy_contract": "keyed_patient_tokens_no_raw_linkage_identifiers",
            "research_use_limit": RESEARCH_USE_LIMIT,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._body(), "manifest_sha256": self.manifest_sha256}

    def public_summary(self) -> dict[str, object]:
        """Return partition counts only, with no patient or source tokens."""

        counts = Counter(item.partition.value for item in self.assignments)
        return {
            "development_count": counts[TemporalPartition.DEVELOPMENT.value],
            "calibration_count": counts[TemporalPartition.CALIBRATION.value],
            "temporal_evaluation_count": counts[TemporalPartition.TEMPORAL_EVALUATION.value],
            "total_patient_count": len(self.assignments),
            "manifest_sha256": self.manifest_sha256,
            "privacy_contract": AGGREGATE_ONLY_LIMIT,
            "research_use_limit": RESEARCH_USE_LIMIT,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TemporalSplitManifest:
        expected = {
            "schema_version",
            "artifact_type",
            "timeline_config_sha256",
            "development_end_utc",
            "calibration_end_utc",
            "assignments",
            "privacy_contract",
            "research_use_limit",
            "manifest_sha256",
        }
        if set(payload) != expected:
            raise LongitudinalIntegrityError("split manifest has missing or unknown fields")
        if payload["schema_version"] != SPLIT_SCHEMA_VERSION:
            raise LongitudinalIntegrityError("unsupported split manifest schema_version")
        if payload["artifact_type"] != SPLIT_ARTIFACT_TYPE:
            raise LongitudinalIntegrityError("invalid split manifest artifact_type")
        if payload["privacy_contract"] != "keyed_patient_tokens_no_raw_linkage_identifiers":
            raise LongitudinalIntegrityError("invalid split privacy contract")
        if payload["research_use_limit"] != RESEARCH_USE_LIMIT:
            raise LongitudinalIntegrityError("invalid split research-use limitation")
        timeline_hash = _sha256(payload["timeline_config_sha256"], "timeline_config_sha256")
        development_end = canonical_utc_timestamp(
            payload["development_end_utc"], "development_end_utc"
        )
        calibration_end = canonical_utc_timestamp(
            payload["calibration_end_utc"], "calibration_end_utc"
        )
        if parse_utc(development_end) >= parse_utc(calibration_end):
            raise LongitudinalIntegrityError("temporal split cutoffs are not chronological")
        rows = _mapping_sequence(payload["assignments"], "assignments")
        assignments = tuple(TemporalSplitAssignment.from_dict(row) for row in rows)
        _validate_assignments(assignments, development_end, calibration_end)
        canonical = tuple(sorted(assignments, key=_assignment_sort_key))
        if assignments != canonical:
            raise LongitudinalIntegrityError("split assignments are not canonically sorted")
        instance = object.__new__(cls)
        object.__setattr__(instance, "timeline_config_sha256", timeline_hash)
        object.__setattr__(instance, "development_end_utc", development_end)
        object.__setattr__(instance, "calibration_end_utc", calibration_end)
        object.__setattr__(instance, "assignments", assignments)
        body = instance._body()
        stored_hash = _sha256(payload["manifest_sha256"], "manifest_sha256")
        if canonical_sha256(body) != stored_hash:
            raise LongitudinalIntegrityError("temporal split manifest SHA-256 mismatch")
        object.__setattr__(instance, "manifest_sha256", stored_hash)
        return instance


def build_temporal_split_manifest(
    cohort: LongitudinalCohort,
    *,
    development_end_utc: str,
    calibration_end_utc: str,
    pseudonymization_key: bytes,
) -> TemporalSplitManifest:
    """Assign each patient once using the last pre-index input encounter time."""

    if not isinstance(cohort, LongitudinalCohort):
        raise TypeError("cohort must be a LongitudinalCohort")
    if not isinstance(pseudonymization_key, bytes):
        raise LongitudinalError("pseudonymization_key must be bytes")
    if len(pseudonymization_key) < 32:
        raise LongitudinalError("pseudonymization_key must contain at least 32 bytes")
    development_end = canonical_utc_timestamp(development_end_utc, "development_end_utc")
    calibration_end = canonical_utc_timestamp(calibration_end_utc, "calibration_end_utc")
    if parse_utc(development_end) >= parse_utc(calibration_end):
        raise LongitudinalError("development_end_utc must precede calibration_end_utc")
    if parse_utc(calibration_end) >= parse_utc(cohort.config.index_time_utc):
        raise LongitudinalError("calibration_end_utc must precede the cohort index time")
    if not cohort.timelines:
        raise LongitudinalError("cannot split an empty longitudinal cohort")

    assignments: list[TemporalSplitAssignment] = []
    for timeline in cohort.timelines:
        assignment_time = timeline.assignment_time_utc
        if parse_utc(assignment_time) <= parse_utc(development_end):
            partition = TemporalPartition.DEVELOPMENT
        elif parse_utc(assignment_time) <= parse_utc(calibration_end):
            partition = TemporalPartition.CALIBRATION
        else:
            partition = TemporalPartition.TEMPORAL_EVALUATION
        patient_token = _keyed_token(
            pseudonymization_key,
            "patient",
            list(timeline.patient_identity),
        )
        source_tokens = tuple(
            sorted(
                _keyed_token(pseudonymization_key, "source_scope", list(identity))
                for identity in timeline.source_identities
            )
        )
        assignments.append(
            TemporalSplitAssignment(
                patient_token=patient_token,
                source_scope_tokens=source_tokens,
                assignment_time_utc=assignment_time,
                partition=partition,
                source_role=timeline.source_role,
            )
        )

    canonical = tuple(sorted(assignments, key=_assignment_sort_key))
    _validate_assignments(canonical, development_end, calibration_end)
    instance = object.__new__(TemporalSplitManifest)
    object.__setattr__(instance, "timeline_config_sha256", cohort.config.config_sha256)
    object.__setattr__(instance, "development_end_utc", development_end)
    object.__setattr__(instance, "calibration_end_utc", calibration_end)
    object.__setattr__(instance, "assignments", canonical)
    object.__setattr__(instance, "manifest_sha256", canonical_sha256(instance._body()))
    return instance


def _validate_assignments(
    assignments: Sequence[TemporalSplitAssignment],
    development_end_utc: str,
    calibration_end_utc: str,
) -> None:
    if not assignments:
        raise LongitudinalIntegrityError("split manifest must contain assignments")
    patient_tokens = [item.patient_token for item in assignments]
    if len(set(patient_tokens)) != len(patient_tokens):
        raise LongitudinalIntegrityError("a patient token appears in more than one split")
    source_roles: dict[str, SourceRole] = {}
    present: set[TemporalPartition] = set()
    development_end = parse_utc(development_end_utc)
    calibration_end = parse_utc(calibration_end_utc)
    for item in assignments:
        for source_token in item.source_scope_tokens:
            prior = source_roles.setdefault(source_token, item.source_role)
            if prior is not item.source_role:
                raise LongitudinalIntegrityError("a source scope appears in multiple source roles")
        timestamp = parse_utc(item.assignment_time_utc)
        expected = (
            TemporalPartition.DEVELOPMENT
            if timestamp <= development_end
            else TemporalPartition.CALIBRATION
            if timestamp <= calibration_end
            else TemporalPartition.TEMPORAL_EVALUATION
        )
        if item.partition is not expected:
            raise LongitudinalIntegrityError("assignment violates temporal split cutoffs")
        present.add(item.partition)
    if present != set(TemporalPartition):
        raise LongitudinalIntegrityError("all temporal partitions must contain patients")


def _assignment_sort_key(item: TemporalSplitAssignment) -> tuple[int, str, str]:
    return (_PARTITION_ORDER[item.partition], item.assignment_time_utc, item.patient_token)


def _keyed_token(key: bytes, domain: str, identity: object) -> str:
    encoded = json.dumps(
        {"domain": domain, "identity": identity},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "hmac-sha256:" + hmac.new(key, encoded, hashlib.sha256).hexdigest()


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise LongitudinalIntegrityError(f"{name} must be text")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise LongitudinalIntegrityError(f"{name} must be a prefixed SHA-256")
    return value


def _mapping_sequence(value: object, name: str) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LongitudinalIntegrityError(f"{name} must be a sequence")
    rows: list[Mapping[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise LongitudinalIntegrityError(f"{name} must contain mappings")
        rows.append(cast(Mapping[str, object], item))
    return tuple(rows)


def _string_sequence(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LongitudinalIntegrityError(f"{name} must be a sequence")
    if any(not isinstance(item, str) for item in value):
        raise LongitudinalIntegrityError(f"{name} must contain text")
    return tuple(cast(str, item) for item in value)
