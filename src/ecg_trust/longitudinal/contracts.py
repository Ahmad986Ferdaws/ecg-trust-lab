"""Immutable contracts for retrospective longitudinal ECG research."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import cast

TIMELINE_CONFIG_SCHEMA_VERSION = 1
TIMELINE_CONFIG_ARTIFACT_TYPE = "ecg_trust.longitudinal_timeline_config"
RESEARCH_USE_LIMIT = "retrospective_research_only_not_for_clinical_decisions"
NON_CAUSAL_TARGET_LIMIT = "retrospective_future_association_not_causal"
AGGREGATE_ONLY_LIMIT = "aggregate_only_no_patient_level_public_results"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")
_TIMESTAMP = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class LongitudinalError(ValueError):
    """Raised when longitudinal metadata violates the research contract."""


class LongitudinalIntegrityError(LongitudinalError):
    """Raised when a serialized longitudinal artifact fails integrity checks."""


class TemporalLeakageError(LongitudinalError):
    """Raised when an index input overlaps a same/future target encounter."""


class RoleIsolationError(LongitudinalError):
    """Raised when a patient or source crosses source-cohort roles."""


class SourceRole(StrEnum):
    """Role assigned to an entire source cohort before analysis."""

    DEVELOPMENT = "development"
    CALIBRATION = "calibration"
    PREVIOUSLY_OBSERVED = "previously_observed"
    UNTOUCHED_LOCKBOX = "untouched_lockbox"


class CensoringPolicy(StrEnum):
    """How incomplete follow-up is represented in a constructed cohort."""

    RETAIN_WITH_STATUS = "retain_with_status"
    EXCLUDE_BELOW_MINIMUM = "exclude_below_minimum"
    REQUIRE_COMPLETE_HORIZON = "require_complete_horizon"


class FollowUpStatus(StrEnum):
    """Observation completeness relative to the configured index and horizon."""

    COMPLETE_HORIZON = "complete_horizon"
    RIGHT_CENSORED = "right_censored"
    INSUFFICIENT_FOLLOW_UP = "insufficient_follow_up"


class TargetStatus(StrEnum):
    """Whether one binary future-event label is evaluable."""

    OBSERVED = "observed"
    RIGHT_CENSORED = "right_censored"
    INSUFFICIENT_FOLLOW_UP = "insufficient_follow_up"


@dataclass(frozen=True, slots=True, init=False)
class ECGEncounter:
    """Strict private metadata for one ECG encounter.

    Patient and encounter keys are retained only for internal linkage. Public
    summaries produced by this package never serialize them.
    """

    source_dataset: str
    source_version: str
    source_site: str
    source_role: SourceRole
    patient_key: str
    encounter_key: str
    occurred_at_utc: str
    observed_through_utc: str
    event_labels: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        source_dataset: str,
        source_version: str,
        source_site: str,
        source_role: SourceRole | str,
        patient_key: str,
        encounter_key: str,
        occurred_at_utc: str,
        observed_through_utc: str,
        event_labels: Sequence[str],
    ) -> ECGEncounter:
        dataset = _identifier(source_dataset, "source_dataset")
        version = _identifier(source_version, "source_version")
        site = _identifier(source_site, "source_site")
        patient = _identifier(patient_key, "patient_key")
        encounter = _identifier(encounter_key, "encounter_key")
        try:
            role = SourceRole(source_role)
        except (TypeError, ValueError) as exc:
            raise LongitudinalError("source_role is invalid") from exc
        occurred = canonical_utc_timestamp(occurred_at_utc, "occurred_at_utc")
        observed = canonical_utc_timestamp(observed_through_utc, "observed_through_utc")
        if parse_utc(observed) < parse_utc(occurred):
            raise LongitudinalError("observed_through_utc cannot precede the encounter")
        labels = _labels(event_labels, "event_labels", allow_empty=True)
        instance = object.__new__(cls)
        object.__setattr__(instance, "source_dataset", dataset)
        object.__setattr__(instance, "source_version", version)
        object.__setattr__(instance, "source_site", site)
        object.__setattr__(instance, "source_role", role)
        object.__setattr__(instance, "patient_key", patient)
        object.__setattr__(instance, "encounter_key", encounter)
        object.__setattr__(instance, "occurred_at_utc", occurred)
        object.__setattr__(instance, "observed_through_utc", observed)
        object.__setattr__(instance, "event_labels", labels)
        return instance

    @property
    def patient_identity(self) -> tuple[str, str, str]:
        return (self.source_dataset, self.source_version, self.patient_key)

    @property
    def source_identity(self) -> tuple[str, str, str]:
        return (self.source_dataset, self.source_version, self.source_site)

    @property
    def encounter_identity(self) -> tuple[str, str, str, str, str]:
        return (
            self.source_dataset,
            self.source_version,
            self.source_site,
            self.patient_key,
            self.encounter_key,
        )

    def to_private_dict(self) -> dict[str, object]:
        """Serialize for controlled internal linkage, never for public results."""

        return {
            "source_dataset": self.source_dataset,
            "source_version": self.source_version,
            "source_site": self.source_site,
            "source_role": self.source_role.value,
            "patient_key": self.patient_key,
            "encounter_key": self.encounter_key,
            "occurred_at_utc": self.occurred_at_utc,
            "observed_through_utc": self.observed_through_utc,
            "event_labels": list(self.event_labels),
            "privacy_classification": "private_linkage_metadata",
        }

    @classmethod
    def from_private_dict(cls, payload: Mapping[str, object]) -> ECGEncounter:
        expected = {
            "source_dataset",
            "source_version",
            "source_site",
            "source_role",
            "patient_key",
            "encounter_key",
            "occurred_at_utc",
            "observed_through_utc",
            "event_labels",
            "privacy_classification",
        }
        _exact_keys(payload, expected, "encounter")
        if payload["privacy_classification"] != "private_linkage_metadata":
            raise LongitudinalIntegrityError("invalid encounter privacy classification")
        return cls.create(
            source_dataset=_string(payload["source_dataset"], "source_dataset"),
            source_version=_string(payload["source_version"], "source_version"),
            source_site=_string(payload["source_site"], "source_site"),
            source_role=_string(payload["source_role"], "source_role"),
            patient_key=_string(payload["patient_key"], "patient_key"),
            encounter_key=_string(payload["encounter_key"], "encounter_key"),
            occurred_at_utc=_string(payload["occurred_at_utc"], "occurred_at_utc"),
            observed_through_utc=_string(payload["observed_through_utc"], "observed_through_utc"),
            event_labels=_string_sequence(payload["event_labels"], "event_labels"),
        )


@dataclass(frozen=True, slots=True, init=False)
class TimelineConfig:
    """Hash-bound fixed-cutoff cohort design."""

    index_time_utc: str
    history_window_days: int
    prediction_horizon_days: int
    minimum_follow_up_days: int
    censoring_policy: CensoringPolicy

    @classmethod
    def create(
        cls,
        *,
        index_time_utc: str,
        history_window_days: int,
        prediction_horizon_days: int,
        minimum_follow_up_days: int,
        censoring_policy: CensoringPolicy | str,
    ) -> TimelineConfig:
        index = canonical_utc_timestamp(index_time_utc, "index_time_utc")
        history = _bounded_days(history_window_days, "history_window_days", minimum=1)
        horizon = _bounded_days(prediction_horizon_days, "prediction_horizon_days", minimum=1)
        follow_up = _bounded_days(
            minimum_follow_up_days,
            "minimum_follow_up_days",
            minimum=0,
        )
        if follow_up > horizon:
            raise LongitudinalError("minimum_follow_up_days cannot exceed the horizon")
        try:
            policy = CensoringPolicy(censoring_policy)
        except (TypeError, ValueError) as exc:
            raise LongitudinalError("censoring_policy is invalid") from exc
        instance = object.__new__(cls)
        object.__setattr__(instance, "index_time_utc", index)
        object.__setattr__(instance, "history_window_days", history)
        object.__setattr__(instance, "prediction_horizon_days", horizon)
        object.__setattr__(instance, "minimum_follow_up_days", follow_up)
        object.__setattr__(instance, "censoring_policy", policy)
        return instance

    def _body(self) -> dict[str, object]:
        return {
            "schema_version": TIMELINE_CONFIG_SCHEMA_VERSION,
            "artifact_type": TIMELINE_CONFIG_ARTIFACT_TYPE,
            "index_time_utc": self.index_time_utc,
            "history_window_days": self.history_window_days,
            "prediction_horizon_days": self.prediction_horizon_days,
            "minimum_follow_up_days": self.minimum_follow_up_days,
            "censoring_policy": self.censoring_policy.value,
            "research_use_limit": RESEARCH_USE_LIMIT,
            "target_interpretation": NON_CAUSAL_TARGET_LIMIT,
        }

    @property
    def config_sha256(self) -> str:
        return canonical_sha256(self._body())

    def to_dict(self) -> dict[str, object]:
        return {**self._body(), "config_sha256": self.config_sha256}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TimelineConfig:
        expected = {
            "schema_version",
            "artifact_type",
            "index_time_utc",
            "history_window_days",
            "prediction_horizon_days",
            "minimum_follow_up_days",
            "censoring_policy",
            "research_use_limit",
            "target_interpretation",
            "config_sha256",
        }
        _exact_keys(payload, expected, "timeline config")
        if payload["schema_version"] != TIMELINE_CONFIG_SCHEMA_VERSION:
            raise LongitudinalIntegrityError("unsupported timeline config schema_version")
        if payload["artifact_type"] != TIMELINE_CONFIG_ARTIFACT_TYPE:
            raise LongitudinalIntegrityError("invalid timeline config artifact_type")
        if payload["research_use_limit"] != RESEARCH_USE_LIMIT:
            raise LongitudinalIntegrityError("invalid research-use limitation")
        if payload["target_interpretation"] != NON_CAUSAL_TARGET_LIMIT:
            raise LongitudinalIntegrityError("invalid non-causal target limitation")
        restored = cls.create(
            index_time_utc=_string(payload["index_time_utc"], "index_time_utc"),
            history_window_days=_integer(payload["history_window_days"], "history_window_days"),
            prediction_horizon_days=_integer(
                payload["prediction_horizon_days"], "prediction_horizon_days"
            ),
            minimum_follow_up_days=_integer(
                payload["minimum_follow_up_days"], "minimum_follow_up_days"
            ),
            censoring_policy=_string(payload["censoring_policy"], "censoring_policy"),
        )
        stored_hash = _prefixed_sha256(payload["config_sha256"], "config_sha256")
        if stored_hash != restored.config_sha256:
            raise LongitudinalIntegrityError("timeline config SHA-256 mismatch")
        return restored


@dataclass(frozen=True, slots=True, init=False)
class FutureEventDefinition:
    """One binary future-event target; multiple definitions form multilabel output."""

    target_name: str
    event_any_of: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        target_name: str,
        event_any_of: Sequence[str],
    ) -> FutureEventDefinition:
        name = _identifier(target_name, "target_name")
        events = _labels(event_any_of, "event_any_of", allow_empty=False)
        instance = object.__new__(cls)
        object.__setattr__(instance, "target_name", name)
        object.__setattr__(instance, "event_any_of", events)
        return instance

    def to_dict(self) -> dict[str, object]:
        return {
            "target_name": self.target_name,
            "event_any_of": list(self.event_any_of),
            "interpretation": NON_CAUSAL_TARGET_LIMIT,
        }


@dataclass(frozen=True, slots=True)
class BinaryFutureTarget:
    """One target value, or an explicit reason that it is unavailable."""

    target_name: str
    value: int | None
    status: TargetStatus

    def __post_init__(self) -> None:
        _identifier(self.target_name, "target_name")
        if self.status is TargetStatus.OBSERVED:
            if self.value not in {0, 1} or isinstance(self.value, bool):
                raise LongitudinalError("observed targets require an integer 0 or 1")
        elif self.value is not None:
            raise LongitudinalError("censored targets must have value=null")


def canonical_sha256(payload: Mapping[str, object]) -> str:
    """Return a prefixed SHA-256 over finite canonical JSON."""

    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LongitudinalError("artifact payload must contain finite canonical JSON") from exc
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def canonical_utc_timestamp(value: object, name: str) -> str:
    if not isinstance(value, str) or _TIMESTAMP.fullmatch(value) is None:
        raise LongitudinalError(f"{name} must use YYYY-MM-DDTHH:MM:SSZ")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise LongitudinalError(f"{name} is not a valid UTC timestamp") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise LongitudinalError(f"{name} is not canonical")
    return value


def parse_utc(value: str) -> datetime:
    """Parse a timestamp already validated by this module."""

    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def format_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != datetime.now(UTC).utcoffset():
        raise LongitudinalError("datetime must be UTC-aware")
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise LongitudinalError(f"{name} must be a safe canonical identifier")
    return value


def _labels(value: Sequence[str], name: str, *, allow_empty: bool) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise LongitudinalError(f"{name} must be a sequence of labels")
    labels = tuple(_identifier(item, name) for item in value)
    if not labels and not allow_empty:
        raise LongitudinalError(f"{name} must not be empty")
    if len(set(labels)) != len(labels):
        raise LongitudinalError(f"{name} must contain unique labels")
    if labels != tuple(sorted(labels)):
        raise LongitudinalError(f"{name} must be lexicographically sorted")
    return labels


def _bounded_days(value: object, name: str, *, minimum: int) -> int:
    parsed = _integer(value, name)
    if parsed < minimum or parsed > 36_525:
        raise LongitudinalError(f"{name} must be in [{minimum}, 36525]")
    return parsed


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LongitudinalError(f"{name} must be an integer")
    return value


def _exact_keys(payload: Mapping[str, object], expected: set[str], context: str) -> None:
    if set(payload) != expected:
        raise LongitudinalIntegrityError(
            f"{context} keys differ: missing={sorted(expected - set(payload))}, "
            f"extra={sorted(set(payload) - expected)}"
        )


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise LongitudinalIntegrityError(f"{name} must be text")
    return value


def _string_sequence(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LongitudinalIntegrityError(f"{name} must be a sequence")
    if any(not isinstance(item, str) for item in value):
        raise LongitudinalIntegrityError(f"{name} must contain text")
    return tuple(cast(str, item) for item in value)


def _prefixed_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise LongitudinalIntegrityError(f"{name} must be a prefixed SHA-256")
    return value


def finite_probability(value: object, name: str = "predicted_risk") -> float:
    """Validate a finite probability without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LongitudinalError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise LongitudinalError(f"{name} must be finite and in [0, 1]")
    return parsed
