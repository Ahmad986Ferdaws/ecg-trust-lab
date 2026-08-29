"""Immutable metadata contracts for leakage-safe multi-site ECG research.

This module handles metadata only. It neither downloads nor opens waveforms and
contains no evaluation entry point. Native SNOMED CT codes are preserved in
source order; conversion into shared research labels requires a separate,
hash-bound artifact containing an explicit human review decision for every
native code present in the bound manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import cast

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_ARTIFACT_TYPE = "ecg_trust.multisite_wfdb_manifest"
ONTOLOGY_SCHEMA_VERSION = 1
ONTOLOGY_ARTIFACT_TYPE = "ecg_trust.reviewed_shared_ontology"
MAX_CODES_TEXT_LENGTH = 100_000

SNOMED_CODE_PATTERN = re.compile(r"^[1-9][0-9]{5,17}$")
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
UTC_TIMESTAMP_PATTERN = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")

CANONICAL_LEADS: frozenset[str] = frozenset(
    ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
)

_CHALLENGE_DATASET_TOKENS = frozenset(
    {
        "physionetchallenge2020",
        "physionetchallenge2021",
        "physionetcomputingincardiologychallenge2020",
        "physionetcomputingincardiologychallenge2021",
        "cincchallenge2020",
        "cincchallenge2021",
        "challenge2020",
        "challenge2021",
    }
)
_PTB_SOURCE_TOKENS = frozenset(
    {
        "ptb",
        "ptbxl",
        "ptbandptbxl",
        "ptbptbxl",
        "physikalischtechnischebundesanstalt",
        "physikalischtechnischebundesanstaltptbptbxl",
    }
)


class MultisiteManifestError(ValueError):
    """Raised when multi-site metadata violates the research contract."""


class MultisiteIntegrityError(MultisiteManifestError):
    """Raised when a hash-bound manifest or ontology artifact has changed."""


class RoleIsolationError(MultisiteManifestError):
    """Raised when a patient or source appears in more than one dataset role."""


class LockboxMutationError(MultisiteIntegrityError):
    """Raised when an untouched-lockbox record is added, removed, or changed."""


class OntologyReviewError(MultisiteManifestError):
    """Raised when a shared-label mapping is missing or has not been reviewed."""


class DatasetRole(StrEnum):
    """Permitted roles for an entire source cohort."""

    DEVELOPMENT = "development"
    CALIBRATION = "calibration"
    PREVIOUSLY_OBSERVED = "previously_observed"
    UNTOUCHED_LOCKBOX = "untouched_lockbox"


class MappingAction(StrEnum):
    """Reviewed disposition of one native SNOMED code."""

    MAP = "map"
    EXCLUDE = "exclude"


class ReviewStatus(StrEnum):
    """Review state accepted by an ontology decision record."""

    UNREVIEWED = "unreviewed"
    REVIEWED = "reviewed"


@dataclass(frozen=True, slots=True, init=False)
class ExternalWFDBRecord:
    """One immutable external-cohort WFDB metadata record."""

    source_dataset: str
    source_version: str
    source_site: str
    record_ref: str
    patient_key: str
    sampling_rate_hz: float
    duration_seconds: float
    ordered_leads: tuple[str, ...]
    native_snomed_codes: tuple[str, ...]
    role: DatasetRole

    @classmethod
    def create(
        cls,
        *,
        source_dataset: str,
        source_version: str,
        source_site: str,
        record_ref: str,
        patient_key: str,
        sampling_rate_hz: float,
        duration_seconds: float,
        ordered_leads: Sequence[str],
        native_snomed_codes: Sequence[str],
        role: DatasetRole | str,
    ) -> ExternalWFDBRecord:
        """Validate one record without reading its header or signal file."""

        dataset = _strict_text(source_dataset, "source_dataset")
        version = _strict_text(source_version, "source_version")
        site = _strict_text(source_site, "source_site")
        reference = _record_reference(record_ref)
        patient = _identifier(patient_key, "patient_key")
        rate = _positive_float(sampling_rate_hz, "sampling_rate_hz")
        duration = _positive_float(duration_seconds, "duration_seconds")
        leads = _ordered_leads(ordered_leads)
        codes = _code_sequence(native_snomed_codes)
        resolved_role = _dataset_role(role)

        instance = object.__new__(cls)
        object.__setattr__(instance, "source_dataset", dataset)
        object.__setattr__(instance, "source_version", version)
        object.__setattr__(instance, "source_site", site)
        object.__setattr__(instance, "record_ref", reference)
        object.__setattr__(instance, "patient_key", patient)
        object.__setattr__(instance, "sampling_rate_hz", rate)
        object.__setattr__(instance, "duration_seconds", duration)
        object.__setattr__(instance, "ordered_leads", leads)
        object.__setattr__(instance, "native_snomed_codes", codes)
        object.__setattr__(instance, "role", resolved_role)
        return instance

    @property
    def record_identity(self) -> tuple[str, str, str, str]:
        return (
            self.source_dataset,
            self.source_version,
            self.source_site,
            self.record_ref,
        )

    @property
    def patient_identity(self) -> tuple[str, str, str]:
        """Patient key scoped to the dataset release, not to an individual site."""

        return (self.source_dataset, self.source_version, self.patient_key)

    @property
    def source_identity(self) -> tuple[str, str, str]:
        return (self.source_dataset, self.source_version, self.source_site)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_dataset": self.source_dataset,
            "source_version": self.source_version,
            "source_site": self.source_site,
            "record_ref": self.record_ref,
            "patient_key": self.patient_key,
            "sampling_rate_hz": self.sampling_rate_hz,
            "duration_seconds": self.duration_seconds,
            "ordered_leads": list(self.ordered_leads),
            "native_snomed_codes": list(self.native_snomed_codes),
            "role": self.role.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ExternalWFDBRecord:
        _expect_exact_keys(
            payload,
            {
                "source_dataset",
                "source_version",
                "source_site",
                "record_ref",
                "patient_key",
                "sampling_rate_hz",
                "duration_seconds",
                "ordered_leads",
                "native_snomed_codes",
                "role",
            },
            context="WFDB record",
        )
        return cls.create(
            source_dataset=_string(payload["source_dataset"], "source_dataset"),
            source_version=_string(payload["source_version"], "source_version"),
            source_site=_string(payload["source_site"], "source_site"),
            record_ref=_string(payload["record_ref"], "record_ref"),
            patient_key=_string(payload["patient_key"], "patient_key"),
            sampling_rate_hz=_number(payload["sampling_rate_hz"], "sampling_rate_hz"),
            duration_seconds=_number(payload["duration_seconds"], "duration_seconds"),
            ordered_leads=_string_sequence(payload["ordered_leads"], "ordered_leads"),
            native_snomed_codes=_string_sequence(
                payload["native_snomed_codes"], "native_snomed_codes"
            ),
            role=_string(payload["role"], "role"),
        )


@dataclass(frozen=True, slots=True, init=False)
class MultisiteManifest:
    """Sorted, self-hashed collection of source-native WFDB metadata."""

    records: tuple[ExternalWFDBRecord, ...]
    manifest_sha256: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> MultisiteManifest:
        _expect_exact_keys(
            payload,
            {"schema_version", "artifact_type", "records", "manifest_sha256"},
            context="multisite manifest",
        )
        if payload["schema_version"] != MANIFEST_SCHEMA_VERSION:
            raise MultisiteIntegrityError("unsupported multisite manifest schema_version")
        if payload["artifact_type"] != MANIFEST_ARTIFACT_TYPE:
            raise MultisiteIntegrityError("unexpected multisite manifest artifact_type")
        stored_hash = _sha256(payload["manifest_sha256"], "manifest_sha256")
        rows = _mapping_sequence(payload["records"], "records")
        records = tuple(ExternalWFDBRecord.from_dict(row) for row in rows)
        rebuilt = build_multisite_manifest(records)
        if rebuilt.manifest_sha256 != stored_hash:
            raise MultisiteIntegrityError("multisite manifest SHA-256 mismatch")
        serialized_order = tuple(record.record_identity for record in records)
        canonical_order = tuple(record.record_identity for record in rebuilt.records)
        if serialized_order != canonical_order:
            raise MultisiteIntegrityError("multisite manifest records are not canonically sorted")
        return rebuilt

    def to_dict(self) -> dict[str, object]:
        return {
            **self._payload_without_hash(),
            "manifest_sha256": self.manifest_sha256,
        }

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "artifact_type": MANIFEST_ARTIFACT_TYPE,
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(frozen=True, slots=True, init=False)
class OntologyMappingDecision:
    """Human review decision for one source-native SNOMED code."""

    source_dataset: str
    source_version: str
    native_snomed_code: str
    action: MappingAction
    shared_label: str | None
    status: ReviewStatus
    reviewed_by: str | None
    reviewed_at_utc: str | None
    review_reference: str | None
    rationale: str

    @classmethod
    def create(
        cls,
        *,
        source_dataset: str,
        source_version: str,
        native_snomed_code: str,
        action: MappingAction | str,
        shared_label: str | None,
        status: ReviewStatus | str,
        rationale: str,
        reviewed_by: str | None = None,
        reviewed_at_utc: str | None = None,
        review_reference: str | None = None,
    ) -> OntologyMappingDecision:
        dataset = _strict_text(source_dataset, "source_dataset")
        version = _strict_text(source_version, "source_version")
        code = _single_snomed_code(native_snomed_code)
        resolved_action = _mapping_action(action)
        resolved_status = _review_status(status)
        reason = _strict_text(rationale, "rationale")

        label: str | None
        if resolved_action is MappingAction.MAP:
            if shared_label is None:
                raise OntologyReviewError("map decisions require a shared_label")
            label = _strict_text(shared_label, "shared_label")
        else:
            if shared_label is not None:
                raise OntologyReviewError("exclude decisions must not contain a shared_label")
            label = None

        reviewer: str | None = None
        reviewed_at: str | None = None
        reference: str | None = None
        if resolved_status is ReviewStatus.REVIEWED:
            reviewer = _strict_optional_text(reviewed_by, "reviewed_by", required=True)
            reviewed_at = _review_timestamp(reviewed_at_utc)
            reference = _strict_optional_text(review_reference, "review_reference", required=True)
        elif any(value is not None for value in (reviewed_by, reviewed_at_utc, review_reference)):
            raise OntologyReviewError("unreviewed decisions must not contain reviewer attestation")

        instance = object.__new__(cls)
        object.__setattr__(instance, "source_dataset", dataset)
        object.__setattr__(instance, "source_version", version)
        object.__setattr__(instance, "native_snomed_code", code)
        object.__setattr__(instance, "action", resolved_action)
        object.__setattr__(instance, "shared_label", label)
        object.__setattr__(instance, "status", resolved_status)
        object.__setattr__(instance, "reviewed_by", reviewer)
        object.__setattr__(instance, "reviewed_at_utc", reviewed_at)
        object.__setattr__(instance, "review_reference", reference)
        object.__setattr__(instance, "rationale", reason)
        return instance

    @property
    def mapping_key(self) -> tuple[str, str, str]:
        return (self.source_dataset, self.source_version, self.native_snomed_code)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_dataset": self.source_dataset,
            "source_version": self.source_version,
            "native_snomed_code": self.native_snomed_code,
            "action": self.action.value,
            "shared_label": self.shared_label,
            "status": self.status.value,
            "reviewed_by": self.reviewed_by,
            "reviewed_at_utc": self.reviewed_at_utc,
            "review_reference": self.review_reference,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> OntologyMappingDecision:
        _expect_exact_keys(
            payload,
            {
                "source_dataset",
                "source_version",
                "native_snomed_code",
                "action",
                "shared_label",
                "status",
                "reviewed_by",
                "reviewed_at_utc",
                "review_reference",
                "rationale",
            },
            context="ontology mapping decision",
        )
        return cls.create(
            source_dataset=_string(payload["source_dataset"], "source_dataset"),
            source_version=_string(payload["source_version"], "source_version"),
            native_snomed_code=_string(payload["native_snomed_code"], "native_snomed_code"),
            action=_string(payload["action"], "action"),
            shared_label=_optional_string(payload["shared_label"], "shared_label"),
            status=_string(payload["status"], "status"),
            reviewed_by=_optional_string(payload["reviewed_by"], "reviewed_by"),
            reviewed_at_utc=_optional_string(payload["reviewed_at_utc"], "reviewed_at_utc"),
            review_reference=_optional_string(payload["review_reference"], "review_reference"),
            rationale=_string(payload["rationale"], "rationale"),
        )


@dataclass(frozen=True, slots=True, init=False)
class ReviewedSharedOntology:
    """Manifest-bound ontology containing only fully reviewed decisions."""

    ontology_name: str
    ontology_version: str
    shared_labels: tuple[str, ...]
    manifest_sha256: str
    decisions: tuple[OntologyMappingDecision, ...]
    artifact_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            **self._payload_without_hash(),
            "artifact_sha256": self.artifact_sha256,
        }

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": ONTOLOGY_SCHEMA_VERSION,
            "artifact_type": ONTOLOGY_ARTIFACT_TYPE,
            "ontology_name": self.ontology_name,
            "ontology_version": self.ontology_version,
            "shared_labels": list(self.shared_labels),
            "manifest_sha256": self.manifest_sha256,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "review_contract": "every_manifest_native_code_explicitly_reviewed",
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        manifest: MultisiteManifest,
    ) -> ReviewedSharedOntology:
        _expect_exact_keys(
            payload,
            {
                "schema_version",
                "artifact_type",
                "ontology_name",
                "ontology_version",
                "shared_labels",
                "manifest_sha256",
                "decisions",
                "review_contract",
                "artifact_sha256",
            },
            context="reviewed shared ontology",
        )
        if payload["schema_version"] != ONTOLOGY_SCHEMA_VERSION:
            raise MultisiteIntegrityError("unsupported ontology schema_version")
        if payload["artifact_type"] != ONTOLOGY_ARTIFACT_TYPE:
            raise MultisiteIntegrityError("unexpected ontology artifact_type")
        if payload["review_contract"] != "every_manifest_native_code_explicitly_reviewed":
            raise MultisiteIntegrityError("unsupported ontology review_contract")
        stored_hash = _sha256(payload["artifact_sha256"], "artifact_sha256")
        decisions = tuple(
            OntologyMappingDecision.from_dict(item)
            for item in _mapping_sequence(payload["decisions"], "decisions")
        )
        rebuilt = build_reviewed_shared_ontology(
            manifest,
            ontology_name=_string(payload["ontology_name"], "ontology_name"),
            ontology_version=_string(payload["ontology_version"], "ontology_version"),
            shared_labels=_string_sequence(payload["shared_labels"], "shared_labels"),
            decisions=decisions,
        )
        if _sha256(payload["manifest_sha256"], "manifest_sha256") != manifest.manifest_sha256:
            raise MultisiteIntegrityError("ontology is bound to a different manifest")
        if rebuilt.artifact_sha256 != stored_hash:
            raise MultisiteIntegrityError("reviewed ontology SHA-256 mismatch")
        serialized_order = tuple(decision.mapping_key for decision in decisions)
        rebuilt_order = tuple(decision.mapping_key for decision in rebuilt.decisions)
        if serialized_order != rebuilt_order:
            raise MultisiteIntegrityError("ontology decisions are not canonically sorted")
        return rebuilt

    def labels_for(self, record: ExternalWFDBRecord) -> tuple[str, ...]:
        """Apply only exact, reviewed decisions and preserve shared-label order."""

        decision_lookup = {decision.mapping_key: decision for decision in self.decisions}
        selected: set[str] = set()
        for code in record.native_snomed_codes:
            key = (record.source_dataset, record.source_version, code)
            decision = decision_lookup.get(key)
            if decision is None or decision.status is not ReviewStatus.REVIEWED:
                raise OntologyReviewError(f"no reviewed ontology decision for {key!r}")
            if decision.action is MappingAction.MAP:
                selected.add(cast(str, decision.shared_label))
        return tuple(label for label in self.shared_labels if label in selected)


def parse_native_snomed_codes(value: object) -> tuple[str, ...]:
    """Parse an exact WFDB ``#Dx:`` code list without numeric coercion.

    The source order is preserved. Duplicate identifiers, empty elements,
    leading zeroes, signs, decimal notation, and non-ASCII digits are rejected.
    """

    if not isinstance(value, str):
        raise MultisiteManifestError("native SNOMED codes must be text")
    if len(value) > MAX_CODES_TEXT_LENGTH:
        raise MultisiteManifestError("native SNOMED code text is unreasonably large")
    text = value
    if text.startswith("#Dx:"):
        text = text[4:]
    text = text.strip()
    if not text:
        raise MultisiteManifestError("native SNOMED code list must not be empty")
    raw_codes = text.split(",")
    codes = tuple(part.strip(" \t") for part in raw_codes)
    if any(not code for code in codes):
        raise MultisiteManifestError("native SNOMED code list contains an empty element")
    for code in codes:
        if SNOMED_CODE_PATTERN.fullmatch(code) is None:
            raise MultisiteManifestError(f"invalid native SNOMED CT identifier: {code!r}")
    if len(set(codes)) != len(codes):
        raise MultisiteManifestError("native SNOMED code list contains duplicates")
    return codes


def build_multisite_manifest(records: Iterable[ExternalWFDBRecord]) -> MultisiteManifest:
    """Validate, canonicalize, and self-hash source-native metadata records."""

    materialized = tuple(records)
    if not materialized:
        raise MultisiteManifestError("multisite manifest must contain records")
    if any(not isinstance(record, ExternalWFDBRecord) for record in materialized):
        raise MultisiteManifestError("manifest entries must be ExternalWFDBRecord values")
    canonical = tuple(sorted(materialized, key=_record_sort_key))
    identities = [record.record_identity for record in canonical]
    duplicate = _first_duplicate(identities)
    if duplicate is not None:
        raise MultisiteManifestError(f"duplicate external record identity: {duplicate!r}")
    assert_patient_role_isolation(canonical)
    assert_source_role_isolation(canonical)
    assert_no_physionet_ptb_overlap(canonical)

    payload: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_type": MANIFEST_ARTIFACT_TYPE,
        "records": [record.to_dict() for record in canonical],
    }
    digest = canonical_sha256(payload)
    instance = object.__new__(MultisiteManifest)
    object.__setattr__(instance, "records", canonical)
    object.__setattr__(instance, "manifest_sha256", digest)
    return instance


def assert_patient_role_isolation(records: Iterable[ExternalWFDBRecord]) -> None:
    """Require every dataset-scoped patient key to have exactly one role."""

    assignments: dict[tuple[str, str, str], DatasetRole] = {}
    for record in records:
        prior = assignments.setdefault(record.patient_identity, record.role)
        if prior is not record.role:
            raise RoleIsolationError(
                f"patient {record.patient_identity!r} occurs in roles "
                f"{prior.value!r} and {record.role.value!r}"
            )


def assert_source_role_isolation(records: Iterable[ExternalWFDBRecord]) -> None:
    """Require an entire dataset-version-site cohort to have exactly one role."""

    assignments: dict[tuple[str, str, str], DatasetRole] = {}
    for record in records:
        prior = assignments.setdefault(record.source_identity, record.role)
        if prior is not record.role:
            raise RoleIsolationError(
                f"source {record.source_identity!r} occurs in roles "
                f"{prior.value!r} and {record.role.value!r}"
            )


def assert_no_physionet_ptb_overlap(records: Iterable[ExternalWFDBRecord]) -> None:
    """Reject the PTB/PTB-XL copy embedded in PhysioNet Challenge collections."""

    for record in records:
        dataset_token = _alphanumeric_token(record.source_dataset)
        site_token = _alphanumeric_token(record.source_site)
        is_challenge = dataset_token in _CHALLENGE_DATASET_TOKENS or (
            "physionet" in dataset_token and "challenge" in dataset_token
        )
        embeds_ptb_name = "ptb" in dataset_token and is_challenge
        if is_challenge and (site_token in _PTB_SOURCE_TOKENS or embeds_ptb_name):
            raise MultisiteManifestError(
                "PhysioNet Challenge PTB/PTB-XL source is excluded to prevent "
                "development-cohort overlap"
            )


def assert_lockbox_unchanged(
    frozen: MultisiteManifest,
    candidate: MultisiteManifest,
) -> None:
    """Prove the untouched-lockbox record set and every field remain unchanged."""

    _verify_manifest_hash(frozen)
    _verify_manifest_hash(candidate)
    frozen_rows = {
        record.record_identity: record.to_dict()
        for record in frozen.records
        if record.role is DatasetRole.UNTOUCHED_LOCKBOX
    }
    candidate_rows = {
        record.record_identity: record.to_dict()
        for record in candidate.records
        if record.role is DatasetRole.UNTOUCHED_LOCKBOX
    }
    if frozen_rows != candidate_rows:
        removed = sorted(set(frozen_rows) - set(candidate_rows))
        added = sorted(set(candidate_rows) - set(frozen_rows))
        changed = sorted(
            identity
            for identity in set(frozen_rows) & set(candidate_rows)
            if frozen_rows[identity] != candidate_rows[identity]
        )
        raise LockboxMutationError(
            f"untouched lockbox changed: removed={removed}, added={added}, changed={changed}"
        )


def build_reviewed_shared_ontology(
    manifest: MultisiteManifest,
    *,
    ontology_name: str,
    ontology_version: str,
    shared_labels: Sequence[str],
    decisions: Iterable[OntologyMappingDecision],
) -> ReviewedSharedOntology:
    """Build an exact manifest-bound ontology or refuse incomplete review."""

    _verify_manifest_hash(manifest)
    name = _strict_text(ontology_name, "ontology_name")
    version = _strict_text(ontology_version, "ontology_version")
    labels = _shared_labels(shared_labels)
    materialized = tuple(decisions)
    if not materialized:
        raise OntologyReviewError("ontology must contain mapping decisions")
    if any(not isinstance(item, OntologyMappingDecision) for item in materialized):
        raise OntologyReviewError("ontology decisions have an invalid type")
    if any(item.status is not ReviewStatus.REVIEWED for item in materialized):
        raise OntologyReviewError("shared ontology refuses unreviewed mapping decisions")
    for item in materialized:
        if item.action is MappingAction.MAP and item.shared_label not in labels:
            raise OntologyReviewError(
                f"mapping target {item.shared_label!r} is not a declared shared label"
            )

    canonical = tuple(sorted(materialized, key=lambda item: item.mapping_key))
    keys = [item.mapping_key for item in canonical]
    duplicate = _first_duplicate(keys)
    if duplicate is not None:
        raise OntologyReviewError(f"duplicate ontology decision: {duplicate!r}")
    required_keys = {
        (record.source_dataset, record.source_version, code)
        for record in manifest.records
        for code in record.native_snomed_codes
    }
    supplied_keys = set(keys)
    if supplied_keys != required_keys:
        raise OntologyReviewError(
            "ontology decisions do not exactly cover manifest-native codes: "
            f"missing={sorted(required_keys - supplied_keys)}, "
            f"extra={sorted(supplied_keys - required_keys)}"
        )

    payload: dict[str, object] = {
        "schema_version": ONTOLOGY_SCHEMA_VERSION,
        "artifact_type": ONTOLOGY_ARTIFACT_TYPE,
        "ontology_name": name,
        "ontology_version": version,
        "shared_labels": list(labels),
        "manifest_sha256": manifest.manifest_sha256,
        "decisions": [decision.to_dict() for decision in canonical],
        "review_contract": "every_manifest_native_code_explicitly_reviewed",
    }
    digest = canonical_sha256(payload)
    instance = object.__new__(ReviewedSharedOntology)
    object.__setattr__(instance, "ontology_name", name)
    object.__setattr__(instance, "ontology_version", version)
    object.__setattr__(instance, "shared_labels", labels)
    object.__setattr__(instance, "manifest_sha256", manifest.manifest_sha256)
    object.__setattr__(instance, "decisions", canonical)
    object.__setattr__(instance, "artifact_sha256", digest)
    return instance


def canonical_sha256(payload: Mapping[str, object]) -> str:
    """Return a deterministic prefixed SHA-256 over finite canonical JSON."""

    try:
        serialized = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise MultisiteManifestError("artifact payload must be finite JSON") from error
    return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _verify_manifest_hash(manifest: MultisiteManifest) -> None:
    if not isinstance(manifest, MultisiteManifest):
        raise MultisiteIntegrityError("expected a MultisiteManifest")
    expected = canonical_sha256(manifest._payload_without_hash())
    if manifest.manifest_sha256 != expected:
        raise MultisiteIntegrityError("in-memory multisite manifest SHA-256 mismatch")


def _record_sort_key(record: ExternalWFDBRecord) -> tuple[str, str, str, str, str]:
    return (
        record.source_dataset,
        record.source_version,
        record.source_site,
        record.record_ref,
        record.role.value,
    )


def _first_duplicate(values: Sequence[object]) -> object | None:
    seen: set[object] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def _record_reference(value: object) -> str:
    text = _strict_text(value, "record_ref")
    if "\x00" in text or ":" in text:
        raise MultisiteManifestError("record_ref contains a forbidden character")
    normalized = text.replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(normalized)
    if posix.is_absolute() or windows.is_absolute():
        raise MultisiteManifestError("record_ref must be relative")
    if any(part in {"", ".", ".."} for part in posix.parts):
        raise MultisiteManifestError("record_ref contains unsafe path traversal")
    if posix.suffix.casefold() in {".hea", ".dat"}:
        raise MultisiteManifestError("record_ref must be a WFDB record stem without suffix")
    return posix.as_posix()


def _ordered_leads(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise MultisiteManifestError("ordered_leads must be a sequence")
    leads = tuple(value)
    if not leads:
        raise MultisiteManifestError("ordered_leads must not be empty")
    if any(not isinstance(lead, str) or lead not in CANONICAL_LEADS for lead in leads):
        raise MultisiteManifestError("ordered_leads contains a non-canonical ECG lead")
    if len(set(leads)) != len(leads):
        raise MultisiteManifestError("ordered_leads contains duplicates")
    return leads


def _code_sequence(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise MultisiteManifestError("native_snomed_codes must be parsed into a sequence first")
    codes = tuple(value)
    if not codes:
        raise MultisiteManifestError("native_snomed_codes must not be empty")
    parsed = parse_native_snomed_codes(",".join(codes))
    if parsed != codes:
        raise MultisiteManifestError("native_snomed_codes must already be canonical")
    return codes


def _single_snomed_code(value: object) -> str:
    if not isinstance(value, str):
        raise OntologyReviewError("native_snomed_code must be text")
    try:
        parsed = parse_native_snomed_codes(value)
    except MultisiteManifestError as error:
        raise OntologyReviewError(str(error)) from error
    if len(parsed) != 1:
        raise OntologyReviewError("ontology decisions require exactly one native code")
    return parsed[0]


def _dataset_role(value: DatasetRole | str) -> DatasetRole:
    try:
        return DatasetRole(value)
    except (TypeError, ValueError) as error:
        raise MultisiteManifestError(f"invalid dataset role: {value!r}") from error


def _mapping_action(value: MappingAction | str) -> MappingAction:
    try:
        return MappingAction(value)
    except (TypeError, ValueError) as error:
        raise OntologyReviewError(f"invalid mapping action: {value!r}") from error


def _review_status(value: ReviewStatus | str) -> ReviewStatus:
    try:
        return ReviewStatus(value)
    except (TypeError, ValueError) as error:
        raise OntologyReviewError(f"invalid review status: {value!r}") from error


def _shared_labels(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise OntologyReviewError("shared_labels must be a sequence")
    labels = tuple(_strict_text(item, "shared label") for item in value)
    if not labels:
        raise OntologyReviewError("shared_labels must not be empty")
    if len(set(labels)) != len(labels):
        raise OntologyReviewError("shared_labels must be unique")
    return labels


def _review_timestamp(value: object) -> str:
    if not isinstance(value, str) or UTC_TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise OntologyReviewError("reviewed_at_utc must use YYYY-MM-DDTHH:MM:SSZ")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as error:
        raise OntologyReviewError("reviewed_at_utc is not a valid UTC timestamp") from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise OntologyReviewError("reviewed_at_utc is not canonical")
    return value


def _strict_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MultisiteManifestError(f"{name} must be non-empty canonical text")
    if len(value) > 500 or any(ord(character) < 32 for character in value):
        raise MultisiteManifestError(f"{name} contains invalid text")
    return value


def _strict_optional_text(value: object, name: str, *, required: bool) -> str | None:
    if value is None:
        if required:
            raise OntologyReviewError(f"reviewed decisions require {name}")
        return None
    try:
        return _strict_text(value, name)
    except MultisiteManifestError as error:
        raise OntologyReviewError(str(error)) from error


def _identifier(value: object, name: str) -> str:
    text = _strict_text(value, name)
    if re.fullmatch(r"[A-Za-z0-9._-]+", text) is None:
        raise MultisiteManifestError(f"{name} contains forbidden characters")
    return text


def _positive_float(value: object, name: str) -> float:
    number = _number(value, name)
    if number <= 0.0:
        raise MultisiteManifestError(f"{name} must be positive")
    return number


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MultisiteManifestError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise MultisiteManifestError(f"{name} must be finite")
    return number


def _alphanumeric_token(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _expect_exact_keys(payload: Mapping[str, object], expected: set[str], *, context: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise MultisiteIntegrityError(
            f"{context} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise MultisiteIntegrityError(f"{name} must be text")
    return value


def _optional_string(value: object, name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise MultisiteIntegrityError(f"{name} must be text or null")
    return value


def _string_sequence(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MultisiteIntegrityError(f"{name} must be a sequence")
    if any(not isinstance(item, str) for item in value):
        raise MultisiteIntegrityError(f"{name} must contain text values")
    return tuple(cast(str, item) for item in value)


def _mapping_sequence(value: object, name: str) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MultisiteIntegrityError(f"{name} must be a sequence")
    rows: list[Mapping[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise MultisiteIntegrityError(f"{name} must contain mappings")
        rows.append(cast(Mapping[str, object], item))
    return tuple(rows)


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise MultisiteIntegrityError(f"{name} must be a prefixed lower-case SHA-256")
    return value
