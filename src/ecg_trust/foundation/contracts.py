"""Strict disclosures and evaluation-mode contracts for external ECG encoders."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import cast
from urllib.parse import urlparse

from ecg_trust.constants import LEADS

FOUNDATION_SPEC_SCHEMA_VERSION = 1
FOUNDATION_SPEC_ARTIFACT_TYPE = "ecg_trust.foundation_model_spec"
RESEARCH_USE_LIMIT = "retrospective_research_only_not_for_clinical_decisions"
EXTERNAL_ONLY_LIMIT = "external_checkpoint_evaluation_no_local_foundation_pretraining"
CANONICAL_INPUT_SAMPLES = 1_000
CANONICAL_SAMPLING_FREQUENCY_HZ = 100
CANONICAL_INPUT_UNIT = "millivolts"
CANONICAL_DTYPE = "float32"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+/-]{0,199}$")
_PARAMETER_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.]*$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class FoundationError(ValueError):
    """Raised when a foundation-representation contract is violated."""


class FoundationIntegrityError(FoundationError):
    """Raised when a serialized foundation artifact fails verification."""


class FoundationAdapterError(FoundationError):
    """Raised when an encoder violates its runtime representation contract."""


class TrainabilityError(FoundationError):
    """Raised when actual trainable parameters differ from the declared set."""


class EvaluationMode(StrEnum):
    """Permitted downstream evaluation modes; none performs pretraining."""

    FROZEN_ENCODER = "frozen_encoder"
    LINEAR_PROBE = "linear_probe"
    PARAMETER_EFFICIENT_TUNING = "parameter_efficient_tuning"


class OverlapStatus(StrEnum):
    """Disclosure state for one pretraining/evaluation dataset pair."""

    CONFIRMED = "confirmed"
    POSSIBLE = "possible"
    NONE_KNOWN = "none_known"


@dataclass(frozen=True, slots=True, init=False)
class PretrainingDatasetDisclosure:
    """One dataset family disclosed by the external checkpoint provider."""

    dataset_name: str
    dataset_version: str
    source_url: str
    patient_deduplication_disclosed: bool

    @classmethod
    def create(
        cls,
        *,
        dataset_name: str,
        dataset_version: str,
        source_url: str,
        patient_deduplication_disclosed: bool,
    ) -> PretrainingDatasetDisclosure:
        name = strict_identifier(dataset_name, "dataset_name")
        version = strict_identifier(dataset_version, "dataset_version")
        url = https_url(source_url, "source_url")
        if not isinstance(patient_deduplication_disclosed, bool):
            raise FoundationError("patient_deduplication_disclosed must be boolean")
        instance = object.__new__(cls)
        object.__setattr__(instance, "dataset_name", name)
        object.__setattr__(instance, "dataset_version", version)
        object.__setattr__(instance, "source_url", url)
        object.__setattr__(
            instance,
            "patient_deduplication_disclosed",
            patient_deduplication_disclosed,
        )
        return instance

    @property
    def dataset_identity(self) -> tuple[str, str]:
        return (self.dataset_name, self.dataset_version)

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "source_url": self.source_url,
            "patient_deduplication_disclosed": self.patient_deduplication_disclosed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PretrainingDatasetDisclosure:
        expect_exact_keys(
            payload,
            {
                "dataset_name",
                "dataset_version",
                "source_url",
                "patient_deduplication_disclosed",
            },
            "pretraining dataset disclosure",
        )
        deduplication = payload["patient_deduplication_disclosed"]
        if not isinstance(deduplication, bool):
            raise FoundationIntegrityError("patient_deduplication_disclosed must be boolean")
        return cls.create(
            dataset_name=string(payload["dataset_name"], "dataset_name"),
            dataset_version=string(payload["dataset_version"], "dataset_version"),
            source_url=string(payload["source_url"], "source_url"),
            patient_deduplication_disclosed=deduplication,
        )


@dataclass(frozen=True, slots=True, init=False)
class KnownOverlapDisclosure:
    """Explicit overlap knowledge for one evaluation/pretraining dataset pair."""

    evaluation_dataset_name: str
    evaluation_dataset_version: str
    pretraining_dataset_name: str
    pretraining_dataset_version: str
    status: OverlapStatus
    rationale: str

    @classmethod
    def create(
        cls,
        *,
        evaluation_dataset_name: str,
        evaluation_dataset_version: str,
        pretraining_dataset_name: str,
        pretraining_dataset_version: str,
        status: OverlapStatus | str,
        rationale: str,
    ) -> KnownOverlapDisclosure:
        try:
            parsed_status = OverlapStatus(status)
        except (TypeError, ValueError) as exc:
            raise FoundationError("overlap status is invalid") from exc
        reason = strict_text(rationale, "rationale", maximum=2_000)
        instance = object.__new__(cls)
        object.__setattr__(
            instance,
            "evaluation_dataset_name",
            strict_identifier(evaluation_dataset_name, "evaluation_dataset_name"),
        )
        object.__setattr__(
            instance,
            "evaluation_dataset_version",
            strict_identifier(evaluation_dataset_version, "evaluation_dataset_version"),
        )
        object.__setattr__(
            instance,
            "pretraining_dataset_name",
            strict_identifier(pretraining_dataset_name, "pretraining_dataset_name"),
        )
        object.__setattr__(
            instance,
            "pretraining_dataset_version",
            strict_identifier(pretraining_dataset_version, "pretraining_dataset_version"),
        )
        object.__setattr__(instance, "status", parsed_status)
        object.__setattr__(instance, "rationale", reason)
        return instance

    @property
    def disclosure_key(self) -> tuple[str, str, str, str]:
        return (
            self.evaluation_dataset_name,
            self.evaluation_dataset_version,
            self.pretraining_dataset_name,
            self.pretraining_dataset_version,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "evaluation_dataset_name": self.evaluation_dataset_name,
            "evaluation_dataset_version": self.evaluation_dataset_version,
            "pretraining_dataset_name": self.pretraining_dataset_name,
            "pretraining_dataset_version": self.pretraining_dataset_version,
            "status": self.status.value,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> KnownOverlapDisclosure:
        expect_exact_keys(
            payload,
            {
                "evaluation_dataset_name",
                "evaluation_dataset_version",
                "pretraining_dataset_name",
                "pretraining_dataset_version",
                "status",
                "rationale",
            },
            "known overlap disclosure",
        )
        return cls.create(
            evaluation_dataset_name=string(
                payload["evaluation_dataset_name"], "evaluation_dataset_name"
            ),
            evaluation_dataset_version=string(
                payload["evaluation_dataset_version"], "evaluation_dataset_version"
            ),
            pretraining_dataset_name=string(
                payload["pretraining_dataset_name"], "pretraining_dataset_name"
            ),
            pretraining_dataset_version=string(
                payload["pretraining_dataset_version"], "pretraining_dataset_version"
            ),
            status=string(payload["status"], "status"),
            rationale=string(payload["rationale"], "rationale"),
        )


@dataclass(frozen=True, slots=True, init=False)
class FoundationModelSpec:
    """Self-hashed external checkpoint and representation contract."""

    model_id: str
    architecture: str
    checkpoint_sha256: str
    checkpoint_source_url: str
    checkpoint_revision: str
    license_identifier: str
    license_url: str
    pretraining_datasets: tuple[PretrainingDatasetDisclosure, ...]
    known_overlaps: tuple[KnownOverlapDisclosure, ...]
    embedding_dimension: int

    @classmethod
    def create(
        cls,
        *,
        model_id: str,
        architecture: str,
        checkpoint_sha256: str,
        checkpoint_source_url: str,
        checkpoint_revision: str,
        license_identifier: str,
        license_url: str,
        pretraining_datasets: Sequence[PretrainingDatasetDisclosure],
        known_overlaps: Sequence[KnownOverlapDisclosure],
        embedding_dimension: int,
    ) -> FoundationModelSpec:
        model = strict_identifier(model_id, "model_id")
        architecture_name = strict_identifier(architecture, "architecture")
        checkpoint_hash = prefixed_sha256(checkpoint_sha256, "checkpoint_sha256")
        checkpoint_url = https_url(checkpoint_source_url, "checkpoint_source_url")
        revision = strict_identifier(checkpoint_revision, "checkpoint_revision")
        license_id = strict_identifier(license_identifier, "license_identifier")
        license_source = https_url(license_url, "license_url")
        if (
            isinstance(pretraining_datasets, (str, bytes))
            or not isinstance(pretraining_datasets, Sequence)
            or not pretraining_datasets
        ):
            raise FoundationError("at least one pretraining dataset disclosure is required")
        if any(not isinstance(item, PretrainingDatasetDisclosure) for item in pretraining_datasets):
            raise FoundationError("pretraining_datasets contains an invalid value")
        datasets = tuple(sorted(pretraining_datasets, key=lambda item: item.dataset_identity))
        dataset_ids = [item.dataset_identity for item in datasets]
        if len(set(dataset_ids)) != len(dataset_ids):
            raise FoundationError("pretraining dataset disclosures must be unique")
        if isinstance(known_overlaps, (str, bytes)) or not isinstance(known_overlaps, Sequence):
            raise FoundationError("known_overlaps must be a sequence")
        if any(not isinstance(item, KnownOverlapDisclosure) for item in known_overlaps):
            raise FoundationError("known_overlaps contains an invalid value")
        overlaps = tuple(sorted(known_overlaps, key=lambda item: item.disclosure_key))
        overlap_keys = [item.disclosure_key for item in overlaps]
        if len(set(overlap_keys)) != len(overlap_keys):
            raise FoundationError("known overlap disclosures must be unique")
        declared_pretraining = set(dataset_ids)
        if any(
            (item.pretraining_dataset_name, item.pretraining_dataset_version)
            not in declared_pretraining
            for item in overlaps
        ):
            raise FoundationError("overlap disclosure references undeclared pretraining data")
        if isinstance(embedding_dimension, bool) or not isinstance(embedding_dimension, int):
            raise FoundationError("embedding_dimension must be an integer")
        if not 1 <= embedding_dimension <= 1_000_000:
            raise FoundationError("embedding_dimension must be in [1, 1000000]")

        instance = object.__new__(cls)
        object.__setattr__(instance, "model_id", model)
        object.__setattr__(instance, "architecture", architecture_name)
        object.__setattr__(instance, "checkpoint_sha256", checkpoint_hash)
        object.__setattr__(instance, "checkpoint_source_url", checkpoint_url)
        object.__setattr__(instance, "checkpoint_revision", revision)
        object.__setattr__(instance, "license_identifier", license_id)
        object.__setattr__(instance, "license_url", license_source)
        object.__setattr__(instance, "pretraining_datasets", datasets)
        object.__setattr__(instance, "known_overlaps", overlaps)
        object.__setattr__(instance, "embedding_dimension", embedding_dimension)
        return instance

    def _body(self) -> dict[str, object]:
        return {
            "schema_version": FOUNDATION_SPEC_SCHEMA_VERSION,
            "artifact_type": FOUNDATION_SPEC_ARTIFACT_TYPE,
            "model_id": self.model_id,
            "architecture": self.architecture,
            "checkpoint": {
                "sha256": self.checkpoint_sha256,
                "source_url": self.checkpoint_source_url,
                "revision": self.checkpoint_revision,
                "origin": "external_checkpoint",
            },
            "license": {
                "identifier": self.license_identifier,
                "url": self.license_url,
            },
            "pretraining_datasets": [item.to_dict() for item in self.pretraining_datasets],
            "known_overlaps": [item.to_dict() for item in self.known_overlaps],
            "input_contract": {
                "ordered_leads": list(LEADS),
                "samples": CANONICAL_INPUT_SAMPLES,
                "sampling_frequency_hz": CANONICAL_SAMPLING_FREQUENCY_HZ,
                "unit": CANONICAL_INPUT_UNIT,
                "dtype": CANONICAL_DTYPE,
            },
            "embedding_contract": {
                "dimension": self.embedding_dimension,
                "dtype": CANONICAL_DTYPE,
            },
            "local_pretraining_performed": False,
            "scope_limit": EXTERNAL_ONLY_LIMIT,
            "research_use_limit": RESEARCH_USE_LIMIT,
        }

    @property
    def spec_sha256(self) -> str:
        return canonical_sha256(self._body())

    def to_dict(self) -> dict[str, object]:
        return {**self._body(), "spec_sha256": self.spec_sha256}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> FoundationModelSpec:
        expected = {
            "schema_version",
            "artifact_type",
            "model_id",
            "architecture",
            "checkpoint",
            "license",
            "pretraining_datasets",
            "known_overlaps",
            "input_contract",
            "embedding_contract",
            "local_pretraining_performed",
            "scope_limit",
            "research_use_limit",
            "spec_sha256",
        }
        expect_exact_keys(payload, expected, "foundation model spec")
        if payload["schema_version"] != FOUNDATION_SPEC_SCHEMA_VERSION:
            raise FoundationIntegrityError("unsupported foundation spec schema_version")
        if payload["artifact_type"] != FOUNDATION_SPEC_ARTIFACT_TYPE:
            raise FoundationIntegrityError("invalid foundation spec artifact_type")
        if payload["local_pretraining_performed"] is not False:
            raise FoundationIntegrityError("local foundation pretraining is prohibited")
        if payload["scope_limit"] != EXTERNAL_ONLY_LIMIT:
            raise FoundationIntegrityError("invalid foundation scope limitation")
        if payload["research_use_limit"] != RESEARCH_USE_LIMIT:
            raise FoundationIntegrityError("invalid foundation research-use limitation")
        checkpoint = mapping(payload["checkpoint"], "checkpoint")
        expect_exact_keys(checkpoint, {"sha256", "source_url", "revision", "origin"}, "checkpoint")
        if checkpoint["origin"] != "external_checkpoint":
            raise FoundationIntegrityError("checkpoint origin must be external")
        license_payload = mapping(payload["license"], "license")
        expect_exact_keys(license_payload, {"identifier", "url"}, "license")
        input_contract = mapping(payload["input_contract"], "input_contract")
        expected_input: dict[str, object] = {
            "ordered_leads": list(LEADS),
            "samples": CANONICAL_INPUT_SAMPLES,
            "sampling_frequency_hz": CANONICAL_SAMPLING_FREQUENCY_HZ,
            "unit": CANONICAL_INPUT_UNIT,
            "dtype": CANONICAL_DTYPE,
        }
        if dict(input_contract) != expected_input:
            raise FoundationIntegrityError("foundation input contract is not canonical")
        embedding_contract = mapping(payload["embedding_contract"], "embedding_contract")
        expect_exact_keys(embedding_contract, {"dimension", "dtype"}, "embedding contract")
        if embedding_contract["dtype"] != CANONICAL_DTYPE:
            raise FoundationIntegrityError("embedding dtype must be float32")
        datasets = tuple(
            PretrainingDatasetDisclosure.from_dict(item)
            for item in mapping_sequence(payload["pretraining_datasets"], "pretraining_datasets")
        )
        overlaps = tuple(
            KnownOverlapDisclosure.from_dict(item)
            for item in mapping_sequence(payload["known_overlaps"], "known_overlaps")
        )
        restored = cls.create(
            model_id=string(payload["model_id"], "model_id"),
            architecture=string(payload["architecture"], "architecture"),
            checkpoint_sha256=string(checkpoint["sha256"], "checkpoint.sha256"),
            checkpoint_source_url=string(checkpoint["source_url"], "checkpoint.source_url"),
            checkpoint_revision=string(checkpoint["revision"], "checkpoint.revision"),
            license_identifier=string(license_payload["identifier"], "license.identifier"),
            license_url=string(license_payload["url"], "license.url"),
            pretraining_datasets=datasets,
            known_overlaps=overlaps,
            embedding_dimension=integer(embedding_contract["dimension"], "embedding.dimension"),
        )
        serialized_dataset_order = tuple(item.dataset_identity for item in datasets)
        canonical_dataset_order = tuple(
            item.dataset_identity for item in restored.pretraining_datasets
        )
        serialized_overlap_order = tuple(item.disclosure_key for item in overlaps)
        canonical_overlap_order = tuple(item.disclosure_key for item in restored.known_overlaps)
        if serialized_dataset_order != canonical_dataset_order or (
            serialized_overlap_order != canonical_overlap_order
        ):
            raise FoundationIntegrityError("foundation disclosures are not canonically sorted")
        stored_hash = prefixed_sha256(payload["spec_sha256"], "spec_sha256")
        if stored_hash != restored.spec_sha256:
            raise FoundationIntegrityError("foundation model spec SHA-256 mismatch")
        return restored


@dataclass(frozen=True, slots=True, init=False)
class TrainabilityPolicy:
    """Exact fail-closed set of parameters permitted to receive gradients."""

    mode: EvaluationMode
    allowed_trainable_parameter_names: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        mode: EvaluationMode | str,
        allowed_trainable_parameter_names: Sequence[str],
    ) -> TrainabilityPolicy:
        try:
            parsed_mode = EvaluationMode(mode)
        except (TypeError, ValueError) as exc:
            raise FoundationError("evaluation mode is invalid") from exc
        if isinstance(allowed_trainable_parameter_names, (str, bytes)) or not isinstance(
            allowed_trainable_parameter_names, Sequence
        ):
            raise FoundationError("allowed trainable parameters must be a sequence")
        names = tuple(allowed_trainable_parameter_names)
        if any(
            not isinstance(name, str) or _PARAMETER_NAME.fullmatch(name) is None for name in names
        ):
            raise FoundationError("allowed trainable parameter name is invalid")
        if len(set(names)) != len(names):
            raise FoundationError("allowed trainable parameter names must be unique")
        if names != tuple(sorted(names)):
            raise FoundationError("allowed trainable parameter names must be sorted")
        if parsed_mode is EvaluationMode.FROZEN_ENCODER and names:
            raise FoundationError("frozen_encoder permits no trainable parameters")
        if parsed_mode is not EvaluationMode.FROZEN_ENCODER and not names:
            raise FoundationError("probe and parameter-efficient modes require an explicit set")
        instance = object.__new__(cls)
        object.__setattr__(instance, "mode", parsed_mode)
        object.__setattr__(instance, "allowed_trainable_parameter_names", names)
        return instance

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "allowed_trainable_parameter_names": list(self.allowed_trainable_parameter_names),
            "verification": "exact_set_fail_closed",
            "scope_limit": EXTERNAL_ONLY_LIMIT,
        }


def canonical_sha256(payload: Mapping[str, object]) -> str:
    try:
        serialized = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise FoundationError("artifact must contain finite canonical JSON") from exc
    return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def strict_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise FoundationError(f"{name} must be a safe canonical identifier")
    return value


def strict_text(value: object, name: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise FoundationError(f"{name} must be non-empty canonical text")
    if len(value) > maximum or any(ord(character) < 32 for character in value):
        raise FoundationError(f"{name} contains invalid text")
    return value


def https_url(value: object, name: str) -> str:
    text = strict_text(value, name, maximum=2_000)
    parsed = urlparse(text)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
    ):
        raise FoundationError(f"{name} must be an absolute credential-free HTTPS URL")
    if parsed.fragment:
        raise FoundationError(f"{name} must not contain a URL fragment")
    return text


def prefixed_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FoundationIntegrityError(f"{name} must be a prefixed lower-case SHA-256")
    return value


def expect_exact_keys(payload: Mapping[str, object], expected: set[str], context: str) -> None:
    if set(payload) != expected:
        raise FoundationIntegrityError(
            f"{context} keys differ: missing={sorted(expected - set(payload))}, "
            f"extra={sorted(set(payload) - expected)}"
        )


def string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise FoundationIntegrityError(f"{name} must be text")
    return value


def integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FoundationIntegrityError(f"{name} must be an integer")
    return value


def mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FoundationIntegrityError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise FoundationIntegrityError(f"{name} keys must be text")
    return cast(Mapping[str, object], value)


def mapping_sequence(value: object, name: str) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FoundationIntegrityError(f"{name} must be a sequence")
    return tuple(mapping(item, name) for item in value)
