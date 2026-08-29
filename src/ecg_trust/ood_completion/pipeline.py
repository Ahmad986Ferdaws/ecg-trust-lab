"""Immutable end-to-end execution of Trust Sentinel OOD completion v1."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray
from pydantic import ValidationError

from ecg_trust.constants import TARGET_COLUMNS
from ecg_trust.data.dataset import NormalizationStats, PTBXLDataset
from ecg_trust.data.manifest import ManifestError, parse_sha256sums
from ecg_trust.demo_backend import DemoInferenceBackend
from ecg_trust.foundation.adapter import model_state_sha256
from ecg_trust.ood_completion.cohorts import (
    CohortCounts,
    OODCohorts,
    OODExpectedCohortCounts,
    OrderedCohort,
    load_ood_cohorts,
)
from ecg_trust.ood_completion.embedding_artifact import (
    EmbeddingArtifact,
    EmbeddingArtifactError,
    EmbeddingRole,
    create_embedding_artifact,
    load_embedding_artifact,
    save_embedding_artifact,
)
from ecg_trust.ood_completion.models import (
    DISTRIBUTION_POLICY_FILENAME,
    MAX_DISTRIBUTION_POLICY_BYTES,
    MAX_OOD_COMPLETION_RESULT_BYTES,
    MAX_OOD_COMPLETION_SUCCESS_BYTES,
    OOD_COMPLETION_FAILURE_FILENAME,
    OOD_COMPLETION_RESULT_FILENAME,
    OOD_COMPLETION_SUCCESS_FILENAME,
    DistributionPolicy,
    DistributionPolicyBinding,
    OODBundleMember,
    OODBundleRelativePath,
    OODClaimBoundary,
    OODCompletionFailureCode,
    OODCompletionFailureReceiptBody,
    OODCompletionIntegrityError,
    OODCompletionResult,
    OODCompletionResultBody,
    OODCompletionSuccessManifest,
    OODCompletionSuccessManifestBody,
    OODIntegritySummary,
    OODLineageProvenance,
    OODPositiveEvaluationSummary,
    ReferenceAndThresholdExecutionSummary,
    ReferenceEmbeddingExecutionSummary,
    SourceOODValidationSummary,
    ThresholdEmbeddingExecutionSummary,
    _assert_result_source_support_eligible,
    canonical_json_bytes,
    distribution_policy_json_bytes,
    load_distribution_policy_bytes,
    load_ood_completion_result_bytes,
    load_ood_completion_success_bytes,
    ood_completion_failure_json_bytes,
    ood_completion_result_json_bytes,
    ood_completion_success_json_bytes,
    seal_ood_completion_failure_receipt,
    seal_ood_completion_result,
    seal_ood_completion_success_manifest,
)
from ecg_trust.ood_completion.runtime import (
    DeterministicCUDARuntime,
    OODDeterminismError,
    OODRuntimeError,
    configure_deterministic_cuda,
    extract_embeddings_twice,
    prepare_resnet_for_embedding,
)
from ecg_trust.ood_completion.statistics import (
    evaluate_source_validation,
    fit_distribution_policy,
)
from ecg_trust.ood_completion.waveform_inventory import (
    OfficialWaveformSubset,
    build_official_waveform_subset,
    verify_official_waveform_subset,
)
from ecg_trust.protocol import ExperimentProtocol
from ecg_trust.refit_runner import FrozenRefitError, load_refit_completion
from ecg_trust.source_calibration import (
    SourceCalibrationConfig,
    SourceCalibrationResult,
    VerifiedSourceInputs,
    load_source_calibration_config,
    load_source_calibration_result_bytes,
    patient_split_role,
    verify_clean_git_revision,
    verify_source_inputs,
)
from ecg_trust.source_calibration.models import canonical_sha256 as source_canonical_sha256

Int64Array = NDArray[np.int64]
Int8Array = NDArray[np.int8]

_EXPECTED_CONFIG_FILE_SHA256 = (
    "sha256:5d12a71e8cd11350580a6d88b3656ca416392bedd3209d558ba116a90d536070"
)
_CONFIG_MAX_BYTES = 1_000_000
_BOUND_FILE_MAX_BYTES = 1_000_000_000
_SOURCE_IDENTITY_NPZ_MAX_BYTES = 256_000_000
_SOURCE_CALIBRATION_RESULT_MAX_BYTES = 2_000_000
_VALIDATION_ACCESS_RECORD_MAX_BYTES = 4096
_VALIDATION_ACCESS_MARKER_FILENAME = "source-validation-access-armed.json"
_VALIDATION_ACCESS_CLAIM_SUFFIX = ".source-validation-one-shot-claim.json"
_BUNDLE_MEMBER_PATHS = (
    "distribution-policy.json",
    "ood-completion-result.json",
    "private/reference-embeddings.json",
    "private/reference-embeddings.npz",
    "private/source-validation-embeddings.json",
    "private/source-validation-embeddings.npz",
    "private/threshold-fit-embeddings.json",
    "private/threshold-fit-embeddings.npz",
    "source-validation-access-armed.json",
)
_GIT_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_OWNER_NONCE = re.compile(r"[0-9a-f]{64}\Z")
_BINDING_NAMES = (
    "source_calibration_result",
    "source_calibration_config",
    "dataset_manifest",
    "official_dataset_checksums",
    "refit_completion",
    "checkpoint",
    "resolved_config",
    "normalization",
    "historical_demo_policy",
    "demo_binding",
    "experiment_protocol",
    "dependency_lock",
    "project_manifest",
)


class OODCompletionConfigError(ValueError):
    """Raised when the exact frozen OOD configuration cannot be loaded."""


class OODCompletionExecutionError(RuntimeError):
    """Raised when an immutable OOD execution or output commit fails."""


class _OODOutputCommitError(OODCompletionExecutionError):
    """Report whether an output rename completed before its durability check failed."""

    def __init__(self, message: str, *, output_root_committed: bool) -> None:
        super().__init__(message)
        self.output_root_committed = output_root_committed


class _OODFitError(RuntimeError):
    """Internal stage marker for sanitized FIT_FAILED receipts."""


class _OODValidationError(RuntimeError):
    """Internal stage marker for sanitized VALIDATION_FAILED receipts."""


@dataclass(frozen=True, slots=True)
class BoundProjectFile:
    name: str
    relative_path: str
    file_sha256: str


@dataclass(frozen=True, slots=True)
class OODCompletionConfig:
    """Only execution-relevant values decoded from the byte-frozen YAML."""

    path: Path
    file_sha256: str
    frozen_at_utc: datetime
    bindings: Mapping[str, BoundProjectFile]
    resolved_protocol_path: str
    resolved_protocol_file_sha256: str
    source_calibration_artifact_sha256: str
    refit_completion_artifact_sha256: str
    resolved_config_sha256: str
    experiment_protocol_sha256: str
    reference_identity_sha256: str
    fold9_identity_sha256: str
    expected_counts: OODExpectedCohortCounts
    patient_split_salt: str
    selected_record_count: int
    selected_file_count: int
    output_root: str
    private_npz_paths: Mapping[EmbeddingRole, str]
    expected_device_name: str
    expected_compute_capability: tuple[int, int]
    expected_python_version: str
    expected_torch_version: str
    expected_cuda_runtime: str
    expected_cudnn_version: int
    expected_nvidia_driver_version: str


@dataclass(frozen=True, slots=True)
class VerifiedOODInputs:
    """Hash-verified inputs, decoded only after their byte identities match."""

    project_root: Path
    paths: Mapping[str, Path]
    experiment_protocol: ExperimentProtocol
    official_checksums: Mapping[str, str]

    @property
    def dataset_root(self) -> Path:
        return self.paths["official_dataset_checksums"].parent


@dataclass(frozen=True, slots=True)
class SavedRoleEmbeddings:
    artifact: EmbeddingArtifact
    repeated_embedding_tensor_sha256: str


@dataclass(frozen=True, slots=True)
class _PostSealSourceArtifacts:
    config: SourceCalibrationConfig
    result: SourceCalibrationResult
    inputs: VerifiedSourceInputs
    assignment_sha256: str


@dataclass(frozen=True, slots=True)
class _StagedOODExecution:
    result: OODCompletionResult
    source_artifacts: _PostSealSourceArtifacts


@dataclass(frozen=True, slots=True)
class SealedDistributionPolicy:
    """Proof that the fitted policy was committed and strictly reloaded before C."""

    policy: DistributionPolicy
    path: Path
    file_sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.policy, DistributionPolicy):
            raise TypeError("policy must be a DistributionPolicy")
        snapshot = _read_bounded_file_snapshot(
            self.path,
            maximum_bytes=MAX_DISTRIBUTION_POLICY_BYTES,
            context="sealed distribution policy",
        )
        if _sha256_bytes(snapshot) != self.file_sha256:
            raise OODCompletionIntegrityError("sealed distribution policy file hash differs")
        if len(snapshot) != self.size_bytes or self.size_bytes <= 0:
            raise OODCompletionIntegrityError("sealed distribution policy size differs")


@dataclass(frozen=True, slots=True, init=False)
class VerifiedOODCompletionBundle:
    """Whole-root proof required by any downstream research-bundle consumer."""

    result: OODCompletionResult
    policy: DistributionPolicy
    success_manifest: OODCompletionSuccessManifest

    @classmethod
    def _create(
        cls,
        *,
        result: OODCompletionResult,
        policy: DistributionPolicy,
        success_manifest: OODCompletionSuccessManifest,
    ) -> VerifiedOODCompletionBundle:
        instance = object.__new__(cls)
        object.__setattr__(instance, "result", result)
        object.__setattr__(instance, "policy", policy)
        object.__setattr__(instance, "success_manifest", success_manifest)
        return instance


@dataclass(frozen=True, slots=True)
class _ExpectedEmbeddingBinding:
    relative_npz_path: str
    role: EmbeddingRole
    folds: tuple[int, ...]
    summary: (
        ReferenceEmbeddingExecutionSummary
        | ThresholdEmbeddingExecutionSummary
        | SourceOODValidationSummary
    )


@dataclass(slots=True)
class _OneShotClaimState:
    owner_nonce: str
    _published_by_this_process: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if _OWNER_NONCE.fullmatch(self.owner_nonce) is None:
            raise ValueError("one-shot claim owner nonce must be 32-byte lowercase hex")

    @property
    def claim_bytes(self) -> bytes:
        return _validation_access_claim_bytes(self.owner_nonce)

    @property
    def published_by_this_process(self) -> bool:
        return self._published_by_this_process

    def _mark_published(self) -> None:
        self._published_by_this_process = True

    def is_valid_other_owner(self, claim_path: Path) -> bool:
        try:
            observed = _read_bounded_file_snapshot(
                claim_path,
                maximum_bytes=_VALIDATION_ACCESS_RECORD_MAX_BYTES,
                context="source-validation claim",
            )
            return _validate_validation_access_claim_bytes(observed) != self.owner_nonce
        except (OSError, OODCompletionIntegrityError):
            return False


def load_ood_completion_config(path: str | Path) -> OODCompletionConfig:
    """Load the exact preregistered YAML, rejecting any byte or schema drift."""

    source = Path(os.path.abspath(os.fspath(path)))
    try:
        _assert_no_reparse_components(source)
        if not source.is_file():
            raise OODCompletionConfigError("OOD configuration is missing or symbolic")
        raw = source.read_bytes()
    except OODCompletionConfigError:
        raise
    except OSError as error:
        raise OODCompletionConfigError("OOD configuration cannot be read") from error
    if not 0 < len(raw) <= _CONFIG_MAX_BYTES:
        raise OODCompletionConfigError("OOD configuration size is invalid")
    file_sha256 = _sha256_bytes(raw)
    if file_sha256 != _EXPECTED_CONFIG_FILE_SHA256:
        raise OODCompletionConfigError("OOD configuration differs from the frozen v1 bytes")
    try:
        text = raw.decode("utf-8")
        _reject_duplicate_yaml_keys(text)
        decoded: object = yaml.safe_load(text)
    except (UnicodeError, yaml.YAMLError) as error:
        raise OODCompletionConfigError("OOD configuration is not valid UTF-8 YAML") from error
    root = _mapping(decoded, "OOD configuration")
    if root.get("schema_version") != 1 or root.get("protocol_id") != (
        "trust-sentinel-ood-completion-v1"
    ):
        raise OODCompletionConfigError("OOD configuration identity is invalid")
    if root.get("status") != "frozen_pre_execution" or root.get("research_only") is not True:
        raise OODCompletionConfigError("OOD configuration is not a frozen research protocol")
    timestamp = root.get("frozen_at_utc")
    if not isinstance(timestamp, datetime) or timestamp.utcoffset() != UTC.utcoffset(timestamp):
        raise OODCompletionConfigError("OOD frozen_at_utc must be timezone-aware UTC")

    bindings_payload = _mapping(root.get("bindings"), "bindings")
    if set(bindings_payload) != set(_BINDING_NAMES):
        raise OODCompletionConfigError("OOD binding inventory is not exact")
    bindings: dict[str, BoundProjectFile] = {}
    for name in _BINDING_NAMES:
        item = _mapping(bindings_payload.get(name), f"bindings.{name}")
        bindings[name] = BoundProjectFile(
            name=name,
            relative_path=_relative_path(item.get("path"), f"bindings.{name}.path"),
            file_sha256=_prefixed_sha256(
                item.get("file_sha256"),
                f"bindings.{name}.file_sha256",
            ),
        )

    experiment_binding = _mapping(
        bindings_payload.get("experiment_protocol"),
        "bindings.experiment_protocol",
    )
    source_binding = _mapping(
        bindings_payload.get("source_calibration_result"),
        "bindings.source_calibration_result",
    )
    refit_binding = _mapping(bindings_payload.get("refit_completion"), "bindings.refit_completion")
    resolved_binding = _mapping(bindings_payload.get("resolved_config"), "bindings.resolved_config")
    identity = _mapping(root.get("cohort_identity"), "cohort_identity")
    roles = _mapping(root.get("roles"), "roles")
    threshold_role = _mapping(roles.get("threshold_fit"), "roles.threshold_fit")
    validation_role = _mapping(roles.get("source_validation"), "roles.source_validation")
    patient_split_salt = _exact_string(
        threshold_role.get("patient_hash_salt"),
        "trust-sentinel-v1",
        "roles.threshold_fit.patient_hash_salt",
    )
    if (
        _exact_string(
            validation_role.get("patient_hash_salt"),
            "trust-sentinel-v1",
            "roles.source_validation.patient_hash_salt",
        )
        != patient_split_salt
    ):
        raise OODCompletionConfigError("source role patient split salts differ")
    expected = OODExpectedCohortCounts(
        reference=_role_counts(roles, "reference"),
        decision_fit=CohortCounts(records=847, patients=751),
        threshold_fit=_role_counts(roles, "threshold_fit"),
        source_validation=_role_counts(roles, "source_validation"),
        full_fold9=CohortCounts(records=2146, patients=1917),
    )
    checksum_subset = _mapping(root.get("official_checksum_subset"), "official_checksum_subset")
    artifacts = _mapping(root.get("artifacts"), "artifacts")
    private = _mapping(artifacts.get("private_embeddings"), "artifacts.private_embeddings")
    private_roles = _mapping(private.get("roles"), "artifacts.private_embeddings.roles")
    runtime = _mapping(root.get("runtime"), "runtime")
    capability = runtime.get("compute_capability")
    if not isinstance(capability, list) or capability != [12, 0]:
        raise OODCompletionConfigError("runtime compute capability must be [12, 0]")

    return OODCompletionConfig(
        path=source.resolve(strict=True),
        file_sha256=file_sha256,
        frozen_at_utc=timestamp,
        bindings=MappingProxyType(bindings),
        resolved_protocol_path=_relative_path(
            experiment_binding.get("resolved_path"),
            "bindings.experiment_protocol.resolved_path",
        ),
        resolved_protocol_file_sha256=_prefixed_sha256(
            experiment_binding.get("resolved_file_sha256"),
            "bindings.experiment_protocol.resolved_file_sha256",
        ),
        source_calibration_artifact_sha256=_prefixed_sha256(
            source_binding.get("artifact_sha256"),
            "bindings.source_calibration_result.artifact_sha256",
        ),
        refit_completion_artifact_sha256=_prefixed_sha256(
            refit_binding.get("artifact_sha256"),
            "bindings.refit_completion.artifact_sha256",
        ),
        resolved_config_sha256=_prefixed_sha256(
            resolved_binding.get("inner_config_sha256"),
            "bindings.resolved_config.inner_config_sha256",
        ),
        experiment_protocol_sha256=_prefixed_sha256(
            experiment_binding.get("protocol_sha256"),
            "bindings.experiment_protocol.protocol_sha256",
        ),
        reference_identity_sha256=_prefixed_sha256(
            identity.get("reference_sha256"),
            "cohort_identity.reference_sha256",
        ),
        fold9_identity_sha256=_prefixed_sha256(
            identity.get("fold9_sha256"),
            "cohort_identity.fold9_sha256",
        ),
        expected_counts=expected,
        patient_split_salt=patient_split_salt,
        selected_record_count=_exact_integer(
            checksum_subset.get("expected_selected_records"),
            18_383,
            "official_checksum_subset.expected_selected_records",
        ),
        selected_file_count=_exact_integer(
            checksum_subset.get("expected_selected_files"),
            36_766,
            "official_checksum_subset.expected_selected_files",
        ),
        output_root=_relative_path(artifacts.get("output_root"), "artifacts.output_root"),
        private_npz_paths=MappingProxyType(
            {
                role: _private_npz_path(private_roles, role)
                for role in EmbeddingRole
            }
        ),
        expected_device_name=_exact_string(
            runtime.get("device_name"),
            "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
            "runtime.device_name",
        ),
        expected_compute_capability=(12, 0),
        expected_python_version=_exact_string(
            runtime.get("python_version"), "3.12.13", "runtime.python_version"
        ),
        expected_torch_version=_exact_string(
            runtime.get("torch_version"), "2.13.0+cu130", "runtime.torch_version"
        ),
        expected_cuda_runtime=_exact_string(
            runtime.get("cuda_runtime"), "13.0", "runtime.cuda_runtime"
        ),
        expected_cudnn_version=_exact_integer(
            runtime.get("cudnn_version_api_integer"),
            92_000,
            "runtime.cudnn_version_api_integer",
        ),
        expected_nvidia_driver_version=_exact_string(
            runtime.get("nvidia_driver_version"),
            "596.49",
            "runtime.nvidia_driver_version",
        ),
    )


def verify_ood_inputs(
    config: OODCompletionConfig,
    *,
    project_root: str | Path,
) -> VerifiedOODInputs:
    """Hash every declared source before decoding any bound scientific file."""

    if not isinstance(config, OODCompletionConfig):
        raise TypeError("config must be an OODCompletionConfig")
    lexical_root = Path(os.path.abspath(os.fspath(project_root)))
    _assert_no_reparse_components(lexical_root)
    root = lexical_root.resolve(strict=True)
    paths: dict[str, Path] = {}
    for name, binding in config.bindings.items():
        path = _resolve_project_path(root, binding.relative_path, require_file=True)
        if path.is_symlink():
            raise OODCompletionIntegrityError("bound project inputs must not be symbolic")
        try:
            size = path.stat().st_size
        except OSError as error:
            raise OODCompletionIntegrityError("bound project input is unavailable") from error
        if not 0 < size <= _BOUND_FILE_MAX_BYTES:
            raise OODCompletionIntegrityError("bound project input size is invalid")
        if _sha256_file(path) != binding.file_sha256:
            raise OODCompletionIntegrityError(f"{name} file hash differs from the protocol")
        paths[name] = path
    resolved_protocol = _resolve_project_path(
        root,
        config.resolved_protocol_path,
        require_file=True,
    )
    if resolved_protocol.is_symlink() or _sha256_file(resolved_protocol) != (
        config.resolved_protocol_file_sha256
    ):
        raise OODCompletionIntegrityError("resolved experiment protocol hash differs")
    paths["resolved_experiment_protocol"] = resolved_protocol

    protocol = ExperimentProtocol.canonical()
    if protocol.protocol_hash != config.experiment_protocol_sha256:
        raise OODCompletionIntegrityError("canonical experiment protocol hash differs")
    try:
        completion = load_refit_completion(
            paths["refit_completion"],
            protocol=protocol,
            verify_sources=True,
        )
    except FrozenRefitError as error:
        raise OODCompletionIntegrityError("refit completion lineage verification failed") from error
    if completion.get("artifact_sha256") != config.refit_completion_artifact_sha256:
        raise OODCompletionIntegrityError("refit completion logical identity differs")

    try:
        checksum_text = paths["official_dataset_checksums"].read_text(encoding="utf-8")
        official = parse_sha256sums(checksum_text)
    except (OSError, UnicodeError, ManifestError) as error:
        raise OODCompletionIntegrityError(
            "official checksum inventory cannot be decoded"
        ) from error
    return VerifiedOODInputs(
        project_root=root,
        paths=MappingProxyType(paths),
        experiment_protocol=protocol,
        official_checksums=MappingProxyType(dict(official)),
    )


def prepare_ood_completion(
    *,
    config_path: str | Path,
    project_root: str | Path,
    code_revision: str,
) -> OODCompletionResult:
    """Execute once into a new output root and retain unfavorable evidence."""

    _validate_code_revision(code_revision)
    config = load_ood_completion_config(config_path)
    inputs = verify_ood_inputs(config, project_root=project_root)
    _assert_clean_code_revision(inputs.project_root, code_revision)
    output_root = _resolve_project_path(
        inputs.project_root,
        config.output_root,
        require_file=False,
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists() or output_root.is_symlink():
        raise OODCompletionExecutionError("immutable OOD output root already exists")
    validation_claim_path = _validation_access_claim_path(output_root)
    _assert_validation_access_unclaimed(validation_claim_path)
    _assert_no_marked_staging_retry(output_root)
    try:
        raw_staging = tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging-",
            dir=output_root.parent,
        )
    except OSError as error:
        raise OODCompletionExecutionError("could not create OOD staging root") from error
    staging = Path(raw_staging).resolve(strict=True)
    validation_claim_state = _OneShotClaimState(owner_nonce=secrets.token_hex(32))
    committed = False
    try:
        staged_execution = _execute_staged(
            config=config,
            inputs=inputs,
            staging_root=staging,
            code_revision=code_revision,
            validation_claim_path=validation_claim_path,
            validation_claim_state=validation_claim_state,
        )
        _commit_staged_directory(staging, output_root)
        committed = True
        _assert_bound_inputs_unchanged(
            config,
            inputs,
            source_artifacts=staged_execution.source_artifacts,
        )
        _assert_clean_code_revision(inputs.project_root, code_revision)
        return _finalize_success_bundle(
            output_root=output_root,
            config=config,
            code_revision=code_revision,
            expected_result_artifact_sha256=staged_execution.result.artifact_sha256,
        )
    except BaseException as error:
        if isinstance(error, _OODOutputCommitError) and error.output_root_committed:
            committed = True
        receipt = seal_ood_completion_failure_receipt(
            OODCompletionFailureReceiptBody(
                schema_version=1,
                artifact_type="ecg_trust.ood_completion_failure",
                protocol_id="trust-sentinel-ood-completion-v1",
                status="FAILED",
                frozen_at_utc=config.frozen_at_utc,
                config_file_sha256=config.file_sha256,
                code_revision=code_revision,
                failure_code=_failure_code(error),
                contains_raw_ids_or_rows=False,
                contains_embeddings=False,
                contains_filesystem_paths=False,
                retry_requires_new_output_root=True,
            )
        )
        receipt_bytes = ood_completion_failure_json_bytes(receipt)
        if committed:
            try:
                _atomic_write_new(
                    output_root / OOD_COMPLETION_FAILURE_FILENAME,
                    receipt_bytes,
                )
            except Exception as committed_receipt_error:
                raise OODCompletionExecutionError(
                    "OOD execution failed and its sanitized receipt could not be committed"
                ) from committed_receipt_error
        elif validation_claim_state.published_by_this_process:
            _retain_owned_post_claim_failure(
                staging_root=staging,
                output_root=output_root,
                claim_state=validation_claim_state,
                receipt_bytes=receipt_bytes,
            )
        elif validation_claim_state.is_valid_other_owner(validation_claim_path):
            # A different nonce owns the fixed claim. This contender never
            # crossed the C boundary, so its pre-claim armed staging is safe to
            # remove and must never race the winner into the output root.
            _remove_staging_root(staging, expected_parent=output_root.parent)
        elif validation_claim_path.exists() or validation_claim_path.is_symlink():
            # A present but unreadable or invalid claim cannot safely be
            # attributed. Preserve the armed staging and forbid automatic
            # recovery rather than risk deleting post-boundary evidence.
            pass
        else:
            # An armed marker without a published claim is crash-conservative
            # evidence. Never delete it automatically; the frozen protocol
            # requires manual forensic review before any new protocol run.
            if not _validation_access_armed(staging):
                _remove_staging_root(staging, expected_parent=output_root.parent)
        raise


def verify_ood_completion_bundle(output_root: str | Path) -> VerifiedOODCompletionBundle:
    """Verify the complete immutable output root before any downstream use."""

    root = _strict_existing_bundle_root(output_root)
    _assert_exact_bundle_tree(root, include_success_manifest=True)
    success_path = root / OOD_COMPLETION_SUCCESS_FILENAME
    success_bytes = _read_bounded_file_snapshot(
        success_path,
        maximum_bytes=MAX_OOD_COMPLETION_SUCCESS_BYTES,
        context="OOD completion success manifest",
    )
    manifest = load_ood_completion_success_bytes(success_bytes)
    if manifest.config_file_sha256 != _EXPECTED_CONFIG_FILE_SHA256:
        raise OODCompletionIntegrityError("success manifest binds a different frozen config")
    result, policy, members, claim_sha256 = _verify_committed_evidence(
        output_root=root,
        expected_result_artifact_sha256=manifest.result_artifact_sha256,
        expected_config_file_sha256=manifest.config_file_sha256,
        expected_code_revision=manifest.code_revision,
        include_success_manifest=True,
    )
    if manifest.distribution_policy_artifact_sha256 != policy.artifact_sha256:
        raise OODCompletionIntegrityError("success manifest policy identity differs")
    if manifest.validation_access_claim_file_sha256 != claim_sha256:
        raise OODCompletionIntegrityError("success manifest one-shot claim hash differs")
    if manifest.members != members:
        raise OODCompletionIntegrityError("success manifest inventory differs from output files")
    return VerifiedOODCompletionBundle._create(
        result=result,
        policy=policy,
        success_manifest=manifest,
    )


def verify_research_bundle_eligible(
    output_root: str | Path,
) -> VerifiedOODCompletionBundle:
    """Return only a whole-root-verified, source-support-eligible bundle."""

    bundle = verify_ood_completion_bundle(output_root)
    _assert_result_source_support_eligible(bundle.result)
    return bundle


def _finalize_success_bundle(
    *,
    output_root: Path,
    config: OODCompletionConfig,
    code_revision: str,
    expected_result_artifact_sha256: str,
) -> OODCompletionResult:
    result, policy, members, claim_sha256 = _verify_committed_evidence(
        output_root=output_root,
        expected_result_artifact_sha256=expected_result_artifact_sha256,
        expected_config_file_sha256=config.file_sha256,
        expected_code_revision=code_revision,
        include_success_manifest=False,
    )
    manifest = seal_ood_completion_success_manifest(
        OODCompletionSuccessManifestBody(
            schema_version=1,
            artifact_type="ecg_trust.ood_completion_success_manifest",
            protocol_id="trust-sentinel-ood-completion-v1",
            status="SUCCESS",
            frozen_at_utc=config.frozen_at_utc,
            config_file_sha256=config.file_sha256,
            code_revision=code_revision,
            result_artifact_sha256=result.artifact_sha256,
            distribution_policy_artifact_sha256=policy.artifact_sha256,
            validation_access_claim_filename=(
                ".ood_completion_v1.source-validation-one-shot-claim.json"
            ),
            validation_access_claim_file_sha256=claim_sha256,
            member_count=9,
            members=members,
            terminal_checks_complete=True,
            failure_receipt_present=False,
        )
    )
    # This atomic create-new link is deliberately the final successful write.
    # No fallible validation or mutation may follow it on the success path.
    _atomic_write_terminal_success(
        output_root,
        ood_completion_success_json_bytes(manifest),
    )
    return result


def _verify_committed_evidence(
    *,
    output_root: Path,
    expected_result_artifact_sha256: str,
    expected_config_file_sha256: str,
    expected_code_revision: str,
    include_success_manifest: bool,
) -> tuple[
    OODCompletionResult,
    DistributionPolicy,
    tuple[OODBundleMember, ...],
    str,
]:
    _assert_exact_bundle_tree(
        output_root,
        include_success_manifest=include_success_manifest,
    )
    claim_sha256, marker_bytes = _verify_validation_access_claim_and_marker(output_root)

    result_path = output_root / OOD_COMPLETION_RESULT_FILENAME
    result_bytes = _read_bounded_file_snapshot(
        result_path,
        maximum_bytes=MAX_OOD_COMPLETION_RESULT_BYTES,
        context="OOD completion result",
    )
    result = load_ood_completion_result_bytes(result_bytes)
    if result.artifact_sha256 != expected_result_artifact_sha256:
        raise OODCompletionIntegrityError("committed OOD result identity changed")
    if (
        result.provenance.ood_config_file_sha256 != expected_config_file_sha256
        or result.provenance.code_revision != expected_code_revision
    ):
        raise OODCompletionIntegrityError("committed OOD result lineage differs")

    policy_path = output_root / DISTRIBUTION_POLICY_FILENAME
    policy_bytes = _read_bounded_file_snapshot(
        policy_path,
        maximum_bytes=MAX_DISTRIBUTION_POLICY_BYTES,
        context="distribution policy",
    )
    policy = load_distribution_policy_bytes(policy_bytes)
    binding = result.distribution_policy
    if (
        policy.artifact_sha256 != binding.artifact_sha256
        or _sha256_bytes(policy_bytes) != binding.file_sha256
        or len(policy_bytes) != binding.size_bytes
        or policy.provenance != result.provenance
    ):
        raise OODCompletionIntegrityError("distribution policy differs from result binding")
    detector = policy.detector
    threshold = result.threshold_fit
    if (
        detector.threshold != threshold.threshold
        or detector.embedding_dim != threshold.embedding_dimension
        or detector.n_fit_samples != threshold.n_reference_samples
        or detector.n_threshold_samples != threshold.n_threshold_samples
        or detector.quantile_rank != threshold.quantile_rank
        or detector.shrinkage != threshold.shrinkage
        or detector.ridge != threshold.ridge
    ):
        raise OODCompletionIntegrityError("distribution policy fit differs from result summary")

    member_by_path = _verify_private_embedding_bundle(output_root, result)
    for relative_path, snapshot in (
        (DISTRIBUTION_POLICY_FILENAME, policy_bytes),
        (OOD_COMPLETION_RESULT_FILENAME, result_bytes),
        (_VALIDATION_ACCESS_MARKER_FILENAME, marker_bytes),
    ):
        member_by_path[relative_path] = OODBundleMember(
            relative_path=cast(OODBundleRelativePath, relative_path),
            size_bytes=len(snapshot),
            file_sha256=_sha256_bytes(snapshot),
        )
    members = tuple(member_by_path[relative_path] for relative_path in _BUNDLE_MEMBER_PATHS)
    return result, policy, members, claim_sha256


def _execute_staged(
    *,
    config: OODCompletionConfig,
    inputs: VerifiedOODInputs,
    staging_root: Path,
    code_revision: str,
    validation_claim_path: Path,
    validation_claim_state: _OneShotClaimState,
) -> _StagedOODExecution:
    cohorts = load_ood_cohorts(
        inputs.paths["dataset_manifest"],
        patient_split_salt=config.patient_split_salt,
        expected_counts=config.expected_counts,
    )
    if cohorts.reference_sha256 != config.reference_identity_sha256:
        raise OODCompletionIntegrityError("reference cohort identity differs from protocol")
    if cohorts.full_fold9_sha256 != config.fold9_identity_sha256:
        raise OODCompletionIntegrityError("fold-9 cohort identity differs from protocol")
    reference_subset, threshold_subset, validation_subset, source_subset, selected_subset = (
        _waveform_subsets(cohorts, inputs)
    )
    if (
        selected_subset.record_count != config.selected_record_count
        or selected_subset.file_count != config.selected_file_count
    ):
        raise OODCompletionIntegrityError("selected waveform subset counts differ from protocol")

    verify_official_waveform_subset(inputs.dataset_root, reference_subset)
    verify_official_waveform_subset(inputs.dataset_root, threshold_subset)

    runtime = configure_deterministic_cuda(
        expected_device_name=config.expected_device_name,
        expected_compute_capability=config.expected_compute_capability,
        expected_python_version=config.expected_python_version,
        expected_torch_version=config.expected_torch_version,
        expected_cuda_runtime=config.expected_cuda_runtime,
        expected_cudnn_version=config.expected_cudnn_version,
        expected_nvidia_driver_version=config.expected_nvidia_driver_version,
    )
    backend = DemoInferenceBackend.load(
        checkpoint_path=inputs.paths["checkpoint"],
        resolved_config_path=inputs.paths["resolved_config"],
        normalization_path=inputs.paths["normalization"],
        decision_policy_path=inputs.paths["historical_demo_policy"],
    )
    model_state_before = model_state_sha256(backend.model)
    model = prepare_resnet_for_embedding(backend.model, runtime=runtime)

    reference_saved = _extract_save_role(
        cohort=cohorts.reference,
        config=config,
        inputs=inputs,
        normalization=backend.normalization,
        model=model,
        runtime=runtime,
        staging_root=staging_root,
    )
    verify_official_waveform_subset(inputs.dataset_root, reference_subset)
    threshold_saved = _extract_save_role(
        cohort=cohorts.threshold_fit,
        config=config,
        inputs=inputs,
        normalization=backend.normalization,
        model=model,
        runtime=runtime,
        staging_root=staging_root,
    )
    verify_official_waveform_subset(inputs.dataset_root, threshold_subset)

    provenance = _lineage_provenance(
        config=config,
        inputs=inputs,
        code_revision=code_revision,
        reference_subset=reference_subset,
        source_subset=source_subset,
        selected_subset=selected_subset,
    )
    try:
        policy, threshold_summary = fit_distribution_policy(
            reference_saved.artifact.embedding,
            threshold_saved.artifact.embedding,
            provenance=provenance,
        )
    except Exception as error:
        raise _OODFitError("distribution-policy fit failed") from error
    sealed_policy = _seal_distribution_policy(staging_root=staging_root, policy=policy)
    source_artifacts = _load_postseal_source_artifacts(
        sealed_policy=sealed_policy,
        config=config,
        inputs=inputs,
        cohorts=cohorts,
    )
    assignment_sha256 = source_artifacts.assignment_sha256

    # Durably arm this contender's nonce-bound marker first. Publishing the
    # adjacent claim is then the sole one-shot C boundary, so every durable
    # winning claim necessarily has a pre-existing matching marker.
    validation_claim_sha256 = _sha256_bytes(validation_claim_state.claim_bytes)
    _mark_validation_access_armed(
        staging_root,
        validation_claim_file_sha256=validation_claim_sha256,
        owner_nonce=validation_claim_state.owner_nonce,
    )
    validation_claim_sha256 = _claim_validation_access(
        validation_claim_path,
        claim_state=validation_claim_state,
    )
    _assert_validation_access_marker_binding(
        staging_root / _VALIDATION_ACCESS_MARKER_FILENAME,
        validation_claim_file_sha256=validation_claim_sha256,
        owner_nonce=validation_claim_state.owner_nonce,
    )
    # C is first decoded only after the fitted policy file is sealed and reloaded.
    verify_official_waveform_subset(inputs.dataset_root, validation_subset)
    validation_saved = _extract_validation_after_policy_seal(
        sealed_policy=sealed_policy,
        cohort=cohorts.source_validation,
        config=config,
        inputs=inputs,
        normalization=backend.normalization,
        model=model,
        runtime=runtime,
        staging_root=staging_root,
    )
    verify_official_waveform_subset(inputs.dataset_root, validation_subset)
    try:
        validation_summary = evaluate_source_validation(
            validation_saved.artifact,
            repeated_embedding_tensor_sha256=(
                validation_saved.repeated_embedding_tensor_sha256
            ),
            policy=sealed_policy.policy,
            source_assignment_sha256=assignment_sha256,
        )
    except Exception as error:
        raise _OODValidationError("source-validation evaluation failed") from error
    if model_state_sha256(model) != model_state_before:
        raise OODCompletionIntegrityError("model state changed during embedding extraction")
    _assert_bound_inputs_unchanged(
        config,
        inputs,
        source_artifacts=source_artifacts,
    )
    _assert_clean_code_revision(inputs.project_root, code_revision)

    execution = _reference_threshold_execution(
        runtime=runtime,
        reference=reference_saved,
        threshold=threshold_saved,
        source_assignment_sha256=assignment_sha256,
    )
    eligible = validation_summary.cluster_bootstrap.one_sided_upper <= 0.05
    result = seal_ood_completion_result(
        OODCompletionResultBody(
            schema_version=1,
            artifact_type="ecg_trust.ood_completion_result",
            protocol_id="trust-sentinel-ood-completion-v1",
            status=(
                "SOURCE_SUPPORT_GATE_COMPLETE"
                if eligible
                else "SOURCE_SUPPORT_GATE_TARGET_MISSED"
            ),
            frozen_at_utc=config.frozen_at_utc,
            source_calibration=source_artifacts.result,
            provenance=provenance,
            distribution_policy=DistributionPolicyBinding(
                filename="distribution-policy.json",
                artifact_sha256=sealed_policy.policy.artifact_sha256,
                file_sha256=sealed_policy.file_sha256,
                size_bytes=sealed_policy.size_bytes,
            ),
            reference_and_threshold_execution=execution,
            threshold_fit=threshold_summary,
            source_validation=validation_summary,
            ood_positive_evaluation=OODPositiveEvaluationSummary(
                status="NOT_EVALUATED",
                evaluation_sources=(),
                records=0,
                semantic_ood_recall="NOT_EVALUATED",
                severe_ood_recall="NOT_EVALUATED",
                ood_auroc="NOT_EVALUATED",
                ood_average_precision="NOT_EVALUATED",
                unseen_site_or_device_performance="NOT_EVALUATED",
                target_site_fitting_performed=False,
                reason_code="NO_FROZEN_OOD_POSITIVE_EVALUATION_SOURCE",
            ),
            integrity=OODIntegritySummary(
                complete=True,
                verified_input_hashes=True,
                source_calibration_verified=True,
                refit_lineage_verified=True,
                waveform_checksums_verified_before_and_after=True,
                source_alignment_verified=True,
                patient_roles_disjoint=True,
                forbidden_sources_not_accessed=True,
                model_state_unchanged=True,
                deterministic_repeat_verified=True,
                distribution_policy_verified=True,
                aggregate_only_result_verified=True,
            ),
            research_bundle_eligible=eligible,
            claims=OODClaimBoundary(
                scope="retrospective_ptbxl_source_domain_development_only",
                research_only=True,
                clinical_validation=False,
                external_ood_positive_validation=False,
                limitations=(
                    "no_external_lockbox_evaluation",
                    "no_ood_positive_evaluation",
                    "no_clinical_validation",
                    "ood_score_does_not_identify_unknown_disease",
                    "source_false_rejection_target_is_provisional",
                    "research_bundle_eligibility_is_not_clinical_release_readiness",
                ),
            ),
        )
    )
    result_path = staging_root / OOD_COMPLETION_RESULT_FILENAME
    _atomic_write_new(result_path, ood_completion_result_json_bytes(result))
    loaded_result = load_ood_completion_result_bytes(
        _read_bounded_file_snapshot(
            result_path,
            maximum_bytes=MAX_OOD_COMPLETION_RESULT_BYTES,
            context="staged OOD completion result",
        )
    )
    if loaded_result.artifact_sha256 != result.artifact_sha256:
        raise OODCompletionIntegrityError("staged OOD result identity changed")
    return _StagedOODExecution(
        result=loaded_result,
        source_artifacts=source_artifacts,
    )


def _extract_save_role(
    *,
    cohort: OrderedCohort,
    config: OODCompletionConfig,
    inputs: VerifiedOODInputs,
    normalization: NormalizationStats,
    model: torch.nn.Module,
    runtime: DeterministicCUDARuntime,
    staging_root: Path,
) -> SavedRoleEmbeddings:
    from ecg_trust.models.resnet1d import ResNet1D

    if not isinstance(model, ResNet1D):
        raise OODCompletionExecutionError("embedding model is not ResNet1D")
    dataset = _dataset_for_cohort(
        cohort,
        inputs=inputs,
        normalization=normalization,
    )
    passes = extract_embeddings_twice(model, dataset, runtime=runtime)
    ecg_id = np.asarray([record.ecg_id for record in cohort.records], dtype=np.int64)
    patient_id = np.asarray(
        [record.patient_id for record in cohort.records], dtype=np.int64
    )
    strat_fold = np.asarray(
        [record.strat_fold for record in cohort.records], dtype=np.int8
    )
    first = create_embedding_artifact(
        ecg_id=ecg_id,
        patient_id=patient_id,
        strat_fold=strat_fold,
        embedding=passes.first,
        role=cohort.role,
        expected_folds=cohort.folds,
        checkpoint_sha256=config.bindings["checkpoint"].file_sha256,
        config_sha256=config.file_sha256,
        normalization_sha256=config.bindings["normalization"].file_sha256,
        manifest_sha256=config.bindings["dataset_manifest"].file_sha256,
        protocol_sha256=config.experiment_protocol_sha256,
        runtime_sha256=runtime.runtime_sha256,
    )
    repeated = create_embedding_artifact(
        ecg_id=ecg_id,
        patient_id=patient_id,
        strat_fold=strat_fold,
        embedding=passes.repeated,
        role=cohort.role,
        expected_folds=cohort.folds,
        checkpoint_sha256=config.bindings["checkpoint"].file_sha256,
        config_sha256=config.file_sha256,
        normalization_sha256=config.bindings["normalization"].file_sha256,
        manifest_sha256=config.bindings["dataset_manifest"].file_sha256,
        protocol_sha256=config.experiment_protocol_sha256,
        runtime_sha256=runtime.runtime_sha256,
    )
    if (
        first.alignment_sha256 != repeated.alignment_sha256
        or first.embedding_tensor_sha256 != repeated.embedding_tensor_sha256
    ):
        raise OODDeterminismError("role embedding repeat hashes differ")
    relative_npz = config.private_npz_paths[cohort.role]
    destination = _resolve_staging_path(staging_root, relative_npz)
    try:
        files = save_embedding_artifact(first, destination)
        loaded = load_embedding_artifact(
            files.npz_path,
            expected_artifact_sha256=files.artifact_sha256,
            expected_npz_file_sha256=files.npz_file_sha256,
            expected_role=cohort.role,
        )
    except (EmbeddingArtifactError, OSError) as error:
        raise OODCompletionExecutionError(
            "private embedding artifact could not be committed and verified"
        ) from error
    return SavedRoleEmbeddings(
        artifact=loaded,
        repeated_embedding_tensor_sha256=repeated.embedding_tensor_sha256,
    )


def _seal_distribution_policy(
    *,
    staging_root: Path,
    policy: DistributionPolicy,
) -> SealedDistributionPolicy:
    policy_path = staging_root / DISTRIBUTION_POLICY_FILENAME
    _atomic_write_new(policy_path, distribution_policy_json_bytes(policy))
    policy_bytes = _read_bounded_file_snapshot(
        policy_path,
        maximum_bytes=MAX_DISTRIBUTION_POLICY_BYTES,
        context="sealed distribution policy",
    )
    loaded = load_distribution_policy_bytes(policy_bytes)
    if loaded.artifact_sha256 != policy.artifact_sha256:
        raise OODCompletionIntegrityError("sealed distribution policy identity changed")
    return SealedDistributionPolicy(
        policy=loaded,
        path=policy_path,
        file_sha256=_sha256_bytes(policy_bytes),
        size_bytes=len(policy_bytes),
    )


def _extract_validation_after_policy_seal(
    *,
    sealed_policy: SealedDistributionPolicy,
    cohort: OrderedCohort,
    config: OODCompletionConfig,
    inputs: VerifiedOODInputs,
    normalization: NormalizationStats,
    model: torch.nn.Module,
    runtime: DeterministicCUDARuntime,
    staging_root: Path,
) -> SavedRoleEmbeddings:
    _verify_sealed_policy_proof(sealed_policy)
    if cohort.role is not EmbeddingRole.SOURCE_VALIDATION:
        raise OODCompletionIntegrityError("post-seal extraction accepts only source validation")
    return _extract_save_role(
        cohort=cohort,
        config=config,
        inputs=inputs,
        normalization=normalization,
        model=model,
        runtime=runtime,
        staging_root=staging_root,
    )


def _verify_sealed_policy_proof(
    sealed_policy: SealedDistributionPolicy,
) -> DistributionPolicy:
    if not isinstance(sealed_policy, SealedDistributionPolicy):
        raise TypeError("source access requires a sealed distribution-policy proof")
    policy_bytes = _read_bounded_file_snapshot(
        sealed_policy.path,
        maximum_bytes=MAX_DISTRIBUTION_POLICY_BYTES,
        context="sealed distribution policy at validation boundary",
    )
    if _sha256_bytes(policy_bytes) != sealed_policy.file_sha256:
        raise OODCompletionIntegrityError("distribution policy changed before validation access")
    loaded = load_distribution_policy_bytes(policy_bytes)
    if loaded.artifact_sha256 != sealed_policy.policy.artifact_sha256:
        raise OODCompletionIntegrityError("distribution policy changed before validation access")
    return loaded


def _load_postseal_source_artifacts(
    *,
    sealed_policy: SealedDistributionPolicy,
    config: OODCompletionConfig,
    inputs: VerifiedOODInputs,
    cohorts: OODCohorts,
) -> _PostSealSourceArtifacts:
    """Decode historical C-bearing lineage only after the detector is sealed."""

    _verify_sealed_policy_proof(sealed_policy)
    try:
        source_config, source_config_hash = load_source_calibration_config(
            inputs.paths["source_calibration_config"]
        )
    except Exception as error:
        raise OODCompletionIntegrityError(
            "post-seal source calibration config verification failed"
        ) from error
    if source_config_hash != config.bindings["source_calibration_config"].file_sha256:
        raise OODCompletionIntegrityError("source calibration config hash differs after decode")
    if source_config.patient_split.salt != config.patient_split_salt:
        raise OODCompletionIntegrityError("source calibration patient split salt differs")

    source_result_path = inputs.paths["source_calibration_result"]
    source_result_bytes = _read_bounded_file_snapshot(
        source_result_path,
        maximum_bytes=_SOURCE_CALIBRATION_RESULT_MAX_BYTES,
        context="source calibration result",
    )
    if _sha256_bytes(source_result_bytes) != config.bindings[
        "source_calibration_result"
    ].file_sha256:
        raise OODCompletionIntegrityError("source calibration result file hash differs")
    try:
        source_result = load_source_calibration_result_bytes(source_result_bytes)
    except Exception as error:
        raise OODCompletionIntegrityError(
            "post-seal source calibration result verification failed"
        ) from error
    if source_result.artifact_sha256 != config.source_calibration_artifact_sha256:
        raise OODCompletionIntegrityError("source calibration logical identity differs")
    if source_result.open_world.status != "PENDING" or source_result.open_world.release_ready:
        raise OODCompletionIntegrityError("source calibration v1 must remain permanently pending")
    try:
        source_inputs = verify_source_inputs(source_config, project_root=inputs.project_root)
    except Exception as error:
        raise OODCompletionIntegrityError(
            "post-seal source prediction verification failed"
        ) from error
    if source_inputs.source_npz_sha256 != source_result.provenance.source_npz_sha256:
        raise OODCompletionIntegrityError("source prediction NPZ differs from nested provenance")
    if source_inputs.source_sidecar_sha256 != source_result.provenance.source_sidecar_sha256:
        raise OODCompletionIntegrityError(
            "source prediction sidecar differs from nested provenance"
        )
    assignment_sha256 = _verify_source_role_alignment(
        cohorts,
        source_config=source_config,
        source_result=source_result,
        source_inputs=source_inputs,
    )
    return _PostSealSourceArtifacts(
        config=source_config,
        result=source_result,
        inputs=source_inputs,
        assignment_sha256=assignment_sha256,
    )


def _dataset_for_cohort(
    cohort: OrderedCohort,
    *,
    inputs: VerifiedOODInputs,
    normalization: NormalizationStats,
) -> PTBXLDataset:
    rows = [
        {
            "ecg_id": record.ecg_id,
            "patient_id": record.patient_id,
            "strat_fold": record.strat_fold,
            "record_path": record.record_path,
            **{target: 0 for target in TARGET_COLUMNS},
        }
        for record in cohort.records
    ]
    frame = pd.DataFrame(rows)
    return PTBXLDataset(
        frame,
        inputs.dataset_root,
        normalization=normalization,
        require_positive_target=False,
        protocol=inputs.experiment_protocol,
    )


def _verify_source_role_alignment(
    cohorts: OODCohorts,
    *,
    source_config: SourceCalibrationConfig,
    source_result: SourceCalibrationResult,
    source_inputs: VerifiedSourceInputs,
) -> str:
    fold9_records = cohorts.full_fold9_records
    expected_ecg = np.asarray([record.ecg_id for record in fold9_records], dtype=np.int64)
    expected_patient = np.asarray(
        [record.patient_id for record in fold9_records], dtype=np.int64
    )
    expected_fold = np.asarray([record.strat_fold for record in fold9_records], dtype=np.int8)
    observed_ecg, observed_patient, observed_fold = _load_source_identity_arrays(
        source_inputs.npz_path
    )
    if not (
        np.array_equal(observed_ecg, expected_ecg)
        and np.array_equal(observed_patient, expected_patient)
        and np.array_equal(observed_fold, expected_fold)
    ):
        raise OODCompletionIntegrityError("source prediction identities differ from manifest")

    assignments = {
        int(patient): patient_split_role(
            patient_id=int(patient),
            salt=source_config.patient_split.salt,
        )
        for patient in np.unique(expected_patient)
    }
    assignment_payload: dict[str, object] = {
        "schema_version": 1,
        "algorithm": "sha256_first8_uint64_fraction_v1",
        "assignments": [
            {"patient_id_base10": str(patient), "role": assignments[patient].value}
            for patient in sorted(assignments)
        ],
    }
    digest = source_canonical_sha256(assignment_payload)
    if digest != source_result.split.assignment_sha256:
        raise OODCompletionIntegrityError("source patient assignment hash differs")
    return digest


def _load_source_identity_arrays(path: Path) -> tuple[Int64Array, Int64Array, Int8Array]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > _SOURCE_IDENTITY_NPZ_MAX_BYTES
    ):
        raise OODCompletionIntegrityError("source prediction NPZ is invalid")
    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            names = {item.filename for item in archive.infolist()}
            required = {"ecg_id.npy", "patient_id.npy", "strat_fold.npy"}
            if not required.issubset(names):
                raise OODCompletionIntegrityError("source prediction identity arrays are missing")
        with np.load(path, allow_pickle=False) as archive:
            ecg = np.asarray(archive["ecg_id"]).copy()
            patient = np.asarray(archive["patient_id"]).copy()
            fold = np.asarray(archive["strat_fold"]).copy()
    except OODCompletionIntegrityError:
        raise
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        raise OODCompletionIntegrityError(
            "source prediction identities cannot be loaded"
        ) from error
    if (
        ecg.dtype != np.dtype(np.int64)
        or patient.dtype != np.dtype(np.int64)
        or fold.dtype != np.dtype(np.int8)
        or ecg.ndim != 1
        or patient.shape != ecg.shape
        or fold.shape != ecg.shape
    ):
        raise OODCompletionIntegrityError("source prediction identity dtypes or shapes differ")
    order = np.argsort(ecg, kind="stable")
    return (
        cast(Int64Array, ecg[order]),
        cast(Int64Array, patient[order]),
        cast(Int8Array, fold[order]),
    )


def _waveform_subsets(
    cohorts: OODCohorts,
    inputs: VerifiedOODInputs,
) -> tuple[
    OfficialWaveformSubset,
    OfficialWaveformSubset,
    OfficialWaveformSubset,
    OfficialWaveformSubset,
    OfficialWaveformSubset,
]:
    reference_records = cohorts.reference.records
    threshold_records = cohorts.threshold_fit.records
    validation_records = cohorts.source_validation.records
    source_records = (*threshold_records, *validation_records)
    selected_records = (*reference_records, *source_records)
    return (
        build_official_waveform_subset(
            reference_records,
            official_checksums=inputs.official_checksums,
        ),
        build_official_waveform_subset(
            threshold_records,
            official_checksums=inputs.official_checksums,
        ),
        build_official_waveform_subset(
            validation_records,
            official_checksums=inputs.official_checksums,
        ),
        build_official_waveform_subset(
            source_records,
            official_checksums=inputs.official_checksums,
        ),
        build_official_waveform_subset(
            selected_records,
            official_checksums=inputs.official_checksums,
        ),
    )


def _lineage_provenance(
    *,
    config: OODCompletionConfig,
    inputs: VerifiedOODInputs,
    code_revision: str,
    reference_subset: OfficialWaveformSubset,
    source_subset: OfficialWaveformSubset,
    selected_subset: OfficialWaveformSubset,
) -> OODLineageProvenance:
    return OODLineageProvenance(
        ood_config_file_sha256=config.file_sha256,
        source_calibration_artifact_sha256=config.source_calibration_artifact_sha256,
        source_calibration_file_sha256=config.bindings["source_calibration_result"].file_sha256,
        source_calibration_config_file_sha256=(
            config.bindings["source_calibration_config"].file_sha256
        ),
        refit_completion_artifact_sha256=config.refit_completion_artifact_sha256,
        refit_completion_file_sha256=config.bindings["refit_completion"].file_sha256,
        checkpoint_file_sha256=config.bindings["checkpoint"].file_sha256,
        resolved_config_sha256=config.resolved_config_sha256,
        resolved_config_file_sha256=config.bindings["resolved_config"].file_sha256,
        dataset_manifest_file_sha256=config.bindings["dataset_manifest"].file_sha256,
        normalization_file_sha256=config.bindings["normalization"].file_sha256,
        experiment_protocol_sha256=config.experiment_protocol_sha256,
        environment_lock_file_sha256=config.bindings["dependency_lock"].file_sha256,
        project_manifest_file_sha256=(
            "sha256:e1de755829678d588784bbcc34becc8c031c742d9b3f05458e76e67f577da3cd"
        ),
        raw_checksum_inventory_file_sha256=(
            config.bindings["official_dataset_checksums"].file_sha256
        ),
        raw_selected_inventory_sha256=selected_subset.subset_sha256,
        selected_record_count=18_383,
        selected_file_count=36_766,
        raw_reference_inventory_sha256=reference_subset.subset_sha256,
        raw_source_inventory_sha256=source_subset.subset_sha256,
        code_revision=code_revision,
        model_member_id="resnet1d-seed2026",
        architecture="resnet1d",
        seed=2026,
    )


def _reference_threshold_execution(
    *,
    runtime: DeterministicCUDARuntime,
    reference: SavedRoleEmbeddings,
    threshold: SavedRoleEmbeddings,
    source_assignment_sha256: str,
) -> ReferenceAndThresholdExecutionSummary:
    reference_artifact = reference.artifact
    threshold_artifact = threshold.artifact
    return ReferenceAndThresholdExecutionSummary(
        runtime=runtime.summary,
        reference=ReferenceEmbeddingExecutionSummary(
            records=reference_artifact.record_count,
            patients=reference_artifact.patient_count,
            embedding_dimension=512,
            alignment_sha256=reference_artifact.alignment_sha256,
            embedding_tensor_sha256=reference_artifact.embedding_tensor_sha256,
            repeated_embedding_tensor_sha256=reference.repeated_embedding_tensor_sha256,
            embedding_artifact_sha256=_required_hash(reference_artifact.artifact_sha256),
            embedding_npz_file_sha256=_required_hash(reference_artifact.npz_file_sha256),
            embedding_sidecar_file_sha256=_required_hash(
                reference_artifact.sidecar_file_sha256
            ),
            runtime_sha256=reference_artifact.runtime_sha256,
            exact_repeat_verified=True,
            public_contains_row_arrays=False,
            role="ptbxl_folds_1_to_8_training_reference",
            folds=(1, 2, 3, 4, 5, 6, 7, 8),
        ),
        threshold_fit=ThresholdEmbeddingExecutionSummary(
            records=threshold_artifact.record_count,
            patients=threshold_artifact.patient_count,
            embedding_dimension=512,
            alignment_sha256=threshold_artifact.alignment_sha256,
            embedding_tensor_sha256=threshold_artifact.embedding_tensor_sha256,
            repeated_embedding_tensor_sha256=threshold.repeated_embedding_tensor_sha256,
            embedding_artifact_sha256=_required_hash(threshold_artifact.artifact_sha256),
            embedding_npz_file_sha256=_required_hash(threshold_artifact.npz_file_sha256),
            embedding_sidecar_file_sha256=_required_hash(
                threshold_artifact.sidecar_file_sha256
            ),
            runtime_sha256=threshold_artifact.runtime_sha256,
            exact_repeat_verified=True,
            public_contains_row_arrays=False,
            role="conformal_and_ood_threshold_fit",
            folds=(9,),
            source_assignment_sha256=source_assignment_sha256,
        ),
    )


def _assert_bound_inputs_unchanged(
    config: OODCompletionConfig,
    inputs: VerifiedOODInputs,
    *,
    source_artifacts: _PostSealSourceArtifacts,
) -> None:
    if _sha256_file(config.path) != config.file_sha256:
        raise OODCompletionIntegrityError("OOD config changed during execution")
    for name, binding in config.bindings.items():
        if _sha256_file(inputs.paths[name]) != binding.file_sha256:
            raise OODCompletionIntegrityError("a bound input changed during execution")
    if _sha256_file(source_artifacts.inputs.npz_path) != (
        source_artifacts.inputs.source_npz_sha256
    ):
        raise OODCompletionIntegrityError("source prediction NPZ changed during execution")
    if _sha256_file(source_artifacts.inputs.sidecar_path) != (
        source_artifacts.inputs.source_sidecar_sha256
    ):
        raise OODCompletionIntegrityError("source prediction sidecar changed during execution")


def _role_counts(roles: Mapping[str, object], name: str) -> CohortCounts:
    role = _mapping(roles.get(name), f"roles.{name}")
    records = role.get("records")
    patients = role.get("patients")
    if isinstance(records, bool) or not isinstance(records, int):
        raise OODCompletionConfigError(f"roles.{name}.records must be an integer")
    if isinstance(patients, bool) or not isinstance(patients, int):
        raise OODCompletionConfigError(f"roles.{name}.patients must be an integer")
    return CohortCounts(records=records, patients=patients)


def _private_npz_path(
    private_roles: Mapping[str, object],
    role: EmbeddingRole,
) -> str:
    item = _mapping(private_roles.get(role.value), f"private role {role.value}")
    return _relative_path(item.get("npz"), f"private role {role.value}.npz")


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise OODCompletionConfigError(f"{context} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _relative_path(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise OODCompletionConfigError(f"{context} must be non-empty text")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or posix.as_posix() != value
    ):
        raise OODCompletionConfigError(f"{context} must be a safe relative POSIX path")
    return value


def _prefixed_sha256(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise OODCompletionConfigError(f"{context} must be SHA-256 text")
    normalized = value if value.startswith("sha256:") else f"sha256:{value}"
    if len(normalized) != 71 or _SHA256.fullmatch(normalized[7:]) is None:
        raise OODCompletionConfigError(f"{context} must be lowercase SHA-256 text")
    return normalized


def _exact_string(value: object, expected: str, context: str) -> str:
    if value != expected or not isinstance(value, str):
        raise OODCompletionConfigError(f"{context} differs from the frozen value")
    return value


def _exact_integer(value: object, expected: int, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise OODCompletionConfigError(f"{context} differs from the frozen value")
    return value


def _required_hash(value: str | None) -> str:
    if value is None:
        raise OODCompletionIntegrityError("sealed artifact is missing a required hash")
    return value


def _resolve_project_path(root: Path, relative_path: str, *, require_file: bool) -> Path:
    if PurePosixPath(relative_path).is_absolute() or PureWindowsPath(relative_path).is_absolute():
        raise OODCompletionIntegrityError("project path must be relative")
    parts = PurePosixPath(relative_path).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise OODCompletionIntegrityError("project path contains a forbidden component")
    lexical_root = Path(os.path.abspath(os.fspath(root)))
    candidate = Path(os.path.abspath(os.fspath(lexical_root.joinpath(*parts))))
    try:
        candidate.relative_to(lexical_root)
    except ValueError as error:
        raise OODCompletionIntegrityError("project path escapes the project root") from error
    _assert_no_reparse_components(candidate)
    if require_file and not candidate.is_file():
        raise OODCompletionIntegrityError("required project file is missing")
    resolved = candidate.resolve(strict=require_file)
    try:
        resolved.relative_to(lexical_root.resolve(strict=True))
    except ValueError as error:
        raise OODCompletionIntegrityError(
            "resolved project path escapes the project root"
        ) from error
    return resolved


def _resolve_staging_path(staging_root: Path, relative_path: str) -> Path:
    parts = PurePosixPath(relative_path).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise OODCompletionExecutionError("private artifact path is invalid")
    lexical_root = Path(os.path.abspath(os.fspath(staging_root)))
    candidate = Path(os.path.abspath(os.fspath(lexical_root.joinpath(*parts))))
    try:
        candidate.relative_to(lexical_root)
    except ValueError as error:
        raise OODCompletionExecutionError("private artifact path escapes staging") from error
    _assert_no_reparse_components(candidate)
    return candidate


def _reject_duplicate_yaml_keys(serialized: str) -> None:
    try:
        root = yaml.compose(serialized, Loader=yaml.SafeLoader)
    except yaml.YAMLError as error:
        raise OODCompletionConfigError("OOD configuration is not valid YAML") from error
    visited: set[int] = set()

    def visit(node: object) -> None:
        identity = id(node)
        if identity in visited:
            raise OODCompletionConfigError("YAML aliases are forbidden")
        visited.add(identity)
        if isinstance(node, yaml.MappingNode):
            keys: set[tuple[str, str]] = set()
            for key_node, value_node in node.value:
                if not isinstance(key_node, yaml.ScalarNode):
                    raise OODCompletionConfigError("YAML mapping keys must be scalar")
                key = (str(key_node.tag), str(key_node.value))
                if key in keys or str(key_node.value) == "<<":
                    raise OODCompletionConfigError("YAML keys must be unique without merges")
                keys.add(key)
                visit(value_node)
        elif isinstance(node, yaml.SequenceNode):
            for item in node.value:
                visit(item)

    if root is not None:
        visit(root)


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise OODCompletionIntegrityError("required file could not be hashed") from error
    return "sha256:" + digest.hexdigest()


def _read_bounded_file_snapshot(
    path: Path,
    *,
    maximum_bytes: int,
    context: str,
) -> bytes:
    """Read one bounded regular-file snapshot for decoding and physical identity."""

    if maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be positive")
    if _is_link_or_junction(path) or not path.is_file():
        raise OODCompletionIntegrityError(f"{context} is missing or indirect")
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum_bytes:
                raise OODCompletionIntegrityError(f"{context} size is invalid")
            payload = handle.read(maximum_bytes + 1)
            after = os.fstat(handle.fileno())
    except OODCompletionIntegrityError:
        raise
    except OSError as error:
        raise OODCompletionIntegrityError(f"{context} cannot be read") from error
    if (
        len(payload) != before.st_size
        or len(payload) > maximum_bytes
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise OODCompletionIntegrityError(f"{context} changed while being read")
    return payload


def _strict_existing_bundle_root(path: str | Path) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    _assert_no_reparse_components(lexical)
    if not lexical.is_dir():
        raise OODCompletionIntegrityError("OOD completion bundle root is missing")
    return lexical


def _is_link_or_junction(path: Path) -> bool:
    try:
        is_junction = getattr(path, "is_junction", None)
        return path.is_symlink() or bool(is_junction is not None and is_junction())
    except OSError as error:
        raise OODCompletionIntegrityError("filesystem link state could not be inspected") from error


def _assert_no_reparse_components(path: Path) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if _is_link_or_junction(current):
            raise OODCompletionIntegrityError("filesystem path contains a link or junction")


def _assert_exact_bundle_tree(
    output_root: Path,
    *,
    include_success_manifest: bool,
) -> None:
    if _is_link_or_junction(output_root) or not output_root.is_dir():
        raise OODCompletionIntegrityError("OOD completion output root is missing or indirect")
    failure_path = output_root / OOD_COMPLETION_FAILURE_FILENAME
    if failure_path.exists() or failure_path.is_symlink():
        raise OODCompletionIntegrityError("failure receipt forbids successful bundle use")

    files: set[str] = set()
    directories: set[str] = set()
    pending = [output_root]
    while pending:
        directory = pending.pop()
        try:
            entries = tuple(directory.iterdir())
        except OSError as error:
            raise OODCompletionIntegrityError("bundle tree could not be enumerated") from error
        for entry in entries:
            if _is_link_or_junction(entry):
                raise OODCompletionIntegrityError("bundle tree contains a link or junction")
            relative = entry.relative_to(output_root).as_posix()
            if entry.is_dir():
                directories.add(relative)
                pending.append(entry)
            elif entry.is_file():
                files.add(relative)
            else:
                raise OODCompletionIntegrityError("bundle tree contains a non-regular entry")

    expected_files = set(_BUNDLE_MEMBER_PATHS)
    if include_success_manifest:
        expected_files.add(OOD_COMPLETION_SUCCESS_FILENAME)
    if files != expected_files or directories != {"private"}:
        raise OODCompletionIntegrityError("bundle tree differs from the exact file inventory")


def _verify_validation_access_claim_and_marker(output_root: Path) -> tuple[str, bytes]:
    claim_path = _validation_access_claim_path(output_root)
    if _is_link_or_junction(claim_path) or not claim_path.is_file():
        raise OODCompletionIntegrityError("source-validation one-shot claim is missing")
    claim_bytes = _read_bounded_file_snapshot(
        claim_path,
        maximum_bytes=_VALIDATION_ACCESS_RECORD_MAX_BYTES,
        context="source-validation claim",
    )
    owner_nonce = _validate_validation_access_claim_bytes(claim_bytes)
    claim_sha256 = _sha256_bytes(claim_bytes)
    marker_path = output_root / _VALIDATION_ACCESS_MARKER_FILENAME
    marker_bytes = _assert_validation_access_marker_binding(
        marker_path,
        validation_claim_file_sha256=claim_sha256,
        owner_nonce=owner_nonce,
    )
    return claim_sha256, marker_bytes


def _verify_private_embedding_bundle(
    output_root: Path,
    result: OODCompletionResult,
) -> dict[str, OODBundleMember]:
    execution = result.reference_and_threshold_execution
    bindings = (
        _ExpectedEmbeddingBinding(
            relative_npz_path="private/reference-embeddings.npz",
            role=EmbeddingRole.REFERENCE,
            folds=(1, 2, 3, 4, 5, 6, 7, 8),
            summary=execution.reference,
        ),
        _ExpectedEmbeddingBinding(
            relative_npz_path="private/source-validation-embeddings.npz",
            role=EmbeddingRole.SOURCE_VALIDATION,
            folds=(9,),
            summary=result.source_validation,
        ),
        _ExpectedEmbeddingBinding(
            relative_npz_path="private/threshold-fit-embeddings.npz",
            role=EmbeddingRole.THRESHOLD_FIT,
            folds=(9,),
            summary=execution.threshold_fit,
        ),
    )
    provenance = result.provenance
    members: dict[str, OODBundleMember] = {}
    for binding in bindings:
        npz_path = output_root.joinpath(*PurePosixPath(binding.relative_npz_path).parts)
        try:
            artifact = load_embedding_artifact(
                npz_path,
                expected_artifact_sha256=binding.summary.embedding_artifact_sha256,
                expected_npz_file_sha256=binding.summary.embedding_npz_file_sha256,
                expected_role=binding.role,
            )
        except (EmbeddingArtifactError, OSError) as error:
            raise OODCompletionIntegrityError(
                "private embedding artifact failed bundle verification"
            ) from error
        if (
            artifact.expected_folds != binding.folds
            or artifact.record_count != binding.summary.records
            or artifact.patient_count != binding.summary.patients
            or artifact.alignment_sha256 != binding.summary.alignment_sha256
            or artifact.embedding_tensor_sha256
            != binding.summary.embedding_tensor_sha256
            or artifact.runtime_sha256 != binding.summary.runtime_sha256
            or artifact.artifact_sha256 != binding.summary.embedding_artifact_sha256
            or artifact.npz_file_sha256 != binding.summary.embedding_npz_file_sha256
            or artifact.sidecar_file_sha256
            != binding.summary.embedding_sidecar_file_sha256
            or artifact.checkpoint_sha256 != provenance.checkpoint_file_sha256
            or artifact.config_sha256 != provenance.ood_config_file_sha256
            or artifact.normalization_sha256 != provenance.normalization_file_sha256
            or artifact.manifest_sha256 != provenance.dataset_manifest_file_sha256
            or artifact.protocol_sha256 != provenance.experiment_protocol_sha256
        ):
            raise OODCompletionIntegrityError(
                "private embedding artifact differs from aggregate result binding"
            )
        if (
            artifact.npz_file_sha256 is None
            or artifact.sidecar_file_sha256 is None
            or artifact.npz_size_bytes is None
            or artifact.sidecar_size_bytes is None
        ):
            raise OODCompletionIntegrityError(
                "private embedding physical identity is incomplete"
            )
        sidecar_relative_path = PurePosixPath(binding.relative_npz_path).with_suffix(
            ".json"
        ).as_posix()
        members[binding.relative_npz_path] = OODBundleMember(
            relative_path=cast(OODBundleRelativePath, binding.relative_npz_path),
            size_bytes=artifact.npz_size_bytes,
            file_sha256=artifact.npz_file_sha256,
        )
        members[sidecar_relative_path] = OODBundleMember(
            relative_path=cast(OODBundleRelativePath, sidecar_relative_path),
            size_bytes=artifact.sidecar_size_bytes,
            file_sha256=artifact.sidecar_file_sha256,
        )
    return members


def _validate_code_revision(value: str) -> None:
    if not isinstance(value, str) or _GIT_REVISION.fullmatch(value) is None:
        raise OODCompletionIntegrityError("code revision must be a lowercase Git hash")


def _assert_clean_code_revision(project_root: Path, expected_revision: str) -> None:
    """Bind every execution boundary to the same clean committed source tree."""

    try:
        observed_revision = verify_clean_git_revision(project_root)
    except Exception as error:
        raise OODCompletionIntegrityError(
            "OOD execution requires the clean committed Git worktree"
        ) from error
    if observed_revision != expected_revision:
        raise OODCompletionIntegrityError("supplied code revision differs from Git HEAD")


def _validation_access_claim_path(output_root: Path) -> Path:
    return output_root.parent / f".{output_root.name}{_VALIDATION_ACCESS_CLAIM_SUFFIX}"


def _assert_validation_access_unclaimed(claim_path: Path) -> None:
    if claim_path.exists() or claim_path.is_symlink():
        raise OODCompletionExecutionError(
            "source-validation one-shot claim already exists; retry is forbidden"
        )


def _validation_access_claim_bytes(owner_nonce: str) -> bytes:
    if _OWNER_NONCE.fullmatch(owner_nonce) is None:
        raise OODCompletionIntegrityError("source-validation claim owner nonce is invalid")
    return canonical_json_bytes(
        {
            "artifact_type": "ecg_trust.source_validation_one_shot_claim",
            "owner_nonce": owner_nonce,
            "protocol_id": "trust-sentinel-ood-completion-v1",
            "retry_forbidden": True,
            "state": "CLAIMED",
        }
    ) + b"\n"


def _validate_validation_access_claim_bytes(payload: bytes) -> str:
    if (
        not payload
        or len(payload) > _VALIDATION_ACCESS_RECORD_MAX_BYTES
        or not payload.endswith(b"\n")
    ):
        raise OODCompletionIntegrityError("source-validation claim byte contract is invalid")
    try:
        decoded: object = json.loads(payload[:-1].decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise OODCompletionIntegrityError("source-validation claim cannot be decoded") from error
    if not isinstance(decoded, dict) or set(decoded) != {
        "artifact_type",
        "owner_nonce",
        "protocol_id",
        "retry_forbidden",
        "state",
    }:
        raise OODCompletionIntegrityError("source-validation claim schema is invalid")
    owner_nonce = decoded.get("owner_nonce")
    if (
        decoded.get("artifact_type") != "ecg_trust.source_validation_one_shot_claim"
        or not isinstance(owner_nonce, str)
        or _OWNER_NONCE.fullmatch(owner_nonce) is None
        or decoded.get("protocol_id") != "trust-sentinel-ood-completion-v1"
        or decoded.get("retry_forbidden") is not True
        or decoded.get("state") != "CLAIMED"
        or canonical_json_bytes(decoded) + b"\n" != payload
    ):
        raise OODCompletionIntegrityError("source-validation claim content is invalid")
    return owner_nonce


def _claim_validation_access(
    claim_path: Path,
    *,
    claim_state: _OneShotClaimState,
) -> str:
    """Atomically consume the fixed one-shot C claim across all processes."""

    _atomic_write_new(
        claim_path,
        claim_state.claim_bytes,
        publication_witness=claim_state._mark_published,
    )
    try:
        observed = _read_bounded_file_snapshot(
            claim_path,
            maximum_bytes=_VALIDATION_ACCESS_RECORD_MAX_BYTES,
            context="source-validation claim",
        )
    except OODCompletionIntegrityError as error:
        raise OODCompletionExecutionError(
            "published source-validation claim could not be verified"
        ) from error
    if observed != claim_state.claim_bytes:
        raise OODCompletionExecutionError("source-validation one-shot claim bytes changed")
    try:
        _validate_validation_access_claim_bytes(observed)
    except OODCompletionIntegrityError as error:
        raise OODCompletionExecutionError(
            "published source-validation claim content is invalid"
        ) from error
    return _sha256_bytes(claim_state.claim_bytes)


def _validation_access_marker_bytes(
    validation_claim_file_sha256: str,
    *,
    owner_nonce: str,
) -> bytes:
    if (
        not isinstance(validation_claim_file_sha256, str)
        or not validation_claim_file_sha256.startswith("sha256:")
        or _SHA256.fullmatch(validation_claim_file_sha256[7:]) is None
    ):
        raise OODCompletionIntegrityError("source-validation claim hash is invalid")
    if _OWNER_NONCE.fullmatch(owner_nonce) is None:
        raise OODCompletionIntegrityError("source-validation marker owner nonce is invalid")
    return canonical_json_bytes(
        {
            "artifact_type": "ecg_trust.source_validation_access_marker",
            "external_claim_file_sha256": validation_claim_file_sha256,
            "owner_nonce": owner_nonce,
            "protocol_id": "trust-sentinel-ood-completion-v1",
            "state": "SOURCE_VALIDATION_ACCESS_ARMED",
        }
    ) + b"\n"


def _mark_validation_access_armed(
    staging_root: Path,
    *,
    validation_claim_file_sha256: str,
    owner_nonce: str,
) -> None:
    _atomic_write_new(
        staging_root / _VALIDATION_ACCESS_MARKER_FILENAME,
        _validation_access_marker_bytes(
            validation_claim_file_sha256,
            owner_nonce=owner_nonce,
        ),
    )


def _validation_access_armed(staging_root: Path) -> bool:
    marker = staging_root / _VALIDATION_ACCESS_MARKER_FILENAME
    return marker.is_file() or marker.is_symlink()


def _assert_validation_access_marker_binding(
    marker_path: Path,
    *,
    validation_claim_file_sha256: str,
    owner_nonce: str,
) -> bytes:
    observed = _read_bounded_file_snapshot(
        marker_path,
        maximum_bytes=_VALIDATION_ACCESS_RECORD_MAX_BYTES,
        context="source-validation access marker",
    )
    expected = _validation_access_marker_bytes(
        validation_claim_file_sha256,
        owner_nonce=owner_nonce,
    )
    if observed != expected:
        raise OODCompletionIntegrityError("source-validation marker claim binding differs")
    return observed


def _retain_owned_post_claim_failure(
    *,
    staging_root: Path,
    output_root: Path,
    claim_state: _OneShotClaimState,
    receipt_bytes: bytes,
) -> None:
    """Retain the already-armed evidence after this process owns C."""

    claim_sha256 = _sha256_bytes(claim_state.claim_bytes)
    marker_error: Exception | None = None
    receipt_error: Exception | None = None
    try:
        _assert_validation_access_marker_binding(
            staging_root / _VALIDATION_ACCESS_MARKER_FILENAME,
            validation_claim_file_sha256=claim_sha256,
            owner_nonce=claim_state.owner_nonce,
        )
    except Exception as error:
        marker_error = error
    try:
        _atomic_write_new(
            staging_root / OOD_COMPLETION_FAILURE_FILENAME,
            receipt_bytes,
        )
    except Exception as error:
        receipt_error = error
    if marker_error is not None or receipt_error is not None:
        raise OODCompletionExecutionError(
            "owned post-claim evidence remains in staging because its marker or failure "
            "receipt could not be verified"
        ) from (marker_error if marker_error is not None else receipt_error)
    try:
        _commit_staged_directory(staging_root, output_root)
    except _OODOutputCommitError as error:
        if error.output_root_committed:
            raise OODCompletionExecutionError(
                "owned post-claim evidence was committed but its parent sync failed"
            ) from error
        raise OODCompletionExecutionError(
            "owned post-claim failure evidence remains in staging; retry is forbidden"
        ) from error


def _assert_no_marked_staging_retry(output_root: Path) -> None:
    """Reject retry when a hard-crashed run left evidence past the C boundary."""

    prefix = f".{output_root.name}.staging-"
    try:
        candidates = tuple(output_root.parent.iterdir())
    except OSError as error:
        raise OODCompletionExecutionError("could not inspect prior OOD staging roots") from error
    for candidate in candidates:
        if not candidate.name.startswith(prefix):
            continue
        marker = candidate / _VALIDATION_ACCESS_MARKER_FILENAME
        if candidate.is_symlink() or marker.is_file() or marker.is_symlink():
            raise OODCompletionExecutionError(
                "marked source-validation staging evidence exists; retry is forbidden"
            )


def _atomic_write_new(
    path: Path,
    payload: bytes,
    *,
    publication_witness: Callable[[], None] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise OODCompletionExecutionError("immutable artifact already exists")
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        if publication_witness is not None:
            publication_witness()
        _fsync_directory(path.parent)
    except FileExistsError as error:
        raise OODCompletionExecutionError("immutable artifact already exists") from error
    except OSError as error:
        raise OODCompletionExecutionError("atomic artifact commit failed") from error
    finally:
        _unlink_temporary_and_sync(temporary, directory=path.parent)


def _atomic_write_terminal_success(output_root: Path, payload: bytes) -> None:
    """Publish the success manifest as the final output-root mutation."""

    target = output_root / OOD_COMPLETION_SUCCESS_FILENAME
    if _is_link_or_junction(output_root) or not output_root.is_dir():
        raise OODCompletionExecutionError("immutable output root is unavailable")
    if target.exists() or target.is_symlink():
        raise OODCompletionExecutionError("immutable success manifest already exists")
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{output_root.name}.success-manifest-",
        suffix=".tmp",
        dir=output_root.parent,
    )
    temporary = Path(raw_temp)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        published = True
    except FileExistsError as error:
        raise OODCompletionExecutionError("immutable success manifest already exists") from error
    except OSError as error:
        raise OODCompletionExecutionError("terminal success-manifest commit failed") from error
    finally:
        if published:
            # Publication is the terminal success transition. Only best-effort
            # durability work on the parent and an adjacent, aggregate-only
            # temporary name may follow; neither can revoke visible success.
            _best_effort_sync_and_remove_terminal_temp(
                output_root=output_root,
                temporary=temporary,
            )
        else:
            _unlink_temporary_and_sync(temporary, directory=output_root.parent)


def _best_effort_sync_and_remove_terminal_temp(
    *,
    output_root: Path,
    temporary: Path,
) -> None:
    with suppress(OSError):
        _fsync_directory(output_root)
    try:
        temporary.unlink()
    except OSError:
        return
    with suppress(OSError):
        _fsync_directory(output_root.parent)


def _unlink_temporary_and_sync(temporary: Path, *, directory: Path) -> None:
    try:
        temporary.unlink()
    except FileNotFoundError:
        return
    except OSError as error:
        raise OODCompletionExecutionError("atomic temporary file could not be removed") from error
    try:
        _fsync_directory(directory)
    except OSError as error:
        raise OODCompletionExecutionError(
            "atomic temporary-file removal could not be synced"
        ) from error


def _commit_staged_directory(staging_root: Path, output_root: Path) -> None:
    if output_root.exists() or _is_link_or_junction(output_root):
        raise OODCompletionExecutionError("immutable output root already exists")
    renamed = False
    try:
        os.rename(staging_root, output_root)
        renamed = True
        _fsync_directory(output_root.parent)
    except FileExistsError as error:
        raise _OODOutputCommitError(
            "immutable output root already exists",
            output_root_committed=False,
        ) from error
    except OSError as error:
        raise _OODOutputCommitError(
            "atomic output-root commit failed",
            output_root_committed=renamed,
        ) from error


def _fsync_directory(path: Path) -> None:
    """Durably flush a directory entry on platforms that expose directory FDs."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        if os.name == "nt":
            return
        raise
    try:
        os.fsync(descriptor)
    except OSError:
        if os.name != "nt":
            raise
    finally:
        os.close(descriptor)


def _remove_staging_root(staging_root: Path, *, expected_parent: Path) -> None:
    resolved = Path(os.path.abspath(os.fspath(staging_root)))
    parent = Path(os.path.abspath(os.fspath(expected_parent)))
    if (
        resolved.parent != parent
        or not resolved.name.startswith(".")
        or ".staging-" not in resolved.name
        or _is_link_or_junction(resolved)
    ):
        raise OODCompletionExecutionError("refusing to remove an unexpected staging root")
    try:
        shutil.rmtree(resolved, ignore_errors=False)
    except FileNotFoundError:
        return
    except OSError as error:
        raise OODCompletionExecutionError("failed staging root could not be removed") from error


def _failure_code(error: BaseException) -> OODCompletionFailureCode:
    if isinstance(error, _OODFitError):
        return OODCompletionFailureCode.FIT_FAILED
    if isinstance(error, _OODValidationError):
        return OODCompletionFailureCode.VALIDATION_FAILED
    if isinstance(error, OODDeterminismError):
        return OODCompletionFailureCode.DETERMINISM_FAILED
    if isinstance(error, OODRuntimeError):
        return OODCompletionFailureCode.EMBEDDING_EXTRACTION_FAILED
    if isinstance(error, OODCompletionIntegrityError):
        return OODCompletionFailureCode.INPUT_CONTRACT_FAILED
    if isinstance(error, (ValidationError, ValueError)):
        return OODCompletionFailureCode.VALIDATION_FAILED
    if isinstance(error, OODCompletionExecutionError):
        return OODCompletionFailureCode.OUTPUT_COMMIT_FAILED
    return OODCompletionFailureCode.INTERNAL_FAILURE


__all__ = [
    "BoundProjectFile",
    "OODCompletionConfig",
    "OODCompletionConfigError",
    "OODCompletionExecutionError",
    "SavedRoleEmbeddings",
    "SealedDistributionPolicy",
    "VerifiedOODInputs",
    "VerifiedOODCompletionBundle",
    "load_ood_completion_config",
    "prepare_ood_completion",
    "verify_ood_completion_bundle",
    "verify_ood_inputs",
    "verify_research_bundle_eligible",
]
