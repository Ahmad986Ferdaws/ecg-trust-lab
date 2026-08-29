"""Immutable whole-root evidence bundles for external OOD v2.

The public result is aggregate-only.  Record identities, patient keys, scores,
embeddings, and bootstrap replicates remain in private bundle members whose
bytes are covered by the final success manifest.  The manifest is always the
last success-path write and is excluded from its own member inventory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final, cast

from ecg_trust.ood_v2.inventory import (
    CHALLENGE_2011_DATASET,
    ZZU_PEDIATRIC_DATASET,
    build_external_inventory,
    load_external_inventory,
)
from ecg_trust.ood_v2.models import (
    OOD_V2_PARENT_CONFIG_SHA256,
    OOD_V2_PROTOCOL_ID,
    OOD_V2_RESULT_FILENAME,
    OODV2Result,
    OODV2Status,
    load_ood_v2_result_bytes,
)

SUCCESS_MANIFEST_FILENAME: Final = "success-manifest.json"
FAILURE_RECEIPT_FILENAME: Final = "failure-receipt.json"
ACCESS_MARKER_FILENAME: Final = "external-access-armed.json"
SUCCESS_MANIFEST_ARTIFACT_TYPE: Final = (
    "ecg_trust.ood_external_v2_1_success_manifest"
)
ACCESS_CLAIM_FILENAME: Final = ".ood_external_v2_1.one-shot-claim.json"
ACCESS_MARKER_ARTIFACT_TYPE: Final = "ecg_trust.ood_external_v2_1_access_marker"
ACCESS_CLAIM_ARTIFACT_TYPE: Final = "ecg_trust.ood_external_v2_1_access_claim"
MAX_JSON_BYTES: Final = 8 * 1024 * 1024
MAX_PRIVATE_JSON_BYTES: Final = 64 * 1024 * 1024
QUALITY_AUDIT_SHARD_MAX_BYTES: Final = 8 * 1024 * 1024
QUALITY_AUDIT_SHARD_RECORDS: Final = 256
QUALITY_AUDIT_EXPECTED_RECORDS: Final = 13_328
QUALITY_AUDIT_SHARD_COUNT: Final = (
    QUALITY_AUDIT_EXPECTED_RECORDS + QUALITY_AUDIT_SHARD_RECORDS - 1
) // QUALITY_AUDIT_SHARD_RECORDS
QUALITY_AUDIT_INDEX_PATH: Final = "private/quality-audit-index.json"
QUALITY_AUDIT_SHARD_PATHS: Final[tuple[str, ...]] = tuple(
    f"private/quality-audit/part-{index:05d}.json"
    for index in range(QUALITY_AUDIT_SHARD_COUNT)
)
CANONICAL_SIGNAL_SIDECAR_PATH: Final = "private/canonical-signals.json"
CANONICAL_SIGNAL_NPZ_PATH: Final = "private/canonical-signals.npz"
CANONICAL_SIGNAL_SHARD_RECORDS: Final = QUALITY_AUDIT_SHARD_RECORDS
CANONICAL_SIGNAL_SHARD_COUNT: Final = QUALITY_AUDIT_SHARD_COUNT
CANONICAL_SIGNAL_MEMBER_MAX_BYTES: Final = 12_500_000

PRIVATE_MEMBER_PATHS: Final[tuple[str, ...]] = (
    "private/bootstrap-replicates.json",
    "private/bootstrap-replicates.npz",
    CANONICAL_SIGNAL_SIDECAR_PATH,
    CANONICAL_SIGNAL_NPZ_PATH,
    "private/external-inventory.json",
    "private/frozen-child-contract.json",
    "private/frozen-demo-policy.json",
    "private/frozen-distribution-policy.json",
    "private/frozen-model.ckpt",
    "private/frozen-normalization.json",
    "private/frozen-parent-config.yaml",
    "private/frozen-resolved-config.json",
    "private/frozen-source-calibration-result.json",
    "private/frozen-v1-ood-completion-result.json",
    "private/quality-pass-embeddings.json",
    "private/quality-pass-embeddings.npz",
    "private/record-evidence.json",
    "private/routing-contract.json",
    QUALITY_AUDIT_INDEX_PATH,
    *QUALITY_AUDIT_SHARD_PATHS,
)
BUNDLE_MEMBER_PATHS: Final[tuple[str, ...]] = tuple(
    sorted(
        (
            ACCESS_MARKER_FILENAME,
            OOD_V2_RESULT_FILENAME,
            *PRIVATE_MEMBER_PATHS,
        )
    )
)

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_OWNER_NONCE = re.compile(r"[0-9a-f]{64}\Z")
FROZEN_ROUTE_ORDER: Final[tuple[str, ...]] = (
    "INVALID_INPUT",
    "REACQUIRE",
    "UNSUPPORTED_INPUT",
    "ABSTAIN",
    "PREDICTION_ALLOWED",
)


class ExternalV2BundleError(ValueError):
    """Raised when a v2 bundle, claim, or manifest fails closed."""


def canonical_json_bytes(value: object) -> bytes:
    """Return canonical ASCII JSON with exactly one final newline."""

    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ExternalV2BundleError("value is not finite canonical JSON") from error
    return encoded + b"\n"


def canonical_sha256(value: object) -> str:
    """Hash a canonical JSON body without its storage newline."""

    payload = canonical_json_bytes(value)[:-1]
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return _stable_file_snapshot(path, context="bundle member").file_sha256


def _strict_digest(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ExternalV2BundleError(f"{context} must be a prefixed SHA-256 digest")
    return value


def _strict_positive_integer(value: object, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise ExternalV2BundleError(f"{context} must be a positive integer")
    return value


def _safe_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ExternalV2BundleError("bundle member path must be canonical POSIX text")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ExternalV2BundleError("bundle member path must be relative")
    if posix.as_posix() != value or any(part in {"", ".", ".."} for part in posix.parts):
        raise ExternalV2BundleError("bundle member path is not canonical")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ExternalV2BundleError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def load_canonical_json_bytes(
    payload: bytes,
    *,
    maximum_bytes: int = MAX_JSON_BYTES,
    context: str,
) -> dict[str, object]:
    """Decode an exact canonical object while rejecting duplicate keys."""

    if (
        not payload
        or len(payload) > maximum_bytes
        or not payload.endswith(b"\n")
        or payload.endswith(b"\n\n")
        or b"\r" in payload
    ):
        raise ExternalV2BundleError(f"{context} has an invalid byte contract")
    try:
        decoded: object = json.loads(
            payload[:-1].decode("ascii"),
            object_pairs_hook=_unique_object,
        )
    except ExternalV2BundleError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ExternalV2BundleError(f"{context} is not canonical JSON") from error
    if not isinstance(decoded, dict) or not all(isinstance(key, str) for key in decoded):
        raise ExternalV2BundleError(f"{context} must contain a string-keyed object")
    result = cast(dict[str, object], decoded)
    if canonical_json_bytes(result) != payload:
        raise ExternalV2BundleError(f"{context} is not in exact canonical form")
    return result


@dataclass(frozen=True, slots=True)
class ExternalV2BundleMember:
    relative_path: str
    file_sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _safe_relative_path(self.relative_path)
        _strict_digest(self.file_sha256, "member file_sha256")
        _strict_positive_integer(self.size_bytes, "member size_bytes")

    def to_dict(self) -> dict[str, object]:
        return {
            "file_sha256": self.file_sha256,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: object) -> ExternalV2BundleMember:
        if not isinstance(value, dict) or set(value) != {
            "file_sha256",
            "relative_path",
            "size_bytes",
        }:
            raise ExternalV2BundleError("success-manifest member fields differ")
        return cls(
            relative_path=_safe_relative_path(value["relative_path"]),
            file_sha256=_strict_digest(value["file_sha256"], "member file_sha256"),
            size_bytes=_strict_positive_integer(value["size_bytes"], "member size_bytes"),
        )


@dataclass(frozen=True, slots=True)
class ExternalV2SuccessManifest:
    parent_config_file_sha256: str
    child_contract_file_sha256: str
    child_contract_artifact_sha256: str
    inventory_sha256: str
    result_artifact_sha256: str
    external_claim_file_sha256: str
    code_revision: str
    members: tuple[ExternalV2BundleMember, ...]
    artifact_sha256: str

    def __post_init__(self) -> None:
        for context, digest in (
            ("parent config", self.parent_config_file_sha256),
            ("child contract", self.child_contract_file_sha256),
            ("child contract artifact", self.child_contract_artifact_sha256),
            ("inventory", self.inventory_sha256),
            ("result artifact", self.result_artifact_sha256),
            ("external claim", self.external_claim_file_sha256),
            ("manifest artifact", self.artifact_sha256),
        ):
            _strict_digest(digest, context)
        if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", self.code_revision) is None:
            raise ExternalV2BundleError("manifest code revision is invalid")
        expected_paths = BUNDLE_MEMBER_PATHS
        observed_paths = tuple(member.relative_path for member in self.members)
        if observed_paths != expected_paths:
            raise ExternalV2BundleError("manifest member inventory differs from protocol")
        if len(observed_paths) != len(set(observed_paths)):
            raise ExternalV2BundleError("manifest contains duplicate member paths")
        if self.artifact_sha256 != canonical_sha256(self.body_dict()):
            raise ExternalV2BundleError("manifest logical hash does not match its body")

    def body_dict(self) -> dict[str, object]:
        return {
            "artifact_type": SUCCESS_MANIFEST_ARTIFACT_TYPE,
            "child_contract_artifact_sha256": self.child_contract_artifact_sha256,
            "child_contract_file_sha256": self.child_contract_file_sha256,
            "code_revision": self.code_revision,
            "external_claim_file_sha256": self.external_claim_file_sha256,
            "failure_receipt_present": False,
            "inventory_sha256": self.inventory_sha256,
            "member_count": len(self.members),
            "members": [member.to_dict() for member in self.members],
            "parent_config_file_sha256": self.parent_config_file_sha256,
            "protocol_id": OOD_V2_PROTOCOL_ID,
            "result_artifact_sha256": self.result_artifact_sha256,
            "schema_version": 1,
            "status": "SUCCESS",
            "terminal_checks_complete": True,
        }

    def to_dict(self) -> dict[str, object]:
        payload = self.body_dict()
        payload["artifact_sha256"] = self.artifact_sha256
        return payload

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def create(
        cls,
        *,
        parent_config_file_sha256: str,
        child_contract_file_sha256: str,
        child_contract_artifact_sha256: str,
        inventory_sha256: str,
        result_artifact_sha256: str,
        external_claim_file_sha256: str,
        code_revision: str,
        members: tuple[ExternalV2BundleMember, ...],
    ) -> ExternalV2SuccessManifest:
        provisional = {
            "artifact_type": SUCCESS_MANIFEST_ARTIFACT_TYPE,
            "child_contract_artifact_sha256": child_contract_artifact_sha256,
            "child_contract_file_sha256": child_contract_file_sha256,
            "code_revision": code_revision,
            "external_claim_file_sha256": external_claim_file_sha256,
            "failure_receipt_present": False,
            "inventory_sha256": inventory_sha256,
            "member_count": len(members),
            "members": [member.to_dict() for member in members],
            "parent_config_file_sha256": parent_config_file_sha256,
            "protocol_id": OOD_V2_PROTOCOL_ID,
            "result_artifact_sha256": result_artifact_sha256,
            "schema_version": 1,
            "status": "SUCCESS",
            "terminal_checks_complete": True,
        }
        return cls(
            parent_config_file_sha256=parent_config_file_sha256,
            child_contract_file_sha256=child_contract_file_sha256,
            child_contract_artifact_sha256=child_contract_artifact_sha256,
            inventory_sha256=inventory_sha256,
            result_artifact_sha256=result_artifact_sha256,
            external_claim_file_sha256=external_claim_file_sha256,
            code_revision=code_revision,
            members=members,
            artifact_sha256=canonical_sha256(provisional),
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> ExternalV2SuccessManifest:
        decoded = load_canonical_json_bytes(payload, context="success manifest")
        expected = {
            "artifact_sha256",
            "artifact_type",
            "child_contract_artifact_sha256",
            "child_contract_file_sha256",
            "code_revision",
            "external_claim_file_sha256",
            "failure_receipt_present",
            "inventory_sha256",
            "member_count",
            "members",
            "parent_config_file_sha256",
            "protocol_id",
            "result_artifact_sha256",
            "schema_version",
            "status",
            "terminal_checks_complete",
        }
        if set(decoded) != expected:
            raise ExternalV2BundleError("success-manifest fields differ from protocol")
        if (
            decoded["artifact_type"] != SUCCESS_MANIFEST_ARTIFACT_TYPE
            or decoded["protocol_id"] != OOD_V2_PROTOCOL_ID
            or decoded["schema_version"] != 1
            or decoded["status"] != "SUCCESS"
            or decoded["failure_receipt_present"] is not False
            or decoded["terminal_checks_complete"] is not True
        ):
            raise ExternalV2BundleError("success-manifest identity or status is invalid")
        raw_members = decoded["members"]
        if not isinstance(raw_members, list):
            raise ExternalV2BundleError("success-manifest members must be an array")
        members = tuple(ExternalV2BundleMember.from_dict(item) for item in raw_members)
        if decoded["member_count"] != len(members):
            raise ExternalV2BundleError("success-manifest member count differs")
        return cls(
            parent_config_file_sha256=_strict_digest(
                decoded["parent_config_file_sha256"], "parent config"
            ),
            child_contract_file_sha256=_strict_digest(
                decoded["child_contract_file_sha256"], "child contract"
            ),
            child_contract_artifact_sha256=_strict_digest(
                decoded["child_contract_artifact_sha256"],
                "child contract artifact",
            ),
            inventory_sha256=_strict_digest(decoded["inventory_sha256"], "inventory"),
            result_artifact_sha256=_strict_digest(
                decoded["result_artifact_sha256"], "result artifact"
            ),
            external_claim_file_sha256=_strict_digest(
                decoded["external_claim_file_sha256"], "external claim"
            ),
            code_revision=cast(str, decoded["code_revision"]),
            members=members,
            artifact_sha256=_strict_digest(decoded["artifact_sha256"], "manifest artifact"),
        )


@dataclass(frozen=True, slots=True)
class VerifiedExternalV2Bundle:
    result: OODV2Result
    success_manifest: ExternalV2SuccessManifest
    claim_file_sha256: str


@dataclass(frozen=True, slots=True)
class _FileSystemEntryIdentity:
    device: int
    inode: int
    size_bytes: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True, slots=True)
class _StableFileSnapshot:
    identity: _FileSystemEntryIdentity
    file_sha256: str


@dataclass(frozen=True, slots=True)
class _ExactTreeSnapshot:
    directories: tuple[tuple[str, _FileSystemEntryIdentity], ...]
    files: tuple[tuple[str, _StableFileSnapshot], ...]


def build_success_manifest(
    output_root: Path,
    *,
    parent_config_file_sha256: str,
    child_contract_file_sha256: str,
    child_contract_artifact_sha256: str,
    inventory_sha256: str,
    result_artifact_sha256: str,
    external_claim_file_sha256: str,
    code_revision: str,
) -> ExternalV2SuccessManifest:
    """Hash the exact pre-manifest root after every member has been finalized."""

    _assert_exact_tree(output_root, include_success_manifest=False)
    members: list[ExternalV2BundleMember] = []
    for relative_path in BUNDLE_MEMBER_PATHS:
        path = output_root / PurePosixPath(relative_path)
        try:
            size = path.stat().st_size
        except OSError as error:
            raise ExternalV2BundleError("bundle member is unavailable") from error
        members.append(
            ExternalV2BundleMember(
                relative_path=relative_path,
                file_sha256=sha256_file(path),
                size_bytes=size,
            )
        )
    return ExternalV2SuccessManifest.create(
        parent_config_file_sha256=parent_config_file_sha256,
        child_contract_file_sha256=child_contract_file_sha256,
        child_contract_artifact_sha256=child_contract_artifact_sha256,
        inventory_sha256=inventory_sha256,
        result_artifact_sha256=result_artifact_sha256,
        external_claim_file_sha256=external_claim_file_sha256,
        code_revision=code_revision,
        members=tuple(members),
    )


def verify_external_v2_bundle(
    output_root: str | Path,
    *,
    claim_path: str | Path | None = None,
    project_root: str | Path | None = None,
    seven_zip_executable: str | Path | None = None,
) -> VerifiedExternalV2Bundle:
    """Verify the terminal bundle against its live frozen external sources."""

    root = Path(os.path.abspath(os.fspath(output_root)))
    tree_before = _exact_tree_snapshot(root, include_success_manifest=True)
    manifest_path = root / SUCCESS_MANIFEST_FILENAME
    manifest = ExternalV2SuccessManifest.from_bytes(
        _read_bounded(manifest_path, MAX_JSON_BYTES, "success manifest")
    )
    return _verify_external_v2_bundle_members(
        root,
        manifest,
        claim_path=claim_path,
        project_root=project_root,
        seven_zip_executable=seven_zip_executable,
        include_success_manifest=True,
        tree_before=tree_before,
    )


def preverify_external_v2_bundle(
    output_root: str | Path,
    manifest: ExternalV2SuccessManifest,
    *,
    claim_path: str | Path | None = None,
    project_root: str | Path | None = None,
    seven_zip_executable: str | Path | None = None,
) -> VerifiedExternalV2Bundle:
    """Run the full verifier before the terminal manifest directory entry exists.

    The supplied manifest must already be a canonical, self-consistent object.
    Its complete member inventory is verified against the finalized root.  A
    caller may therefore make the manifest bytes the final success-path write.
    """

    if not isinstance(manifest, ExternalV2SuccessManifest):
        raise TypeError("manifest must be ExternalV2SuccessManifest")
    root = Path(os.path.abspath(os.fspath(output_root)))
    tree_before = _exact_tree_snapshot(root, include_success_manifest=False)
    # Exercise the exact public byte parser before terminal publication.
    reparsed = ExternalV2SuccessManifest.from_bytes(manifest.to_bytes())
    if reparsed != manifest:
        raise ExternalV2BundleError("success manifest changed during canonical round-trip")
    return _verify_external_v2_bundle_members(
        root,
        manifest,
        claim_path=claim_path,
        project_root=project_root,
        seven_zip_executable=seven_zip_executable,
        include_success_manifest=False,
        tree_before=tree_before,
    )


def _verify_external_v2_bundle_members(
    root: Path,
    manifest: ExternalV2SuccessManifest,
    *,
    claim_path: str | Path | None,
    project_root: str | Path | None,
    seven_zip_executable: str | Path | None,
    include_success_manifest: bool,
    tree_before: _ExactTreeSnapshot,
) -> VerifiedExternalV2Bundle:
    adjacent = Path(os.path.abspath(os.fspath(root.parent / ACCESS_CLAIM_FILENAME)))
    if claim_path is not None:
        requested_claim = Path(os.path.abspath(os.fspath(claim_path)))
        if requested_claim != adjacent:
            raise ExternalV2BundleError(
                "external claim must use the exact canonical adjacent path"
            )
    claim_before = _stable_file_snapshot(
        adjacent,
        context="canonical adjacent external claim",
    )
    for member in manifest.members:
        member_path = root / PurePosixPath(member.relative_path)
        try:
            size = member_path.stat().st_size
        except OSError as error:
            raise ExternalV2BundleError("bundle member is unavailable") from error
        if size != member.size_bytes or sha256_file(member_path) != member.file_sha256:
            raise ExternalV2BundleError("bundle member differs from success manifest")

    result_bytes = _read_bounded(
        root / OOD_V2_RESULT_FILENAME,
        MAX_JSON_BYTES,
        "aggregate result",
    )
    try:
        result = load_ood_v2_result_bytes(result_bytes)
    except Exception as error:
        raise ExternalV2BundleError("aggregate result verification failed") from error
    if result.artifact_sha256 != manifest.result_artifact_sha256:
        raise ExternalV2BundleError("result identity differs from success manifest")
    if (
        result.code_revision != manifest.code_revision
        or manifest.parent_config_file_sha256 != OOD_V2_PARENT_CONFIG_SHA256
        or result.preregistration_sha256 != manifest.parent_config_file_sha256
        or result.cohort_role_manifest_sha256
        != manifest.child_contract_artifact_sha256
    ):
        raise ExternalV2BundleError("result and success-manifest lineage differs")
    hard = result.hard_gates
    if not all(
        (
            result.integrity.complete,
            hard.v1_policy_bytes_unchanged_before_and_after,
            hard.exact_v1_whole_bundle_verifier_passes,
            hard.external_raw_sources_verified_before_and_after,
            hard.exact_dataset_roots_verified,
            hard.exact_selected_input_inventory_verified_before_and_after,
            hard.semantic_roles_rederived_before_and_after,
            hard.challenge_reference_label_alignment_complete,
            hard.raw_canonical_lead_and_data_file_bindings_verified,
            hard.active_scientific_package_versions_match_child,
            hard.deterministic_repeated_embeddings_match,
            hard.raw_source_to_canonical_signal_replay_matches,
            hard.canonical_signal_to_full_backbone_embedding_replay_matches,
            hard.aggregate_only_publication_verified,
            hard.immutable_success_bundle_verifies,
            not hard.failure_receipt_exists,
        )
    ):
        raise ExternalV2BundleError(
            "terminal success bundle contains incomplete provenance or integrity gates"
        )
    try:
        private_inventory = load_external_inventory(
            root / "private" / "external-inventory.json"
        )
    except Exception as error:
        raise ExternalV2BundleError("private inventory verification failed") from error
    if private_inventory.inventory_sha256 != manifest.inventory_sha256:
        raise ExternalV2BundleError("private inventory identity differs from manifest")
    fixed_external = {
        "challenge_external_distribution_recall": (
            "PhysioNet Challenge 2011 Set A",
            "1.0.0",
            "ODC-By-1.0",
        ),
        "zzu_external_distribution_recall": (
            "ZZU pediatric ECG",
            "1",
            "CC-BY-4.0",
        ),
    }
    subsets = {
        dataset: build_external_inventory(
            tuple(record for record in private_inventory.records if record.dataset == dataset)
        )
        for dataset in (CHALLENGE_2011_DATASET, ZZU_PEDIATRIC_DATASET)
    }
    endpoint_dataset = {
        "challenge_external_distribution_recall": CHALLENGE_2011_DATASET,
        "zzu_external_distribution_recall": ZZU_PEDIATRIC_DATASET,
    }
    private_rows = _load_private_route_rows(root / "private" / "record-evidence.json")
    if len(private_rows) != len(private_inventory.records):
        raise ExternalV2BundleError("private route rows do not cover the inventory")
    for row, record in zip(private_rows, private_inventory.records, strict=True):
        if (
            row.get("dataset") != record.dataset
            or row.get("record_ref") != record.record_ref
            or row.get("patient_key") != record.patient_key
            or row.get("challenge_quality_label")
            != record.challenge_quality_label
        ):
            raise ExternalV2BundleError(
                "private route row order or inventory identity differs"
            )
    quality_pass_counts = {
        dataset: sum(
            row.get("dataset") == dataset and row.get("quality_status") == "pass"
            for row in private_rows
        )
        for dataset in (CHALLENGE_2011_DATASET, ZZU_PEDIATRIC_DATASET)
    }
    external_keys = tuple(endpoint.endpoint_key for endpoint in result.external_cohorts)
    expected_external_keys = tuple(
        key
        for key, dataset in (
            ("challenge_external_distribution_recall", CHALLENGE_2011_DATASET),
            ("zzu_external_distribution_recall", ZZU_PEDIATRIC_DATASET),
        )
        if quality_pass_counts[dataset] > 0
    )
    challenge_records = tuple(
        record
        for record in private_inventory.records
        if record.dataset == CHALLENGE_2011_DATASET
    )
    zzu_records = tuple(
        record
        for record in private_inventory.records
        if record.dataset == ZZU_PEDIATRIC_DATASET
    )
    zzu_patients = {record.patient_key for record in zzu_records}
    if (
        len(challenge_records) != 1_000
        or len(zzu_records) != 12_328
        or None in zzu_patients
        or len(zzu_patients) != 10_350
    ):
        raise ExternalV2BundleError("private inventory counts differ from protocol")
    expected_technical_counts = {
        "challenge_group3_technical_block_sensitivity": sum(
            record.challenge_quality_label == "unacceptable"
            for record in challenge_records
        ),
        "challenge_group1_quality_pass_rate": sum(
            record.challenge_quality_label == "acceptable" for record in challenge_records
        ),
    }
    if (
        expected_technical_counts["challenge_group3_technical_block_sensitivity"]
        != 225
        or expected_technical_counts["challenge_group1_quality_pass_rate"] != 773
        or sum(record.challenge_quality_label == "indeterminate" for record in challenge_records)
        != 2
    ):
        raise ExternalV2BundleError("Challenge quality-role counts differ from protocol")
    expected_technical_keys = tuple(
        key
        for key in (
            "challenge_group3_technical_block_sensitivity",
            "challenge_group1_quality_pass_rate",
        )
        if expected_technical_counts[key] > 0
    )
    technical_keys = tuple(
        endpoint.endpoint_key for endpoint in result.technical_quality_endpoints
    )
    if external_keys != expected_external_keys or technical_keys != expected_technical_keys:
        raise ExternalV2BundleError(
            "endpoint presence differs from the exact defined denominators"
        )
    four_endpoints_present = len(external_keys) == 2 and len(technical_keys) == 2
    if (
        result.status is OODV2Status.EXTERNAL_OOD_INSUFFICIENT_EVIDENCE
    ) is four_endpoints_present:
        raise ExternalV2BundleError(
            "insufficient-evidence status does not match an undefined endpoint"
        )
    for external_endpoint in result.external_cohorts:
        expected = fixed_external.get(external_endpoint.endpoint_key)
        if expected is None or (
            external_endpoint.dataset_name,
            external_endpoint.dataset_version,
            external_endpoint.license_identifier,
        ) != expected:
            raise ExternalV2BundleError("external endpoint source metadata differs")
        if (
            external_endpoint.role_assignment_sha256
            != result.cohort_role_manifest_sha256
        ):
            raise ExternalV2BundleError("external endpoint role assignment differs")
        dataset = endpoint_dataset[external_endpoint.endpoint_key]
        if (
            external_endpoint.cohort_manifest_sha256
            != subsets[dataset].inventory_sha256
            or external_endpoint.records != quality_pass_counts[dataset]
            or (
                dataset == CHALLENGE_2011_DATASET
                and external_endpoint.subjects != quality_pass_counts[dataset]
            )
        ):
            raise ExternalV2BundleError("external endpoint cohort identity differs")
    zzu_quality_pass_patients = {
        row.get("patient_key")
        for row in private_rows
        if row.get("dataset") == ZZU_PEDIATRIC_DATASET
        and row.get("quality_status") == "pass"
    }
    if None in zzu_quality_pass_patients:
        raise ExternalV2BundleError("ZZU quality-pass row has a null patient key")
    zzu_endpoint = next(
        (
            endpoint
            for endpoint in result.external_cohorts
            if endpoint.endpoint_key == "zzu_external_distribution_recall"
        ),
        None,
    )
    if zzu_endpoint is not None and zzu_endpoint.subjects != len(zzu_quality_pass_patients):
        raise ExternalV2BundleError("ZZU endpoint patient denominator differs")
    for technical_endpoint in result.technical_quality_endpoints:
        if (
            technical_endpoint.records
            != expected_technical_counts[technical_endpoint.endpoint_key]
            or technical_endpoint.subjects != technical_endpoint.records
        ):
            raise ExternalV2BundleError("technical endpoint denominator differs")

    invalid_counts = {
        dataset: sum(
            row.get("dataset") == dataset and row.get("quality_status") == "invalid"
            for row in private_rows
        )
        for dataset in (CHALLENGE_2011_DATASET, ZZU_PEDIATRIC_DATASET)
    }
    for row in private_rows:
        if (row.get("quality_status") == "invalid") is not (
            row.get("route") == "INVALID_INPUT"
        ):
            raise ExternalV2BundleError("INVALID_INPUT route semantics differ")
    group3_prediction_allowed = sum(
        row.get("dataset") == CHALLENGE_2011_DATASET
        and row.get("challenge_quality_label") == "unacceptable"
        and row.get("route") == "PREDICTION_ALLOWED"
        for row in private_rows
    )
    expected_record_coverage = quality_pass_counts[ZZU_PEDIATRIC_DATASET] / len(
        zzu_records
    )
    expected_patient_coverage = len(zzu_quality_pass_patients) / len(zzu_patients)
    if (
        hard.challenge_invalid_input_count != invalid_counts[CHALLENGE_2011_DATASET]
        or hard.challenge_quality_pass_records
        != quality_pass_counts[CHALLENGE_2011_DATASET]
        or hard.zzu_invalid_input_count != invalid_counts[ZZU_PEDIATRIC_DATASET]
        or hard.zzu_selected_records != len(zzu_records)
        or hard.zzu_quality_pass_records != quality_pass_counts[ZZU_PEDIATRIC_DATASET]
        or hard.zzu_quality_pass_record_coverage != expected_record_coverage
        or hard.zzu_selected_patients != len(zzu_patients)
        or hard.zzu_quality_pass_patients != len(zzu_quality_pass_patients)
        or hard.zzu_quality_pass_patient_coverage != expected_patient_coverage
        or hard.challenge_group3_prediction_allowed_count
        != group3_prediction_allowed
        or hard.skipped_selected_records != 0
    ):
        raise ExternalV2BundleError("result hard-gate denominators differ from private rows")

    # Local import avoids a module cycle while making the public verifier a
    # full semantic verifier: every private signal/report, embedding/score,
    # route, endpoint, and bootstrap replicate is independently rederived
    # from manifest-covered frozen routing evidence.
    try:
        from ecg_trust.ood_v2.pipeline import (
            verify_private_external_v2_bundle_semantics,
        )

        verify_private_external_v2_bundle_semantics(
            root,
            result=result,
            inventory=private_inventory,
            parent_config_file_sha256=manifest.parent_config_file_sha256,
            child_contract_file_sha256=manifest.child_contract_file_sha256,
            project_root=project_root,
            seven_zip_executable=seven_zip_executable,
        )
    except Exception as error:
        raise ExternalV2BundleError(
            "private bundle semantic verification failed"
        ) from error

    if _is_indirect(adjacent) or not adjacent.is_file():
        raise ExternalV2BundleError("canonical adjacent external claim is unavailable")
    claim_bytes = _read_bounded(adjacent, 16_384, "external access claim")
    if sha256_bytes(claim_bytes) != manifest.external_claim_file_sha256:
        raise ExternalV2BundleError("external claim differs from success manifest")
    claim = _load_access_record(claim_bytes, marker=False)
    marker = _load_access_record(
        _read_bounded(root / ACCESS_MARKER_FILENAME, 16_384, "external access marker"),
        marker=True,
    )
    if (
        claim["owner_nonce"] != marker["owner_nonce"]
        or marker["external_claim_file_sha256"] != manifest.external_claim_file_sha256
        or claim["parent_config_file_sha256"] != manifest.parent_config_file_sha256
        or claim["child_contract_file_sha256"] != manifest.child_contract_file_sha256
        or claim["inventory_sha256"] != manifest.inventory_sha256
        or marker["parent_config_file_sha256"] != manifest.parent_config_file_sha256
        or marker["child_contract_file_sha256"] != manifest.child_contract_file_sha256
        or marker["inventory_sha256"] != manifest.inventory_sha256
    ):
        raise ExternalV2BundleError("external claim and marker binding differs")
    tree_after = _exact_tree_snapshot(
        root,
        include_success_manifest=include_success_manifest,
    )
    claim_after = _stable_file_snapshot(
        adjacent,
        context="canonical adjacent external claim",
    )
    if tree_after != tree_before or claim_after != claim_before:
        raise ExternalV2BundleError(
            "bundle tree or adjacent claim changed during semantic verification"
        )
    return VerifiedExternalV2Bundle(
        result=result,
        success_manifest=manifest,
        claim_file_sha256=manifest.external_claim_file_sha256,
    )


def _load_access_record(payload: bytes, *, marker: bool) -> dict[str, object]:
    context = "external access marker" if marker else "external access claim"
    decoded = load_canonical_json_bytes(payload, maximum_bytes=16_384, context=context)
    common = {
        "artifact_type",
        "child_contract_file_sha256",
        "contains_embeddings_or_scores",
        "contains_record_or_patient_identifiers",
        "inventory_sha256",
        "owner_nonce",
        "parent_config_file_sha256",
        "protocol_id",
        "schema_version",
        "state",
    }
    expected = common | ({"external_claim_file_sha256"} if marker else set())
    if set(decoded) != expected:
        raise ExternalV2BundleError(f"{context} fields differ from protocol")
    expected_type = ACCESS_MARKER_ARTIFACT_TYPE if marker else ACCESS_CLAIM_ARTIFACT_TYPE
    expected_state = "EXTERNAL_ACCESS_ARMED" if marker else "EXTERNAL_ACCESS_CLAIMED"
    if (
        decoded["artifact_type"] != expected_type
        or decoded["protocol_id"] != OOD_V2_PROTOCOL_ID
        or decoded["schema_version"] != 1
        or decoded["state"] != expected_state
        or decoded["contains_embeddings_or_scores"] is not False
        or decoded["contains_record_or_patient_identifiers"] is not False
        or not isinstance(decoded["owner_nonce"], str)
        or _OWNER_NONCE.fullmatch(decoded["owner_nonce"]) is None
    ):
        raise ExternalV2BundleError(f"{context} identity or privacy fields are invalid")
    for key in (
        "child_contract_file_sha256",
        "inventory_sha256",
        "parent_config_file_sha256",
    ):
        _strict_digest(decoded[key], f"{context} {key}")
    if marker:
        _strict_digest(
            decoded["external_claim_file_sha256"],
            "external access marker claim hash",
        )
    return decoded


def _load_private_route_rows(path: Path) -> tuple[dict[str, object], ...]:
    payload = load_canonical_json_bytes(
        _read_bounded(path, MAX_PRIVATE_JSON_BYTES, "private record evidence"),
        maximum_bytes=MAX_PRIVATE_JSON_BYTES,
        context="private record evidence",
    )
    expected_fields = {
        "artifact_sha256",
        "artifact_type",
        "child_contract_file_sha256",
        "decision_bindings",
        "inventory_sha256",
        "parent_config_file_sha256",
        "protocol_id",
        "record_count",
        "records",
        "route_counts",
        "schema_version",
        "threshold",
    }
    if (
        set(payload) != expected_fields
        or payload.get("artifact_type")
        != "ecg_trust.ood_external_v2_1_record_evidence"
        or payload.get("protocol_id") != OOD_V2_PROTOCOL_ID
        or payload.get("schema_version") != 1
    ):
        raise ExternalV2BundleError("private record evidence identity differs")
    claimed = _strict_digest(
        payload["artifact_sha256"],
        "private record evidence artifact",
    )
    body = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    if claimed != canonical_sha256(body):
        raise ExternalV2BundleError("private record evidence self-hash differs")
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise ExternalV2BundleError("private record evidence rows are missing")
    result: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ExternalV2BundleError("private record evidence row is invalid")
        dataset = row.get("dataset")
        quality = row.get("quality_status")
        route = row.get("route")
        if dataset not in {CHALLENGE_2011_DATASET, ZZU_PEDIATRIC_DATASET} or quality not in {
            "invalid",
            "limited",
            "pass",
            "reacquire",
        } or route not in {
            "INVALID_INPUT",
            "REACQUIRE",
            "UNSUPPORTED_INPUT",
            "ABSTAIN",
            "PREDICTION_ALLOWED",
        }:
            raise ExternalV2BundleError("private route row identity is invalid")
        result.append(cast(dict[str, object], row))
    if payload.get("record_count") != len(result):
        raise ExternalV2BundleError("private route row count differs")
    observed_route_counts: dict[str, int] = {route: 0 for route in FROZEN_ROUTE_ORDER}
    for row in result:
        route = cast(str, row["route"])
        observed_route_counts[route] = observed_route_counts.get(route, 0) + 1
    if payload.get("route_counts") != observed_route_counts:
        raise ExternalV2BundleError("private route counts differ from rows")
    return tuple(result)


def _stat_identity(value: os.stat_result) -> _FileSystemEntryIdentity:
    return _FileSystemEntryIdentity(
        device=value.st_dev,
        inode=value.st_ino,
        size_bytes=value.st_size,
        modified_ns=value.st_mtime_ns,
        changed_ns=value.st_ctime_ns,
    )


def _same_open_file(left: os.stat_result, right: os.stat_result) -> bool:
    """Compare path/handle identity without Windows' divergent handle ctime."""

    return (
        left.st_dev,
        left.st_ino,
        left.st_size,
        left.st_mtime_ns,
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_size,
        right.st_mtime_ns,
    )


def _stable_file_snapshot(path: Path, *, context: str) -> _StableFileSnapshot:
    direct = _assert_direct_ancestry(path, context=context)
    if not direct.is_file():
        raise ExternalV2BundleError(f"{context} is missing or indirect")
    digest = hashlib.sha256()
    try:
        before = direct.stat()
        with direct.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if not _same_open_file(opened, before):
                raise ExternalV2BundleError(
                    f"{context} changed before its stable handle was opened"
                )
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
            closed = os.fstat(handle.fileno())
        after = direct.stat()
    except ExternalV2BundleError:
        raise
    except OSError as error:
        raise ExternalV2BundleError(f"{context} could not be hashed") from error
    identity = _stat_identity(before)
    if (
        not _same_open_file(closed, before)
        or _stat_identity(after) != identity
        or _assert_direct_ancestry(direct, context=context) != direct
    ):
        raise ExternalV2BundleError(f"{context} changed while being hashed")
    return _StableFileSnapshot(
        identity=identity,
        file_sha256="sha256:" + digest.hexdigest(),
    )


def _read_bounded(path: Path, maximum_bytes: int, context: str) -> bytes:
    direct = _assert_direct_ancestry(path, context=context)
    if not direct.is_file():
        raise ExternalV2BundleError(f"{context} is missing or indirect")
    try:
        before = direct.stat()
        if not 0 < before.st_size <= maximum_bytes:
            raise ExternalV2BundleError(f"{context} size is invalid")
        with direct.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if not _same_open_file(opened, before):
                raise ExternalV2BundleError(
                    f"{context} changed before its stable handle was opened"
                )
            payload = handle.read(maximum_bytes + 1)
            closed = os.fstat(handle.fileno())
        after = direct.stat()
    except ExternalV2BundleError:
        raise
    except OSError as error:
        raise ExternalV2BundleError(f"{context} could not be read") from error
    if (
        len(payload) != before.st_size
        or len(payload) > maximum_bytes
        or not _same_open_file(opened, before)
        or not _same_open_file(closed, before)
        or _stat_identity(after) != _stat_identity(before)
        or _assert_direct_ancestry(direct, context=context) != direct
    ):
        raise ExternalV2BundleError(f"{context} changed while being read")
    return payload


def _is_indirect(path: Path) -> bool:
    try:
        junction = getattr(path, "is_junction", None)
        return path.is_symlink() or bool(junction is not None and junction())
    except OSError as error:
        raise ExternalV2BundleError("filesystem link state could not be inspected") from error


def _assert_direct_ancestry(path: Path, *, context: str) -> Path:
    """Resolve only after rejecting every symlink or junction ancestor."""

    lexical = Path(os.path.abspath(os.fspath(path)))
    try:
        for component in (lexical, *lexical.parents):
            if _is_indirect(component):
                raise ExternalV2BundleError(
                    f"{context} contains an indirect filesystem component"
                )
        resolved = lexical.resolve(strict=True)
    except ExternalV2BundleError:
        raise
    except OSError as error:
        raise ExternalV2BundleError(f"{context} is unavailable") from error
    if resolved != lexical:
        raise ExternalV2BundleError(
            f"{context} does not resolve to its exact lexical path"
        )
    return resolved


def _assert_exact_tree(output_root: Path, *, include_success_manifest: bool) -> None:
    output_root = _assert_direct_ancestry(
        output_root,
        context="external v2 bundle root",
    )
    if not output_root.is_dir():
        raise ExternalV2BundleError("external v2 bundle root is missing or indirect")
    files: set[str] = set()
    directories: set[str] = set()
    pending = [output_root]
    while pending:
        directory = _assert_direct_ancestry(
            pending.pop(),
            context="bundle tree directory",
        )
        try:
            entries = tuple(directory.iterdir())
        except OSError as error:
            raise ExternalV2BundleError("bundle tree could not be enumerated") from error
        for entry in entries:
            entry = _assert_direct_ancestry(entry, context="bundle tree entry")
            relative = entry.relative_to(output_root).as_posix()
            if entry.is_dir():
                directories.add(relative)
                pending.append(entry)
            elif entry.is_file():
                files.add(relative)
            else:
                raise ExternalV2BundleError("bundle tree contains a non-regular entry")
    expected_files = set(BUNDLE_MEMBER_PATHS)
    if include_success_manifest:
        expected_files.add(SUCCESS_MANIFEST_FILENAME)
    if FAILURE_RECEIPT_FILENAME in files:
        raise ExternalV2BundleError("failure receipt forbids successful bundle use")
    if files != expected_files or directories != {"private", "private/quality-audit"}:
        raise ExternalV2BundleError("bundle tree differs from the exact protocol inventory")


def _exact_tree_snapshot(
    output_root: Path,
    *,
    include_success_manifest: bool,
) -> _ExactTreeSnapshot:
    """Hash and file-ID bind the complete exact tree, then enumerate it again."""

    _assert_exact_tree(output_root, include_success_manifest=include_success_manifest)
    expected_files = set(BUNDLE_MEMBER_PATHS)
    if include_success_manifest:
        expected_files.add(SUCCESS_MANIFEST_FILENAME)
    directories: list[tuple[str, _FileSystemEntryIdentity]] = []
    for relative in (".", "private", "private/quality-audit"):
        directory = output_root if relative == "." else output_root / relative
        directory = _assert_direct_ancestry(
            directory,
            context="bundle snapshot directory",
        )
        if not directory.is_dir():
            raise ExternalV2BundleError("bundle directory became unavailable")
        try:
            before = _stat_identity(directory.stat())
            tuple(directory.iterdir())
            after = _stat_identity(directory.stat())
        except OSError as error:
            raise ExternalV2BundleError("bundle directory could not be snapshotted") from error
        if before != after or _assert_direct_ancestry(
            directory,
            context="bundle snapshot directory",
        ) != directory:
            raise ExternalV2BundleError("bundle directory changed during snapshot")
        directories.append((relative, before))
    files = tuple(
        (
            relative,
            _stable_file_snapshot(
                output_root / PurePosixPath(relative),
                context="bundle snapshot member",
            ),
        )
        for relative in sorted(expected_files)
    )
    _assert_exact_tree(output_root, include_success_manifest=include_success_manifest)
    for relative, identity in directories:
        directory = output_root if relative == "." else output_root / relative
        if (
            _stat_identity(directory.stat()) != identity
            or _assert_direct_ancestry(
                directory,
                context="bundle snapshot directory",
            )
            != directory
        ):
            raise ExternalV2BundleError("bundle directory changed during tree hashing")
    return _ExactTreeSnapshot(tuple(directories), files)


__all__ = [
    "ACCESS_CLAIM_ARTIFACT_TYPE",
    "ACCESS_CLAIM_FILENAME",
    "ACCESS_MARKER_ARTIFACT_TYPE",
    "ACCESS_MARKER_FILENAME",
    "BUNDLE_MEMBER_PATHS",
    "ExternalV2BundleError",
    "ExternalV2BundleMember",
    "ExternalV2SuccessManifest",
    "FAILURE_RECEIPT_FILENAME",
    "PRIVATE_MEMBER_PATHS",
    "QUALITY_AUDIT_EXPECTED_RECORDS",
    "QUALITY_AUDIT_INDEX_PATH",
    "QUALITY_AUDIT_SHARD_COUNT",
    "QUALITY_AUDIT_SHARD_MAX_BYTES",
    "QUALITY_AUDIT_SHARD_PATHS",
    "QUALITY_AUDIT_SHARD_RECORDS",
    "SUCCESS_MANIFEST_FILENAME",
    "VerifiedExternalV2Bundle",
    "build_success_manifest",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_canonical_json_bytes",
    "preverify_external_v2_bundle",
    "sha256_bytes",
    "sha256_file",
    "verify_external_v2_bundle",
]
