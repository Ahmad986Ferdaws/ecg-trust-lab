"""Materialize one provenance-bound local demo without test-set selection."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]

from ecg_trust.decisioning import load_calibration_decisions
from ecg_trust.demo_backend import FrozenDecisionPolicy
from ecg_trust.predictions import load_prediction_artifact
from ecg_trust.protocol import (
    CALIBRATION_FOLDS,
    MODEL_SELECTION_FOLDS,
    ExperimentProtocol,
    FoldRole,
)
from ecg_trust.release_gates import (
    CalibrationBundle,
    CalibrationMember,
    RefitBundle,
    RefitMember,
    canonical_sha256,
    load_calibration_bundle,
    load_refit_bundle,
    materialize_demo_policy_payload,
    read_json_mapping,
    sha256_file,
    write_new_hashed_json,
)

DEMO_MEMBER_ID = "resnet1d-seed2026"
DEMO_TARGET_COVERAGE = 0.8
DEMO_EXAMPLE_COUNT = 5
DEMO_BINDING_SCHEMA_VERSION = 1
DEMO_BINDING_TYPE = "ecg_trust.demo_materialization_binding"
DEMO_POLICY_FILENAME = "resnet1d-seed2026.coverage80.demo-policy.json"
DEMO_EXAMPLES_FILENAME = "fold8-label-free.examples.json"
DEMO_BINDING_FILENAME = "resnet1d-seed2026.coverage80.demo-binding.json"
_EXAMPLE_COLUMNS = ("ecg_id", "patient_id", "strat_fold", "record_path")


class DemoMaterializationError(ValueError):
    """Raised when a demo cannot be bound safely to the frozen release."""


@dataclass(frozen=True, slots=True)
class DemoMaterializationResult:
    """Paths and hashes for one immutable demo materialization."""

    policy_path: Path
    policy_file_sha256: str
    examples_path: Path
    examples_file_sha256: str
    binding_path: Path
    binding_artifact_sha256: str
    binding_file_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "member_id": DEMO_MEMBER_ID,
            "target_coverage": DEMO_TARGET_COVERAGE,
            "policy": {
                "path": str(self.policy_path),
                "file_sha256": self.policy_file_sha256,
            },
            "examples": {
                "path": str(self.examples_path),
                "file_sha256": self.examples_file_sha256,
            },
            "binding": {
                "path": str(self.binding_path),
                "artifact_sha256": self.binding_artifact_sha256,
                "file_sha256": self.binding_file_sha256,
            },
        }


@dataclass(frozen=True, slots=True)
class _ExampleSelection:
    manifest_payload: dict[str, object]
    provenance: dict[str, object]


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise DemoMaterializationError(f"{context} must be a string-keyed object")
    return cast(Mapping[str, Any], value)


def _exact_keys(value: Mapping[str, object], required: set[str], context: str) -> None:
    missing = sorted(required.difference(value))
    unexpected = sorted(set(value).difference(required))
    if missing or unexpected:
        raise DemoMaterializationError(
            f"{context} keys differ; missing={missing}, unexpected={unexpected}"
        )


def _required_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DemoMaterializationError(f"{context} must be a non-empty string")
    return value


def _prefixed_hash(value: str) -> str:
    return "sha256:" + value.removeprefix("sha256:")


def _verified_file_hash(path: Path, expected: str, context: str) -> str:
    actual = sha256_file(path)
    if actual != expected.removeprefix("sha256:"):
        raise DemoMaterializationError(f"{context} changed after bundle verification")
    return "sha256:" + actual


def _find_refit_member(bundle: RefitBundle) -> RefitMember:
    matches = [member for member in bundle.members if member.member_id == DEMO_MEMBER_ID]
    if len(matches) != 1:
        raise DemoMaterializationError(
            f"refit bundle must contain exactly one {DEMO_MEMBER_ID!r} member"
        )
    member = matches[0]
    if member.architecture != "resnet1d" or member.seed != 2026:
        raise DemoMaterializationError("fixed demo member identity is inconsistent")
    return member


def _find_calibration_member(bundle: CalibrationBundle) -> CalibrationMember:
    matches = [member for member in bundle.members if member.member_id == DEMO_MEMBER_ID]
    if len(matches) != 1:
        raise DemoMaterializationError(
            f"calibration bundle must contain exactly one {DEMO_MEMBER_ID!r} member"
        )
    member = matches[0]
    if member.architecture != "resnet1d" or member.seed != 2026:
        raise DemoMaterializationError("fixed calibration member identity is inconsistent")
    return member


def _validate_cross_bundle_lineage(
    refits: RefitBundle,
    calibrations: CalibrationBundle,
    refit: RefitMember,
    calibration: CalibrationMember,
) -> None:
    if refits.artifact_sha256 is None or calibrations.artifact_sha256 is None:
        raise DemoMaterializationError("demo sources must be saved integrity-bound bundles")
    if calibrations.refit_bundle_sha256 != refits.artifact_sha256:
        raise DemoMaterializationError("calibration bundle does not bind the supplied refit bundle")
    if (
        calibrations.protocol_hash != refits.protocol_hash
        or calibrations.manifest_sha256 != refits.manifest_sha256
        or calibrations.normalization_sha256 != refits.normalization_sha256
        or calibrations.label_order != refits.label_order
    ):
        raise DemoMaterializationError("refit and calibration bundle contracts differ")
    comparisons: tuple[tuple[object, object, str], ...] = (
        (calibration.refit_lineage_sha256, refit.lineage_sha256, "refit lineage"),
        (calibration.model_name, refit.run_name, "model name"),
        (
            calibration.checkpoint_path.resolve(),
            refit.final_checkpoint_path.resolve(),
            "checkpoint",
        ),
        (calibration.checkpoint_sha256, refit.final_checkpoint_sha256, "checkpoint hash"),
        (
            calibration.resolved_config_path.resolve(),
            refit.resolved_config_path.resolve(),
            "resolved config",
        ),
        (
            calibration.resolved_config_file_sha256,
            refit.resolved_config_file_sha256,
            "resolved config file hash",
        ),
        (calibration.resolved_config_hash, refit.resolved_config_hash, "config hash"),
        (
            calibration.normalization_path.resolve(),
            refit.normalization_path.resolve(),
            "normalization",
        ),
        (
            calibration.normalization_sha256,
            refit.normalization_sha256,
            "normalization hash",
        ),
    )
    mismatches = [name for observed, expected, name in comparisons if observed != expected]
    if mismatches:
        raise DemoMaterializationError(
            "calibration member differs from the refit member: " + ", ".join(mismatches)
        )


def _selection_payload() -> dict[str, object]:
    return {
        "member_id": DEMO_MEMBER_ID,
        "architecture": "resnet1d",
        "seed": 2026,
        "target_coverage": DEMO_TARGET_COVERAGE,
        "member_rule": (
            "development-selected primary architecture and first fixed confirmation seed"
        ),
        "coverage_rule": "preexisting materializer default frozen fold-9 coverage gate",
        "fold10_predictions_read": False,
        "fold10_performance_used": False,
    }


def _verify_bound_stage_artifact(
    binding_value: object,
    *,
    context: str,
    hash_field: str,
) -> None:
    binding = _mapping(binding_value, f"{context} binding")
    path = Path(_required_string(binding.get("path"), f"{context}.path")).resolve()
    expected_file_hash = _required_string(
        binding.get("file_sha256"), f"{context}.file_sha256"
    )
    _verified_file_hash(path, expected_file_hash, context)
    payload = dict(read_json_mapping(path, context=context))
    stored = _prefixed_hash(
        _required_string(payload.pop(hash_field, None), f"{context}.{hash_field}")
    )
    expected_artifact_hash = _prefixed_hash(
        _required_string(binding.get("artifact_sha256"), f"{context}.artifact_sha256")
    )
    if canonical_sha256(payload) != stored or stored != expected_artifact_hash:
        raise DemoMaterializationError(f"{context} self-hash or binding differs")


def _verify_calibration_sources_for_demo(
    bundle: CalibrationBundle, *, protocol: ExperimentProtocol
) -> None:
    """Verify calibration sources without replaying the historical runtime envelope."""

    if bundle.stage_provenance is None:
        raise DemoMaterializationError("calibration bundle has no sealed stage provenance")
    provenance = _mapping(bundle.stage_provenance, "calibration stage provenance")
    for name, hash_field in (
        ("final_evaluation_spec", "artifact_sha256"),
        ("refit_bundle", "artifact_sha256"),
        ("fold9_export_plan", "plan_sha256"),
        ("fold9_export_completion", "artifact_sha256"),
        ("calibration_fit_plan", "plan_sha256"),
        ("calibration_fit_completion", "artifact_sha256"),
    ):
        _verify_bound_stage_artifact(
            provenance.get(name), context=name, hash_field=hash_field
        )

    for member in bundle.members:
        _verified_file_hash(
            member.checkpoint_path,
            member.checkpoint_sha256,
            f"{member.member_id} checkpoint",
        )
        _verified_file_hash(
            member.resolved_config_path,
            member.resolved_config_file_sha256,
            f"{member.member_id} resolved config",
        )
        _verified_file_hash(
            member.normalization_path,
            member.normalization_sha256,
            f"{member.member_id} normalization",
        )
        _verified_file_hash(
            member.prediction_path,
            member.prediction_npz_sha256,
            f"{member.member_id} fold-9 prediction",
        )
        _verified_file_hash(
            member.prediction_sidecar_path,
            member.prediction_sidecar_sha256,
            f"{member.member_id} fold-9 sidecar",
        )
        _verified_file_hash(
            member.decision_path,
            member.decision_file_sha256,
            f"{member.member_id} fold-9 decision",
        )
        prediction = load_prediction_artifact(
            member.prediction_path,
            protocol=protocol,
            expected_config_hash=member.resolved_config_hash,
            expected_manifest_hash=bundle.manifest_sha256,
        )
        if (
            prediction.fold_role is not FoldRole.CALIBRATION
            or prediction.folds != CALIBRATION_FOLDS
            or prediction.integrity_sha256 != member.prediction_artifact_sha256
            or prediction.alignment_sha256 != member.prediction_alignment_sha256
            or prediction.model_name != member.model_name
            or prediction.model_seed != member.seed
        ):
            raise DemoMaterializationError(
                f"{member.member_id} fold-9 prediction lineage differs"
            )
        decision = load_calibration_decisions(member.decision_path, protocol=protocol)
        decision_gates = tuple(gate.to_dict() for gate in decision.coverage_gates)
        expected_gates = tuple(dict(gate) for gate in member.entropy_gates)
        if (
            decision.integrity_sha256 != member.decision_artifact_sha256
            or decision.model_name != member.model_name
            or decision.model_seed != member.seed
            or decision.config_hash != member.resolved_config_hash
            or decision.manifest_hash != bundle.manifest_sha256
            or decision.source_prediction_sha256 != member.prediction_artifact_sha256
            or decision.source_alignment_sha256 != member.prediction_alignment_sha256
            or decision.temperature_scaling.temperature != member.temperature
            or decision.threshold_optimization.thresholds != member.thresholds
            or decision_gates != expected_gates
        ):
            raise DemoMaterializationError(
                f"{member.member_id} calibration decision lineage differs"
            )


def _source_binding_payload(
    *,
    refit_path: Path,
    calibration_path: Path,
    refits: RefitBundle,
    calibrations: CalibrationBundle,
    refit: RefitMember,
    calibration: CalibrationMember,
) -> dict[str, object]:
    checkpoint_file_sha256 = _verified_file_hash(
        refit.final_checkpoint_path,
        refit.final_checkpoint_sha256,
        "authoritative checkpoint",
    )
    resolved_config_file_sha256 = _verified_file_hash(
        refit.resolved_config_path,
        refit.resolved_config_file_sha256,
        "resolved refit config",
    )
    normalization_file_sha256 = _verified_file_hash(
        refit.normalization_path,
        refit.normalization_sha256,
        "normalization artifact",
    )
    decision_file_sha256 = _verified_file_hash(
        calibration.decision_path,
        calibration.decision_file_sha256,
        "fold-9 calibration decision",
    )
    fold9_npz_file_sha256 = _verified_file_hash(
        calibration.prediction_path,
        calibration.prediction_npz_sha256,
        "fold-9 prediction NPZ",
    )
    fold9_sidecar_file_sha256 = _verified_file_hash(
        calibration.prediction_sidecar_path,
        calibration.prediction_sidecar_sha256,
        "fold-9 prediction sidecar",
    )
    return {
        "refit_bundle": {
            "path": str(refit_path),
            "artifact_sha256": refits.artifact_sha256,
            "file_sha256": "sha256:" + sha256_file(refit_path),
        },
        "calibration_bundle": {
            "path": str(calibration_path),
            "artifact_sha256": calibrations.artifact_sha256,
            "file_sha256": "sha256:" + sha256_file(calibration_path),
            "refit_bundle_sha256": calibrations.refit_bundle_sha256,
        },
        "checkpoint": {
            "path": str(refit.final_checkpoint_path.resolve()),
            "file_sha256": checkpoint_file_sha256,
            "refit_lineage_sha256": refit.lineage_sha256,
        },
        "resolved_config": {
            "path": str(refit.resolved_config_path.resolve()),
            "file_sha256": resolved_config_file_sha256,
            "config_hash": refit.resolved_config_hash,
        },
        "normalization": {
            "path": str(refit.normalization_path.resolve()),
            "file_sha256": normalization_file_sha256,
        },
        "fold9_prediction": {
            "path": str(calibration.prediction_path.resolve()),
            "sidecar_path": str(calibration.prediction_sidecar_path.resolve()),
            "npz_file_sha256": fold9_npz_file_sha256,
            "sidecar_file_sha256": fold9_sidecar_file_sha256,
            "artifact_sha256": calibration.prediction_artifact_sha256,
            "alignment_sha256": calibration.prediction_alignment_sha256,
        },
        "fold9_decision": {
            "path": str(calibration.decision_path.resolve()),
            "file_sha256": decision_file_sha256,
            "artifact_sha256": calibration.decision_artifact_sha256,
            "independent_fit_sha256": calibration.independent_fit_sha256,
        },
    }


def _resolved_dataset_root(member: RefitMember) -> Path:
    wrapper = read_json_mapping(member.resolved_config_path, context="resolved refit config")
    config = _mapping(wrapper.get("config"), "resolved refit config body")
    data = _mapping(config.get("data"), "resolved refit data")
    dataset_root = Path(
        _required_string(data.get("dataset_root"), "resolved refit data.dataset_root")
    ).resolve()
    configured_manifest = Path(
        _required_string(data.get("manifest"), "resolved refit data.manifest")
    ).resolve()
    configured_normalization = Path(
        _required_string(data.get("normalization"), "resolved refit data.normalization")
    ).resolve()
    if configured_manifest != member.manifest_path.resolve():
        raise DemoMaterializationError("resolved config manifest differs from the refit bundle")
    if configured_normalization != member.normalization_path.resolve():
        raise DemoMaterializationError(
            "resolved config normalization differs from the refit bundle"
        )
    if not dataset_root.is_dir():
        raise DemoMaterializationError(f"resolved dataset root is missing: {dataset_root}")
    return dataset_root


def _record_root(dataset_root: Path, value: object) -> Path:
    record = Path(_required_string(value, "manifest record_path"))
    if record.is_absolute() or any(part in {"", ".", ".."} for part in record.parts):
        raise DemoMaterializationError("manifest record_path must be a safe relative path")
    resolved = (dataset_root / record).resolve()
    if not resolved.is_relative_to(dataset_root):
        raise DemoMaterializationError("manifest record_path escapes the dataset root")
    return resolved


def _select_examples(member: RefitMember, *, dataset_root: Path) -> _ExampleSelection:
    try:
        frame = pd.read_parquet(member.manifest_path, columns=list(_EXAMPLE_COLUMNS))
    except (OSError, ValueError) as error:
        raise DemoMaterializationError(
            f"could not read example source manifest: {error}"
        ) from error
    if tuple(frame.columns) != _EXAMPLE_COLUMNS:
        raise DemoMaterializationError("example source manifest columns are not canonical")
    if frame.empty or frame.loc[:, list(_EXAMPLE_COLUMNS)].isna().any(axis=None):
        raise DemoMaterializationError("example source manifest contains missing values")

    fold = MODEL_SELECTION_FOLDS[0]
    candidates = frame.loc[frame["strat_fold"] == fold].sort_values(
        ["ecg_id", "patient_id", "record_path"], kind="stable"
    )
    candidates = candidates.drop_duplicates(subset=["patient_id"], keep="first")
    if len(candidates) < DEMO_EXAMPLE_COUNT:
        raise DemoMaterializationError("model-selection fold has too few distinct example patients")

    examples: list[dict[str, str]] = []
    records: list[dict[str, object]] = []
    for row in candidates.head(DEMO_EXAMPLE_COUNT).itertuples(index=False):
        try:
            ecg_id = int(row.ecg_id)
            patient_id = int(row.patient_id)
            strat_fold = int(row.strat_fold)
        except (TypeError, ValueError) as error:
            raise DemoMaterializationError("example identities must be integers") from error
        if ecg_id < 1 or patient_id < 1 or strat_fold != fold:
            raise DemoMaterializationError("example identities violate the fold-8 contract")
        root = _record_root(dataset_root, row.record_path)
        header = root.with_suffix(".hea")
        signal = root.with_suffix(".dat")
        if not header.is_file() or not signal.is_file():
            raise DemoMaterializationError(f"example WFDB pair is incomplete: {root}")
        example_id = f"ptbxl-f8-{ecg_id:05d}"
        examples.append(
            {
                "id": example_id,
                "label": f"PTB-XL fold 8 example ECG {ecg_id}",
                "record_path": str(root),
            }
        )
        records.append(
            {
                "id": example_id,
                "ecg_id": ecg_id,
                "patient_id": patient_id,
                "strat_fold": strat_fold,
                "record_path": str(root),
                "header_file_sha256": "sha256:" + sha256_file(header),
                "signal_file_sha256": "sha256:" + sha256_file(signal),
            }
        )
    return _ExampleSelection(
        manifest_payload={"examples": examples},
        provenance={
            "source_manifest": {
                "path": str(member.manifest_path.resolve()),
                "file_sha256": _prefixed_hash(member.manifest_sha256),
            },
            "dataset_root": str(dataset_root),
            "selection_fold": list(MODEL_SELECTION_FOLDS),
            "selection_algorithm": (
                "ascending_ecg_id_then_patient_id_then_record_path; "
                "first_record_per_patient; first_five"
            ),
            "columns_read": list(_EXAMPLE_COLUMNS),
            "diagnostic_target_columns_read": False,
            "fold10_rows_eligible": False,
            "count": len(examples),
            "records": records,
        },
    )


def _serialize_json(payload: Mapping[str, object]) -> bytes:
    try:
        text = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise DemoMaterializationError("demo artifact must contain finite JSON") from error
    return (text + "\n").encode("utf-8")


def _ensure_exact_json(path: Path, payload: Mapping[str, object]) -> str:
    """Create an immutable JSON file or adopt an identical crash-recovery file."""

    expected = _serialize_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            observed = path.read_bytes()
        except OSError as error:
            raise DemoMaterializationError(
                f"could not read existing demo artifact: {error}"
            ) from error
        if observed != expected:
            raise DemoMaterializationError(f"immutable demo artifact differs: {path}")
        return "sha256:" + hashlib.sha256(observed).hexdigest()

    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            observed = path.read_bytes()
            if observed != expected:
                raise DemoMaterializationError(
                    f"immutable demo artifact differs: {path}"
                ) from error
    finally:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
    return "sha256:" + hashlib.sha256(expected).hexdigest()


def _validate_policy_payload(payload: Mapping[str, object]) -> None:
    with tempfile.TemporaryDirectory(prefix="ecg-demo-policy-validation-") as raw_directory:
        path = Path(raw_directory) / "policy.json"
        path.write_bytes(_serialize_json(payload))
        FrozenDecisionPolicy.load(path)


def _ensure_self_hashed_binding(
    path: Path, payload: Mapping[str, object]
) -> tuple[str, str]:
    digest = canonical_sha256(payload)
    committed = dict(payload)
    committed["artifact_sha256"] = digest
    if path.exists():
        file_digest = _ensure_exact_json(path, committed)
        return digest, file_digest
    saved_path, saved_digest = write_new_hashed_json(
        path, payload, hash_field="artifact_sha256"
    )
    return saved_digest, "sha256:" + sha256_file(saved_path)


def _assert_output_outside_release(
    output_directory: Path, refit_bundle_path: Path, calibration_bundle_path: Path
) -> None:
    output = output_directory.resolve()
    for source in (refit_bundle_path.resolve(), calibration_bundle_path.resolve()):
        release_root = source.parent
        if output == release_root or output.is_relative_to(release_root):
            raise DemoMaterializationError(
                "demo output must be outside the immutable release-artifact directory"
            )


def _binding_payload(
    *,
    policy_path: Path,
    policy_payload: Mapping[str, object],
    policy_file_sha256: str,
    examples_path: Path,
    examples: _ExampleSelection,
    examples_file_sha256: str,
    sources: Mapping[str, object],
    protocol_hash: str,
    label_order: tuple[str, ...],
) -> dict[str, object]:
    return {
        "schema_version": DEMO_BINDING_SCHEMA_VERSION,
        "artifact_type": DEMO_BINDING_TYPE,
        "selection": _selection_payload(),
        "policy": {
            "path": str(policy_path),
            "payload_sha256": canonical_sha256(policy_payload),
            "file_sha256": policy_file_sha256,
        },
        "examples": {
            "path": str(examples_path),
            "payload_sha256": canonical_sha256(examples.manifest_payload),
            "file_sha256": examples_file_sha256,
            **examples.provenance,
        },
        "sources": dict(sources),
        "protocol_hash": protocol_hash,
        "label_order": list(label_order),
    }


def materialize_demo(
    *,
    refit_bundle_path: str | Path,
    calibration_bundle_path: str | Path,
    output_directory: str | Path,
    protocol: ExperimentProtocol,
) -> DemoMaterializationResult:
    """Create the fixed ResNet/coverage-0.8 demo and its complete binding."""

    refit_path = Path(refit_bundle_path).resolve()
    calibration_path = Path(calibration_bundle_path).resolve()
    output = Path(output_directory).resolve()
    _assert_output_outside_release(output, refit_path, calibration_path)

    refits = load_refit_bundle(refit_path, protocol=protocol, verify_sources=True)
    # Loading verifies the bundle schema and self-hash. Source verification is
    # performed below without requiring today's code revision to equal the
    # historical final-evaluation runtime envelope.
    calibrations = load_calibration_bundle(
        calibration_path, protocol=protocol, verify_sources=False
    )
    _verify_calibration_sources_for_demo(calibrations, protocol=protocol)
    refit = _find_refit_member(refits)
    calibration = _find_calibration_member(calibrations)
    _validate_cross_bundle_lineage(refits, calibrations, refit, calibration)
    dataset_root = _resolved_dataset_root(refit)

    policy_payload = materialize_demo_policy_payload(
        calibrations,
        DEMO_MEMBER_ID,
        target_coverage=DEMO_TARGET_COVERAGE,
    )
    _validate_policy_payload(policy_payload)
    examples = _select_examples(refit, dataset_root=dataset_root)

    policy_path = output / DEMO_POLICY_FILENAME
    examples_path = output / DEMO_EXAMPLES_FILENAME
    binding_path = output / DEMO_BINDING_FILENAME
    policy_file_sha256 = _ensure_exact_json(policy_path, policy_payload)
    examples_file_sha256 = _ensure_exact_json(examples_path, examples.manifest_payload)
    FrozenDecisionPolicy.load(policy_path)

    sources = _source_binding_payload(
        refit_path=refit_path,
        calibration_path=calibration_path,
        refits=refits,
        calibrations=calibrations,
        refit=refit,
        calibration=calibration,
    )
    binding = _binding_payload(
        policy_path=policy_path,
        policy_payload=policy_payload,
        policy_file_sha256=policy_file_sha256,
        examples_path=examples_path,
        examples=examples,
        examples_file_sha256=examples_file_sha256,
        sources=sources,
        protocol_hash=refits.protocol_hash,
        label_order=refits.label_order,
    )
    binding_artifact_sha256, binding_file_sha256 = _ensure_self_hashed_binding(
        binding_path, binding
    )
    return DemoMaterializationResult(
        policy_path=policy_path,
        policy_file_sha256=policy_file_sha256,
        examples_path=examples_path,
        examples_file_sha256=examples_file_sha256,
        binding_path=binding_path,
        binding_artifact_sha256=binding_artifact_sha256,
        binding_file_sha256=binding_file_sha256,
    )


def load_and_verify_demo_binding(
    path: str | Path, *, protocol: ExperimentProtocol
) -> DemoMaterializationResult:
    """Load one demo binding and re-verify every bound source without writing."""

    binding_path = Path(path).resolve()
    if binding_path.name != DEMO_BINDING_FILENAME:
        raise DemoMaterializationError(
            f"demo binding filename must be exactly {DEMO_BINDING_FILENAME}"
        )
    committed = dict(read_json_mapping(binding_path, context="demo materialization binding"))
    _exact_keys(
        committed,
        {
            "schema_version",
            "artifact_type",
            "selection",
            "policy",
            "examples",
            "sources",
            "protocol_hash",
            "label_order",
            "artifact_sha256",
        },
        "demo materialization binding",
    )
    stored_artifact_sha256 = _prefixed_hash(
        _required_string(
            committed.pop("artifact_sha256"), "demo binding artifact_sha256"
        )
    )
    if canonical_sha256(committed) != stored_artifact_sha256:
        raise DemoMaterializationError("demo materialization binding self-hash differs")
    if committed.get("schema_version") != DEMO_BINDING_SCHEMA_VERSION or (
        committed.get("artifact_type") != DEMO_BINDING_TYPE
    ):
        raise DemoMaterializationError("unsupported demo materialization binding")
    if committed.get("protocol_hash") != protocol.protocol_hash:
        raise DemoMaterializationError("demo binding protocol differs from supplied protocol")

    sources_value = _mapping(committed.get("sources"), "demo binding sources")
    refit_binding = _mapping(sources_value.get("refit_bundle"), "refit bundle source")
    calibration_binding = _mapping(
        sources_value.get("calibration_bundle"), "calibration bundle source"
    )
    refit_path = Path(
        _required_string(refit_binding.get("path"), "refit bundle source path")
    ).resolve()
    calibration_path = Path(
        _required_string(
            calibration_binding.get("path"), "calibration bundle source path"
        )
    ).resolve()
    _assert_output_outside_release(binding_path.parent, refit_path, calibration_path)

    refits = load_refit_bundle(refit_path, protocol=protocol, verify_sources=True)
    calibrations = load_calibration_bundle(
        calibration_path, protocol=protocol, verify_sources=False
    )
    _verify_calibration_sources_for_demo(calibrations, protocol=protocol)
    refit = _find_refit_member(refits)
    calibration = _find_calibration_member(calibrations)
    _validate_cross_bundle_lineage(refits, calibrations, refit, calibration)
    dataset_root = _resolved_dataset_root(refit)

    policy_path = binding_path.parent / DEMO_POLICY_FILENAME
    examples_path = binding_path.parent / DEMO_EXAMPLES_FILENAME
    policy_payload = dict(read_json_mapping(policy_path, context="demo decision policy"))
    expected_policy = materialize_demo_policy_payload(
        calibrations,
        DEMO_MEMBER_ID,
        target_coverage=DEMO_TARGET_COVERAGE,
    )
    if policy_payload != expected_policy:
        raise DemoMaterializationError("demo policy differs from frozen calibration sources")
    FrozenDecisionPolicy.load(policy_path)
    policy_file_sha256 = "sha256:" + sha256_file(policy_path)

    examples = _select_examples(refit, dataset_root=dataset_root)
    observed_examples = dict(
        read_json_mapping(examples_path, context="deterministic demo examples")
    )
    if observed_examples != examples.manifest_payload:
        raise DemoMaterializationError("demo examples differ from deterministic selection")
    examples_file_sha256 = "sha256:" + sha256_file(examples_path)
    expected_sources = _source_binding_payload(
        refit_path=refit_path,
        calibration_path=calibration_path,
        refits=refits,
        calibrations=calibrations,
        refit=refit,
        calibration=calibration,
    )
    expected_binding = _binding_payload(
        policy_path=policy_path,
        policy_payload=policy_payload,
        policy_file_sha256=policy_file_sha256,
        examples_path=examples_path,
        examples=examples,
        examples_file_sha256=examples_file_sha256,
        sources=expected_sources,
        protocol_hash=refits.protocol_hash,
        label_order=refits.label_order,
    )
    if committed != expected_binding:
        raise DemoMaterializationError("demo binding differs from reverified sources")
    return DemoMaterializationResult(
        policy_path=policy_path,
        policy_file_sha256=policy_file_sha256,
        examples_path=examples_path,
        examples_file_sha256=examples_file_sha256,
        binding_path=binding_path,
        binding_artifact_sha256=stored_artifact_sha256,
        binding_file_sha256="sha256:" + sha256_file(binding_path),
    )


__all__ = [
    "DEMO_BINDING_TYPE",
    "DEMO_EXAMPLE_COUNT",
    "DEMO_MEMBER_ID",
    "DEMO_TARGET_COVERAGE",
    "DemoMaterializationError",
    "DemoMaterializationResult",
    "load_and_verify_demo_binding",
    "materialize_demo",
]
