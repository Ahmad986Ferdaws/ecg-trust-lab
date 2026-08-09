"""Preregistered, resumable robustness audit over the sealed fold-10 release.

The audit deliberately corrupts physical millivolt waveforms and only then
applies each member's frozen normalization.  No temperature, classification
threshold, or abstention gate is fitted here.  The exact clean-logit gate is
evaluated for all six release members before any non-clean result is accepted.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ecg_trust.audit_artifacts import (
    AuditArrayArtifact,
    AuditArtifactError,
    load_audit_array_artifact,
    save_audit_array_artifact,
)
from ecg_trust.audit_runtime import (
    AlignedAuditInference,
    AuditMemberRuntime,
    CompletedAuditRuntime,
    PhysicalBatchTransform,
)
from ecg_trust.evaluation import compute_multilabel_metrics, stable_sigmoid
from ecg_trust.post_analysis import mean_normalized_binary_entropy, multilabel_log_loss
from ecg_trust.post_evaluation import (
    PostEvaluationSpec,
    _robustness_protocol,
    canonical_sha256,
)
from ecg_trust.predictions import IdentifierArray
from ecg_trust.protocol import LABEL_ORDER
from ecg_trust.robustness import (
    amplitude_scale,
    baseline_wander,
    dc_offset,
    drop_leads,
    gaussian_noise_at_snr,
    mask_contiguous_time,
    permute_leads,
    powerline_interference,
    zero_padded_time_shift,
)

ROBUSTNESS_MEMBER_CASE_TYPE = "ecg_trust.robustness_member_case"
ROBUSTNESS_MANIFEST_TYPE = "ecg_trust.robustness_audit_manifest"
ROBUSTNESS_MANIFEST_SCHEMA_VERSION = 1
EXPECTED_CASE_COUNT = 41
EXPECTED_MEMBER_COUNT = 6
EXPECTED_MEMBER_CASE_COUNT = EXPECTED_CASE_COUNT * EXPECTED_MEMBER_COUNT
EXPECTED_SAMPLING_FREQUENCY_HZ = 100.0
EXPECTED_RANDOM_SEED = 20_260_808
EXPECTED_BOOTSTRAP_RESAMPLES = 1_000
EXPECTED_BOOTSTRAP_CONFIDENCE = 0.95
EXPECTED_BOOTSTRAP_BASE_SEED = 20_260_908
EXPECTED_CORRUPTION_DOMAIN = (
    "physical_millivolts_before_each_member_frozen_normalization"
)
EXPECTED_EXECUTION = {
    "physical_corruption_precision": "cpu_float32",
    "model_inference_precision": "bf16_as_frozen_in_final_batch",
    "metric_precision": "cpu_float64",
    "inference_batching": "reuse_frozen_final_evaluation_settings",
    "stochastic_transform_randomness": "stateless_per_ecg_sha256",
}
EXPECTED_TRANSFORM_DEFINITIONS = {
    "record_scale": "whole_record_all_12_lead_rms_per_ecg_in_physical_mv",
    "baseline_wander": "shared_across_leads_sinusoid_scaled_by_record_rms",
    "powerline": (
        "shared_across_leads_sinusoid_scaled_by_record_rms;50hz_nyquist_allowed"
    ),
    "gaussian_noise": (
        "standard_normal_draw_rescaled_per_ecg_to_exact_realized_rms_snr"
    ),
    "amplitude_scale": "multiply_all_leads_and_samples_by_factor",
    "dc_offset": "shared_constant_scaled_by_signed_fraction_of_record_rms",
    "time_shift": "zero_padded_non_circular_shift",
    "contiguous_mask": (
        "zero_half_open_interval_on_all_listed_leads;stateless_valid_start_per_ecg"
    ),
    "lead_dropout": "zero_all_samples_of_listed_leads",
    "lead_permutation": "explicit_full_canonical_lead_axis_permutation",
}
EXPECTED_DENSE_RISK_COVERAGE = {
    "ordering": "ascending_mean_normalized_binary_entropy_stable_index_tiebreak",
    "coverage_prefixes": "one_through_all_records",
    "hamming_risk": "mean_label_error_per_record_then_cumulative_prefix_mean",
    "log_loss_risk": "mean_label_log_loss_per_record_then_cumulative_prefix_mean",
    "area_method": "arithmetic_mean_over_all_prefix_coverages",
    "oracle_reference": "ascending_per_record_loss_stable_index_tiebreak",
    "random_reference": "analytical_constant_full_coverage_risk",
}
EXPECTED_BOOTSTRAP_POLICY = {
    "method": "patient_cluster_percentile_bootstrap",
    "pairing": "corrupted_minus_clean_within_member_and_patient",
    "cluster_unit": "patient_id",
    "cluster_sampling": (
        "sample_n_unique_patients_with_replacement_then_include_all_cluster_records"
    ),
    "record_weighting": "resampled_patient_multiplicity_with_all_cluster_records",
    "confidence_interval": "percentile",
}
_BOOTSTRAP_METADATA_KEYS = {
    "method",
    "resamples",
    "confidence",
    "seed",
    "pairing",
    "cluster_unit",
    "cluster_sampling",
    "record_weighting",
    "confidence_interval",
    "metric_names",
    "statistics",
    "invalid_value_storage",
}
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9+_.-]+$")
_MEMBER_CASE_ARRAYS = {
    "bootstrap_delta",
    "bootstrap_valid",
    "calibrated_probabilities",
    "dense_coverage",
    "dense_entropy_hamming_risk",
    "dense_entropy_log_loss_risk",
    "dense_entropy_order",
    "dense_oracle_hamming_risk",
    "dense_oracle_log_loss_risk",
    "dense_random_hamming_risk",
    "dense_random_log_loss_risk",
    "ecg_id",
    "gate_selected",
    "patient_id",
    "predictions",
    "raw_logits",
    "raw_probabilities",
    "targets",
    "uncertainty",
}
_MEMBER_CASE_METADATA = {
    "bootstrap",
    "case",
    "case_protocol_sha256",
    "clean_gate",
    "decision_policy",
    "delta_summary",
    "label_order",
    "member_binding_sha256",
    "member_id",
    "metric_summary",
    "n_samples",
    "post_evaluation_spec_sha256",
}
_MANIFEST_KEYS = {
    "schema_version",
    "artifact_type",
    "post_evaluation_spec_sha256",
    "post_evaluation_spec",
    "member_count",
    "case_count",
    "member_case_count",
    "clean_gate",
    "artifacts",
    "artifact_sha256",
}

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


class RobustnessAuditError(RuntimeError):
    """Raised when the preregistered robustness audit cannot proceed safely."""


class RobustnessAuditIntegrityError(RobustnessAuditError):
    """Raised when a spec, source, resumed artifact, or manifest differs."""


class RobustnessCleanGateError(RobustnessAuditIntegrityError):
    """Raised before output when the exact-six clean gate is not bit exact."""


@dataclass(frozen=True, slots=True)
class RobustnessCase:
    """One immutable case from the frozen 41-case severity matrix."""

    case_id: str
    corruption: str
    _parameters_json: str

    @property
    def parameters(self) -> dict[str, object]:
        decoded: object = json.loads(self._parameters_json)
        if not isinstance(decoded, dict):  # pragma: no cover - constructor invariant
            raise RobustnessAuditIntegrityError("case parameters are not an object")
        return cast(dict[str, object], decoded)

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "corruption": self.corruption,
            "parameters": self.parameters,
        }

    @property
    def protocol_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class RobustnessArtifactRecord:
    """Verified identity for one immutable member/case artifact pair."""

    member_id: str
    case_id: str
    artifact: AuditArrayArtifact
    npz_file_sha256: str
    sidecar_file_sha256: str

    def to_manifest_entry(self) -> dict[str, object]:
        return {
            "member_id": self.member_id,
            "case_id": self.case_id,
            "artifact_sha256": self.artifact.artifact_sha256,
            "npz": {
                "path": str(self.artifact.npz_path),
                "file_sha256": self.npz_file_sha256,
            },
            "sidecar": {
                "path": str(self.artifact.json_path),
                "file_sha256": self.sidecar_file_sha256,
            },
        }


@dataclass(frozen=True, slots=True)
class RobustnessAuditManifest:
    """Strict, self-hashed top-level manifest for all 246 artifacts."""

    path: Path
    artifact_sha256: str
    _canonical_payload: str

    @property
    def payload(self) -> dict[str, object]:
        decoded: object = json.loads(self._canonical_payload)
        if not isinstance(decoded, dict):  # pragma: no cover - constructor invariant
            raise RobustnessAuditIntegrityError("manifest is not an object")
        return cast(dict[str, object], decoded)


@dataclass(frozen=True, slots=True)
class RobustnessAuditProgress:
    """Result of a complete or deliberately partial resumable invocation."""

    completed_member_cases: int
    expected_member_cases: int
    created_member_cases: int
    resumed_member_cases: int
    manifest: RobustnessAuditManifest | None

    def to_dict(self) -> dict[str, object]:
        return {
            "completed_member_cases": self.completed_member_cases,
            "expected_member_cases": self.expected_member_cases,
            "created_member_cases": self.created_member_cases,
            "resumed_member_cases": self.resumed_member_cases,
            "complete": self.completed_member_cases == self.expected_member_cases,
            "manifest_path": str(self.manifest.path) if self.manifest else None,
            "manifest_sha256": self.manifest.artifact_sha256 if self.manifest else None,
        }


def expand_robustness_cases(spec: PostEvaluationSpec) -> tuple[RobustnessCase, ...]:
    """Parse and strictly validate the exact 41 cases frozen in ``spec``."""

    if not isinstance(spec, PostEvaluationSpec):
        raise TypeError("spec must be a PostEvaluationSpec")
    payload = spec.payload
    protocols = _mapping(payload.get("audit_protocols"), "audit_protocols")
    robustness = _mapping(protocols.get("robustness"), "robustness protocol")
    _assert_robustness_protocol_semantics(robustness)
    if dict(robustness) != _robustness_protocol():
        raise RobustnessAuditIntegrityError(
            "robustness protocol differs from the canonical frozen 41-case plan"
        )
    if robustness.get("retuning_allowed") is not False:
        raise RobustnessAuditIntegrityError("robustness retuning must be forbidden")
    if robustness.get("calibration_and_decision_policy") != (
        "reuse_each_member_frozen_temperature_thresholds_and_entropy_gates"
    ):
        raise RobustnessAuditIntegrityError("frozen decision policy differs")
    matrix = _sequence(robustness.get("severity_matrix"), "severity_matrix")
    if len(matrix) != EXPECTED_CASE_COUNT:
        raise RobustnessAuditIntegrityError("severity matrix must contain exactly 41 cases")
    cases: list[RobustnessCase] = []
    for index, raw_case in enumerate(matrix):
        case = _mapping(raw_case, f"severity_matrix[{index}]")
        if set(case) != {"case_id", "corruption", "parameters"}:
            raise RobustnessAuditIntegrityError("robustness case keys are not canonical")
        case_id = _safe_component(case.get("case_id"), "case_id")
        corruption = _string(case.get("corruption"), "corruption")
        parameters = dict(_mapping(case.get("parameters"), "parameters"))
        _validate_case_parameters(corruption, parameters)
        cases.append(
            RobustnessCase(
                case_id=case_id,
                corruption=corruption,
                _parameters_json=_canonical_json(parameters),
            )
        )
    if len({case.case_id for case in cases}) != EXPECTED_CASE_COUNT:
        raise RobustnessAuditIntegrityError("robustness case IDs must be unique")
    if cases[0].to_dict() != {
        "case_id": "clean",
        "corruption": "clean",
        "parameters": {},
    }:
        raise RobustnessAuditIntegrityError("the first robustness case must be clean")
    expected_counts = {
        "clean": 1,
        "baseline_wander": 3,
        "powerline": 3,
        "gaussian_noise": 3,
        "amplitude_scale": 4,
        "dc_offset": 4,
        "time_shift": 4,
        "contiguous_mask": 3,
        "lead_dropout": 14,
        "lead_permutation": 2,
    }
    observed = {name: sum(case.corruption == name for case in cases) for name in expected_counts}
    if observed != expected_counts:
        raise RobustnessAuditIntegrityError("robustness corruption counts differ")
    return tuple(cases)


def assert_runtime_matches_post_evaluation_spec(
    spec: PostEvaluationSpec, runtime: CompletedAuditRuntime
) -> None:
    """Fail closed unless every loaded release source equals the frozen spec.

    The post-evaluation spec loader verifies the files it names, while the
    completed runtime loader verifies the files it was asked to load.  This
    cross-binding prevents those two independently valid source sets from
    differing.
    """

    payload = spec.payload
    protocol_binding = _mapping(payload.get("protocol"), "spec protocol")
    runtime_protocol = runtime.protocol
    if protocol_binding.get("protocol_hash") != runtime_protocol.protocol_hash:
        raise RobustnessAuditIntegrityError("runtime protocol differs from the frozen spec")
    runtime_member_ids = tuple(member.member_id for member in runtime.members)
    if runtime_member_ids != spec.member_ids:
        raise RobustnessAuditIntegrityError("runtime member grid differs from the frozen spec")

    sealed = _mapping(payload.get("sealed_evaluation"), "sealed_evaluation")
    final_spec_binding = _mapping(
        sealed.get("final_evaluation_spec"), "sealed final-evaluation spec"
    )
    runtime_final_spec = runtime.final_evaluation_spec
    if runtime_final_spec.path is None:
        raise RobustnessAuditIntegrityError("runtime final-evaluation spec has no path")
    _assert_path_hash_artifact_binding(
        final_spec_binding,
        path=runtime_final_spec.path,
        artifact_sha256=runtime_final_spec.artifact_sha256,
        context="final-evaluation spec",
    )

    refit_binding = _mapping(sealed.get("refit_bundle"), "sealed refit bundle")
    calibration_binding = _mapping(
        sealed.get("calibration_bundle"), "sealed calibration bundle"
    )
    refit_bundle = runtime.refit_bundle
    calibration_bundle = runtime.calibration_bundle
    _assert_bundle_identity(
        refit_binding,
        artifact_sha256=refit_bundle.artifact_sha256,
        protocol_hash=refit_bundle.protocol_hash,
        manifest_sha256=refit_bundle.manifest_sha256,
        normalization_sha256=refit_bundle.normalization_sha256,
        member_count=len(refit_bundle.members),
        context="refit bundle",
    )
    _assert_bundle_identity(
        calibration_binding,
        artifact_sha256=calibration_bundle.artifact_sha256,
        protocol_hash=calibration_bundle.protocol_hash,
        manifest_sha256=calibration_bundle.manifest_sha256,
        normalization_sha256=calibration_bundle.normalization_sha256,
        member_count=len(calibration_bundle.members),
        context="calibration bundle",
    )
    if calibration_bundle.refit_bundle_sha256 != refit_bundle.artifact_sha256:
        raise RobustnessAuditIntegrityError(
            "runtime calibration bundle binds a different refit bundle"
        )

    ledger_binding = _mapping(sealed.get("opening_ledger"), "sealed opening ledger")
    ledger = runtime.ledger
    if (
        Path(_string(ledger_binding.get("path"), "ledger path")).resolve()
        != ledger.path.resolve()
        or _normalized_hash(ledger_binding.get("file_sha256"), "ledger file hash")
        != _file_sha256(ledger.path)
        or ledger_binding.get("ledger_sha256") != ledger.ledger_sha256
        or ledger_binding.get("batch_sha256") != ledger.batch_sha256
        or ledger_binding.get("purpose") != ledger.purpose
        or ledger_binding.get("operator") != ledger.operator
        or ledger_binding.get("state") != "complete"
        or Path(
            _string(ledger_binding.get("opening_marker_path"), "opening marker path")
        ).resolve()
        != ledger.opening_marker_path.resolve()
        or _normalized_hash(
            ledger_binding.get("opening_marker_file_sha256"),
            "opening marker file hash",
        )
        != _file_sha256(ledger.opening_marker_path)
    ):
        raise RobustnessAuditIntegrityError("runtime ledger differs from the frozen spec")

    summary_binding = _mapping(
        sealed.get("final_batch_summary"), "sealed final-batch summary"
    )
    ledger_outputs = _mapping(ledger.payload.get("outputs"), "runtime ledger outputs")
    if (
        Path(
            _string(ledger_outputs.get("batch_summary_path"), "ledger summary path")
        ).resolve()
        != Path(_string(summary_binding.get("path"), "summary path")).resolve()
        or _normalized_hash(summary_binding.get("file_sha256"), "summary file hash")
        != _file_sha256(
            Path(_string(summary_binding.get("path"), "summary path")).resolve()
        )
        or ledger_outputs.get("batch_summary_sha256")
        != summary_binding.get("artifact_sha256")
        or summary_binding.get("batch_sha256") != ledger.batch_sha256
    ):
        raise RobustnessAuditIntegrityError(
            "runtime final-batch summary differs from the frozen spec"
        )

    spec_members = {
        _string(_mapping(item, "spec member").get("member_id"), "spec member_id"): _mapping(
            item, "spec member"
        )
        for item in _sequence(payload.get("members"), "spec members")
    }
    refits = {member.member_id: member for member in refit_bundle.members}
    calibrations = {member.member_id: member for member in calibration_bundle.members}
    ledger_plan_members = {
        _string(_mapping(item, "ledger plan member").get("member_id"), "member_id"): _mapping(
            item, "ledger plan member"
        )
        for item in _sequence(ledger.plan.get("members"), "ledger plan members")
    }
    ledger_members = _mapping(ledger.members, "ledger members")
    for member in runtime.members:
        member_id = member.member_id
        frozen = spec_members[member_id]
        refit = refits[member_id]
        calibration = calibrations[member_id]
        planned = ledger_plan_members[member_id]
        completed = _mapping(ledger_members.get(member_id), "completed ledger member")
        prediction = _mapping(frozen.get("prediction"), "frozen prediction")
        checkpoint = _mapping(frozen.get("checkpoint"), "frozen checkpoint")
        config = _mapping(frozen.get("resolved_config"), "frozen resolved config")
        decision = _mapping(
            frozen.get("calibration_decision"), "frozen calibration decision"
        )
        report = _mapping(frozen.get("final_report"), "frozen final report")
        sealed_prediction = member.sealed_prediction
        if (
            frozen.get("architecture") != member.architecture
            or frozen.get("architecture") != refit.architecture
            or frozen.get("seed") != member.seed
            or frozen.get("seed") != refit.seed
            or frozen.get("model_name") != refit.run_name
            or sealed_prediction.model_name != refit.run_name
            or sealed_prediction.model_seed != refit.seed
            or sealed_prediction.protocol_hash != runtime.protocol.protocol_hash
            or sealed_prediction.config_hash != refit.resolved_config_hash
            or sealed_prediction.manifest_hash != refit.manifest_sha256
            or frozen.get("refit_lineage_sha256") != refit.lineage_sha256
            or calibration.refit_lineage_sha256 != refit.lineage_sha256
        ):
            raise RobustnessAuditIntegrityError(
                f"runtime model/lineage differs for {member_id}"
            )
        if (
            Path(_string(checkpoint.get("path"), "checkpoint path")).resolve()
            != refit.final_checkpoint_path.resolve()
            or _normalized_hash(checkpoint.get("file_sha256"), "checkpoint file hash")
            != refit.final_checkpoint_sha256
            or member.checkpoint_sha256 != refit.final_checkpoint_sha256
            or Path(_string(config.get("path"), "config path")).resolve()
            != refit.resolved_config_path.resolve()
            or _normalized_hash(config.get("file_sha256"), "config file hash")
            != refit.resolved_config_file_sha256
            or config.get("config_hash") != refit.resolved_config_hash
        ):
            raise RobustnessAuditIntegrityError(
                f"runtime checkpoint/config differs for {member_id}"
            )
        if (
            Path(_string(decision.get("path"), "decision path")).resolve()
            != calibration.decision_path.resolve()
            or _normalized_hash(decision.get("file_sha256"), "decision file hash")
            != _normalized_hash(
                calibration.decision_file_sha256,
                "runtime decision file hash",
            )
            or decision.get("artifact_sha256") != calibration.decision_artifact_sha256
            or member.decisions.integrity_sha256 != calibration.decision_artifact_sha256
        ):
            raise RobustnessAuditIntegrityError(
                f"runtime calibration decision differs for {member_id}"
            )
        prediction_path = Path(
            _string(prediction.get("npz_path"), "prediction npz path")
        ).resolve()
        sidecar_path = Path(
            _string(prediction.get("sidecar_path"), "prediction sidecar path")
        ).resolve()
        if (
            prediction_path
            != Path(_string(planned.get("final_prediction_path"), "planned prediction")).resolve()
            or prediction_path
            != Path(
                _string(completed.get("final_prediction_path"), "completed prediction")
            ).resolve()
            or _normalized_hash(prediction.get("npz_file_sha256"), "prediction file hash")
            != _normalized_hash(
                completed.get("final_prediction_file_sha256"), "ledger prediction file hash"
            )
            or _normalized_hash(
                prediction.get("sidecar_file_sha256"), "prediction sidecar hash"
            )
            != _normalized_hash(
                completed.get("final_prediction_sidecar_sha256"),
                "ledger prediction sidecar hash",
            )
            or sidecar_path != prediction_path.with_suffix(".json")
            or prediction.get("artifact_sha256") != sealed_prediction.integrity_sha256
            or prediction.get("artifact_sha256")
            != completed.get("final_prediction_artifact_sha256")
            or prediction.get("alignment_sha256") != sealed_prediction.alignment_sha256
            or prediction.get("record_count") != sealed_prediction.n_samples
        ):
            raise RobustnessAuditIntegrityError(
                f"runtime sealed prediction differs for {member_id}"
            )
        if (
            Path(_string(report.get("path"), "final report path")).resolve()
            != Path(_string(completed.get("final_report_path"), "ledger report path")).resolve()
            or report.get("artifact_sha256") != completed.get("final_report_sha256")
        ):
            raise RobustnessAuditIntegrityError(
                f"runtime final report differs for {member_id}"
            )


def stateless_seed(base_seed: int, ecg_id: object, *, purpose: str) -> int:
    """Derive a stable positive int64 seed from purpose, base seed, and ECG ID."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise RobustnessAuditError("base_seed must be a non-negative integer")
    if not isinstance(purpose, str) or not purpose.strip():
        raise RobustnessAuditError("purpose must be a non-empty string")
    identity = ecg_id.item() if isinstance(ecg_id, np.generic) else ecg_id
    if not isinstance(identity, (str, int)) or isinstance(identity, bool):
        raise RobustnessAuditError("ECG IDs must be strings or integers")
    material = f"ecg_trust:robustness:{purpose}:{base_seed}:{identity}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def build_case_transform(
    case: RobustnessCase, *, base_seed: int
) -> Callable[[Tensor, IdentifierArray], Tensor]:
    """Build a physical-mV batch transform for one frozen case."""

    parameters = case.parameters

    def transform(signals_mv: Tensor, ecg_id: IdentifierArray) -> Tensor:
        if signals_mv.ndim != 3 or signals_mv.shape[0] != len(ecg_id):
            raise RobustnessAuditError("physical batch and ECG IDs do not align")
        if signals_mv.device.type != "cpu" or signals_mv.dtype != torch.float32:
            raise RobustnessAuditError("corruptions require CPU float32 physical-mV input")
        if case.corruption == "clean":
            return signals_mv.clone()
        if case.corruption == "baseline_wander":
            return baseline_wander(
                signals_mv,
                amplitude_fraction=_number(parameters["amplitude_fraction"], "amplitude"),
                frequency_hz=_number(parameters["frequency_hz"], "frequency"),
                sampling_frequency_hz=EXPECTED_SAMPLING_FREQUENCY_HZ,
                phase_radians=_number(parameters["phase_radians"], "phase"),
            )
        if case.corruption == "powerline":
            return powerline_interference(
                signals_mv,
                amplitude_fraction=_number(parameters["amplitude_fraction"], "amplitude"),
                frequency_hz=_number(parameters["frequency_hz"], "frequency"),
                sampling_frequency_hz=EXPECTED_SAMPLING_FREQUENCY_HZ,
                phase_radians=_number(parameters["phase_radians"], "phase"),
            )
        if case.corruption == "gaussian_noise":
            output = torch.empty_like(signals_mv)
            for index, identity in enumerate(ecg_id.tolist()):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(stateless_seed(base_seed, identity, purpose="noise"))
                output[index : index + 1] = gaussian_noise_at_snr(
                    signals_mv[index : index + 1],
                    snr_db=_number(parameters["snr_db"], "snr_db"),
                    generator=generator,
                )
            return output
        if case.corruption == "amplitude_scale":
            return amplitude_scale(signals_mv, factor=_number(parameters["factor"], "factor"))
        if case.corruption == "dc_offset":
            return dc_offset(
                signals_mv,
                offset_fraction=_number(parameters["offset_fraction"], "offset_fraction"),
            )
        if case.corruption == "time_shift":
            return zero_padded_time_shift(
                signals_mv, samples=_integer(parameters["samples"], "samples", signed=True)
            )
        if case.corruption == "contiguous_mask":
            width = _integer(parameters["width_samples"], "width_samples")
            lead_indices = _integer_sequence(parameters["lead_indices"], "lead_indices")
            output = signals_mv.clone()
            possible_starts = signals_mv.shape[2] - width + 1
            if possible_starts < 1:
                raise RobustnessAuditError("mask width exceeds the signal")
            mask_seed = _integer(parameters["seed"], "mask seed")
            for index, identity in enumerate(ecg_id.tolist()):
                start = stateless_seed(mask_seed, identity, purpose="mask") % possible_starts
                output[index : index + 1] = mask_contiguous_time(
                    signals_mv[index : index + 1],
                    start_sample=start,
                    width_samples=width,
                    lead_indices=lead_indices,
                )
            return output
        if case.corruption == "lead_dropout":
            return drop_leads(
                signals_mv,
                lead_indices=_integer_sequence(parameters["lead_indices"], "lead_indices"),
            )
        if case.corruption == "lead_permutation":
            return permute_leads(
                signals_mv,
                permutation=_integer_sequence(parameters["permutation"], "permutation"),
            )
        raise RobustnessAuditIntegrityError(f"unsupported corruption {case.corruption!r}")

    return transform


def run_robustness_audit(
    *,
    spec: PostEvaluationSpec,
    runtime: CompletedAuditRuntime,
    member_ids: Sequence[str] | None = None,
    case_ids: Sequence[str] | None = None,
    finalize: bool = True,
    progress: Callable[[str], None] | None = None,
) -> RobustnessAuditProgress:
    """Run or safely resume member/case artifacts and conditionally finalize."""

    cases = preflight_robustness_artifact_paths(spec)
    all_member_ids = spec.member_ids
    runtime_ids = tuple(member.member_id for member in runtime.members)
    if (
        len(all_member_ids) != EXPECTED_MEMBER_COUNT
        or runtime_ids != all_member_ids
        or len(set(all_member_ids)) != EXPECTED_MEMBER_COUNT
    ):
        raise RobustnessAuditIntegrityError("runtime/spec membership is not the ordered exact six")
    assert_runtime_matches_post_evaluation_spec(spec, runtime)

    # This repeat is intentional.  It occurs before output discovery or creation,
    # so even resumed corruptions are accepted only after a fresh exact-six gate.
    try:
        clean_evidence = runtime.assert_clean_logit_equivalence()
    except Exception as error:
        raise RobustnessCleanGateError(f"exact-six clean gate failed: {error}") from error
    if tuple(item.member_id for item in clean_evidence) != all_member_ids or any(
        not item.exact
        or item.mismatch_count != 0
        or item.maximum_absolute_error != 0.0
        or item.mean_absolute_error != 0.0
        for item in clean_evidence
    ):
        raise RobustnessCleanGateError("exact-six clean evidence is not bit exact")
    if progress is not None:
        progress("exact-six clean-logit gate passed")

    selected_members = _selection(member_ids, all_member_ids, "member")
    all_case_ids = tuple(case.case_id for case in cases)
    selected_cases = _selection(case_ids, all_case_ids, "case")
    selected_case_set = set(selected_cases)
    paths = _audit_paths(spec)
    manifest_path, robustness_root = paths

    if manifest_path.exists():
        existing_manifest = load_robustness_manifest(
            manifest_path, spec=spec, verify_sources=True
        )
        if progress is not None:
            progress("verified existing complete 246-unit robustness manifest")
        return RobustnessAuditProgress(
            completed_member_cases=EXPECTED_MEMBER_CASE_COUNT,
            expected_member_cases=EXPECTED_MEMBER_CASE_COUNT,
            created_member_cases=0,
            resumed_member_cases=EXPECTED_MEMBER_CASE_COUNT,
            manifest=existing_manifest,
        )

    base_seed, resamples, confidence, bootstrap_base_seed = _robustness_settings(spec)
    created = 0
    resumed = 0
    clean_records: dict[str, RobustnessArtifactRecord] = {}
    selected_non_clean = sum(
        case.case_id in selected_case_set for case in cases[1:]
    )
    selected_unit_count = len(selected_members) * (1 + selected_non_clean)

    def report(record: RobustnessArtifactRecord, was_created: bool) -> None:
        if progress is None:
            return
        completed = created + resumed
        disposition = "created" if was_created else "resumed"
        progress(
            f"[{completed}/{selected_unit_count}] {disposition} "
            f"{record.member_id}/{record.case_id}"
        )

    # Materialize/resume every selected member's clean artifact first.
    for member_id in selected_members:
        member = runtime.member(member_id)
        clean_case = cases[0]
        record, was_created = _load_or_create_case(
            spec=spec,
            member=member,
            case=clean_case,
            clean=None,
            robustness_root=robustness_root,
            base_seed=base_seed,
            bootstrap_resamples=resamples,
            bootstrap_confidence=confidence,
            bootstrap_base_seed=bootstrap_base_seed,
        )
        clean_records[member_id] = record
        created += int(was_created)
        resumed += int(not was_created)
        report(record, was_created)

    for member_id in selected_members:
        member = runtime.member(member_id)
        for case in cases[1:]:
            if case.case_id not in selected_case_set:
                continue
            record, was_created = _load_or_create_case(
                spec=spec,
                member=member,
                case=case,
                clean=clean_records[member_id],
                robustness_root=robustness_root,
                base_seed=base_seed,
                bootstrap_resamples=resamples,
                bootstrap_confidence=confidence,
                bootstrap_base_seed=bootstrap_base_seed,
            )
            created += int(was_created)
            resumed += int(not was_created)
            report(record, was_created)

    if progress is not None:
        progress("verifying all discoverable immutable member/case artifacts")
    complete_records = _discover_all_records(
        spec=spec,
        runtime=runtime,
        cases=cases,
        robustness_root=robustness_root,
    )
    manifest: RobustnessAuditManifest | None = None
    if len(complete_records) == EXPECTED_MEMBER_CASE_COUNT and finalize:
        if progress is not None:
            progress("publishing verified complete robustness manifest")
        manifest = save_robustness_manifest(
            manifest_path,
            spec=spec,
            records=complete_records,
            clean_evidence=[item.to_dict() for item in clean_evidence],
        )
    return RobustnessAuditProgress(
        completed_member_cases=len(complete_records),
        expected_member_cases=EXPECTED_MEMBER_CASE_COUNT,
        created_member_cases=created,
        resumed_member_cases=resumed,
        manifest=manifest,
    )


def preflight_robustness_artifact_paths(
    spec: PostEvaluationSpec,
) -> tuple[RobustnessCase, ...]:
    """Prove the complete frozen member/case path mapping is injective and contained."""

    cases = expand_robustness_cases(spec)
    _, robustness_root = _audit_paths(spec)
    root = robustness_root.resolve()
    owners: dict[Path, str] = {}
    for member_id in spec.member_ids:
        for case in cases:
            npz_path = _case_npz_path(root, member_id, case.case_id).resolve()
            bindings = (
                (npz_path, "npz"),
                (npz_path.with_suffix(".json").resolve(), "sidecar"),
            )
            for path, kind in bindings:
                owner = f"{member_id}/{case.case_id}/{kind}"
                if root not in path.parents:
                    raise RobustnessAuditIntegrityError(
                        f"robustness artifact path escapes root: {owner} -> {path}"
                    )
                previous = owners.get(path)
                if previous is not None:
                    raise RobustnessAuditIntegrityError(
                        "robustness artifact path collision: "
                        f"{previous} and {owner} both resolve to {path}"
                    )
                owners[path] = owner
    expected = len(spec.member_ids) * EXPECTED_CASE_COUNT * 2
    if len(owners) != expected:  # pragma: no cover - loop/collision invariant
        raise RobustnessAuditIntegrityError(
            f"robustness artifact preflight expected {expected} unique paths, "
            f"observed {len(owners)}"
        )
    return cases


def save_robustness_manifest(
    path: str | Path,
    *,
    spec: PostEvaluationSpec,
    records: Sequence[RobustnessArtifactRecord],
    clean_evidence: Sequence[Mapping[str, object]],
) -> RobustnessAuditManifest:
    """Atomically save the final manifest only for the verified 6 x 41 grid."""

    destination = Path(path).resolve()
    canonical_path, robustness_root = _audit_paths(spec)
    if destination != canonical_path:
        raise RobustnessAuditIntegrityError("manifest path differs from the frozen contract")
    cases = expand_robustness_cases(spec)
    expected_pairs = [
        (member_id, case.case_id) for member_id in spec.member_ids for case in cases
    ]
    observed_pairs = [(record.member_id, record.case_id) for record in records]
    if observed_pairs != expected_pairs or len(records) != EXPECTED_MEMBER_CASE_COUNT:
        raise RobustnessAuditIntegrityError("manifest requires the ordered complete 246 grid")
    case_lookup = {case.case_id: case for case in cases}
    clean_hashes: dict[str, str] = {}
    for record in records:
        if (
            _file_sha256(record.artifact.npz_path) != record.npz_file_sha256
            or _file_sha256(record.artifact.json_path) != record.sidecar_file_sha256
        ):
            raise RobustnessAuditIntegrityError(
                "member/case source changed before manifest publication"
            )
        verified = _load_expected_artifact(
            record.artifact.npz_path,
            spec=spec,
            member_id=record.member_id,
            case=case_lookup[record.case_id],
            member_binding_sha256=_member_binding_sha256(spec, record.member_id),
            robustness_root=robustness_root,
        )
        if verified.artifact_sha256 != record.artifact.artifact_sha256:
            raise RobustnessAuditIntegrityError(
                "member/case artifact changed before manifest publication"
            )
        verified_gate = _mapping(verified.metadata["clean_gate"], "clean gate")
        if record.case_id == "clean":
            if verified_gate.get("clean_artifact_sha256") is not None:
                raise RobustnessCleanGateError("clean artifact has a clean parent")
            clean_hashes[record.member_id] = verified.artifact_sha256
        elif verified_gate.get("clean_artifact_sha256") != clean_hashes.get(
            record.member_id
        ):
            raise RobustnessCleanGateError(
                "corruption does not bind its verified member clean artifact"
            )
    if spec.path is None or not spec.path.is_file():
        raise RobustnessAuditIntegrityError("saved post-evaluation spec is required")
    evidence = [dict(item) for item in clean_evidence]
    if len(evidence) != EXPECTED_MEMBER_COUNT or any(
        item.get("member_id") != spec.member_ids[index]
        or item.get("exact") is not True
        or item.get("mismatch_count") != 0
        or item.get("maximum_absolute_error") != 0.0
        for index, item in enumerate(evidence)
    ):
        raise RobustnessCleanGateError("manifest clean-gate evidence is incomplete")
    body: dict[str, object] = {
        "schema_version": ROBUSTNESS_MANIFEST_SCHEMA_VERSION,
        "artifact_type": ROBUSTNESS_MANIFEST_TYPE,
        "post_evaluation_spec_sha256": spec.artifact_sha256,
        "post_evaluation_spec": {
            "path": str(spec.path.resolve()),
            "file_sha256": _file_sha256(spec.path),
        },
        "member_count": EXPECTED_MEMBER_COUNT,
        "case_count": EXPECTED_CASE_COUNT,
        "member_case_count": EXPECTED_MEMBER_CASE_COUNT,
        "clean_gate": {
            "comparison": "np.array_equal_against_sealed_raw_logits",
            "maximum_absolute_logit_error": 0.0,
            "all_six_passed_before_corruptions": True,
            "members": evidence,
        },
        "artifacts": [record.to_manifest_entry() for record in records],
    }
    payload = {**body, "artifact_sha256": canonical_sha256(body)}
    _write_new_json(destination, payload)
    return load_robustness_manifest(destination, spec=spec, verify_sources=True)


def load_robustness_manifest(
    path: str | Path,
    *,
    spec: PostEvaluationSpec,
    verify_sources: bool = True,
) -> RobustnessAuditManifest:
    """Load a strict manifest and verify every nested file binding."""

    source = Path(path).resolve()
    canonical_path, robustness_root = _audit_paths(spec)
    if source != canonical_path:
        raise RobustnessAuditIntegrityError("manifest is outside the frozen output contract")
    root = _read_json(source, "robustness manifest")
    if set(root) != _MANIFEST_KEYS:
        raise RobustnessAuditIntegrityError("manifest keys are not canonical")
    if root.get("schema_version") != ROBUSTNESS_MANIFEST_SCHEMA_VERSION or root.get(
        "artifact_type"
    ) != ROBUSTNESS_MANIFEST_TYPE:
        raise RobustnessAuditIntegrityError("manifest schema/type differs")
    stored_hash = _hash(root.get("artifact_sha256"), "manifest artifact_sha256")
    body = dict(root)
    del body["artifact_sha256"]
    if canonical_sha256(body) != stored_hash:
        raise RobustnessAuditIntegrityError("manifest self-hash mismatch")
    if root.get("post_evaluation_spec_sha256") != spec.artifact_sha256:
        raise RobustnessAuditIntegrityError("manifest post-evaluation spec hash differs")
    spec_binding = _validate_file_binding(
        root.get("post_evaluation_spec"), "post_evaluation_spec"
    )
    bound_spec_path = Path(_string(spec_binding["path"], "spec path")).resolve()
    if spec.path is None or bound_spec_path != spec.path.resolve():
        raise RobustnessAuditIntegrityError("manifest post-evaluation spec path differs")
    if (
        root.get("member_count") != EXPECTED_MEMBER_COUNT
        or root.get("case_count") != EXPECTED_CASE_COUNT
        or root.get("member_case_count") != EXPECTED_MEMBER_CASE_COUNT
    ):
        raise RobustnessAuditIntegrityError("manifest completion counts differ")
    cases = expand_robustness_cases(spec)
    artifacts = _sequence(root.get("artifacts"), "manifest artifacts")
    expected_pairs = [
        (member_id, case.case_id) for member_id in spec.member_ids for case in cases
    ]
    observed_pairs: list[tuple[str, str]] = []
    for raw in artifacts:
        entry = _mapping(raw, "manifest artifact")
        if set(entry) != {"member_id", "case_id", "artifact_sha256", "npz", "sidecar"}:
            raise RobustnessAuditIntegrityError("manifest artifact keys differ")
        member_id = _string(entry.get("member_id"), "manifest member_id")
        case_id = _string(entry.get("case_id"), "manifest case_id")
        observed_pairs.append((member_id, case_id))
        _hash(entry.get("artifact_sha256"), "member/case artifact hash")
        for key in ("npz", "sidecar"):
            _validate_file_binding(entry.get(key), f"manifest {key}")
    if observed_pairs != expected_pairs or len(artifacts) != EXPECTED_MEMBER_CASE_COUNT:
        raise RobustnessAuditIntegrityError("manifest does not contain the ordered 246 grid")
    clean_gate = _mapping(root.get("clean_gate"), "manifest clean gate")
    if (
        clean_gate.get("comparison") != "np.array_equal_against_sealed_raw_logits"
        or clean_gate.get("maximum_absolute_logit_error") != 0.0
        or clean_gate.get("all_six_passed_before_corruptions") is not True
    ):
        raise RobustnessCleanGateError("manifest clean gate differs")
    evidence = _sequence(clean_gate.get("members"), "manifest clean members")
    if len(evidence) != EXPECTED_MEMBER_COUNT or any(
        _mapping(item, "manifest clean member").get("member_id")
        != spec.member_ids[index]
        or _mapping(item, "manifest clean member").get("exact") is not True
        or _mapping(item, "manifest clean member").get("mismatch_count") != 0
        or _mapping(item, "manifest clean member").get("maximum_absolute_error")
        != 0.0
        for index, item in enumerate(evidence)
    ):
        raise RobustnessCleanGateError("manifest exact-six evidence differs")
    if verify_sources:
        _verify_nested_file_bindings(root)
        case_lookup = {case.case_id: case for case in cases}
        clean_hashes: dict[str, str] = {}
        for raw in artifacts:
            entry = _mapping(raw, "manifest artifact")
            npz = _mapping(entry["npz"], "manifest npz")
            member_id = _string(entry["member_id"], "member_id")
            case_id = _string(entry["case_id"], "case_id")
            artifact = _load_expected_artifact(
                Path(_string(npz["path"], "npz path")),
                spec=spec,
                member_id=member_id,
                case=case_lookup[case_id],
                member_binding_sha256=_member_binding_sha256(spec, member_id),
                robustness_root=robustness_root,
            )
            if artifact.artifact_sha256 != entry["artifact_sha256"]:
                raise RobustnessAuditIntegrityError("nested artifact hash differs")
            artifact_gate = _mapping(artifact.metadata["clean_gate"], "clean gate")
            if case_id == "clean":
                if artifact_gate.get("clean_artifact_sha256") is not None:
                    raise RobustnessCleanGateError("clean artifact has a clean parent")
                clean_hashes[member_id] = artifact.artifact_sha256
            elif artifact_gate.get("clean_artifact_sha256") != clean_hashes.get(member_id):
                raise RobustnessCleanGateError(
                    "corruption does not bind its member's manifest clean artifact"
                )
    return RobustnessAuditManifest(
        path=source,
        artifact_sha256=stored_hash,
        _canonical_payload=_canonical_json(root),
    )


def compute_member_case_arrays(
    *,
    inference: AlignedAuditInference,
    clean_logits: FloatArray,
    temperature: float,
    thresholds: Sequence[float],
    gates: Sequence[Mapping[str, object]],
    bootstrap_resamples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
    clean_case_arrays: Mapping[str, NDArray[np.generic]] | None,
) -> tuple[dict[str, NDArray[np.generic]], dict[str, object]]:
    """Compute all FP64/CPU metrics and paired patient-cluster bootstrap arrays."""

    logits = np.asarray(inference.raw_logits, dtype=np.float64)
    baseline_logits = np.asarray(clean_logits, dtype=np.float64)
    targets = np.asarray(inference.targets, dtype=np.int8)
    if logits.shape != baseline_logits.shape or logits.shape != targets.shape:
        raise RobustnessAuditIntegrityError("case and clean arrays do not align")
    if not np.all(np.isfinite(logits)) or not np.all(np.isfinite(baseline_logits)):
        raise RobustnessAuditIntegrityError("robustness logits must be finite")
    threshold_array = np.asarray(tuple(thresholds), dtype=np.float64)
    if threshold_array.shape != (len(LABEL_ORDER),):
        raise RobustnessAuditIntegrityError("frozen thresholds must contain five values")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise RobustnessAuditIntegrityError("frozen temperature must be positive")
    gate_values = tuple(dict(gate) for gate in gates)
    if not gate_values:
        raise RobustnessAuditIntegrityError("at least one frozen gate is required")

    raw_probabilities = stable_sigmoid(logits)
    calibrated = stable_sigmoid(logits / temperature)
    uncertainty = mean_normalized_binary_entropy(calibrated)
    predictions = calibrated >= threshold_array[None, :]
    gate_selected = np.column_stack(
        [
            uncertainty <= _number(gate.get("maximum_entropy"), "maximum_entropy")
            for gate in gate_values
        ]
    ).astype(np.bool_, copy=False)
    dense = _dense_curves(targets, calibrated, predictions, uncertainty)
    summary = _case_summary(
        targets=targets,
        logits=logits,
        clean_logits=baseline_logits,
        raw_probabilities=raw_probabilities,
        calibrated_probabilities=calibrated,
        predictions=predictions,
        uncertainty=uncertainty,
        gate_selected=gate_selected,
        gates=gate_values,
        dense=dense,
    )

    if clean_case_arrays is None:
        clean_raw = raw_probabilities
        clean_calibrated = calibrated
        clean_uncertainty = uncertainty
        clean_predictions = predictions
        clean_gate_selected = gate_selected
        clean_dense = dense
    else:
        clean_raw = np.asarray(clean_case_arrays["raw_probabilities"], dtype=np.float64)
        clean_calibrated = np.asarray(
            clean_case_arrays["calibrated_probabilities"], dtype=np.float64
        )
        clean_uncertainty = np.asarray(clean_case_arrays["uncertainty"], dtype=np.float64)
        clean_predictions = np.asarray(clean_case_arrays["predictions"], dtype=np.bool_)
        clean_gate_selected = np.asarray(clean_case_arrays["gate_selected"], dtype=np.bool_)
        clean_dense = {
            "aurc_hamming": float(
                np.asarray(clean_case_arrays["dense_entropy_hamming_risk"]).mean()
            ),
            "aurc_log_loss": float(
                np.asarray(clean_case_arrays["dense_entropy_log_loss_risk"]).mean()
            ),
            "oracle_aurc_hamming": float(
                np.asarray(clean_case_arrays["dense_oracle_hamming_risk"]).mean()
            ),
            "oracle_aurc_log_loss": float(
                np.asarray(clean_case_arrays["dense_oracle_log_loss_risk"]).mean()
            ),
            "random_aurc_hamming": float(
                np.asarray(clean_case_arrays["dense_random_hamming_risk"]).mean()
            ),
            "random_aurc_log_loss": float(
                np.asarray(clean_case_arrays["dense_random_log_loss_risk"]).mean()
            ),
        }
    clean_summary = _case_summary(
        targets=targets,
        logits=baseline_logits,
        clean_logits=baseline_logits,
        raw_probabilities=clean_raw,
        calibrated_probabilities=clean_calibrated,
        predictions=clean_predictions,
        uncertainty=clean_uncertainty,
        gate_selected=clean_gate_selected,
        gates=gate_values,
        dense=clean_dense,
    )
    delta_summary = _summary_delta(summary, clean_summary)

    metric_names = _bootstrap_metric_names(len(gate_values))
    bootstrap_delta, bootstrap_valid = _paired_patient_bootstrap(
        targets=targets,
        patient_id=np.asarray(inference.patient_id),
        raw_probabilities=raw_probabilities,
        calibrated_probabilities=calibrated,
        predictions=predictions,
        uncertainty=uncertainty,
        gate_selected=gate_selected,
        clean_raw_probabilities=clean_raw,
        clean_calibrated_probabilities=clean_calibrated,
        clean_predictions=clean_predictions,
        clean_uncertainty=clean_uncertainty,
        clean_gate_selected=clean_gate_selected,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    bootstrap_summary = _bootstrap_summary(
        metric_names,
        bootstrap_delta,
        bootstrap_valid,
        confidence=bootstrap_confidence,
    )
    arrays: dict[str, NDArray[np.generic]] = {
        "ecg_id": np.asarray(inference.ecg_id),
        "patient_id": np.asarray(inference.patient_id),
        "targets": targets,
        "raw_logits": logits,
        "raw_probabilities": raw_probabilities,
        "calibrated_probabilities": calibrated,
        "uncertainty": uncertainty,
        "predictions": predictions,
        "gate_selected": gate_selected,
        "dense_coverage": cast(FloatArray, dense["coverage"]),
        "dense_entropy_order": cast(IntArray, dense["entropy_order"]),
        "dense_entropy_hamming_risk": cast(FloatArray, dense["entropy_hamming"]),
        "dense_entropy_log_loss_risk": cast(FloatArray, dense["entropy_log_loss"]),
        "dense_oracle_hamming_risk": cast(FloatArray, dense["oracle_hamming"]),
        "dense_oracle_log_loss_risk": cast(FloatArray, dense["oracle_log_loss"]),
        "dense_random_hamming_risk": cast(FloatArray, dense["random_hamming"]),
        "dense_random_log_loss_risk": cast(FloatArray, dense["random_log_loss"]),
        "bootstrap_delta": bootstrap_delta,
        "bootstrap_valid": bootstrap_valid,
    }
    analysis: dict[str, object] = {
        "metric_summary": summary,
        "delta_summary": delta_summary,
        "bootstrap": {
            "method": "patient_cluster_percentile_bootstrap",
            "resamples": bootstrap_resamples,
            "confidence": bootstrap_confidence,
            "seed": bootstrap_seed,
            "pairing": "corrupted_minus_clean_within_member_and_patient",
            "cluster_unit": "patient_id",
            "cluster_sampling": (
                "sample_n_unique_patients_with_replacement_then_include_all_cluster_records"
            ),
            "record_weighting": (
                "resampled_patient_multiplicity_with_all_cluster_records"
            ),
            "confidence_interval": "percentile",
            "metric_names": list(metric_names),
            "statistics": bootstrap_summary,
            "invalid_value_storage": (
                "zero_placeholder_with_parallel_bootstrap_valid_mask"
            ),
        },
    }
    return arrays, analysis


def _load_or_create_case(
    *,
    spec: PostEvaluationSpec,
    member: AuditMemberRuntime,
    case: RobustnessCase,
    clean: RobustnessArtifactRecord | None,
    robustness_root: Path,
    base_seed: int,
    bootstrap_resamples: int,
    bootstrap_confidence: float,
    bootstrap_base_seed: int,
) -> tuple[RobustnessArtifactRecord, bool]:
    npz_path = _case_npz_path(robustness_root, member.member_id, case.case_id)
    json_path = npz_path.with_suffix(".json")
    member_hash = _member_binding_sha256(spec, member.member_id)
    if npz_path.exists() or json_path.exists():
        if not (npz_path.is_file() and json_path.is_file()):
            raise RobustnessAuditIntegrityError("partial member/case artifact cannot resume")
        artifact = _load_expected_artifact(
            npz_path,
            spec=spec,
            member_id=member.member_id,
            case=case,
            member_binding_sha256=member_hash,
            robustness_root=robustness_root,
        )
        _assert_artifact_alignment(artifact, member)
        if case.corruption == "clean" and not np.array_equal(
            artifact.arrays["raw_logits"], member.sealed_prediction.raw_logits
        ):
            raise RobustnessCleanGateError(
                "resumed clean artifact logits differ from sealed logits"
            )
        if clean is not None:
            resumed_gate = _mapping(artifact.metadata["clean_gate"], "clean gate")
            if resumed_gate.get("clean_artifact_sha256") != clean.artifact.artifact_sha256:
                raise RobustnessCleanGateError(
                    "resumed corruption binds a different clean artifact"
                )
        return _artifact_record(member.member_id, case.case_id, artifact), False

    if case.corruption == "clean":
        sealed = member.sealed_prediction
        inference = AlignedAuditInference(
            member_id=member.member_id,
            ecg_id=sealed.ecg_id,
            patient_id=sealed.patient_id,
            strat_fold=sealed.strat_fold,
            targets=sealed.targets,
            raw_logits=sealed.raw_logits,
        )
        clean_arrays = None
        clean_hash = None
    else:
        if clean is None:
            raise RobustnessCleanGateError("non-clean case has no verified clean artifact")
        transform = build_case_transform(case, base_seed=base_seed)
        inference = member.infer_logits(transform=cast(PhysicalBatchTransform, transform))
        clean_arrays = clean.artifact.arrays
        clean_hash = clean.artifact.artifact_sha256
    if case.corruption == "clean" and not np.array_equal(
        inference.raw_logits, member.sealed_prediction.raw_logits
    ):
        raise RobustnessCleanGateError("clean artifact logits differ from sealed logits")
    bootstrap_seed = _derived_bootstrap_seed(
        bootstrap_base_seed, member.member_id, case.case_id
    )
    arrays, analysis = compute_member_case_arrays(
        inference=inference,
        clean_logits=np.asarray(member.sealed_prediction.raw_logits, dtype=np.float64),
        temperature=member.decisions.temperature_scaling.temperature,
        thresholds=member.decisions.threshold_optimization.thresholds,
        gates=tuple(gate.to_dict() for gate in member.decisions.coverage_gates),
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_confidence=bootstrap_confidence,
        bootstrap_seed=bootstrap_seed,
        clean_case_arrays=clean_arrays,
    )
    metadata: dict[str, object] = {
        "post_evaluation_spec_sha256": spec.artifact_sha256,
        "member_id": member.member_id,
        "member_binding_sha256": member_hash,
        "case_protocol_sha256": case.protocol_sha256,
        "case": case.to_dict(),
        "label_order": list(LABEL_ORDER),
        "n_samples": int(inference.targets.shape[0]),
        "clean_gate": {
            "comparison": "np.array_equal_against_sealed_raw_logits",
            "exact": True,
            "mismatch_count": 0,
            "maximum_absolute_logit_error": 0.0,
            "clean_artifact_sha256": clean_hash,
        },
        "decision_policy": {
            "temperature": member.decisions.temperature_scaling.temperature,
            "thresholds": list(member.decisions.threshold_optimization.thresholds),
            "entropy_gates": [gate.to_dict() for gate in member.decisions.coverage_gates],
            "retuned": False,
        },
        **analysis,
    }
    save_audit_array_artifact(
        npz_path,
        artifact_type=ROBUSTNESS_MEMBER_CASE_TYPE,
        arrays=arrays,
        metadata=metadata,
    )
    artifact = _load_expected_artifact(
        npz_path,
        spec=spec,
        member_id=member.member_id,
        case=case,
        member_binding_sha256=member_hash,
        robustness_root=robustness_root,
    )
    _assert_artifact_alignment(artifact, member)
    return _artifact_record(member.member_id, case.case_id, artifact), True


def _case_summary(
    *,
    targets: NDArray[np.generic],
    logits: FloatArray,
    clean_logits: FloatArray,
    raw_probabilities: FloatArray,
    calibrated_probabilities: FloatArray,
    predictions: BoolArray,
    uncertainty: FloatArray,
    gate_selected: BoolArray,
    gates: Sequence[Mapping[str, object]],
    dense: Mapping[str, object],
) -> dict[str, object]:
    absolute_drift = np.abs(logits - clean_logits)
    frozen_gates: list[dict[str, object]] = []
    truth = np.asarray(targets, dtype=np.bool_)
    for index, gate in enumerate(gates):
        selected = gate_selected[:, index]
        count = int(selected.sum())
        errors = predictions[selected] != truth[selected]
        frozen_gates.append(
            {
                "target_coverage": _number(gate.get("target_coverage"), "target_coverage"),
                "maximum_entropy": _number(
                    gate.get("maximum_entropy"), "maximum_entropy"
                ),
                "selected_count": count,
                "coverage": float(selected.mean()),
                "hamming_risk": float(errors.mean()) if count else None,
                "exact_match_accuracy": (
                    float((~errors.any(axis=1)).mean()) if count else None
                ),
            }
        )
    dense_summary = {
        "ordering": "ascending_mean_normalized_binary_entropy_stable_index_tiebreak",
        "random_reference": "analytical_constant_full_coverage_risk",
        "oracle_reference": "ascending_per_record_loss_stable_index_tiebreak",
        "area_method": "arithmetic_mean_over_all_prefix_coverages",
        "aurc_hamming": _number(dense["aurc_hamming"], "aurc_hamming"),
        "aurc_log_loss": _number(dense["aurc_log_loss"], "aurc_log_loss"),
        "oracle_aurc_hamming": _number(
            dense.get("oracle_aurc_hamming", 0.0), "oracle_aurc_hamming"
        ),
        "oracle_aurc_log_loss": _number(
            dense.get("oracle_aurc_log_loss", 0.0), "oracle_aurc_log_loss"
        ),
        "random_aurc_hamming": _number(
            dense.get("random_aurc_hamming", 0.0), "random_aurc_hamming"
        ),
        "random_aurc_log_loss": _number(
            dense.get("random_aurc_log_loss", 0.0), "random_aurc_log_loss"
        ),
    }
    return {
        "raw": compute_multilabel_metrics(targets, raw_probabilities).to_dict(),
        "calibrated": compute_multilabel_metrics(
            targets, calibrated_probabilities
        ).to_dict(),
        "calibrated_log_loss": multilabel_log_loss(targets, calibrated_probabilities),
        "uncertainty": _distribution_summary(uncertainty),
        "raw_logit_drift": {
            **_distribution_summary(absolute_drift.reshape(-1)),
            "maximum": float(absolute_drift.max()),
            "root_mean_square": float(np.sqrt(np.square(logits - clean_logits).mean())),
        },
        "full_coverage": {
            "hamming_risk": float((predictions != truth).mean()),
            "exact_match_accuracy": float((predictions == truth).all(axis=1).mean()),
        },
        "frozen_gates": frozen_gates,
        "dense_risk_coverage": dense_summary,
    }


def _dense_curves(
    targets: NDArray[np.generic],
    probabilities: FloatArray,
    predictions: BoolArray,
    uncertainty: FloatArray,
) -> dict[str, object]:
    truth = np.asarray(targets, dtype=np.bool_)
    hamming = np.not_equal(predictions, truth).mean(axis=1).astype(np.float64)
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    target_float = truth.astype(np.float64)
    log_loss = -(
        target_float * np.log(clipped) + (1.0 - target_float) * np.log(1.0 - clipped)
    ).mean(axis=1)
    count = hamming.size
    coverage = np.arange(1, count + 1, dtype=np.float64) / count
    entropy_order = np.argsort(uncertainty, kind="stable").astype(np.int64)
    entropy_hamming = _cumulative_risk(hamming, entropy_order)
    entropy_log_loss = _cumulative_risk(log_loss, entropy_order)
    oracle_hamming = _cumulative_risk(hamming, np.argsort(hamming, kind="stable"))
    oracle_log_loss = _cumulative_risk(log_loss, np.argsort(log_loss, kind="stable"))
    random_hamming = np.full(count, float(hamming.mean()), dtype=np.float64)
    random_log_loss = np.full(count, float(log_loss.mean()), dtype=np.float64)
    return {
        "coverage": coverage,
        "entropy_order": entropy_order,
        "entropy_hamming": entropy_hamming,
        "entropy_log_loss": entropy_log_loss,
        "oracle_hamming": oracle_hamming,
        "oracle_log_loss": oracle_log_loss,
        "random_hamming": random_hamming,
        "random_log_loss": random_log_loss,
        "aurc_hamming": float(entropy_hamming.mean()),
        "aurc_log_loss": float(entropy_log_loss.mean()),
        "oracle_aurc_hamming": float(oracle_hamming.mean()),
        "oracle_aurc_log_loss": float(oracle_log_loss.mean()),
        "random_aurc_hamming": float(random_hamming.mean()),
        "random_aurc_log_loss": float(random_log_loss.mean()),
    }


def _cumulative_risk(values: FloatArray, order: NDArray[np.integer]) -> FloatArray:
    denominators = np.arange(1, values.size + 1, dtype=np.float64)
    return np.asarray(np.cumsum(values[order], dtype=np.float64) / denominators)


def _bootstrap_metric_names(gate_count: int) -> tuple[str, ...]:
    names = [
        "raw_macro_roc_auc",
        "raw_macro_average_precision",
        "raw_macro_brier_score",
        "raw_macro_ece",
        "calibrated_macro_roc_auc",
        "calibrated_macro_average_precision",
        "calibrated_macro_brier_score",
        "calibrated_macro_ece",
        "calibrated_log_loss",
        "uncertainty_mean",
        "full_hamming_risk",
        "full_exact_match_error",
        "aurc_hamming",
        "aurc_log_loss",
    ]
    for index in range(gate_count):
        names.extend(
            [
                f"gate_{index}_coverage",
                f"gate_{index}_hamming_risk",
                f"gate_{index}_exact_match_error",
            ]
        )
    return tuple(names)


def _bootstrap_vector(
    targets: NDArray[np.generic],
    raw_probabilities: FloatArray,
    calibrated_probabilities: FloatArray,
    predictions: BoolArray,
    uncertainty: FloatArray,
    gate_selected: BoolArray,
) -> tuple[FloatArray, BoolArray]:
    raw = compute_multilabel_metrics(targets, raw_probabilities)
    calibrated = compute_multilabel_metrics(targets, calibrated_probabilities)
    dense = _dense_curves(targets, calibrated_probabilities, predictions, uncertainty)
    truth = np.asarray(targets, dtype=np.bool_)
    errors = predictions != truth
    optional = [
        raw.macro.roc_auc,
        raw.macro.average_precision,
        raw.macro.brier_score,
        raw.macro.ece,
        calibrated.macro.roc_auc,
        calibrated.macro.average_precision,
        calibrated.macro.brier_score,
        calibrated.macro.ece,
        multilabel_log_loss(targets, calibrated_probabilities),
        float(uncertainty.mean()),
        float(errors.mean()),
        float(errors.any(axis=1).mean()),
        _number(dense["aurc_hamming"], "aurc_hamming"),
        _number(dense["aurc_log_loss"], "aurc_log_loss"),
    ]
    for index in range(gate_selected.shape[1]):
        selected = gate_selected[:, index]
        selected_count = int(selected.sum())
        selected_errors = errors[selected]
        optional.extend(
            [
                float(selected.mean()),
                float(selected_errors.mean()) if selected_count else None,
                float(selected_errors.any(axis=1).mean()) if selected_count else None,
            ]
        )
    valid_values: list[bool] = []
    stored_values: list[float] = []
    for value in optional:
        is_valid = value is not None and math.isfinite(value)
        valid_values.append(is_valid)
        stored_values.append(0.0 if value is None or not is_valid else value)
    valid: BoolArray = np.asarray(valid_values, dtype=np.bool_)
    values: FloatArray = np.asarray(stored_values, dtype=np.float64)
    return values, valid


def _paired_patient_bootstrap(
    *,
    targets: NDArray[np.generic],
    patient_id: NDArray[np.generic],
    raw_probabilities: FloatArray,
    calibrated_probabilities: FloatArray,
    predictions: BoolArray,
    uncertainty: FloatArray,
    gate_selected: BoolArray,
    clean_raw_probabilities: FloatArray,
    clean_calibrated_probabilities: FloatArray,
    clean_predictions: BoolArray,
    clean_uncertainty: FloatArray,
    clean_gate_selected: BoolArray,
    resamples: int,
    seed: int,
) -> tuple[FloatArray, BoolArray]:
    clusters: dict[object, list[int]] = {}
    for index, raw_identity in enumerate(patient_id.tolist()):
        identity = raw_identity.item() if isinstance(raw_identity, np.generic) else raw_identity
        clusters.setdefault(identity, []).append(index)
    cluster_rows = [np.asarray(rows, dtype=np.int64) for rows in clusters.values()]
    if not cluster_rows:
        raise RobustnessAuditError("patient bootstrap requires patient identities")
    rng = np.random.Generator(np.random.PCG64(seed))
    metric_count = len(_bootstrap_metric_names(gate_selected.shape[1]))
    deltas = np.zeros((resamples, metric_count), dtype=np.float64)
    valid = np.zeros((resamples, metric_count), dtype=np.bool_)
    for replicate in range(resamples):
        selected_clusters = rng.integers(0, len(cluster_rows), size=len(cluster_rows))
        rows = np.concatenate([cluster_rows[int(index)] for index in selected_clusters])
        observed, observed_valid = _bootstrap_vector(
            targets[rows],
            raw_probabilities[rows],
            calibrated_probabilities[rows],
            predictions[rows],
            uncertainty[rows],
            gate_selected[rows],
        )
        clean, clean_valid = _bootstrap_vector(
            targets[rows],
            clean_raw_probabilities[rows],
            clean_calibrated_probabilities[rows],
            clean_predictions[rows],
            clean_uncertainty[rows],
            clean_gate_selected[rows],
        )
        replicate_valid = observed_valid & clean_valid
        deltas[replicate, replicate_valid] = observed[replicate_valid] - clean[replicate_valid]
        valid[replicate] = replicate_valid
    return deltas, valid


def _bootstrap_summary(
    names: Sequence[str],
    values: FloatArray,
    valid: BoolArray,
    *,
    confidence: float,
) -> list[dict[str, object]]:
    alpha = (1.0 - confidence) / 2.0
    summaries: list[dict[str, object]] = []
    for index, name in enumerate(names):
        selected = values[valid[:, index], index]
        summaries.append(
            {
                "metric": name,
                "valid_resamples": int(selected.size),
                "invalid_resamples": int(values.shape[0] - selected.size),
                "mean_delta": float(selected.mean()) if selected.size else None,
                "confidence_interval": (
                    [
                        float(np.quantile(selected, alpha)),
                        float(np.quantile(selected, 1.0 - alpha)),
                    ]
                    if selected.size
                    else None
                ),
            }
        )
    return summaries


def _summary_delta(
    observed: Mapping[str, object], clean: Mapping[str, object]
) -> dict[str, object]:
    return {
        "direction": "corrupted_minus_clean",
        "raw": _metric_report_delta(observed["raw"], clean["raw"]),
        "calibrated": _metric_report_delta(observed["calibrated"], clean["calibrated"]),
        "calibrated_log_loss": _optional_difference(
            observed["calibrated_log_loss"], clean["calibrated_log_loss"]
        ),
        "uncertainty": _mapping_delta(observed["uncertainty"], clean["uncertainty"]),
        "dense_risk_coverage": _mapping_delta(
            observed["dense_risk_coverage"], clean["dense_risk_coverage"],
            keys=("aurc_hamming", "aurc_log_loss"),
        ),
        "frozen_gates": [
            {
                "target_coverage": observed_gate["target_coverage"],
                "coverage": _optional_difference(
                    observed_gate["coverage"], clean_gate["coverage"]
                ),
                "hamming_risk": _optional_difference(
                    observed_gate["hamming_risk"], clean_gate["hamming_risk"]
                ),
                "exact_match_accuracy": _optional_difference(
                    observed_gate["exact_match_accuracy"],
                    clean_gate["exact_match_accuracy"],
                ),
            }
            for observed_gate, clean_gate in zip(
                cast(Sequence[Mapping[str, object]], observed["frozen_gates"]),
                cast(Sequence[Mapping[str, object]], clean["frozen_gates"]),
                strict=True,
            )
        ],
    }


def _metric_report_delta(observed: object, clean: object) -> dict[str, object]:
    observed_map = _mapping(observed, "observed metrics")
    clean_map = _mapping(clean, "clean metrics")
    observed_macro = _mapping(observed_map["macro"], "observed macro")
    clean_macro = _mapping(clean_map["macro"], "clean macro")
    observed_labels = _sequence(observed_map["per_label"], "observed per-label")
    clean_labels = _sequence(clean_map["per_label"], "clean per-label")
    keys = ("roc_auc", "average_precision", "brier_score", "ece")
    return {
        "macro": {
            key: _optional_difference(observed_macro[key], clean_macro[key]) for key in keys
        },
        "per_label": [
            {
                "label": _mapping(left, "observed label")["label"],
                **{
                    key: _optional_difference(
                        _mapping(left, "observed label")[key],
                        _mapping(right, "clean label")[key],
                    )
                    for key in keys
                },
            }
            for left, right in zip(observed_labels, clean_labels, strict=True)
        ],
    }


def _mapping_delta(
    observed: object,
    clean: object,
    *,
    keys: Sequence[str] = ("mean", "median", "p95"),
) -> dict[str, object]:
    left = _mapping(observed, "observed summary")
    right = _mapping(clean, "clean summary")
    return {key: _optional_difference(left[key], right[key]) for key in keys}


def _optional_difference(left: object, right: object) -> float | None:
    if left is None or right is None:
        return None
    return float(cast(float, left)) - float(cast(float, right))


def _distribution_summary(values: FloatArray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _discover_all_records(
    *,
    spec: PostEvaluationSpec,
    runtime: CompletedAuditRuntime,
    cases: Sequence[RobustnessCase],
    robustness_root: Path,
) -> list[RobustnessArtifactRecord]:
    records: list[RobustnessArtifactRecord] = []
    for member_id in spec.member_ids:
        member = runtime.member(member_id)
        member_hash = _member_binding_sha256(spec, member_id)
        clean_hash: str | None = None
        for case in cases:
            path = _case_npz_path(robustness_root, member_id, case.case_id)
            sidecar = path.with_suffix(".json")
            if not path.exists() and not sidecar.exists():
                continue
            if not path.is_file() or not sidecar.is_file():
                raise RobustnessAuditIntegrityError("partial member/case artifact exists")
            artifact = _load_expected_artifact(
                path,
                spec=spec,
                member_id=member_id,
                case=case,
                member_binding_sha256=member_hash,
                robustness_root=robustness_root,
            )
            _assert_artifact_alignment(artifact, member)
            artifact_gate = _mapping(artifact.metadata["clean_gate"], "clean gate")
            if case.corruption == "clean":
                if (
                    not np.array_equal(
                        artifact.arrays["raw_logits"], member.sealed_prediction.raw_logits
                    )
                    or artifact_gate.get("clean_artifact_sha256") is not None
                ):
                    raise RobustnessCleanGateError("discovered clean artifact differs")
                clean_hash = artifact.artifact_sha256
            elif artifact_gate.get("clean_artifact_sha256") != clean_hash:
                raise RobustnessCleanGateError(
                    "discovered corruption binds a different clean artifact"
                )
            records.append(_artifact_record(member_id, case.case_id, artifact))
    return records


def _load_expected_artifact(
    path: Path,
    *,
    spec: PostEvaluationSpec,
    member_id: str,
    case: RobustnessCase,
    member_binding_sha256: str,
    robustness_root: Path,
) -> AuditArrayArtifact:
    expected = _case_npz_path(robustness_root, member_id, case.case_id)
    if path.resolve() != expected:
        raise RobustnessAuditIntegrityError("member/case artifact path differs")
    try:
        artifact = load_audit_array_artifact(
            path, expected_artifact_type=ROBUSTNESS_MEMBER_CASE_TYPE
        )
    except (AuditArtifactError, OSError) as error:
        raise RobustnessAuditIntegrityError(f"member/case artifact failed: {error}") from error
    if set(artifact.arrays) != _MEMBER_CASE_ARRAYS or set(artifact.metadata) != (
        _MEMBER_CASE_METADATA
    ):
        raise RobustnessAuditIntegrityError("member/case schema differs")
    metadata = artifact.metadata
    if (
        metadata["post_evaluation_spec_sha256"] != spec.artifact_sha256
        or metadata["member_id"] != member_id
        or metadata["member_binding_sha256"] != member_binding_sha256
        or metadata["case_protocol_sha256"] != case.protocol_sha256
        or metadata["case"] != case.to_dict()
        or metadata["label_order"] != list(LABEL_ORDER)
    ):
        raise RobustnessAuditIntegrityError("member/case source binding differs")
    n_samples = _integer(metadata["n_samples"], "n_samples")
    arrays = artifact.arrays
    if (
        arrays["targets"].shape != (n_samples, len(LABEL_ORDER))
        or arrays["raw_logits"].shape != (n_samples, len(LABEL_ORDER))
        or arrays["raw_probabilities"].shape != (n_samples, len(LABEL_ORDER))
        or arrays["calibrated_probabilities"].shape != (n_samples, len(LABEL_ORDER))
        or arrays["predictions"].shape != (n_samples, len(LABEL_ORDER))
        or arrays["uncertainty"].shape != (n_samples,)
        or arrays["ecg_id"].shape != (n_samples,)
        or arrays["patient_id"].shape != (n_samples,)
    ):
        raise RobustnessAuditIntegrityError("member/case core array shapes differ")
    expected_dtypes = {
        "targets": np.dtype(np.int8),
        "raw_logits": np.dtype(np.float64),
        "raw_probabilities": np.dtype(np.float64),
        "calibrated_probabilities": np.dtype(np.float64),
        "uncertainty": np.dtype(np.float64),
        "predictions": np.dtype(np.bool_),
        "gate_selected": np.dtype(np.bool_),
        "dense_coverage": np.dtype(np.float64),
        "dense_entropy_order": np.dtype(np.int64),
        "dense_entropy_hamming_risk": np.dtype(np.float64),
        "dense_entropy_log_loss_risk": np.dtype(np.float64),
        "dense_oracle_hamming_risk": np.dtype(np.float64),
        "dense_oracle_log_loss_risk": np.dtype(np.float64),
        "dense_random_hamming_risk": np.dtype(np.float64),
        "dense_random_log_loss_risk": np.dtype(np.float64),
        "bootstrap_delta": np.dtype(np.float64),
        "bootstrap_valid": np.dtype(np.bool_),
    }
    if any(arrays[name].dtype != dtype for name, dtype in expected_dtypes.items()):
        raise RobustnessAuditIntegrityError("member/case array dtypes differ")
    gate_count = len(
        _sequence(
            _mapping(metadata["decision_policy"], "decision policy")["entropy_gates"],
            "entropy gates",
        )
    )
    if arrays["gate_selected"].shape != (n_samples, gate_count):
        raise RobustnessAuditIntegrityError("gate selection shape differs")

    bootstrap = _mapping(metadata["bootstrap"], "bootstrap")
    if set(bootstrap) != _BOOTSTRAP_METADATA_KEYS:
        raise RobustnessAuditIntegrityError("bootstrap metadata keys differ")
    resamples = _integer(bootstrap["resamples"], "bootstrap resamples")
    confidence = _probability(bootstrap["confidence"], "bootstrap confidence")
    seed = _integer(bootstrap["seed"], "bootstrap seed")
    metric_names = _sequence(bootstrap["metric_names"], "bootstrap metric names")
    _, expected_resamples, expected_confidence, bootstrap_base_seed = (
        _robustness_settings(spec)
    )
    expected_metric_names = _bootstrap_metric_names(gate_count)
    if (
        any(
            bootstrap.get(key) != value
            for key, value in EXPECTED_BOOTSTRAP_POLICY.items()
        )
        or resamples != expected_resamples
        or confidence != expected_confidence
        or seed != _derived_bootstrap_seed(bootstrap_base_seed, member_id, case.case_id)
        or tuple(metric_names) != expected_metric_names
        or bootstrap.get("invalid_value_storage")
        != "zero_placeholder_with_parallel_bootstrap_valid_mask"
    ):
        raise RobustnessAuditIntegrityError("bootstrap metadata differs from protocol")
    if arrays["bootstrap_delta"].shape != (resamples, len(metric_names)) or arrays[
        "bootstrap_valid"
    ].shape != (resamples, len(metric_names)):
        raise RobustnessAuditIntegrityError("bootstrap array shapes differ")
    bootstrap_delta = cast(FloatArray, arrays["bootstrap_delta"])
    bootstrap_valid = cast(BoolArray, arrays["bootstrap_valid"])
    if np.any(bootstrap_delta[~bootstrap_valid] != 0.0):
        raise RobustnessAuditIntegrityError("invalid bootstrap values are not placeholders")
    if case.corruption == "clean" and np.any(bootstrap_delta != 0.0):
        raise RobustnessAuditIntegrityError("clean bootstrap deltas must be exactly zero")
    expected_statistics = _bootstrap_summary(
        expected_metric_names,
        bootstrap_delta,
        bootstrap_valid,
        confidence=confidence,
    )
    if bootstrap.get("statistics") != expected_statistics:
        raise RobustnessAuditIntegrityError("bootstrap statistics differ from stored arrays")
    if arrays["dense_entropy_order"].shape != (n_samples,):
        raise RobustnessAuditIntegrityError("dense entropy order shape differs")
    for name in _MEMBER_CASE_ARRAYS:
        if (
            name.startswith("dense_")
            and name != "dense_entropy_order"
            and arrays[name].shape != (n_samples,)
        ):
            raise RobustnessAuditIntegrityError("dense curve shape differs")
    clean_gate = _mapping(metadata["clean_gate"], "clean gate")
    if (
        clean_gate.get("exact") is not True
        or clean_gate.get("mismatch_count") != 0
        or clean_gate.get("maximum_absolute_logit_error") != 0.0
    ):
        raise RobustnessCleanGateError("member/case clean gate differs")
    return artifact


def _assert_artifact_alignment(
    artifact: AuditArrayArtifact, member: AuditMemberRuntime
) -> None:
    arrays = artifact.arrays
    sealed = member.sealed_prediction
    if (
        not np.array_equal(arrays["ecg_id"], sealed.ecg_id)
        or not np.array_equal(arrays["patient_id"], sealed.patient_id)
        or not np.array_equal(arrays["targets"], sealed.targets)
        or arrays["ecg_id"].dtype != sealed.ecg_id.dtype
        or arrays["patient_id"].dtype != sealed.patient_id.dtype
    ):
        raise RobustnessAuditIntegrityError("resumed artifact differs from sealed alignment")
    decision = _mapping(artifact.metadata["decision_policy"], "decision policy")
    if decision != {
        "temperature": member.decisions.temperature_scaling.temperature,
        "thresholds": list(member.decisions.threshold_optimization.thresholds),
        "entropy_gates": [gate.to_dict() for gate in member.decisions.coverage_gates],
        "retuned": False,
    }:
        raise RobustnessAuditIntegrityError("resumed artifact frozen decisions differ")
    raw_logits = np.asarray(arrays["raw_logits"], dtype=np.float64)
    expected_raw = stable_sigmoid(raw_logits)
    expected_calibrated = stable_sigmoid(
        raw_logits / member.decisions.temperature_scaling.temperature
    )
    expected_uncertainty = mean_normalized_binary_entropy(expected_calibrated)
    expected_predictions = expected_calibrated >= np.asarray(
        member.decisions.threshold_optimization.thresholds, dtype=np.float64
    )[None, :]
    expected_gate_selected = np.column_stack(
        [
            expected_uncertainty <= gate.maximum_entropy
            for gate in member.decisions.coverage_gates
        ]
    )
    dense = _dense_curves(
        sealed.targets, expected_calibrated, expected_predictions, expected_uncertainty
    )
    expected_dense = {
        "dense_coverage": dense["coverage"],
        "dense_entropy_order": dense["entropy_order"],
        "dense_entropy_hamming_risk": dense["entropy_hamming"],
        "dense_entropy_log_loss_risk": dense["entropy_log_loss"],
        "dense_oracle_hamming_risk": dense["oracle_hamming"],
        "dense_oracle_log_loss_risk": dense["oracle_log_loss"],
        "dense_random_hamming_risk": dense["random_hamming"],
        "dense_random_log_loss_risk": dense["random_log_loss"],
    }
    if (
        not np.array_equal(arrays["raw_probabilities"], expected_raw)
        or not np.array_equal(arrays["calibrated_probabilities"], expected_calibrated)
        or not np.array_equal(arrays["uncertainty"], expected_uncertainty)
        or not np.array_equal(arrays["predictions"], expected_predictions)
        or not np.array_equal(arrays["gate_selected"], expected_gate_selected)
        or any(
            not np.array_equal(arrays[name], cast(NDArray[np.generic], value))
            for name, value in expected_dense.items()
        )
    ):
        raise RobustnessAuditIntegrityError("resumed artifact derived arrays differ")

    targets = np.asarray(sealed.targets, dtype=np.int8)
    clean_logits = np.asarray(sealed.raw_logits, dtype=np.float64)
    gates = tuple(gate.to_dict() for gate in member.decisions.coverage_gates)
    expected_summary = _case_summary(
        targets=targets,
        logits=raw_logits,
        clean_logits=clean_logits,
        raw_probabilities=expected_raw,
        calibrated_probabilities=expected_calibrated,
        predictions=expected_predictions,
        uncertainty=expected_uncertainty,
        gate_selected=expected_gate_selected,
        gates=gates,
        dense=dense,
    )
    clean_raw = stable_sigmoid(clean_logits)
    clean_calibrated = stable_sigmoid(
        clean_logits / member.decisions.temperature_scaling.temperature
    )
    clean_uncertainty = mean_normalized_binary_entropy(clean_calibrated)
    clean_predictions = clean_calibrated >= np.asarray(
        member.decisions.threshold_optimization.thresholds, dtype=np.float64
    )[None, :]
    clean_gate_selected = np.column_stack(
        [
            clean_uncertainty <= gate.maximum_entropy
            for gate in member.decisions.coverage_gates
        ]
    )
    clean_dense = _dense_curves(
        targets, clean_calibrated, clean_predictions, clean_uncertainty
    )
    clean_summary = _case_summary(
        targets=targets,
        logits=clean_logits,
        clean_logits=clean_logits,
        raw_probabilities=clean_raw,
        calibrated_probabilities=clean_calibrated,
        predictions=clean_predictions,
        uncertainty=clean_uncertainty,
        gate_selected=clean_gate_selected,
        gates=gates,
        dense=clean_dense,
    )
    expected_delta = _summary_delta(expected_summary, clean_summary)
    if artifact.metadata["metric_summary"] != expected_summary:
        raise RobustnessAuditIntegrityError("resumed artifact metric summary differs")
    if artifact.metadata["delta_summary"] != expected_delta:
        raise RobustnessAuditIntegrityError("resumed artifact delta summary differs")


def _artifact_record(
    member_id: str, case_id: str, artifact: AuditArrayArtifact
) -> RobustnessArtifactRecord:
    return RobustnessArtifactRecord(
        member_id=member_id,
        case_id=case_id,
        artifact=artifact,
        npz_file_sha256=_file_sha256(artifact.npz_path),
        sidecar_file_sha256=_file_sha256(artifact.json_path),
    )


def _audit_paths(spec: PostEvaluationSpec) -> tuple[Path, Path]:
    output = _mapping(spec.payload["output_contract"], "output_contract")
    artifacts = _mapping(output["artifacts"], "output artifacts")
    manifest = Path(_string(artifacts["robustness_manifest"], "robustness manifest")).resolve()
    output_root = Path(_string(output["root"], "output root")).resolve()
    if manifest.parent.parent != output_root or manifest.name != "manifest.json":
        raise RobustnessAuditIntegrityError("robustness output path differs from contract")
    return manifest, manifest.parent


def _case_path(root: Path, member_id: str, case_id: str) -> Path:
    member = _safe_component(member_id, "member_id")
    case = _safe_component(case_id, "case_id")
    result = (root / "member_cases" / member / case).resolve()
    if root.resolve() not in result.parents:
        raise RobustnessAuditIntegrityError("member/case path escapes robustness root")
    return result


def _case_npz_path(root: Path, member_id: str, case_id: str) -> Path:
    """Append ``.npz`` without interpreting decimal case IDs as suffixes."""

    base = _case_path(root, member_id, case_id)
    return base.parent / f"{base.name}.npz"


def _member_binding_sha256(spec: PostEvaluationSpec, member_id: str) -> str:
    members = _sequence(spec.payload["members"], "spec members")
    matches = [
        dict(_mapping(item, "spec member"))
        for item in members
        if _mapping(item, "spec member").get("member_id") == member_id
    ]
    if len(matches) != 1:
        raise RobustnessAuditIntegrityError("member binding is unavailable")
    return canonical_sha256(matches[0])


def _robustness_settings(spec: PostEvaluationSpec) -> tuple[int, int, float, int]:
    protocols = _mapping(spec.payload["audit_protocols"], "audit protocols")
    robustness = _mapping(protocols["robustness"], "robustness protocol")
    _assert_robustness_protocol_semantics(robustness)
    bootstrap = _mapping(robustness["patient_resampling"], "patient_resampling")
    return (
        _integer(robustness["random_seed"], "random seed"),
        _integer(bootstrap["resamples"], "bootstrap resamples"),
        _probability(bootstrap["confidence"], "bootstrap confidence"),
        _integer(bootstrap["base_seed"], "bootstrap base seed"),
    )


def _assert_robustness_protocol_semantics(
    robustness: Mapping[str, object],
) -> None:
    """Validate every execution semantic consumed by this runner."""

    if robustness.get("sampling_frequency_hz") != EXPECTED_SAMPLING_FREQUENCY_HZ:
        raise RobustnessAuditIntegrityError("robustness sampling frequency differs")
    if robustness.get("random_seed") != EXPECTED_RANDOM_SEED:
        raise RobustnessAuditIntegrityError("robustness random seed differs")
    if robustness.get("corruption_domain") != EXPECTED_CORRUPTION_DOMAIN:
        raise RobustnessAuditIntegrityError("robustness corruption domain differs")
    if dict(_mapping(robustness.get("execution"), "robustness execution")) != (
        EXPECTED_EXECUTION
    ):
        raise RobustnessAuditIntegrityError("robustness execution semantics differ")
    if dict(
        _mapping(robustness.get("transform_definitions"), "transform definitions")
    ) != EXPECTED_TRANSFORM_DEFINITIONS:
        raise RobustnessAuditIntegrityError("robustness transform semantics differ")
    if dict(
        _mapping(robustness.get("dense_risk_coverage"), "dense risk/coverage")
    ) != EXPECTED_DENSE_RISK_COVERAGE:
        raise RobustnessAuditIntegrityError("dense risk/coverage semantics differ")

    bootstrap = _mapping(robustness.get("patient_resampling"), "patient_resampling")
    expected_keys = set(EXPECTED_BOOTSTRAP_POLICY) | {
        "resamples",
        "confidence",
        "base_seed",
    }
    observed_policy = {key: bootstrap.get(key) for key in EXPECTED_BOOTSTRAP_POLICY}
    if set(bootstrap) != expected_keys or observed_policy != EXPECTED_BOOTSTRAP_POLICY:
        raise RobustnessAuditIntegrityError("bootstrap policy differs")
    if (
        bootstrap.get("resamples") != EXPECTED_BOOTSTRAP_RESAMPLES
        or bootstrap.get("confidence") != EXPECTED_BOOTSTRAP_CONFIDENCE
        or bootstrap.get("base_seed") != EXPECTED_BOOTSTRAP_BASE_SEED
    ):
        raise RobustnessAuditIntegrityError("bootstrap settings differ")


def _derived_bootstrap_seed(base_seed: int, member_id: str, case_id: str) -> int:
    material = f"ecg_trust:robustness:bootstrap:{base_seed}:{member_id}:{case_id}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _validate_case_parameters(corruption: str, parameters: Mapping[str, object]) -> None:
    expected: dict[str, set[str]] = {
        "clean": set(),
        "baseline_wander": {"amplitude_fraction", "frequency_hz", "phase_radians"},
        "powerline": {"amplitude_fraction", "frequency_hz", "phase_radians"},
        "gaussian_noise": {"snr_db", "seed_strategy"},
        "amplitude_scale": {"factor"},
        "dc_offset": {"offset_fraction"},
        "time_shift": {"samples", "padding"},
        "contiguous_mask": {
            "width_samples",
            "lead_indices",
            "start_strategy",
            "seed",
        },
        "lead_dropout": {"lead_indices", "lead_names"},
        "lead_permutation": {"permutation"},
    }
    if corruption not in expected or set(parameters) != expected[corruption]:
        raise RobustnessAuditIntegrityError(
            f"parameters for corruption {corruption!r} are not canonical"
        )
    # Parse all numeric/index fields now so malformed frozen values fail early.
    numeric_keys = (
        "amplitude_fraction",
        "frequency_hz",
        "phase_radians",
        "snr_db",
        "factor",
        "offset_fraction",
    )
    for key in numeric_keys:
        if key in parameters:
            _number(parameters[key], key)
    for key in ("samples", "width_samples", "seed"):
        if key in parameters:
            _integer(parameters[key], key, signed=key == "samples")
    for key in ("lead_indices", "permutation"):
        if key in parameters:
            _integer_sequence(parameters[key], key)


def _selection(
    requested: Sequence[str] | None, available: Sequence[str], context: str
) -> tuple[str, ...]:
    if requested is None:
        return tuple(available)
    values = tuple(requested)
    if not values or len(set(values)) != len(values):
        raise RobustnessAuditError(f"{context} selection must be non-empty and unique")
    unknown = sorted(set(values).difference(available))
    if unknown:
        raise RobustnessAuditError(f"unknown {context} selection: {unknown}")
    requested_set = set(values)
    return tuple(value for value in available if value in requested_set)


def _validate_file_binding(value: object, context: str) -> Mapping[str, object]:
    binding = _mapping(value, context)
    if set(binding) != {"path", "file_sha256"}:
        raise RobustnessAuditIntegrityError(f"{context} keys differ")
    _string(binding["path"], f"{context}.path")
    _hash(binding["file_sha256"], f"{context}.file_sha256")
    return binding


def _verify_nested_file_bindings(value: object) -> None:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        if set(mapping) == {"path", "file_sha256"}:
            binding = _validate_file_binding(mapping, "nested file binding")
            path = Path(_string(binding["path"], "nested path")).resolve()
            if _file_sha256(path) != binding["file_sha256"]:
                raise RobustnessAuditIntegrityError(f"nested file hash mismatch: {path}")
            return
        for nested in mapping.values():
            _verify_nested_file_bindings(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            _verify_nested_file_bindings(nested)


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable manifest already exists: {path}")
    serialized = json.dumps(
        payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True
    ) + "\n"
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"immutable manifest already exists: {path}") from error
    finally:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)


def _read_json(path: Path, context: str) -> Mapping[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RobustnessAuditIntegrityError(f"could not decode {context}: {error}") from error
    return _mapping(value, context)


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise RobustnessAuditIntegrityError(f"bound file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
    except (TypeError, ValueError) as error:
        raise RobustnessAuditIntegrityError("value is not finite canonical JSON") from error


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise RobustnessAuditIntegrityError(f"{context} must be a string-keyed object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise RobustnessAuditIntegrityError(f"{context} must be a sequence")
    return cast(Sequence[object], value)


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RobustnessAuditIntegrityError(f"{context} must be a non-empty string")
    return value


def _safe_component(value: object, context: str) -> str:
    text = _string(value, context)
    if _SAFE_COMPONENT.fullmatch(text) is None or text in {".", ".."}:
        raise RobustnessAuditIntegrityError(f"{context} is not a safe path component")
    return text


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RobustnessAuditIntegrityError(f"{context} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise RobustnessAuditIntegrityError(f"{context} must be finite")
    return parsed


def _probability(value: object, context: str) -> float:
    parsed = _number(value, context)
    if not 0.0 < parsed < 1.0:
        raise RobustnessAuditIntegrityError(f"{context} must lie strictly inside (0, 1)")
    return parsed


def _integer(
    value: object, context: str, *, signed: bool = False
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or (not signed and value < 1):
        qualifier = "an integer" if signed else "a positive integer"
        raise RobustnessAuditIntegrityError(f"{context} must be {qualifier}")
    return value


def _integer_sequence(value: object, context: str) -> tuple[int, ...]:
    sequence = _sequence(value, context)
    result = tuple(_integer(item, context, signed=True) for item in sequence)
    if not result:
        raise RobustnessAuditIntegrityError(f"{context} must not be empty")
    return result


def _hash(value: object, context: str) -> str:
    text = _string(value, context)
    digest = text.removeprefix("sha256:")
    if (
        not text.startswith("sha256:")
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RobustnessAuditIntegrityError(f"{context} must be a prefixed SHA-256")
    return text


def _normalized_hash(value: object, context: str) -> str:
    text = _string(value, context)
    digest = text.removeprefix("sha256:")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise RobustnessAuditIntegrityError(f"{context} must contain a SHA-256")
    return "sha256:" + digest


def _assert_path_hash_artifact_binding(
    binding: Mapping[str, object],
    *,
    path: Path,
    artifact_sha256: str,
    context: str,
) -> None:
    if (
        Path(_string(binding.get("path"), f"{context} path")).resolve()
        != path.resolve()
        or _normalized_hash(binding.get("file_sha256"), f"{context} file hash")
        != _file_sha256(path)
        or binding.get("artifact_sha256") != artifact_sha256
    ):
        raise RobustnessAuditIntegrityError(
            f"runtime {context} differs from the frozen spec"
        )


def _assert_bundle_identity(
    binding: Mapping[str, object],
    *,
    artifact_sha256: str | None,
    protocol_hash: str,
    manifest_sha256: str,
    normalization_sha256: str,
    member_count: int,
    context: str,
) -> None:
    if (
        artifact_sha256 is None
        or binding.get("artifact_sha256") != artifact_sha256
        or binding.get("protocol_hash") != protocol_hash
        or binding.get("manifest_sha256") != manifest_sha256
        or binding.get("normalization_sha256") != normalization_sha256
        or binding.get("member_count") != member_count
    ):
        raise RobustnessAuditIntegrityError(
            f"runtime {context} differs from the frozen spec"
        )


__all__ = [
    "EXPECTED_CASE_COUNT",
    "EXPECTED_MEMBER_CASE_COUNT",
    "ROBUSTNESS_MANIFEST_TYPE",
    "ROBUSTNESS_MEMBER_CASE_TYPE",
    "RobustnessArtifactRecord",
    "RobustnessAuditError",
    "RobustnessAuditIntegrityError",
    "RobustnessAuditManifest",
    "RobustnessAuditProgress",
    "RobustnessCase",
    "RobustnessCleanGateError",
    "build_case_transform",
    "assert_runtime_matches_post_evaluation_spec",
    "compute_member_case_arrays",
    "expand_robustness_cases",
    "load_robustness_manifest",
    "run_robustness_audit",
    "save_robustness_manifest",
    "stateless_seed",
]
