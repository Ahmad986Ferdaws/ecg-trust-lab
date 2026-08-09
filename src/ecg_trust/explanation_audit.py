"""Deterministic, integrity-bound explanation audits for the sealed exact-six release.

Attributions are model-behavior diagnostics.  They are not physiological ground
truth, causal explanations, or clinical localization evidence.  This runner
therefore pairs each map with repeatability, perturbation, random-ranking,
parameter-randomization, and cross-method controls and preserves the arrays
needed to reproduce every reported summary.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from ecg_trust.audit_artifacts import (
    AuditArrayArtifact,
    AuditArrayFiles,
    load_audit_array_artifact,
    save_audit_array_artifact,
)
from ecg_trust.audit_runtime import AuditMemberRuntime, CompletedAuditRuntime
from ecg_trust.explain import (
    attribution_stability_similarity,
    cross_method_temporal_similarity,
    grad_cam_1d,
    integrated_gradients_with_delta,
    lead_ablation_faithfulness_curve,
    parameter_randomization_comparison,
    randomized_model_copy,
    temporal_faithfulness_curve,
    temporal_occlusion,
)
from ecg_trust.models import ResNet1D
from ecg_trust.post_evaluation import (
    PostEvaluationSpec,
    _explanation_settings,
    canonical_sha256,
)
from ecg_trust.protocol import LABEL_ORDER
from ecg_trust.release_gates import sha256_file

EXPLANATION_MANIFEST_SCHEMA_VERSION = 1
EXPLANATION_MANIFEST_TYPE = "ecg_trust.explanation_audit_manifest"
EXPLANATION_ARRAY_TYPE = "ecg_trust.explanation_method_audit_arrays"
EXPLANATION_CROSS_METHOD_TYPE = "ecg_trust.explanation_cross_method_arrays"
_METHODS_BY_ARCHITECTURE: Mapping[str, tuple[str, ...]] = {
    "resnet1d": ("grad_cam_1d", "integrated_gradients", "temporal_occlusion"),
    "ecg_transformer": ("integrated_gradients", "temporal_occlusion"),
}


class ExplanationAuditError(RuntimeError):
    """Raised when an explanation audit cannot follow its frozen contract."""


class ExplanationAuditIntegrityError(ExplanationAuditError):
    """Raised when an existing explanation artifact fails verification."""


@dataclass(frozen=True, slots=True)
class ExplanationManifest:
    """Verified self-hashed top-level explanation manifest."""

    path: Path
    artifact_sha256: str
    payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _Cohort:
    ecg_id: NDArray[np.int64]
    patient_id: NDArray[np.int64]
    target_index: NDArray[np.int64]
    target_value: NDArray[np.int8]
    target_bits: NDArray[np.int8]
    selection_sha256: tuple[str, ...]
    payload_sha256: str

    @property
    def size(self) -> int:
        return int(self.ecg_id.shape[0])


@dataclass(frozen=True, slots=True)
class _Settings:
    ig_steps: int
    ig_internal_batch_size: int
    occlusion_window: int
    occlusion_stride: int
    occlusion_perturbations: int
    fractions: tuple[float, ...]
    random_ranking_replicates: int
    random_ranking_seed: int
    randomization_seeds: tuple[int, ...]
    stability_snr_db: float
    stability_replicates: int
    outer_batch_size: int


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ExplanationAuditIntegrityError(f"{context} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _sequence(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ExplanationAuditIntegrityError(f"{context} must be a sequence")
    return cast(Sequence[object], value)


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExplanationAuditIntegrityError(f"{context} must be a non-empty string")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ExplanationAuditIntegrityError(f"{context} must be an integer >= {minimum}")
    return value


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExplanationAuditIntegrityError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ExplanationAuditIntegrityError(f"{context} must be finite")
    return result


def _hash(value: object, context: str) -> str:
    text = _string(value, context)
    digest = text.removeprefix("sha256:")
    if (
        not text.startswith("sha256:")
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ExplanationAuditIntegrityError(f"{context} must be a prefixed SHA-256")
    return text


def _finite_json(value: Mapping[str, object], context: str) -> dict[str, object]:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        decoded: object = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ExplanationAuditError(f"{context} must contain finite JSON") from error
    return dict(_mapping(decoded, context))


def _cohort_from_spec(spec: PostEvaluationSpec) -> _Cohort:
    protocols = _mapping(spec.payload["audit_protocols"], "audit_protocols")
    explanations = _mapping(protocols["explanations"], "explanations")
    payload = _mapping(explanations["cohort"], "explanation cohort")
    records = _sequence(payload["records"], "explanation cohort records")
    ecg_ids: list[int] = []
    patient_ids: list[int] = []
    target_indices: list[int] = []
    target_values: list[int] = []
    target_bits: list[list[int]] = []
    selections: list[str] = []
    for expected_rank, raw_record in enumerate(records):
        record = _mapping(raw_record, "explanation cohort record")
        if record.get("rank") != expected_rank:
            raise ExplanationAuditIntegrityError("cohort ranks are not canonical")
        index = _integer(record["target_index"], "target_index")
        value = _integer(record["target_value"], "target_value")
        bits = [_integer(bit, "target bit") for bit in _sequence(record["target_bits"], "bits")]
        if (
            index >= len(LABEL_ORDER)
            or value not in {0, 1}
            or len(bits) != len(LABEL_ORDER)
            or any(bit not in {0, 1} for bit in bits)
            or bits[index] != value
            or record["target_label"] != LABEL_ORDER[index]
        ):
            raise ExplanationAuditIntegrityError("cohort target assignment is invalid")
        ecg_ids.append(_integer(record["ecg_id"], "ecg_id", minimum=1))
        patient_ids.append(_integer(record["patient_id"], "patient_id", minimum=1))
        target_indices.append(index)
        target_values.append(value)
        target_bits.append(bits)
        selections.append(_hash(record["selection_sha256"], "selection_sha256"))
    if len(ecg_ids) != 60 or len(set(ecg_ids)) != 60 or len(set(patient_ids)) != 60:
        raise ExplanationAuditIntegrityError("explanation cohort must contain 60 unique patients")
    return _Cohort(
        ecg_id=np.asarray(ecg_ids, dtype=np.int64),
        patient_id=np.asarray(patient_ids, dtype=np.int64),
        target_index=np.asarray(target_indices, dtype=np.int64),
        target_value=np.asarray(target_values, dtype=np.int8),
        target_bits=np.asarray(target_bits, dtype=np.int8),
        selection_sha256=tuple(selections),
        payload_sha256=canonical_sha256(dict(payload)),
    )


def _settings_from_spec(spec: PostEvaluationSpec, *, outer_batch_size: int) -> _Settings:
    if isinstance(outer_batch_size, bool) or outer_batch_size < 1:
        raise ExplanationAuditError("outer_batch_size must be positive")
    protocols = _mapping(spec.payload["audit_protocols"], "audit_protocols")
    explanations = _mapping(protocols["explanations"], "explanations")
    settings = _mapping(explanations["settings"], "explanation settings")
    if dict(settings) != _explanation_settings():
        raise ExplanationAuditIntegrityError(
            "explanation settings differ from the frozen canonical plan"
        )
    target_score = _mapping(settings["target_score"], "target score")
    grad_cam = _mapping(settings["grad_cam_1d"], "Grad-CAM settings")
    execution = _mapping(settings["execution"], "explanation execution")
    frozen_outer_batch_size = _integer(
        execution["outer_attribution_batch_size"],
        "outer attribution batch size",
        minimum=1,
    )
    if (
        execution.get("numeric_precision") != "float32"
        or execution.get("sealed_clean_equivalence_precision") != "bf16_as_frozen_in_final_batch"
        or execution.get("torch_deterministic_algorithms") is not True
        or execution.get("identical_rerun_requirement") != "torch_equal"
        or execution.get("tf32_allowed") is not False
        or execution.get("fp32_vs_sealed_cohort_logit_drift_required") is not True
        or execution.get("faithfulness_scoring_batch_size") != 60
        or outer_batch_size != frozen_outer_batch_size
    ):
        raise ExplanationAuditIntegrityError(
            "explanation execution settings differ from the frozen plan"
        )
    ig = _mapping(settings["integrated_gradients"], "integrated_gradients")
    occlusion = _mapping(settings["temporal_occlusion"], "temporal_occlusion")
    faithfulness = _mapping(settings["faithfulness"], "faithfulness")
    fractions = tuple(
        _number(value, "deletion fraction")
        for value in _sequence(faithfulness["temporal_deletion_fractions"], "deletion fractions")
    )
    result = _Settings(
        ig_steps=_integer(ig["n_steps"], "IG steps", minimum=2),
        ig_internal_batch_size=_integer(
            ig["internal_batch_size"], "IG internal batch size", minimum=1
        ),
        occlusion_window=_integer(occlusion["window_samples"], "occlusion window", minimum=1),
        occlusion_stride=_integer(occlusion["stride_samples"], "occlusion stride", minimum=1),
        occlusion_perturbations=_integer(
            occlusion["perturbations_per_eval"],
            "occlusion perturbations",
            minimum=1,
        ),
        fractions=fractions,
        random_ranking_replicates=_integer(
            faithfulness["random_ranking_replicates"],
            "random ranking replicates",
            minimum=1,
        ),
        random_ranking_seed=_integer(faithfulness["random_ranking_seed"], "random ranking seed"),
        randomization_seeds=tuple(
            _integer(value, "randomization seed")
            for value in _sequence(
                faithfulness["parameter_randomization_seeds"],
                "parameter randomization seeds",
            )
        ),
        stability_snr_db=_number(faithfulness["stability_noise_snr_db"], "stability SNR"),
        stability_replicates=_integer(
            faithfulness["stability_replicates"],
            "stability replicates",
            minimum=1,
        ),
        outer_batch_size=frozen_outer_batch_size,
    )
    lead_ablation = _mapping(faithfulness["lead_ablation"], "lead ablation")
    cross_method = _mapping(faithfulness["cross_method_agreement"], "cross-method agreement")
    operation_rankings = _mapping(faithfulness["operation_rankings"], "operation rankings")
    stability_noise = _mapping(faithfulness["stability_noise"], "stability noise")
    if (
        target_score.get("positive_cell") != "+1_times_target_label_logit"
        or target_score.get("negative_cell") != "-1_times_target_label_logit"
        or target_score.get("probability")
        != "sigmoid(signed_correct_status_logit_over_frozen_temperature)"
        or target_score.get("attribution_orientation") != "multiply_target_label_map_by_cell_sign"
        or grad_cam.get("feature_map") != "resnet_final_temporal_feature_map_before_global_pool"
        or grad_cam.get("channel_weights") != "mean_target_gradient_over_time"
        or grad_cam.get("signed") is not True
        or grad_cam.get("relu_applied") is not False
        or grad_cam.get("upsampling") != "linear_to_1000_samples_align_corners_false"
        or grad_cam.get("normalization") != "per_ecg_unit_maximum_absolute_value"
        or occlusion.get("mask_unit") != "one_temporal_window_across_all_12_leads"
        or occlusion.get("baseline") != "zero_normalized_input"
        or occlusion.get("returned_lead_axis") != "duplicated_not_lead_resolved"
        or ig.get("multiply_by_inputs") is not True
        or ig.get("integration_method") != "gausslegendre"
        or result.ig_steps != 32
        or result.ig_internal_batch_size != 8
        or ig.get("normalize") is not True
        or ig.get("completeness_delta_basis") != "pre_normalization_signed_raw_ig"
        or result.occlusion_window != 50
        or result.occlusion_stride != 25
        or result.occlusion_perturbations != 16
        or occlusion.get("normalize") is not True
        or result.fractions != (0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
        or list(_sequence(operation_rankings.get("deletion"), "deletion rankings"))
        != ["most_important", "least_important", "random"]
        or list(_sequence(operation_rankings.get("insertion"), "insertion rankings"))
        != ["most_important"]
        or faithfulness.get("temporal_importance")
        != "mean_absolute_attribution_across_attribution_channels_per_sample"
        or faithfulness.get("temporal_perturbation_unit")
        != "one_ranked_sample_index_across_all_12_input_leads"
        or faithfulness.get("area_under_curve")
        != ("trapezoidal_integral_of_calibrated_correct_status_probability_over_fraction")
        or faithfulness.get("guided_vs_random_advantage") != "mean_random_auc_minus_guided_auc"
        or result.random_ranking_replicates != 20
        or result.random_ranking_seed != 20_261_008
        or result.randomization_seeds != (2_026_801, 2_026_802, 2_026_803)
        or result.stability_snr_db != 40.0
        or result.stability_replicates != 3
        or faithfulness.get("parameter_randomization_scope") != "full_model_reset_parameters_copy"
        or faithfulness.get("parameter_randomization_similarity")
        != "signed_cosine_of_flattened_attribution_tensors"
        or faithfulness.get("stability_seed_strategy") != "sha256_of_replicate_and_ecg_id"
        or faithfulness.get("stability_similarity")
        != "signed_cosine_of_flattened_attribution_tensors"
        or stability_noise.get("domain") != "after_frozen_normalization"
        or stability_noise.get("distribution") != "iid_zero_mean_gaussian"
        or stability_noise.get("scale") != "whole_record_all_lead_rms_per_ecg"
        or stability_noise.get("snr_definition")
        != "nominal_expected_power_snr_from_gaussian_sigma;realized_draw_not_renormalized"
        or faithfulness.get("integrated_gradients_completeness_delta_required") is not True
        or lead_ablation.get("applicable_to_lead_specific_maps_only") is not True
        or list(_sequence(lead_ablation.get("applicable_methods"), "lead methods"))
        != ["integrated_gradients"]
        or list(_sequence(lead_ablation.get("rankings"), "lead rankings"))
        != ["most_important", "least_important", "random"]
        or lead_ablation.get("prefixes") != "zero_through_all_12_leads"
        or lead_ablation.get("lead_importance") != "mean_absolute_attribution_over_time_per_lead"
        or cross_method.get("aggregation") != "absolute_attribution_mean_across_leads_to_time"
        or list(_sequence(cross_method.get("metrics"), "cross-method metrics"))
        != ["cosine", "spearman"]
    ):
        raise ExplanationAuditIntegrityError("explanation settings differ from the frozen plan")
    return result


def _method_names(architecture: str) -> tuple[str, ...]:
    try:
        return _METHODS_BY_ARCHITECTURE[architecture]
    except KeyError as error:
        raise ExplanationAuditIntegrityError(
            f"unsupported audit architecture {architecture!r}"
        ) from error


def _assert_runtime_bound_to_spec(spec: PostEvaluationSpec, runtime: CompletedAuditRuntime) -> None:
    payload = spec.payload
    sealed = _mapping(payload["sealed_evaluation"], "sealed_evaluation")
    final_spec = _mapping(sealed["final_evaluation_spec"], "final_evaluation_spec")
    refit = _mapping(sealed["refit_bundle"], "refit_bundle")
    calibration = _mapping(sealed["calibration_bundle"], "calibration_bundle")
    ledger = _mapping(sealed["opening_ledger"], "opening_ledger")
    if (
        runtime.final_evaluation_spec.artifact_sha256 != final_spec["artifact_sha256"]
        or runtime.refit_bundle.artifact_sha256 != refit["artifact_sha256"]
        or runtime.calibration_bundle.artifact_sha256 != calibration["artifact_sha256"]
        or runtime.ledger.ledger_sha256 != ledger["ledger_sha256"]
        or runtime.ledger.batch_sha256 != ledger["batch_sha256"]
    ):
        raise ExplanationAuditIntegrityError(
            "audit runtime release anchors differ from the post-evaluation spec"
        )
    spec_members = {
        _string(member["member_id"], "spec member_id"): member
        for member in (
            _mapping(raw_member, "spec member")
            for raw_member in _sequence(payload["members"], "spec members")
        )
    }
    if tuple(spec_members) != spec.member_ids:
        raise ExplanationAuditIntegrityError("spec member order differs")
    for member in runtime.members:
        bound = spec_members.get(member.member_id)
        if bound is None:
            raise ExplanationAuditIntegrityError("runtime contains an unbound member")
        prediction = _mapping(bound["prediction"], "spec prediction")
        checkpoint = _mapping(bound["checkpoint"], "spec checkpoint")
        resolved = _mapping(bound["resolved_config"], "spec resolved config")
        decision = _mapping(bound["calibration_decision"], "spec calibration decision")
        if (
            bound["architecture"] != member.architecture
            or bound["seed"] != member.seed
            or bound["model_name"] != member.refit.run_name
            or bound["refit_lineage_sha256"] != member.refit.lineage_sha256
            or checkpoint["file_sha256"] != member.checkpoint_sha256
            or resolved["config_hash"] != member.refit.resolved_config_hash
            or decision["artifact_sha256"] != member.decisions.integrity_sha256
            or prediction["artifact_sha256"] != member.sealed_prediction.integrity_sha256
            or prediction["alignment_sha256"] != member.sealed_prediction.alignment_sha256
        ):
            raise ExplanationAuditIntegrityError(
                f"audit runtime member {member.member_id} differs from its spec binding"
            )


def _stable_uint32(*parts: object) -> int:
    encoded = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big")


def _artifact_binding(files: AuditArrayFiles) -> dict[str, object]:
    return {
        "artifact_sha256": files.artifact_sha256,
        "npz": {"path": str(files.npz_path), "file_sha256": files.npz_sha256},
        "sidecar": {
            "path": str(files.json_path),
            "file_sha256": files.json_file_sha256,
        },
    }


def _runtime_block(member: AuditMemberRuntime) -> dict[str, object]:
    device = member.runtime.device
    device_name: str | None = None
    compute_capability: list[int] | None = None
    if device.type == "cuda":
        index = 0 if device.index is None else device.index
        device_name = torch.cuda.get_device_name(index)
        major, minor = torch.cuda.get_device_capability(index)
        compute_capability = [major, minor]
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "captum": importlib.metadata.version("captum"),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
        "device": str(device),
        "device_name": device_name,
        "compute_capability": compute_capability,
        "model_dtype": "float32",
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cuda_matmul_tf32": False,
        "cudnn_tf32": False,
        "sealed_clean_equivalence_precision": "bf16",
    }


def _expected_metadata(
    *,
    spec: PostEvaluationSpec,
    cohort: _Cohort,
    member: AuditMemberRuntime,
    method: str,
    settings: _Settings,
) -> dict[str, object]:
    decision_hash = member.decisions.integrity_sha256
    prediction_hash = member.sealed_prediction.integrity_sha256
    if decision_hash is None or prediction_hash is None:
        raise ExplanationAuditIntegrityError("member sources are not integrity-bound")
    return {
        "post_evaluation_spec_sha256": spec.artifact_sha256,
        "cohort_sha256": cohort.payload_sha256,
        "member_id": member.member_id,
        "architecture": member.architecture,
        "seed": member.seed,
        "method": method,
        "checkpoint_sha256": member.checkpoint_sha256,
        "calibration_decision_sha256": decision_hash,
        "sealed_prediction_sha256": prediction_hash,
        "temperature": member.decisions.temperature_scaling.temperature,
        "target_score": (
            "signed_correct_status_logit; sign=+1 for positive cell and -1 for "
            "negative cell; probability=sigmoid(signed_logit/temperature)"
        ),
        "attribution_runtime": _runtime_block(member),
        "settings": {
            "ig_steps": settings.ig_steps,
            "ig_internal_batch_size": settings.ig_internal_batch_size,
            "occlusion_window": settings.occlusion_window,
            "occlusion_stride": settings.occlusion_stride,
            "occlusion_perturbations": settings.occlusion_perturbations,
            "fractions": list(settings.fractions),
            "random_ranking_replicates": settings.random_ranking_replicates,
            "random_ranking_seed": settings.random_ranking_seed,
            "randomization_seeds": list(settings.randomization_seeds),
            "stability_snr_db": settings.stability_snr_db,
            "stability_replicates": settings.stability_replicates,
            "outer_batch_size": settings.outer_batch_size,
            "faithfulness_scoring_batch_size": 60,
            "numeric_precision": "float32_model_and_attributions_float64_summaries",
        },
    }


@contextmanager
def _deterministic_torch(seed: int) -> Iterator[None]:
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(previous_deterministic)
            torch.backends.cudnn.benchmark = previous_benchmark
            torch.backends.cudnn.deterministic = previous_cudnn_deterministic
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
            torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32


def _cohort_inputs(member: AuditMemberRuntime, cohort: _Cohort) -> Tensor:
    manifest = member.selected_manifest
    positions = {int(ecg_id): index for index, ecg_id in enumerate(manifest["ecg_id"].tolist())}
    physical: list[Tensor] = []
    for row_index, ecg_id in enumerate(cohort.ecg_id.tolist()):
        if ecg_id not in positions:
            raise ExplanationAuditIntegrityError(f"cohort ECG {ecg_id} is unavailable")
        position = positions[ecg_id]
        row = manifest.iloc[position]
        if int(row["patient_id"]) != int(cohort.patient_id[row_index]):
            raise ExplanationAuditIntegrityError("cohort patient identity differs")
        signal, target = member.physical_dataset[position]
        observed_target = target.detach().cpu().numpy().astype(np.int8, copy=False)
        if not np.array_equal(observed_target, cohort.target_bits[row_index]):
            raise ExplanationAuditIntegrityError("cohort target bits differ from physical data")
        physical.append(signal.detach().to(device="cpu", dtype=torch.float32))
    normalized = member.normalize_physical_batch(torch.stack(physical, dim=0))
    return normalized.contiguous()


def _method_attributions(
    method: str,
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    settings: _Settings,
) -> tuple[Tensor, Tensor | None]:
    if method == "grad_cam_1d":
        if not isinstance(model, ResNet1D):
            raise ExplanationAuditIntegrityError("Grad-CAM requires the ResNet member")
        return grad_cam_1d(model, inputs, targets, normalize=True), None
    if method == "integrated_gradients":
        return integrated_gradients_with_delta(
            model,
            inputs,
            targets,
            n_steps=settings.ig_steps,
            internal_batch_size=settings.ig_internal_batch_size,
            normalize=True,
        )
    if method == "temporal_occlusion":
        return (
            temporal_occlusion(
                model,
                inputs,
                targets,
                window_samples=settings.occlusion_window,
                stride_samples=settings.occlusion_stride,
                perturbations_per_eval=settings.occlusion_perturbations,
                normalize=True,
            ),
            None,
        )
    raise ExplanationAuditIntegrityError(f"unsupported explanation method {method!r}")


def _batched_attributions(
    method: str,
    model: nn.Module,
    inputs_cpu: Tensor,
    targets_cpu: Tensor,
    target_signs_cpu: Tensor,
    settings: _Settings,
) -> tuple[Tensor, Tensor | None]:
    device = next(model.parameters()).device
    attributions: list[Tensor] = []
    deltas: list[Tensor] = []
    for start in range(0, inputs_cpu.shape[0], settings.outer_batch_size):
        stop = min(start + settings.outer_batch_size, inputs_cpu.shape[0])
        inputs = inputs_cpu[start:stop].to(device=device, dtype=torch.float32)
        targets = targets_cpu[start:stop].to(device=device, dtype=torch.int64)
        signs = target_signs_cpu[start:stop].to(device=device, dtype=torch.float32)
        batch_attributions, batch_delta = _method_attributions(
            method, model, inputs, targets, settings
        )
        batch_attributions = batch_attributions * signs[:, None, None]
        if batch_delta is not None:
            batch_delta = batch_delta * signs.detach().cpu().to(batch_delta.dtype)
        attributions.append(batch_attributions.detach().cpu().to(torch.float32))
        if batch_delta is not None:
            deltas.append(batch_delta.detach().cpu().to(torch.float64))
    combined = torch.cat(attributions, dim=0)
    return combined, (torch.cat(deltas) if deltas else None)


def _noisy_inputs(
    inputs: Tensor,
    ecg_id: NDArray[np.int64],
    *,
    snr_db: float,
    replicate: int,
) -> Tensor:
    result = inputs.detach().cpu().clone().to(torch.float32)
    for index, identifier in enumerate(ecg_id.tolist()):
        signal = result[index]
        rms = float(torch.sqrt(torch.mean(signal.square())).item())
        scale = (rms if rms > 0.0 else 1.0) / (10.0 ** (snr_db / 20.0))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_uint32("explanation-stability", replicate, int(identifier)))
        result[index] = (
            signal
            + torch.randn(
                signal.shape,
                generator=generator,
                dtype=torch.float32,
            )
            * scale
        )
    return result


def _fp32_sealed_logit_bridge(
    member: AuditMemberRuntime,
    cohort: _Cohort,
    inputs_cpu: Tensor,
    *,
    batch_size: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    model = member.model.float().eval()
    device = next(model.parameters()).device
    observed: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, inputs_cpu.shape[0], batch_size):
            stop = min(start + batch_size, inputs_cpu.shape[0])
            logits = model(inputs_cpu[start:stop].to(device=device, dtype=torch.float32))
            if not isinstance(logits, Tensor) or logits.shape != (
                stop - start,
                len(LABEL_ORDER),
            ):
                raise ExplanationAuditIntegrityError("FP32 cohort logits are invalid")
            observed.append(logits.detach().cpu().to(torch.float32))
    fp32 = torch.cat(observed).numpy().astype(np.float64, copy=False)
    sealed_positions = {
        int(ecg_id): index for index, ecg_id in enumerate(member.sealed_prediction.ecg_id.tolist())
    }
    try:
        sealed = np.asarray(
            [
                member.sealed_prediction.raw_logits[sealed_positions[int(ecg_id)]]
                for ecg_id in cohort.ecg_id.tolist()
            ],
            dtype=np.float64,
        )
    except KeyError as error:
        raise ExplanationAuditIntegrityError(
            "cohort ECG is absent from the sealed prediction"
        ) from error
    if fp32.shape != sealed.shape or not np.all(np.isfinite(fp32)):
        raise ExplanationAuditIntegrityError("FP32/sealed cohort logits do not align")
    return fp32, sealed, fp32 - sealed


def _curve_arrays(curve: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    logits = curve.target_logits.detach().cpu().numpy().astype(np.float64, copy=False)
    probabilities = curve.target_probabilities.detach().cpu().numpy().astype(np.float64, copy=False)
    return logits, probabilities


def _faithfulness_arrays(
    *,
    model: nn.Module,
    inputs_cpu: Tensor,
    targets_cpu: Tensor,
    target_signs_cpu: Tensor,
    attributions_cpu: Tensor,
    temperature: float,
    settings: _Settings,
    lead_ablation_applicable: bool,
) -> dict[str, NDArray[np.generic]]:
    device = next(model.parameters()).device
    inputs = inputs_cpu.to(device=device, dtype=torch.float32)
    targets = targets_cpu.to(device=device, dtype=torch.int64)
    target_signs = target_signs_cpu.to(device=device, dtype=torch.float32)
    attributions = attributions_cpu.to(device=device, dtype=torch.float32)
    fractions = settings.fractions
    result: dict[str, NDArray[np.generic]] = {"fractions": np.asarray(fractions, dtype=np.float64)}
    for name, operation, ranking in (
        ("deletion", "deletion", "most_important"),
        ("insertion", "insertion", "most_important"),
        ("least_deletion", "deletion", "least_important"),
    ):
        curve = temporal_faithfulness_curve(
            model,
            inputs,
            attributions,
            targets,
            fractions=fractions,
            operation=cast(Literal["deletion", "insertion"], operation),
            ranking=cast(Literal["most_important", "least_important", "random"], ranking),
            temperature=temperature,
            target_signs=target_signs,
        )
        logits, probabilities = _curve_arrays(curve)
        result[f"{name}_logits"] = logits
        result[f"{name}_probabilities"] = probabilities
    random_logits: list[NDArray[np.float64]] = []
    random_probabilities: list[NDArray[np.float64]] = []
    for replicate in range(settings.random_ranking_replicates):
        curve = temporal_faithfulness_curve(
            model,
            inputs,
            attributions,
            targets,
            fractions=fractions,
            operation="deletion",
            ranking="random",
            random_seed=settings.random_ranking_seed + replicate,
            temperature=temperature,
            target_signs=target_signs,
        )
        logits, probabilities = _curve_arrays(curve)
        random_logits.append(logits)
        random_probabilities.append(probabilities)
    result["random_deletion_logits"] = np.stack(random_logits)
    result["random_deletion_probabilities"] = np.stack(random_probabilities)

    if lead_ablation_applicable:
        if attributions.shape[1] != 12:
            raise ExplanationAuditIntegrityError(
                "lead-ablation method did not produce lead-resolved attributions"
            )
        result["lead_fractions"] = np.arange(13, dtype=np.float64) / 12.0
        for name, ranking in (
            ("lead_ablation", "most_important"),
            ("least_lead_ablation", "least_important"),
        ):
            curve = lead_ablation_faithfulness_curve(
                model,
                inputs,
                attributions,
                targets,
                ranking=cast(Literal["most_important", "least_important", "random"], ranking),
                temperature=temperature,
                target_signs=target_signs,
            )
            logits, probabilities = _curve_arrays(curve)
            result[f"{name}_logits"] = logits
            result[f"{name}_probabilities"] = probabilities
        lead_random_logits: list[NDArray[np.float64]] = []
        lead_random_probabilities: list[NDArray[np.float64]] = []
        for replicate in range(settings.random_ranking_replicates):
            curve = lead_ablation_faithfulness_curve(
                model,
                inputs,
                attributions,
                targets,
                ranking="random",
                random_seed=settings.random_ranking_seed + replicate,
                temperature=temperature,
                target_signs=target_signs,
            )
            logits, probabilities = _curve_arrays(curve)
            lead_random_logits.append(logits)
            lead_random_probabilities.append(probabilities)
        result["random_lead_ablation_logits"] = np.stack(lead_random_logits)
        result["random_lead_ablation_probabilities"] = np.stack(lead_random_probabilities)
    return result


def _per_example_auc(
    values: NDArray[np.float64], fractions: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.asarray(np.trapezoid(values, x=fractions, axis=-1), dtype=np.float64)


def _run_method_arrays(
    *,
    member: AuditMemberRuntime,
    method: str,
    cohort: _Cohort,
    inputs_cpu: Tensor,
    settings: _Settings,
) -> dict[str, NDArray[np.generic]]:
    model = member.model.float().eval()
    targets_cpu = torch.from_numpy(cohort.target_index.copy()).to(torch.int64)
    target_signs_cpu = torch.where(
        torch.from_numpy(cohort.target_value.copy()).to(torch.int8) == 1,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    )
    seed = _stable_uint32("explanation-method", member.member_id, method)
    with _deterministic_torch(seed):
        fp32_logits, sealed_logits, logit_delta = _fp32_sealed_logit_bridge(
            member,
            cohort,
            inputs_cpu,
            batch_size=settings.outer_batch_size,
        )
        attributions, delta = _batched_attributions(
            method, model, inputs_cpu, targets_cpu, target_signs_cpu, settings
        )
        repeated, repeated_delta = _batched_attributions(
            method, model, inputs_cpu, targets_cpu, target_signs_cpu, settings
        )
        if not torch.equal(attributions, repeated):
            maximum = float((attributions - repeated).abs().max().item())
            raise ExplanationAuditError(
                f"deterministic rerun differed for {member.member_id}/{method}: "
                f"max_abs={maximum:.9g}"
            )
        if (delta is None) != (repeated_delta is None) or (
            delta is not None
            and repeated_delta is not None
            and not torch.equal(delta, repeated_delta)
        ):
            raise ExplanationAuditError(
                f"deterministic completeness delta differed for {member.member_id}/{method}"
            )

        repeat_similarity = (
            attribution_stability_similarity(attributions, repeated)
            .values.numpy()
            .astype(np.float64, copy=False)
        )
        stability: list[NDArray[np.float64]] = []
        for replicate in range(settings.stability_replicates):
            noisy = _noisy_inputs(
                inputs_cpu,
                cohort.ecg_id,
                snr_db=settings.stability_snr_db,
                replicate=replicate,
            )
            noisy_attributions, _ = _batched_attributions(
                method, model, noisy, targets_cpu, target_signs_cpu, settings
            )
            similarity = attribution_stability_similarity(attributions, noisy_attributions).values
            stability.append(similarity.numpy().astype(np.float64, copy=False))

        randomization: list[NDArray[np.float64]] = []
        for randomization_seed in settings.randomization_seeds:
            randomized = randomized_model_copy(model, seed=randomization_seed).float().eval()
            randomized_attributions, _ = _batched_attributions(
                method,
                randomized,
                inputs_cpu,
                targets_cpu,
                target_signs_cpu,
                settings,
            )
            similarity = parameter_randomization_comparison(
                attributions,
                randomized_attributions,
                seed=randomization_seed,
            ).values
            randomization.append(similarity.numpy().astype(np.float64, copy=False))
            del randomized, randomized_attributions
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        arrays: dict[str, NDArray[np.generic]] = {
            "ecg_id": cohort.ecg_id,
            "patient_id": cohort.patient_id,
            "target_index": cohort.target_index,
            "target_value": cohort.target_value,
            "target_sign": target_signs_cpu.numpy().astype(np.int8, copy=False),
            "target_bits": cohort.target_bits,
            "fp32_raw_logits": fp32_logits,
            "sealed_bf16_raw_logits": sealed_logits,
            "fp32_minus_sealed_logits": logit_delta,
            "attributions": attributions.numpy().astype(np.float32, copy=False),
            "deterministic_repeat_exact": np.ones(cohort.size, dtype=np.uint8),
            "deterministic_repeat_cosine": repeat_similarity,
            "stability_cosine": np.stack(stability),
            "parameter_randomization_cosine": np.stack(randomization),
        }
        if delta is not None:
            arrays["ig_completeness_delta"] = delta.numpy().astype(np.float64, copy=False)
        arrays.update(
            _faithfulness_arrays(
                model=model,
                inputs_cpu=inputs_cpu,
                targets_cpu=targets_cpu,
                target_signs_cpu=target_signs_cpu,
                attributions_cpu=attributions,
                temperature=member.decisions.temperature_scaling.temperature,
                settings=settings,
                lead_ablation_applicable=(method == "integrated_gradients"),
            )
        )
    fractions = cast(NDArray[np.float64], arrays["fractions"])
    deletion = cast(NDArray[np.float64], arrays["deletion_probabilities"])
    insertion = cast(NDArray[np.float64], arrays["insertion_probabilities"])
    least = cast(NDArray[np.float64], arrays["least_deletion_probabilities"])
    random_deletion = cast(NDArray[np.float64], arrays["random_deletion_probabilities"])
    arrays["deletion_auc"] = _per_example_auc(deletion, fractions)
    arrays["insertion_auc"] = _per_example_auc(insertion, fractions)
    arrays["least_deletion_auc"] = _per_example_auc(least, fractions)
    random_auc = _per_example_auc(random_deletion, fractions)
    arrays["random_deletion_auc"] = random_auc
    arrays["guided_vs_random_deletion_advantage"] = random_auc.mean(axis=0) - cast(
        NDArray[np.float64], arrays["deletion_auc"]
    )
    if "lead_ablation_probabilities" in arrays:
        lead_fractions = cast(NDArray[np.float64], arrays["lead_fractions"])
        lead = cast(NDArray[np.float64], arrays["lead_ablation_probabilities"])
        least_lead = cast(NDArray[np.float64], arrays["least_lead_ablation_probabilities"])
        random_lead = cast(NDArray[np.float64], arrays["random_lead_ablation_probabilities"])
        arrays["lead_ablation_auc"] = _per_example_auc(lead, lead_fractions)
        arrays["least_lead_ablation_auc"] = _per_example_auc(least_lead, lead_fractions)
        random_lead_auc = _per_example_auc(random_lead, lead_fractions)
        arrays["random_lead_ablation_auc"] = random_lead_auc
        arrays["guided_vs_random_lead_advantage"] = random_lead_auc.mean(axis=0) - cast(
            NDArray[np.float64], arrays["lead_ablation_auc"]
        )
    return arrays


def _mean(values: NDArray[np.generic]) -> float:
    return float(np.asarray(values, dtype=np.float64).mean())


def _method_summary(artifact: AuditArrayArtifact) -> dict[str, object]:
    arrays = artifact.arrays
    attribution = np.asarray(arrays["attributions"])
    summary: dict[str, object] = {
        "examples": int(attribution.shape[0]),
        "attribution_shape": list(attribution.shape),
        "deterministic_repeat_exact": bool(
            np.all(np.asarray(arrays["deterministic_repeat_exact"]) == 1)
        ),
        "mean_repeat_cosine": _mean(arrays["deterministic_repeat_cosine"]),
        "mean_stability_cosine_40db": _mean(arrays["stability_cosine"]),
        "mean_parameter_randomization_cosine": _mean(arrays["parameter_randomization_cosine"]),
        "mean_deletion_auc": _mean(arrays["deletion_auc"]),
        "mean_insertion_auc": _mean(arrays["insertion_auc"]),
        "mean_least_deletion_auc": _mean(arrays["least_deletion_auc"]),
        "mean_random_deletion_auc": _mean(arrays["random_deletion_auc"]),
        "mean_guided_vs_random_deletion_advantage": _mean(
            arrays["guided_vs_random_deletion_advantage"]
        ),
        "fp32_vs_sealed_bf16_logit_drift": {
            "mean_absolute": float(
                np.abs(np.asarray(arrays["fp32_minus_sealed_logits"], dtype=np.float64)).mean()
            ),
            "maximum_absolute": float(
                np.abs(np.asarray(arrays["fp32_minus_sealed_logits"], dtype=np.float64)).max()
            ),
            "root_mean_square": float(
                np.sqrt(
                    np.square(
                        np.asarray(arrays["fp32_minus_sealed_logits"], dtype=np.float64)
                    ).mean()
                )
            ),
        },
    }
    if "ig_completeness_delta" in arrays:
        delta = np.abs(np.asarray(arrays["ig_completeness_delta"], dtype=np.float64))
        summary["mean_absolute_ig_completeness_delta"] = float(delta.mean())
        summary["maximum_absolute_ig_completeness_delta"] = float(delta.max())
    if "lead_ablation_auc" in arrays:
        summary.update(
            {
                "mean_lead_ablation_auc": _mean(arrays["lead_ablation_auc"]),
                "mean_least_lead_ablation_auc": _mean(arrays["least_lead_ablation_auc"]),
                "mean_random_lead_ablation_auc": _mean(arrays["random_lead_ablation_auc"]),
                "mean_guided_vs_random_lead_advantage": _mean(
                    arrays["guided_vs_random_lead_advantage"]
                ),
            }
        )
    target_index = np.asarray(arrays["target_index"], dtype=np.int64)
    target_value = np.asarray(arrays["target_value"], dtype=np.int8)
    cells: list[dict[str, object]] = []
    for label_index, label in enumerate(LABEL_ORDER):
        for value, status in ((0, "negative"), (1, "positive")):
            selected = (target_index == label_index) & (target_value == value)
            if int(selected.sum()) != 6:
                raise ExplanationAuditIntegrityError(
                    "method artifact label/status cells are not balanced"
                )
            cell: dict[str, object] = {
                "target_label": label,
                "target_status": status,
                "count": 6,
                "mean_stability_cosine_40db": float(
                    np.asarray(arrays["stability_cosine"], dtype=np.float64)[:, selected].mean()
                ),
                "mean_parameter_randomization_cosine": float(
                    np.asarray(arrays["parameter_randomization_cosine"], dtype=np.float64)[
                        :, selected
                    ].mean()
                ),
                "mean_deletion_auc": float(
                    np.asarray(arrays["deletion_auc"], dtype=np.float64)[selected].mean()
                ),
                "mean_insertion_auc": float(
                    np.asarray(arrays["insertion_auc"], dtype=np.float64)[selected].mean()
                ),
                "mean_guided_vs_random_deletion_advantage": float(
                    np.asarray(
                        arrays["guided_vs_random_deletion_advantage"],
                        dtype=np.float64,
                    )[selected].mean()
                ),
            }
            if "lead_ablation_auc" in arrays:
                cell["mean_guided_vs_random_lead_advantage"] = float(
                    np.asarray(arrays["guided_vs_random_lead_advantage"], dtype=np.float64)[
                        selected
                    ].mean()
                )
            cells.append(cell)
    summary["label_status_cells"] = cells
    return summary


def _required_array(
    artifact: AuditArrayArtifact,
    name: str,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
) -> NDArray[np.generic]:
    if name not in artifact.arrays:
        raise ExplanationAuditIntegrityError(f"explanation array {name!r} is missing")
    array = artifact.arrays[name]
    if array.dtype != dtype or array.shape != shape:
        raise ExplanationAuditIntegrityError(
            f"explanation array {name!r} has an invalid dtype or shape"
        )
    return array


def _validate_method_artifact(
    artifact: AuditArrayArtifact,
    *,
    method: str,
    expected_cohort: _Cohort,
    expected_summary: object | None = None,
) -> None:
    if set(artifact.metadata) != {
        "post_evaluation_spec_sha256",
        "cohort_sha256",
        "member_id",
        "architecture",
        "seed",
        "method",
        "checkpoint_sha256",
        "calibration_decision_sha256",
        "sealed_prediction_sha256",
        "temperature",
        "target_score",
        "attribution_runtime",
        "settings",
    }:
        raise ExplanationAuditIntegrityError("method artifact metadata keys are not canonical")
    if (
        artifact.metadata.get("method") != method
        or artifact.metadata.get("cohort_sha256") != expected_cohort.payload_sha256
        or artifact.metadata.get("target_score")
        != (
            "signed_correct_status_logit; sign=+1 for positive cell and -1 for "
            "negative cell; probability=sigmoid(signed_logit/temperature)"
        )
    ):
        raise ExplanationAuditIntegrityError("method artifact protocol metadata differs")
    settings = _mapping(artifact.metadata.get("settings"), "method settings")
    expected_settings = {
        "ig_steps": 32,
        "ig_internal_batch_size": 8,
        "occlusion_window": 50,
        "occlusion_stride": 25,
        "occlusion_perturbations": 16,
        "fractions": [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        "random_ranking_replicates": 20,
        "random_ranking_seed": 20_261_008,
        "randomization_seeds": [2_026_801, 2_026_802, 2_026_803],
        "stability_snr_db": 40.0,
        "stability_replicates": 3,
        "outer_batch_size": 4,
        "faithfulness_scoring_batch_size": 60,
        "numeric_precision": "float32_model_and_attributions_float64_summaries",
    }
    if dict(settings) != expected_settings:
        raise ExplanationAuditIntegrityError("method artifact settings differ")
    cohort_size = 60
    fraction_count = 8
    random_count = 20
    attribution_channels = 1 if method == "grad_cam_1d" else 12
    base_names = {
        "ecg_id",
        "patient_id",
        "target_index",
        "target_value",
        "target_sign",
        "target_bits",
        "fp32_raw_logits",
        "sealed_bf16_raw_logits",
        "fp32_minus_sealed_logits",
        "attributions",
        "deterministic_repeat_exact",
        "deterministic_repeat_cosine",
        "stability_cosine",
        "parameter_randomization_cosine",
        "fractions",
        "deletion_logits",
        "deletion_probabilities",
        "insertion_logits",
        "insertion_probabilities",
        "least_deletion_logits",
        "least_deletion_probabilities",
        "random_deletion_logits",
        "random_deletion_probabilities",
        "deletion_auc",
        "insertion_auc",
        "least_deletion_auc",
        "random_deletion_auc",
        "guided_vs_random_deletion_advantage",
    }
    lead_names = {
        "lead_fractions",
        "lead_ablation_logits",
        "lead_ablation_probabilities",
        "least_lead_ablation_logits",
        "least_lead_ablation_probabilities",
        "random_lead_ablation_logits",
        "random_lead_ablation_probabilities",
        "lead_ablation_auc",
        "least_lead_ablation_auc",
        "random_lead_ablation_auc",
        "guided_vs_random_lead_advantage",
    }
    expected_names = set(base_names)
    if method == "integrated_gradients":
        expected_names.add("ig_completeness_delta")
        expected_names.update(lead_names)
    if set(artifact.arrays) != expected_names:
        raise ExplanationAuditIntegrityError("method artifact array set is not canonical")
    ecg_id = _required_array(artifact, "ecg_id", dtype=np.dtype(np.int64), shape=(cohort_size,))
    patient_id = _required_array(
        artifact, "patient_id", dtype=np.dtype(np.int64), shape=(cohort_size,)
    )
    target_index = _required_array(
        artifact, "target_index", dtype=np.dtype(np.int64), shape=(cohort_size,)
    )
    target_value = _required_array(
        artifact, "target_value", dtype=np.dtype(np.int8), shape=(cohort_size,)
    )
    target_sign = _required_array(
        artifact, "target_sign", dtype=np.dtype(np.int8), shape=(cohort_size,)
    )
    target_bits = _required_array(
        artifact,
        "target_bits",
        dtype=np.dtype(np.int8),
        shape=(cohort_size, len(LABEL_ORDER)),
    )
    if (
        np.unique(ecg_id).size != cohort_size
        or np.unique(patient_id).size != cohort_size
        or np.any(np.asarray(target_index) < 0)
        or np.any(np.asarray(target_index) >= len(LABEL_ORDER))
        or not np.logical_or(target_value == 0, target_value == 1).all()
        or not np.array_equal(target_sign, np.where(target_value == 1, 1, -1).astype(np.int8))
        or not np.logical_or(target_bits == 0, target_bits == 1).all()
        or not np.array_equal(
            np.asarray(target_bits)[np.arange(cohort_size), np.asarray(target_index)],
            target_value,
        )
    ):
        raise ExplanationAuditIntegrityError("method artifact cohort arrays are invalid")
    if (
        not np.array_equal(ecg_id, expected_cohort.ecg_id)
        or not np.array_equal(patient_id, expected_cohort.patient_id)
        or not np.array_equal(target_index, expected_cohort.target_index)
        or not np.array_equal(target_value, expected_cohort.target_value)
        or not np.array_equal(target_bits, expected_cohort.target_bits)
    ):
        raise ExplanationAuditIntegrityError("method artifact arrays differ from the frozen cohort")
    fp32_logits = _required_array(
        artifact,
        "fp32_raw_logits",
        dtype=np.dtype(np.float64),
        shape=(cohort_size, len(LABEL_ORDER)),
    )
    sealed_logits = _required_array(
        artifact,
        "sealed_bf16_raw_logits",
        dtype=np.dtype(np.float64),
        shape=(cohort_size, len(LABEL_ORDER)),
    )
    logit_delta = _required_array(
        artifact,
        "fp32_minus_sealed_logits",
        dtype=np.dtype(np.float64),
        shape=(cohort_size, len(LABEL_ORDER)),
    )
    if not np.array_equal(
        logit_delta,
        np.asarray(fp32_logits, dtype=np.float64) - np.asarray(sealed_logits, dtype=np.float64),
    ):
        raise ExplanationAuditIntegrityError("FP32/sealed logit drift is not reproducible")
    _required_array(
        artifact,
        "attributions",
        dtype=np.dtype(np.float32),
        shape=(cohort_size, attribution_channels, 1000),
    )
    repeat_exact = _required_array(
        artifact,
        "deterministic_repeat_exact",
        dtype=np.dtype(np.uint8),
        shape=(cohort_size,),
    )
    repeat_cosine = _required_array(
        artifact,
        "deterministic_repeat_cosine",
        dtype=np.dtype(np.float64),
        shape=(cohort_size,),
    )
    if not np.all(repeat_exact == 1) or not np.allclose(repeat_cosine, 1.0, rtol=0.0, atol=1e-12):
        raise ExplanationAuditIntegrityError("deterministic repeat evidence differs")
    for name, shape in (
        ("stability_cosine", (3, cohort_size)),
        ("parameter_randomization_cosine", (3, cohort_size)),
    ):
        raw_values = _required_array(artifact, name, dtype=np.dtype(np.float64), shape=shape)
        values = np.asarray(raw_values, dtype=np.float64)
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ExplanationAuditIntegrityError(f"{name} lies outside [-1, 1]")
    fractions = _required_array(
        artifact,
        "fractions",
        dtype=np.dtype(np.float64),
        shape=(fraction_count,),
    )
    if not np.array_equal(fractions, np.asarray([0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])):
        raise ExplanationAuditIntegrityError("faithfulness fractions differ")
    for prefix in ("deletion", "insertion", "least_deletion"):
        logits = _required_array(
            artifact,
            f"{prefix}_logits",
            dtype=np.dtype(np.float64),
            shape=(cohort_size, fraction_count),
        )
        probabilities = _required_array(
            artifact,
            f"{prefix}_probabilities",
            dtype=np.dtype(np.float64),
            shape=(cohort_size, fraction_count),
        )
        _validate_probability_logits(logits, probabilities, artifact)
    random_logits = _required_array(
        artifact,
        "random_deletion_logits",
        dtype=np.dtype(np.float64),
        shape=(random_count, cohort_size, fraction_count),
    )
    random_probabilities = _required_array(
        artifact,
        "random_deletion_probabilities",
        dtype=np.dtype(np.float64),
        shape=(random_count, cohort_size, fraction_count),
    )
    _validate_probability_logits(random_logits, random_probabilities, artifact)
    deletion_probabilities = np.asarray(artifact.arrays["deletion_probabilities"], dtype=np.float64)
    insertion_probabilities = np.asarray(
        artifact.arrays["insertion_probabilities"], dtype=np.float64
    )
    least_probabilities = np.asarray(
        artifact.arrays["least_deletion_probabilities"], dtype=np.float64
    )
    if not (
        np.allclose(
            deletion_probabilities[:, 0],
            insertion_probabilities[:, -1],
            rtol=0.0,
            atol=1e-7,
        )
        and np.allclose(
            deletion_probabilities[:, 0],
            least_probabilities[:, 0],
            rtol=0.0,
            atol=1e-7,
        )
        and np.allclose(
            random_probabilities[:, :, 0],
            deletion_probabilities[None, :, 0],
            rtol=0.0,
            atol=1e-7,
        )
        and np.allclose(
            deletion_probabilities[:, -1],
            insertion_probabilities[:, 0],
            rtol=0.0,
            atol=1e-7,
        )
        and np.allclose(
            deletion_probabilities[:, -1],
            least_probabilities[:, -1],
            rtol=0.0,
            atol=1e-7,
        )
        and np.allclose(
            random_probabilities[:, :, -1],
            deletion_probabilities[None, :, -1],
            rtol=0.0,
            atol=1e-7,
        )
    ):
        raise ExplanationAuditIntegrityError("faithfulness curve endpoints differ")
    expected_auc = {
        "deletion_auc": _per_example_auc(
            cast(NDArray[np.float64], artifact.arrays["deletion_probabilities"]),
            cast(NDArray[np.float64], fractions),
        ),
        "insertion_auc": _per_example_auc(
            cast(NDArray[np.float64], artifact.arrays["insertion_probabilities"]),
            cast(NDArray[np.float64], fractions),
        ),
        "least_deletion_auc": _per_example_auc(
            cast(NDArray[np.float64], artifact.arrays["least_deletion_probabilities"]),
            cast(NDArray[np.float64], fractions),
        ),
        "random_deletion_auc": _per_example_auc(
            cast(NDArray[np.float64], artifact.arrays["random_deletion_probabilities"]),
            cast(NDArray[np.float64], fractions),
        ),
    }
    for name, expected in expected_auc.items():
        observed = _required_array(artifact, name, dtype=np.dtype(np.float64), shape=expected.shape)
        if not np.array_equal(observed, expected):
            raise ExplanationAuditIntegrityError(f"stored {name} is not reproducible")
    expected_advantage = (
        expected_auc["random_deletion_auc"].mean(axis=0) - expected_auc["deletion_auc"]
    )
    advantage = _required_array(
        artifact,
        "guided_vs_random_deletion_advantage",
        dtype=np.dtype(np.float64),
        shape=(cohort_size,),
    )
    if not np.array_equal(advantage, expected_advantage):
        raise ExplanationAuditIntegrityError("deletion advantage is not reproducible")
    if method == "integrated_gradients":
        _required_array(
            artifact,
            "ig_completeness_delta",
            dtype=np.dtype(np.float64),
            shape=(cohort_size,),
        )
        _validate_lead_arrays(
            artifact,
            cohort_size=cohort_size,
            random_count=random_count,
            temporal_clean=deletion_probabilities[:, 0],
            temporal_baseline=deletion_probabilities[:, -1],
        )
    recomputed_summary = _method_summary(artifact)
    if expected_summary is not None and expected_summary != recomputed_summary:
        raise ExplanationAuditIntegrityError("method summary is not reproducible")


def _validate_probability_logits(
    logits: NDArray[np.generic],
    probabilities: NDArray[np.generic],
    artifact: AuditArrayArtifact,
) -> None:
    temperature = _number(artifact.metadata.get("temperature"), "temperature")
    observed = np.asarray(probabilities, dtype=np.float64)
    expected = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64) / temperature))
    if (
        np.any(observed < 0.0)
        or np.any(observed > 1.0)
        or not np.allclose(observed, expected, rtol=0.0, atol=1e-7)
    ):
        raise ExplanationAuditIntegrityError("stored probabilities differ from logits")


def _validate_lead_arrays(
    artifact: AuditArrayArtifact,
    *,
    cohort_size: int,
    random_count: int,
    temporal_clean: NDArray[np.float64],
    temporal_baseline: NDArray[np.float64],
) -> None:
    lead_count = 13
    fractions = _required_array(
        artifact,
        "lead_fractions",
        dtype=np.dtype(np.float64),
        shape=(lead_count,),
    )
    if not np.array_equal(fractions, np.arange(lead_count, dtype=np.float64) / 12.0):
        raise ExplanationAuditIntegrityError("lead fractions differ")
    for prefix in ("lead_ablation", "least_lead_ablation"):
        logits = _required_array(
            artifact,
            f"{prefix}_logits",
            dtype=np.dtype(np.float64),
            shape=(cohort_size, lead_count),
        )
        probabilities = _required_array(
            artifact,
            f"{prefix}_probabilities",
            dtype=np.dtype(np.float64),
            shape=(cohort_size, lead_count),
        )
        _validate_probability_logits(logits, probabilities, artifact)
    random_logits = _required_array(
        artifact,
        "random_lead_ablation_logits",
        dtype=np.dtype(np.float64),
        shape=(random_count, cohort_size, lead_count),
    )
    random_probabilities = _required_array(
        artifact,
        "random_lead_ablation_probabilities",
        dtype=np.dtype(np.float64),
        shape=(random_count, cohort_size, lead_count),
    )
    _validate_probability_logits(random_logits, random_probabilities, artifact)
    lead_probabilities = np.asarray(
        artifact.arrays["lead_ablation_probabilities"], dtype=np.float64
    )
    least_probabilities = np.asarray(
        artifact.arrays["least_lead_ablation_probabilities"], dtype=np.float64
    )
    if not (
        np.allclose(lead_probabilities[:, 0], temporal_clean, rtol=0.0, atol=1e-7)
        and np.allclose(least_probabilities[:, 0], temporal_clean, rtol=0.0, atol=1e-7)
        and np.allclose(random_probabilities[:, :, 0], temporal_clean[None, :], rtol=0.0, atol=1e-7)
        and np.allclose(lead_probabilities[:, -1], temporal_baseline, rtol=0.0, atol=1e-7)
        and np.allclose(least_probabilities[:, -1], temporal_baseline, rtol=0.0, atol=1e-7)
        and np.allclose(
            random_probabilities[:, :, -1],
            temporal_baseline[None, :],
            rtol=0.0,
            atol=1e-7,
        )
    ):
        raise ExplanationAuditIntegrityError("lead-ablation endpoints differ")
    expected_auc = {
        "lead_ablation_auc": _per_example_auc(
            cast(NDArray[np.float64], artifact.arrays["lead_ablation_probabilities"]),
            cast(NDArray[np.float64], fractions),
        ),
        "least_lead_ablation_auc": _per_example_auc(
            cast(
                NDArray[np.float64],
                artifact.arrays["least_lead_ablation_probabilities"],
            ),
            cast(NDArray[np.float64], fractions),
        ),
        "random_lead_ablation_auc": _per_example_auc(
            cast(NDArray[np.float64], artifact.arrays["random_lead_ablation_probabilities"]),
            cast(NDArray[np.float64], fractions),
        ),
    }
    for name, expected in expected_auc.items():
        observed = _required_array(artifact, name, dtype=np.dtype(np.float64), shape=expected.shape)
        if not np.array_equal(observed, expected):
            raise ExplanationAuditIntegrityError(f"stored {name} is not reproducible")
    expected_advantage = (
        expected_auc["random_lead_ablation_auc"].mean(axis=0) - expected_auc["lead_ablation_auc"]
    )
    observed_advantage = _required_array(
        artifact,
        "guided_vs_random_lead_advantage",
        dtype=np.dtype(np.float64),
        shape=(cohort_size,),
    )
    if not np.array_equal(observed_advantage, expected_advantage):
        raise ExplanationAuditIntegrityError("lead advantage is not reproducible")


def _method_artifact(
    *,
    output_directory: Path,
    spec: PostEvaluationSpec,
    cohort: _Cohort,
    member: AuditMemberRuntime,
    method: str,
    inputs_cpu: Tensor,
    settings: _Settings,
) -> tuple[AuditArrayArtifact, dict[str, object]]:
    path = (output_directory / "members" / member.member_id / f"{method}.npz").resolve()
    metadata = _expected_metadata(
        spec=spec,
        cohort=cohort,
        member=member,
        method=method,
        settings=settings,
    )
    if path.exists() or path.with_suffix(".json").exists():
        artifact = load_audit_array_artifact(
            path,
            expected_artifact_type=EXPLANATION_ARRAY_TYPE,
            expected_metadata=metadata,
        )
        files = AuditArrayFiles(
            npz_path=artifact.npz_path,
            json_path=artifact.json_path,
            artifact_sha256=artifact.artifact_sha256,
            npz_sha256=artifact.npz_sha256,
            json_file_sha256="sha256:" + sha256_file(artifact.json_path),
        )
    else:
        arrays = _run_method_arrays(
            member=member,
            method=method,
            cohort=cohort,
            inputs_cpu=inputs_cpu,
            settings=settings,
        )
        files = save_audit_array_artifact(
            path,
            artifact_type=EXPLANATION_ARRAY_TYPE,
            arrays=arrays,
            metadata=metadata,
        )
        artifact = load_audit_array_artifact(
            path,
            expected_artifact_type=EXPLANATION_ARRAY_TYPE,
            expected_metadata=metadata,
        )
    if not np.array_equal(artifact.arrays["ecg_id"], cohort.ecg_id):
        raise ExplanationAuditIntegrityError("method artifact cohort alignment differs")
    _validate_method_artifact(
        artifact,
        method=method,
        expected_cohort=cohort,
    )
    binding = _artifact_binding(files)
    binding["method"] = method
    binding["summary"] = _method_summary(artifact)
    return artifact, binding


def _cross_method_metadata(
    *, spec: PostEvaluationSpec, cohort: _Cohort, member: AuditMemberRuntime
) -> dict[str, object]:
    return {
        "post_evaluation_spec_sha256": spec.artifact_sha256,
        "cohort_sha256": cohort.payload_sha256,
        "member_id": member.member_id,
        "architecture": member.architecture,
        "seed": member.seed,
        "aggregation": "absolute_attribution_mean_across_leads_to_time",
        "metrics": ["cosine", "spearman"],
    }


def _cross_method_artifact(
    *,
    output_directory: Path,
    spec: PostEvaluationSpec,
    cohort: _Cohort,
    member: AuditMemberRuntime,
    methods: Mapping[str, AuditArrayArtifact],
) -> dict[str, object]:
    method_names = tuple(methods)
    pair_names: list[str] = []
    cosine: list[NDArray[np.float64]] = []
    spearman: list[NDArray[np.float64]] = []
    cosine_valid: list[NDArray[np.bool_]] = []
    spearman_valid: list[NDArray[np.bool_]] = []
    for left_index, left in enumerate(method_names):
        for right in method_names[left_index + 1 :]:
            pair_names.append(f"{left}__vs__{right}")
            result = cross_method_temporal_similarity(
                torch.from_numpy(np.asarray(methods[left].arrays["attributions"])),
                torch.from_numpy(np.asarray(methods[right].arrays["attributions"])),
            )
            cosine.append(result.cosine.numpy().astype(np.float64, copy=False))
            spearman.append(result.spearman.numpy().astype(np.float64, copy=False))
            cosine_valid.append(result.cosine_valid.numpy().astype(np.bool_, copy=False))
            spearman_valid.append(result.spearman_valid.numpy().astype(np.bool_, copy=False))
    if not pair_names:
        raise ExplanationAuditIntegrityError("cross-method audit requires two methods")
    path = (output_directory / "members" / member.member_id / "cross_method.npz").resolve()
    metadata = _cross_method_metadata(spec=spec, cohort=cohort, member=member)
    if path.exists() or path.with_suffix(".json").exists():
        artifact = load_audit_array_artifact(
            path,
            expected_artifact_type=EXPLANATION_CROSS_METHOD_TYPE,
            expected_metadata=metadata,
        )
        files = AuditArrayFiles(
            npz_path=artifact.npz_path,
            json_path=artifact.json_path,
            artifact_sha256=artifact.artifact_sha256,
            npz_sha256=artifact.npz_sha256,
            json_file_sha256="sha256:" + sha256_file(artifact.json_path),
        )
    else:
        files = save_audit_array_artifact(
            path,
            artifact_type=EXPLANATION_CROSS_METHOD_TYPE,
            arrays={
                "ecg_id": cohort.ecg_id,
                "pair_name": np.asarray(pair_names, dtype=np.str_),
                "cosine": np.stack(cosine),
                "spearman": np.stack(spearman),
                "cosine_valid": np.stack(cosine_valid),
                "spearman_valid": np.stack(spearman_valid),
            },
            metadata=metadata,
        )
        artifact = load_audit_array_artifact(
            path,
            expected_artifact_type=EXPLANATION_CROSS_METHOD_TYPE,
            expected_metadata=metadata,
        )
    observed_pairs = tuple(str(value) for value in artifact.arrays["pair_name"].tolist())
    if observed_pairs != tuple(pair_names):
        raise ExplanationAuditIntegrityError("cross-method pair order differs")
    _validate_cross_method_artifact(
        artifact,
        cohort=cohort,
        methods=methods,
    )
    binding = _artifact_binding(files)
    binding["summary"] = _cross_method_summary(artifact)
    return binding


def _cross_method_summary(artifact: AuditArrayArtifact) -> dict[str, object]:
    pairs = [str(value) for value in artifact.arrays["pair_name"].tolist()]
    cosine = np.asarray(artifact.arrays["cosine"], dtype=np.float64)
    spearman = np.asarray(artifact.arrays["spearman"], dtype=np.float64)
    cosine_valid = np.asarray(artifact.arrays["cosine_valid"], dtype=np.bool_)
    spearman_valid = np.asarray(artifact.arrays["spearman_valid"], dtype=np.bool_)

    def valid_means(values: NDArray[np.float64], valid: NDArray[np.bool_]) -> list[float | None]:
        return [
            float(values[index][valid[index]].mean()) if np.any(valid[index]) else None
            for index in range(values.shape[0])
        ]

    return {
        "pairs": pairs,
        "mean_cosine": valid_means(cosine, cosine_valid),
        "mean_spearman": valid_means(spearman, spearman_valid),
        "valid_cosine_examples": [int(row.sum()) for row in cosine_valid],
        "valid_spearman_examples": [int(row.sum()) for row in spearman_valid],
    }


def _validate_cross_method_artifact(
    artifact: AuditArrayArtifact,
    *,
    cohort: _Cohort,
    methods: Mapping[str, AuditArrayArtifact],
    expected_summary: object | None = None,
) -> None:
    if artifact.metadata.get("cohort_sha256") != cohort.payload_sha256:
        raise ExplanationAuditIntegrityError("cross-method cohort metadata differs")
    method_names = tuple(methods)
    expected_pairs = tuple(
        f"{left}__vs__{right}"
        for left_index, left in enumerate(method_names)
        for right in method_names[left_index + 1 :]
    )
    pair_count = len(expected_pairs)
    if set(artifact.arrays) != {
        "ecg_id",
        "pair_name",
        "cosine",
        "spearman",
        "cosine_valid",
        "spearman_valid",
    }:
        raise ExplanationAuditIntegrityError("cross-method array set is not canonical")
    ecg_id = _required_array(artifact, "ecg_id", dtype=np.dtype(np.int64), shape=(cohort.size,))
    if not np.array_equal(ecg_id, cohort.ecg_id):
        raise ExplanationAuditIntegrityError("cross-method cohort differs")
    pair_name = artifact.arrays["pair_name"]
    if pair_name.dtype.kind != "U" or pair_name.shape != (pair_count,):
        raise ExplanationAuditIntegrityError("cross-method pair names are invalid")
    if tuple(str(value) for value in pair_name.tolist()) != expected_pairs:
        raise ExplanationAuditIntegrityError("cross-method pair order differs")
    for name, dtype in (
        ("cosine", np.dtype(np.float64)),
        ("spearman", np.dtype(np.float64)),
        ("cosine_valid", np.dtype(np.bool_)),
        ("spearman_valid", np.dtype(np.bool_)),
    ):
        _required_array(
            artifact,
            name,
            dtype=dtype,
            shape=(pair_count, cohort.size),
        )
    expected_cosine: list[NDArray[np.float64]] = []
    expected_spearman: list[NDArray[np.float64]] = []
    expected_cosine_valid: list[NDArray[np.bool_]] = []
    expected_spearman_valid: list[NDArray[np.bool_]] = []
    for left_index, left in enumerate(method_names):
        for right in method_names[left_index + 1 :]:
            result = cross_method_temporal_similarity(
                torch.from_numpy(np.array(methods[left].arrays["attributions"], copy=True)),
                torch.from_numpy(np.array(methods[right].arrays["attributions"], copy=True)),
            )
            expected_cosine.append(result.cosine.numpy().astype(np.float64, copy=False))
            expected_spearman.append(result.spearman.numpy().astype(np.float64, copy=False))
            expected_cosine_valid.append(result.cosine_valid.numpy().astype(np.bool_, copy=False))
            expected_spearman_valid.append(
                result.spearman_valid.numpy().astype(np.bool_, copy=False)
            )
    for name, expected in (
        ("cosine", np.stack(expected_cosine)),
        ("spearman", np.stack(expected_spearman)),
        ("cosine_valid", np.stack(expected_cosine_valid)),
        ("spearman_valid", np.stack(expected_spearman_valid)),
    ):
        if not np.array_equal(artifact.arrays[name], expected):
            raise ExplanationAuditIntegrityError(f"cross-method {name} is not reproducible")
    summary = _cross_method_summary(artifact)
    if expected_summary is not None and expected_summary != summary:
        raise ExplanationAuditIntegrityError("cross-method summary is not reproducible")


def _cohort_summary(cohort: _Cohort) -> dict[str, object]:
    cells: list[dict[str, object]] = []
    for label_index, label in enumerate(LABEL_ORDER):
        for target_value, status in ((0, "negative"), (1, "positive")):
            count = int(
                np.count_nonzero(
                    (cohort.target_index == label_index) & (cohort.target_value == target_value)
                )
            )
            cells.append(
                {
                    "target_label": label,
                    "target_status": status,
                    "count": count,
                }
            )
    if any(cell["count"] != 6 for cell in cells):
        raise ExplanationAuditIntegrityError("cohort cells are not balanced")
    return {
        "payload_sha256": cohort.payload_sha256,
        "records": cohort.size,
        "unique_ecgs": int(np.unique(cohort.ecg_id).size),
        "unique_patients": int(np.unique(cohort.patient_id).size),
        "cells": cells,
        "selection_used_predictions": False,
        "selection_used_metrics": False,
    }


def _explanation_limitations() -> dict[str, object]:
    return {
        "clinical_localization_ground_truth_available": False,
        "shuffled_label_localization_control_available": False,
        "interpretation": (
            "model-behavior attribution audit only; highlights are not causal, "
            "physiological, diagnostic, or clinical localization evidence"
        ),
        "analysis_status": "post_evaluation_descriptive_audit",
        "example_replacement_or_retuning_performed": False,
    }


def _manifest_path(spec: PostEvaluationSpec) -> Path:
    contract = _mapping(spec.payload["output_contract"], "output_contract")
    artifacts = _mapping(contract["artifacts"], "output artifacts")
    return Path(_string(artifacts["explanations_manifest"], "manifest path")).resolve()


def _output_directory(spec: PostEvaluationSpec) -> Path:
    return _manifest_path(spec).parent


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable explanation manifest already exists: {path}")
    serialized = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"immutable explanation manifest already exists: {path}"
            ) from error
    finally:
        with suppress(OSError):
            temporary.unlink()


def _binding_paths(binding: Mapping[str, object]) -> tuple[Path, Path]:
    npz = _mapping(binding["npz"], "artifact npz binding")
    sidecar = _mapping(binding["sidecar"], "artifact sidecar binding")
    return (
        Path(_string(npz["path"], "artifact npz path")).resolve(),
        Path(_string(sidecar["path"], "artifact sidecar path")).resolve(),
    )


def _verify_array_binding(
    binding: Mapping[str, object],
    *,
    artifact_type: str,
    spec_sha256: str,
    allowed_root: Path,
) -> AuditArrayArtifact:
    expected_binding_keys = {"artifact_sha256", "npz", "sidecar"}
    if artifact_type == EXPLANATION_ARRAY_TYPE:
        expected_binding_keys.update({"method", "summary"})
    elif artifact_type == EXPLANATION_CROSS_METHOD_TYPE:
        expected_binding_keys.add("summary")
    if set(binding) != expected_binding_keys:
        raise ExplanationAuditIntegrityError("array binding keys are not canonical")
    npz_path, sidecar_path = _binding_paths(binding)
    for name in ("npz", "sidecar"):
        if set(_mapping(binding[name], f"{name} binding")) != {"path", "file_sha256"}:
            raise ExplanationAuditIntegrityError(f"{name} binding keys are not canonical")
    if npz_path.parent != sidecar_path.parent or sidecar_path != npz_path.with_suffix(".json"):
        raise ExplanationAuditIntegrityError("array artifact paths are inconsistent")
    if allowed_root != npz_path and allowed_root not in npz_path.parents:
        raise ExplanationAuditIntegrityError("array artifact escapes explanation root")
    npz_binding = _mapping(binding["npz"], "npz binding")
    sidecar_binding = _mapping(binding["sidecar"], "sidecar binding")
    if "sha256:" + sha256_file(npz_path) != _hash(
        npz_binding["file_sha256"], "npz file_sha256"
    ) or "sha256:" + sha256_file(sidecar_path) != _hash(
        sidecar_binding["file_sha256"], "sidecar file_sha256"
    ):
        raise ExplanationAuditIntegrityError("array artifact file hash differs")
    artifact = load_audit_array_artifact(npz_path, expected_artifact_type=artifact_type)
    if artifact.artifact_sha256 != _hash(binding["artifact_sha256"], "array artifact_sha256"):
        raise ExplanationAuditIntegrityError("array artifact identity differs")
    if artifact.metadata.get("post_evaluation_spec_sha256") != spec_sha256:
        raise ExplanationAuditIntegrityError("array artifact binds a different spec")
    return artifact


def run_explanation_audit(
    *,
    spec: PostEvaluationSpec,
    runtime: CompletedAuditRuntime,
    outer_batch_size: int = 4,
    progress: Callable[[str], None] | None = None,
) -> ExplanationManifest:
    """Run or safely resume the exact-six frozen explanation audit."""

    if not isinstance(spec, PostEvaluationSpec):
        raise TypeError("spec must be a PostEvaluationSpec")
    if not isinstance(runtime, CompletedAuditRuntime):
        raise TypeError("runtime must be a CompletedAuditRuntime")
    _assert_runtime_bound_to_spec(spec, runtime)
    if tuple(member.member_id for member in runtime.members) != spec.member_ids:
        raise ExplanationAuditIntegrityError("runtime member grid differs from the spec")
    manifest_path = _manifest_path(spec)
    if manifest_path.exists():
        return load_explanation_manifest(
            manifest_path,
            expected_spec_sha256=spec.artifact_sha256,
            spec=spec,
            verify_sources=True,
        )
    cohort = _cohort_from_spec(spec)
    settings = _settings_from_spec(spec, outer_batch_size=outer_batch_size)
    output_directory = _output_directory(spec)
    member_payloads: list[dict[str, object]] = []
    for member in runtime.members:
        if progress is not None:
            progress(f"loading explanation cohort for {member.member_id}")
        inputs_cpu = _cohort_inputs(member, cohort)
        method_artifacts: dict[str, AuditArrayArtifact] = {}
        method_bindings: list[dict[str, object]] = []
        for method in _method_names(member.architecture):
            if progress is not None:
                progress(f"auditing {member.member_id}/{method}")
            artifact, binding = _method_artifact(
                output_directory=output_directory,
                spec=spec,
                cohort=cohort,
                member=member,
                method=method,
                inputs_cpu=inputs_cpu,
                settings=settings,
            )
            method_artifacts[method] = artifact
            method_bindings.append(binding)
        cross_method = _cross_method_artifact(
            output_directory=output_directory,
            spec=spec,
            cohort=cohort,
            member=member,
            methods=method_artifacts,
        )
        member_payloads.append(
            {
                "member_id": member.member_id,
                "architecture": member.architecture,
                "seed": member.seed,
                "methods": method_bindings,
                "cross_method": cross_method,
            }
        )
        if progress is not None:
            progress(f"completed {member.member_id}")
    protocol = _mapping(spec.payload["protocol"], "protocol")
    body: dict[str, object] = {
        "schema_version": EXPLANATION_MANIFEST_SCHEMA_VERSION,
        "artifact_type": EXPLANATION_MANIFEST_TYPE,
        "post_evaluation_spec_sha256": spec.artifact_sha256,
        "protocol_hash": _hash(protocol["protocol_hash"], "protocol_hash"),
        "cohort": _cohort_summary(cohort),
        "settings": _finite_json(
            _mapping(
                _mapping(
                    _mapping(spec.payload["audit_protocols"], "audit protocols")["explanations"],
                    "explanations",
                )["settings"],
                "explanation settings",
            ),
            "explanation settings",
        ),
        "attribution_runtime": {
            member.member_id: _runtime_block(member) for member in runtime.members
        },
        "clean_logit_equivalence": [evidence.to_dict() for evidence in runtime.clean_equivalence],
        "members": member_payloads,
        "limitations": _explanation_limitations(),
    }
    payload = {**body, "artifact_sha256": canonical_sha256(body)}
    _write_new_json(manifest_path, payload)
    return load_explanation_manifest(
        manifest_path,
        expected_spec_sha256=spec.artifact_sha256,
        spec=spec,
        verify_sources=True,
    )


def load_explanation_manifest(
    path: str | Path,
    *,
    expected_spec_sha256: str,
    spec: PostEvaluationSpec | None = None,
    verify_sources: bool = True,
) -> ExplanationManifest:
    """Load and strictly verify a completed explanation manifest."""

    source = Path(path).resolve()
    try:
        decoded: object = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExplanationAuditIntegrityError(
            f"could not decode explanation manifest: {error}"
        ) from error
    payload = _mapping(decoded, "explanation manifest")
    expected_keys = {
        "schema_version",
        "artifact_type",
        "post_evaluation_spec_sha256",
        "protocol_hash",
        "cohort",
        "settings",
        "attribution_runtime",
        "clean_logit_equivalence",
        "members",
        "limitations",
        "artifact_sha256",
    }
    if set(payload) != expected_keys:
        raise ExplanationAuditIntegrityError("explanation manifest keys are not canonical")
    if (
        payload["schema_version"] != EXPLANATION_MANIFEST_SCHEMA_VERSION
        or payload["artifact_type"] != EXPLANATION_MANIFEST_TYPE
        or payload["post_evaluation_spec_sha256"] != expected_spec_sha256
    ):
        raise ExplanationAuditIntegrityError("explanation manifest identity differs")
    _hash(payload["protocol_hash"], "protocol_hash")
    _mapping(payload["settings"], "manifest settings")
    expected_cohort: _Cohort | None = None
    expected_members_by_id: dict[str, Mapping[str, object]] = {}
    if spec is not None:
        if spec.artifact_sha256 != expected_spec_sha256:
            raise ExplanationAuditIntegrityError("supplied spec identity differs")
        protocol = _mapping(spec.payload["protocol"], "protocol")
        if payload["protocol_hash"] != protocol["protocol_hash"]:
            raise ExplanationAuditIntegrityError("explanation protocol hash differs from the spec")
        protocols = _mapping(spec.payload["audit_protocols"], "audit protocols")
        explanation_protocol = _mapping(protocols["explanations"], "explanations")
        expected_settings = dict(_mapping(explanation_protocol["settings"], "explanation settings"))
        if dict(_mapping(payload["settings"], "manifest settings")) != expected_settings:
            raise ExplanationAuditIntegrityError("explanation settings differ from the spec")
        expected_cohort = _cohort_from_spec(spec)
        expected_members_by_id = {
            _string(member["member_id"], "spec member_id"): member
            for member in (
                _mapping(value, "spec member")
                for value in _sequence(spec.payload["members"], "spec members")
            )
        }
    stored_hash = _hash(payload["artifact_sha256"], "artifact_sha256")
    body = dict(payload)
    del body["artifact_sha256"]
    if canonical_sha256(body) != stored_hash:
        raise ExplanationAuditIntegrityError("explanation manifest self-hash mismatch")
    members = _sequence(payload["members"], "members")
    expected_member_ids = tuple(
        f"{architecture}-seed{seed}"
        for architecture in _METHODS_BY_ARCHITECTURE
        for seed in (2026, 2027, 2028)
    )
    observed_member_ids = tuple(
        _string(_mapping(member, "member")["member_id"], "member_id") for member in members
    )
    if observed_member_ids != expected_member_ids:
        raise ExplanationAuditIntegrityError("explanation manifest member grid differs")
    runtime_blocks = _mapping(payload["attribution_runtime"], "attribution runtime")
    if tuple(runtime_blocks) != expected_member_ids:
        raise ExplanationAuditIntegrityError("attribution runtime member grid differs")
    runtime_keys = {
        "python",
        "numpy",
        "torch",
        "captum",
        "cuda_runtime",
        "cudnn",
        "device",
        "device_name",
        "compute_capability",
        "model_dtype",
        "deterministic_algorithms",
        "cudnn_benchmark",
        "cuda_matmul_tf32",
        "cudnn_tf32",
        "sealed_clean_equivalence_precision",
    }
    for member_id in expected_member_ids:
        block = _mapping(runtime_blocks[member_id], "member attribution runtime")
        if set(block) != runtime_keys:
            raise ExplanationAuditIntegrityError("attribution runtime keys are not canonical")
        for name in ("python", "numpy", "torch", "captum", "device"):
            _string(block[name], f"attribution runtime {name}")
        if block["cuda_runtime"] is not None:
            _string(block["cuda_runtime"], "CUDA runtime")
        if block["cudnn"] is not None:
            _integer(block["cudnn"], "cuDNN version", minimum=1)
        if block["device_name"] is not None:
            _string(block["device_name"], "device name")
        capability = block["compute_capability"]
        if capability is not None:
            components = _sequence(capability, "compute capability")
            if len(components) != 2:
                raise ExplanationAuditIntegrityError("compute capability is not canonical")
            for component in components:
                _integer(component, "compute capability component", minimum=0)
        if (
            block["model_dtype"] != "float32"
            or block["deterministic_algorithms"] is not True
            or block["cudnn_benchmark"] is not False
            or block["cuda_matmul_tf32"] is not False
            or block["cudnn_tf32"] is not False
            or block["sealed_clean_equivalence_precision"] != "bf16"
        ):
            raise ExplanationAuditIntegrityError("attribution runtime policy differs")
    cohort = _mapping(payload["cohort"], "cohort")
    if (
        cohort.get("records") != 60
        or cohort.get("unique_ecgs") != 60
        or cohort.get("unique_patients") != 60
        or cohort.get("selection_used_predictions") is not False
        or cohort.get("selection_used_metrics") is not False
    ):
        raise ExplanationAuditIntegrityError("explanation cohort summary differs")
    _hash(cohort.get("payload_sha256"), "cohort payload_sha256")
    cells = _sequence(cohort.get("cells"), "cohort cells")
    if len(cells) != 10 or any(_mapping(cell, "cohort cell").get("count") != 6 for cell in cells):
        raise ExplanationAuditIntegrityError("explanation cohort cells differ")
    if expected_cohort is not None and dict(cohort) != _cohort_summary(expected_cohort):
        raise ExplanationAuditIntegrityError("manifest cohort summary differs from the spec")
    clean = _sequence(payload["clean_logit_equivalence"], "clean equivalence")
    clean_ids: list[str] = []
    for raw_evidence in clean:
        evidence = _mapping(raw_evidence, "clean equivalence member")
        if set(evidence) != {
            "member_id",
            "record_count",
            "logit_count",
            "sealed_prediction_sha256",
            "exact",
            "mismatch_count",
            "maximum_absolute_error",
            "mean_absolute_error",
        }:
            raise ExplanationAuditIntegrityError("clean-equivalence keys are not canonical")
        member_id = _string(evidence.get("member_id"), "clean member_id")
        clean_ids.append(member_id)
        record_count = _integer(evidence.get("record_count"), "clean record_count", minimum=1)
        logit_count = _integer(evidence.get("logit_count"), "clean logit_count", minimum=1)
        if (
            evidence.get("exact") is not True
            or _integer(evidence.get("mismatch_count"), "clean mismatch_count", minimum=0) != 0
            or _number(evidence.get("maximum_absolute_error"), "clean maximum error") != 0.0
            or _number(evidence.get("mean_absolute_error"), "clean mean error") != 0.0
            or logit_count != record_count * len(LABEL_ORDER)
        ):
            raise ExplanationAuditIntegrityError("clean-logit equivalence is not exact")
        sealed_prediction_sha256 = _hash(
            evidence.get("sealed_prediction_sha256"),
            "clean sealed_prediction_sha256",
        )
        if expected_members_by_id:
            try:
                spec_member = expected_members_by_id[member_id]
            except KeyError as error:
                raise ExplanationAuditIntegrityError(
                    "clean equivalence contains an unknown member"
                ) from error
            prediction = _mapping(spec_member["prediction"], "spec member prediction")
            if (
                sealed_prediction_sha256 != prediction["artifact_sha256"]
                or record_count != prediction["record_count"]
            ):
                raise ExplanationAuditIntegrityError(
                    "clean equivalence differs from the sealed prediction binding"
                )
    if tuple(clean_ids) != expected_member_ids:
        raise ExplanationAuditIntegrityError("clean equivalence member grid differs")
    limitations = _mapping(payload["limitations"], "limitations")
    if dict(limitations) != _explanation_limitations():
        raise ExplanationAuditIntegrityError("explanation limitations differ")
    if verify_sources:
        if spec is None:
            raise ExplanationAuditIntegrityError(
                "source verification requires the frozen post-evaluation spec"
            )
        if expected_cohort is None:  # pragma: no cover - guarded by spec requirement
            raise ExplanationAuditIntegrityError("explanation cohort could not be resolved")
        allowed_root = source.parent.resolve()
        for raw_member in members:
            member = _mapping(raw_member, "member")
            if set(member) != {
                "member_id",
                "architecture",
                "seed",
                "methods",
                "cross_method",
            }:
                raise ExplanationAuditIntegrityError("member keys are not canonical")
            member_id = _string(member["member_id"], "member_id")
            architecture = _string(member["architecture"], "member architecture")
            seed = _integer(member["seed"], "member seed")
            if member_id != f"{architecture}-seed{seed}":
                raise ExplanationAuditIntegrityError("member identity is inconsistent")
            try:
                spec_member = expected_members_by_id[member_id]
            except KeyError as error:  # pragma: no cover - spec grid is validated upstream
                raise ExplanationAuditIntegrityError(
                    "manifest member is absent from the spec"
                ) from error
            if architecture != spec_member["architecture"] or seed != spec_member["seed"]:
                raise ExplanationAuditIntegrityError("manifest member differs from the spec")
            checkpoint = _mapping(spec_member["checkpoint"], "spec member checkpoint")
            decision = _mapping(
                spec_member["calibration_decision"], "spec member calibration decision"
            )
            prediction = _mapping(spec_member["prediction"], "spec member prediction")
            methods = _sequence(member["methods"], "member methods")
            observed_methods: list[str] = []
            method_artifacts: dict[str, AuditArrayArtifact] = {}
            for raw_method in methods:
                binding = _mapping(raw_method, "method binding")
                method = _string(binding["method"], "method")
                observed_methods.append(method)
                artifact = _verify_array_binding(
                    binding,
                    artifact_type=EXPLANATION_ARRAY_TYPE,
                    spec_sha256=expected_spec_sha256,
                    allowed_root=allowed_root,
                )
                if (
                    artifact.metadata.get("member_id") != member_id
                    or artifact.metadata.get("architecture") != architecture
                    or artifact.metadata.get("seed") != seed
                    or artifact.metadata.get("method") != method
                    or artifact.metadata.get("checkpoint_sha256") != checkpoint["file_sha256"]
                    or artifact.metadata.get("calibration_decision_sha256")
                    != decision["artifact_sha256"]
                    or artifact.metadata.get("sealed_prediction_sha256")
                    != prediction["artifact_sha256"]
                    or artifact.metadata.get("attribution_runtime") != runtime_blocks[member_id]
                ):
                    raise ExplanationAuditIntegrityError(
                        "method artifact member/method binding differs"
                    )
                _validate_method_artifact(
                    artifact,
                    method=method,
                    expected_cohort=expected_cohort,
                    expected_summary=binding["summary"],
                )
                method_artifacts[method] = artifact
            if tuple(observed_methods) != _method_names(architecture):
                raise ExplanationAuditIntegrityError("explanation method grid differs")
            cross_artifact = _verify_array_binding(
                _mapping(member["cross_method"], "cross-method binding"),
                artifact_type=EXPLANATION_CROSS_METHOD_TYPE,
                spec_sha256=expected_spec_sha256,
                allowed_root=allowed_root,
            )
            if cross_artifact.metadata.get("member_id") != member_id:
                raise ExplanationAuditIntegrityError("cross-method artifact member binding differs")
            if dict(cross_artifact.metadata) != {
                "post_evaluation_spec_sha256": expected_spec_sha256,
                "cohort_sha256": expected_cohort.payload_sha256,
                "member_id": member_id,
                "architecture": architecture,
                "seed": seed,
                "aggregation": "absolute_attribution_mean_across_leads_to_time",
                "metrics": ["cosine", "spearman"],
            }:
                raise ExplanationAuditIntegrityError(
                    "cross-method artifact metadata differs from the spec"
                )
            _validate_cross_method_artifact(
                cross_artifact,
                cohort=expected_cohort,
                methods=method_artifacts,
                expected_summary=_mapping(member["cross_method"], "cross-method binding")[
                    "summary"
                ],
            )
    return ExplanationManifest(
        path=source,
        artifact_sha256=stored_hash,
        payload=_finite_json(payload, "explanation manifest"),
    )


__all__ = [
    "EXPLANATION_ARRAY_TYPE",
    "EXPLANATION_CROSS_METHOD_TYPE",
    "EXPLANATION_MANIFEST_TYPE",
    "ExplanationAuditError",
    "ExplanationAuditIntegrityError",
    "ExplanationManifest",
    "load_explanation_manifest",
    "run_explanation_audit",
]
