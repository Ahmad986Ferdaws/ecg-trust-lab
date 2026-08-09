"""Immutable specification for read-only post-evaluation audits.

The fold-10 release is an input to this module, never an output.  A post-
evaluation specification verifies the completed six-member release, binds the
analysis revision, and freezes robustness, explanation, and demo choices before
any derived analysis is run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ecg_trust.constants import LEADS
from ecg_trust.final_evaluation_spec import load_final_evaluation_spec
from ecg_trust.protocol import FINAL_TEST_FOLDS, LABEL_ORDER, ExperimentProtocol
from ecg_trust.release_gates import (
    EXPECTED_ARCHITECTURES,
    EXPECTED_SEEDS,
    CalibrationBundle,
    CalibrationMember,
    RefitBundle,
    RefitMember,
    load_calibration_bundle,
    load_refit_bundle,
)

POST_EVALUATION_SPEC_SCHEMA_VERSION = 1
POST_EVALUATION_SUPERSESSION_SCHEMA_VERSION = 2
POST_EVALUATION_SPEC_TYPE = "ecg_trust.post_evaluation_audit_specification"
POST_EVALUATION_DIRECTORY = "post_evaluation"
POST_EVALUATION_FILENAME = "audit_spec.json"
EXPLANATION_COHORT_SIZE = 60
EXPLANATION_CELL_SIZE = 6
EXPLANATION_SELECTION_SEED = 20_260_808
ROBUSTNESS_RANDOM_SEED = 20_260_808
SUPERSESSION_REASON_DECIMAL_CASE_PATH_COLLISION = "decimal_case_id_suffix_collision"
SUPERSESSION_STATUS_ABORTED = "aborted_incomplete_no_final_manifest"
DEMO_MEMBER_ID = "resnet1d-seed2026"
DEMO_TARGET_COVERAGE = 0.8
EXPECTED_MEMBER_IDS: tuple[str, ...] = tuple(
    f"{architecture}-seed{seed}"
    for architecture in EXPECTED_ARCHITECTURES
    for seed in EXPECTED_SEEDS
)


class PostEvaluationError(ValueError):
    """Raised when a post-evaluation freeze request is not canonical."""


class PostEvaluationIntegrityError(PostEvaluationError):
    """Raised when a bound release input or stored specification is invalid."""


@dataclass(frozen=True, slots=True)
class PostEvaluationSpec:
    """Validated self-hashed post-evaluation audit specification."""

    path: Path | None
    artifact_sha256: str
    _canonical_payload: str

    @property
    def payload(self) -> dict[str, object]:
        decoded: object = json.loads(self._canonical_payload)
        if not isinstance(decoded, dict):  # pragma: no cover - constructor invariant
            raise PostEvaluationIntegrityError("stored specification is not an object")
        return cast(dict[str, object], decoded)

    @property
    def output_root(self) -> Path:
        contract = _mapping(self.payload["output_contract"], "output_contract")
        return Path(_string(contract["root"], "output_contract.root"))

    @property
    def member_ids(self) -> tuple[str, ...]:
        return tuple(
            _string(_mapping(item, "member")["member_id"], "member.member_id")
            for item in _sequence(self.payload["members"], "members")
        )

    def to_payload(self) -> dict[str, object]:
        return self.payload


def canonical_sha256(payload: Mapping[str, object]) -> str:
    """Return a prefixed SHA-256 over finite canonical JSON."""

    try:
        serialized = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise PostEvaluationError("specification must be finite JSON") from error
    return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise PostEvaluationIntegrityError(f"required source is missing: {source}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise PostEvaluationIntegrityError(
            f"could not hash required source {source}: {error}"
        ) from error
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path, context: str) -> Mapping[str, object]:
    if not path.is_file():
        raise PostEvaluationIntegrityError(f"{context} is missing: {path}")
    if path.stat().st_size > 100_000_000:
        raise PostEvaluationIntegrityError(f"{context} is unreasonably large")
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PostEvaluationIntegrityError(f"could not decode {context}: {error}") from error
    return _mapping(decoded, context)


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise PostEvaluationIntegrityError(f"{context} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _sequence(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise PostEvaluationIntegrityError(f"{context} must be a sequence")
    return cast(Sequence[object], value)


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PostEvaluationIntegrityError(f"{context} must be a non-empty string")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PostEvaluationIntegrityError(f"{context} must be an integer >= {minimum}")
    return value


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PostEvaluationIntegrityError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PostEvaluationIntegrityError(f"{context} must be finite")
    return result


def _hash(value: object, context: str) -> str:
    text = _string(value, context)
    digest = text.removeprefix("sha256:")
    if (
        not text.startswith("sha256:")
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise PostEvaluationIntegrityError(f"{context} must be a prefixed lower-case SHA-256")
    return text


def _normalized_hash(value: object, context: str) -> str:
    text = _string(value, context)
    digest = text.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise PostEvaluationIntegrityError(f"{context} must contain a lower-case SHA-256")
    return "sha256:" + digest


def _exact_keys(value: Mapping[str, object], expected: set[str], context: str) -> None:
    if set(value) != expected:
        missing = sorted(expected.difference(value))
        unknown = sorted(set(value).difference(expected))
        raise PostEvaluationIntegrityError(
            f"{context} keys are not canonical; missing={missing}, unknown={unknown}"
        )


def _verified_hashed_json(
    path: Path,
    *,
    context: str,
    hash_field: str,
    type_field: str,
    artifact_type: str,
) -> Mapping[str, object]:
    payload = _read_json(path, context)
    if payload.get(type_field) != artifact_type:
        raise PostEvaluationIntegrityError(f"{context} has an unexpected artifact type")
    stored = _hash(payload.get(hash_field), f"{context}.{hash_field}")
    unhashed = dict(payload)
    del unhashed[hash_field]
    if canonical_sha256(unhashed) != stored:
        raise PostEvaluationIntegrityError(f"{context} self-hash mismatch")
    return payload


def _file_binding(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "file_sha256": _file_sha256(path)}


def _artifact_binding(
    path: Path,
    payload: Mapping[str, object],
    *,
    hash_field: str,
) -> dict[str, object]:
    return {
        **_file_binding(path),
        "artifact_sha256": _hash(payload[hash_field], hash_field),
    }


def _robustness_protocol() -> dict[str, object]:
    cases: list[dict[str, object]] = [
        {"case_id": "clean", "corruption": "clean", "parameters": {}},
    ]
    cases.extend(
        {
            "case_id": f"baseline-wander-{amplitude:.2f}",
            "corruption": "baseline_wander",
            "parameters": {
                "amplitude_fraction": amplitude,
                "frequency_hz": 0.33,
                "phase_radians": 0.0,
            },
        }
        for amplitude in (0.05, 0.10, 0.20)
    )
    cases.extend(
        {
            "case_id": f"powerline-{amplitude:.2f}",
            "corruption": "powerline",
            "parameters": {
                "amplitude_fraction": amplitude,
                "frequency_hz": 50.0,
                "phase_radians": math.pi / 2.0,
            },
        }
        for amplitude in (0.01, 0.03, 0.05)
    )
    cases.extend(
        {
            "case_id": f"gaussian-noise-{snr_db}db",
            "corruption": "gaussian_noise",
            "parameters": {
                "snr_db": float(snr_db),
                "seed_strategy": "sha256_of_base_seed_and_ecg_id",
            },
        }
        for snr_db in (30, 20, 10)
    )
    cases.extend(
        {
            "case_id": f"amplitude-scale-{factor:.2f}",
            "corruption": "amplitude_scale",
            "parameters": {"factor": factor},
        }
        for factor in (0.75, 1.25, 0.50, 1.50)
    )
    cases.extend(
        {
            "case_id": f"dc-offset-{offset:+.2f}",
            "corruption": "dc_offset",
            "parameters": {"offset_fraction": offset},
        }
        for offset in (-0.05, 0.05, -0.15, 0.15)
    )
    cases.extend(
        {
            "case_id": f"time-shift-{samples:+d}",
            "corruption": "time_shift",
            "parameters": {"samples": samples, "padding": "zero"},
        }
        for samples in (-10, 10, -25, 25)
    )
    cases.extend(
        {
            "case_id": f"contiguous-mask-{width}",
            "corruption": "contiguous_mask",
            "parameters": {
                "width_samples": width,
                "lead_indices": list(range(len(LEADS))),
                "start_strategy": "sha256_modulo_valid_start_per_ecg",
                "seed": ROBUSTNESS_RANDOM_SEED,
            },
        }
        for width in (50, 100, 200)
    )
    cases.extend(
        {
            "case_id": f"lead-drop-{LEADS[index]}",
            "corruption": "lead_dropout",
            "parameters": {"lead_indices": [index], "lead_names": [LEADS[index]]},
        }
        for index in range(len(LEADS))
    )
    cases.extend(
        [
            {
                "case_id": "lead-drop-limb",
                "corruption": "lead_dropout",
                "parameters": {
                    "lead_indices": list(range(6)),
                    "lead_names": list(LEADS[:6]),
                },
            },
            {
                "case_id": "lead-drop-precordial",
                "corruption": "lead_dropout",
                "parameters": {
                    "lead_indices": list(range(6, 12)),
                    "lead_names": list(LEADS[6:]),
                },
            },
            {
                "case_id": "lead-permutation-swap-I-II",
                "corruption": "lead_permutation",
                "parameters": {"permutation": [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
            },
            {
                "case_id": "lead-permutation-reverse-all",
                "corruption": "lead_permutation",
                "parameters": {"permutation": list(reversed(range(len(LEADS))))},
            },
        ]
    )
    if len(cases) != 41:  # pragma: no cover - static construction invariant
        raise PostEvaluationIntegrityError("robustness matrix must contain 41 cases")
    return {
        "sampling_frequency_hz": 100.0,
        "random_seed": ROBUSTNESS_RANDOM_SEED,
        "corruption_domain": ("physical_millivolts_before_each_member_frozen_normalization"),
        "execution": {
            "physical_corruption_precision": "cpu_float32",
            "model_inference_precision": "bf16_as_frozen_in_final_batch",
            "metric_precision": "cpu_float64",
            "inference_batching": "reuse_frozen_final_evaluation_settings",
            "stochastic_transform_randomness": "stateless_per_ecg_sha256",
        },
        "transform_definitions": {
            "record_scale": "whole_record_all_12_lead_rms_per_ecg_in_physical_mv",
            "baseline_wander": "shared_across_leads_sinusoid_scaled_by_record_rms",
            "powerline": ("shared_across_leads_sinusoid_scaled_by_record_rms;50hz_nyquist_allowed"),
            "gaussian_noise": ("standard_normal_draw_rescaled_per_ecg_to_exact_realized_rms_snr"),
            "amplitude_scale": "multiply_all_leads_and_samples_by_factor",
            "dc_offset": "shared_constant_scaled_by_signed_fraction_of_record_rms",
            "time_shift": "zero_padded_non_circular_shift",
            "contiguous_mask": (
                "zero_half_open_interval_on_all_listed_leads;stateless_valid_start_per_ecg"
            ),
            "lead_dropout": "zero_all_samples_of_listed_leads",
            "lead_permutation": "explicit_full_canonical_lead_axis_permutation",
        },
        "clean_baseline_gate": {
            "required_before_corruptions": True,
            "comparison": "np.array_equal_against_sealed_raw_logits",
            "maximum_absolute_logit_error": 0.0,
            "failure_policy": "reject_all_corruption_results",
        },
        "calibration_and_decision_policy": (
            "reuse_each_member_frozen_temperature_thresholds_and_entropy_gates"
        ),
        "patient_resampling": {
            "method": "patient_cluster_percentile_bootstrap",
            "resamples": 1000,
            "confidence": 0.95,
            "base_seed": 20_260_908,
            "pairing": "corrupted_minus_clean_within_member_and_patient",
            "cluster_unit": "patient_id",
            "cluster_sampling": (
                "sample_n_unique_patients_with_replacement_then_include_all_cluster_records"
            ),
            "record_weighting": "resampled_patient_multiplicity_with_all_cluster_records",
            "confidence_interval": "percentile",
        },
        "dense_risk_coverage": {
            "ordering": "ascending_mean_normalized_binary_entropy_stable_index_tiebreak",
            "coverage_prefixes": "one_through_all_records",
            "hamming_risk": "mean_label_error_per_record_then_cumulative_prefix_mean",
            "log_loss_risk": "mean_label_log_loss_per_record_then_cumulative_prefix_mean",
            "area_method": "arithmetic_mean_over_all_prefix_coverages",
            "oracle_reference": "ascending_per_record_loss_stable_index_tiebreak",
            "random_reference": "analytical_constant_full_coverage_risk",
        },
        "metrics": [
            "macro_roc_auc",
            "macro_average_precision",
            "macro_brier_score",
            "macro_ece",
            "dense_entropy_ranked_risk_coverage_curve",
            "area_under_risk_coverage_curve",
            "uncertainty_drift",
            "raw_logit_drift",
            "hamming_risk_by_frozen_gate",
            "exact_match_accuracy_by_frozen_gate",
            "coverage_by_frozen_gate",
            "corrupted_minus_clean_metric_deltas",
        ],
        "severity_matrix": cases,
        "retuning_allowed": False,
    }


def _explanation_settings() -> dict[str, object]:
    return {
        "target_labels": list(LABEL_ORDER),
        "target_assignment": "one_preregistered_label_status_cell_per_selected_ecg",
        "target_score": {
            "positive_cell": "+1_times_target_label_logit",
            "negative_cell": "-1_times_target_label_logit",
            "probability": "sigmoid(signed_correct_status_logit_over_frozen_temperature)",
            "attribution_orientation": "multiply_target_label_map_by_cell_sign",
        },
        "baseline": "all_zero_normalized_input",
        "execution": {
            "numeric_precision": "float32",
            "sealed_clean_equivalence_precision": "bf16_as_frozen_in_final_batch",
            "outer_attribution_batch_size": 4,
            "faithfulness_scoring_batch_size": 60,
            "torch_deterministic_algorithms": True,
            "identical_rerun_requirement": "torch_equal",
            "tf32_allowed": False,
            "fp32_vs_sealed_cohort_logit_drift_required": True,
        },
        "methods_by_architecture": {
            "resnet1d": ["grad_cam_1d", "integrated_gradients", "temporal_occlusion"],
            "ecg_transformer": ["integrated_gradients", "temporal_occlusion"],
        },
        "grad_cam_1d": {
            "feature_map": "resnet_final_temporal_feature_map_before_global_pool",
            "channel_weights": "mean_target_gradient_over_time",
            "signed": True,
            "relu_applied": False,
            "upsampling": "linear_to_1000_samples_align_corners_false",
            "normalization": "per_ecg_unit_maximum_absolute_value",
        },
        "integrated_gradients": {
            "n_steps": 32,
            "internal_batch_size": 8,
            "multiply_by_inputs": True,
            "integration_method": "gausslegendre",
            "normalize": True,
            "completeness_delta_basis": "pre_normalization_signed_raw_ig",
        },
        "temporal_occlusion": {
            "window_samples": 50,
            "stride_samples": 25,
            "perturbations_per_eval": 16,
            "normalize": True,
            "mask_unit": "one_temporal_window_across_all_12_leads",
            "baseline": "zero_normalized_input",
            "returned_lead_axis": "duplicated_not_lead_resolved",
        },
        "faithfulness": {
            "temporal_importance": (
                "mean_absolute_attribution_across_attribution_channels_per_sample"
            ),
            "temporal_perturbation_unit": ("one_ranked_sample_index_across_all_12_input_leads"),
            "operation_rankings": {
                "deletion": ["most_important", "least_important", "random"],
                "insertion": ["most_important"],
            },
            "temporal_deletion_fractions": [
                0.0,
                0.05,
                0.1,
                0.2,
                0.4,
                0.6,
                0.8,
                1.0,
            ],
            "random_ranking_replicates": 20,
            "random_ranking_seed": 20_261_008,
            "lead_ablation": {
                "applicable_to_lead_specific_maps_only": True,
                "applicable_methods": ["integrated_gradients"],
                "rankings": ["most_important", "least_important", "random"],
                "prefixes": "zero_through_all_12_leads",
                "lead_importance": "mean_absolute_attribution_over_time_per_lead",
            },
            "area_under_curve": (
                "trapezoidal_integral_of_calibrated_correct_status_probability_over_fraction"
            ),
            "guided_vs_random_advantage": "mean_random_auc_minus_guided_auc",
            "parameter_randomization_seeds": [2_026_801, 2_026_802, 2_026_803],
            "parameter_randomization_scope": "full_model_reset_parameters_copy",
            "parameter_randomization_similarity": (
                "signed_cosine_of_flattened_attribution_tensors"
            ),
            "stability_noise_snr_db": 40.0,
            "stability_replicates": 3,
            "stability_seed_strategy": "sha256_of_replicate_and_ecg_id",
            "stability_similarity": "signed_cosine_of_flattened_attribution_tensors",
            "stability_noise": {
                "domain": "after_frozen_normalization",
                "distribution": "iid_zero_mean_gaussian",
                "scale": "whole_record_all_lead_rms_per_ecg",
                "snr_definition": (
                    "nominal_expected_power_snr_from_gaussian_sigma;realized_draw_not_renormalized"
                ),
            },
            "integrated_gradients_completeness_delta_required": True,
            "cross_method_agreement": {
                "aggregation": "absolute_attribution_mean_across_leads_to_time",
                "metrics": ["cosine", "spearman"],
            },
        },
        "selection_data_policy": {
            "targets": "allowed_only_for_preregistered_label_status_stratification",
            "predictions": "forbidden",
            "metrics": "forbidden",
        },
        "retuning_or_example_replacement_allowed": False,
    }


def _output_contract(
    project_root: Path,
    comparison_id: str,
    *,
    audit_revision: int = 1,
) -> dict[str, object]:
    allowed_parent = (project_root / "runs" / POST_EVALUATION_DIRECTORY).resolve()
    if isinstance(audit_revision, bool) or not isinstance(audit_revision, int):
        raise PostEvaluationError("audit revision must be an integer")
    if audit_revision == 1:
        directory_name = comparison_id
    elif audit_revision >= 2:
        directory_name = f"{comparison_id}__audit-r{audit_revision}"
    else:
        raise PostEvaluationError("audit revision must be 1 or >= 2")
    root = (allowed_parent / directory_name).resolve()
    return {
        "root": str(root),
        "allowed_parent": str(allowed_parent),
        "policy": {
            "release_inputs_read_only": True,
            "overwrite_allowed": False,
            "all_derived_artifacts_must_remain_under_root": True,
        },
        "artifacts": {
            "audit_spec": str(root / POST_EVALUATION_FILENAME),
            "derived_manifest": str(root / "derived_artifacts.manifest.json"),
            "final_results_markdown": str(root / "reports" / "FINAL_RESULTS.md"),
            "probability_audit": str(root / "probability_audit.json"),
            "robustness_manifest": str(root / "robustness" / "manifest.json"),
            "explanations_manifest": str(root / "explanations" / "manifest.json"),
            "demo_directory": str(root / "demo"),
            "demo_policy": str(root / "demo" / "resnet1d-seed2026.coverage80.demo-policy.json"),
            "demo_examples": str(root / "demo" / "fold8-label-free.examples.json"),
            "demo_binding": str(root / "demo" / "resnet1d-seed2026.coverage80.demo-binding.json"),
            "publication_tables_directory": str(root / "publication" / "tables"),
            "publication_figures_directory": str(root / "publication" / "figures"),
        },
    }


def _audit_revision_from_output_root(
    project_root: Path,
    comparison_id: str,
    output_root: Path,
) -> int:
    allowed_parent = (project_root / "runs" / POST_EVALUATION_DIRECTORY).resolve()
    root = output_root.resolve()
    if root.parent != allowed_parent:
        raise PostEvaluationIntegrityError("output root escapes runs/post_evaluation")
    if root.name == comparison_id:
        return 1
    prefix = f"{comparison_id}__audit-r"
    suffix = root.name.removeprefix(prefix)
    if (
        not root.name.startswith(prefix)
        or not suffix.isascii()
        or not suffix.isdigit()
        or suffix.startswith("0")
    ):
        raise PostEvaluationIntegrityError("versioned output root name is not canonical")
    revision = int(suffix)
    if revision < 2 or suffix != str(revision):
        raise PostEvaluationIntegrityError("versioned output root revision must be >= 2")
    return revision


def _run_git(project_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={project_root}", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise PostEvaluationError(f"Git state is unavailable: {error}") from error
    return result.stdout.strip()


def _capture_clean_git(project_root: Path) -> dict[str, object]:
    if not project_root.is_dir():
        raise PostEvaluationError(f"project root is missing: {project_root}")
    top_level = Path(_run_git(project_root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != project_root.resolve():
        raise PostEvaluationError("project root must be the Git worktree top level")
    status = _run_git(project_root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise PostEvaluationError("post-evaluation freeze requires a clean committed Git worktree")
    revision = _run_git(project_root, "rev-parse", "HEAD").casefold()
    if len(revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise PostEvaluationError("Git revision is not a full hexadecimal object ID")
    return {"project_root": str(project_root.resolve()), "git_revision": revision, "clean": True}


def _bundle_binding(
    path: Path,
    bundle: RefitBundle | CalibrationBundle,
) -> dict[str, object]:
    if bundle.artifact_sha256 is None:
        raise PostEvaluationIntegrityError("release bundle is not self-hashed")
    return {
        **_file_binding(path),
        "artifact_sha256": _hash(bundle.artifact_sha256, "bundle artifact_sha256"),
        "protocol_hash": _hash(bundle.protocol_hash, "bundle protocol_hash"),
        "manifest_sha256": _hash(bundle.manifest_sha256, "bundle manifest_sha256"),
        "normalization_sha256": _hash(bundle.normalization_sha256, "bundle normalization_sha256"),
        "member_count": len(bundle.members),
    }


def _infer_opening_ledger_path(final_spec_path: Path, final_spec_sha256: str) -> Path:
    release_root = final_spec_path.resolve().parent.parent
    return (
        release_root
        / ".final-test-openings"
        / (final_spec_sha256.removeprefix("sha256:") + ".opening-ledger.json")
    ).resolve()


def _opening_ledger_binding(
    path: Path,
    *,
    protocol: ExperimentProtocol,
    final_spec_binding: Mapping[str, object],
    refit_binding: Mapping[str, object],
    calibration_binding: Mapping[str, object],
) -> tuple[dict[str, object], Mapping[str, object]]:
    ledger = _verified_hashed_json(
        path,
        context="completed final-test opening ledger",
        hash_field="ledger_sha256",
        type_field="artifact_type",
        artifact_type="ecg_trust.final_test_opening_ledger",
    )
    if ledger.get("state") != "complete":
        raise PostEvaluationIntegrityError("final-test opening ledger is not complete")
    plan = _mapping(ledger.get("plan"), "opening ledger plan")
    if plan.get("protocol_hash") != protocol.protocol_hash:
        raise PostEvaluationIntegrityError("opening ledger protocol differs")
    if plan.get("refit_bundle_sha256") != refit_binding["artifact_sha256"]:
        raise PostEvaluationIntegrityError("opening ledger refit bundle differs")
    if plan.get("calibration_bundle_sha256") != calibration_binding["artifact_sha256"]:
        raise PostEvaluationIntegrityError("opening ledger calibration bundle differs")
    if dict(_mapping(plan.get("final_evaluation_spec"), "ledger final spec")) != dict(
        final_spec_binding
    ):
        raise PostEvaluationIntegrityError("opening ledger final specification differs")
    members = _mapping(ledger.get("members"), "opening ledger members")
    if set(members) != set(EXPECTED_MEMBER_IDS):
        raise PostEvaluationIntegrityError("opening ledger member grid is not exact-six")
    if any(
        _mapping(members[member_id], f"ledger member {member_id}").get("state") != "report_saved"
        for member_id in EXPECTED_MEMBER_IDS
    ):
        raise PostEvaluationIntegrityError("opening ledger contains an incomplete member")
    events = _sequence(ledger.get("events"), "opening ledger events")
    if not events or _mapping(events[-1], "last opening event").get("event") != (
        "exact_six_member_final_batch_complete"
    ):
        raise PostEvaluationIntegrityError("opening ledger has no terminal completion event")
    opening = _mapping(ledger.get("opening"), "opening ledger opening")
    purpose = _string(opening.get("purpose"), "opening purpose")
    operator = _string(opening.get("operator"), "opening operator")
    batch_sha256 = _hash(plan.get("batch_sha256"), "opening ledger batch_sha256")
    marker_path = Path(_string(plan.get("opening_marker_path"), "opening marker path")).resolve()
    marker = _verified_hashed_json(
        marker_path,
        context="canonical final-test opening marker",
        hash_field="marker_sha256",
        type_field="artifact_type",
        artifact_type="ecg_trust.final_test_canonical_opening_marker",
    )
    if (
        marker.get("batch_sha256") != batch_sha256
        or marker.get("refit_bundle_sha256") != refit_binding["artifact_sha256"]
        or marker.get("calibration_bundle_sha256") != calibration_binding["artifact_sha256"]
        or Path(_string(marker.get("ledger_path"), "marker ledger path")).resolve()
        != path.resolve()
        or marker.get("marker_precedes_fold10_access") is not True
    ):
        raise PostEvaluationIntegrityError("canonical opening marker differs")
    return (
        {
            **_file_binding(path),
            "ledger_sha256": _hash(ledger["ledger_sha256"], "ledger_sha256"),
            "state": "complete",
            "purpose": purpose,
            "operator": operator,
            "batch_sha256": batch_sha256,
            "opening_marker_path": str(marker_path),
            "opening_marker_file_sha256": _file_sha256(marker_path),
            "opening_marker_sha256": _hash(marker["marker_sha256"], "opening marker_sha256"),
            "terminal_event": "exact_six_member_final_batch_complete",
        },
        ledger,
    )


def _final_spec_binding(
    path: Path, *, protocol: ExperimentProtocol
) -> tuple[dict[str, object], Mapping[str, object]]:
    spec = load_final_evaluation_spec(
        path,
        protocol=protocol,
        verify_sources=True,
        # The sealed evaluation intentionally binds its earlier execution revision.
        verify_runtime=False,
    )
    payload = spec.payload
    runtime = _mapping(payload["runtime_envelope"], "final spec runtime_envelope")
    git = _mapping(runtime["git"], "final spec runtime git")
    binding = {
        **_file_binding(path),
        "artifact_sha256": spec.artifact_sha256,
        "evaluation_git_revision": _string(git["revision"], "final evaluation Git revision"),
    }
    return binding, payload


def _protocol_deviation_binding(
    path: Path,
    *,
    final_spec_payload: Mapping[str, object],
) -> dict[str, object]:
    frozen = _mapping(final_spec_payload["protocol_deviations"], "frozen protocol deviations")
    binding = {
        **_file_binding(path),
        "required_in_all_reporting": True,
    }
    if Path(_string(frozen["path"], "frozen deviations path")).resolve() != path:
        raise PostEvaluationIntegrityError("protocol deviation path differs from frozen spec")
    if frozen["file_sha256"] != binding["file_sha256"]:
        raise PostEvaluationIntegrityError("protocol deviations changed after final freeze")
    if frozen.get("required_in_final_reporting") is not True:
        raise PostEvaluationIntegrityError("frozen deviations are not required in reporting")
    return binding


def _validate_release_bundles(
    refit_path: Path,
    calibration_path: Path,
    *,
    protocol: ExperimentProtocol,
) -> tuple[RefitBundle, CalibrationBundle, dict[str, object], dict[str, object]]:
    try:
        refit = load_refit_bundle(refit_path, protocol=protocol, verify_sources=True)
        # Full calibration source verification intentionally replays the earlier
        # final-spec runtime gate and therefore rejects every legitimate later
        # analysis revision.  Load its strict self-hashed schema here, then
        # independently verify every bound decision/checkpoint/config below.
        calibration = load_calibration_bundle(
            calibration_path, protocol=protocol, verify_sources=False
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise PostEvaluationIntegrityError(
            f"could not verify sealed release bundles: {error}"
        ) from error
    refit_binding = _bundle_binding(refit_path, refit)
    calibration_binding = _bundle_binding(calibration_path, calibration)
    if len(refit.members) != 6 or len(calibration.members) != 6:
        raise PostEvaluationIntegrityError("release bundles must contain exactly six members")
    if {member.member_id for member in refit.members} != set(EXPECTED_MEMBER_IDS) or {
        member.member_id for member in calibration.members
    } != set(EXPECTED_MEMBER_IDS):
        raise PostEvaluationIntegrityError("release bundle member grid is not canonical")
    if (
        calibration.refit_bundle_sha256 != refit.artifact_sha256
        or calibration.protocol_hash != refit.protocol_hash
        or calibration.manifest_sha256 != refit.manifest_sha256
        or calibration.normalization_sha256 != refit.normalization_sha256
        or calibration.label_order != refit.label_order
        or refit.label_order != LABEL_ORDER
    ):
        raise PostEvaluationIntegrityError("refit and calibration bundles disagree")
    return refit, calibration, refit_binding, calibration_binding


def _identity_arrays(
    path: Path,
    *,
    expected_count: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int8]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            raw_ecg_id = np.asarray(archive["ecg_id"])
            raw_patient_id = np.asarray(archive["patient_id"])
            raw_strat_fold = np.asarray(archive["strat_fold"])
            raw_targets = np.asarray(archive["targets"])
    except (OSError, ValueError, KeyError) as error:
        raise PostEvaluationIntegrityError(
            f"could not read prediction identity arrays {path}: {error}"
        ) from error
    if (
        any(
            not np.issubdtype(values.dtype, np.integer)
            for values in (raw_ecg_id, raw_patient_id, raw_strat_fold, raw_targets)
        )
        or not np.logical_or(raw_targets == 0, raw_targets == 1).all()
    ):
        raise PostEvaluationIntegrityError(
            "prediction identity arrays must be integers and targets must be binary"
        )
    ecg_id = raw_ecg_id.astype(np.int64, copy=False)
    patient_id = raw_patient_id.astype(np.int64, copy=False)
    strat_fold = raw_strat_fold.astype(np.int64, copy=False)
    targets = raw_targets.astype(np.int8, copy=False)
    if (
        ecg_id.shape != (expected_count,)
        or patient_id.shape != (expected_count,)
        or strat_fold.shape != (expected_count,)
        or targets.shape != (expected_count, len(LABEL_ORDER))
        or len(np.unique(ecg_id)) != expected_count
        or np.any(ecg_id <= 0)
        or np.any(patient_id <= 0)
        or not np.all(strat_fold == FINAL_TEST_FOLDS[0])
        or not np.logical_or(targets == 0, targets == 1).all()
    ):
        raise PostEvaluationIntegrityError("prediction identity arrays are not fold-10 canonical")
    return ecg_id, patient_id, targets


def _member_report_hashes(summary: Mapping[str, object]) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in _sequence(summary.get("member_reports"), "summary member_reports"):
        entry = _mapping(item, "summary member report")
        _exact_keys(entry, {"member_id", "report_sha256"}, "summary member report")
        member_id = _string(entry["member_id"], "summary member_id")
        if member_id in values:
            raise PostEvaluationIntegrityError("summary repeats a member report")
        values[member_id] = _hash(entry["report_sha256"], "summary report_sha256")
    if set(values) != set(EXPECTED_MEMBER_IDS):
        raise PostEvaluationIntegrityError("summary does not contain exact-six reports")
    return values


def _build_member_bindings(
    *,
    final_root: Path,
    summary: Mapping[str, object],
    ledger: Mapping[str, object],
    refit_bundle: RefitBundle,
    calibration_bundle: CalibrationBundle,
    protocol: ExperimentProtocol,
    final_spec_binding: Mapping[str, object],
    deviation_binding: Mapping[str, object],
) -> tuple[
    list[dict[str, object]],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int8],
    str,
]:
    summary_hashes = _member_report_hashes(summary)
    ledger_members = _mapping(ledger["members"], "opening ledger members")
    refits = {member.member_id: member for member in refit_bundle.members}
    calibrations = {member.member_id: member for member in calibration_bundle.members}
    members: list[dict[str, object]] = []
    reference_ecg: NDArray[np.int64] | None = None
    reference_patient: NDArray[np.int64] | None = None
    reference_targets: NDArray[np.int8] | None = None
    reference_alignment: str | None = None
    expected_npz_names = {f"{member_id}.fold10.npz" for member_id in EXPECTED_MEMBER_IDS}
    expected_sidecar_names = {f"{member_id}.fold10.json" for member_id in EXPECTED_MEMBER_IDS}
    expected_report_names = {f"{member_id}.final-report.json" for member_id in EXPECTED_MEMBER_IDS}
    if (
        {path.name for path in final_root.glob("*.fold10.npz")} != expected_npz_names
        or {path.name for path in final_root.glob("*.fold10.json")} != expected_sidecar_names
        or {path.name for path in final_root.glob("*.final-report.json")} != expected_report_names
    ):
        raise PostEvaluationIntegrityError(
            "final output directory does not contain the exact six prediction/report pairs"
        )

    for member_id in EXPECTED_MEMBER_IDS:
        refit: RefitMember = refits[member_id]
        calibration: CalibrationMember = calibrations[member_id]
        if (
            refit.architecture != calibration.architecture
            or refit.seed != calibration.seed
            or refit.run_name != calibration.model_name
            or refit.lineage_sha256 != calibration.refit_lineage_sha256
            or refit.final_checkpoint_path.resolve() != calibration.checkpoint_path.resolve()
            or refit.final_checkpoint_sha256 != calibration.checkpoint_sha256
            or refit.resolved_config_path.resolve() != calibration.resolved_config_path.resolve()
            or refit.resolved_config_hash != calibration.resolved_config_hash
        ):
            raise PostEvaluationIntegrityError(f"release lineage differs for {member_id}")

        prediction_path = (final_root / f"{member_id}.fold10.npz").resolve()
        sidecar_path = prediction_path.with_suffix(".json")
        report_path = (final_root / f"{member_id}.final-report.json").resolve()
        sidecar = _verified_hashed_json(
            sidecar_path,
            context=f"{member_id} fold-10 prediction sidecar",
            hash_field="artifact_sha256",
            type_field="artifact_type",
            artifact_type="ecg_trust.multilabel_predictions",
        )
        report = _verified_hashed_json(
            report_path,
            context=f"{member_id} final report",
            hash_field="report_sha256",
            type_field="report_type",
            artifact_type="ecg_trust.final_evaluation_report",
        )
        record_count = _integer(sidecar.get("record_count"), "prediction record_count", minimum=1)
        alignment = _hash(sidecar.get("alignment_sha256"), "prediction alignment_sha256")
        model = _mapping(sidecar.get("model"), "prediction model")
        if (
            sidecar.get("fold_role") != "final_test"
            or list(_sequence(sidecar.get("folds"), "prediction folds")) != [10]
            or list(_sequence(sidecar.get("label_order"), "prediction label_order"))
            != list(LABEL_ORDER)
            or model.get("name") != refit.run_name
            or model.get("seed") != refit.seed
            or sidecar.get("protocol_hash") != protocol.protocol_hash
            or sidecar.get("config_hash") != refit.resolved_config_hash
            or sidecar.get("manifest_hash") != refit.manifest_sha256
            or sidecar.get("npz_file") != prediction_path.name
            or sidecar.get("npz_size_bytes") != prediction_path.stat().st_size
            or _normalized_hash(sidecar.get("npz_sha256"), "prediction npz_sha256")
            != _file_sha256(prediction_path)
        ):
            raise PostEvaluationIntegrityError(f"fold-10 prediction differs for {member_id}")
        report_model = _mapping(report.get("model"), "final report model")
        sources = _mapping(report.get("sources"), "final report sources")
        report_spec = _mapping(report.get("final_evaluation_spec"), "report final evaluation spec")
        report_deviation = _mapping(report.get("protocol_deviations"), "report protocol deviations")
        if (
            report["report_sha256"] != summary_hashes[member_id]
            or report_model.get("name") != refit.run_name
            or report_model.get("seed") != refit.seed
            or report.get("protocol_hash") != protocol.protocol_hash
            or report.get("config_hash") != refit.resolved_config_hash
            or sources.get("final_prediction_sha256") != sidecar["artifact_sha256"]
            or sources.get("final_alignment_sha256") != alignment
            or sources.get("calibration_artifact_sha256") != calibration.decision_artifact_sha256
            or dict(report_spec) != dict(final_spec_binding)
            or Path(_string(report_deviation.get("path"), "report deviations path")).resolve()
            != Path(_string(deviation_binding["path"], "deviation path"))
            or report_deviation.get("file_sha256") != deviation_binding["file_sha256"]
        ):
            raise PostEvaluationIntegrityError(f"final report differs for {member_id}")

        ledger_member = _mapping(ledger_members[member_id], f"ledger member {member_id}")
        if (
            Path(
                _string(
                    ledger_member.get("final_prediction_path"),
                    "ledger final prediction path",
                )
            ).resolve()
            != prediction_path
            or ledger_member.get("final_prediction_artifact_sha256") != sidecar["artifact_sha256"]
            or _normalized_hash(
                ledger_member.get("final_prediction_file_sha256"),
                "ledger prediction file hash",
            )
            != _file_sha256(prediction_path)
            or _normalized_hash(
                ledger_member.get("final_prediction_sidecar_sha256"),
                "ledger prediction sidecar hash",
            )
            != _file_sha256(sidecar_path)
            or Path(_string(ledger_member.get("final_report_path"), "ledger report path")).resolve()
            != report_path
            or ledger_member.get("final_report_sha256") != report["report_sha256"]
        ):
            raise PostEvaluationIntegrityError(f"opening ledger differs for {member_id}")

        decision = _verified_hashed_json(
            calibration.decision_path,
            context=f"{member_id} calibration decisions",
            hash_field="artifact_sha256",
            type_field="artifact_type",
            artifact_type="ecg_trust.calibration_decisions",
        )
        if (
            decision["artifact_sha256"] != calibration.decision_artifact_sha256
            or _file_sha256(calibration.decision_path)
            != _normalized_hash(calibration.decision_file_sha256, "calibration decision file hash")
            or decision.get("config_hash") != refit.resolved_config_hash
            or decision.get("protocol_hash") != protocol.protocol_hash
        ):
            raise PostEvaluationIntegrityError(f"calibration decision differs for {member_id}")

        ecg_id, patient_id, targets = _identity_arrays(prediction_path, expected_count=record_count)
        if reference_ecg is None:
            reference_ecg = ecg_id
            reference_patient = patient_id
            reference_targets = targets
            reference_alignment = alignment
        else:
            if reference_patient is None or reference_targets is None:
                raise PostEvaluationIntegrityError(
                    "fold-10 prediction alignment state is incomplete"
                )
            if (
                alignment != reference_alignment
                or not np.array_equal(ecg_id, reference_ecg)
                or not np.array_equal(patient_id, reference_patient)
                or not np.array_equal(targets, reference_targets)
            ):
                raise PostEvaluationIntegrityError("fold-10 prediction identities are not aligned")

        members.append(
            {
                "member_id": member_id,
                "architecture": refit.architecture,
                "seed": refit.seed,
                "model_name": refit.run_name,
                "prediction": {
                    "npz_path": str(prediction_path),
                    "npz_file_sha256": _file_sha256(prediction_path),
                    "sidecar_path": str(sidecar_path),
                    "sidecar_file_sha256": _file_sha256(sidecar_path),
                    "artifact_sha256": sidecar["artifact_sha256"],
                    "alignment_sha256": alignment,
                    "record_count": record_count,
                },
                "final_report": _artifact_binding(report_path, report, hash_field="report_sha256"),
                "checkpoint": {
                    "path": str(refit.final_checkpoint_path.resolve()),
                    "file_sha256": _file_sha256(refit.final_checkpoint_path),
                },
                "resolved_config": {
                    "path": str(refit.resolved_config_path.resolve()),
                    "file_sha256": _file_sha256(refit.resolved_config_path),
                    "config_hash": refit.resolved_config_hash,
                },
                "calibration_decision": _artifact_binding(
                    calibration.decision_path,
                    decision,
                    hash_field="artifact_sha256",
                ),
                "refit_lineage_sha256": refit.lineage_sha256,
            }
        )

    if (
        reference_ecg is None
        or reference_patient is None
        or reference_targets is None
        or reference_alignment is None
    ):
        raise PostEvaluationIntegrityError("no aligned final predictions were verified")
    return (
        members,
        reference_ecg,
        reference_patient,
        reference_targets,
        reference_alignment,
    )


def _summary_binding(
    path: Path,
    *,
    final_spec_binding: Mapping[str, object],
    deviation_binding: Mapping[str, object],
) -> tuple[dict[str, object], Mapping[str, object]]:
    summary = _verified_hashed_json(
        path,
        context="completed final batch summary",
        hash_field="artifact_sha256",
        type_field="artifact_type",
        artifact_type="ecg_trust.final_batch_summary",
    )
    if summary.get("retuning_performed") is not False:
        raise PostEvaluationIntegrityError("final batch summary records retuning")
    preregistration = _mapping(summary.get("preregistration"), "summary preregistration")
    if dict(_mapping(preregistration.get("final_evaluation_spec"), "summary final spec")) != {
        key: final_spec_binding[key] for key in ("path", "file_sha256", "artifact_sha256")
    }:
        raise PostEvaluationIntegrityError("final summary specification binding differs")
    summary_deviation = _mapping(preregistration.get("protocol_deviations"), "summary deviations")
    if (
        Path(_string(summary_deviation.get("path"), "summary deviation path")).resolve()
        != Path(_string(deviation_binding["path"], "deviation path"))
        or summary_deviation.get("file_sha256") != deviation_binding["file_sha256"]
        or summary_deviation.get("required_in_final_reporting") is not True
    ):
        raise PostEvaluationIntegrityError("final summary deviations binding differs")
    batch_sha256 = _hash(summary.get("batch_sha256"), "summary batch_sha256")
    return (
        {
            **_artifact_binding(path, summary, hash_field="artifact_sha256"),
            "batch_sha256": batch_sha256,
        },
        summary,
    )


def _build_aggregate_bindings(
    *,
    summary: Mapping[str, object],
    ledger: Mapping[str, object],
    batch_sha256: str,
) -> dict[str, object]:
    ledger_outputs = _mapping(ledger.get("outputs"), "opening ledger outputs")
    architecture_entries = _mapping(
        summary.get("architecture_reports"), "summary architecture reports"
    )
    if set(architecture_entries) != set(EXPECTED_ARCHITECTURES):
        raise PostEvaluationIntegrityError("summary architecture grid is not canonical")
    architecture_summaries: list[dict[str, object]] = []
    for architecture in EXPECTED_ARCHITECTURES:
        summary_entry = _mapping(
            architecture_entries[architecture], f"{architecture} summary entry"
        )
        path = Path(_string(summary_entry.get("path"), "architecture path")).resolve()
        report = _verified_hashed_json(
            path,
            context=f"{architecture} architecture summary",
            hash_field="artifact_sha256",
            type_field="artifact_type",
            artifact_type="ecg_trust.final_architecture_aggregate",
        )
        if (
            report.get("architecture") != architecture
            or report.get("batch_sha256") != batch_sha256
            or report.get("artifact_sha256") != summary_entry.get("artifact_sha256")
            or Path(
                _string(
                    ledger_outputs.get(f"architecture_{architecture}_path"),
                    "ledger architecture path",
                )
            ).resolve()
            != path
            or ledger_outputs.get(f"architecture_{architecture}_sha256")
            != report.get("artifact_sha256")
        ):
            raise PostEvaluationIntegrityError(f"{architecture} architecture summary differs")
        architecture_summaries.append(
            {
                "architecture": architecture,
                **_artifact_binding(path, report, hash_field="artifact_sha256"),
            }
        )

    manifest_entry = _mapping(summary.get("paired_bootstrap_manifest"), "summary paired manifest")
    manifest_path = Path(_string(manifest_entry.get("path"), "paired manifest path")).resolve()
    manifest = _verified_hashed_json(
        manifest_path,
        context="paired patient-bootstrap manifest",
        hash_field="artifact_sha256",
        type_field="artifact_type",
        artifact_type="ecg_trust.paired_patient_bootstrap_manifest",
    )
    if (
        manifest.get("batch_sha256") != batch_sha256
        or manifest.get("artifact_sha256") != manifest_entry.get("artifact_sha256")
        or Path(_string(ledger_outputs.get("paired_manifest_path"), "ledger paired path")).resolve()
        != manifest_path
        or ledger_outputs.get("paired_manifest_sha256") != manifest.get("artifact_sha256")
    ):
        raise PostEvaluationIntegrityError("paired bootstrap manifest differs")
    entries = _sequence(manifest.get("entries"), "paired manifest entries")
    if len(entries) != len(EXPECTED_SEEDS):
        raise PostEvaluationIntegrityError("paired manifest must contain three seeds")
    by_seed: dict[int, Mapping[str, object]] = {}
    for raw_entry in entries:
        entry = _mapping(raw_entry, "paired manifest entry")
        seed = _integer(entry.get("seed"), "paired seed")
        if seed in by_seed:
            raise PostEvaluationIntegrityError("paired manifest repeats a seed")
        by_seed[seed] = entry
    if set(by_seed) != set(EXPECTED_SEEDS):
        raise PostEvaluationIntegrityError("paired manifest seed grid is not canonical")
    paired_reports: list[dict[str, object]] = []
    for seed in EXPECTED_SEEDS:
        entry = by_seed[seed]
        path = Path(_string(entry.get("path"), "paired report path")).resolve()
        report = _verified_hashed_json(
            path,
            context=f"paired seed {seed} report",
            hash_field="artifact_sha256",
            type_field="artifact_type",
            artifact_type="ecg_trust.paired_patient_bootstrap_report",
        )
        if (
            report.get("seed") != seed
            or report.get("batch_sha256") != batch_sha256
            or report.get("artifact_sha256") != entry.get("artifact_sha256")
            or entry.get("direction") != "ecg_transformer_minus_resnet1d"
        ):
            raise PostEvaluationIntegrityError(f"paired seed {seed} report differs")
        paired_reports.append(
            {
                "seed": seed,
                "alignment_sha256": _hash(entry.get("alignment_sha256"), "paired alignment_sha256"),
                **_artifact_binding(path, report, hash_field="artifact_sha256"),
            }
        )
    return {
        "architecture_summaries": architecture_summaries,
        "paired_manifest": _artifact_binding(manifest_path, manifest, hash_field="artifact_sha256"),
        "paired_reports": paired_reports,
    }


def _explanation_cohort(
    ecg_id: NDArray[np.int64],
    patient_id: NDArray[np.int64],
    targets: NDArray[np.int8],
    *,
    alignment_sha256: str,
) -> dict[str, object]:
    candidates: list[tuple[str, int, int, int, int, tuple[int, ...]]] = []
    for row_index, (raw_ecg, raw_patient) in enumerate(
        zip(ecg_id.tolist(), patient_id.tolist(), strict=True)
    ):
        ecg = int(raw_ecg)
        patient = int(raw_patient)
        bits = tuple(int(value) for value in targets[row_index].tolist())
        for label_index, target_value in enumerate(bits):
            label = LABEL_ORDER[label_index]
            status = "positive" if target_value == 1 else "negative"
            key = (
                "ecg_trust:post-evaluation:explanation-cohort:v2:"
                f"{EXPLANATION_SELECTION_SEED}:{label}:{status}:{ecg}"
            )
            digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
            candidates.append((digest, ecg, patient, label_index, target_value, bits))
    candidates.sort(key=lambda value: (value[0], value[1], value[3], value[4]))
    records: list[dict[str, object]] = []
    selected_patients: set[int] = set()
    selected_ecgs: set[int] = set()
    cell_counts = {
        (label_index, target_value): 0
        for label_index in range(len(LABEL_ORDER))
        for target_value in (0, 1)
    }
    for digest, ecg, patient, label_index, target_value, bits in candidates:
        cell = (label_index, target_value)
        if (
            cell_counts[cell] >= EXPLANATION_CELL_SIZE
            or patient in selected_patients
            or ecg in selected_ecgs
        ):
            continue
        selected_patients.add(patient)
        selected_ecgs.add(ecg)
        cell_counts[cell] += 1
        records.append(
            {
                "rank": len(records),
                "ecg_id": ecg,
                "patient_id": patient,
                "target_label": LABEL_ORDER[label_index],
                "target_index": label_index,
                "target_status": "positive" if target_value == 1 else "negative",
                "target_value": target_value,
                "target_bits": list(bits),
                "selection_sha256": "sha256:" + digest,
            }
        )
        if len(records) == EXPLANATION_COHORT_SIZE:
            break
    if len(records) != EXPLANATION_COHORT_SIZE or any(
        count != EXPLANATION_CELL_SIZE for count in cell_counts.values()
    ):
        raise PostEvaluationIntegrityError(
            "fold 10 cannot satisfy the deterministic label/status explanation grid"
        )
    return {
        "selection_rule": "global_sha256_greedy_label_status_cells_v2",
        "selection_seed": EXPLANATION_SELECTION_SEED,
        "requested_records": EXPLANATION_COHORT_SIZE,
        "selected_records": len(records),
        "records_per_label_status_cell": EXPLANATION_CELL_SIZE,
        "cell_order": [
            {"target_label": label, "target_status": status}
            for label in LABEL_ORDER
            for status in ("negative", "positive")
        ],
        "one_ecg_per_patient": True,
        "source_alignment_sha256": alignment_sha256,
        "selection_data_policy": {
            "targets": "used_only_for_label_status_cell_membership",
            "predictions": "not_used",
            "metrics": "not_used",
        },
        "records": records,
    }


def _demo_protocol(
    members: Sequence[Mapping[str, object]],
    calibration_bundle: CalibrationBundle,
) -> dict[str, object]:
    member = next((item for item in members if item.get("member_id") == DEMO_MEMBER_ID), None)
    calibration = next(
        (item for item in calibration_bundle.members if item.member_id == DEMO_MEMBER_ID),
        None,
    )
    if member is None or calibration is None:
        raise PostEvaluationIntegrityError("canonical demo member is unavailable")
    matches = [
        gate
        for gate in calibration.entropy_gates
        if math.isclose(
            _number(gate.get("target_coverage"), "demo target coverage"),
            DEMO_TARGET_COVERAGE,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    if len(matches) != 1:
        raise PostEvaluationIntegrityError("canonical demo gate is unavailable")
    gate = dict(matches[0])
    return {
        "member_id": DEMO_MEMBER_ID,
        "architecture": "resnet1d",
        "seed": 2026,
        "target_coverage": DEMO_TARGET_COVERAGE,
        "entropy_method": "mean_normalized_binary_entropy",
        "gate": gate,
        "checkpoint": dict(_mapping(member["checkpoint"], "demo checkpoint")),
        "resolved_config": dict(_mapping(member["resolved_config"], "demo resolved config")),
        "calibration_decision": dict(
            _mapping(member["calibration_decision"], "demo calibration decision")
        ),
        "selection_basis": "fixed_operational_demo_default_not_fold10_model_selection",
        "retuning_allowed": False,
    }


def _build_body(
    *,
    protocol: ExperimentProtocol,
    final_batch_summary_path: Path,
    opening_ledger_path: Path | None,
    refit_bundle_path: Path,
    calibration_bundle_path: Path,
    final_evaluation_spec_path: Path,
    protocol_deviations_path: Path,
    project_root: Path,
    output_root: Path | None,
    analysis_runtime: Mapping[str, object],
    supersedes_spec_path: Path | None = None,
    supersession_reason: str | None = None,
) -> dict[str, object]:
    refit, calibration, refit_binding, calibration_binding = _validate_release_bundles(
        refit_bundle_path,
        calibration_bundle_path,
        protocol=protocol,
    )
    final_spec_binding, final_spec_payload = _final_spec_binding(
        final_evaluation_spec_path, protocol=protocol
    )
    frozen_refit = _mapping(final_spec_payload["refit_bundle"], "frozen refit bundle")
    if (
        frozen_refit.get("artifact_sha256") != refit_binding["artifact_sha256"]
        or frozen_refit.get("file_sha256") != refit_binding["file_sha256"]
        or Path(_string(frozen_refit.get("path"), "frozen refit path")).resolve()
        != refit_bundle_path
    ):
        raise PostEvaluationIntegrityError("final spec refit binding differs")
    deviation_binding = _protocol_deviation_binding(
        protocol_deviations_path, final_spec_payload=final_spec_payload
    )
    expected_ledger = _infer_opening_ledger_path(
        final_evaluation_spec_path,
        _string(final_spec_binding["artifact_sha256"], "final spec artifact hash"),
    )
    ledger_path = expected_ledger if opening_ledger_path is None else opening_ledger_path
    if ledger_path.resolve() != expected_ledger:
        raise PostEvaluationIntegrityError(
            "opening ledger path is not the canonical final-spec registry path"
        )
    ledger_binding, ledger = _opening_ledger_binding(
        ledger_path,
        protocol=protocol,
        final_spec_binding={
            key: final_spec_binding[key] for key in ("path", "file_sha256", "artifact_sha256")
        },
        refit_binding=refit_binding,
        calibration_binding=calibration_binding,
    )
    summary_binding, summary = _summary_binding(
        final_batch_summary_path,
        final_spec_binding=final_spec_binding,
        deviation_binding=deviation_binding,
    )
    batch_sha256 = _string(summary_binding["batch_sha256"], "batch_sha256")
    if ledger_binding["batch_sha256"] != batch_sha256:
        raise PostEvaluationIntegrityError("summary and opening ledger batches differ")
    ledger_outputs = _mapping(ledger["outputs"], "opening ledger outputs")
    if (
        Path(_string(ledger_outputs.get("batch_summary_path"), "ledger summary path")).resolve()
        != final_batch_summary_path
        or ledger_outputs.get("batch_summary_sha256") != summary_binding["artifact_sha256"]
    ):
        raise PostEvaluationIntegrityError("opening ledger summary binding differs")
    final_root = final_batch_summary_path.parent.resolve()
    members, ecg_id, patient_id, targets, alignment_sha256 = _build_member_bindings(
        final_root=final_root,
        summary=summary,
        ledger=ledger,
        refit_bundle=refit,
        calibration_bundle=calibration,
        protocol=protocol,
        final_spec_binding={
            key: final_spec_binding[key] for key in ("path", "file_sha256", "artifact_sha256")
        },
        deviation_binding=deviation_binding,
    )
    aggregates = _build_aggregate_bindings(
        summary=summary, ledger=ledger, batch_sha256=batch_sha256
    )
    comparison_ids = {member.comparison_id for member in refit.members}
    if len(comparison_ids) != 1:
        raise PostEvaluationIntegrityError("refit comparison identity is inconsistent")
    comparison_id = next(iter(comparison_ids))
    if (supersedes_spec_path is None) != (supersession_reason is None):
        raise PostEvaluationError(
            "superseded spec and supersession reason must be supplied together"
        )
    superseded: PostEvaluationSpec | None = None
    supersession: dict[str, object] | None = None
    if supersedes_spec_path is None:
        schema_version = POST_EVALUATION_SPEC_SCHEMA_VERSION
        audit_revision = 1
    else:
        superseded, supersession = _build_supersession_binding(
            supersedes_spec_path,
            reason=cast(str, supersession_reason),
        )
        old_protocol = _mapping(superseded.payload["protocol"], "superseded protocol")
        if old_protocol["comparison_id"] != comparison_id:
            raise PostEvaluationIntegrityError(
                "superseded specification comparison identity differs"
            )
        old_revision = _audit_revision_from_output_root(
            project_root,
            comparison_id,
            superseded.output_root,
        )
        schema_version = POST_EVALUATION_SUPERSESSION_SCHEMA_VERSION
        audit_revision = old_revision + 1
    outputs = _output_contract(
        project_root,
        comparison_id,
        audit_revision=audit_revision,
    )
    canonical_root = Path(_string(outputs["root"], "canonical output root"))
    if output_root is not None and output_root.resolve() != canonical_root:
        raise PostEvaluationError(f"post-evaluation output root must be exactly {canonical_root}")
    body: dict[str, object] = {
        "schema_version": schema_version,
        "artifact_type": POST_EVALUATION_SPEC_TYPE,
        "protocol": {
            "protocol_hash": protocol.protocol_hash,
            "comparison_id": comparison_id,
            "manifest_sha256": refit.manifest_sha256,
            "normalization_sha256": refit.normalization_sha256,
            "label_order": list(LABEL_ORDER),
            "final_folds": list(FINAL_TEST_FOLDS),
        },
        "analysis_runtime": dict(analysis_runtime),
        "sealed_evaluation": {
            "final_batch_summary": summary_binding,
            "opening_ledger": ledger_binding,
            "final_evaluation_spec": final_spec_binding,
            "protocol_deviations": deviation_binding,
            "refit_bundle": refit_binding,
            "calibration_bundle": calibration_binding,
        },
        "members": members,
        "aggregate_outputs": aggregates,
        "audit_protocols": {
            "robustness": _robustness_protocol(),
            "explanations": {
                "cohort": _explanation_cohort(
                    ecg_id,
                    patient_id,
                    targets,
                    alignment_sha256=alignment_sha256,
                ),
                "settings": _explanation_settings(),
            },
            "demo": _demo_protocol(members, calibration),
        },
        "output_contract": outputs,
    }
    if supersession is not None:
        body["supersession"] = supersession
        if superseded is None:  # pragma: no cover - construction invariant
            raise PostEvaluationIntegrityError("superseded specification is unavailable")
        old_payload = superseded.payload
        for key in (
            "protocol",
            "sealed_evaluation",
            "members",
            "aggregate_outputs",
            "audit_protocols",
        ):
            if body[key] != old_payload[key]:
                raise PostEvaluationIntegrityError(
                    f"superseding audit {key} differs from the superseded specification"
                )
    return body


def _validate_file_binding(
    value: object,
    *,
    context: str,
    expected_keys: set[str],
) -> Mapping[str, object]:
    binding = _mapping(value, context)
    _exact_keys(binding, expected_keys, context)
    Path(_string(binding["path"], f"{context}.path"))
    _hash(binding["file_sha256"], f"{context}.file_sha256")
    if "artifact_sha256" in binding:
        _hash(binding["artifact_sha256"], f"{context}.artifact_sha256")
    return binding


def _validate_cohort(value: object) -> None:
    cohort = _mapping(value, "explanation cohort")
    _exact_keys(
        cohort,
        {
            "selection_rule",
            "selection_seed",
            "requested_records",
            "selected_records",
            "records_per_label_status_cell",
            "cell_order",
            "one_ecg_per_patient",
            "source_alignment_sha256",
            "selection_data_policy",
            "records",
        },
        "explanation cohort",
    )
    if (
        cohort["selection_rule"] != "global_sha256_greedy_label_status_cells_v2"
        or cohort["selection_seed"] != EXPLANATION_SELECTION_SEED
        or cohort["requested_records"] != EXPLANATION_COHORT_SIZE
        or cohort["selected_records"] != EXPLANATION_COHORT_SIZE
        or cohort["records_per_label_status_cell"] != EXPLANATION_CELL_SIZE
        or cohort["one_ecg_per_patient"] is not True
    ):
        raise PostEvaluationIntegrityError("explanation cohort policy is not canonical")
    expected_cells = [
        {"target_label": label, "target_status": status}
        for label in LABEL_ORDER
        for status in ("negative", "positive")
    ]
    if list(_sequence(cohort["cell_order"], "cohort cell_order")) != expected_cells:
        raise PostEvaluationIntegrityError("explanation cohort cell order differs")
    if dict(_mapping(cohort["selection_data_policy"], "cohort selection data policy")) != {
        "targets": "used_only_for_label_status_cell_membership",
        "predictions": "not_used",
        "metrics": "not_used",
    }:
        raise PostEvaluationIntegrityError("cohort selection data policy differs")
    _hash(cohort["source_alignment_sha256"], "cohort source_alignment_sha256")
    records = _sequence(cohort["records"], "cohort records")
    if len(records) != EXPLANATION_COHORT_SIZE:
        raise PostEvaluationIntegrityError("explanation cohort size is not canonical")
    ecg_ids: set[int] = set()
    patient_ids: set[int] = set()
    cell_counts = {
        (label, status): 0 for label in LABEL_ORDER for status in ("negative", "positive")
    }
    for expected_rank, raw in enumerate(records):
        record = _mapping(raw, "cohort record")
        _exact_keys(
            record,
            {
                "rank",
                "ecg_id",
                "patient_id",
                "target_label",
                "target_index",
                "target_status",
                "target_value",
                "target_bits",
                "selection_sha256",
            },
            "cohort record",
        )
        ecg_id = _integer(record["ecg_id"], "cohort ecg_id", minimum=1)
        patient_id = _integer(record["patient_id"], "cohort patient_id", minimum=1)
        if record["rank"] != expected_rank or ecg_id in ecg_ids or patient_id in patient_ids:
            raise PostEvaluationIntegrityError("cohort ranks or identities are not unique")
        target_label = _string(record["target_label"], "cohort target_label")
        target_index = _integer(record["target_index"], "cohort target_index")
        target_status = _string(record["target_status"], "cohort target_status")
        target_value = _integer(record["target_value"], "cohort target_value")
        bits = list(_sequence(record["target_bits"], "cohort target_bits"))
        if (
            target_index >= len(LABEL_ORDER)
            or target_label != LABEL_ORDER[target_index]
            or target_status not in {"negative", "positive"}
            or target_value not in {0, 1}
            or target_status != ("positive" if target_value == 1 else "negative")
            or len(bits) != len(LABEL_ORDER)
            or any(value not in {0, 1} for value in bits)
            or bits[target_index] != target_value
        ):
            raise PostEvaluationIntegrityError("cohort target assignment is invalid")
        key = (
            "ecg_trust:post-evaluation:explanation-cohort:v2:"
            f"{EXPLANATION_SELECTION_SEED}:{target_label}:{target_status}:{ecg_id}"
        )
        expected_hash = "sha256:" + hashlib.sha256(key.encode("utf-8")).hexdigest()
        if record["selection_sha256"] != expected_hash:
            raise PostEvaluationIntegrityError("cohort selection hash is invalid")
        ecg_ids.add(ecg_id)
        patient_ids.add(patient_id)
        cell_counts[(target_label, target_status)] += 1
    if any(count != EXPLANATION_CELL_SIZE for count in cell_counts.values()):
        raise PostEvaluationIntegrityError("cohort label/status cells are unbalanced")


def _validate_payload(root: Mapping[str, object]) -> None:
    schema_version = _integer(root.get("schema_version"), "schema_version", minimum=1)
    if schema_version not in {
        POST_EVALUATION_SPEC_SCHEMA_VERSION,
        POST_EVALUATION_SUPERSESSION_SCHEMA_VERSION,
    }:
        raise PostEvaluationIntegrityError("unsupported post-evaluation schema")
    expected_root_keys = {
        "schema_version",
        "artifact_type",
        "protocol",
        "analysis_runtime",
        "sealed_evaluation",
        "members",
        "aggregate_outputs",
        "audit_protocols",
        "output_contract",
        "artifact_sha256",
    }
    if schema_version == POST_EVALUATION_SUPERSESSION_SCHEMA_VERSION:
        expected_root_keys.add("supersession")
    _exact_keys(
        root,
        expected_root_keys,
        "post-evaluation specification",
    )
    if root["artifact_type"] != POST_EVALUATION_SPEC_TYPE:
        raise PostEvaluationIntegrityError("unexpected post-evaluation artifact type")
    protocol = _mapping(root["protocol"], "protocol")
    _exact_keys(
        protocol,
        {
            "protocol_hash",
            "comparison_id",
            "manifest_sha256",
            "normalization_sha256",
            "label_order",
            "final_folds",
        },
        "protocol",
    )
    for key in ("protocol_hash", "manifest_sha256", "normalization_sha256"):
        _hash(protocol[key], f"protocol.{key}")
    comparison_id = _string(protocol["comparison_id"], "protocol.comparison_id")
    if list(_sequence(protocol["label_order"], "protocol label_order")) != list(
        LABEL_ORDER
    ) or list(_sequence(protocol["final_folds"], "protocol final_folds")) != [10]:
        raise PostEvaluationIntegrityError("protocol task contract is not canonical")

    runtime = _mapping(root["analysis_runtime"], "analysis_runtime")
    _exact_keys(runtime, {"project_root", "git_revision", "clean"}, "analysis_runtime")
    project_root = Path(_string(runtime["project_root"], "analysis project_root")).resolve()
    revision = _string(runtime["git_revision"], "analysis git_revision")
    if (
        runtime["clean"] is not True
        or len(revision) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise PostEvaluationIntegrityError("analysis Git binding is invalid")

    sealed = _mapping(root["sealed_evaluation"], "sealed_evaluation")
    _exact_keys(
        sealed,
        {
            "final_batch_summary",
            "opening_ledger",
            "final_evaluation_spec",
            "protocol_deviations",
            "refit_bundle",
            "calibration_bundle",
        },
        "sealed_evaluation",
    )
    _validate_file_binding(
        sealed["final_batch_summary"],
        context="final_batch_summary",
        expected_keys={"path", "file_sha256", "artifact_sha256", "batch_sha256"},
    )
    ledger_binding = _validate_file_binding(
        sealed["opening_ledger"],
        context="opening_ledger",
        expected_keys={
            "path",
            "file_sha256",
            "ledger_sha256",
            "state",
            "purpose",
            "operator",
            "batch_sha256",
            "opening_marker_path",
            "opening_marker_file_sha256",
            "opening_marker_sha256",
            "terminal_event",
        },
    )
    for key in (
        "ledger_sha256",
        "batch_sha256",
        "opening_marker_file_sha256",
        "opening_marker_sha256",
    ):
        _hash(ledger_binding[key], f"opening_ledger.{key}")
    if (
        ledger_binding["state"] != "complete"
        or ledger_binding["terminal_event"] != "exact_six_member_final_batch_complete"
    ):
        raise PostEvaluationIntegrityError("opening ledger binding is not terminal")
    _string(ledger_binding["purpose"], "opening_ledger.purpose")
    _string(ledger_binding["operator"], "opening_ledger.operator")
    Path(_string(ledger_binding["opening_marker_path"], "opening marker path"))
    _validate_file_binding(
        sealed["final_evaluation_spec"],
        context="final_evaluation_spec",
        expected_keys={
            "path",
            "file_sha256",
            "artifact_sha256",
            "evaluation_git_revision",
        },
    )
    final_spec_binding = _mapping(sealed["final_evaluation_spec"], "final_evaluation_spec")
    evaluation_revision = _string(
        final_spec_binding["evaluation_git_revision"], "evaluation Git revision"
    )
    if len(evaluation_revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in evaluation_revision
    ):
        raise PostEvaluationIntegrityError("evaluation Git revision is invalid")
    deviation = _validate_file_binding(
        sealed["protocol_deviations"],
        context="protocol_deviations",
        expected_keys={"path", "file_sha256", "required_in_all_reporting"},
    )
    if deviation["required_in_all_reporting"] is not True:
        raise PostEvaluationIntegrityError("protocol deviations must remain disclosed")
    bundle_keys = {
        "path",
        "file_sha256",
        "artifact_sha256",
        "protocol_hash",
        "manifest_sha256",
        "normalization_sha256",
        "member_count",
    }
    for name in ("refit_bundle", "calibration_bundle"):
        binding = _validate_file_binding(sealed[name], context=name, expected_keys=bundle_keys)
        for key in (
            "protocol_hash",
            "manifest_sha256",
            "normalization_sha256",
        ):
            _hash(binding[key], f"{name}.{key}")
        if binding["member_count"] != 6:
            raise PostEvaluationIntegrityError(f"{name} must bind six members")

    members = _sequence(root["members"], "members")
    if len(members) != 6:
        raise PostEvaluationIntegrityError("specification must bind exactly six members")
    observed_ids: list[str] = []
    for raw_member in members:
        member = _mapping(raw_member, "member")
        _exact_keys(
            member,
            {
                "member_id",
                "architecture",
                "seed",
                "model_name",
                "prediction",
                "final_report",
                "checkpoint",
                "resolved_config",
                "calibration_decision",
                "refit_lineage_sha256",
            },
            "member",
        )
        member_id = _string(member["member_id"], "member_id")
        observed_ids.append(member_id)
        architecture = _string(member["architecture"], "member architecture")
        seed = _integer(member["seed"], "member seed")
        if member_id != f"{architecture}-seed{seed}":
            raise PostEvaluationIntegrityError("member identity is inconsistent")
        _string(member["model_name"], "member model_name")
        _hash(member["refit_lineage_sha256"], "member refit lineage")
        prediction = _mapping(member["prediction"], "member prediction")
        _exact_keys(
            prediction,
            {
                "npz_path",
                "npz_file_sha256",
                "sidecar_path",
                "sidecar_file_sha256",
                "artifact_sha256",
                "alignment_sha256",
                "record_count",
            },
            "member prediction",
        )
        for key in (
            "npz_file_sha256",
            "sidecar_file_sha256",
            "artifact_sha256",
            "alignment_sha256",
        ):
            _hash(prediction[key], f"member prediction {key}")
        Path(_string(prediction["npz_path"], "prediction npz_path"))
        Path(_string(prediction["sidecar_path"], "prediction sidecar_path"))
        _integer(prediction["record_count"], "prediction record_count", minimum=1)
        _validate_file_binding(
            member["final_report"],
            context="member final_report",
            expected_keys={"path", "file_sha256", "artifact_sha256"},
        )
        _validate_file_binding(
            member["checkpoint"],
            context="member checkpoint",
            expected_keys={"path", "file_sha256"},
        )
        resolved = _validate_file_binding(
            member["resolved_config"],
            context="member resolved_config",
            expected_keys={"path", "file_sha256", "config_hash"},
        )
        _hash(resolved["config_hash"], "resolved config hash")
        _validate_file_binding(
            member["calibration_decision"],
            context="member calibration_decision",
            expected_keys={"path", "file_sha256", "artifact_sha256"},
        )
    if tuple(observed_ids) != EXPECTED_MEMBER_IDS:
        raise PostEvaluationIntegrityError("member order/grid is not canonical")

    aggregate = _mapping(root["aggregate_outputs"], "aggregate_outputs")
    _exact_keys(
        aggregate,
        {"architecture_summaries", "paired_manifest", "paired_reports"},
        "aggregate_outputs",
    )
    architectures = _sequence(aggregate["architecture_summaries"], "architecture_summaries")
    if len(architectures) != 2:
        raise PostEvaluationIntegrityError("two architecture summaries are required")
    observed_architectures: list[str] = []
    for raw in architectures:
        binding = _validate_file_binding(
            raw,
            context="architecture summary",
            expected_keys={"architecture", "path", "file_sha256", "artifact_sha256"},
        )
        observed_architectures.append(
            _string(binding["architecture"], "architecture summary architecture")
        )
    if tuple(observed_architectures) != EXPECTED_ARCHITECTURES:
        raise PostEvaluationIntegrityError("architecture summary order is not canonical")
    _validate_file_binding(
        aggregate["paired_manifest"],
        context="paired manifest",
        expected_keys={"path", "file_sha256", "artifact_sha256"},
    )
    paired = _sequence(aggregate["paired_reports"], "paired_reports")
    if len(paired) != 3:
        raise PostEvaluationIntegrityError("three paired reports are required")
    paired_seeds: list[int] = []
    for raw in paired:
        binding = _validate_file_binding(
            raw,
            context="paired report",
            expected_keys={
                "seed",
                "alignment_sha256",
                "path",
                "file_sha256",
                "artifact_sha256",
            },
        )
        paired_seeds.append(_integer(binding["seed"], "paired report seed"))
        _hash(binding["alignment_sha256"], "paired report alignment")
    if tuple(paired_seeds) != EXPECTED_SEEDS:
        raise PostEvaluationIntegrityError("paired report seeds are not canonical")

    protocols = _mapping(root["audit_protocols"], "audit_protocols")
    _exact_keys(protocols, {"robustness", "explanations", "demo"}, "audit_protocols")
    if dict(_mapping(protocols["robustness"], "robustness protocol")) != (_robustness_protocol()):
        raise PostEvaluationIntegrityError("robustness severity matrix is not canonical")
    explanations = _mapping(protocols["explanations"], "explanations protocol")
    _exact_keys(explanations, {"cohort", "settings"}, "explanations protocol")
    _validate_cohort(explanations["cohort"])
    if dict(_mapping(explanations["settings"], "explanation settings")) != (
        _explanation_settings()
    ):
        raise PostEvaluationIntegrityError("explanation settings are not canonical")
    demo = _mapping(protocols["demo"], "demo protocol")
    _exact_keys(
        demo,
        {
            "member_id",
            "architecture",
            "seed",
            "target_coverage",
            "entropy_method",
            "gate",
            "checkpoint",
            "resolved_config",
            "calibration_decision",
            "selection_basis",
            "retuning_allowed",
        },
        "demo protocol",
    )
    if (
        demo["member_id"] != DEMO_MEMBER_ID
        or demo["architecture"] != "resnet1d"
        or demo["seed"] != 2026
        or demo["target_coverage"] != DEMO_TARGET_COVERAGE
        or demo["entropy_method"] != "mean_normalized_binary_entropy"
        or demo["retuning_allowed"] is not False
    ):
        raise PostEvaluationIntegrityError("demo member/gate is not canonical")
    gate = _mapping(demo["gate"], "demo gate")
    if _number(gate.get("target_coverage"), "demo gate coverage") != DEMO_TARGET_COVERAGE:
        raise PostEvaluationIntegrityError("demo gate target coverage differs")

    output = _mapping(root["output_contract"], "output_contract")
    output_root = Path(_string(output["root"], "output root")).resolve()
    try:
        audit_revision = _audit_revision_from_output_root(
            project_root,
            comparison_id,
            output_root,
        )
    except PostEvaluationIntegrityError as error:
        raise PostEvaluationIntegrityError("output contract root is not canonical") from error
    if (schema_version == POST_EVALUATION_SPEC_SCHEMA_VERSION and audit_revision != 1) or (
        schema_version == POST_EVALUATION_SUPERSESSION_SCHEMA_VERSION and audit_revision < 2
    ):
        raise PostEvaluationIntegrityError("schema and output-root revision differ")
    expected_output = _output_contract(
        project_root,
        comparison_id,
        audit_revision=audit_revision,
    )
    if dict(output) != expected_output:
        raise PostEvaluationIntegrityError("output contract is not canonical")
    artifacts = _mapping(output["artifacts"], "output artifacts")
    for value in artifacts.values():
        candidate = Path(_string(value, "planned output path")).resolve()
        if candidate != output_root and output_root not in candidate.parents:
            raise PostEvaluationIntegrityError("planned output escapes fixed output root")

    stored_hash = _hash(root["artifact_sha256"], "artifact_sha256")
    unhashed = dict(root)
    del unhashed["artifact_sha256"]
    if canonical_sha256(unhashed) != stored_hash:
        raise PostEvaluationIntegrityError("post-evaluation specification self-hash mismatch")
    if schema_version == POST_EVALUATION_SUPERSESSION_SCHEMA_VERSION:
        _verify_supersession(root)


def _spec_from_payload(payload: Mapping[str, object], *, path: Path | None) -> PostEvaluationSpec:
    _validate_payload(payload)
    try:
        canonical = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:  # pragma: no cover - validation invariant
        raise PostEvaluationIntegrityError("specification is not finite JSON") from error
    return PostEvaluationSpec(
        path=path.resolve() if path is not None else None,
        artifact_sha256=cast(str, payload["artifact_sha256"]),
        _canonical_payload=canonical,
    )


def _is_junction(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    return bool(checker()) if checker is not None else False


def _output_tree_snapshot(root: Path) -> dict[str, object]:
    """Hash a closed, regular-file-only snapshot of a superseded output tree."""

    source_root = root.absolute()
    if not source_root.is_dir() or source_root.is_symlink() or _is_junction(source_root):
        raise PostEvaluationIntegrityError("superseded output root must be a regular directory")
    resolved_root = source_root.resolve()
    files: list[dict[str, object]] = []
    pending = [resolved_root]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise PostEvaluationIntegrityError(
                f"could not enumerate superseded output root: {error}"
            ) from error
        for entry in entries:
            path = Path(entry.path)
            if entry.is_symlink() or _is_junction(path):
                raise PostEvaluationIntegrityError(
                    "superseded output tree contains a link or junction"
                )
            if entry.is_dir(follow_symlinks=False):
                pending.append(path)
                continue
            if not entry.is_file(follow_symlinks=False):
                raise PostEvaluationIntegrityError(
                    "superseded output tree contains a non-regular entry"
                )
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(resolved_root)
            except ValueError as error:  # pragma: no cover - guarded by link rejection
                raise PostEvaluationIntegrityError(
                    "superseded output tree entry escapes its root"
                ) from error
            try:
                size = resolved.stat().st_size
            except OSError as error:
                raise PostEvaluationIntegrityError(
                    f"could not stat superseded output file: {error}"
                ) from error
            files.append(
                {
                    "path": relative.as_posix(),
                    "file_sha256": _file_sha256(resolved),
                    "size_bytes": size,
                }
            )
    files.sort(key=lambda item: cast(str, item["path"]))
    if not files:
        raise PostEvaluationIntegrityError("superseded output tree is empty")
    return {
        "files": files,
        "file_count": len(files),
        "tree_sha256": canonical_sha256({"files": files}),
    }


def _load_canonical_superseded_spec(path: Path) -> PostEvaluationSpec:
    source = path.resolve()
    spec = _spec_from_payload(
        _read_json(source, "superseded post-evaluation specification"),
        path=source,
    )
    expected = (spec.output_root / POST_EVALUATION_FILENAME).resolve()
    if source != expected:
        raise PostEvaluationIntegrityError(
            "superseded post-evaluation specification path is not canonical"
        )
    return spec


def _assert_superseded_manifests_absent(spec: PostEvaluationSpec) -> None:
    output = _mapping(spec.payload["output_contract"], "superseded output contract")
    artifacts = _mapping(output["artifacts"], "superseded output artifacts")
    for name in ("robustness_manifest", "derived_manifest"):
        path = Path(_string(artifacts[name], f"superseded {name}")).resolve()
        if path.exists():
            raise PostEvaluationIntegrityError(
                f"superseded {name} must be absent for an incomplete audit"
            )


def _build_supersession_binding(
    path: Path,
    *,
    reason: str,
) -> tuple[PostEvaluationSpec, dict[str, object]]:
    if reason != SUPERSESSION_REASON_DECIMAL_CASE_PATH_COLLISION:
        raise PostEvaluationError("supersession reason must be decimal_case_id_suffix_collision")
    old = _load_canonical_superseded_spec(path)
    if old.path is None:  # pragma: no cover - loader invariant
        raise PostEvaluationIntegrityError("superseded specification has no path")
    _assert_superseded_manifests_absent(old)
    runtime = _mapping(old.payload["analysis_runtime"], "superseded analysis runtime")
    binding = {
        "superseded_spec": {
            "path": str(old.path),
            "file_sha256": _file_sha256(old.path),
            "artifact_sha256": old.artifact_sha256,
            "git_revision": _string(runtime["git_revision"], "superseded analysis Git revision"),
            "output_root": str(old.output_root.resolve()),
        },
        "output_tree": _output_tree_snapshot(old.output_root),
        "reason": reason,
        "status": SUPERSESSION_STATUS_ABORTED,
        "derived_artifact_reuse_allowed": False,
    }
    return old, binding


def _verify_supersession(root: Mapping[str, object]) -> None:
    supersession = _mapping(root.get("supersession"), "supersession")
    _exact_keys(
        supersession,
        {
            "superseded_spec",
            "output_tree",
            "reason",
            "status",
            "derived_artifact_reuse_allowed",
        },
        "supersession",
    )
    if (
        supersession["reason"] != SUPERSESSION_REASON_DECIMAL_CASE_PATH_COLLISION
        or supersession["status"] != SUPERSESSION_STATUS_ABORTED
        or supersession["derived_artifact_reuse_allowed"] is not False
    ):
        raise PostEvaluationIntegrityError("supersession policy differs")
    binding = _mapping(supersession["superseded_spec"], "superseded spec binding")
    _exact_keys(
        binding,
        {"path", "file_sha256", "artifact_sha256", "git_revision", "output_root"},
        "superseded spec binding",
    )
    old_path_text = _string(binding["path"], "superseded spec path")
    old_root_text = _string(binding["output_root"], "superseded output root")
    old_path = Path(old_path_text).resolve()
    old_root = Path(old_root_text).resolve()
    if old_path_text != str(old_path) or old_root_text != str(old_root):
        raise PostEvaluationIntegrityError(
            "superseded spec path and output root must use canonical absolute spelling"
        )
    old_file_sha256 = _hash(binding["file_sha256"], "superseded spec file hash")
    _hash(binding["artifact_sha256"], "superseded spec artifact hash")
    old_git_revision = _string(binding["git_revision"], "superseded Git revision")
    if len(old_git_revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in old_git_revision
    ):
        raise PostEvaluationIntegrityError("superseded Git revision is invalid")

    protocol = _mapping(root["protocol"], "protocol")
    comparison_id = _string(protocol["comparison_id"], "protocol comparison_id")
    runtime = _mapping(root["analysis_runtime"], "analysis runtime")
    project_root = Path(_string(runtime["project_root"], "analysis project_root")).resolve()
    new_output = _mapping(root["output_contract"], "output contract")
    new_root = Path(_string(new_output["root"], "output root")).resolve()
    old_revision = _audit_revision_from_output_root(project_root, comparison_id, old_root)
    new_revision = _audit_revision_from_output_root(project_root, comparison_id, new_root)
    if old_root == new_root or new_revision != old_revision + 1:
        raise PostEvaluationIntegrityError(
            "superseding audit must use the next distinct sibling output root"
        )
    if old_path != (old_root / POST_EVALUATION_FILENAME).resolve():
        raise PostEvaluationIntegrityError("superseded spec path and root differ")
    if _file_sha256(old_path) != old_file_sha256:
        raise PostEvaluationIntegrityError("superseded spec file changed")
    old = _load_canonical_superseded_spec(old_path)
    old_payload = old.payload
    old_runtime = _mapping(old_payload["analysis_runtime"], "superseded analysis runtime")
    if (
        old.artifact_sha256 != binding["artifact_sha256"]
        or old_runtime["git_revision"] != old_git_revision
        or old.output_root.resolve() != old_root
        or Path(_string(old_runtime["project_root"], "superseded project root")).resolve()
        != project_root
    ):
        raise PostEvaluationIntegrityError("superseded spec identity differs")
    if runtime["git_revision"] == old_git_revision:
        raise PostEvaluationIntegrityError(
            "superseding audit must bind a different committed Git revision"
        )
    _assert_superseded_manifests_absent(old)
    snapshot = _mapping(supersession["output_tree"], "superseded output tree")
    _exact_keys(
        snapshot,
        {"files", "file_count", "tree_sha256"},
        "superseded output tree",
    )
    raw_files = _sequence(snapshot["files"], "superseded output files")
    paths: list[str] = []
    normalized_files: list[dict[str, object]] = []
    for raw_file in raw_files:
        entry = _mapping(raw_file, "superseded output file")
        _exact_keys(
            entry,
            {"path", "file_sha256", "size_bytes"},
            "superseded output file",
        )
        relative = _string(entry["path"], "superseded relative path")
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or relative_path.as_posix() != relative
            or ".." in relative_path.parts
        ):
            raise PostEvaluationIntegrityError("superseded output relative path is not canonical")
        paths.append(relative)
        normalized_files.append(
            {
                "path": relative,
                "file_sha256": _hash(entry["file_sha256"], "superseded output file hash"),
                "size_bytes": _integer(entry["size_bytes"], "superseded output file size"),
            }
        )
    if paths != sorted(paths) or len(set(paths)) != len(paths):
        raise PostEvaluationIntegrityError("superseded output file snapshot is not uniquely sorted")
    file_count = _integer(snapshot["file_count"], "superseded file_count", minimum=1)
    if file_count != len(normalized_files):
        raise PostEvaluationIntegrityError("superseded output file count differs")
    tree_sha256 = _hash(snapshot["tree_sha256"], "superseded tree hash")
    if tree_sha256 != canonical_sha256({"files": normalized_files}):
        raise PostEvaluationIntegrityError("superseded output tree hash differs")
    if dict(snapshot) != _output_tree_snapshot(old_root):
        raise PostEvaluationIntegrityError("superseded output tree changed")
    for key in (
        "protocol",
        "sealed_evaluation",
        "members",
        "aggregate_outputs",
        "audit_protocols",
    ):
        if root[key] != old_payload[key]:
            raise PostEvaluationIntegrityError(
                f"superseding audit {key} differs from the superseded specification"
            )


def create_post_evaluation_spec(
    *,
    protocol: ExperimentProtocol,
    final_batch_summary_path: str | Path,
    refit_bundle_path: str | Path,
    calibration_bundle_path: str | Path,
    final_evaluation_spec_path: str | Path,
    protocol_deviations_path: str | Path,
    project_root: str | Path,
    opening_ledger_path: str | Path | None = None,
    output_root: str | Path | None = None,
    supersedes_spec_path: str | Path | None = None,
    supersession_reason: str | None = None,
) -> PostEvaluationSpec:
    """Verify the completed release and create a deterministic audit freeze."""

    if not isinstance(protocol, ExperimentProtocol):
        raise TypeError("protocol must be an ExperimentProtocol")
    resolved_project = Path(project_root).resolve()
    runtime = _capture_clean_git(resolved_project)
    body = _build_body(
        protocol=protocol,
        final_batch_summary_path=Path(final_batch_summary_path).resolve(),
        opening_ledger_path=(
            None if opening_ledger_path is None else Path(opening_ledger_path).resolve()
        ),
        refit_bundle_path=Path(refit_bundle_path).resolve(),
        calibration_bundle_path=Path(calibration_bundle_path).resolve(),
        final_evaluation_spec_path=Path(final_evaluation_spec_path).resolve(),
        protocol_deviations_path=Path(protocol_deviations_path).resolve(),
        project_root=resolved_project,
        output_root=None if output_root is None else Path(output_root).resolve(),
        analysis_runtime=runtime,
        supersedes_spec_path=(
            None if supersedes_spec_path is None else Path(supersedes_spec_path).resolve()
        ),
        supersession_reason=supersession_reason,
    )
    payload = dict(body)
    payload["artifact_sha256"] = canonical_sha256(body)
    return _spec_from_payload(payload, path=None)


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable post-evaluation spec already exists: {path}")
    try:
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
    except (TypeError, ValueError) as error:
        raise PostEvaluationError("specification must be finite JSON") from error
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
                f"immutable post-evaluation spec already exists: {path}"
            ) from error
    finally:
        with suppress(OSError):
            temporary.unlink()


def save_post_evaluation_spec(spec: PostEvaluationSpec, path: str | Path) -> PostEvaluationSpec:
    """Atomically publish the spec at its sole canonical no-overwrite path."""

    if not isinstance(spec, PostEvaluationSpec):
        raise TypeError("spec must be a PostEvaluationSpec")
    payload = spec.to_payload()
    _validate_payload(payload)
    output = _mapping(payload["output_contract"], "output_contract")
    artifacts = _mapping(output["artifacts"], "output artifacts")
    canonical_path = Path(_string(artifacts["audit_spec"], "audit spec path")).resolve()
    destination = Path(path).resolve()
    if destination != canonical_path or destination.name != POST_EVALUATION_FILENAME:
        raise PostEvaluationError(
            f"post-evaluation specification must be saved at {canonical_path}"
        )
    _write_new_json(destination, payload)
    return _spec_from_payload(payload, path=destination)


def load_post_evaluation_spec(
    path: str | Path,
    *,
    protocol: ExperimentProtocol,
    verify_sources: bool = True,
    verify_git: bool = True,
) -> PostEvaluationSpec:
    """Load a strict spec and optionally reverify every release input and Git."""

    if not isinstance(protocol, ExperimentProtocol):
        raise TypeError("protocol must be an ExperimentProtocol")
    source = Path(path).resolve()
    spec = _spec_from_payload(_read_json(source, "post-evaluation specification"), path=source)
    payload = spec.payload
    protocol_binding = _mapping(payload["protocol"], "protocol")
    if protocol_binding["protocol_hash"] != protocol.protocol_hash:
        raise PostEvaluationIntegrityError("spec protocol differs from supplied protocol")
    output = _mapping(payload["output_contract"], "output_contract")
    artifacts = _mapping(output["artifacts"], "output artifacts")
    if source != Path(_string(artifacts["audit_spec"], "audit spec path")).resolve():
        raise PostEvaluationIntegrityError("spec is not stored at its canonical path")
    runtime = _mapping(payload["analysis_runtime"], "analysis_runtime")
    project_root = Path(_string(runtime["project_root"], "analysis project_root")).resolve()
    if verify_git and _capture_clean_git(project_root) != dict(runtime):
        raise PostEvaluationIntegrityError(
            "current clean Git revision differs from the post-evaluation freeze"
        )
    if verify_sources:
        sealed = _mapping(payload["sealed_evaluation"], "sealed_evaluation")
        supersedes_spec_path: Path | None = None
        supersession_reason: str | None = None
        if payload["schema_version"] == POST_EVALUATION_SUPERSESSION_SCHEMA_VERSION:
            supersession = _mapping(payload["supersession"], "supersession")
            superseded_binding = _mapping(
                supersession["superseded_spec"], "superseded spec binding"
            )
            supersedes_spec_path = Path(
                _string(superseded_binding["path"], "superseded spec path")
            ).resolve()
            supersession_reason = _string(supersession["reason"], "supersession reason")
        recreated_body = _build_body(
            protocol=protocol,
            final_batch_summary_path=Path(
                _string(
                    _mapping(sealed["final_batch_summary"], "final_batch_summary")["path"],
                    "final batch summary path",
                )
            ).resolve(),
            opening_ledger_path=Path(
                _string(
                    _mapping(sealed["opening_ledger"], "opening_ledger")["path"],
                    "opening ledger path",
                )
            ).resolve(),
            refit_bundle_path=Path(
                _string(
                    _mapping(sealed["refit_bundle"], "refit_bundle")["path"],
                    "refit bundle path",
                )
            ).resolve(),
            calibration_bundle_path=Path(
                _string(
                    _mapping(sealed["calibration_bundle"], "calibration_bundle")["path"],
                    "calibration bundle path",
                )
            ).resolve(),
            final_evaluation_spec_path=Path(
                _string(
                    _mapping(sealed["final_evaluation_spec"], "final_evaluation_spec")["path"],
                    "final evaluation spec path",
                )
            ).resolve(),
            protocol_deviations_path=Path(
                _string(
                    _mapping(sealed["protocol_deviations"], "protocol_deviations")["path"],
                    "protocol deviations path",
                )
            ).resolve(),
            project_root=project_root,
            output_root=Path(_string(output["root"], "output root")).resolve(),
            analysis_runtime=runtime,
            supersedes_spec_path=supersedes_spec_path,
            supersession_reason=supersession_reason,
        )
        expected_body = dict(payload)
        del expected_body["artifact_sha256"]
        if recreated_body != expected_body:
            raise PostEvaluationIntegrityError(
                "bound release inputs differ from the post-evaluation freeze"
            )
    return spec


def freeze_post_evaluation_spec(
    output_path: str | Path | None,
    *,
    protocol: ExperimentProtocol,
    final_batch_summary_path: str | Path,
    refit_bundle_path: str | Path,
    calibration_bundle_path: str | Path,
    final_evaluation_spec_path: str | Path,
    protocol_deviations_path: str | Path,
    project_root: str | Path,
    opening_ledger_path: str | Path | None = None,
    output_root: str | Path | None = None,
    supersedes_spec_path: str | Path | None = None,
    supersession_reason: str | None = None,
) -> PostEvaluationSpec:
    """Create and atomically save one post-evaluation specification."""

    created = create_post_evaluation_spec(
        protocol=protocol,
        final_batch_summary_path=final_batch_summary_path,
        opening_ledger_path=opening_ledger_path,
        refit_bundle_path=refit_bundle_path,
        calibration_bundle_path=calibration_bundle_path,
        final_evaluation_spec_path=final_evaluation_spec_path,
        protocol_deviations_path=protocol_deviations_path,
        project_root=project_root,
        output_root=output_root,
        supersedes_spec_path=supersedes_spec_path,
        supersession_reason=supersession_reason,
    )
    destination = (
        created.output_root / POST_EVALUATION_FILENAME
        if output_path is None
        else Path(output_path).resolve()
    )
    saved = save_post_evaluation_spec(created, destination)
    if saved.path is None:  # pragma: no cover - save invariant
        raise PostEvaluationIntegrityError("saved specification has no path")
    return load_post_evaluation_spec(
        saved.path,
        protocol=protocol,
        verify_sources=False,
        # Git cleanliness was captured immediately before the ignored runs/ write.
        verify_git=False,
    )


__all__ = [
    "DEMO_MEMBER_ID",
    "DEMO_TARGET_COVERAGE",
    "EXPLANATION_COHORT_SIZE",
    "PostEvaluationError",
    "PostEvaluationIntegrityError",
    "PostEvaluationSpec",
    "canonical_sha256",
    "create_post_evaluation_spec",
    "freeze_post_evaluation_spec",
    "load_post_evaluation_spec",
    "save_post_evaluation_spec",
]
