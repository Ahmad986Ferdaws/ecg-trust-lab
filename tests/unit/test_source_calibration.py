from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

import ecg_trust.source_calibration.pipeline as pipeline_module
from ecg_trust.open_world import max_normalized_bernoulli_entropy, normalized_bernoulli_entropy
from ecg_trust.predictions import create_prediction_artifact, save_prediction_artifact
from ecg_trust.protocol import ExperimentProtocol, FoldRole
from ecg_trust.source_calibration import (
    FAILURE_RECEIPT_FILENAME,
    RESULT_FILENAME,
    FittedEntropyGate,
    SourceCalibrationConfig,
    SourceCalibrationConfigError,
    SourceCalibrationIntegrityError,
    SourceCalibrationOutputError,
    SourceCalibrationResult,
    SourcePredictionArrays,
    SourceRole,
    VerifiedSourceInputs,
    assert_complete_release_ready,
    build_source_calibration_result,
    evaluate_source_validation,
    fit_entropy_gate,
    fit_source_components,
    load_source_calibration_config,
    load_source_calibration_result_bytes,
    load_verified_source_predictions,
    partition_source_predictions,
    patient_split_fraction,
    patient_split_role,
    prepare_source_calibration,
    replace_validation_role,
    verify_clean_git_revision,
    verify_source_inputs,
)
from ecg_trust.source_calibration.models import (
    EntropyGateSummary,
    canonical_json_bytes,
    canonical_sha256,
)

LABELS = ("NORM", "MI", "STTC", "CD", "HYP")
REVISION = "a" * 40


@dataclass(frozen=True, slots=True)
class SyntheticProject:
    root: Path
    config_path: Path
    npz_path: Path
    sidecar_path: Path
    output_root: Path
    patient_id: np.ndarray[tuple[int], np.dtype[np.int64]]
    targets: np.ndarray[tuple[int, int], np.dtype[np.int8]]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _positive_mapping(targets: np.ndarray[tuple[int, int], np.dtype[np.int8]]) -> dict[str, int]:
    values = tuple(int(value) for value in targets.sum(axis=0))
    return dict(zip(LABELS, values, strict=True))


def _build_synthetic_project(root: Path) -> SyntheticProject:
    protocol = ExperimentProtocol.canonical()
    salt = "synthetic-sentinel-v1"
    patient_values: list[int] = []
    for patient in range(1, 91):
        patient_values.extend([patient] * (2 if patient % 5 == 0 else 1))
    patient_id = np.asarray(patient_values, dtype=np.int64)
    records = int(patient_id.size)
    ecg_id = np.arange(100_001, 100_001 + records, dtype=np.int64)
    targets = np.zeros((records, len(LABELS)), dtype=np.int8)
    logits = np.empty((records, len(LABELS)), dtype=np.float64)
    for row, patient in enumerate(patient_values):
        for label_index in range(len(LABELS)):
            target_digest = hashlib.sha256(
                f"target|{patient}|{row}|{label_index}".encode()
            ).digest()
            target = int.from_bytes(target_digest[:8], "big") % 100 < 45 - 4 * label_index
            targets[row, label_index] = int(target)
            logit_digest = hashlib.sha256(f"logit|{patient}|{row}|{label_index}".encode()).digest()
            base = (int.from_bytes(logit_digest[:8], "big") % 4_001) / 1_000 - 2.0
            logits[row, label_index] = base + (0.8 if target else -0.2)

    relative_npz = Path("runs/synthetic/resnet1d-seed2026.fold9.npz")
    npz_path = root / relative_npz
    artifact = create_prediction_artifact(
        ecg_id=ecg_id,
        patient_id=patient_id,
        strat_fold=np.full(records, 9, dtype=np.int8),
        targets=targets,
        raw_logits=logits,
        model_name="resnet1d-seed2026",
        model_seed=2026,
        protocol=protocol,
        config_hash="sha256:" + "1" * 64,
        manifest_hash="sha256:" + "2" * 64,
        fold_role=FoldRole.CALIBRATION,
        created_at_utc="2026-08-24T07:25:28Z",
        producer="tests.synthetic_source_calibration",
    )
    save_prediction_artifact(artifact, npz_path, protocol=protocol)
    sidecar_path = npz_path.with_suffix(".json")

    demo_path = root / "artifacts/synthetic/demo-binding.json"
    historical_path = root / "artifacts/synthetic/historical-policy.json"
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    demo_path.write_bytes(b'{"artifact":"synthetic-demo-binding"}\n')
    historical_path.write_bytes(b'{"artifact":"synthetic-historical-policy"}\n')

    roles = np.asarray(
        [patient_split_role(patient_id=int(value), salt=salt).value for value in patient_id]
    )
    expected: dict[str, object] = {}
    for role in SourceRole:
        mask = roles == role.value
        expected[role.value] = {
            "records": int(mask.sum()),
            "patients": int(np.unique(patient_id[mask]).size),
            "positive_records": _positive_mapping(targets[mask]),
        }

    config_payload: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": "trust-sentinel-source-calibration-v1",
        "status": "frozen_pre_execution",
        "frozen_at_utc": "2026-08-24T07:25:28Z",
        "research_only": True,
        "purpose": "Synthetic-only verification of the frozen source-calibration pipeline.",
        "model": {
            "member_id": "resnet1d-seed2026",
            "source_artifact_model_name": "resnet1d-seed2026",
            "architecture": "resnet1d",
            "seed": 2026,
            "selection_rule": "development_selected_primary_architecture_and_first_fixed_seed",
            "checkpoint_sha256": "3" * 64,
            "demo_binding": {
                "path": demo_path.relative_to(root).as_posix(),
                "file_sha256": _sha256(demo_path),
            },
            "historical_policy": {
                "path": historical_path.relative_to(root).as_posix(),
                "file_sha256": _sha256(historical_path),
                "use": "comparison_only_not_reused_for_vnext_fitting",
            },
        },
        "source_prediction": {
            "role": "ptbxl_fold9_source_calibration_pool",
            "npz_path": relative_npz.as_posix(),
            "npz_sha256": _sha256(npz_path),
            "sidecar_path": relative_npz.with_suffix(".json").as_posix(),
            "sidecar_sha256": _sha256(sidecar_path),
            "expected_records": records,
            "label_order": list(LABELS),
        },
        "patient_split": {
            "unit": "patient_id",
            "algorithm": (
                'sha256(utf8(salt + "|" + base10_patient_id)); take the first 8 digest bytes '
                "as an unsigned big-endian integer; divide by 2^64"
            ),
            "salt": salt,
            "ranges": {
                "decision_fit": "[0.0,0.4)",
                "conformal_and_ood_threshold_fit": "[0.4,0.8)",
                "source_validation": "[0.8,1.0)",
            },
            "expected": expected,
            "design_provenance": "Proportions frozen before viewing synthetic scores.",
        },
        "decision_fit": {
            "temperature_scaling": "single_positive_temperature_binary_nll",
            "classification_thresholds": "per_label_maximum_f1",
            "classification_threshold_tie_rule": (
                "maximum_f1_then_closest_to_0.5_then_higher_threshold"
            ),
            "legacy_entropy_gate": {
                "method": "mean_normalized_binary_entropy",
                "target_coverage": 0.8,
                "tie_rule": ("retain_all_scores_less_than_or_equal_to_frozen_order_statistic"),
            },
        },
        "conformal": {
            "method": "labelwise_binary_split_conformal",
            "alpha": 0.1,
            "fit_role": "conformal_and_ood_threshold_fit",
            "coverage_scope": "labelwise_marginal_under_exchangeability",
            "individual_certainty_guarantee": False,
        },
        "open_world": {
            "primary_method": "shrinkage_mahalanobis_embedding_distance",
            "embedding": "frozen_resnet_preclassifier_global_average_pool",
            "reference_role": "ptbxl_folds_1_to_8_training_reference",
            "threshold_role": "conformal_and_ood_threshold_fit",
            "threshold_inlier_coverage": 0.95,
            "shrinkage": 0.1,
            "ridge": 0.000001,
            "target_site_fitting": "forbidden",
        },
        "source_validation": {
            "tuning_allowed": False,
            "report": [
                "labelwise_conformal_coverage_and_set_size",
                "legacy_gate_coverage",
                "hamming_loss_and_exact_match_at_frozen_thresholds",
                "auroc_average_precision_brier_ece15",
                "ood_source_false_rejection",
            ],
            "minimum_positive_records_for_label_statement": 2,
        },
        "forbidden_fit_or_selection_sources": [
            "ptbxl_fold10",
            "sph",
            "future_external_observed_sites",
            "future_external_lockbox_sites",
        ],
        "execution": {
            "require_clean_committed_revision": True,
            "require_verified_input_hashes": True,
            "output_root": "artifacts/synthetic/source-calibration-v1",
            "output_root_must_be_absent": True,
            "automatic_publication": False,
            "raw_ids_or_row_arrays_public": False,
        },
        "claims": {
            "scope": "retrospective_source_domain_development_only",
            "limitations": [
                "no_external_lockbox_evaluation",
                "no_clinical_validation",
                "conformal_coverage_is_marginal_not_individual",
                "ood_score_does_not_identify_unknown_disease",
                "thresholds_are_provisional_research_values",
            ],
        },
    }
    config_path = root / "configs/source-calibration.yaml"
    _write_yaml(config_path, config_payload)
    return SyntheticProject(
        root=root,
        config_path=config_path,
        npz_path=npz_path,
        sidecar_path=sidecar_path,
        output_root=root / "artifacts/synthetic/source-calibration-v1",
        patient_id=patient_id,
        targets=targets,
    )


@pytest.fixture
def synthetic_project(tmp_path: Path) -> SyntheticProject:
    return _build_synthetic_project(tmp_path / "project")


def _loaded_source(
    project: SyntheticProject,
) -> tuple[SourceCalibrationConfig, VerifiedSourceInputs, SourcePredictionArrays]:
    config, _ = load_source_calibration_config(project.config_path)
    verified = verify_source_inputs(config, project_root=project.root)
    source = load_verified_source_predictions(config, verified)
    return config, verified, source


def _rewrite_npz_hash(project: SyntheticProject) -> SourceCalibrationConfig:
    decoded = cast(
        dict[str, object], yaml.safe_load(project.config_path.read_text(encoding="utf-8"))
    )
    source = cast(dict[str, object], decoded["source_prediction"])
    source["npz_sha256"] = _sha256(project.npz_path)
    _write_yaml(project.config_path, decoded)
    return load_source_calibration_config(project.config_path)[0]


def test_frozen_config_is_strict_and_official_counts_are_bound(
    synthetic_project: SyntheticProject,
) -> None:
    config, digest = load_source_calibration_config(synthetic_project.config_path)
    assert config.status == "frozen_pre_execution"
    assert digest.startswith("sha256:") and len(digest) == 71

    original_yaml = synthetic_project.config_path.read_text(encoding="utf-8")
    decoded = cast(dict[str, object], yaml.safe_load(original_yaml))
    decoded["unexpected"] = True
    _write_yaml(synthetic_project.config_path, decoded)
    with pytest.raises(SourceCalibrationConfigError):
        load_source_calibration_config(synthetic_project.config_path)
    synthetic_project.config_path.write_text(
        original_yaml + "status: frozen_pre_execution\n", encoding="utf-8"
    )
    with pytest.raises(SourceCalibrationConfigError):
        load_source_calibration_config(synthetic_project.config_path)
    for field, value in (
        ("automatic_publication", 0),
        ("require_clean_committed_revision", 1),
    ):
        numeric_boolean = cast(dict[str, object], yaml.safe_load(original_yaml))
        execution = cast(dict[str, object], numeric_boolean["execution"])
        execution[field] = value
        _write_yaml(synthetic_project.config_path, numeric_boolean)
        with pytest.raises(SourceCalibrationConfigError):
            load_source_calibration_config(synthetic_project.config_path)
    numeric_version = cast(dict[str, object], yaml.safe_load(original_yaml))
    numeric_version["schema_version"] = 1.0
    _write_yaml(synthetic_project.config_path, numeric_version)
    with pytest.raises(SourceCalibrationConfigError):
        load_source_calibration_config(synthetic_project.config_path)

    official = (
        Path(__file__).resolve().parents[2] / "configs/trust_sentinel_source_calibration_v1.yaml"
    )
    official_config, _ = load_source_calibration_config(official)
    expected = official_config.patient_split.expected
    assert (expected.decision_fit.records, expected.decision_fit.patients) == (847, 751)
    assert (
        expected.conformal_and_ood_threshold_fit.records,
        expected.conformal_and_ood_threshold_fit.patients,
    ) == (834, 757)
    assert (expected.source_validation.records, expected.source_validation.patients) == (465, 409)
    assert expected.source_validation.positive_records.model_dump() == {
        "NORM": 216,
        "MI": 117,
        "STTC": 110,
        "CD": 104,
        "HYP": 45,
    }


def test_patient_split_and_assignment_hash_are_exact_and_deterministic(
    synthetic_project: SyntheticProject,
) -> None:
    config, _, source = _loaded_source(synthetic_project)
    patient = 42
    digest = hashlib.sha256(f"{config.patient_split.salt}|{patient}".encode()).digest()
    expected_fraction = int.from_bytes(digest[:8], "big") / float(1 << 64)
    assert patient_split_fraction(patient_id=patient, salt=config.patient_split.salt) == (
        expected_fraction
    )
    expected_role = (
        SourceRole.DECISION_FIT
        if expected_fraction < 0.4
        else SourceRole.CONFORMAL_AND_OOD_THRESHOLD_FIT
        if expected_fraction < 0.8
        else SourceRole.SOURCE_VALIDATION
    )
    assert patient_split_role(patient_id=patient, salt=config.patient_split.salt) is expected_role

    first = partition_source_predictions(source, config)
    second = partition_source_predictions(source, config)
    assignments = [
        {
            "patient_id_base10": str(value),
            "role": patient_split_role(patient_id=value, salt=config.patient_split.salt).value,
        }
        for value in sorted(int(item) for item in np.unique(source.patient_id))
    ]
    expected_hash = canonical_sha256(
        {
            "schema_version": 1,
            "algorithm": "sha256_first8_uint64_fraction_v1",
            "assignments": assignments,
        }
    )
    assert first.evidence.assignment_sha256 == second.evidence.assignment_sha256 == expected_hash
    patient_roles: dict[int, set[SourceRole]] = {}
    for role in SourceRole:
        for value in first.for_role(role).patient_id:
            patient_roles.setdefault(int(value), set()).add(role)
    assert patient_roles and all(len(roles) == 1 for roles in patient_roles.values())


def test_entropy_gate_retains_all_cutoff_ties() -> None:
    probabilities = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.5, 0.5, 0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    gate = fit_entropy_gate(probabilities, target_coverage=0.5)
    assert gate.fit_count == 4
    assert gate.selected_count == 3
    assert gate.achieved_coverage == 0.75


def _order_statistic_gate(uncertainty: np.ndarray, target_coverage: float) -> tuple[float, int]:
    requested = min(uncertainty.size, max(1, int(np.ceil(target_coverage * uncertainty.size))))
    cutoff = float(np.sort(uncertainty, kind="stable")[requested - 1])
    return cutoff, int(np.count_nonzero(uncertainty <= cutoff))


def _pre_change_mean_gate(probabilities: np.ndarray, target_coverage: float) -> tuple[float, int]:
    return _order_statistic_gate(normalized_bernoulli_entropy(probabilities), target_coverage)


def test_worst_label_gate_rejects_a_borderline_label_the_mean_gate_admits() -> None:
    generator = np.random.default_rng(20260923)
    # Every fit label is moderately confident: its entropy lies in [H(0.2), H(0.02)] < 1.
    magnitude = generator.uniform(0.02, 0.2, size=(200, 5))
    fit = np.where(generator.random((200, 5)) < 0.5, magnitude, 1.0 - magnitude)
    borderline = np.asarray([[0.5, 0.001, 0.999, 0.001, 0.001]], dtype=np.float64)
    confident = np.asarray([[0.001, 0.999, 0.001, 0.001, 0.999]], dtype=np.float64)

    mean_gate = fit_entropy_gate(fit, target_coverage=0.8)
    max_gate = fit_entropy_gate(fit, target_coverage=0.8, aggregation="max")

    assert (mean_gate.fit_count, max_gate.fit_count) == (200, 200)
    # Each cutoff is the 160th smallest fit score of its own aggregation.
    mean_scores = normalized_bernoulli_entropy(fit)
    max_scores = max_normalized_bernoulli_entropy(fit)
    assert mean_gate.maximum_entropy == float(np.sort(mean_scores, kind="stable")[159])
    assert max_gate.maximum_entropy == float(np.sort(max_scores, kind="stable")[159])
    assert max_gate.maximum_entropy > mean_gate.maximum_entropy
    assert int(mean_gate.retains(fit).sum()) == mean_gate.selected_count >= 160
    assert int(max_gate.retains(fit).sum()) == max_gate.selected_count >= 160
    assert normalized_bernoulli_entropy(borderline)[0] == pytest.approx(0.2091262, abs=1e-6)
    assert max_normalized_bernoulli_entropy(borderline)[0] == pytest.approx(1.0)
    # Failing before: the frozen mean gate admits one label at probability one half.
    assert mean_gate.retains(borderline).tolist() == [True]
    # Passing after: the worst-label gate judges that label on its own.
    assert max_gate.maximum_entropy < 1.0
    assert max_gate.retains(borderline).tolist() == [False]
    assert mean_gate.retains(confident).tolist() == max_gate.retains(confident).tolist() == [True]


def test_entropy_gate_default_is_the_unchanged_mean_gate() -> None:
    generator = np.random.default_rng(7)
    probabilities = generator.uniform(0.0, 1.0, size=(97, 5))
    probabilities[:5] = 0.1

    default = fit_entropy_gate(probabilities, target_coverage=0.8)
    explicit = fit_entropy_gate(probabilities, target_coverage=0.8, aggregation="mean")
    cutoff, selected = _pre_change_mean_gate(probabilities, 0.8)

    assert default == explicit
    assert default.aggregation == "mean"
    assert default.method == "mean_normalized_binary_entropy"
    assert (default.maximum_entropy, default.selected_count, default.fit_count) == (
        cutoff,
        selected,
        97,
    )
    assert np.array_equal(
        default.retains(probabilities), normalized_bernoulli_entropy(probabilities) <= cutoff
    )
    worst = fit_entropy_gate(probabilities, target_coverage=0.8, aggregation="max")
    assert worst.method == "max_normalized_binary_entropy"
    assert (worst.maximum_entropy, worst.selected_count) == _order_statistic_gate(
        max_normalized_bernoulli_entropy(probabilities), 0.8
    )
    assert np.array_equal(
        worst.retains(probabilities),
        max_normalized_bernoulli_entropy(probabilities) <= worst.maximum_entropy,
    )


def test_worst_label_gate_cutoff_stays_inside_the_sealed_unit_interval() -> None:
    # One label a few ulps below one half: its entropy rounds one ulp above one bit.
    fit = np.tile([0.49999999999999983, 0.01, 0.99, 0.01, 0.01], (10, 1))
    gate = fit_entropy_gate(fit, target_coverage=0.8, aggregation="max")

    assert gate.maximum_entropy == 1.0
    assert int(gate.retains(fit).sum()) == gate.selected_count == 10
    summary = EntropyGateSummary(
        method=gate.method,
        fit_role=SourceRole.DECISION_FIT,
        target_coverage=0.8,
        tie_rule="retain_all_scores_less_than_or_equal_to_frozen_order_statistic",
        maximum_entropy=gate.maximum_entropy,
        selected_count=gate.selected_count,
        fit_count=gate.fit_count,
        achieved_coverage=gate.achieved_coverage,
    )
    assert summary.method == "max_normalized_binary_entropy"
    assert summary.maximum_entropy == 1.0


@pytest.mark.parametrize("aggregation", ["median", "MAX", "", None, 1, True])
def test_entropy_gate_rejects_unknown_aggregations(aggregation: object) -> None:
    probabilities = np.full((4, 5), 0.25, dtype=np.float64)
    with pytest.raises(SourceCalibrationIntegrityError, match="aggregation"):
        fit_entropy_gate(
            probabilities,
            target_coverage=0.8,
            aggregation=aggregation,  # type: ignore[arg-type]
        )
    with pytest.raises(SourceCalibrationIntegrityError, match="aggregation"):
        FittedEntropyGate(
            target_coverage=0.8,
            maximum_entropy=0.5,
            selected_count=4,
            fit_count=4,
            aggregation=aggregation,  # type: ignore[arg-type]
        )


def test_default_source_components_and_summaries_are_unchanged(
    synthetic_project: SyntheticProject,
) -> None:
    config, verified, source = _loaded_source(synthetic_project)
    partitions = partition_source_predictions(source, config)
    default = fit_source_components(partitions, config)
    explicit = fit_source_components(partitions, config, entropy_aggregation="mean")

    assert default.entropy_aggregation == explicit.entropy_aggregation == "mean"
    assert default.summary == explicit.summary
    gate = default.summary.entropy_gate
    assert gate.method == "mean_normalized_binary_entropy"
    assert set(gate.model_dump(mode="json")) == {
        "method",
        "fit_role",
        "target_coverage",
        "tie_rule",
        "maximum_entropy",
        "selected_count",
        "fit_count",
        "achieved_coverage",
    }
    decision_probabilities = default.temperature.predict_proba(partitions.decision_fit.raw_logits)
    cutoff, selected = _pre_change_mean_gate(decision_probabilities, 0.8)
    assert (gate.maximum_entropy, gate.selected_count) == (cutoff, selected)
    assert default.entropy_cutoff == cutoff

    validation = evaluate_source_validation(partitions, default, config)
    validation_probabilities = default.temperature.predict_proba(
        partitions.source_validation.raw_logits
    )
    expected_retained = int(
        np.count_nonzero(normalized_bernoulli_entropy(validation_probabilities) <= cutoff)
    )
    assert validation.entropy_gate.selected_count == expected_retained
    assert set(validation.entropy_gate.model_dump(mode="json")) == {
        "frozen_component_sha256",
        "maximum_entropy",
        "selected_count",
        "validation_count",
        "achieved_coverage",
        "retained_hamming_loss",
        "retained_exact_match_accuracy",
    }
    result = build_source_calibration_result(
        config=config,
        source=source,
        verified=verified,
        config_file_sha256=load_source_calibration_config(synthetic_project.config_path)[1],
        code_revision=REVISION,
    )
    assert result.frozen_components == default.summary
    assert result.source_validation == validation


def test_worst_label_components_bind_fit_and_validation_to_one_aggregation(
    synthetic_project: SyntheticProject,
) -> None:
    config, verified, source = _loaded_source(synthetic_project)
    partitions = partition_source_predictions(source, config)
    mean = fit_source_components(partitions, config)
    worst = fit_source_components(partitions, config, entropy_aggregation="max")

    assert worst.entropy_aggregation == "max"
    assert worst.summary.entropy_gate.method == "max_normalized_binary_entropy"
    assert worst.entropy_cutoff == worst.summary.entropy_gate.maximum_entropy
    decision_probabilities = worst.temperature.predict_proba(partitions.decision_fit.raw_logits)
    assert (worst.entropy_cutoff, worst.summary.entropy_gate.selected_count) == (
        _order_statistic_gate(max_normalized_bernoulli_entropy(decision_probabilities), 0.8)
    )
    assert worst.entropy_cutoff != mean.entropy_cutoff
    assert worst.summary.temperature == mean.summary.temperature
    assert worst.summary.thresholds == mean.summary.thresholds
    assert worst.summary.conformal == mean.summary.conformal
    assert worst.summary.component_sha256 != mean.summary.component_sha256

    validation = evaluate_source_validation(partitions, worst, config)
    probabilities = worst.temperature.predict_proba(partitions.source_validation.raw_logits)
    expected_retained = int(
        np.count_nonzero(max_normalized_bernoulli_entropy(probabilities) <= worst.entropy_cutoff)
    )
    assert validation.entropy_gate.selected_count == expected_retained
    assert validation.entropy_gate.frozen_component_sha256 == worst.summary.component_sha256

    with pytest.raises(SourceCalibrationIntegrityError, match="aggregation"):
        replace(worst, entropy_aggregation="mean")
    with pytest.raises(SourceCalibrationIntegrityError, match="aggregation"):
        replace(mean, entropy_aggregation="max")
    with pytest.raises(SourceCalibrationIntegrityError, match="cutoff"):
        replace(mean, entropy_cutoff=worst.entropy_cutoff)
    with pytest.raises(SourceCalibrationIntegrityError, match="cutoff"):
        replace(worst, entropy_cutoff=mean.entropy_cutoff)
    with pytest.raises(SourceCalibrationIntegrityError, match="aggregation"):
        fit_source_components(
            partitions,
            config,
            entropy_aggregation="median",  # type: ignore[arg-type]
        )

    # Protocol v1 results keep accepting only the frozen mean gate.
    result = build_source_calibration_result(
        config=config,
        source=source,
        verified=verified,
        config_file_sha256=load_source_calibration_config(synthetic_project.config_path)[1],
        code_revision=REVISION,
    )
    payload = result.model_dump(mode="json")
    payload["frozen_components"] = worst.summary.model_dump(mode="json")
    payload["source_validation"] = validation.model_dump(mode="json")
    del payload["artifact_sha256"]
    payload["artifact_sha256"] = canonical_sha256(payload)
    with pytest.raises(ValidationError, match="frozen mean entropy gate"):
        SourceCalibrationResult.model_validate(payload)
    with pytest.raises(SourceCalibrationIntegrityError, match="schema"):
        load_source_calibration_result_bytes(canonical_json_bytes(payload) + b"\n")


def test_validation_and_conformal_roles_cannot_leak_into_other_fits(
    synthetic_project: SyntheticProject,
) -> None:
    config, _, source = _loaded_source(synthetic_project)
    partitions = partition_source_predictions(source, config)
    original = fit_source_components(partitions, config)
    with pytest.raises(SourceCalibrationIntegrityError):
        replace(
            partitions,
            decision_fit=partitions.conformal_and_ood_threshold_fit,
            conformal_and_ood_threshold_fit=partitions.decision_fit,
        )

    mutated_validation = replace_validation_role(
        partitions,
        raw_logits=np.asarray(-9.0 * partitions.source_validation.raw_logits + 3.0),
        targets=np.asarray(1 - partitions.source_validation.targets, dtype=np.int8),
    )
    after_validation_mutation = fit_source_components(mutated_validation, config)
    assert after_validation_mutation.summary == original.summary

    conformal_role = partitions.conformal_and_ood_threshold_fit
    offsets = np.linspace(-4.0, 4.0, conformal_role.records, dtype=np.float64)[:, None]
    mutated_conformal = replace(
        partitions,
        conformal_and_ood_threshold_fit=replace(
            conformal_role,
            raw_logits=np.asarray(conformal_role.raw_logits + offsets, dtype=np.float64),
        ),
    )
    after_conformal_mutation = fit_source_components(mutated_conformal, config)
    assert after_conformal_mutation.summary.temperature == original.summary.temperature
    assert after_conformal_mutation.summary.thresholds == original.summary.thresholds
    assert after_conformal_mutation.summary.entropy_gate == original.summary.entropy_gate
    assert after_conformal_mutation.summary.conformal != original.summary.conformal


def test_preparation_is_deterministic_private_and_explicitly_not_release_ready(
    synthetic_project: SyntheticProject,
) -> None:
    result = prepare_source_calibration(
        config_path=synthetic_project.config_path,
        project_root=synthetic_project.root,
        code_revision=REVISION,
    )
    result_path = synthetic_project.output_root / RESULT_FILENAME
    assert result_path.is_file()
    assert not list(synthetic_project.output_root.parent.glob(".*.staging-*"))
    loaded = load_source_calibration_result_bytes(result_path.read_bytes())
    assert loaded == result
    assert result.status == "PREPARED_NOT_RELEASE_READY"
    assert result.open_world.status == "PENDING"
    assert result.open_world.artifact_sha256 is None
    assert result.open_world.threshold_fitted is False
    assert result.open_world.source_false_rejection_evaluated is False
    assert result.open_world.embedding_device is None
    assert result.open_world.embedding_precision is None
    assert result.open_world.release_ready is False
    with pytest.raises(SourceCalibrationIntegrityError):
        assert_complete_release_ready(result)

    serialized = result_path.read_text(encoding="ascii")
    for forbidden in (
        '"patient_id"',
        '"ecg_id"',
        '"raw_logits"',
        '"targets"',
        '"path"',
        str(synthetic_project.root),
    ):
        assert forbidden not in serialized
    assert result.provenance.source_alignment_sha256.startswith("sha256:")
    assert (
        result.frozen_components.thresholds.tie_rule
        == "maximum_f1_then_closest_to_0.5_then_higher_threshold"
    )
    assert all(
        item.frozen_component_sha256 == result.frozen_components.component_sha256
        for item in (
            result.source_validation.threshold_decisions,
            result.source_validation.entropy_gate,
            result.source_validation.conformal,
        )
    )

    with pytest.raises(SourceCalibrationOutputError):
        prepare_source_calibration(
            config_path=synthetic_project.config_path,
            project_root=synthetic_project.root,
            code_revision=REVISION,
        )


def test_result_self_hash_and_canonical_bytes_fail_closed(
    synthetic_project: SyntheticProject,
) -> None:
    config, verified, source = _loaded_source(synthetic_project)
    config_hash = load_source_calibration_config(synthetic_project.config_path)[1]
    first = build_source_calibration_result(
        config=config,
        source=source,
        verified=verified,
        config_file_sha256=config_hash,
        code_revision=REVISION,
    )
    second = build_source_calibration_result(
        config=config,
        source=source,
        verified=verified,
        config_file_sha256=config_hash,
        code_revision=REVISION,
    )
    assert first.artifact_sha256 == second.artifact_sha256
    payload = first.model_dump(mode="json")
    validation = cast(dict[str, object], payload["source_validation"])
    validation["records"] = cast(int, validation["records"]) + 1
    tampered = (
        json.dumps(
            payload, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
        + "\n"
    ).encode("ascii")
    with pytest.raises(SourceCalibrationIntegrityError):
        load_source_calibration_result_bytes(tampered)


@pytest.mark.parametrize("malformation", ["extra_key", "wrong_dtype", "wrong_fold", "nonfinite"])
def test_safe_npz_rejects_exact_contract_violations(
    synthetic_project: SyntheticProject,
    malformation: str,
) -> None:
    with np.load(synthetic_project.npz_path, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]).copy() for name in loaded.files}
    if malformation == "extra_key":
        arrays["extra"] = np.asarray([1], dtype=np.int8)
    elif malformation == "wrong_dtype":
        arrays["targets"] = arrays["targets"].astype(np.int64)
    elif malformation == "wrong_fold":
        arrays["strat_fold"][:] = 8
    else:
        arrays["raw_logits"][0, 0] = np.nan
    with synthetic_project.npz_path.open("wb") as handle:
        if malformation == "extra_key":
            np.savez_compressed(
                handle,
                ecg_id=arrays["ecg_id"],
                patient_id=arrays["patient_id"],
                strat_fold=arrays["strat_fold"],
                targets=arrays["targets"],
                raw_logits=arrays["raw_logits"],
                extra=arrays["extra"],
            )
        else:
            np.savez_compressed(
                handle,
                ecg_id=arrays["ecg_id"],
                patient_id=arrays["patient_id"],
                strat_fold=arrays["strat_fold"],
                targets=arrays["targets"],
                raw_logits=arrays["raw_logits"],
            )
    config = _rewrite_npz_hash(synthetic_project)
    verified = verify_source_inputs(config, project_root=synthetic_project.root)
    with pytest.raises(SourceCalibrationIntegrityError):
        load_verified_source_predictions(config, verified)


@pytest.mark.parametrize("target", ["npz", "sidecar"])
def test_declared_hash_failure_happens_before_output_acquisition(
    synthetic_project: SyntheticProject,
    target: str,
) -> None:
    path = synthetic_project.npz_path if target == "npz" else synthetic_project.sidecar_path
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(SourceCalibrationIntegrityError):
        prepare_source_calibration(
            config_path=synthetic_project.config_path,
            project_root=synthetic_project.root,
            code_revision=REVISION,
        )
    assert not synthetic_project.output_root.exists()
    assert not list(synthetic_project.output_root.parent.glob(".*.staging-*"))


def test_post_commit_failure_receipt_is_sanitized(
    synthetic_project: SyntheticProject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = pipeline_module.load_source_calibration_result_bytes
    calls = 0

    def fail_after_commit(payload: bytes) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise SourceCalibrationIntegrityError("sensitive /absolute/path patient_id=123")
        return original(payload)

    monkeypatch.setattr(pipeline_module, "load_source_calibration_result_bytes", fail_after_commit)
    with pytest.raises(SourceCalibrationIntegrityError):
        prepare_source_calibration(
            config_path=synthetic_project.config_path,
            project_root=synthetic_project.root,
            code_revision=REVISION,
        )
    receipt_path = synthetic_project.output_root / FAILURE_RECEIPT_FILENAME
    assert receipt_path.is_file()
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    assert receipt["status"] == "FAILED"
    assert receipt["failure_code"] == "SOURCE_CONTRACT_FAILED"
    assert receipt["contains_raw_ids_or_rows"] is False
    serialized = json.dumps(receipt)
    assert "patient_id" not in serialized
    assert str(synthetic_project.root) not in serialized
    assert "sensitive" not in serialized


def test_clean_git_preflight_accepts_commit_and_rejects_dirty_tree(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("Git is unavailable")
    root = tmp_path / "repository"
    root.mkdir()

    def git(*arguments: str) -> None:
        subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init", "--quiet")
    git("config", "user.email", "source-calibration@example.invalid")
    git("config", "user.name", "Source Calibration Test")
    (root / "tracked.txt").write_text("frozen\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "--quiet", "-m", "frozen source")
    revision = verify_clean_git_revision(root)
    assert len(revision) == 40
    (root / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(SourceCalibrationIntegrityError):
        verify_clean_git_revision(root)
