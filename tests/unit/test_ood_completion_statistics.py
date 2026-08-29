from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ecg_trust.ood_completion.embedding_artifact import (
    EmbeddingRole,
    create_embedding_artifact,
    load_embedding_artifact,
    save_embedding_artifact,
)
from ecg_trust.ood_completion.models import (
    OODLineageProvenance,
    distribution_policy_json_bytes,
)
from ecg_trust.ood_completion.statistics import (
    evaluate_source_validation,
    fit_distribution_policy,
    patient_cluster_bootstrap_interval,
    score_quantiles,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _provenance() -> OODLineageProvenance:
    return OODLineageProvenance(
        ood_config_file_sha256=_digest("1"),
        source_calibration_artifact_sha256=_digest("2"),
        source_calibration_file_sha256=_digest("3"),
        source_calibration_config_file_sha256=_digest("4"),
        refit_completion_artifact_sha256=_digest("5"),
        refit_completion_file_sha256=_digest("6"),
        checkpoint_file_sha256=_digest("7"),
        resolved_config_sha256=_digest("8"),
        resolved_config_file_sha256=_digest("9"),
        dataset_manifest_file_sha256=_digest("a"),
        normalization_file_sha256=_digest("b"),
        experiment_protocol_sha256=_digest("c"),
        environment_lock_file_sha256=_digest("d"),
        project_manifest_file_sha256=(
            "sha256:e1de755829678d588784bbcc34becc8c031c742d9b3f05458e76e67f577da3cd"
        ),
        raw_checksum_inventory_file_sha256=_digest("e"),
        raw_selected_inventory_sha256=_digest("2"),
        selected_record_count=18383,
        selected_file_count=36766,
        raw_reference_inventory_sha256=_digest("f"),
        raw_source_inventory_sha256=_digest("0"),
        code_revision="1" * 40,
        model_member_id="resnet1d-seed2026",
        architecture="resnet1d",
        seed=2026,
    )


def test_score_quantiles_use_the_frozen_linear_definition() -> None:
    scores = np.asarray([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float64)

    observed = score_quantiles(scores)

    assert observed.min == 0.0
    assert observed.p50 == 20.0
    assert observed.p90 == pytest.approx(36.0)
    assert observed.p95 == pytest.approx(38.0)
    assert observed.p99 == pytest.approx(39.6)
    assert observed.max == 40.0


def test_patient_cluster_bootstrap_is_exactly_reproducible() -> None:
    patients = np.asarray([1, 1, 2, 3, 3, 3], dtype=np.int64)
    rejected = np.asarray([False, True, False, True, False, False], dtype=np.bool_)

    first = patient_cluster_bootstrap_interval(patients, rejected)
    second = patient_cluster_bootstrap_interval(patients, rejected)

    assert first == second
    assert first.seed == 20_260_829
    assert first.replicates == 10_000
    assert first.two_sided_lower <= 2 / 6 <= first.two_sided_upper
    assert first.one_sided_upper >= 2 / 6


def test_threshold_role_can_change_only_threshold_not_reference_parameters() -> None:
    generator = np.random.Generator(np.random.PCG64(71))
    reference = generator.normal(size=(520, 512)).astype(np.float32)
    threshold_fit = generator.normal(size=(60, 512)).astype(np.float32)
    shifted_threshold_fit = threshold_fit + np.float32(4.0)

    first, first_summary = fit_distribution_policy(
        reference,
        threshold_fit,
        provenance=_provenance(),
    )
    shifted, shifted_summary = fit_distribution_policy(
        reference,
        shifted_threshold_fit,
        provenance=_provenance(),
    )

    assert first.detector.mean == shifted.detector.mean
    assert first.detector.precision == shifted.detector.precision
    assert first.detector.threshold != shifted.detector.threshold
    assert first_summary.threshold == first.detector.threshold
    assert shifted_summary.threshold == shifted.detector.threshold
    assert first.detector.quantile_rank == 58


def test_validation_evaluation_cannot_mutate_or_refit_sealed_policy(tmp_path: Path) -> None:
    destination_root = tmp_path
    generator = np.random.Generator(np.random.PCG64(79))
    reference = generator.normal(size=(520, 512)).astype(np.float32)
    threshold_fit = generator.normal(size=(80, 512)).astype(np.float32)
    policy, _ = fit_distribution_policy(
        reference,
        threshold_fit,
        provenance=_provenance(),
    )
    before = distribution_policy_json_bytes(policy)

    validation = generator.normal(size=(465, 512)).astype(np.float32)
    patient_id = np.concatenate(
        (
            np.arange(1, 410, dtype=np.int64),
            np.arange(1, 57, dtype=np.int64),
        )
    )
    private = create_embedding_artifact(
        ecg_id=np.arange(1, 466, dtype=np.int64),
        patient_id=patient_id,
        strat_fold=np.full(465, 9, dtype=np.int8),
        embedding=validation,
        role=EmbeddingRole.SOURCE_VALIDATION,
        expected_folds=(9,),
        checkpoint_sha256=_digest("7"),
        config_sha256=_digest("1"),
        normalization_sha256=_digest("b"),
        manifest_sha256=_digest("a"),
        protocol_sha256=_digest("c"),
        runtime_sha256=_digest("e"),
    )
    files = save_embedding_artifact(private, destination_root / "validation.npz")
    sealed = load_embedding_artifact(files.npz_path)

    summary = evaluate_source_validation(
        sealed,
        repeated_embedding_tensor_sha256=sealed.embedding_tensor_sha256,
        policy=policy,
        source_assignment_sha256=_digest("f"),
    )

    assert distribution_policy_json_bytes(policy) == before
    assert summary.records == 465
    assert summary.patients == 409
    assert summary.threshold == policy.detector.threshold
    assert summary.embedding_artifact_sha256 == files.artifact_sha256
    assert summary.exact_repeat_verified is True


@pytest.mark.parametrize(
    ("patients", "rejected", "match"),
    [
        ([1, 2], [False], "align"),
        ([1, 2], [0, 1], "boolean"),
        ([1, 0], [False, True], "positive"),
        ([], [], "integer"),
    ],
)
def test_bootstrap_rejects_invalid_arrays(
    patients: Any,
    rejected: Any,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        patient_cluster_bootstrap_interval(patients, rejected)
