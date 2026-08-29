"""Frozen fitting and source-rejection statistics for OOD completion v1.

The fitting API accepts only the reference and threshold-fit embeddings.  The
source-validation API accepts an already sealed distribution policy, which
makes accidental validation-driven refitting structurally difficult.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ecg_trust.ood_completion.embedding_artifact import (
    EMBEDDING_DIMENSION,
    EmbeddingArtifact,
    EmbeddingRole,
)
from ecg_trust.ood_completion.models import (
    DistributionPolicy,
    DistributionPolicyBody,
    EmbeddingContract,
    MahalanobisDetectorArtifact,
    OODLineageProvenance,
    PatientClusterBootstrapInterval,
    ScoreQuantiles,
    SourceOODValidationSummary,
    ThresholdFitSummary,
    seal_distribution_policy,
)
from ecg_trust.open_world import ShrinkageMahalanobisDetector

Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

_SHRINKAGE = 0.1
_RIDGE = 0.000001
_INLIER_COVERAGE = 0.95
_BOOTSTRAP_REPLICATES = 10_000
_BOOTSTRAP_SEED = 20_260_829


def score_quantiles(scores: ArrayLike) -> ScoreQuantiles:
    """Summarize finite nonnegative scores with NumPy's frozen linear method."""

    values = _score_vector(scores)
    probabilities = np.asarray(
        [0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0],
        dtype=np.float64,
    )
    quantiles = np.quantile(values, probabilities, method="linear")
    return ScoreQuantiles(
        min=float(quantiles[0]),
        p01=float(quantiles[1]),
        p05=float(quantiles[2]),
        p25=float(quantiles[3]),
        p50=float(quantiles[4]),
        p75=float(quantiles[5]),
        p90=float(quantiles[6]),
        p95=float(quantiles[7]),
        p99=float(quantiles[8]),
        max=float(quantiles[9]),
    )


def fit_distribution_policy(
    reference_embeddings: ArrayLike,
    threshold_fit_embeddings: ArrayLike,
    *,
    provenance: OODLineageProvenance,
) -> tuple[DistributionPolicy, ThresholdFitSummary]:
    """Fit the v1 detector from R and its threshold from B—never from C."""

    reference = _embedding_matrix(reference_embeddings, context="reference embeddings")
    threshold_fit = _embedding_matrix(
        threshold_fit_embeddings,
        context="threshold-fit embeddings",
    )
    if reference.shape[1] != threshold_fit.shape[1]:
        raise ValueError("reference and threshold-fit embedding dimensions differ")
    if reference.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(f"OOD completion v1 requires {EMBEDDING_DIMENSION} embedding values")

    detector = ShrinkageMahalanobisDetector.fit(
        reference,
        threshold_fit,
        shrinkage=_SHRINKAGE,
        ridge=_RIDGE,
        inlier_coverage=_INLIER_COVERAGE,
    )
    threshold_scores = detector.score(threshold_fit)
    rejected = threshold_scores > detector.threshold
    rejected_count = int(np.count_nonzero(rejected))
    accepted_count = int(threshold_scores.shape[0] - rejected_count)
    summary = ThresholdFitSummary(
        method="shrinkage_mahalanobis_embedding_distance",
        fit_role="conformal_and_ood_threshold_fit",
        n_reference_samples=int(reference.shape[0]),
        n_threshold_samples=int(threshold_fit.shape[0]),
        embedding_dimension=512,
        shrinkage=_SHRINKAGE,
        ridge=_RIDGE,
        target_inlier_coverage=_INLIER_COVERAGE,
        quantile_rank=detector.quantile_rank,
        threshold=detector.threshold,
        threshold_comparison="score_strictly_greater_than_threshold",
        observed_inlier_count=accepted_count,
        observed_rejection_count=rejected_count,
        observed_inlier_fraction=accepted_count / threshold_fit.shape[0],
        observed_false_rejection_rate=rejected_count / threshold_fit.shape[0],
        score_quantiles=score_quantiles(threshold_scores),
    )
    policy = seal_distribution_policy(
        DistributionPolicyBody(
            schema_version=1,
            artifact_type="ecg_trust.distribution_policy",
            protocol_id="trust-sentinel-ood-completion-v1",
            method="shrinkage_mahalanobis_embedding_distance",
            score_direction="higher_is_more_out_of_distribution",
            threshold_comparison="score_strictly_greater_than_threshold",
            embedding_contract=EmbeddingContract(
                architecture="resnet1d",
                extraction_point="frozen_resnet_preclassifier_global_average_pool",
                pooling="adaptive_average_pool_1d",
                embedding_dimension=512,
                tensor_dtype="float32",
                detector_numeric_dtype="float64",
                lead_order=(
                    "I",
                    "II",
                    "III",
                    "aVR",
                    "aVL",
                    "aVF",
                    "V1",
                    "V2",
                    "V3",
                    "V4",
                    "V5",
                    "V6",
                ),
                sampling_frequency_hz=100.0,
                samples_per_lead=1000,
                physical_units="mV",
            ),
            detector=MahalanobisDetectorArtifact.from_detector(detector),
            provenance=provenance,
            research_only=True,
        )
    )
    return policy, summary


def patient_cluster_bootstrap_interval(
    patient_id: ArrayLike,
    rejected: ArrayLike,
) -> PatientClusterBootstrapInterval:
    """Return the preregistered 10,000-replicate patient-cluster interval.

    Each replicate samples the observed patients with replacement, carries all
    records belonging to every sampled patient, and then recomputes the
    record-weighted false-rejection rate.  Percentiles use NumPy ``linear``
    interpolation.
    """

    patients, rejected_values = _aligned_patient_rejections(patient_id, rejected)
    unique_patients, inverse = np.unique(patients, return_inverse=True)
    patient_count = int(unique_patients.shape[0])
    records_per_patient = np.bincount(inverse, minlength=patient_count).astype(
        np.int64,
        copy=False,
    )
    rejected_per_patient = np.bincount(
        inverse,
        weights=rejected_values.astype(np.int64),
        minlength=patient_count,
    ).astype(np.int64, copy=False)

    generator = np.random.Generator(np.random.PCG64(_BOOTSTRAP_SEED))
    sampled = generator.integers(
        0,
        patient_count,
        size=(_BOOTSTRAP_REPLICATES, patient_count),
        endpoint=False,
    )
    denominators = records_per_patient[sampled].sum(axis=1, dtype=np.int64)
    numerators = rejected_per_patient[sampled].sum(axis=1, dtype=np.int64)
    if np.any(denominators <= 0):  # pragma: no cover - guaranteed by validated inputs
        raise ValueError("bootstrap produced an empty patient replicate")
    rates = numerators.astype(np.float64) / denominators.astype(np.float64)
    lower, upper, one_sided_upper = np.quantile(
        rates,
        np.asarray([0.025, 0.975, 0.95], dtype=np.float64),
        method="linear",
    )
    return PatientClusterBootstrapInterval(
        method="patient_cluster_percentile_bootstrap",
        estimator="record_false_rejection_rate",
        resampling_unit="patient",
        sampling_with_replacement=True,
        seed=20_260_829,
        replicates=10_000,
        two_sided_confidence_level=0.95,
        two_sided_lower=float(lower),
        two_sided_upper=float(upper),
        one_sided_confidence_level=0.95,
        one_sided_upper=float(one_sided_upper),
    )


def evaluate_source_validation(
    artifact: EmbeddingArtifact,
    *,
    repeated_embedding_tensor_sha256: str,
    policy: DistributionPolicy,
    source_assignment_sha256: str,
) -> SourceOODValidationSummary:
    """Evaluate a sealed policy once on the private C-role embedding artifact."""

    if not isinstance(artifact, EmbeddingArtifact):
        raise TypeError("artifact must be an EmbeddingArtifact")
    if artifact.role is not EmbeddingRole.SOURCE_VALIDATION:
        raise ValueError("source evaluation requires the SOURCE_VALIDATION artifact")
    if artifact.artifact_sha256 is None:
        raise ValueError("source-validation embedding artifact must already be sealed")
    if artifact.npz_file_sha256 is None or artifact.sidecar_file_sha256 is None:
        raise ValueError("source-validation embedding files must already be sealed")
    repeat_hash = _normalize_sha256(
        repeated_embedding_tensor_sha256,
        context="repeated embedding tensor hash",
    )
    if repeat_hash != artifact.embedding_tensor_sha256:
        raise ValueError("source-validation embedding repeat does not match the sealed artifact")

    scores = policy.to_detector().score(artifact.embedding)
    threshold = policy.detector.threshold
    rejected = scores > threshold
    rejected_count = int(np.count_nonzero(rejected))
    accepted_count = artifact.record_count - rejected_count
    unique_patients, inverse = np.unique(artifact.patient_id, return_inverse=True)
    patient_count = int(unique_patients.shape[0])
    per_patient_rates = np.empty(patient_count, dtype=np.float64)
    patient_any = np.empty(patient_count, dtype=np.bool_)
    for index in range(patient_count):
        mask = inverse == index
        per_patient_rates[index] = float(np.mean(rejected[mask], dtype=np.float64))
        patient_any[index] = bool(np.any(rejected[mask]))
    record_rate = rejected_count / artifact.record_count
    interval = patient_cluster_bootstrap_interval(artifact.patient_id, rejected)
    return SourceOODValidationSummary(
        records=artifact.record_count,
        patients=artifact.patient_count,
        embedding_dimension=512,
        alignment_sha256=artifact.alignment_sha256,
        embedding_tensor_sha256=artifact.embedding_tensor_sha256,
        repeated_embedding_tensor_sha256=repeat_hash,
        embedding_artifact_sha256=artifact.artifact_sha256,
        embedding_npz_file_sha256=artifact.npz_file_sha256,
        embedding_sidecar_file_sha256=artifact.sidecar_file_sha256,
        runtime_sha256=artifact.runtime_sha256,
        exact_repeat_verified=True,
        public_contains_row_arrays=False,
        evaluation_role="source_validation",
        tuning_allowed=False,
        rejected_records=rejected_count,
        accepted_records=accepted_count,
        rejected_patients_any=int(np.count_nonzero(patient_any)),
        record_false_rejection_rate=record_rate,
        source_record_support_coverage=accepted_count / artifact.record_count,
        patient_equalized_false_rejection_rate=float(per_patient_rates.mean()),
        patient_any_false_rejection_rate=float(patient_any.mean(dtype=np.float64)),
        maximum_allowed_record_false_rejection_rate=0.05,
        cluster_bootstrap=interval,
        score_quantiles=score_quantiles(scores),
        threshold=threshold,
        threshold_tie_count=int(np.count_nonzero(scores == threshold)),
        threshold_comparison="score_strictly_greater_than_threshold",
        source_assignment_sha256=_normalize_sha256(
            source_assignment_sha256,
            context="source assignment hash",
        ),
        target_met=interval.one_sided_upper <= 0.05,
    )


def _embedding_matrix(values: ArrayLike, *, context: str) -> Float64Array:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be numeric") from error
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{context} must contain a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{context} must contain only finite values")
    return matrix


def _score_vector(values: ArrayLike) -> Float64Array:
    try:
        scores = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("scores must be numeric") from error
    if scores.ndim != 1 or scores.shape[0] == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.0):
        raise ValueError("scores must be finite and nonnegative")
    return scores


def _aligned_patient_rejections(
    patient_id: ArrayLike,
    rejected: ArrayLike,
) -> tuple[Int64Array, BoolArray]:
    raw_patients = np.asarray(patient_id)
    raw_rejected = np.asarray(rejected)
    if raw_patients.ndim != 1 or raw_patients.dtype.kind not in {"i", "u"}:
        raise ValueError("patient_id must be a one-dimensional integer array")
    if raw_patients.shape[0] == 0 or np.any(raw_patients <= 0):
        raise ValueError("patient_id must contain positive identifiers")
    if raw_rejected.ndim != 1 or raw_rejected.dtype != np.dtype(np.bool_):
        raise ValueError("rejected must be a one-dimensional boolean array")
    if raw_rejected.shape != raw_patients.shape:
        raise ValueError("patient_id and rejected must align one-to-one")
    if raw_patients.dtype.kind == "u" and int(raw_patients.max()) > np.iinfo(np.int64).max:
        raise ValueError("patient_id contains a value outside int64 range")
    return (
        cast(Int64Array, raw_patients.astype(np.int64, copy=False)),
        cast(BoolArray, raw_rejected),
    )


def _normalize_sha256(value: str, *, context: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{context} must be a prefixed SHA-256 digest")
    suffix = value[7:]
    if any(character not in "0123456789abcdef" for character in suffix):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


__all__ = [
    "evaluate_source_validation",
    "fit_distribution_policy",
    "patient_cluster_bootstrap_interval",
    "score_quantiles",
]
