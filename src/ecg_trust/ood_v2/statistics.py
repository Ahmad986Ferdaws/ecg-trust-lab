"""Deterministic binary-rate uncertainty and gates for open-world v2."""

from __future__ import annotations

from decimal import Decimal
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ecg_trust.ood_v2.models import (
    ExternalCohortRole,
    ExternalCohortSummary,
    OODAxis,
    ProportionBootstrapInterval,
    ResamplingUnit,
    SourceGateSummary,
    TechnicalQualityEndpointSummary,
    TechnicalQualityEventDefinition,
)

BoolArray = NDArray[np.bool_]
Int64Array = NDArray[np.int64]
Float64Array = NDArray[np.float64]

DEFAULT_BOOTSTRAP_SEED = 20_260_829
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_CONFIDENCE_LEVEL = 0.95
_MAX_SAMPLED_INDICES_PER_CHUNK = 1_000_000


def bootstrap_proportion_interval(
    events: ArrayLike,
    *,
    resampling_unit: ResamplingUnit | str,
    cluster_labels: ArrayLike | None = None,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> ProportionBootstrapInterval:
    """Estimate deterministic percentile intervals for a binary record rate.

    ``events`` must be a strict one-dimensional boolean vector. With
    ``patient_cluster`` resampling, ``cluster_labels`` must be aligned positive
    integer labels; each sampled patient carries all their records. With
    ``record`` resampling, labels are forbidden and each replicate draws
    ``n`` record indices with replacement exactly as preregistered. Sampling is
    chunked only to bound memory; the PCG64 draw stream and record order are
    unchanged.

    The two-sided interval uses alpha/2 and 1-alpha/2 percentiles. The one-sided
    interval uses alpha and 1-alpha percentiles, where alpha is
    ``1 - confidence_level``. NumPy's ``linear`` quantile definition is frozen
    in the returned contract.
    """

    event_values = _boolean_vector(events)
    unit = _resampling_unit(resampling_unit)
    _validate_bootstrap_parameters(
        seed=seed,
        replicates=replicates,
        confidence_level=confidence_level,
    )
    records = int(event_values.shape[0])
    event_count = int(np.count_nonzero(event_values))
    generator = np.random.Generator(np.random.PCG64(seed))

    if unit is ResamplingUnit.RECORD:
        if cluster_labels is not None:
            raise ValueError("cluster_labels must be omitted for record-level resampling")
        rates = _record_bootstrap_rates(
            events=event_values,
            generator=generator,
            replicates=replicates,
        )
        resampling_units = records
    else:
        labels = _cluster_vector(cluster_labels, expected_records=records)
        unique_labels, inverse = np.unique(labels, return_inverse=True)
        resampling_units = int(unique_labels.shape[0])
        records_per_cluster = np.bincount(
            inverse,
            minlength=resampling_units,
        ).astype(np.int64, copy=False)
        events_per_cluster = np.bincount(
            inverse,
            weights=event_values.astype(np.int64),
            minlength=resampling_units,
        ).astype(np.int64, copy=False)
        rates = _cluster_bootstrap_rates(
            records_per_cluster=records_per_cluster,
            events_per_cluster=events_per_cluster,
            generator=generator,
            replicates=replicates,
        )

    # The protocol freezes decimal tail probabilities (not the result of a
    # binary-float subtraction).  In particular, 1 - 0.9875 must be the exact
    # float literal 0.0125 used by the preregistered quantile contract.
    alpha = float(Decimal("1") - Decimal(str(confidence_level)))
    quantiles = np.quantile(
        rates,
        np.asarray(
            [alpha / 2.0, 1.0 - alpha / 2.0, alpha, 1.0 - alpha],
            dtype=np.float64,
        ),
        method="linear",
    )
    return ProportionBootstrapInterval(
        method="percentile_bootstrap",
        estimator="record_weighted_event_rate",
        resampling_unit=unit,
        sampling_with_replacement=True,
        random_generator="numpy.random.Generator_PCG64",
        seed=seed,
        replicates=replicates,
        percentile_function="numpy.quantile",
        quantile_method="linear",
        confidence_level=confidence_level,
        records=records,
        resampling_units=resampling_units,
        event_count=event_count,
        point_estimate=event_count / records,
        two_sided_lower=float(quantiles[0]),
        two_sided_upper=float(quantiles[1]),
        one_sided_lower=float(quantiles[2]),
        one_sided_upper=float(quantiles[3]),
    )


def evaluate_source_gate(
    rejected: ArrayLike,
    *,
    cohort_key: str,
    cohort_manifest_sha256: str,
    resampling_unit: ResamplingUnit | str,
    cluster_labels: ArrayLike | None = None,
    subjects: int | None = None,
    maximum_false_rejection_rate: float = 0.05,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> SourceGateSummary:
    """Build the source gate from rejected/not-rejected record indicators."""

    interval = bootstrap_proportion_interval(
        rejected,
        resampling_unit=resampling_unit,
        cluster_labels=cluster_labels,
        seed=seed,
        replicates=replicates,
        confidence_level=confidence_level,
    )
    subject_count = _subject_count(subjects, interval=interval)
    rejected_count = interval.event_count
    retained_count = interval.records - rejected_count
    return SourceGateSummary(
        cohort_key=cohort_key,
        cohort_manifest_sha256=cohort_manifest_sha256,
        evaluation_role="source_retention",
        records=interval.records,
        subjects=subject_count,
        rejected_records=rejected_count,
        retained_records=retained_count,
        false_rejection_rate=interval.point_estimate,
        support_coverage=retained_count / interval.records,
        maximum_false_rejection_rate=maximum_false_rejection_rate,
        interval=interval,
        gate_passed=interval.one_sided_upper <= maximum_false_rejection_rate,
        sealed_v1_source_validation_used_for_tuning=False,
        public_contains_record_level_outputs=False,
    )


def evaluate_external_ood_gate(
    detected: ArrayLike,
    *,
    endpoint_key: str,
    cohort_key: str,
    dataset_name: str,
    dataset_version: str,
    license_identifier: str,
    cohort_manifest_sha256: str,
    role_assignment_sha256: str,
    evaluation_role: ExternalCohortRole | str,
    ood_axis: OODAxis | str,
    resampling_unit: ResamplingUnit | str,
    cluster_labels: ArrayLike | None = None,
    subjects: int | None = None,
    minimum_ood_recall: float = 0.90,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> ExternalCohortSummary:
    """Build an OOD gate from detected/not-detected OOD-positive indicators."""

    interval = bootstrap_proportion_interval(
        detected,
        resampling_unit=resampling_unit,
        cluster_labels=cluster_labels,
        seed=seed,
        replicates=replicates,
        confidence_level=confidence_level,
    )
    subject_count = _subject_count(subjects, interval=interval)
    detected_count = interval.event_count
    missed_count = interval.records - detected_count
    return ExternalCohortSummary(
        endpoint_key=endpoint_key,
        cohort_key=cohort_key,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        license_identifier=license_identifier,
        cohort_manifest_sha256=cohort_manifest_sha256,
        role_assignment_sha256=role_assignment_sha256,
        evaluation_role=_enum_value(
            ExternalCohortRole,
            evaluation_role,
            context="evaluation_role",
        ),
        ood_axis=_enum_value(OODAxis, ood_axis, context="ood_axis"),
        records=interval.records,
        subjects=subject_count,
        detected_records=detected_count,
        missed_records=missed_count,
        ood_recall=interval.point_estimate,
        minimum_ood_recall=minimum_ood_recall,
        interval=interval,
        gate_passed=interval.one_sided_lower >= minimum_ood_recall,
        target_site_fitting_performed=False,
        public_contains_record_level_outputs=False,
    )


def evaluate_technical_quality_gate(
    events: ArrayLike,
    *,
    endpoint_key: str,
    cohort_key: str,
    event_definition: TechnicalQualityEventDefinition | str,
    resampling_unit: ResamplingUnit | str,
    cluster_labels: ArrayLike | None = None,
    subjects: int | None = None,
    minimum_rate: float,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> TechnicalQualityEndpointSummary:
    """Build one preregistered technical-quality co-primary endpoint."""

    interval = bootstrap_proportion_interval(
        events,
        resampling_unit=resampling_unit,
        cluster_labels=cluster_labels,
        seed=seed,
        replicates=replicates,
        confidence_level=confidence_level,
    )
    subject_count = _subject_count(subjects, interval=interval)
    return TechnicalQualityEndpointSummary(
        endpoint_key=endpoint_key,
        cohort_key=cohort_key,
        event_definition=_enum_value(
            TechnicalQualityEventDefinition,
            event_definition,
            context="event_definition",
        ),
        records=interval.records,
        subjects=subject_count,
        events=interval.event_count,
        non_events=interval.records - interval.event_count,
        point_rate=interval.point_estimate,
        minimum_rate=minimum_rate,
        interval=interval,
        gate_passed=interval.one_sided_lower >= minimum_rate,
        public_contains_record_level_outputs=False,
    )


def _cluster_bootstrap_rates(
    *,
    records_per_cluster: Int64Array,
    events_per_cluster: Int64Array,
    generator: np.random.Generator,
    replicates: int,
) -> Float64Array:
    clusters = int(records_per_cluster.shape[0])
    chunk_size = max(1, min(replicates, _MAX_SAMPLED_INDICES_PER_CHUNK // clusters))
    rates = np.empty(replicates, dtype=np.float64)
    start = 0
    while start < replicates:
        stop = min(replicates, start + chunk_size)
        sampled = generator.integers(
            0,
            clusters,
            size=(stop - start, clusters),
            endpoint=False,
        )
        denominators = records_per_cluster[sampled].sum(axis=1, dtype=np.int64)
        numerators = events_per_cluster[sampled].sum(axis=1, dtype=np.int64)
        if np.any(denominators <= 0):  # pragma: no cover - validated nonempty clusters
            raise ValueError("bootstrap produced an empty cluster replicate")
        rates[start:stop] = numerators.astype(np.float64) / denominators.astype(np.float64)
        start = stop
    return rates


def _record_bootstrap_rates(
    *,
    events: BoolArray,
    generator: np.random.Generator,
    replicates: int,
) -> Float64Array:
    """Draw records with replacement using the frozen PCG64 index rule."""

    records = int(events.shape[0])
    chunk_size = max(1, min(replicates, _MAX_SAMPLED_INDICES_PER_CHUNK // records))
    rates = np.empty(replicates, dtype=np.float64)
    start = 0
    while start < replicates:
        stop = min(replicates, start + chunk_size)
        sampled = generator.integers(
            0,
            records,
            size=(stop - start, records),
            endpoint=False,
        )
        rates[start:stop] = events[sampled].mean(axis=1, dtype=np.float64)
        start = stop
    return rates


def _boolean_vector(values: ArrayLike) -> BoolArray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.shape[0] == 0:
        raise ValueError("events must be a non-empty one-dimensional boolean array")
    if raw.dtype != np.dtype(np.bool_):
        raise ValueError("events must be a strict boolean array")
    return cast(BoolArray, raw)


def _cluster_vector(values: ArrayLike | None, *, expected_records: int) -> Int64Array:
    if values is None:
        raise ValueError("cluster_labels are required for patient-cluster resampling")
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in {"i", "u"}:
        raise ValueError("cluster_labels must be a one-dimensional integer array")
    if raw.shape[0] != expected_records:
        raise ValueError("cluster_labels and events must align one-to-one")
    if np.any(raw <= 0):
        raise ValueError("cluster_labels must contain positive private identifiers")
    if raw.dtype.kind == "u" and int(raw.max()) > np.iinfo(np.int64).max:
        raise ValueError("cluster_labels contain a value outside int64 range")
    return cast(Int64Array, raw.astype(np.int64, copy=False))


def _resampling_unit(value: ResamplingUnit | str) -> ResamplingUnit:
    try:
        return ResamplingUnit(value)
    except (TypeError, ValueError) as error:
        raise ValueError("resampling_unit must be 'patient_cluster' or 'record'") from error


def _enum_value[EnumType: (ExternalCohortRole, OODAxis, TechnicalQualityEventDefinition)](
    enum_type: type[EnumType],
    value: EnumType | str,
    *,
    context: str,
) -> EnumType:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} is invalid") from error


def _validate_bootstrap_parameters(
    *,
    seed: int,
    replicates: int,
    confidence_level: float,
) -> None:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(replicates) is not int or replicates < 1_000:
        raise ValueError("replicates must be an integer of at least 1000")
    if type(confidence_level) is not float or not 0.5 <= confidence_level < 1.0:
        raise ValueError("confidence_level must be a float in [0.5, 1.0)")


def _subject_count(
    declared: int | None,
    *,
    interval: ProportionBootstrapInterval,
) -> int:
    if declared is None:
        return interval.resampling_units
    if type(declared) is not int or declared < 1 or declared > interval.records:
        raise ValueError("subjects must be a positive integer no greater than records")
    if (
        interval.resampling_unit is ResamplingUnit.PATIENT_CLUSTER
        and declared != interval.resampling_units
    ):
        raise ValueError("subjects must equal the number of patient clusters")
    return declared


__all__ = [
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CONFIDENCE_LEVEL",
    "bootstrap_proportion_interval",
    "evaluate_external_ood_gate",
    "evaluate_source_gate",
    "evaluate_technical_quality_gate",
]
