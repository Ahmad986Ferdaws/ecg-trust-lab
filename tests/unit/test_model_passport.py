from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from typing import Literal, cast

import pytest
from pydantic import ValidationError

from ecg_trust.constants import SUPERCLASSES
from ecg_trust.passport import (
    AGGREGATE_METRIC_SCHEMA_VERSION,
    COHORT_EVIDENCE_SCHEMA_VERSION,
    CONFIDENCE_INTERVAL_SCHEMA_VERSION,
    CONFORMAL_EVIDENCE_SCHEMA_VERSION,
    CONFORMAL_SCOPE_TEXT,
    EXTERNAL_TRANSPORT_SCHEMA_VERSION,
    LABEL_PERFORMANCE_SCHEMA_VERSION,
    MACRO_PERFORMANCE_SCHEMA_VERSION,
    OOD_EVIDENCE_SCHEMA_VERSION,
    QUALITY_EVIDENCE_SCHEMA_VERSION,
    SELECTIVE_EVIDENCE_SCHEMA_VERSION,
    SUBGROUP_EVIDENCE_SCHEMA_VERSION,
    AggregateConfidenceInterval,
    AggregateMetric,
    CohortEvidence,
    CohortRole,
    ConformalEvidenceSummary,
    EvidenceStatus,
    ExternalTransportEvidence,
    LabelPerformance,
    MacroPerformance,
    MetricDirection,
    MinimumEvidenceStatus,
    ModelPassport,
    ModelPassportIntegrityError,
    ModelPassportPrivacyError,
    OODEvidenceSummary,
    QualityEvidenceSummary,
    SelectiveEvidenceSummary,
    SubgroupEvidence,
    SupportedInputContract,
    build_model_passport,
    model_passport_from_json_bytes,
    model_passport_to_json_bytes,
    render_model_passport_markdown,
    validate_model_passport,
)

NOW = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
TestSuperclass = Literal["NORM", "MI", "STTC", "CD", "HYP"]


def _hash(index: int) -> str:
    return "sha256:" + f"{index:064x}"


def _metric(
    metric_id: str,
    value: float = 0.9,
    *,
    status: EvidenceStatus = EvidenceStatus.AVAILABLE,
    sample_count: int = 100,
    interval: bool = False,
) -> AggregateMetric:
    confidence = None
    if interval:
        confidence = AggregateConfidenceInterval(
            schema_version=CONFIDENCE_INTERVAL_SCHEMA_VERSION,
            lower=value - 0.02,
            upper=value + 0.02,
            confidence_level=0.95,
            method="patient_bootstrap",
        )
    return AggregateMetric(
        schema_version=AGGREGATE_METRIC_SCHEMA_VERSION,
        metric_id=metric_id,
        display_name=metric_id.replace("_", " ").title(),
        definition=f"Aggregate definition for {metric_id}.",
        direction=MetricDirection.HIGHER_IS_BETTER,
        status=status,
        value=value if status is EvidenceStatus.AVAILABLE else None,
        unit="proportion",
        sample_count=sample_count,
        confidence_interval=confidence,
    )


def _cohorts() -> tuple[CohortEvidence, ...]:
    return (
        CohortEvidence(
            schema_version=COHORT_EVIDENCE_SCHEMA_VERSION,
            cohort_id="sph-external",
            dataset_name="SPH",
            dataset_version="frozen-evaluation-release",
            site_name="External source site",
            role=CohortRole.EXTERNAL_TRANSPORT,
            sample_count=500,
            patient_count=450,
            manifest_sha256=_hash(4),
            definition="Frozen no-adaptation external transport cohort.",
        ),
        CohortEvidence(
            schema_version=COHORT_EVIDENCE_SCHEMA_VERSION,
            cohort_id="ptbxl-fold10",
            dataset_name="PTB-XL",
            dataset_version="1.0.3",
            site_name="Source benchmark site",
            role=CohortRole.SEALED_TEST,
            sample_count=100,
            patient_count=90,
            manifest_sha256=_hash(3),
            definition="Protocol-defined sealed evaluation cohort.",
        ),
    )


def _label_performance() -> tuple[LabelPerformance, ...]:
    return tuple(
        LabelPerformance(
            schema_version=LABEL_PERFORMANCE_SCHEMA_VERSION,
            label=cast(TestSuperclass, label),
            discrimination=_metric("roc_auc", 0.9, interval=True),
            calibration=_metric("brier_score", 0.1),
        )
        for label in reversed(SUPERCLASSES)
    )


def _macro_performance() -> MacroPerformance:
    return MacroPerformance(
        schema_version=MACRO_PERFORMANCE_SCHEMA_VERSION,
        discrimination=_metric("macro_roc_auc", 0.91, interval=True),
        calibration=_metric("macro_brier_score", 0.11),
    )


def _subgroups() -> tuple[SubgroupEvidence, ...]:
    return (
        SubgroupEvidence(
            schema_version=SUBGROUP_EVIDENCE_SCHEMA_VERSION,
            evidence_id="sex-unknown",
            cohort_id="ptbxl-fold10",
            attribute="sex",
            group_name="Unknown",
            sample_count=4,
            patient_count=4,
            minimum_sample_count=20,
            minimum_patient_count=20,
            status=MinimumEvidenceStatus.INSUFFICIENT_EVIDENCE,
            metrics=(),
            reason_codes=("MINIMUM_EVIDENCE_NOT_MET",),
        ),
        SubgroupEvidence(
            schema_version=SUBGROUP_EVIDENCE_SCHEMA_VERSION,
            evidence_id="age-older",
            cohort_id="ptbxl-fold10",
            attribute="age_band",
            group_name="Older adults",
            sample_count=50,
            patient_count=45,
            minimum_sample_count=20,
            minimum_patient_count=20,
            status=MinimumEvidenceStatus.SUFFICIENT_EVIDENCE,
            metrics=(_metric("macro_roc_auc", 0.88, sample_count=50),),
            reason_codes=("MINIMUM_EVIDENCE_MET",),
        ),
    )


def _external_transport() -> ExternalTransportEvidence:
    return ExternalTransportEvidence(
        schema_version=EXTERNAL_TRANSPORT_SCHEMA_VERSION,
        evidence_id="sph-transport",
        status=EvidenceStatus.AVAILABLE,
        method="frozen_no_adaptation_transport",
        artifact_sha256=_hash(10),
        cohort_ids=("sph-external",),
        sample_count=500,
        patient_count=450,
        metrics=(_metric("macro_roc_auc", 0.84, sample_count=500),),
        summary="The frozen source model was evaluated without target adaptation.",
        limitations=("External label mapping limits direct diagnostic interpretation.",),
        frozen_source_model=True,
        target_adaptation="NONE",
    )


def _ood() -> OODEvidenceSummary:
    return OODEvidenceSummary(
        schema_version=OOD_EVIDENCE_SCHEMA_VERSION,
        evidence_id="mahalanobis-ood",
        status=EvidenceStatus.AVAILABLE,
        method="shrinkage_mahalanobis",
        artifact_sha256=_hash(11),
        cohort_ids=("ptbxl-fold10", "sph-external"),
        sample_count=600,
        patient_count=540,
        metrics=(_metric("ood_auroc", 0.78, sample_count=600),),
        summary="Aggregate source-versus-external OOD discrimination evidence.",
        limitations=("OOD score is not a probability of clinical abnormality.",),
        score_direction="HIGHER_IS_MORE_OUT_OF_DISTRIBUTION",
        threshold_scope="SOURCE_CALIBRATION_ONLY",
    )


def _quality() -> QualityEvidenceSummary:
    return QualityEvidenceSummary(
        schema_version=QUALITY_EVIDENCE_SCHEMA_VERSION,
        evidence_id="quality-audit",
        status=EvidenceStatus.AVAILABLE,
        method="deterministic_signal_quality",
        artifact_sha256=_hash(12),
        cohort_ids=("ptbxl-fold10",),
        sample_count=100,
        patient_count=90,
        metrics=(_metric("quality_pass_rate", 0.96),),
        summary="Aggregate signal-quality gate dispositions.",
        limitations=("Quality checks do not establish diagnostic validity.",),
        policy_scope="DETERMINISTIC_SIGNAL_QUALITY",
        frozen_policy=True,
    )


def _selective() -> SelectiveEvidenceSummary:
    return SelectiveEvidenceSummary(
        schema_version=SELECTIVE_EVIDENCE_SCHEMA_VERSION,
        evidence_id="selective-audit",
        status=EvidenceStatus.AVAILABLE,
        method="mean_normalized_binary_entropy",
        artifact_sha256=_hash(13),
        cohort_ids=("ptbxl-fold10",),
        sample_count=100,
        patient_count=90,
        metrics=(_metric("retained_fraction", 0.8),),
        summary="Aggregate retained coverage under the frozen abstention gate.",
        limitations=("Selective behavior is not a guarantee for an individual case.",),
        policy_scope="FROZEN_ABSTENTION_GATE",
        frozen_policy=True,
    )


def _conformal() -> ConformalEvidenceSummary:
    return ConformalEvidenceSummary(
        schema_version=CONFORMAL_EVIDENCE_SCHEMA_VERSION,
        evidence_id="conformal-audit",
        status=EvidenceStatus.AVAILABLE,
        method="labelwise_binary_split_conformal",
        artifact_sha256=_hash(14),
        cohort_ids=("ptbxl-fold10",),
        sample_count=100,
        patient_count=90,
        metrics=(
            _metric("marginal_coverage", 0.9),
            _metric("mean_set_size", 1.2),
        ),
        summary="Aggregate label-wise prediction-set evidence.",
        limitations=("The coverage statement is marginal, not simultaneous.",),
        coverage_scope="labelwise_marginal_under_exchangeability",
        coverage_scope_text=CONFORMAL_SCOPE_TEXT,
        calibration_scope="SOURCE_CALIBRATION_ONLY",
    )


def _passport(*, limitations: tuple[str, ...] | None = None) -> ModelPassport:
    return build_model_passport(
        passport_id="passport-release-001",
        generated_at=NOW,
        release_id="release-001",
        release_sha256=_hash(1),
        bundle_sha256=_hash(2),
        protocol_sha256=_hash(5),
        supported_input=SupportedInputContract.canonical(),
        cohorts=_cohorts(),
        label_performance=_label_performance(),
        macro_performance=_macro_performance(),
        subgroup_evidence=_subgroups(),
        external_transport=(_external_transport(),),
        ood_evidence=_ood(),
        quality_evidence=_quality(),
        selective_evidence=_selective(),
        conformal_evidence=_conformal(),
        limitations=(
            "This is retrospective research evidence, not clinical validation.",
            "Performance may not transport to unstudied sites or populations.",
        )
        if limitations is None
        else limitations,
    )


def test_builder_seals_complete_aggregate_only_passport_and_sorts_sets() -> None:
    passport = _passport()
    assert tuple(cohort.cohort_id for cohort in passport.cohorts) == (
        "ptbxl-fold10",
        "sph-external",
    )
    assert tuple(item.label for item in passport.label_performance) == SUPERCLASSES
    assert tuple(item.evidence_id for item in passport.subgroup_evidence) == (
        "age-older",
        "sex-unknown",
    )
    assert passport.research_only
    assert not passport.clinically_validated
    assert not passport.clinical_use_permitted
    assert passport.passport_sha256.startswith("sha256:")
    with pytest.raises(ValidationError, match="frozen"):
        passport.release_id = "changed"


def test_canonical_json_and_hash_are_deterministic_and_round_trip() -> None:
    first = _passport()
    second = _passport()
    assert first.passport_sha256 == second.passport_sha256
    payload = model_passport_to_json_bytes(first)
    assert payload == model_passport_to_json_bytes(second)

    restored = model_passport_from_json_bytes(
        payload,
        expected_release_sha256=_hash(1),
        expected_bundle_sha256=_hash(2),
        expected_protocol_sha256=_hash(5),
        expected_release_id="release-001",
    )
    assert restored == first

    pretty = json.dumps(first.model_dump(mode="json"), indent=2).encode() + b"\n"
    with pytest.raises(ModelPassportIntegrityError, match="not canonical"):
        model_passport_from_json_bytes(
            pretty,
            expected_release_sha256=_hash(1),
            expected_bundle_sha256=_hash(2),
            expected_protocol_sha256=_hash(5),
        )


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("expected_release_sha256", _hash(20)),
        ("expected_bundle_sha256", _hash(21)),
        ("expected_protocol_sha256", _hash(22)),
        ("expected_release_id", "release-other"),
    ],
)
def test_validator_fails_closed_on_release_identity_mismatch(field: str, expected: str) -> None:
    arguments: dict[str, str] = {
        "expected_release_sha256": _hash(1),
        "expected_bundle_sha256": _hash(2),
        "expected_protocol_sha256": _hash(5),
        "expected_release_id": "release-001",
    }
    arguments[field] = expected
    with pytest.raises(ModelPassportIntegrityError, match="differs from expectation"):
        validate_model_passport(_passport(), **arguments)


def test_tampered_self_hash_and_unknown_fields_are_rejected() -> None:
    passport = _passport()
    payload = passport.model_dump(mode="python")
    with pytest.raises(ValidationError, match="passport_sha256"):
        ModelPassport.model_validate({**payload, "release_id": "release-tampered"})
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelPassport.model_validate({**payload, "unknown": "field"})
    with pytest.raises(ValidationError, match="model_passport.v1"):
        ModelPassport.model_validate({**payload, "schema_version": "v2"})


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("patient_ids", ("patient-1",), "forbidden row-level"),
        ("record_ids", ("record-1",), "forbidden row-level"),
        ("raw_signals", ((0.1, 0.2),), "forbidden row-level"),
        ("per_record_scores", (0.1, 0.2), "forbidden row-level"),
    ],
)
def test_privacy_boundary_rejects_identifiers_and_row_arrays(
    field: str, value: object, match: str
) -> None:
    payload = _passport().model_dump(mode="python")
    with pytest.raises(ModelPassportPrivacyError, match=match):
        ModelPassport.model_validate({**payload, field: value})


@pytest.mark.parametrize(
    ("limitation", "match"),
    [
        ("C:\\private\\cohort.csv", "absolute path"),
        ("/srv/private/cohort.csv", "absolute path"),
        ("Evidence was read from C:\\private\\cohort.csv", "absolute path"),
        ("Evidence was read from /srv/private/cohort.csv", "absolute path"),
        ("Bearer abcdefghijklmnopqrstuvwxyz", "secret-like"),
        ("-----BEGIN PRIVATE KEY-----", "secret-like"),
    ],
)
def test_privacy_boundary_rejects_paths_and_secret_material(limitation: str, match: str) -> None:
    with pytest.raises(ModelPassportPrivacyError, match=match):
        _passport(limitations=(limitation,))


def test_metric_and_subgroup_minimum_evidence_semantics_are_fail_closed() -> None:
    with pytest.raises(ValidationError, match="finite number"):
        _metric("macro_roc_auc", float("nan"))
    with pytest.raises(ValidationError, match="cannot publish"):
        AggregateMetric.model_validate(
            {
                **_metric("macro_roc_auc").model_dump(mode="python"),
                "status": EvidenceStatus.NOT_EVALUATED,
            }
        )
    sufficient = _subgroups()[1]
    with pytest.raises(ValidationError, match="suppress metrics"):
        SubgroupEvidence.model_validate(
            {
                **sufficient.model_dump(mode="python"),
                "sample_count": 1,
                "patient_count": 1,
                "status": MinimumEvidenceStatus.INSUFFICIENT_EVIDENCE,
                "reason_codes": ("MINIMUM_EVIDENCE_NOT_MET",),
            }
        )
    with pytest.raises(ValidationError, match="minimum-evidence rule"):
        SubgroupEvidence.model_validate(
            {
                **sufficient.model_dump(mode="python"),
                "status": MinimumEvidenceStatus.INSUFFICIENT_EVIDENCE,
            }
        )


def test_scalar_types_input_shape_and_utc_timestamp_are_strict() -> None:
    cohort_payload = _cohorts()[0].model_dump(mode="python")
    with pytest.raises(ValidationError, match="valid integer"):
        CohortEvidence.model_validate({**cohort_payload, "sample_count": "500"})
    with pytest.raises(ValidationError, match="canonical 12-lead order"):
        SupportedInputContract.model_validate(
            {
                **SupportedInputContract.canonical().model_dump(mode="python"),
                "lead_order": tuple(reversed(SupportedInputContract.canonical().lead_order)),
            }
        )
    with pytest.raises(ValidationError, match="generated_at must use UTC"):
        build_model_passport(
            passport_id="passport-non-utc",
            generated_at=NOW.astimezone(timezone(timedelta(hours=1))),
            release_id="release-001",
            release_sha256=_hash(1),
            bundle_sha256=_hash(2),
            protocol_sha256=_hash(5),
            supported_input=SupportedInputContract.canonical(),
            cohorts=_cohorts(),
            label_performance=_label_performance(),
            macro_performance=_macro_performance(),
            subgroup_evidence=_subgroups(),
            external_transport=(_external_transport(),),
            ood_evidence=_ood(),
            quality_evidence=_quality(),
            selective_evidence=_selective(),
            conformal_evidence=_conformal(),
            limitations=("Research only; no clinical validation.",),
        )


def test_revalidation_blocks_constructed_nested_model_with_private_path() -> None:
    safe = _cohorts()[0]
    constructed = CohortEvidence.model_construct(
        **{
            **safe.model_dump(mode="python"),
            "definition": "Aggregate evidence stored at C:\\private\\cohort.csv",
        }
    )
    with pytest.raises(ModelPassportPrivacyError, match="absolute path"):
        build_model_passport(
            passport_id="passport-constructed-private-value",
            generated_at=NOW,
            release_id="release-001",
            release_sha256=_hash(1),
            bundle_sha256=_hash(2),
            protocol_sha256=_hash(5),
            supported_input=SupportedInputContract.canonical(),
            cohorts=(constructed, _cohorts()[1]),
            label_performance=_label_performance(),
            macro_performance=_macro_performance(),
            subgroup_evidence=_subgroups(),
            external_transport=(_external_transport(),),
            ood_evidence=_ood(),
            quality_evidence=_quality(),
            selective_evidence=_selective(),
            conformal_evidence=_conformal(),
            limitations=("Research only; no clinical validation.",),
        )


def test_external_transport_must_bind_an_external_cohort() -> None:
    source_only_transport = ExternalTransportEvidence.model_validate(
        {
            **_external_transport().model_dump(mode="python"),
            "cohort_ids": ("ptbxl-fold10",),
        }
    )
    with pytest.raises(ValidationError, match="EXTERNAL_TRANSPORT cohort"):
        build_model_passport(
            passport_id="passport-source-only-transport",
            generated_at=NOW,
            release_id="release-001",
            release_sha256=_hash(1),
            bundle_sha256=_hash(2),
            protocol_sha256=_hash(5),
            supported_input=SupportedInputContract.canonical(),
            cohorts=_cohorts(),
            label_performance=_label_performance(),
            macro_performance=_macro_performance(),
            subgroup_evidence=_subgroups(),
            external_transport=(source_only_transport,),
            ood_evidence=_ood(),
            quality_evidence=_quality(),
            selective_evidence=_selective(),
            conformal_evidence=_conformal(),
            limitations=("Research only; no clinical validation.",),
        )


def test_evidence_references_must_resolve_to_declared_cohorts() -> None:
    conformal = ConformalEvidenceSummary.model_validate(
        {
            **_conformal().model_dump(mode="python"),
            "cohort_ids": ("undeclared-site",),
        }
    )
    with pytest.raises(ValidationError, match="undeclared cohorts"):
        build_model_passport(
            passport_id="passport-invalid-reference",
            generated_at=NOW,
            release_id="release-001",
            release_sha256=_hash(1),
            bundle_sha256=_hash(2),
            protocol_sha256=_hash(5),
            supported_input=SupportedInputContract.canonical(),
            cohorts=_cohorts(),
            label_performance=_label_performance(),
            macro_performance=_macro_performance(),
            subgroup_evidence=_subgroups(),
            external_transport=(_external_transport(),),
            ood_evidence=_ood(),
            quality_evidence=_quality(),
            selective_evidence=_selective(),
            conformal_evidence=conformal,
            limitations=("Research only; no clinical validation.",),
        )


def test_renderer_is_deterministic_escaped_and_explicit_about_safety() -> None:
    passport = _passport(
        limitations=(
            "Known | limitation in retrospective evidence.",
            "This is not clinical validation.",
        )
    )
    first = render_model_passport_markdown(passport)
    second = render_model_passport_markdown(passport)
    assert first == second
    assert "## Dataset and site evidence" in first
    assert "## External transport" in first
    assert "## Open-world / OOD evidence" in first
    assert "## Signal-quality evidence" in first
    assert "## Selective-prediction evidence" in first
    assert "## Conformal evidence" in first
    assert passport.passport_sha256 in first
    assert "Known \\| limitation" in first
    assert "not an individual certainty guarantee" in first
    assert "Clinically validated: `false`" in first
    assert "Clinical use permitted: `false`" in first
    assert "patient_id" not in first
    assert "raw_signal" not in first
    assert first.endswith("\n")
