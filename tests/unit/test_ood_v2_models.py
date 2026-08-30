from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import ecg_trust.ood_v2.models as contract_models
from ecg_trust.ood_v2 import (
    OOD_V2_ARTIFACT_TYPE,
    OOD_V2_PARENT_CONFIG_SHA256,
    OOD_V2_PROTOCOL_ID,
    OOD_V2_RESULT_FILENAME,
    AggregateRouteCounts,
    EvidenceRequirements,
    ExternalCohortSummary,
    ExternalOODHardGates,
    HistoricalSourceBootstrapInterval,
    OODV2IntegrityError,
    OODV2IntegritySummary,
    OODV2Result,
    OODV2ResultBody,
    OODV2Status,
    ResamplingUnit,
    SourceGateSummary,
    TechnicalQualityEndpointSummary,
    assert_aggregate_only_ood_v2_result,
    canonical_sha256,
    evaluate_external_ood_gate,
    evaluate_source_gate,
    evaluate_technical_quality_gate,
    load_ood_v2_result_bytes,
    ood_v2_result_json_bytes,
    seal_ood_v2_result,
)

NOW = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
REVISION = "1" * 40


def _digest(character: str) -> str:
    return "sha256:" + character * 64


@lru_cache(maxsize=1)
def _requirements() -> EvidenceRequirements:
    return EvidenceRequirements(
        family_wise_alpha=0.05,
        multiplicity_method="bonferroni",
        co_primary_endpoint_count=4,
        one_sided_alpha_per_endpoint=0.0125,
        co_primary_confidence_level=0.9875,
        bootstrap_replicates=10_000,
        challenge_bootstrap_seed=20_260_901,
        zzu_bootstrap_seed=20_260_902,
    )


def _integrity(*, complete: bool = True) -> OODV2IntegritySummary:
    return OODV2IntegritySummary(
        preregistration_frozen_before_evaluation=complete,
        cohort_roles_frozen_before_model_outputs=complete,
        dataset_hashes_verified=complete,
        overlap_exclusions_verified=complete,
        frozen_detector_verified=complete,
        evaluation_alignment_verified=complete,
        aggregate_only_result_verified=complete,
        sealed_v1_unchanged_verified=complete,
        sealed_v1_source_validation_used_for_tuning=False,
        target_site_fitting_performed=False,
        complete=complete,
    )


@lru_cache(maxsize=1)
def _historical_v1_source() -> SourceGateSummary:
    interval = HistoricalSourceBootstrapInterval(
        method="historical_patient_cluster_percentile_bootstrap",
        estimator="record_weighted_event_rate",
        resampling_unit=ResamplingUnit.PATIENT_CLUSTER,
        sampling_with_replacement=True,
        random_generator="numpy.random.Generator_PCG64",
        seed=20_260_829,
        replicates=10_000,
        percentile_function="numpy.quantile",
        quantile_method="linear",
        confidence_level=0.95,
        records=465,
        resampling_units=409,
        event_count=25,
        point_estimate=25 / 465,
        two_sided_lower=0.03239566265733224,
        two_sided_upper=0.07725321888412018,
        one_sided_upper=0.07296137339055794,
        one_sided_lower_published=False,
    )
    return SourceGateSummary(
        cohort_key="sealed-v1-source-validation",
        cohort_manifest_sha256=(
            "sha256:87992206fcbfc2b091d8f8dd08998a5d9bae3d55a2d2056f1ab674a316b0675b"
        ),
        evaluation_role="source_retention",
        records=465,
        subjects=409,
        rejected_records=25,
        retained_records=440,
        false_rejection_rate=25 / 465,
        support_coverage=440 / 465,
        maximum_false_rejection_rate=0.05,
        interval=interval,
        gate_passed=False,
        sealed_v1_source_validation_used_for_tuning=False,
        public_contains_record_level_outputs=False,
    )


@lru_cache(maxsize=4)
def _external(kind: str, *, missed: bool = False) -> ExternalCohortSummary:
    detected = (
        np.asarray([True] * 150 + [False] * 50, dtype=np.bool_)
        if missed
        else np.ones(200, dtype=np.bool_)
    )
    if kind == "challenge":
        return evaluate_external_ood_gate(
            detected,
            endpoint_key="challenge_external_distribution_recall",
            cohort_key="physionet-challenge-2011-set-a",
            dataset_name="PhysioNet Challenge 2011 Set A",
            dataset_version="1.0.0",
            license_identifier="ODC-By-1.0",
            cohort_manifest_sha256=_digest("b"),
            role_assignment_sha256=_digest("2"),
            evaluation_role="physionet_challenge_2011_set_a",
            ood_axis="external_acquisition_and_population_domain",
            resampling_unit="record",
            minimum_ood_recall=0.90,
            seed=20_260_901,
            replicates=10_000,
            confidence_level=0.9875,
        )
    if kind != "zzu":
        raise ValueError("unknown external test cohort")
    return evaluate_external_ood_gate(
        detected,
        endpoint_key="zzu_external_distribution_recall",
        cohort_key="zzu-pecg-v1",
        dataset_name="ZZU pediatric ECG",
        dataset_version="1",
        license_identifier="CC-BY-4.0",
        cohort_manifest_sha256=_digest("d"),
        role_assignment_sha256=_digest("2"),
        evaluation_role="zzu_pecg_v1",
        ood_axis="pediatric_population_and_external_acquisition_domain",
        resampling_unit="patient_cluster",
        cluster_labels=np.arange(1, 201, dtype=np.int64),
        subjects=200,
        minimum_ood_recall=0.90,
        seed=20_260_902,
        replicates=10_000,
        confidence_level=0.9875,
    )


@lru_cache(maxsize=4)
def _technical(kind: str, *, missed: bool = False) -> TechnicalQualityEndpointSummary:
    events = (
        np.asarray([True] * 150 + [False] * 50, dtype=np.bool_)
        if missed
        else np.ones(200, dtype=np.bool_)
    )
    if kind == "group3":
        endpoint_key = "challenge_group3_technical_block_sensitivity"
        definition = "block_unacceptable"
        minimum = 0.95
    elif kind == "group1":
        endpoint_key = "challenge_group1_quality_pass_rate"
        definition = "pass_acceptable"
        minimum = 0.90
    else:
        raise ValueError("unknown technical test endpoint")
    return evaluate_technical_quality_gate(
        events,
        endpoint_key=endpoint_key,
        cohort_key="physionet-challenge-2011-set-a",
        event_definition=definition,
        resampling_unit="record",
        minimum_rate=minimum,
        seed=20_260_901,
        replicates=10_000,
        confidence_level=0.9875,
    )


def _hard_gates(*, missed: bool = False) -> ExternalOODHardGates:
    return ExternalOODHardGates(
        challenge_reference_label_alignment_complete=True,
        challenge_invalid_input_count=0,
        challenge_quality_pass_records=200,
        zzu_invalid_input_count=0,
        zzu_selected_records=200,
        zzu_quality_pass_records=200,
        zzu_quality_pass_record_coverage=1.0,
        zzu_selected_patients=200,
        zzu_quality_pass_patients=200,
        zzu_quality_pass_patient_coverage=1.0,
        challenge_group3_prediction_allowed_count=0,
        skipped_selected_records=1 if missed else 0,
        target_site_fitting_performed=False,
        v1_policy_bytes_unchanged_before_and_after=True,
        exact_v1_whole_bundle_verifier_passes=True,
        external_raw_sources_verified_before_and_after=True,
        exact_dataset_roots_verified=True,
        exact_selected_input_inventory_verified_before_and_after=True,
        semantic_roles_rederived_before_and_after=True,
        raw_canonical_lead_and_data_file_bindings_verified=True,
        active_scientific_package_versions_match_child=True,
        deterministic_repeated_embeddings_match=True,
        raw_source_to_canonical_signal_replay_matches=True,
        canonical_signal_to_full_backbone_embedding_replay_matches=True,
        aggregate_only_publication_verified=True,
        immutable_success_bundle_verifies=True,
        failure_receipt_exists=False,
        all_passed=not missed,
    )


def _body(
    *,
    status: OODV2Status | str = OODV2Status.EXTERNAL_OOD_EVIDENCE_COMPLETE,
    external: tuple[ExternalCohortSummary, ...] | None = None,
    technical: tuple[TechnicalQualityEndpointSummary, ...] | None = None,
    hard_gates: ExternalOODHardGates | None = None,
    integrity: OODV2IntegritySummary | None = None,
    external_evidence_eligible: bool = True,
) -> OODV2ResultBody:
    return OODV2ResultBody(
        schema_version=1,
        artifact_type="ecg_trust.ood_external_v2_1_result",
        protocol_id="trust-sentinel-ood-external-v2-1-parent",
        frozen_at_utc=NOW,
        status=OODV2Status(status),
        preregistration_sha256=OOD_V2_PARENT_CONFIG_SHA256,
        cohort_role_manifest_sha256=_digest("2"),
        detector_policy_sha256=(
            "sha256:817d6e5c4a3058c064cdc7bdceafb774c7ea4bb0b6cf725be1b8f12c7aae9c1c"
        ),
        sealed_v1_result_sha256=(
            "sha256:844bbe7f2a85b229f553cd12df14f7db712b9e0090fe6fd6823319a557777c12"
        ),
        sealed_v1_claim_sha256=(
            "sha256:956c16e6d9ce4575274f040e44a822e7c8952b98642cc243f165a262f1b5a2f8"
        ),
        code_revision=REVISION,
        evidence_requirements=_requirements(),
        source_gate=_historical_v1_source(),
        external_cohorts=(
            (_external("challenge"), _external("zzu")) if external is None else external
        ),
        technical_quality_endpoints=(
            (_technical("group3"), _technical("group1"))
            if technical is None
            else technical
        ),
        final_route_counts=AggregateRouteCounts(
            INVALID_INPUT=0,
            REACQUIRE=0,
            UNSUPPORTED_INPUT=13_328,
            ABSTAIN=0,
            PREDICTION_ALLOWED=0,
            total_records=13_328,
        ),
        hard_gates=_hard_gates() if hard_gates is None else hard_gates,
        integrity=_integrity() if integrity is None else integrity,
        external_evidence_eligible=external_evidence_eligible,
        integration_permitted=False,
        aggregate_only=True,
        research_only=True,
        clinical_validation=False,
    )


@pytest.fixture(scope="module")
def complete_result() -> OODV2Result:
    return seal_ood_v2_result(_body())


def test_constants_match_authoritative_frozen_parent() -> None:
    assert OOD_V2_PROTOCOL_ID == "trust-sentinel-ood-external-v2-1-parent"
    assert OOD_V2_ARTIFACT_TYPE == "ecg_trust.ood_external_v2_1_result"
    assert OOD_V2_RESULT_FILENAME == "ood-external-v2-1-result.json"
    assert OOD_V2_PARENT_CONFIG_SHA256 == (
        "sha256:9b0358be1d4a12ca1771c57d8387c1b332bbef5698e01d3da2707f59157a586c"
    )
    parent_bytes = Path(
        "configs/trust_sentinel_ood_external_v2_1.yaml"
    ).read_bytes()
    assert "sha256:" + hashlib.sha256(parent_bytes).hexdigest() == (
        OOD_V2_PARENT_CONFIG_SHA256
    )


def test_external_evidence_can_complete_while_historical_source_blocks_integration(
    complete_result: OODV2Result,
) -> None:
    assert complete_result.status is OODV2Status.EXTERNAL_OOD_EVIDENCE_COMPLETE
    assert complete_result.external_evidence_eligible is True
    assert complete_result.source_gate.gate_passed is False
    assert isinstance(complete_result.source_gate.interval, HistoricalSourceBootstrapInterval)
    assert complete_result.integration_permitted is False


def test_historical_v1_context_preserves_only_published_bounds() -> None:
    source = _historical_v1_source()
    interval = source.interval
    assert isinstance(interval, HistoricalSourceBootstrapInterval)
    assert interval.one_sided_lower_published is False
    assert interval.one_sided_upper == pytest.approx(0.07296137339055794)
    assert source.false_rejection_rate == 25 / 465


@pytest.mark.parametrize("confidence_level", [0.949, "0.95", float("nan")])
def test_historical_source_confidence_is_strict_finite_and_exact(
    confidence_level: object,
) -> None:
    interval = _historical_v1_source().interval
    assert isinstance(interval, HistoricalSourceBootstrapInterval)
    payload = interval.model_dump(mode="python")
    payload["confidence_level"] = confidence_level

    with pytest.raises(ValidationError):
        HistoricalSourceBootstrapInterval.model_validate(payload)


def test_nonhistorical_source_summary_cannot_be_substituted() -> None:
    labels = np.concatenate(
        (np.arange(1, 410, dtype=np.int64), np.arange(1, 57, dtype=np.int64))
    )
    rejected = np.asarray([True] * 25 + [False] * 440, dtype=np.bool_)
    nonhistorical = evaluate_source_gate(
        rejected,
        cohort_key="sealed-v1-source-validation",
        cohort_manifest_sha256=_digest("f"),
        resampling_unit="patient_cluster",
        cluster_labels=labels,
        subjects=409,
        maximum_false_rejection_rate=0.05,
        seed=20_260_829,
        replicates=1_000,
        confidence_level=0.95,
    )
    payload = _body().model_dump(mode="python")
    payload["source_gate"] = nonhistorical.model_dump(mode="python")

    with pytest.raises(ValidationError, match="historical"):
        OODV2ResultBody.model_validate(payload)


def test_historical_source_split_assignment_digest_is_exact() -> None:
    payload = _body().model_dump(mode="python")
    payload["source_gate"]["cohort_manifest_sha256"] = _digest("9")

    with pytest.raises(ValidationError, match="published v1 aggregate"):
        OODV2ResultBody.model_validate(payload)


@pytest.mark.parametrize("missed_kind", ["external", "technical", "hard"])
def test_any_complete_external_or_hard_gate_miss_is_completed_unfavorable_evidence(
    missed_kind: str,
) -> None:
    external = (
        _external("challenge", missed=missed_kind == "external"),
        _external("zzu"),
    )
    technical = (
        _technical("group3", missed=missed_kind == "technical"),
        _technical("group1"),
    )
    result = seal_ood_v2_result(
        _body(
            status="EXTERNAL_OOD_TARGET_MISSED",
            external=external,
            technical=technical,
            hard_gates=_hard_gates(missed=missed_kind == "hard"),
            external_evidence_eligible=False,
        )
    )

    assert result.status is OODV2Status.EXTERNAL_OOD_TARGET_MISSED
    assert result.external_evidence_eligible is False
    assert result.integration_permitted is False


@pytest.mark.parametrize("missing_kind", ["external", "technical", "integrity"])
def test_missing_evidence_fails_closed_as_insufficient(missing_kind: str) -> None:
    external = (
        (_external("challenge"),)
        if missing_kind == "external"
        else (_external("challenge"), _external("zzu"))
    )
    technical = (
        (_technical("group3"),)
        if missing_kind == "technical"
        else (_technical("group3"), _technical("group1"))
    )
    result = seal_ood_v2_result(
        _body(
            status="EXTERNAL_OOD_INSUFFICIENT_EVIDENCE",
            external=external,
            technical=technical,
            integrity=_integrity(complete=False) if missing_kind == "integrity" else _integrity(),
            external_evidence_eligible=False,
        )
    )

    assert result.status is OODV2Status.EXTERNAL_OOD_INSUFFICIENT_EVIDENCE
    assert result.external_evidence_eligible is False


def test_status_and_external_eligibility_cannot_be_self_asserted() -> None:
    payload = _body().model_dump(mode="python")
    payload["status"] = "EXTERNAL_OOD_TARGET_MISSED"
    payload["external_evidence_eligible"] = False
    with pytest.raises(ValidationError, match="status"):
        OODV2ResultBody.model_validate(payload)

    payload = _body().model_dump(mode="python")
    payload["external_evidence_eligible"] = False
    with pytest.raises(ValidationError, match="external_evidence_eligible"):
        OODV2ResultBody.model_validate(payload)

    payload = _body().model_dump(mode="python")
    payload["integration_permitted"] = True
    with pytest.raises(ValidationError):
        OODV2ResultBody.model_validate(payload)


def test_parent_file_hash_is_exactly_bound() -> None:
    payload = _body().model_dump(mode="python")
    payload["preregistration_sha256"] = _digest("9")

    with pytest.raises(ValidationError, match="frozen parent file"):
        OODV2ResultBody.model_validate(payload)

    payload = _body().model_dump(mode="python")
    payload["detector_policy_sha256"] = _digest("9")
    with pytest.raises(ValidationError, match="v1 file bindings"):
        OODV2ResultBody.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("family_wise_alpha", 0.04),
        ("co_primary_endpoint_count", 3),
        ("one_sided_alpha_per_endpoint", 0.025),
        ("co_primary_confidence_level", 0.95),
        ("bootstrap_replicates", 9_999),
        ("challenge_bootstrap_seed", 1),
        ("zzu_bootstrap_seed", 2),
    ],
)
def test_frozen_multiplicity_and_bootstrap_requirements_cannot_be_weakened(
    field: str,
    value: object,
) -> None:
    payload = _requirements().model_dump(mode="python")
    payload[field] = value

    with pytest.raises(ValidationError, match="frozen parent"):
        EvidenceRequirements.model_validate(payload)


def test_endpoint_identity_threshold_resampling_and_seed_are_frozen() -> None:
    payload = _body().model_dump(mode="python")
    payload["external_cohorts"][0]["interval"]["seed"] = 1
    with pytest.raises(ValidationError, match="external endpoint"):
        OODV2ResultBody.model_validate(payload)

    payload = _body().model_dump(mode="python")
    payload["technical_quality_endpoints"][0]["minimum_rate"] = 0.90
    payload["technical_quality_endpoints"][0]["gate_passed"] = True
    with pytest.raises(ValidationError, match="technical endpoint"):
        OODV2ResultBody.model_validate(payload)

    payload = _body().model_dump(mode="python")
    payload["external_cohorts"][0]["endpoint_key"] = "unregistered-endpoint"
    with pytest.raises(ValidationError, match="not declared"):
        OODV2ResultBody.model_validate(payload)


@pytest.mark.parametrize(
    ("summary", "subjects"),
    [
        (_external("challenge"), 199),
        (_external("zzu"), 199),
        (_technical("group3"), 199),
    ],
)
def test_endpoint_subjects_match_the_declared_resampling_units(
    summary: ExternalCohortSummary | TechnicalQualityEndpointSummary,
    subjects: int,
) -> None:
    payload = summary.model_dump(mode="python")
    payload["subjects"] = subjects

    with pytest.raises(ValidationError, match="subjects"):
        type(summary).model_validate(payload)


def test_hard_gate_all_passed_flag_is_derived() -> None:
    payload = _hard_gates().model_dump(mode="python")
    payload["skipped_selected_records"] = 1

    with pytest.raises(ValidationError, match="all_passed"):
        ExternalOODHardGates.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("zzu_invalid_input_count", 1),
        ("challenge_quality_pass_records", 0),
        ("challenge_reference_label_alignment_complete", False),
        ("exact_v1_whole_bundle_verifier_passes", False),
        ("external_raw_sources_verified_before_and_after", False),
        ("semantic_roles_rederived_before_and_after", False),
        ("raw_canonical_lead_and_data_file_bindings_verified", False),
        ("active_scientific_package_versions_match_child", False),
        ("raw_source_to_canonical_signal_replay_matches", False),
        ("canonical_signal_to_full_backbone_embedding_replay_matches", False),
    ],
)
def test_successor_integrity_hard_gates_fail_closed(field: str, value: object) -> None:
    payload = _hard_gates().model_dump(mode="python")
    payload[field] = value

    with pytest.raises(ValidationError, match="all_passed"):
        ExternalOODHardGates.model_validate(payload)


def test_zzu_quality_pass_coverage_is_count_derived_and_at_least_eighty_percent() -> None:
    payload = _hard_gates().model_dump(mode="python")
    payload["zzu_quality_pass_records"] = 159
    payload["zzu_quality_pass_record_coverage"] = 159 / 200
    with pytest.raises(ValidationError, match="all_passed"):
        ExternalOODHardGates.model_validate(payload)

    payload = _hard_gates().model_dump(mode="python")
    payload["zzu_quality_pass_record_coverage"] = 0.99
    with pytest.raises(ValidationError, match="coverage differs"):
        ExternalOODHardGates.model_validate(payload)


def test_result_is_frozen_strict_and_forbids_unknown_fields(
    complete_result: OODV2Result,
) -> None:
    with pytest.raises(ValidationError):
        complete_result.external_evidence_eligible = False

    payload = complete_result.model_dump(mode="python")
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        OODV2Result.model_validate(payload)

    requirements = _requirements().model_dump(mode="python")
    requirements["family_wise_alpha"] = "0.05"
    with pytest.raises(ValidationError):
        EvidenceRequirements.model_validate(requirements)


def test_external_endpoint_metadata_and_child_role_hash_are_exactly_cross_linked() -> None:
    payload = _body().model_dump(mode="python")
    payload["external_cohorts"][0]["dataset_version"] = "latest"
    with pytest.raises(ValidationError, match="external endpoint"):
        OODV2ResultBody.model_validate(payload)

    payload = _body().model_dump(mode="python")
    payload["external_cohorts"][1]["role_assignment_sha256"] = _digest("9")
    with pytest.raises(ValidationError, match="role assignment"):
        OODV2ResultBody.model_validate(payload)


def test_canonical_round_trip_hash_and_tamper_detection(
    complete_result: OODV2Result,
) -> None:
    payload = ood_v2_result_json_bytes(complete_result)

    assert payload.endswith(b"\n") and b"\r" not in payload
    assert load_ood_v2_result_bytes(payload) == complete_result
    assert complete_result.artifact_sha256 == canonical_sha256(
        complete_result.model_dump(mode="json", exclude={"artifact_sha256"})
    )

    tampered = json.loads(payload)
    tampered["external_evidence_eligible"] = False
    tampered_payload = contract_models.canonical_json_bytes(tampered) + b"\n"
    with pytest.raises(OODV2IntegrityError):
        load_ood_v2_result_bytes(tampered_payload)


def test_loader_rejects_noncanonical_and_duplicate_json_keys(
    complete_result: OODV2Result,
) -> None:
    payload = ood_v2_result_json_bytes(complete_result)
    pretty = json.dumps(json.loads(payload), indent=2).encode() + b"\n"
    with pytest.raises(OODV2IntegrityError, match="canonical"):
        load_ood_v2_result_bytes(pretty)

    duplicate = payload.replace(
        b'"aggregate_only":true',
        b'"aggregate_only":true,"aggregate_only":true',
        1,
    )
    with pytest.raises(OODV2IntegrityError, match="duplicate"):
        load_ood_v2_result_bytes(duplicate)

    with pytest.raises(OODV2IntegrityError, match="trailing newline"):
        load_ood_v2_result_bytes(payload + b"\n")


def test_public_result_has_no_row_identifiers_paths_or_private_arrays(
    complete_result: OODV2Result,
) -> None:
    assert_aggregate_only_ood_v2_result(complete_result)
    serialized = complete_result.model_dump(mode="json")

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value).union(*(keys(child) for child in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(child) for child in value))
        return set()

    forbidden = {
        "patient_id",
        "subject_id",
        "record_id",
        "ecg_id",
        "rows",
        "waveforms",
        "embeddings",
        "scores",
        "predictions",
        "file_path",
    }
    assert forbidden.isdisjoint(keys(serialized))

    external = _external("challenge").model_dump(mode="python")
    external["dataset_name"] = r"C:\private\external-ecg"
    with pytest.raises(ValidationError, match="paths"):
        contract_models.ExternalCohortSummary.model_validate(external)


def test_v1_binding_and_no_tuning_flags_cannot_be_relaxed() -> None:
    payload = _body().model_dump(mode="python")
    payload["source_gate"]["sealed_v1_source_validation_used_for_tuning"] = True
    with pytest.raises(ValidationError):
        OODV2ResultBody.model_validate(payload)

    payload = _body().model_dump(mode="python")
    payload["integrity"]["sealed_v1_unchanged_verified"] = False
    payload["integrity"]["complete"] = False
    payload["status"] = "EXTERNAL_OOD_INSUFFICIENT_EVIDENCE"
    payload["external_evidence_eligible"] = False
    result = OODV2ResultBody.model_validate(payload)
    assert result.status is OODV2Status.EXTERNAL_OOD_INSUFFICIENT_EVIDENCE


def test_timestamp_must_be_utc_and_endpoint_keys_must_be_unique() -> None:
    payload = _body().model_dump(mode="python")
    payload["frozen_at_utc"] = NOW.astimezone(timezone(timedelta(hours=-5)))
    with pytest.raises(ValidationError, match="UTC"):
        OODV2ResultBody.model_validate(payload)

    payload = _body().model_dump(mode="python")
    payload["external_cohorts"][1]["endpoint_key"] = payload["external_cohorts"][0][
        "endpoint_key"
    ]
    with pytest.raises(ValidationError, match="unique"):
        OODV2ResultBody.model_validate(payload)


def test_loader_enforces_artifact_size_bound(
    complete_result: OODV2Result,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = ood_v2_result_json_bytes(complete_result)
    monkeypatch.setattr(contract_models, "MAX_OOD_V2_RESULT_BYTES", len(payload) - 1)

    with pytest.raises(OODV2IntegrityError, match="byte length"):
        load_ood_v2_result_bytes(payload)
