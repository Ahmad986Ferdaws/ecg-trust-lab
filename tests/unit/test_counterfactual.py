from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from ecg_trust.counterfactual.counterfactual_review import (
    INTERPRETATION_BOUNDARY,
    BlindedCardiologyReview,
    CardiologyReviewSummary,
    CompleteSentinelAnalyzer,
    CounterfactualEvaluation,
    CounterfactualEvaluator,
    CounterfactualProposal,
    CounterfactualReason,
    CounterfactualStatus,
    CounterfactualValidationError,
    PlausibilityConfig,
    QualityGateStatus,
    ReviewEvidenceStatus,
    ReviewPreregistration,
    ReviewValidationError,
    SentinelAnalysis,
    TargetDirection,
    TargetSuperclass,
    TrustDecision,
    aggregate_cardiology_reviews,
    canonical_json,
    canonical_waveform_sha256,
)

FloatArray = NDArray[np.float64]
METHOD_HASH = "b" * 64


def _canonical_signal() -> FloatArray:
    time = np.linspace(0.0, 10.0, 1_000, endpoint=False)
    lead_i = 0.20 * np.sin(2.0 * np.pi * 1.1 * time)
    lead_ii = 0.28 * np.sin(2.0 * np.pi * 1.1 * time + 0.15)
    signal = np.zeros((12, 1_000), dtype=np.float64)
    signal[0] = lead_i
    signal[1] = lead_ii
    signal[2] = lead_ii - lead_i
    signal[3] = -(lead_i + lead_ii) / 2.0
    signal[4] = lead_i - lead_ii / 2.0
    signal[5] = lead_ii - lead_i / 2.0
    for index in range(6, 12):
        signal[index] = (0.15 + index * 0.01) * np.sin(
            2.0 * np.pi * (1.0 + index * 0.02) * time + index * 0.1
        )
    return signal


def _candidate(original: FloatArray) -> FloatArray:
    candidate = original.copy()
    candidate[6, 100:120] += 0.05
    return candidate


def _scores(target_score: float) -> tuple[tuple[TargetSuperclass, float], ...]:
    return (
        (TargetSuperclass.NORM, 0.70),
        (TargetSuperclass.MI, target_score),
        (TargetSuperclass.STTC, 0.10),
        (TargetSuperclass.CD, 0.08),
        (TargetSuperclass.HYP, 0.04),
    )


def _allowed_analysis(target_score: float) -> SentinelAnalysis:
    return SentinelAnalysis(
        decision=TrustDecision.PREDICTION_ALLOWED,
        reason_codes=("ALL_TRUST_GATES_PASSED",),
        quality_status=QualityGateStatus.PASS,
        artifact_free=True,
        ood_supported=True,
        uncertainty_passed=True,
        superclass_scores=_scores(target_score),
    )


def _gated_analysis(
    *,
    decision: TrustDecision = TrustDecision.ABSTAIN,
    quality_status: QualityGateStatus = QualityGateStatus.PASS,
    artifact_free: bool = True,
    ood_supported: bool = True,
    uncertainty_passed: bool = False,
) -> SentinelAnalysis:
    return SentinelAnalysis(
        decision=decision,
        reason_codes=("TRUST_GATE_REJECTED",),
        quality_status=quality_status,
        artifact_free=artifact_free,
        ood_supported=ood_supported,
        uncertainty_passed=uncertainty_passed,
    )


class QueueAnalyzer:
    def __init__(self, outcomes: Sequence[SentinelAnalysis | Exception | object]) -> None:
        self.outcomes = tuple(outcomes)
        self.calls = 0
        self.observed_read_only: list[bool] = []

    def analyze(self, signal_mv: FloatArray) -> SentinelAnalysis:
        self.observed_read_only.append(not signal_mv.flags.writeable)
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return cast(SentinelAnalysis, outcome)


def _proposal(
    original: FloatArray,
    candidate: FloatArray,
    *,
    direction: TargetDirection = TargetDirection.INCREASE,
) -> CounterfactualProposal:
    return CounterfactualProposal.bind(
        original_signal_mv=original,
        candidate_signal_mv=candidate,
        target_superclass=TargetSuperclass.MI,
        target_direction=direction,
        method_artifact_sha256=METHOD_HASH,
        method_version="method-v1",
        seed=42,
    )


def _evaluate(
    original: FloatArray,
    candidate: FloatArray,
    outcomes: Sequence[SentinelAnalysis | Exception | object],
    *,
    proposal: CounterfactualProposal | None = None,
    config: PlausibilityConfig | None = None,
) -> tuple[object, QueueAnalyzer]:
    analyzer = QueueAnalyzer(outcomes)
    evaluator = CounterfactualEvaluator(
        cast(CompleteSentinelAnalyzer, analyzer),
        config=config,
    )
    result = evaluator.evaluate(proposal or _proposal(original, candidate), original, candidate)
    return result, analyzer


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def test_proposal_binds_canonical_hashes_and_is_deterministic() -> None:
    original = _canonical_signal()
    candidate = _candidate(original)
    first = _proposal(original, candidate)
    second = _proposal(original.copy(), candidate.copy())

    assert first == second
    assert first.original_waveform_sha256 == canonical_waveform_sha256(original)
    assert first.candidate_waveform_sha256 == canonical_waveform_sha256(candidate)
    assert first.metadata_sha256() == second.metadata_sha256()
    assert first.canonical_json() == canonical_json(first.to_mapping())
    assert "[[" not in first.canonical_json()
    assert "waveform_mv" not in first.canonical_json()


def test_waveform_hash_is_shape_dtype_and_value_sensitive() -> None:
    original = _canonical_signal()
    changed = original.copy()
    changed[0, 0] += 1e-12
    assert canonical_waveform_sha256(original) != canonical_waveform_sha256(changed)
    assert original.flags.writeable is True
    assert changed.flags.writeable is True

    with pytest.raises(CounterfactualValidationError):
        canonical_waveform_sha256(original[:, :-1])
    with pytest.raises(CounterfactualValidationError):
        canonical_waveform_sha256(np.full((12, 1_000), np.nan))
    with pytest.raises(CounterfactualValidationError):
        canonical_waveform_sha256(np.full((12, 1_000), "1.0"))


def test_proposal_metadata_rejects_bad_hash_version_and_seed() -> None:
    with pytest.raises(CounterfactualValidationError):
        CounterfactualProposal(
            original_waveform_sha256="not-a-hash",
            candidate_waveform_sha256="c" * 64,
            target_superclass=TargetSuperclass.MI,
            target_direction=TargetDirection.INCREASE,
            method_artifact_sha256=METHOD_HASH,
            method_version="method-v1",
            seed=1,
        )
    original = _canonical_signal()
    candidate = _candidate(original)
    with pytest.raises(CounterfactualValidationError):
        CounterfactualProposal.bind(
            original_signal_mv=original,
            candidate_signal_mv=candidate,
            target_superclass=TargetSuperclass.MI,
            target_direction=TargetDirection.INCREASE,
            method_artifact_sha256=METHOD_HASH,
            method_version="../../method",
            seed=-1,
        )


def test_accepted_model_sensitivity_reruns_both_and_discloses_bounded_evidence() -> None:
    original = _canonical_signal()
    candidate = _candidate(original)
    result_object, analyzer = _evaluate(
        original,
        candidate,
        [_allowed_analysis(0.20), _allowed_analysis(0.45)],
    )
    result = cast("CounterfactualEvaluation", result_object)

    assert result.status is CounterfactualStatus.ACCEPTED_MODEL_SENSITIVITY
    assert result.reason_codes == (CounterfactualReason.ACCEPTED_MODEL_SENSITIVITY,)
    assert result.model_sensitivity_effect == pytest.approx(0.25)
    assert result.metrics is not None
    assert result.metrics.global_rms_delta_mv > 0.0
    assert result.metrics.global_linf_delta_mv == pytest.approx(0.05)
    assert result.metrics.change_support_fraction == pytest.approx(20 / 12_000)
    assert result.metrics.change_support_sparsity == pytest.approx(1.0 - 20 / 12_000)
    assert len(result.metrics.per_lead) == 12
    assert analyzer.calls == 2
    assert analyzer.observed_read_only == [True, True]

    public = result.to_public_summary()
    assert public["status"] == "accepted_model_sensitivity"
    assert public["model_sensitivity_effect"] == pytest.approx(0.25)
    assert public["interpretation_boundary"] == INTERPRETATION_BOUNDARY
    serialized = result.canonical_public_json()
    assert result.proposal_metadata_sha256 not in serialized
    assert METHOD_HASH not in serialized
    assert "original_waveform_sha256" not in serialized
    assert "candidate_waveform_sha256" not in serialized
    assert len(result.public_summary_sha256()) == 64


def test_decrease_direction_uses_target_aligned_effect() -> None:
    original = _canonical_signal()
    candidate = _candidate(original)
    result_object, _ = _evaluate(
        original,
        candidate,
        [_allowed_analysis(0.70), _allowed_analysis(0.40)],
        proposal=_proposal(original, candidate, direction=TargetDirection.DECREASE),
    )
    result = cast("CounterfactualEvaluation", result_object)
    assert result.status is CounterfactualStatus.ACCEPTED_MODEL_SENSITIVITY
    assert result.model_sensitivity_effect == pytest.approx(0.30)


def test_rejected_constraint_keeps_internal_effect_but_public_gate_omits_scores() -> None:
    original = _canonical_signal()
    candidate = original.copy()
    candidate[6] += 0.50
    config = PlausibilityConfig(max_per_lead_rms_delta_mv=0.20)
    result_object, analyzer = _evaluate(
        original,
        candidate,
        [_allowed_analysis(0.20), _allowed_analysis(0.50)],
        config=config,
    )
    result = cast("CounterfactualEvaluation", result_object)

    assert result.status is CounterfactualStatus.REJECTED_CONSTRAINT
    assert CounterfactualReason.PER_LEAD_RMS_DELTA_EXCEEDED in result.reason_codes
    assert result.model_sensitivity_effect == pytest.approx(0.30)
    public = result.to_public_summary()
    assert "original_target_score" not in public
    assert "candidate_target_score" not in public
    assert "model_sensitivity_effect" not in public
    assert analyzer.calls == 2


def test_wrong_direction_and_small_effect_are_rejected() -> None:
    original = _canonical_signal()
    candidate = _candidate(original)
    wrong_object, _ = _evaluate(
        original,
        candidate,
        [_allowed_analysis(0.50), _allowed_analysis(0.40)],
    )
    wrong = cast("CounterfactualEvaluation", wrong_object)
    assert wrong.status is CounterfactualStatus.REJECTED_CONSTRAINT
    assert CounterfactualReason.TARGET_DIRECTION_NOT_MET in wrong.reason_codes

    small_object, _ = _evaluate(
        original,
        candidate,
        [_allowed_analysis(0.20), _allowed_analysis(0.21)],
    )
    small = cast("CounterfactualEvaluation", small_object)
    assert small.status is CounterfactualStatus.REJECTED_CONSTRAINT
    assert CounterfactualReason.MODEL_EFFECT_BELOW_MINIMUM in small.reason_codes


def test_identical_waveforms_cannot_be_accepted_from_nondeterministic_scores() -> None:
    original = _canonical_signal()
    candidate = original.copy()
    result_object, _ = _evaluate(
        original,
        candidate,
        [_allowed_analysis(0.20), _allowed_analysis(0.50)],
    )
    result = cast("CounterfactualEvaluation", result_object)
    assert result.status is CounterfactualStatus.REJECTED_CONSTRAINT
    assert CounterfactualReason.NO_MATERIAL_SIGNAL_CHANGE in result.reason_codes


@pytest.mark.parametrize(
    ("candidate_analysis", "expected_reason"),
    [
        (
            _gated_analysis(ood_supported=False, uncertainty_passed=True),
            CounterfactualReason.CANDIDATE_OOD_FAILED,
        ),
        (
            _gated_analysis(uncertainty_passed=False),
            CounterfactualReason.CANDIDATE_UNCERTAINTY_FAILED,
        ),
    ],
)
def test_ood_and_uncertainty_failures_make_evidence_unavailable(
    candidate_analysis: SentinelAnalysis,
    expected_reason: CounterfactualReason,
) -> None:
    original = _canonical_signal()
    candidate = _candidate(original)
    result_object, analyzer = _evaluate(
        original,
        candidate,
        [_allowed_analysis(0.20), candidate_analysis],
    )
    result = cast("CounterfactualEvaluation", result_object)

    assert result.status is CounterfactualStatus.EVIDENCE_UNAVAILABLE
    assert expected_reason in result.reason_codes
    assert "model_sensitivity_effect" not in result.to_public_summary()
    assert analyzer.calls == 2


@pytest.mark.parametrize(
    ("candidate_analysis", "expected_reason"),
    [
        (
            _gated_analysis(
                decision=TrustDecision.REACQUIRE,
                quality_status=QualityGateStatus.REACQUIRE,
                uncertainty_passed=True,
            ),
            CounterfactualReason.CANDIDATE_QUALITY_FAILED,
        ),
        (
            _gated_analysis(
                decision=TrustDecision.REACQUIRE,
                artifact_free=False,
                uncertainty_passed=True,
            ),
            CounterfactualReason.CANDIDATE_ARTIFACT_FAILED,
        ),
    ],
)
def test_candidate_quality_and_artifact_failures_reject_constraint(
    candidate_analysis: SentinelAnalysis,
    expected_reason: CounterfactualReason,
) -> None:
    original = _canonical_signal()
    candidate = _candidate(original)
    result_object, _ = _evaluate(
        original,
        candidate,
        [_allowed_analysis(0.20), candidate_analysis],
    )
    result = cast("CounterfactualEvaluation", result_object)
    assert result.status is CounterfactualStatus.REJECTED_CONSTRAINT
    assert expected_reason in result.reason_codes


def test_original_gate_failure_makes_baseline_evidence_unavailable() -> None:
    original = _canonical_signal()
    candidate = _candidate(original)
    original_failure = _gated_analysis(
        decision=TrustDecision.REACQUIRE,
        quality_status=QualityGateStatus.REACQUIRE,
        uncertainty_passed=True,
    )
    result_object, _ = _evaluate(
        original,
        candidate,
        [original_failure, _allowed_analysis(0.40)],
    )
    result = cast("CounterfactualEvaluation", result_object)
    assert result.status is CounterfactualStatus.EVIDENCE_UNAVAILABLE
    assert CounterfactualReason.ORIGINAL_QUALITY_FAILED in result.reason_codes


def test_analyzer_failures_are_sanitized_and_both_reruns_are_attempted() -> None:
    original = _canonical_signal()
    candidate = _candidate(original)
    result_object, analyzer = _evaluate(
        original,
        candidate,
        [RuntimeError("C:\\private\\model.pt token=do-not-leak"), object()],
    )
    result = cast("CounterfactualEvaluation", result_object)

    assert result.status is CounterfactualStatus.EVIDENCE_UNAVAILABLE
    assert result.reason_codes == (
        CounterfactualReason.ORIGINAL_ANALYZER_FAILURE,
        CounterfactualReason.CANDIDATE_ANALYZER_FAILURE,
    )
    assert analyzer.calls == 2
    serialized = result.canonical_public_json()
    assert "private" not in serialized.lower()
    assert "do-not-leak" not in serialized


@pytest.mark.parametrize(
    ("candidate", "expected_reason"),
    [
        (np.zeros((12, 999), dtype=np.float64), CounterfactualReason.CANDIDATE_SIGNAL_INVALID),
        (
            np.full((12, 1_000), np.nan, dtype=np.float64),
            CounterfactualReason.CANDIDATE_SIGNAL_NONFINITE,
        ),
        (
            np.full((12, 1_000), np.inf, dtype=np.float64),
            CounterfactualReason.CANDIDATE_SIGNAL_NONFINITE,
        ),
    ],
)
def test_malformed_or_nonfinite_candidates_fail_before_analysis(
    candidate: FloatArray,
    expected_reason: CounterfactualReason,
) -> None:
    original = _canonical_signal()
    proposal = CounterfactualProposal(
        original_waveform_sha256=canonical_waveform_sha256(original),
        candidate_waveform_sha256="c" * 64,
        target_superclass=TargetSuperclass.MI,
        target_direction=TargetDirection.INCREASE,
        method_artifact_sha256=METHOD_HASH,
        method_version="method-v1",
        seed=1,
    )
    result_object, analyzer = _evaluate(
        original,
        candidate,
        [_allowed_analysis(0.20), _allowed_analysis(0.40)],
        proposal=proposal,
    )
    result = cast("CounterfactualEvaluation", result_object)
    assert result.status is CounterfactualStatus.REJECTED_CONSTRAINT
    assert result.reason_codes == (expected_reason,)
    assert analyzer.calls == 0
    assert "perturbation_metrics" not in result.to_public_summary()


def test_malformed_original_makes_evidence_unavailable() -> None:
    original = np.zeros((11, 1_000), dtype=np.float64)
    candidate = _canonical_signal()
    proposal = CounterfactualProposal(
        original_waveform_sha256="a" * 64,
        candidate_waveform_sha256=canonical_waveform_sha256(candidate),
        target_superclass=TargetSuperclass.MI,
        target_direction=TargetDirection.INCREASE,
        method_artifact_sha256=METHOD_HASH,
        method_version="method-v1",
        seed=1,
    )
    result_object, analyzer = _evaluate(
        original,
        candidate,
        [_allowed_analysis(0.20), _allowed_analysis(0.40)],
        proposal=proposal,
    )
    result = cast("CounterfactualEvaluation", result_object)
    assert result.status is CounterfactualStatus.EVIDENCE_UNAVAILABLE
    assert result.reason_codes == (CounterfactualReason.ORIGINAL_SIGNAL_INVALID,)
    assert analyzer.calls == 0


def test_hash_mismatch_rejects_without_analyzer() -> None:
    original = _canonical_signal()
    candidate = _candidate(original)
    proposal = _proposal(original, candidate)
    candidate[6, 200] += 0.01

    result_object, analyzer = _evaluate(
        original,
        candidate,
        [_allowed_analysis(0.20), _allowed_analysis(0.40)],
        proposal=proposal,
    )
    result = cast("CounterfactualEvaluation", result_object)
    assert result.status is CounterfactualStatus.REJECTED_CONSTRAINT
    assert result.reason_codes == (CounterfactualReason.CANDIDATE_HASH_MISMATCH,)
    assert analyzer.calls == 0


def test_limb_lead_identity_is_rechecked_and_degradation_rejected() -> None:
    original = _canonical_signal()
    candidate = original.copy()
    candidate[0] += 0.20
    result_object, analyzer = _evaluate(
        original,
        candidate,
        [_allowed_analysis(0.20), _allowed_analysis(0.45)],
    )
    result = cast("CounterfactualEvaluation", result_object)

    assert result.status is CounterfactualStatus.REJECTED_CONSTRAINT
    assert CounterfactualReason.CANDIDATE_LIMB_IDENTITY_FAILED in result.reason_codes
    assert CounterfactualReason.LIMB_IDENTITY_DEGRADATION_EXCEEDED in result.reason_codes
    assert result.metrics is not None
    assert result.metrics.original_limb_identity_rms_mv < 1e-12
    assert result.metrics.candidate_limb_identity_rms_mv > 0.05
    assert analyzer.calls == 2


def test_independent_amplitude_and_step_bounds_reject_artifact_like_candidate() -> None:
    original = _canonical_signal()
    candidate = original.copy()
    candidate[6, 500] = 9.0
    result_object, _ = _evaluate(
        original,
        candidate,
        [_allowed_analysis(0.20), _allowed_analysis(0.45)],
    )
    result = cast("CounterfactualEvaluation", result_object)
    assert result.status is CounterfactualStatus.REJECTED_CONSTRAINT
    assert CounterfactualReason.CANDIDATE_AMPLITUDE_FAILED in result.reason_codes
    assert CounterfactualReason.CANDIDATE_STEP_FAILED in result.reason_codes


def test_sentinel_analysis_structurally_gates_scores() -> None:
    with pytest.raises(CounterfactualValidationError, match="scores are forbidden"):
        SentinelAnalysis(
            decision=TrustDecision.ABSTAIN,
            reason_codes=("CONFIDENCE_GATE_ABSTAINED",),
            quality_status=QualityGateStatus.PASS,
            artifact_free=True,
            ood_supported=True,
            uncertainty_passed=False,
            superclass_scores=_scores(0.2),
        )
    with pytest.raises(CounterfactualValidationError, match="complete canonical order"):
        SentinelAnalysis(
            decision=TrustDecision.PREDICTION_ALLOWED,
            reason_codes=("ALL_TRUST_GATES_PASSED",),
            quality_status=QualityGateStatus.PASS,
            artifact_free=True,
            ood_supported=True,
            uncertainty_passed=True,
            superclass_scores=((TargetSuperclass.MI, 0.2),),
        )
    with pytest.raises(CounterfactualValidationError, match="finite probabilities"):
        SentinelAnalysis(
            decision=TrustDecision.PREDICTION_ALLOWED,
            reason_codes=("ALL_TRUST_GATES_PASSED",),
            quality_status=QualityGateStatus.PASS,
            artifact_free=True,
            ood_supported=True,
            uncertainty_passed=True,
            superclass_scores=_scores(math.nan),
        )


def test_plausibility_configuration_rejects_incoherent_bounds() -> None:
    with pytest.raises(CounterfactualValidationError):
        PlausibilityConfig(max_change_support_fraction=0.0)
    with pytest.raises(CounterfactualValidationError):
        PlausibilityConfig(change_epsilon_mv=2.0, max_global_linf_delta_mv=1.0)
    with pytest.raises(CounterfactualValidationError):
        PlausibilityConfig(minimum_model_sensitivity_effect=1.1)


def _review(
    proposal_label: str,
    pseudonym: str,
    *,
    rating: int = 4,
    artifact: bool = False,
    useful: bool = True,
) -> BlindedCardiologyReview:
    return BlindedCardiologyReview(
        proposal_metadata_sha256=_digest(proposal_label),
        reviewer_pseudonym=pseudonym,
        morphology_plausibility_rating=rating,
        artifact_present=artifact,
        useful_for_model_review=useful,
    )


def _small_preregistration(*, threshold: float = 0.75) -> ReviewPreregistration:
    return ReviewPreregistration(
        version="review-v1",
        minimum_reviews_per_proposal=2,
        minimum_proposal_count=2,
        minimum_unique_reviewer_count=2,
        minimum_total_review_count=4,
        usefulness_vote_threshold=threshold,
    )


def _threshold_batch(
    useful_votes: tuple[bool, bool, bool, bool],
) -> tuple[BlindedCardiologyReview, ...]:
    pseudonyms = ("REV-ABCDEFGH", "REV-BCDEFGJK")
    return (
        _review("proposal-a", pseudonyms[0], rating=4, useful=useful_votes[0]),
        _review("proposal-a", pseudonyms[1], rating=5, useful=useful_votes[1]),
        _review("proposal-b", pseudonyms[0], rating=3, useful=useful_votes[2]),
        _review("proposal-b", pseudonyms[1], rating=4, useful=useful_votes[3]),
    )


def test_blinded_review_contract_is_canonical_and_contains_no_free_text() -> None:
    review = _review("proposal-a", "REV-ABCDEFGH")
    mapping = review.to_mapping()
    assert mapping["blinded"] is True
    assert set(mapping) == {
        "schema_version",
        "proposal_metadata_sha256",
        "reviewer_pseudonym",
        "morphology_plausibility_rating",
        "artifact_present",
        "useful_for_model_review",
        "blinded",
    }
    assert review.canonical_json() == canonical_json(mapping)
    assert len(review.sha256()) == 64
    assert "comment" not in review.canonical_json()
    assert "name" not in review.canonical_json()


@pytest.mark.parametrize(
    ("pseudonym", "rating"),
    [
        ("doctor@example.com", 4),
        ("Dr-Smith", 4),
        ("REV-SHORT", 4),
        ("REV-ABCDEFGH", 0),
        ("REV-ABCDEFGH", 6),
    ],
)
def test_review_contract_rejects_identifiers_and_invalid_ratings(
    pseudonym: str,
    rating: int,
) -> None:
    with pytest.raises(ReviewValidationError):
        _review("proposal-a", pseudonym, rating=rating)


def test_aggregate_review_threshold_met_only_after_preregistered_counts_and_rate() -> None:
    summary = aggregate_cardiology_reviews(
        _threshold_batch((True, True, True, False)),
        _small_preregistration(threshold=0.75),
    )
    assert summary.status is ReviewEvidenceStatus.PREREGISTERED_THRESHOLD_MET
    assert summary.usefulness_claim_allowed is True
    assert summary.total_review_count == 4
    assert summary.eligible_proposal_count == 2
    assert summary.unique_reviewer_count == 2
    assert summary.model_review_usefulness_vote_rate == pytest.approx(0.75)
    assert summary.mean_morphology_rating == pytest.approx(4.0)
    assert summary.median_morphology_rating == pytest.approx(4.0)
    assert summary.pairwise_morphology_absolute_difference == pytest.approx(1.0)
    assert summary.pairwise_usefulness_agreement == pytest.approx(0.5)
    assert summary.pairwise_usefulness_kappa == pytest.approx(-1.0 / 3.0)
    assert summary.preregistered_usefulness_vote_threshold == pytest.approx(0.75)
    assert summary.preregistration_sha256 == _small_preregistration().sha256()


def test_no_usefulness_claim_when_rate_or_count_threshold_is_not_met() -> None:
    low_rate = aggregate_cardiology_reviews(
        _threshold_batch((True, False, True, False)),
        _small_preregistration(threshold=0.75),
    )
    assert low_rate.status is ReviewEvidenceStatus.DESCRIPTIVE_ONLY
    assert low_rate.usefulness_claim_allowed is False

    insufficient = aggregate_cardiology_reviews(
        (_review("proposal-a", "REV-ABCDEFGH"),),
        _small_preregistration(),
    )
    assert insufficient.status is ReviewEvidenceStatus.INSUFFICIENT_REVIEW_COUNT
    assert insufficient.usefulness_claim_allowed is False
    assert insufficient.eligible_review_count == 0


def test_aggregate_public_summary_omits_reviewer_and_proposal_identifiers() -> None:
    reviews = _threshold_batch((True, True, True, True))
    summary = aggregate_cardiology_reviews(reviews, _small_preregistration())
    public = summary.to_public_summary()
    serialized = summary.canonical_public_json()

    assert public["interpretation_boundary"] == INTERPRETATION_BOUNDARY
    assert public["review_scope"] == ("blinded cardiology review of model-sensitivity proposals")
    assert "reviewer_pseudonym" not in serialized
    assert "proposal_metadata_sha256" not in serialized
    assert "preregistration_sha256" not in serialized
    assert all(review.reviewer_pseudonym not in serialized for review in reviews)
    assert all(review.proposal_metadata_sha256 not in serialized for review in reviews)
    assert len(summary.public_summary_sha256()) == 64


def test_duplicate_reviewer_for_one_proposal_is_rejected() -> None:
    duplicate = _review("proposal-a", "REV-ABCDEFGH")
    with pytest.raises(ReviewValidationError, match="only one review"):
        aggregate_cardiology_reviews((duplicate, duplicate), _small_preregistration())


def test_preregistration_is_canonical_and_rejects_underpowered_contract() -> None:
    preregistration = _small_preregistration()
    assert preregistration.canonical_json() == canonical_json(preregistration.to_mapping())
    assert len(preregistration.sha256()) == 64

    with pytest.raises(ReviewValidationError, match="cover every eligible proposal"):
        ReviewPreregistration(
            minimum_reviews_per_proposal=3,
            minimum_proposal_count=5,
            minimum_unique_reviewer_count=3,
            minimum_total_review_count=10,
            usefulness_vote_threshold=0.75,
        )


def test_empty_review_batch_returns_insufficient_aggregate_without_identifiers() -> None:
    summary: CardiologyReviewSummary = aggregate_cardiology_reviews(
        (),
        _small_preregistration(),
    )
    assert summary.status is ReviewEvidenceStatus.INSUFFICIENT_REVIEW_COUNT
    assert summary.total_review_count == 0
    assert summary.mean_morphology_rating is None
    assert summary.pairwise_usefulness_agreement is None
    assert summary.pairwise_usefulness_kappa is None
    assert summary.usefulness_claim_allowed is False


def test_module_contains_no_generation_or_treatment_output_contract() -> None:
    original = _canonical_signal()
    candidate = _candidate(original)
    result_object, _ = _evaluate(
        original,
        candidate,
        [_allowed_analysis(0.20), _allowed_analysis(0.45)],
    )
    result = cast("CounterfactualEvaluation", result_object)
    public = result.to_public_summary()
    assert "generated_waveform" not in public
    assert "treatment" not in public
    assert public["interpretation_boundary"] == (
        "not physiological truth, not causal, not treatment advice"
    )
