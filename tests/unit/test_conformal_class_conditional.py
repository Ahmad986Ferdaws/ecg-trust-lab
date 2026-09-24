from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from ecg_trust.conformal import (
    BinaryDecision,
    BinaryPredictionSets,
    ClassConditionalLabelwiseConformal,
    ConformalValidationError,
    LabelwiseBinaryConformal,
    UncertaintyKind,
    evaluate_prediction_sets,
)
from ecg_trust.constants import SUPERCLASSES
from ecg_trust.contract_adapters import conformal_prediction_sets_to_contracts
from ecg_trust.contracts import ARTIFACT_REFERENCE_SCHEMA_VERSION, ArtifactReference, TrustDecision
from ecg_trust.quality.signal_quality import QualityStatus, SignalQualityReport
from ecg_trust.trust_policy import TrustPolicyInputs, TrustReasonCode, evaluate_trust_policy

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int8]


def _calibration_split() -> tuple[FloatArray, IntArray]:
    probabilities = np.asarray(
        [
            [0.9, 0.7],
            [0.8, 0.05],
            [0.6, 0.1],
            [0.1, 0.15],
            [0.2, 0.2],
            [0.3, 0.4],
        ],
        dtype=np.float64,
    )
    targets = np.asarray(
        [[1, 1], [1, 0], [1, 0], [0, 0], [0, 0], [0, 0]],
        dtype=np.int8,
    )
    return probabilities, targets


def _calibrator() -> ClassConditionalLabelwiseConformal:
    probabilities, targets = _calibration_split()
    return ClassConditionalLabelwiseConformal.fit(
        probabilities,
        targets,
        label_names=("common", "rare"),
        negative_alphas={"common": 0.5, "rare": 0.4},
        positive_alphas={"common": 0.25, "rare": 0.5},
    )


def _rare_positive_cohort(
    generator: np.random.Generator,
    n_samples: int,
) -> tuple[FloatArray, IntArray]:
    """NORM is balanced; MI is rare and its positives are poorly separated."""

    targets = (generator.uniform(size=(n_samples, 2)) < (0.5, 0.08)).astype(np.int8)
    positive = generator.beta((5.0, 3.0), (2.0, 5.0), size=(n_samples, 2))
    negative = generator.beta((2.0, 2.0), (5.0, 8.0), size=(n_samples, 2))
    return np.where(targets == 1, positive, negative), targets


def _outcome_coverage(
    prediction_sets: BinaryPredictionSets,
    targets: IntArray,
    *,
    label: int,
    outcome: int,
) -> float:
    mask = prediction_sets.supported_mask if outcome == 1 else prediction_sets.not_supported_mask
    return float(mask[targets[:, label] == outcome, label].mean())


def test_fit_uses_only_same_outcome_cases_with_corrected_ranks() -> None:
    calibrator = _calibrator()

    assert calibrator.n_calibration_samples == 6
    assert calibrator.negative_counts == (3, 5)
    assert calibrator.positive_counts == (3, 1)
    assert calibrator.negative_ranks == (2, 4)
    assert calibrator.positive_ranks == (3, 1)
    assert calibrator.negative_thresholds == pytest.approx((0.2, 0.2))
    assert calibrator.positive_thresholds == pytest.approx((0.4, 0.3))
    assert calibrator.negative_budget_attainable == (True, True)
    assert calibrator.positive_budget_attainable == (True, True)


def test_budgets_bind_to_label_names_not_mapping_order() -> None:
    probabilities, targets = _calibration_split()
    reordered = ClassConditionalLabelwiseConformal.fit(
        probabilities,
        targets,
        label_names=("common", "rare"),
        negative_alphas={"rare": 0.4, "common": 0.5},
        positive_alphas={"rare": 0.5, "common": 0.25},
    )

    assert reordered == _calibrator()
    assert reordered.negative_alphas == (0.5, 0.4)
    assert reordered.positive_alphas == (0.25, 0.5)


def test_prediction_includes_an_outcome_whose_score_ties_its_threshold() -> None:
    # Saturated or quantized sigmoid outputs repeat, so calibration scores tie and a
    # new case can score exactly at a threshold. The finite-sample guarantee counts
    # such a case as covered, so both comparisons must be inclusive.
    calibrator = ClassConditionalLabelwiseConformal.fit(
        [[0.2], [0.2], [0.2], [0.4], [0.7], [0.7], [0.7], [0.9]],
        [[0], [0], [0], [0], [1], [1], [1], [1]],
        label_names=("MI",),
        negative_alphas={"MI": 0.5},
        positive_alphas={"MI": 0.5},
    )

    assert (calibrator.negative_ranks, calibrator.positive_ranks) == ((3,), (3,))
    assert calibrator.negative_thresholds == (0.2,)
    assert calibrator.positive_thresholds == (1.0 - 0.7,)
    at_threshold = calibrator.predict([[0.2], [0.7]])
    assert at_threshold.decisions == (
        (BinaryDecision.NOT_SUPPORTED,),
        (BinaryDecision.SUPPORTED,),
    )
    one_ulp_beyond = calibrator.predict([[np.nextafter(0.2, 1.0)], [np.nextafter(0.7, 0.0)]])
    assert not one_ulp_beyond.not_supported_mask[0, 0]
    assert not one_ulp_beyond.supported_mask[1, 0]


def test_prediction_uses_separate_outcome_thresholds_and_existing_set_type() -> None:
    prediction_sets = _calibrator().predict([[0.65, 0.25], [0.5, 0.75], [0.15, 0.1]])

    assert isinstance(prediction_sets, BinaryPredictionSets)
    assert prediction_sets.decisions == (
        (BinaryDecision.SUPPORTED, BinaryDecision.UNCERTAIN),
        (BinaryDecision.UNCERTAIN, BinaryDecision.SUPPORTED),
        (BinaryDecision.NOT_SUPPORTED, BinaryDecision.NOT_SUPPORTED),
    )
    assert prediction_sets.uncertainty_kinds == (
        (None, UncertaintyKind.EMPTY),
        (UncertaintyKind.EMPTY, None),
        (None, None),
    )


def test_rare_positive_coverage_meets_budget_where_pooled_calibration_does_not() -> None:
    generator = np.random.default_rng(20260923)
    budgets = {"NORM": 0.1, "MI": 0.1}
    pooled_marginal: list[float] = []
    pooled_positive: list[float] = []
    conditional_positive: list[float] = []
    conditional_negative: list[float] = []
    for _ in range(40):
        calibration_probabilities, calibration_targets = _rare_positive_cohort(generator, 2_500)
        test_probabilities, test_targets = _rare_positive_cohort(generator, 2_500)
        pooled = LabelwiseBinaryConformal.fit(
            calibration_probabilities,
            calibration_targets,
            label_names=("NORM", "MI"),
            alpha=0.1,
        ).predict(test_probabilities)
        conditional = ClassConditionalLabelwiseConformal.fit(
            calibration_probabilities,
            calibration_targets,
            label_names=("NORM", "MI"),
            negative_alphas=budgets,
            positive_alphas=budgets,
        ).predict(test_probabilities)
        pooled_marginal.append(evaluate_prediction_sets(pooled, test_targets).labelwise_coverage[1])
        pooled_positive.append(_outcome_coverage(pooled, test_targets, label=1, outcome=1))
        conditional_positive.append(
            _outcome_coverage(conditional, test_targets, label=1, outcome=1)
        )
        conditional_negative.append(
            _outcome_coverage(conditional, test_targets, label=1, outcome=0)
        )

    # The pooled artifact meets its marginal guarantee while missing most true MI.
    assert np.mean(pooled_marginal) >= 0.88
    assert np.mean(pooled_positive) < 0.5
    # About 200 positives per draw; 0.02 is roughly four Monte-Carlo standard errors.
    assert 0.88 <= np.mean(conditional_positive) <= 0.93
    assert np.mean(conditional_negative) >= 0.88


def test_unattainable_or_empty_stratum_uses_conservative_threshold_one() -> None:
    generator = np.random.default_rng(7)
    probabilities = generator.uniform(0.0, 1.0, size=(40, 2))
    targets = np.zeros((40, 2), dtype=np.int8)
    targets[:5, 0] = 1
    calibrator = ClassConditionalLabelwiseConformal.fit(
        probabilities,
        targets,
        label_names=("MI", "HYP"),
        negative_alphas={"MI": 0.1, "HYP": 0.1},
        positive_alphas={"MI": 0.1, "HYP": 0.1},
    )

    assert calibrator.positive_counts == (5, 0)
    assert calibrator.positive_ranks == (6, 1)
    assert calibrator.positive_thresholds == (1.0, 1.0)
    assert calibrator.positive_budget_attainable == (False, False)
    assert calibrator.negative_counts == (35, 40)
    assert calibrator.negative_budget_attainable == (True, True)
    prediction_sets = calibrator.predict([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    assert prediction_sets.supported_mask.all()
    assert BinaryDecision.NOT_SUPPORTED not in {
        decision for row in prediction_sets.decisions for decision in row
    }


def test_stricter_budget_on_one_label_widens_only_that_label() -> None:
    generator = np.random.default_rng(11)
    labels = ("NORM", "MI", "HYP")
    probabilities = generator.uniform(0.0, 1.0, size=(1_000, 3))
    targets = generator.binomial(1, probabilities).astype(np.int8)
    negative = dict.fromkeys(labels, 0.1)
    baseline = ClassConditionalLabelwiseConformal.fit(
        probabilities,
        targets,
        label_names=labels,
        negative_alphas=negative,
        positive_alphas=dict.fromkeys(labels, 0.1),
    )
    stricter = ClassConditionalLabelwiseConformal.fit(
        probabilities,
        targets,
        label_names=labels,
        negative_alphas=negative,
        positive_alphas={"NORM": 0.1, "MI": 0.02, "HYP": 0.1},
    )

    assert stricter.negative_thresholds == baseline.negative_thresholds
    assert stricter.positive_thresholds[0] == baseline.positive_thresholds[0]
    assert stricter.positive_thresholds[2] == baseline.positive_thresholds[2]
    assert stricter.positive_thresholds[1] > baseline.positive_thresholds[1]
    evaluation = generator.uniform(0.0, 1.0, size=(2_000, 3))
    before = baseline.predict(evaluation)
    after = stricter.predict(evaluation)
    assert np.array_equal(after.not_supported_mask, before.not_supported_mask)
    assert np.array_equal(after.supported_mask[:, [0, 2]], before.supported_mask[:, [0, 2]])
    assert np.all(after.supported_mask[:, 1] >= before.supported_mask[:, 1])
    assert after.supported_mask[:, 1].sum() > before.supported_mask[:, 1].sum()


def test_decisions_feed_the_existing_trust_policy_unchanged() -> None:
    generator = np.random.default_rng(5)
    targets = generator.binomial(1, 0.5, size=(400, len(SUPERCLASSES))).astype(np.int8)
    probabilities = np.where(
        targets == 1,
        generator.uniform(0.8, 1.0, size=targets.shape),
        generator.uniform(0.0, 0.2, size=targets.shape),
    )
    budgets = dict.fromkeys(SUPERCLASSES, 0.1)
    calibrator = ClassConditionalLabelwiseConformal.fit(
        probabilities,
        targets,
        label_names=SUPERCLASSES,
        negative_alphas=budgets,
        positive_alphas=budgets,
    )
    decisions = calibrator.predict(
        [[0.95, 0.05, 0.05, 0.05, 0.05], [0.95, 0.05, 0.05, 0.05, 0.5]]
    ).decisions
    quality = SignalQualityReport(
        status=QualityStatus.PASS,
        config_version="test-quality-v1",
        global_issues=(),
        leads=(),
        reversal_evidence=None,
    )

    results = [
        evaluate_trust_policy(
            TrustPolicyInputs(
                release_integrity_verified=True,
                input_contract_valid=True,
                quality_report=quality,
                distribution_supported=True,
                legacy_entropy_gate_accepted=True,
                conformal_decisions=row,
            )
        )
        for row in decisions
    ]

    assert results[0].decision is TrustDecision.PREDICTION_ALLOWED
    assert results[1].decision is TrustDecision.ABSTAIN
    assert results[1].reason_codes == (TrustReasonCode.CONFORMAL_SET_UNCERTAIN,)
    assert results[1].uncertain_labels == ("HYP",)


def test_case_contract_adapter_cannot_tell_these_sets_from_pooled_ones() -> None:
    # Characterizes a known hazard. BinaryPredictionSets carry no provenance, and the
    # v1 case contract admits only the pooled artifact type and scope, so the adapter
    # stamps those on class-conditional sets. The module docstring and
    # docs/TRUST_SENTINEL_VNEXT.md forbid this conversion. When a new case-contract
    # version lets the adapter take the calibrator, replace this with a refusal test.
    generator = np.random.default_rng(3)
    targets = generator.binomial(1, 0.5, size=(200, len(SUPERCLASSES))).astype(np.int8)
    probabilities = np.where(
        targets == 1,
        generator.uniform(0.6, 1.0, size=targets.shape),
        generator.uniform(0.0, 0.4, size=targets.shape),
    )
    calibrator = ClassConditionalLabelwiseConformal.fit(
        probabilities,
        targets,
        label_names=SUPERCLASSES,
        negative_alphas=dict.fromkeys(SUPERCLASSES, 0.1),
        positive_alphas={**dict.fromkeys(SUPERCLASSES, 0.1), "MI": 0.02},
    )
    row = [0.9, 0.6, 0.1, 0.1, 0.1]

    contracts = conformal_prediction_sets_to_contracts(
        calibrator.predict([row]),
        np.asarray(row),
        calibration_artifact=ArtifactReference(
            schema_version=ARTIFACT_REFERENCE_SCHEMA_VERSION,
            artifact_id="class-conditional-conformal",
            file_sha256="sha256:" + "a" * 64,
            size_bytes=10,
            media_type="application/json",
            sensitive=False,
        ),
    )

    assert {item.calibration_artifact_type for item in contracts} == {
        "ecg_trust.labelwise_binary_conformal"
    }
    assert {item.coverage_scope for item in contracts} == {
        "labelwise_marginal_under_exchangeability"
    }
    assert calibrator.to_dict()["artifact_type"] != contracts[0].calibration_artifact_type
    assert calibrator.to_dict()["coverage_scope"] != contracts[0].coverage_scope


def test_default_labelwise_artifact_and_prediction_sets_are_unchanged() -> None:
    calibrator = LabelwiseBinaryConformal.fit(
        [[0.9, 0.4], [0.8, 0.3], [0.7, 0.2], [0.6, 0.1]],
        [[1, 1]] * 4,
        label_names=("low_error", "high_error"),
        alpha=0.5,
    )

    assert json.dumps(calibrator.to_dict(), sort_keys=True) == (
        '{"alpha": 0.5, "artifact_type": "ecg_trust.labelwise_binary_conformal", '
        '"coverage_scope": "labelwise_marginal_under_exchangeability", '
        '"label_names": ["low_error", "high_error"], "n_calibration_samples": 4, '
        '"quantile_level": 0.75, "quantile_rank": 3, "schema_version": 1, '
        '"thresholds": [0.30000000000000004, 0.8]}'
    )
    assert json.dumps(calibrator.predict([[0.8, 0.5], [0.5, 0.5]]).to_dict(), sort_keys=True) == (
        '{"artifact_type": "ecg_trust.binary_prediction_sets", '
        '"decisions": [["supported", "uncertain"], ["uncertain", "uncertain"]], '
        '"include_not_supported": [[false, true], [false, true]], '
        '"include_supported": [[true, true], [false, true]], '
        '"label_names": ["low_error", "high_error"], "schema_version": 1, '
        '"uncertainty_kind": [[null, "both"], ["empty", "both"]]}'
    )
    with pytest.raises(ConformalValidationError, match="keys differ"):
        LabelwiseBinaryConformal.from_dict(_calibrator().to_dict())
    with pytest.raises(ConformalValidationError, match="keys differ"):
        ClassConditionalLabelwiseConformal.from_dict(calibrator.to_dict())


def test_artifact_round_trips_through_json() -> None:
    calibrator = _calibrator()
    payload = json.loads(json.dumps(calibrator.to_dict(), allow_nan=False))

    assert payload["artifact_type"] == "ecg_trust.class_conditional_labelwise_binary_conformal"
    assert payload["coverage_scope"] == (
        "labelwise_class_conditional_under_within_class_exchangeability"
    )
    restored = ClassConditionalLabelwiseConformal.from_dict(payload)
    assert restored == calibrator
    probabilities = [[0.65, 0.25], [0.5, 0.75]]
    assert restored.predict(probabilities) == calibrator.predict(probabilities)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", 2, "schema_version"),
        ("schema_version", True, "schema_version"),
        ("artifact_type", "ecg_trust.labelwise_binary_conformal", "artifact_type"),
        ("coverage_scope", "labelwise_marginal_under_exchangeability", "coverage_scope"),
        ("positive_ranks", [2, 1], "positive_ranks do not match"),
        ("negative_counts", [3, 4], "negative_ranks do not match"),
        ("n_calibration_samples", 7, "sum to n_calibration_samples"),
        ("positive_alphas", [0.25], "2 values"),
        ("positive_thresholds", [0.4, 1.5], r"\[0.0, 1.0\]"),
        ("negative_counts", [3.0, 5], "integers"),
        ("label_names", "ab", "sequence"),
    ],
)
def test_from_dict_rejects_inconsistent_payloads(field: str, value: object, match: str) -> None:
    payload = _calibrator().to_dict()
    payload[field] = value

    with pytest.raises(ConformalValidationError, match=match):
        ClassConditionalLabelwiseConformal.from_dict(payload)


def test_from_dict_rejects_unknown_keys() -> None:
    payload = _calibrator().to_dict()
    payload["alpha"] = 0.1

    with pytest.raises(ConformalValidationError, match="keys differ"):
        ClassConditionalLabelwiseConformal.from_dict(payload)


@pytest.mark.parametrize(
    ("negative_alphas", "positive_alphas", "match"),
    [
        ([0.1, 0.1], {"a": 0.1, "b": 0.1}, "must map every label"),
        ({"a": 0.1, "b": 0.1}, {"a": 0.1}, "keys differ"),
        ({"a": 0.1, "b": 0.1, "c": 0.1}, {"a": 0.1, "b": 0.1}, "keys differ"),
        ({"a": 0.1, 0: 0.1}, {"a": 0.1, "b": 0.1}, "keys must be label names"),
        ({"a": 0.0, "b": 0.1}, {"a": 0.1, "b": 0.1}, "strictly"),
        ({"a": 0.1, "b": 0.1}, {"a": 0.1, "b": 1.0}, "strictly"),
        ({"a": 0.1, "b": 0.1}, {"a": float("nan"), "b": 0.1}, "must lie in"),
        ({"a": True, "b": 0.1}, {"a": 0.1, "b": 0.1}, "numeric"),
        ({"a": "0.1", "b": 0.1}, {"a": 0.1, "b": 0.1}, "numeric"),
    ],
)
def test_fit_requires_explicit_valid_budgets_for_every_label(
    negative_alphas: object,
    positive_alphas: object,
    match: str,
) -> None:
    with pytest.raises(ConformalValidationError, match=match):
        ClassConditionalLabelwiseConformal.fit(
            [[0.2, 0.8], [0.7, 0.3]],
            [[0, 1], [1, 0]],
            label_names=("a", "b"),
            negative_alphas=negative_alphas,  # type: ignore[arg-type]
            positive_alphas=positive_alphas,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("probabilities", "targets", "labels", "match"),
    [
        ([[0.5, 1.1]], [[0, 1]], ("a", "b"), r"\[0, 1\]"),
        ([[0.5, np.nan]], [[0, 1]], ("a", "b"), "finite"),
        ([[0.5, 0.5]], [[0, 0.5]], ("a", "b"), "binary"),
        ([[0.5, 0.5]], [[0, 1]], ("a",), "expected 1"),
        ([[0.5, 0.5]], [[0, 1]], ("a", "a"), "unique"),
    ],
)
def test_fit_rejects_malformed_calibration_inputs(
    probabilities: object,
    targets: object,
    labels: tuple[str, ...],
    match: str,
) -> None:
    with pytest.raises(ConformalValidationError, match=match):
        ClassConditionalLabelwiseConformal.fit(
            probabilities,
            targets,
            label_names=labels,
            negative_alphas=dict.fromkeys(labels, 0.1),
            positive_alphas=dict.fromkeys(labels, 0.1),
        )


def test_direct_construction_enforces_conservative_threshold_and_freezes_inputs() -> None:
    with pytest.raises(ConformalValidationError, match="conservative threshold one"):
        ClassConditionalLabelwiseConformal(
            ("MI",), (0.1,), (0.1,), (0.3,), (0.5,), 40, (35,), (5,), (33,), (6,)
        )
    with pytest.raises(ConformalValidationError, match="integers of at least 0"):
        replace(_calibrator(), negative_counts=(True, 5))
    with pytest.raises(ConformalValidationError, match="integers of at least 1"):
        replace(_calibrator(), positive_ranks=(3, 0))

    thresholds = [0.4, 0.3]
    calibrator = replace(_calibrator(), positive_thresholds=thresholds)
    thresholds[0] = 1.0
    assert calibrator.positive_thresholds == (0.4, 0.3)
    with pytest.raises(ConformalValidationError, match="expected 2"):
        calibrator.predict([[0.5]])
