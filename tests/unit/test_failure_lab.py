from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

import numpy as np
import pytest
import torch

from ecg_trust.constants import SUPERCLASSES
from ecg_trust.contracts import SignalSourceKind, TrustDecision
from ecg_trust.failure_lab import FailureLabError, FailureLabRunner
from ecg_trust.quality.signal_quality import SignalMetadata
from ecg_trust.sentinel_engine import SentinelCaseResult
from ecg_trust.stress import StressKind, StressScenario
from ecg_trust.trust_policy import TrustPolicyResult, TrustReasonCode

NOW = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)


@dataclass(frozen=True)
class FakeResult:
    decision: TrustDecision
    policy: TrustPolicyResult
    calibrated_probabilities: tuple[float, ...] | None


def _result(
    decision: TrustDecision,
    probabilities: tuple[float, ...] | None = None,
) -> SentinelCaseResult:
    reason = {
        TrustDecision.PREDICTION_ALLOWED: TrustReasonCode.ALL_TRUST_GATES_PASSED,
        TrustDecision.REACQUIRE: TrustReasonCode.SIGNAL_REACQUISITION_REQUIRED,
        TrustDecision.UNSUPPORTED_INPUT: TrustReasonCode.OUTSIDE_VALIDATED_DISTRIBUTION,
        TrustDecision.ABSTAIN: TrustReasonCode.CONFORMAL_SET_UNCERTAIN,
        TrustDecision.INVALID_INPUT: TrustReasonCode.INPUT_CONTRACT_INVALID,
    }[decision]
    fake = FakeResult(
        decision=decision,
        policy=TrustPolicyResult(
            policy_version="trust-policy-v1",
            decision=decision,
            reason_codes=(reason,),
            uncertain_labels=("MI",) if reason is TrustReasonCode.CONFORMAL_SET_UNCERTAIN else (),
        ),
        calibrated_probabilities=probabilities,
    )
    return cast(SentinelCaseResult, fake)


class FakeAnalyzer:
    def __init__(self, results: list[SentinelCaseResult]) -> None:
        self.results = results
        self.calls: list[np.ndarray] = []

    def analyze(self, **kwargs: object) -> SentinelCaseResult:
        waveform = np.asarray(kwargs["signal_mv"])
        self.calls.append(waveform.copy())
        return self.results[len(self.calls) - 1]


def _waveform() -> torch.Tensor:
    time = torch.arange(1_000, dtype=torch.float32) / 100.0
    base = 0.5 * torch.sin(2.0 * torch.pi * time)
    return torch.stack(tuple((index + 1) / 12.0 * base for index in range(12))).contiguous()


def _scenario() -> StressScenario:
    return StressScenario(
        scenario_id="gain-150",
        kind=StressKind.GAIN,
        parameters={"factor": 1.5},
    )


def _run(
    analyzer: FakeAnalyzer,
    *,
    source_kind: SignalSourceKind = SignalSourceKind.SYNTHETIC,
):
    return FailureLabRunner(analyzer).run(
        signal_id="case-001",
        waveform_mv=_waveform(),
        metadata=SignalMetadata.canonical(),
        source_kind=source_kind,
        scenario=_scenario(),
        evaluated_at=NOW,
        release_integrity_verified=True,
    )


def test_allowed_pair_reports_exact_probability_deltas_without_waveform_identity() -> None:
    baseline = (0.9, 0.1, 0.2, 0.3, 0.4)
    stressed = (0.8, 0.2, 0.25, 0.25, 0.5)
    analyzer = FakeAnalyzer(
        [
            _result(TrustDecision.PREDICTION_ALLOWED, baseline),
            _result(TrustDecision.PREDICTION_ALLOWED, stressed),
        ]
    )

    trial = _run(analyzer)

    assert trial.probability_delta == pytest.approx((-0.1, 0.1, 0.05, -0.05, 0.1))
    assert not trial.decision_changed
    assert len(analyzer.calls) == 2
    assert not np.array_equal(analyzer.calls[0], analyzer.calls[1])
    public = trial.to_public_dict()
    assert set(public["probability_delta"]) == set(SUPERCLASSES)  # type: ignore[arg-type]
    text = repr(public).lower()
    assert "parent_waveform_sha256" not in text
    assert "output_waveform_sha256" not in text
    assert "case-001" not in text


@pytest.mark.parametrize(
    "blocked",
    [TrustDecision.REACQUIRE, TrustDecision.UNSUPPORTED_INPUT, TrustDecision.ABSTAIN],
)
def test_stressed_blocking_state_hides_stressed_and_delta_probabilities(
    blocked: TrustDecision,
) -> None:
    analyzer = FakeAnalyzer(
        [
            _result(
                TrustDecision.PREDICTION_ALLOWED,
                (0.9, 0.1, 0.2, 0.3, 0.4),
            ),
            _result(blocked),
        ]
    )

    trial = _run(analyzer)
    public = trial.to_public_dict()

    assert trial.decision_changed
    assert trial.probability_delta is None
    assert "probability_delta" not in public
    assert "probabilities" not in public["stressed"]  # type: ignore[operator]


def test_blocked_baseline_stops_before_stress_and_second_analysis() -> None:
    analyzer = FakeAnalyzer([_result(TrustDecision.REACQUIRE)])

    with pytest.raises(FailureLabError, match="baseline"):
        _run(analyzer)

    assert len(analyzer.calls) == 1


def test_trial_is_deterministic_for_the_same_scenario_and_results() -> None:
    results = [
        _result(TrustDecision.PREDICTION_ALLOWED, (0.9, 0.1, 0.2, 0.3, 0.4)),
        _result(TrustDecision.PREDICTION_ALLOWED, (0.8, 0.2, 0.2, 0.3, 0.4)),
    ]

    first = _run(FakeAnalyzer(results)).to_public_dict()
    second = _run(FakeAnalyzer(results)).to_public_dict()

    assert first == second
    assert first["interpretation"] == "controlled_model_sensitivity_not_clinical_shift"


@pytest.mark.parametrize(
    "source_kind",
    [SignalSourceKind.APPROVED_EXAMPLE, SignalSourceKind.AUTHORIZED_UPLOAD],
)
def test_non_synthetic_trial_cannot_cross_the_public_boundary(
    source_kind: SignalSourceKind,
) -> None:
    trial = _run(
        FakeAnalyzer(
            [
                _result(TrustDecision.PREDICTION_ALLOWED, (0.9, 0.1, 0.2, 0.3, 0.4)),
                _result(TrustDecision.ABSTAIN),
            ]
        ),
        source_kind=source_kind,
    )

    with pytest.raises(FailureLabError, match="only synthetic"):
        trial.to_public_dict()
