"""Validation and blinded review of externally proposed ECG perturbations.

This module does not generate ECGs and does not implement a generative medical
model. It evaluates immutable, hash-bound proposals against complete Sentinel
reruns and conservative signal constraints. Every public result carries the
interpretation boundary: not physiological truth, not causal, not treatment advice.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ecg_trust.constants import LEADS
from ecg_trust.contracts import CANONICAL_SAMPLES_PER_LEAD, TrustDecision

FloatArray = NDArray[np.float64]

INTERPRETATION_BOUNDARY: Final = "not physiological truth, not causal, not treatment advice"
PROPOSAL_SCHEMA_VERSION: Final = "counterfactual-proposal-v1"
PUBLIC_RESULT_SCHEMA_VERSION: Final = "counterfactual-validation-summary-v1"
REVIEW_SCHEMA_VERSION: Final = "blinded-cardiology-review-v1"
REVIEW_PREREGISTRATION_SCHEMA_VERSION: Final = "cardiology-review-preregistration-v1"
REVIEW_SUMMARY_SCHEMA_VERSION: Final = "cardiology-review-summary-v1"

CANONICAL_LEADS: Final = LEADS
CANONICAL_SHAPE: Final = (len(LEADS), CANONICAL_SAMPLES_PER_LEAD)

_HASH_RE: Final = re.compile(r"^[a-f0-9]{64}$")
_VERSION_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")
_REASON_RE: Final = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_REVIEWER_PSEUDONYM_RE: Final = re.compile(r"^REV-[A-Z2-7]{8,16}$")


class CounterfactualValidationError(ValueError):
    """A proposal, result contract, or configuration is malformed."""


class ReviewValidationError(ValueError):
    """A blinded review contract or batch is malformed."""


class TargetSuperclass(StrEnum):
    """PTB-XL diagnostic superclasses used only as model outputs."""

    NORM = "NORM"
    MI = "MI"
    STTC = "STTC"
    CD = "CD"
    HYP = "HYP"


class TargetDirection(StrEnum):
    INCREASE = "increase"
    DECREASE = "decrease"


class QualityGateStatus(StrEnum):
    PASS = "pass"
    LIMITED = "limited"
    REACQUIRE = "reacquire"
    INVALID = "invalid"


class CounterfactualStatus(StrEnum):
    ACCEPTED_MODEL_SENSITIVITY = "accepted_model_sensitivity"
    REJECTED_CONSTRAINT = "rejected_constraint"
    EVIDENCE_UNAVAILABLE = "evidence_unavailable"


class CounterfactualReason(StrEnum):
    ACCEPTED_MODEL_SENSITIVITY = "ACCEPTED_MODEL_SENSITIVITY"
    ORIGINAL_SIGNAL_INVALID = "ORIGINAL_SIGNAL_INVALID"
    CANDIDATE_SIGNAL_INVALID = "CANDIDATE_SIGNAL_INVALID"
    ORIGINAL_SIGNAL_NONFINITE = "ORIGINAL_SIGNAL_NONFINITE"
    CANDIDATE_SIGNAL_NONFINITE = "CANDIDATE_SIGNAL_NONFINITE"
    ORIGINAL_HASH_MISMATCH = "ORIGINAL_HASH_MISMATCH"
    CANDIDATE_HASH_MISMATCH = "CANDIDATE_HASH_MISMATCH"
    ORIGINAL_ANALYZER_FAILURE = "ORIGINAL_ANALYZER_FAILURE"
    CANDIDATE_ANALYZER_FAILURE = "CANDIDATE_ANALYZER_FAILURE"
    ORIGINAL_QUALITY_FAILED = "ORIGINAL_QUALITY_FAILED"
    CANDIDATE_QUALITY_FAILED = "CANDIDATE_QUALITY_FAILED"
    ORIGINAL_ARTIFACT_FAILED = "ORIGINAL_ARTIFACT_FAILED"
    CANDIDATE_ARTIFACT_FAILED = "CANDIDATE_ARTIFACT_FAILED"
    ORIGINAL_OOD_FAILED = "ORIGINAL_OOD_FAILED"
    CANDIDATE_OOD_FAILED = "CANDIDATE_OOD_FAILED"
    ORIGINAL_UNCERTAINTY_FAILED = "ORIGINAL_UNCERTAINTY_FAILED"
    CANDIDATE_UNCERTAINTY_FAILED = "CANDIDATE_UNCERTAINTY_FAILED"
    ORIGINAL_SENTINEL_NOT_ALLOWED = "ORIGINAL_SENTINEL_NOT_ALLOWED"
    CANDIDATE_SENTINEL_NOT_ALLOWED = "CANDIDATE_SENTINEL_NOT_ALLOWED"
    ORIGINAL_LIMB_IDENTITY_FAILED = "ORIGINAL_LIMB_IDENTITY_FAILED"
    CANDIDATE_LIMB_IDENTITY_FAILED = "CANDIDATE_LIMB_IDENTITY_FAILED"
    LIMB_IDENTITY_DEGRADATION_EXCEEDED = "LIMB_IDENTITY_DEGRADATION_EXCEEDED"
    GLOBAL_RMS_DELTA_EXCEEDED = "GLOBAL_RMS_DELTA_EXCEEDED"
    GLOBAL_LINF_DELTA_EXCEEDED = "GLOBAL_LINF_DELTA_EXCEEDED"
    NO_MATERIAL_SIGNAL_CHANGE = "NO_MATERIAL_SIGNAL_CHANGE"
    CHANGE_SUPPORT_EXCEEDED = "CHANGE_SUPPORT_EXCEEDED"
    PER_LEAD_RMS_DELTA_EXCEEDED = "PER_LEAD_RMS_DELTA_EXCEEDED"
    PER_LEAD_LINF_DELTA_EXCEEDED = "PER_LEAD_LINF_DELTA_EXCEEDED"
    ORIGINAL_AMPLITUDE_FAILED = "ORIGINAL_AMPLITUDE_FAILED"
    CANDIDATE_AMPLITUDE_FAILED = "CANDIDATE_AMPLITUDE_FAILED"
    ORIGINAL_STEP_FAILED = "ORIGINAL_STEP_FAILED"
    CANDIDATE_STEP_FAILED = "CANDIDATE_STEP_FAILED"
    TARGET_DIRECTION_NOT_MET = "TARGET_DIRECTION_NOT_MET"
    MODEL_EFFECT_BELOW_MINIMUM = "MODEL_EFFECT_BELOW_MINIMUM"


class ReviewEvidenceStatus(StrEnum):
    INSUFFICIENT_REVIEW_COUNT = "insufficient_review_count"
    DESCRIPTIVE_ONLY = "descriptive_only"
    PREREGISTERED_THRESHOLD_MET = "preregistered_threshold_met"


def canonical_json(value: object) -> str:
    """Encode deterministic JSON suitable for hashing and review artifacts."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise CounterfactualValidationError("value is not canonical JSON data") from error


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class _SignalContractError(CounterfactualValidationError):
    def __init__(self, *, nonfinite: bool = False) -> None:
        super().__init__("waveform does not satisfy the canonical physical-mV contract")
        self.nonfinite = nonfinite


def _canonical_signal(signal_mv: ArrayLike) -> FloatArray:
    try:
        raw = np.asarray(signal_mv)
    except (TypeError, ValueError) as error:
        raise _SignalContractError from error
    if (
        raw.shape != CANONICAL_SHAPE
        or np.iscomplexobj(raw)
        or np.issubdtype(raw.dtype, np.bool_)
        or not np.issubdtype(raw.dtype, np.number)
    ):
        raise _SignalContractError
    try:
        # Detach so the read-only analyzer view cannot change caller ownership.
        signal = np.array(raw, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError) as error:
        raise _SignalContractError from error
    if not np.isfinite(signal).all():
        raise _SignalContractError(nonfinite=True)
    signal.flags.writeable = False
    return signal


def canonical_waveform_sha256(signal_mv: ArrayLike) -> str:
    """Hash one canonical 12x1000 physical-mV waveform without serializing it."""

    signal = _canonical_signal(signal_mv)
    little_endian = signal.astype(np.dtype("<f8"), copy=False)
    digest = hashlib.sha256()
    digest.update(b"ecg-trust:canonical-12x1000-physical-mv-f64le:v1\0")
    digest.update(little_endian.tobytes(order="C"))
    return digest.hexdigest()


def _validate_hash(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise CounterfactualValidationError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _validate_reason_codes(value: object) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or not 1 <= len(value) <= 16:
        raise CounterfactualValidationError("one to sixteen reason codes are required")
    reasons: list[str] = []
    for reason in value:
        if not isinstance(reason, str) or _REASON_RE.fullmatch(reason) is None:
            raise CounterfactualValidationError("reason codes must use the machine format")
        reasons.append(reason)
    if len(reasons) != len(set(reasons)):
        raise CounterfactualValidationError("reason codes must be unique")
    return tuple(reasons)


@dataclass(frozen=True, slots=True)
class CounterfactualProposal:
    """Immutable provenance binding two waveform hashes to one requested direction."""

    original_waveform_sha256: str
    candidate_waveform_sha256: str
    target_superclass: TargetSuperclass
    target_direction: TargetDirection
    method_artifact_sha256: str
    method_version: str
    seed: int

    def __post_init__(self) -> None:
        _validate_hash(self.original_waveform_sha256, field_name="original_waveform_sha256")
        _validate_hash(self.candidate_waveform_sha256, field_name="candidate_waveform_sha256")
        _validate_hash(self.method_artifact_sha256, field_name="method_artifact_sha256")
        if not isinstance(self.target_superclass, TargetSuperclass):
            raise CounterfactualValidationError("target_superclass is invalid")
        if not isinstance(self.target_direction, TargetDirection):
            raise CounterfactualValidationError("target_direction is invalid")
        if (
            not isinstance(self.method_version, str)
            or _VERSION_RE.fullmatch(self.method_version) is None
        ):
            raise CounterfactualValidationError("method_version must be a bounded version token")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**63
        ):
            raise CounterfactualValidationError("seed must be a non-negative signed 64-bit integer")

    @classmethod
    def bind(
        cls,
        *,
        original_signal_mv: ArrayLike,
        candidate_signal_mv: ArrayLike,
        target_superclass: TargetSuperclass,
        target_direction: TargetDirection,
        method_artifact_sha256: str,
        method_version: str,
        seed: int,
    ) -> Self:
        return cls(
            original_waveform_sha256=canonical_waveform_sha256(original_signal_mv),
            candidate_waveform_sha256=canonical_waveform_sha256(candidate_signal_mv),
            target_superclass=target_superclass,
            target_direction=target_direction,
            method_artifact_sha256=method_artifact_sha256,
            method_version=method_version,
            seed=seed,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": PROPOSAL_SCHEMA_VERSION,
            "original_waveform_sha256": self.original_waveform_sha256,
            "candidate_waveform_sha256": self.candidate_waveform_sha256,
            "target_superclass": self.target_superclass.value,
            "target_direction": self.target_direction.value,
            "method_artifact_sha256": self.method_artifact_sha256,
            "method_version": self.method_version,
            "seed": self.seed,
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_mapping())

    def metadata_sha256(self) -> str:
        return canonical_sha256(self.to_mapping())


@dataclass(frozen=True, slots=True)
class SentinelAnalysis:
    """Complete injected Sentinel result; scores exist only after every gate passes."""

    decision: TrustDecision
    reason_codes: tuple[str, ...]
    quality_status: QualityGateStatus
    artifact_free: bool
    ood_supported: bool
    uncertainty_passed: bool
    superclass_scores: tuple[tuple[TargetSuperclass, float], ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, TrustDecision):
            raise CounterfactualValidationError("analysis decision is invalid")
        if not isinstance(self.quality_status, QualityGateStatus):
            raise CounterfactualValidationError("analysis quality status is invalid")
        reasons = _validate_reason_codes(self.reason_codes)
        object.__setattr__(self, "reason_codes", reasons)
        if any(
            not isinstance(flag, bool)
            for flag in (self.artifact_free, self.ood_supported, self.uncertainty_passed)
        ):
            raise CounterfactualValidationError("analysis gate flags must be booleans")

        allowed = self.decision is TrustDecision.PREDICTION_ALLOWED
        if allowed and reasons != ("ALL_TRUST_GATES_PASSED",):
            raise CounterfactualValidationError(
                "PREDICTION_ALLOWED requires only ALL_TRUST_GATES_PASSED"
            )
        if not allowed and "ALL_TRUST_GATES_PASSED" in reasons:
            raise CounterfactualValidationError(
                "ALL_TRUST_GATES_PASSED is reserved for allowed analysis"
            )
        gates_passed = (
            self.quality_status is QualityGateStatus.PASS
            and self.artifact_free
            and self.ood_supported
            and self.uncertainty_passed
        )
        if allowed and not gates_passed:
            raise CounterfactualValidationError(
                "PREDICTION_ALLOWED requires quality, artifact, OOD, and uncertainty passes"
            )
        if not allowed:
            if self.superclass_scores is not None:
                raise CounterfactualValidationError(
                    "scores are forbidden when prediction is not allowed"
                )
            return
        if self.superclass_scores is None:
            raise CounterfactualValidationError("a complete allowed analysis requires all scores")
        if not isinstance(self.superclass_scores, tuple) or any(
            not isinstance(item, tuple) or len(item) != 2 for item in self.superclass_scores
        ):
            raise CounterfactualValidationError("superclass scores must be immutable pairs")
        expected = tuple(TargetSuperclass)
        observed = tuple(item[0] for item in self.superclass_scores)
        if observed != expected:
            raise CounterfactualValidationError(
                "superclass scores must use the complete canonical order"
            )
        for _, score in self.superclass_scores:
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
            ):
                raise CounterfactualValidationError(
                    "superclass scores must be finite probabilities"
                )

    def score_for(self, target: TargetSuperclass) -> float:
        if self.superclass_scores is None:
            raise CounterfactualValidationError("scores are unavailable for a gated analysis")
        return float(dict(self.superclass_scores)[target])


class CompleteSentinelAnalyzer(Protocol):
    """Injected full-pipeline analyzer; the evaluator never bypasses its gates."""

    def analyze(self, signal_mv: FloatArray) -> SentinelAnalysis: ...


@dataclass(frozen=True, slots=True)
class PlausibilityConfig:
    """Frozen research bounds; these thresholds are not physiological truth."""

    version: str = "counterfactual-plausibility-v1"
    change_epsilon_mv: float = 0.005
    max_global_rms_delta_mv: float = 0.25
    max_global_linf_delta_mv: float = 1.0
    max_change_support_fraction: float = 0.20
    max_per_lead_rms_delta_mv: float = 0.40
    max_per_lead_linf_delta_mv: float = 1.20
    max_limb_identity_rms_mv: float = 0.05
    max_limb_identity_degradation_mv: float = 0.02
    max_signal_absolute_mv: float = 8.0
    max_signal_step_mv: float = 4.0
    minimum_model_sensitivity_effect: float = 0.02

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or _VERSION_RE.fullmatch(self.version) is None:
            raise CounterfactualValidationError("plausibility version is invalid")
        positive = (
            self.change_epsilon_mv,
            self.max_global_rms_delta_mv,
            self.max_global_linf_delta_mv,
            self.max_per_lead_rms_delta_mv,
            self.max_per_lead_linf_delta_mv,
            self.max_limb_identity_rms_mv,
            self.max_limb_identity_degradation_mv,
            self.max_signal_absolute_mv,
            self.max_signal_step_mv,
            self.minimum_model_sensitivity_effect,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in positive
        ):
            raise CounterfactualValidationError(
                "plausibility thresholds must be finite and positive"
            )
        if (
            isinstance(self.max_change_support_fraction, bool)
            or not isinstance(self.max_change_support_fraction, (int, float))
            or not math.isfinite(float(self.max_change_support_fraction))
            or not 0.0 < float(self.max_change_support_fraction) <= 1.0
        ):
            raise CounterfactualValidationError("support fraction must be in (0, 1]")
        if self.change_epsilon_mv >= self.max_global_linf_delta_mv:
            raise CounterfactualValidationError("change epsilon must be below the L-infinity bound")
        if self.max_global_linf_delta_mv > self.max_per_lead_linf_delta_mv:
            raise CounterfactualValidationError(
                "global L-infinity cannot exceed its per-lead bound"
            )
        if self.minimum_model_sensitivity_effect > 1.0:
            raise CounterfactualValidationError("model-sensitivity effect cannot exceed one")


@dataclass(frozen=True, slots=True)
class LeadDeltaSummary:
    lead_name: str
    rms_delta_mv: float
    linf_delta_mv: float
    changed_fraction: float

    def to_mapping(self) -> dict[str, object]:
        return {
            "lead_name": self.lead_name,
            "rms_delta_mv": self.rms_delta_mv,
            "linf_delta_mv": self.linf_delta_mv,
            "changed_fraction": self.changed_fraction,
        }


@dataclass(frozen=True, slots=True)
class PerturbationMetrics:
    global_rms_delta_mv: float
    global_linf_delta_mv: float
    change_support_fraction: float
    change_support_sparsity: float
    original_limb_identity_rms_mv: float
    candidate_limb_identity_rms_mv: float
    limb_identity_degradation_mv: float
    original_maximum_absolute_mv: float
    candidate_maximum_absolute_mv: float
    original_maximum_step_mv: float
    candidate_maximum_step_mv: float
    per_lead: tuple[LeadDeltaSummary, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "global_rms_delta_mv": self.global_rms_delta_mv,
            "global_linf_delta_mv": self.global_linf_delta_mv,
            "change_support_fraction": self.change_support_fraction,
            "change_support_sparsity": self.change_support_sparsity,
            "original_limb_identity_rms_mv": self.original_limb_identity_rms_mv,
            "candidate_limb_identity_rms_mv": self.candidate_limb_identity_rms_mv,
            "limb_identity_degradation_mv": self.limb_identity_degradation_mv,
            "original_maximum_absolute_mv": self.original_maximum_absolute_mv,
            "candidate_maximum_absolute_mv": self.candidate_maximum_absolute_mv,
            "original_maximum_step_mv": self.original_maximum_step_mv,
            "candidate_maximum_step_mv": self.candidate_maximum_step_mv,
            "per_lead_deltas": [lead.to_mapping() for lead in self.per_lead],
        }


def _limb_identity_rms(signal: FloatArray) -> float:
    lead_i, lead_ii, lead_iii, avr, avl, avf = signal[:6]
    residuals = np.stack(
        (
            lead_i + lead_iii - lead_ii,
            avr + (lead_i + lead_ii) / 2.0,
            avl - (lead_i - lead_ii / 2.0),
            avf - (lead_ii - lead_i / 2.0),
        )
    )
    return float(np.sqrt(np.mean(np.square(residuals))))


def _maximum_step(signal: FloatArray) -> float:
    return float(np.max(np.abs(np.diff(signal, axis=1))))


def _perturbation_metrics(
    original: FloatArray,
    candidate: FloatArray,
    config: PlausibilityConfig,
) -> PerturbationMetrics:
    delta = candidate - original
    absolute = np.abs(delta)
    support = absolute > config.change_epsilon_mv
    per_lead = tuple(
        LeadDeltaSummary(
            lead_name=lead_name,
            rms_delta_mv=float(np.sqrt(np.mean(np.square(delta[index])))),
            linf_delta_mv=float(np.max(absolute[index])),
            changed_fraction=float(np.mean(support[index])),
        )
        for index, lead_name in enumerate(CANONICAL_LEADS)
    )
    original_identity = _limb_identity_rms(original)
    candidate_identity = _limb_identity_rms(candidate)
    support_fraction = float(np.mean(support))
    values = (
        float(np.sqrt(np.mean(np.square(delta)))),
        float(np.max(absolute)),
        support_fraction,
        1.0 - support_fraction,
        original_identity,
        candidate_identity,
        max(0.0, candidate_identity - original_identity),
        float(np.max(np.abs(original))),
        float(np.max(np.abs(candidate))),
        _maximum_step(original),
        _maximum_step(candidate),
    )
    if any(not math.isfinite(value) for value in values):
        raise CounterfactualValidationError("perturbation metrics must be finite")
    return PerturbationMetrics(
        global_rms_delta_mv=values[0],
        global_linf_delta_mv=values[1],
        change_support_fraction=values[2],
        change_support_sparsity=values[3],
        original_limb_identity_rms_mv=values[4],
        candidate_limb_identity_rms_mv=values[5],
        limb_identity_degradation_mv=values[6],
        original_maximum_absolute_mv=values[7],
        candidate_maximum_absolute_mv=values[8],
        original_maximum_step_mv=values[9],
        candidate_maximum_step_mv=values[10],
        per_lead=per_lead,
    )


@dataclass(frozen=True, slots=True)
class CounterfactualEvaluation:
    """Internal validation result; waveform data are never retained."""

    status: CounterfactualStatus
    reason_codes: tuple[CounterfactualReason, ...]
    proposal_metadata_sha256: str
    target_superclass: TargetSuperclass
    target_direction: TargetDirection
    method_version: str
    plausibility_config_version: str
    metrics: PerturbationMetrics | None = None
    original_target_score: float | None = None
    candidate_target_score: float | None = None
    model_sensitivity_effect: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, CounterfactualStatus):
            raise CounterfactualValidationError("evaluation status is invalid")
        if not isinstance(self.target_superclass, TargetSuperclass) or not isinstance(
            self.target_direction, TargetDirection
        ):
            raise CounterfactualValidationError("evaluation target contract is invalid")
        if (
            _VERSION_RE.fullmatch(self.method_version) is None
            or _VERSION_RE.fullmatch(self.plausibility_config_version) is None
        ):
            raise CounterfactualValidationError("evaluation versions are invalid")
        _validate_hash(self.proposal_metadata_sha256, field_name="proposal_metadata_sha256")
        if not self.reason_codes or len(set(self.reason_codes)) != len(self.reason_codes):
            raise CounterfactualValidationError("evaluation reasons must be non-empty and unique")
        if any(not isinstance(reason, CounterfactualReason) for reason in self.reason_codes):
            raise CounterfactualValidationError("evaluation reasons must use the closed vocabulary")
        if self.status is CounterfactualStatus.ACCEPTED_MODEL_SENSITIVITY:
            if self.reason_codes != (CounterfactualReason.ACCEPTED_MODEL_SENSITIVITY,):
                raise CounterfactualValidationError(
                    "accepted status requires its exact reason code"
                )
        elif CounterfactualReason.ACCEPTED_MODEL_SENSITIVITY in self.reason_codes:
            raise CounterfactualValidationError(
                "accepted reason is forbidden for non-accepted status"
            )
        if self.status is CounterfactualStatus.ACCEPTED_MODEL_SENSITIVITY and (
            self.metrics is None
            or self.original_target_score is None
            or self.candidate_target_score is None
            or self.model_sensitivity_effect is None
        ):
            raise CounterfactualValidationError("accepted evaluations require complete evidence")
        numeric = (
            self.original_target_score,
            self.candidate_target_score,
            self.model_sensitivity_effect,
        )
        if any(value is not None and not math.isfinite(value) for value in numeric):
            raise CounterfactualValidationError("evaluation scores and effect must be finite")
        if any(
            value is not None and not 0.0 <= value <= 1.0
            for value in (self.original_target_score, self.candidate_target_score)
        ):
            raise CounterfactualValidationError("evaluation target scores must be probabilities")
        if (
            self.model_sensitivity_effect is not None
            and not -1.0 <= self.model_sensitivity_effect <= 1.0
        ):
            raise CounterfactualValidationError("model-sensitivity effect must be in [-1, 1]")

    def to_public_summary(self) -> dict[str, object]:
        """Return a disclosure-gated summary with no waveform hashes or identifiers."""

        summary: dict[str, object] = {
            "schema_version": PUBLIC_RESULT_SCHEMA_VERSION,
            "status": self.status.value,
            "reason_codes": [reason.value for reason in self.reason_codes],
            "target_superclass": self.target_superclass.value,
            "target_direction": self.target_direction.value,
            "method_version": self.method_version,
            "plausibility_config_version": self.plausibility_config_version,
            "interpretation_boundary": INTERPRETATION_BOUNDARY,
        }
        if self.metrics is not None:
            summary["perturbation_metrics"] = self.metrics.to_mapping()
        if self.status is CounterfactualStatus.ACCEPTED_MODEL_SENSITIVITY:
            summary["original_target_score"] = self.original_target_score
            summary["candidate_target_score"] = self.candidate_target_score
            summary["model_sensitivity_effect"] = self.model_sensitivity_effect
        return summary

    def canonical_public_json(self) -> str:
        return canonical_json(self.to_public_summary())

    def public_summary_sha256(self) -> str:
        return canonical_sha256(self.to_public_summary())


def _analysis_gate_reasons(
    analysis: SentinelAnalysis,
    *,
    original: bool,
) -> tuple[CounterfactualReason, ...]:
    prefix = "ORIGINAL" if original else "CANDIDATE"
    reasons: list[CounterfactualReason] = []
    if analysis.quality_status is not QualityGateStatus.PASS:
        reasons.append(CounterfactualReason[f"{prefix}_QUALITY_FAILED"])
    if not analysis.artifact_free:
        reasons.append(CounterfactualReason[f"{prefix}_ARTIFACT_FAILED"])
    if not analysis.ood_supported:
        reasons.append(CounterfactualReason[f"{prefix}_OOD_FAILED"])
    if not analysis.uncertainty_passed:
        reasons.append(CounterfactualReason[f"{prefix}_UNCERTAINTY_FAILED"])
    if analysis.decision is not TrustDecision.PREDICTION_ALLOWED:
        reasons.append(CounterfactualReason[f"{prefix}_SENTINEL_NOT_ALLOWED"])
    return tuple(reasons)


def _constraint_reasons(
    metrics: PerturbationMetrics,
    config: PlausibilityConfig,
) -> tuple[CounterfactualReason, ...]:
    reasons: list[CounterfactualReason] = []
    if metrics.original_limb_identity_rms_mv > config.max_limb_identity_rms_mv:
        reasons.append(CounterfactualReason.ORIGINAL_LIMB_IDENTITY_FAILED)
    if metrics.candidate_limb_identity_rms_mv > config.max_limb_identity_rms_mv:
        reasons.append(CounterfactualReason.CANDIDATE_LIMB_IDENTITY_FAILED)
    if metrics.limb_identity_degradation_mv > config.max_limb_identity_degradation_mv:
        reasons.append(CounterfactualReason.LIMB_IDENTITY_DEGRADATION_EXCEEDED)
    if metrics.global_rms_delta_mv > config.max_global_rms_delta_mv:
        reasons.append(CounterfactualReason.GLOBAL_RMS_DELTA_EXCEEDED)
    if metrics.global_linf_delta_mv > config.max_global_linf_delta_mv:
        reasons.append(CounterfactualReason.GLOBAL_LINF_DELTA_EXCEEDED)
    if metrics.change_support_fraction == 0.0:
        reasons.append(CounterfactualReason.NO_MATERIAL_SIGNAL_CHANGE)
    if metrics.change_support_fraction > config.max_change_support_fraction:
        reasons.append(CounterfactualReason.CHANGE_SUPPORT_EXCEEDED)
    if any(lead.rms_delta_mv > config.max_per_lead_rms_delta_mv for lead in metrics.per_lead):
        reasons.append(CounterfactualReason.PER_LEAD_RMS_DELTA_EXCEEDED)
    if any(lead.linf_delta_mv > config.max_per_lead_linf_delta_mv for lead in metrics.per_lead):
        reasons.append(CounterfactualReason.PER_LEAD_LINF_DELTA_EXCEEDED)
    if metrics.original_maximum_absolute_mv > config.max_signal_absolute_mv:
        reasons.append(CounterfactualReason.ORIGINAL_AMPLITUDE_FAILED)
    if metrics.candidate_maximum_absolute_mv > config.max_signal_absolute_mv:
        reasons.append(CounterfactualReason.CANDIDATE_AMPLITUDE_FAILED)
    if metrics.original_maximum_step_mv > config.max_signal_step_mv:
        reasons.append(CounterfactualReason.ORIGINAL_STEP_FAILED)
    if metrics.candidate_maximum_step_mv > config.max_signal_step_mv:
        reasons.append(CounterfactualReason.CANDIDATE_STEP_FAILED)
    return tuple(reasons)


class CounterfactualEvaluator:
    """Validate externally supplied proposals without generating or repairing ECGs."""

    def __init__(
        self,
        analyzer: CompleteSentinelAnalyzer,
        *,
        config: PlausibilityConfig | None = None,
    ) -> None:
        self._analyzer = analyzer
        self._config = config or PlausibilityConfig()

    def _result(
        self,
        proposal: CounterfactualProposal,
        *,
        status: CounterfactualStatus,
        reasons: tuple[CounterfactualReason, ...],
        metrics: PerturbationMetrics | None = None,
        original_score: float | None = None,
        candidate_score: float | None = None,
        effect: float | None = None,
    ) -> CounterfactualEvaluation:
        return CounterfactualEvaluation(
            status=status,
            reason_codes=tuple(dict.fromkeys(reasons)),
            proposal_metadata_sha256=proposal.metadata_sha256(),
            target_superclass=proposal.target_superclass,
            target_direction=proposal.target_direction,
            method_version=proposal.method_version,
            plausibility_config_version=self._config.version,
            metrics=metrics,
            original_target_score=original_score,
            candidate_target_score=candidate_score,
            model_sensitivity_effect=effect,
        )

    def evaluate(
        self,
        proposal: CounterfactualProposal,
        original_signal_mv: ArrayLike,
        candidate_signal_mv: ArrayLike,
    ) -> CounterfactualEvaluation:
        if not isinstance(proposal, CounterfactualProposal):
            raise CounterfactualValidationError("evaluate requires immutable proposal metadata")

        original: FloatArray | None = None
        candidate: FloatArray | None = None
        input_reasons: list[CounterfactualReason] = []
        try:
            original = _canonical_signal(original_signal_mv)
        except _SignalContractError as error:
            input_reasons.append(
                CounterfactualReason.ORIGINAL_SIGNAL_NONFINITE
                if error.nonfinite
                else CounterfactualReason.ORIGINAL_SIGNAL_INVALID
            )
        try:
            candidate = _canonical_signal(candidate_signal_mv)
        except _SignalContractError as error:
            input_reasons.append(
                CounterfactualReason.CANDIDATE_SIGNAL_NONFINITE
                if error.nonfinite
                else CounterfactualReason.CANDIDATE_SIGNAL_INVALID
            )
        if input_reasons:
            status = (
                CounterfactualStatus.EVIDENCE_UNAVAILABLE
                if any(reason.name.startswith("ORIGINAL_") for reason in input_reasons)
                else CounterfactualStatus.REJECTED_CONSTRAINT
            )
            return self._result(proposal, status=status, reasons=tuple(input_reasons))
        assert original is not None and candidate is not None

        hash_reasons: list[CounterfactualReason] = []
        if canonical_waveform_sha256(original) != proposal.original_waveform_sha256:
            hash_reasons.append(CounterfactualReason.ORIGINAL_HASH_MISMATCH)
        if canonical_waveform_sha256(candidate) != proposal.candidate_waveform_sha256:
            hash_reasons.append(CounterfactualReason.CANDIDATE_HASH_MISMATCH)
        if hash_reasons:
            status = (
                CounterfactualStatus.EVIDENCE_UNAVAILABLE
                if CounterfactualReason.ORIGINAL_HASH_MISMATCH in hash_reasons
                else CounterfactualStatus.REJECTED_CONSTRAINT
            )
            return self._result(proposal, status=status, reasons=tuple(hash_reasons))

        metrics = _perturbation_metrics(original, candidate, self._config)
        original_analysis: SentinelAnalysis | None = None
        candidate_analysis: SentinelAnalysis | None = None
        analyzer_reasons: list[CounterfactualReason] = []
        try:
            observed = self._analyzer.analyze(original)
            if not isinstance(observed, SentinelAnalysis):
                raise CounterfactualValidationError("analyzer returned an invalid result")
            original_analysis = observed
        except Exception:
            analyzer_reasons.append(CounterfactualReason.ORIGINAL_ANALYZER_FAILURE)
        try:
            observed = self._analyzer.analyze(candidate)
            if not isinstance(observed, SentinelAnalysis):
                raise CounterfactualValidationError("analyzer returned an invalid result")
            candidate_analysis = observed
        except Exception:
            analyzer_reasons.append(CounterfactualReason.CANDIDATE_ANALYZER_FAILURE)
        if analyzer_reasons:
            return self._result(
                proposal,
                status=CounterfactualStatus.EVIDENCE_UNAVAILABLE,
                reasons=tuple(analyzer_reasons),
                metrics=metrics,
            )
        assert original_analysis is not None and candidate_analysis is not None

        original_gates = _analysis_gate_reasons(original_analysis, original=True)
        candidate_gates = _analysis_gate_reasons(candidate_analysis, original=False)
        if original_gates or candidate_gates:
            candidate_constraint_failure = any(
                reason
                in {
                    CounterfactualReason.CANDIDATE_QUALITY_FAILED,
                    CounterfactualReason.CANDIDATE_ARTIFACT_FAILED,
                }
                for reason in candidate_gates
            )
            status = (
                CounterfactualStatus.REJECTED_CONSTRAINT
                if not original_gates and candidate_constraint_failure
                else CounterfactualStatus.EVIDENCE_UNAVAILABLE
            )
            return self._result(
                proposal,
                status=status,
                reasons=original_gates + candidate_gates,
                metrics=metrics,
            )

        original_score = original_analysis.score_for(proposal.target_superclass)
        candidate_score = candidate_analysis.score_for(proposal.target_superclass)
        signed_effect = candidate_score - original_score
        effect = (
            signed_effect
            if proposal.target_direction is TargetDirection.INCREASE
            else -signed_effect
        )
        if not all(math.isfinite(value) for value in (original_score, candidate_score, effect)):
            return self._result(
                proposal,
                status=CounterfactualStatus.EVIDENCE_UNAVAILABLE,
                reasons=(CounterfactualReason.CANDIDATE_ANALYZER_FAILURE,),
                metrics=metrics,
            )

        constraints = list(_constraint_reasons(metrics, self._config))
        if CounterfactualReason.ORIGINAL_LIMB_IDENTITY_FAILED in constraints or any(
            reason
            in {
                CounterfactualReason.ORIGINAL_AMPLITUDE_FAILED,
                CounterfactualReason.ORIGINAL_STEP_FAILED,
            }
            for reason in constraints
        ):
            return self._result(
                proposal,
                status=CounterfactualStatus.EVIDENCE_UNAVAILABLE,
                reasons=tuple(constraints),
                metrics=metrics,
                original_score=original_score,
                candidate_score=candidate_score,
                effect=effect,
            )
        if effect <= 0.0:
            constraints.append(CounterfactualReason.TARGET_DIRECTION_NOT_MET)
        elif effect < self._config.minimum_model_sensitivity_effect:
            constraints.append(CounterfactualReason.MODEL_EFFECT_BELOW_MINIMUM)
        if constraints:
            return self._result(
                proposal,
                status=CounterfactualStatus.REJECTED_CONSTRAINT,
                reasons=tuple(constraints),
                metrics=metrics,
                original_score=original_score,
                candidate_score=candidate_score,
                effect=effect,
            )
        return self._result(
            proposal,
            status=CounterfactualStatus.ACCEPTED_MODEL_SENSITIVITY,
            reasons=(CounterfactualReason.ACCEPTED_MODEL_SENSITIVITY,),
            metrics=metrics,
            original_score=original_score,
            candidate_score=candidate_score,
            effect=effect,
        )


@dataclass(frozen=True, slots=True)
class ReviewPreregistration:
    """Frozen threshold contract established before blinded review aggregation."""

    version: str = "cardiology-review-v1"
    minimum_reviews_per_proposal: int = 3
    minimum_proposal_count: int = 20
    minimum_unique_reviewer_count: int = 3
    minimum_total_review_count: int = 60
    usefulness_vote_threshold: float = 0.75

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or _VERSION_RE.fullmatch(self.version) is None:
            raise ReviewValidationError("review preregistration version is invalid")
        integer_fields = (
            self.minimum_reviews_per_proposal,
            self.minimum_proposal_count,
            self.minimum_unique_reviewer_count,
            self.minimum_total_review_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 2
            for value in integer_fields
        ):
            raise ReviewValidationError("review count thresholds must be integers of at least two")
        required_total = self.minimum_reviews_per_proposal * self.minimum_proposal_count
        if self.minimum_total_review_count < required_total:
            raise ReviewValidationError("minimum total reviews must cover every eligible proposal")
        if self.minimum_unique_reviewer_count > self.minimum_total_review_count:
            raise ReviewValidationError("unique reviewer minimum cannot exceed total reviews")
        if (
            isinstance(self.usefulness_vote_threshold, bool)
            or not isinstance(self.usefulness_vote_threshold, (int, float))
            or not math.isfinite(float(self.usefulness_vote_threshold))
            or not 0.5 < float(self.usefulness_vote_threshold) <= 1.0
        ):
            raise ReviewValidationError("usefulness threshold must be in (0.5, 1]")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": REVIEW_PREREGISTRATION_SCHEMA_VERSION,
            "version": self.version,
            "minimum_reviews_per_proposal": self.minimum_reviews_per_proposal,
            "minimum_proposal_count": self.minimum_proposal_count,
            "minimum_unique_reviewer_count": self.minimum_unique_reviewer_count,
            "minimum_total_review_count": self.minimum_total_review_count,
            "usefulness_vote_threshold": self.usefulness_vote_threshold,
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_mapping())

    def sha256(self) -> str:
        return canonical_sha256(self.to_mapping())


@dataclass(frozen=True, slots=True)
class BlindedCardiologyReview:
    """Structured individual input; no names, free text, waveform, or clinical advice."""

    proposal_metadata_sha256: str
    reviewer_pseudonym: str
    morphology_plausibility_rating: int
    artifact_present: bool
    useful_for_model_review: bool

    def __post_init__(self) -> None:
        try:
            _validate_hash(self.proposal_metadata_sha256, field_name="proposal_metadata_sha256")
        except CounterfactualValidationError as error:
            raise ReviewValidationError("proposal digest is invalid") from error
        if (
            not isinstance(self.reviewer_pseudonym, str)
            or _REVIEWER_PSEUDONYM_RE.fullmatch(self.reviewer_pseudonym) is None
        ):
            raise ReviewValidationError("reviewer pseudonym must use the blinded token format")
        if (
            isinstance(self.morphology_plausibility_rating, bool)
            or not isinstance(self.morphology_plausibility_rating, int)
            or not 1 <= self.morphology_plausibility_rating <= 5
        ):
            raise ReviewValidationError("morphology rating must be an integer from one to five")
        if not isinstance(self.artifact_present, bool) or not isinstance(
            self.useful_for_model_review, bool
        ):
            raise ReviewValidationError("review votes must be booleans")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "proposal_metadata_sha256": self.proposal_metadata_sha256,
            "reviewer_pseudonym": self.reviewer_pseudonym,
            "morphology_plausibility_rating": self.morphology_plausibility_rating,
            "artifact_present": self.artifact_present,
            "useful_for_model_review": self.useful_for_model_review,
            "blinded": True,
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_mapping())

    def sha256(self) -> str:
        return canonical_sha256(self.to_mapping())


@dataclass(frozen=True, slots=True)
class CardiologyReviewSummary:
    """Aggregate-only blinded review result with a preregistered claim gate."""

    status: ReviewEvidenceStatus
    preregistration_version: str
    preregistration_sha256: str
    preregistered_usefulness_vote_threshold: float
    total_review_count: int
    eligible_review_count: int
    eligible_proposal_count: int
    unique_reviewer_count: int
    mean_morphology_rating: float | None
    median_morphology_rating: float | None
    morphology_rating_population_sd: float | None
    artifact_rate: float | None
    model_review_usefulness_vote_rate: float | None
    pairwise_usefulness_agreement: float | None
    pairwise_usefulness_kappa: float | None
    pairwise_morphology_absolute_difference: float | None
    usefulness_claim_allowed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.status, ReviewEvidenceStatus):
            raise ReviewValidationError("review summary status is invalid")
        if _HASH_RE.fullmatch(self.preregistration_sha256) is None:
            raise ReviewValidationError("review summary preregistration digest is invalid")
        if (
            isinstance(self.preregistered_usefulness_vote_threshold, bool)
            or not isinstance(self.preregistered_usefulness_vote_threshold, (int, float))
            or not math.isfinite(float(self.preregistered_usefulness_vote_threshold))
            or not 0.5 < float(self.preregistered_usefulness_vote_threshold) <= 1.0
        ):
            raise ReviewValidationError("review summary threshold is invalid")
        counts = (
            self.total_review_count,
            self.eligible_review_count,
            self.eligible_proposal_count,
            self.unique_reviewer_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts
        ):
            raise ReviewValidationError("review summary counts must be non-negative integers")
        if self.eligible_review_count > self.total_review_count:
            raise ReviewValidationError("eligible reviews cannot exceed total reviews")
        if self.unique_reviewer_count > self.eligible_review_count:
            raise ReviewValidationError("reviewer count cannot exceed eligible reviews")
        unit_interval_values = (
            self.artifact_rate,
            self.model_review_usefulness_vote_rate,
            self.pairwise_usefulness_agreement,
        )
        if any(
            value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0)
            for value in unit_interval_values
        ):
            raise ReviewValidationError("review summary rates must be finite in [0, 1]")
        if self.pairwise_usefulness_kappa is not None and (
            not math.isfinite(self.pairwise_usefulness_kappa)
            or not -1.0 <= self.pairwise_usefulness_kappa <= 1.0
        ):
            raise ReviewValidationError("review summary kappa must be finite in [-1, 1]")
        if self.usefulness_claim_allowed != (
            self.status is ReviewEvidenceStatus.PREREGISTERED_THRESHOLD_MET
        ):
            raise ReviewValidationError("review claim gate must match the evidence status")
        if self.usefulness_claim_allowed and (
            self.model_review_usefulness_vote_rate is None
            or self.model_review_usefulness_vote_rate < self.preregistered_usefulness_vote_threshold
        ):
            raise ReviewValidationError("review claim gate requires the preregistered rate")

    def to_public_summary(self) -> dict[str, object]:
        return {
            "schema_version": REVIEW_SUMMARY_SCHEMA_VERSION,
            "status": self.status.value,
            "preregistration_version": self.preregistration_version,
            "preregistered_usefulness_vote_threshold": (
                self.preregistered_usefulness_vote_threshold
            ),
            "total_review_count": self.total_review_count,
            "eligible_review_count": self.eligible_review_count,
            "eligible_proposal_count": self.eligible_proposal_count,
            "unique_reviewer_count": self.unique_reviewer_count,
            "mean_morphology_rating": self.mean_morphology_rating,
            "median_morphology_rating": self.median_morphology_rating,
            "morphology_rating_population_sd": self.morphology_rating_population_sd,
            "artifact_rate": self.artifact_rate,
            "model_review_usefulness_vote_rate": self.model_review_usefulness_vote_rate,
            "pairwise_usefulness_agreement": self.pairwise_usefulness_agreement,
            "pairwise_usefulness_kappa": self.pairwise_usefulness_kappa,
            "pairwise_morphology_absolute_difference": (
                self.pairwise_morphology_absolute_difference
            ),
            "usefulness_claim_allowed": self.usefulness_claim_allowed,
            "review_scope": "blinded cardiology review of model-sensitivity proposals",
            "interpretation_boundary": INTERPRETATION_BOUNDARY,
        }

    def canonical_public_json(self) -> str:
        return canonical_json(self.to_public_summary())

    def public_summary_sha256(self) -> str:
        return canonical_sha256(self.to_public_summary())


def _pairwise_review_statistics(
    reviews_by_proposal: Mapping[str, Sequence[BlindedCardiologyReview]],
) -> tuple[float | None, float | None, float | None]:
    agreement_count = 0
    absolute_difference_sum = 0.0
    pair_count = 0
    for reviews in reviews_by_proposal.values():
        for left_index, left in enumerate(reviews):
            for right in reviews[left_index + 1 :]:
                pair_count += 1
                agreement_count += left.useful_for_model_review == right.useful_for_model_review
                absolute_difference_sum += abs(
                    left.morphology_plausibility_rating - right.morphology_plausibility_rating
                )
    if pair_count == 0:
        return None, None, None
    observed_agreement = agreement_count / pair_count
    all_reviews = tuple(review for reviews in reviews_by_proposal.values() for review in reviews)
    usefulness_prevalence = statistics.fmean(
        review.useful_for_model_review for review in all_reviews
    )
    expected_agreement = usefulness_prevalence**2 + (1.0 - usefulness_prevalence) ** 2
    kappa = (
        (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
        if expected_agreement < 1.0
        else None
    )
    return observed_agreement, kappa, absolute_difference_sum / pair_count


def aggregate_cardiology_reviews(
    reviews: Sequence[BlindedCardiologyReview],
    preregistration: ReviewPreregistration,
) -> CardiologyReviewSummary:
    """Aggregate blinded reviews without emitting pseudonyms or proposal digests."""

    if not isinstance(preregistration, ReviewPreregistration):
        raise ReviewValidationError("a frozen preregistration is required")
    materialized = tuple(reviews)
    if any(not isinstance(review, BlindedCardiologyReview) for review in materialized):
        raise ReviewValidationError("all reviews must satisfy the blinded review contract")
    observed_pairs: set[tuple[str, str]] = set()
    grouped: dict[str, list[BlindedCardiologyReview]] = defaultdict(list)
    for review in materialized:
        pair = (review.proposal_metadata_sha256, review.reviewer_pseudonym)
        if pair in observed_pairs:
            raise ReviewValidationError("one reviewer may submit only one review per proposal")
        observed_pairs.add(pair)
        grouped[review.proposal_metadata_sha256].append(review)

    eligible_groups = {
        proposal: tuple(group)
        for proposal, group in grouped.items()
        if len(group) >= preregistration.minimum_reviews_per_proposal
    }
    eligible_reviews = tuple(review for group in eligible_groups.values() for review in group)
    unique_reviewers = {review.reviewer_pseudonym for review in eligible_reviews}
    count_prerequisites_met = (
        len(eligible_groups) >= preregistration.minimum_proposal_count
        and len(eligible_reviews) >= preregistration.minimum_total_review_count
        and len(unique_reviewers) >= preregistration.minimum_unique_reviewer_count
    )

    ratings = [review.morphology_plausibility_rating for review in eligible_reviews]
    usefulness_votes = [review.useful_for_model_review for review in eligible_reviews]
    artifact_votes = [review.artifact_present for review in eligible_reviews]
    if eligible_reviews:
        mean_rating = float(statistics.fmean(ratings))
        median_rating = float(statistics.median(ratings))
        rating_sd = float(statistics.pstdev(ratings))
        artifact_rate = float(statistics.fmean(artifact_votes))
        usefulness_rate = float(statistics.fmean(usefulness_votes))
    else:
        mean_rating = None
        median_rating = None
        rating_sd = None
        artifact_rate = None
        usefulness_rate = None
    (
        pairwise_agreement,
        pairwise_kappa,
        pairwise_rating_difference,
    ) = _pairwise_review_statistics(eligible_groups)

    threshold_met = (
        count_prerequisites_met
        and usefulness_rate is not None
        and usefulness_rate >= preregistration.usefulness_vote_threshold
    )
    if threshold_met:
        status = ReviewEvidenceStatus.PREREGISTERED_THRESHOLD_MET
    elif count_prerequisites_met:
        status = ReviewEvidenceStatus.DESCRIPTIVE_ONLY
    else:
        status = ReviewEvidenceStatus.INSUFFICIENT_REVIEW_COUNT
    return CardiologyReviewSummary(
        status=status,
        preregistration_version=preregistration.version,
        preregistration_sha256=preregistration.sha256(),
        preregistered_usefulness_vote_threshold=preregistration.usefulness_vote_threshold,
        total_review_count=len(materialized),
        eligible_review_count=len(eligible_reviews),
        eligible_proposal_count=len(eligible_groups),
        unique_reviewer_count=len(unique_reviewers),
        mean_morphology_rating=mean_rating,
        median_morphology_rating=median_rating,
        morphology_rating_population_sd=rating_sd,
        artifact_rate=artifact_rate,
        model_review_usefulness_vote_rate=usefulness_rate,
        pairwise_usefulness_agreement=pairwise_agreement,
        pairwise_usefulness_kappa=pairwise_kappa,
        pairwise_morphology_absolute_difference=pairwise_rating_difference,
        usefulness_claim_allowed=threshold_met,
    )
