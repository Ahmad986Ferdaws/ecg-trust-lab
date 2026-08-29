"""End-to-end, fail-closed orchestration for the ECG Trust Sentinel.

The engine deliberately separates four questions that are often conflated:
whether an input obeys the physical ECG contract, whether the signal is usable,
whether a model representation is within its frozen reference distribution, and
whether label-wise uncertainty supports a singleton decision.  Only the final
``PREDICTION_ALLOWED`` state can carry probabilities across a public boundary.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ecg_trust.conformal import BinaryDecision, LabelwiseBinaryConformal
from ecg_trust.constants import SUPERCLASSES
from ecg_trust.contract_adapters import (
    case_distribution_assessment_from_score,
    conformal_prediction_sets_to_contracts,
    quality_report_to_contract,
    unavailable_case_distribution_assessment,
)
from ecg_trust.contracts import (
    ArtifactReference,
    CaseDistributionAssessment,
    LabelPredictionSetDecision,
    QualityReport,
    TrustDecision,
)
from ecg_trust.quality.signal_quality import (
    QualityStatus,
    SignalMetadata,
    SignalQualityConfig,
    assess_signal_quality,
)
from ecg_trust.registry import ArtifactRole, VerifiedTrustBundle
from ecg_trust.runtime_binding import (
    BoundRuntimeComponent,
    RuntimeArtifact,
    RuntimeArtifactIdentity,
    RuntimeBindingError,
    RuntimeTrustBinding,
)
from ecg_trust.trust_policy import (
    TrustPolicyConfig,
    TrustPolicyInputs,
    TrustPolicyResult,
    evaluate_trust_policy,
)

FloatArray = NDArray[np.float64]


class SentinelValidationError(ValueError):
    """Raised when a component violates the Sentinel integration contract."""


class SentinelComponentUnavailable(RuntimeError):
    """Raised without private details when required model inference cannot run."""


@dataclass(frozen=True, slots=True)
class SentinelModelEvidence:
    """Minimal frozen-model output needed by trust gates.

    Raw logits, waveform data, filesystem paths, and explanations are omitted on
    purpose.  ``embedding`` is consumed by the OOD detector and never exposed by
    :class:`SentinelCaseResult`.
    """

    release_id: str
    label_order: tuple[str, ...]
    calibrated_probabilities: tuple[float, ...]
    embedding: tuple[float, ...]
    legacy_entropy_gate_accepted: bool

    def __post_init__(self) -> None:
        if not self.release_id.strip():
            raise SentinelValidationError("release_id must be non-empty")
        if self.label_order != SUPERCLASSES:
            raise SentinelValidationError("model labels must use canonical superclass order")
        if len(self.calibrated_probabilities) != len(SUPERCLASSES):
            raise SentinelValidationError("model must return five calibrated probabilities")
        if any(
            isinstance(value, bool) or not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.calibrated_probabilities
        ):
            raise SentinelValidationError(
                "calibrated probabilities must be finite values in [0, 1]"
            )
        if not self.embedding or any(
            isinstance(value, bool) or not math.isfinite(value) for value in self.embedding
        ):
            raise SentinelValidationError("embedding must be a non-empty finite vector")
        if not isinstance(self.legacy_entropy_gate_accepted, bool):
            raise SentinelValidationError("legacy entropy decision must be boolean")


class FrozenModelRunner(Protocol):
    """Inference boundary implemented by a verified, immutable model release."""

    bound_manifest_sha256: str
    bound_checkpoint_sha256s: tuple[str, ...]

    def infer(self, signal_mv: FloatArray) -> SentinelModelEvidence: ...


class FrozenDistributionDetector(Protocol):
    """Higher-is-more-OOD detector fitted without evaluation-site adaptation."""

    threshold: float

    def score(self, embeddings: ArrayLike) -> FloatArray: ...


@dataclass(frozen=True, slots=True)
class SentinelModelArtifactInputs:
    """Exact verified files supplied to the frozen-model loader."""

    release_id: str
    manifest_sha256: str
    checkpoints: tuple[RuntimeArtifact, ...]
    resolved_config: RuntimeArtifact
    normalization: RuntimeArtifact
    decision_policy: RuntimeArtifact


@dataclass(frozen=True, slots=True)
class LoadedDistributionPolicy:
    """A fitted detector and the method identity encoded by its policy file."""

    detector: FrozenDistributionDetector | None
    method: str
    schema_version: int

    def __post_init__(self) -> None:
        if not self.method.strip():
            raise SentinelValidationError("distribution method must be non-empty")
        if self.schema_version < 1:
            raise SentinelValidationError("distribution schema version must be positive")


@dataclass(frozen=True, slots=True)
class SentinelRuntimeLoaders:
    """Trusted loaders that receive only freshly verified TrustBundle parents."""

    model_runner: Callable[[SentinelModelArtifactInputs], FrozenModelRunner]
    quality_policy: Callable[[RuntimeArtifact], SignalQualityConfig]
    decision_policy: Callable[[RuntimeArtifact], TrustPolicyConfig]
    distribution_policy: Callable[[RuntimeArtifact], LoadedDistributionPolicy]
    conformal_policy: Callable[[RuntimeArtifact], LabelwiseBinaryConformal | None]


@dataclass(frozen=True, slots=True)
class SentinelArtifacts:
    """Release-bound identities for every non-model trust component."""

    release_id: str
    manifest_sha256: str
    distribution_method: str
    distribution_method_schema_version: int
    quality_policy_version: str
    decision_policy_version: str
    checkpoint_artifacts: tuple[ArtifactReference, ...]
    resolved_config_artifact: ArtifactReference
    normalization_artifact: ArtifactReference
    quality_policy_artifact: ArtifactReference
    decision_policy_artifact: ArtifactReference
    distribution_artifact: ArtifactReference
    conformal_artifact: ArtifactReference

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (
                self.release_id,
                self.manifest_sha256,
                self.distribution_method,
                self.quality_policy_version,
                self.decision_policy_version,
            )
        ):
            raise SentinelValidationError("release and policy identities must be non-empty")
        if self.distribution_method_schema_version < 1:
            raise SentinelValidationError("distribution schema version must be positive")
        if len(self.manifest_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.manifest_sha256
        ):
            raise SentinelValidationError("manifest_sha256 must be a lowercase SHA-256 digest")
        if not self.checkpoint_artifacts:
            raise SentinelValidationError("at least one checkpoint artifact is required")


@dataclass(frozen=True, slots=True)
class SentinelCaseResult:
    """One case disposition with a structural prediction-disclosure invariant."""

    release_id: str
    decision: TrustDecision
    policy: TrustPolicyResult
    quality: QualityReport
    distribution: CaseDistributionAssessment | None
    label_prediction_sets: tuple[LabelPredictionSetDecision, ...] | None
    calibrated_probabilities: tuple[float, ...] | None

    def __post_init__(self) -> None:
        exposed = self.decision is TrustDecision.PREDICTION_ALLOWED
        if self.policy.decision is not self.decision:
            raise SentinelValidationError("policy and case decisions must match")
        if exposed:
            if self.calibrated_probabilities is None or self.label_prediction_sets is None:
                raise SentinelValidationError("allowed results require probabilities and sets")
            if len(self.calibrated_probabilities) != len(SUPERCLASSES):
                raise SentinelValidationError("allowed results require five probabilities")
            if len(self.label_prediction_sets) != len(SUPERCLASSES):
                raise SentinelValidationError("allowed results require five prediction sets")
        elif self.calibrated_probabilities is not None or self.label_prediction_sets is not None:
            raise SentinelValidationError("blocked results cannot carry label-level outputs")

    @property
    def predictions_exposed(self) -> bool:
        return self.decision is TrustDecision.PREDICTION_ALLOWED

    def to_public_dict(self) -> dict[str, object]:
        """Return a privacy-minimal response that cannot leak blocked predictions."""

        result: dict[str, object] = {
            "release_id": self.release_id,
            "decision": self.decision.value,
            "predictions_exposed": self.predictions_exposed,
            "reason_codes": [reason.value for reason in self.policy.reason_codes],
            "quality": {
                "passed": self.quality.passed,
                "decision": self.quality.decision.value,
                "finding_codes": [finding.code for finding in self.quality.findings],
            },
            "distribution_status": (
                None if self.distribution is None else self.distribution.status.value
            ),
            "research_only": True,
            "boundary": "Not for diagnosis, treatment, or clinical decision-making.",
        }
        if self.predictions_exposed:
            probabilities = cast(tuple[float, ...], self.calibrated_probabilities)
            sets = cast(tuple[LabelPredictionSetDecision, ...], self.label_prediction_sets)
            result["probabilities"] = dict(zip(SUPERCLASSES, probabilities, strict=True))
            result["prediction_sets"] = [
                {
                    "label": item.label,
                    "decision": item.decision.value,
                    "include_supported": item.include_supported,
                    "include_not_supported": item.include_not_supported,
                }
                for item in sets
            ]
        return result


class TrustSentinelEngine:
    """Compose quality, model, OOD, entropy, and conformal evidence in order."""

    def __init__(
        self,
        *,
        runtime_binding: RuntimeTrustBinding,
        model_runner: BoundRuntimeComponent[FrozenModelRunner],
        quality_policy: BoundRuntimeComponent[SignalQualityConfig],
        decision_policy: BoundRuntimeComponent[TrustPolicyConfig],
        distribution_policy: BoundRuntimeComponent[LoadedDistributionPolicy],
        conformal_policy: BoundRuntimeComponent[LabelwiseBinaryConformal | None],
    ) -> None:
        if not isinstance(runtime_binding, RuntimeTrustBinding):
            raise TypeError("runtime_binding must be a RuntimeTrustBinding")
        try:
            checkpoints = runtime_binding.identities_for_role(ArtifactRole.CHECKPOINT)
            model_roles = (
                *((ArtifactRole.CHECKPOINT,) * len(checkpoints)),
                ArtifactRole.RESOLVED_CONFIG,
                ArtifactRole.NORMALIZATION,
                ArtifactRole.DECISION_POLICY,
            )
            runtime_binding.assert_component_roles(
                cast(BoundRuntimeComponent[object], model_runner), model_roles
            )
            runtime_binding.assert_component_roles(
                cast(BoundRuntimeComponent[object], quality_policy),
                (ArtifactRole.QUALITY_POLICY,),
            )
            runtime_binding.assert_component_roles(
                cast(BoundRuntimeComponent[object], decision_policy),
                (ArtifactRole.DECISION_POLICY,),
            )
            runtime_binding.assert_component_roles(
                cast(BoundRuntimeComponent[object], distribution_policy),
                (ArtifactRole.DISTRIBUTION_POLICY,),
            )
            runtime_binding.assert_component_roles(
                cast(BoundRuntimeComponent[object], conformal_policy),
                (ArtifactRole.CONFORMAL_POLICY,),
            )
            runtime_binding.verify_intact()
        except RuntimeBindingError:
            raise SentinelValidationError("verified runtime binding is invalid") from None

        quality_config = quality_policy.value
        policy_config = decision_policy.value
        distribution_component = distribution_policy.value
        if not isinstance(quality_config, SignalQualityConfig):
            raise SentinelValidationError("quality policy loader returned an invalid component")
        if not isinstance(policy_config, TrustPolicyConfig):
            raise SentinelValidationError("decision policy loader returned an invalid component")
        if not isinstance(distribution_component, LoadedDistributionPolicy):
            raise SentinelValidationError(
                "distribution policy loader returned an invalid component"
            )
        if conformal_policy.value is not None and not isinstance(
            conformal_policy.value, LabelwiseBinaryConformal
        ):
            raise SentinelValidationError("conformal policy loader returned an invalid component")
        expected_checkpoint_digests = tuple(identity.unprefixed_sha256 for identity in checkpoints)
        if (
            getattr(model_runner.value, "bound_manifest_sha256", None)
            != runtime_binding.service_manifest_sha256
            or getattr(model_runner.value, "bound_checkpoint_sha256s", None)
            != expected_checkpoint_digests
        ):
            raise SentinelValidationError(
                "model runner identity does not match the verified release"
            )

        checkpoint_references = tuple(_artifact_reference(identity) for identity in checkpoints)
        self._runtime_binding = runtime_binding
        self._model_runner = model_runner.value
        self._distribution_detector = distribution_component.detector
        self._conformal_predictor = conformal_policy.value
        self._policy_config = policy_config
        self._quality_config = quality_config
        self._artifacts = SentinelArtifacts(
            release_id=runtime_binding.release_id,
            manifest_sha256=runtime_binding.service_manifest_sha256,
            distribution_method=distribution_component.method,
            distribution_method_schema_version=distribution_component.schema_version,
            quality_policy_version=quality_config.version,
            decision_policy_version=policy_config.version,
            checkpoint_artifacts=checkpoint_references,
            resolved_config_artifact=_artifact_reference(
                runtime_binding.require_single(ArtifactRole.RESOLVED_CONFIG).identity
            ),
            normalization_artifact=_artifact_reference(
                runtime_binding.require_single(ArtifactRole.NORMALIZATION).identity
            ),
            quality_policy_artifact=_artifact_reference(
                runtime_binding.require_single(ArtifactRole.QUALITY_POLICY).identity
            ),
            decision_policy_artifact=_artifact_reference(
                runtime_binding.require_single(ArtifactRole.DECISION_POLICY).identity
            ),
            distribution_artifact=_artifact_reference(
                runtime_binding.require_single(ArtifactRole.DISTRIBUTION_POLICY).identity
            ),
            conformal_artifact=_artifact_reference(
                runtime_binding.require_single(ArtifactRole.CONFORMAL_POLICY).identity
            ),
        )

    @classmethod
    def from_verified_bundle(
        cls,
        verified_bundle: VerifiedTrustBundle,
        *,
        loaders: SentinelRuntimeLoaders,
    ) -> TrustSentinelEngine:
        """Load every runtime component from freshly verified role-bound parents."""

        if not isinstance(loaders, SentinelRuntimeLoaders):
            raise TypeError("loaders must be SentinelRuntimeLoaders")
        try:
            binding = RuntimeTrustBinding(verified_bundle)
            checkpoints = binding.require_checkpoints()
            resolved_config = binding.require_single(ArtifactRole.RESOLVED_CONFIG)
            normalization = binding.require_single(ArtifactRole.NORMALIZATION)
            decision_artifact = binding.require_single(ArtifactRole.DECISION_POLICY)
            quality_artifact = binding.require_single(ArtifactRole.QUALITY_POLICY)
            distribution_artifact = binding.require_single(ArtifactRole.DISTRIBUTION_POLICY)
            conformal_artifact = binding.require_single(ArtifactRole.CONFORMAL_POLICY)

            model_artifacts = (*checkpoints, resolved_config, normalization, decision_artifact)

            def load_model(_: tuple[RuntimeArtifact, ...]) -> FrozenModelRunner:
                return loaders.model_runner(
                    SentinelModelArtifactInputs(
                        release_id=binding.release_id,
                        manifest_sha256=binding.service_manifest_sha256,
                        checkpoints=checkpoints,
                        resolved_config=resolved_config,
                        normalization=normalization,
                        decision_policy=decision_artifact,
                    )
                )

            model_runner = binding.load_component(model_artifacts, load_model)
            quality_policy = binding.load_component(
                (quality_artifact,), lambda items: loaders.quality_policy(items[0])
            )
            decision_policy = binding.load_component(
                (decision_artifact,), lambda items: loaders.decision_policy(items[0])
            )
            distribution_policy = binding.load_component(
                (distribution_artifact,), lambda items: loaders.distribution_policy(items[0])
            )
            conformal_policy = binding.load_component(
                (conformal_artifact,), lambda items: loaders.conformal_policy(items[0])
            )
            return cls(
                runtime_binding=binding,
                model_runner=model_runner,
                quality_policy=quality_policy,
                decision_policy=decision_policy,
                distribution_policy=distribution_policy,
                conformal_policy=conformal_policy,
            )
        except (RuntimeBindingError, SentinelValidationError):
            raise SentinelValidationError("verified Sentinel runtime assembly failed") from None

    def is_ready(self) -> bool:
        """Readiness means every component required to permit a prediction exists."""

        return (
            self._runtime_binding.is_intact()
            and self._distribution_detector is not None
            and self._conformal_predictor is not None
        )

    @property
    def release_id(self) -> str:
        """Immutable release identity expected by every component and request."""

        return self._artifacts.release_id

    @property
    def bound_manifest_sha256(self) -> str:
        """Manifest digest that must exactly match the service's verified release."""

        return self._artifacts.manifest_sha256

    def analyze(
        self,
        *,
        signal_id: str,
        signal_mv: ArrayLike,
        metadata: SignalMetadata,
        evaluated_at: datetime,
        release_integrity_verified: bool,
    ) -> SentinelCaseResult:
        """Run the trust gates, skipping model inference after input/quality failure."""

        if not signal_id.strip():
            raise SentinelValidationError("signal_id must be non-empty")
        quality_core = assess_signal_quality(signal_mv, metadata, config=self._quality_config)
        quality = quality_report_to_contract(quality_core, evaluated_at=evaluated_at)
        input_contract_valid = quality_core.status is not QualityStatus.INVALID
        runtime_integrity_verified = (
            release_integrity_verified and self._runtime_binding.is_intact()
        )

        preliminary = evaluate_trust_policy(
            TrustPolicyInputs(
                release_integrity_verified=runtime_integrity_verified,
                input_contract_valid=input_contract_valid,
                quality_report=quality_core,
                distribution_supported=None,
            ),
            config=self._policy_config,
        )
        if preliminary.decision in {TrustDecision.INVALID_INPUT, TrustDecision.REACQUIRE}:
            return self._blocked(preliminary, quality)

        try:
            model_evidence = self._model_runner.infer(np.asarray(signal_mv, dtype=np.float64))
        except Exception as error:
            raise SentinelComponentUnavailable("frozen model inference is unavailable") from error
        if model_evidence.release_id != self._artifacts.release_id:
            raise SentinelComponentUnavailable("model evidence release does not match")

        distribution = self._distribution_assessment(signal_id, model_evidence.embedding)
        distribution_supported = (
            None
            if distribution.is_out_of_distribution is None
            else not distribution.is_out_of_distribution
        )
        distribution_reasons = tuple(distribution.reason_codes)
        runtime_integrity_verified = (
            release_integrity_verified and self._runtime_binding.is_intact()
        )
        distribution_policy = evaluate_trust_policy(
            TrustPolicyInputs(
                release_integrity_verified=runtime_integrity_verified,
                input_contract_valid=True,
                quality_report=quality_core,
                distribution_supported=distribution_supported,
                distribution_reason_codes=distribution_reasons,
                legacy_entropy_gate_accepted=model_evidence.legacy_entropy_gate_accepted,
                # A temporary all-singleton value lets the shared policy evaluate
                # only gates that precede the real conformal calculation below.
                conformal_decisions=(BinaryDecision.NOT_SUPPORTED,) * len(SUPERCLASSES),
            ),
            config=self._policy_config,
        )
        if distribution_policy.decision is not TrustDecision.PREDICTION_ALLOWED:
            return self._blocked(distribution_policy, quality, distribution)

        conformal = self._conformal_predictor
        if conformal is None:
            final_policy = evaluate_trust_policy(
                TrustPolicyInputs(
                    release_integrity_verified=(
                        release_integrity_verified and self._runtime_binding.is_intact()
                    ),
                    input_contract_valid=True,
                    quality_report=quality_core,
                    distribution_supported=True,
                    distribution_reason_codes=distribution_reasons,
                    legacy_entropy_gate_accepted=True,
                    conformal_decisions=None,
                ),
                config=self._policy_config,
            )
            return self._blocked(final_policy, quality, distribution)

        try:
            prediction_sets = conformal.predict(
                np.asarray(model_evidence.calibrated_probabilities, dtype=np.float64)[None, :]
            )
            decisions: tuple[BinaryDecision, ...] = prediction_sets.decisions[0]
            set_contracts = conformal_prediction_sets_to_contracts(
                prediction_sets,
                model_evidence.calibrated_probabilities,
                calibration_artifact=self._artifacts.conformal_artifact,
            )
        except Exception:
            final_policy = evaluate_trust_policy(
                TrustPolicyInputs(
                    release_integrity_verified=(
                        release_integrity_verified and self._runtime_binding.is_intact()
                    ),
                    input_contract_valid=True,
                    quality_report=quality_core,
                    distribution_supported=True,
                    distribution_reason_codes=distribution_reasons,
                    legacy_entropy_gate_accepted=True,
                    conformal_decisions=None,
                ),
                config=self._policy_config,
            )
            return self._blocked(final_policy, quality, distribution)

        final_policy = evaluate_trust_policy(
            TrustPolicyInputs(
                release_integrity_verified=(
                    release_integrity_verified and self._runtime_binding.is_intact()
                ),
                input_contract_valid=True,
                quality_report=quality_core,
                distribution_supported=True,
                distribution_reason_codes=distribution_reasons,
                legacy_entropy_gate_accepted=True,
                conformal_decisions=decisions,
            ),
            config=self._policy_config,
        )
        if final_policy.decision is not TrustDecision.PREDICTION_ALLOWED:
            return self._blocked(final_policy, quality, distribution)
        return SentinelCaseResult(
            release_id=self._artifacts.release_id,
            decision=final_policy.decision,
            policy=final_policy,
            quality=quality,
            distribution=distribution,
            label_prediction_sets=set_contracts,
            calibrated_probabilities=model_evidence.calibrated_probabilities,
        )

    def _distribution_assessment(
        self,
        signal_id: str,
        embedding: tuple[float, ...],
    ) -> CaseDistributionAssessment:
        detector = self._distribution_detector
        if detector is None:
            return unavailable_case_distribution_assessment(
                assessment_id=f"ood-{signal_id}",
                signal_id=signal_id,
                release_id=self._artifacts.release_id,
                method=self._artifacts.distribution_method,
                method_artifact=self._artifacts.distribution_artifact,
                expected_method_schema_version=(self._artifacts.distribution_method_schema_version),
                observed_method_schema_version=None,
                artifact_available=False,
                reason_code="ARTIFACT_UNAVAILABLE",
            )
        try:
            scores = np.asarray(
                detector.score(np.asarray(embedding, dtype=np.float64)[None, :]),
                dtype=np.float64,
            )
            if scores.shape != (1,) or not np.isfinite(scores).all():
                raise SentinelValidationError("distribution detector returned invalid scores")
            score = float(scores[0])
            threshold = float(detector.threshold)
            if not math.isfinite(threshold):
                raise SentinelValidationError("distribution threshold must be finite")
        except Exception:
            return unavailable_case_distribution_assessment(
                assessment_id=f"ood-{signal_id}",
                signal_id=signal_id,
                release_id=self._artifacts.release_id,
                method=self._artifacts.distribution_method,
                method_artifact=self._artifacts.distribution_artifact,
                expected_method_schema_version=(self._artifacts.distribution_method_schema_version),
                observed_method_schema_version=(self._artifacts.distribution_method_schema_version),
                artifact_available=True,
                reason_code="SCORING_UNAVAILABLE",
            )
        return case_distribution_assessment_from_score(
            assessment_id=f"ood-{signal_id}",
            signal_id=signal_id,
            release_id=self._artifacts.release_id,
            method=self._artifacts.distribution_method,
            method_artifact=self._artifacts.distribution_artifact,
            method_schema_version=self._artifacts.distribution_method_schema_version,
            score=score,
            threshold=threshold,
        )

    def _blocked(
        self,
        policy: TrustPolicyResult,
        quality: QualityReport,
        distribution: CaseDistributionAssessment | None = None,
    ) -> SentinelCaseResult:
        return SentinelCaseResult(
            release_id=self._artifacts.release_id,
            decision=policy.decision,
            policy=policy,
            quality=quality,
            distribution=distribution,
            label_prediction_sets=None,
            calibrated_probabilities=None,
        )


def _artifact_reference(identity: RuntimeArtifactIdentity) -> ArtifactReference:
    """Project one verified parent identity without exposing its local path."""

    return ArtifactReference(
        schema_version="ecg_trust.artifact_reference.v1",
        artifact_id=identity.artifact_id,
        file_sha256=identity.file_sha256,
        size_bytes=identity.size_bytes,
        media_type=identity.media_type,
        sensitive=True,
    )


__all__ = [
    "FrozenDistributionDetector",
    "FrozenModelRunner",
    "LoadedDistributionPolicy",
    "SentinelArtifacts",
    "SentinelCaseResult",
    "SentinelComponentUnavailable",
    "SentinelModelArtifactInputs",
    "SentinelModelEvidence",
    "SentinelRuntimeLoaders",
    "SentinelValidationError",
    "TrustSentinelEngine",
]
