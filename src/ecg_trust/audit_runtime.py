"""Verified post-evaluation inference over the sealed fold-10 release.

The final-test ledger predates this module, so post-evaluation analysis cannot
require the *current* Git revision to equal the revision frozen for the final
evaluation.  This module instead verifies the immutable specification and all
of its bound sources, independently verifies the completed ledger and opening
marker, and only then issues a protocol-bound fold-10 token.

Waveform transforms are deliberately applied to physical millivolt signals.
The authoritative training-fold normalization is applied afterwards, before
model inference.  A clean pass must reproduce every stored float32 logit
exactly (after the prediction artifact's canonical float64 promotion).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence, Sized
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from ecg_trust.constants import LEADS, TARGET_COLUMNS
from ecg_trust.data.dataset import NormalizationStats, PTBXLDataset
from ecg_trust.decisioning import (
    CalibrationDecisionArtifact,
    load_calibration_decisions,
)
from ecg_trust.final_batch import (
    FINAL_BATCH_PLAN_SCHEMA_VERSION,
    FINAL_BATCH_PLAN_TYPE,
    FINAL_LEDGER_SCHEMA_VERSION,
    FINAL_LEDGER_TYPE,
    FINAL_OPENING_MARKER_SCHEMA_VERSION,
    FINAL_OPENING_MARKER_TYPE,
)
from ecg_trust.final_evaluation_spec import (
    FinalEvaluationSpec,
    load_final_evaluation_spec,
)
from ecg_trust.prediction_export import (
    PredictionExportRequest,
    _load_inference_model,
    _load_resolved_run,
    _seed_inference,
    _validate_inputs,
    _validate_requested_lineage,
)
from ecg_trust.predictions import (
    IdentifierArray,
    PredictionArtifact,
    assert_prediction_artifacts_aligned,
    load_prediction_artifact,
)
from ecg_trust.protocol import (
    FINAL_TEST_CONFIRMATION,
    FINAL_TEST_FOLDS,
    LABEL_ORDER,
    ExperimentProtocol,
    FinalTestAccessToken,
    FoldRole,
    authorize_final_test_access,
)
from ecg_trust.release_gates import (
    EXPECTED_ARCHITECTURES,
    EXPECTED_SEEDS,
    CalibrationBundle,
    CalibrationMember,
    RefitBundle,
    RefitMember,
    canonical_sha256,
    load_calibration_bundle,
    load_refit_bundle,
    sha256_file,
)
from ecg_trust.training import TrainingRuntime, seed_dataloader_worker, select_device

EXPECTED_AUDIT_MEMBER_IDS: tuple[str, ...] = tuple(
    f"{architecture}-seed{seed}"
    for architecture in EXPECTED_ARCHITECTURES
    for seed in EXPECTED_SEEDS
)


class AuditRuntimeError(RuntimeError):
    """Raised when sealed post-evaluation inference cannot proceed safely."""


class AuditRuntimeIntegrityError(AuditRuntimeError):
    """Raised when a release source or in-memory alignment differs."""


class CleanLogitMismatchError(AuditRuntimeIntegrityError):
    """Raised when clean authoritative inference differs by even one logit."""

    def __init__(
        self,
        member_id: str,
        *,
        mismatch_count: int,
        maximum_absolute_error: float,
        mean_absolute_error: float,
    ) -> None:
        self.member_id = member_id
        self.mismatch_count = mismatch_count
        self.maximum_absolute_error = maximum_absolute_error
        self.mean_absolute_error = mean_absolute_error
        super().__init__(
            f"clean logits for {member_id} are not bit-exact: "
            f"mismatches={mismatch_count}, "
            f"max_abs_error={maximum_absolute_error:.9g}, "
            f"mean_abs_error={mean_absolute_error:.9g}"
        )


class PhysicalBatchTransform(Protocol):
    """Transform a CPU batch in physical millivolts before normalization."""

    def __call__(
        self,
        signals_mv: Tensor,
        ecg_id: IdentifierArray,
    ) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class AuditInferenceSettings:
    """The exact inference settings recorded in the final-test plan."""

    batch_size: int
    num_workers: int
    device: str
    bf16: bool
    seed: int
    pin_memory: bool = True
    persistent_workers: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise AuditRuntimeIntegrityError("audit batch_size must be positive")
        if isinstance(self.num_workers, bool) or self.num_workers < 0:
            raise AuditRuntimeIntegrityError("audit num_workers must be non-negative")
        if not isinstance(self.device, str) or not self.device.strip():
            raise AuditRuntimeIntegrityError("audit device must be non-empty")
        if not isinstance(self.bf16, bool):
            raise AuditRuntimeIntegrityError("audit bf16 setting must be boolean")
        if isinstance(self.seed, bool) or not 0 <= self.seed < 2**32:
            raise AuditRuntimeIntegrityError("audit seed must be a uint32 integer")
        if not isinstance(self.pin_memory, bool) or not isinstance(
            self.persistent_workers, bool
        ):
            raise AuditRuntimeIntegrityError(
                "audit pin_memory and persistent_workers must be boolean"
            )


@dataclass(frozen=True, slots=True)
class AlignedAuditInference:
    """One inference pass aligned to the canonical sealed prediction rows."""

    member_id: str
    ecg_id: IdentifierArray
    patient_id: IdentifierArray
    strat_fold: NDArray[np.int8]
    targets: NDArray[np.int8]
    raw_logits: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CleanLogitEquivalence:
    """Evidence that one clean pass exactly reproduced its sealed logits."""

    member_id: str
    record_count: int
    logit_count: int
    sealed_prediction_sha256: str
    exact: bool = True
    mismatch_count: int = 0
    maximum_absolute_error: float = 0.0
    mean_absolute_error: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "member_id": self.member_id,
            "record_count": self.record_count,
            "logit_count": self.logit_count,
            "sealed_prediction_sha256": self.sealed_prediction_sha256,
            "exact": self.exact,
            "mismatch_count": self.mismatch_count,
            "maximum_absolute_error": self.maximum_absolute_error,
            "mean_absolute_error": self.mean_absolute_error,
        }


@dataclass(frozen=True, slots=True)
class VerifiedCompletedLedger:
    """Immutable in-memory view of a fully verified completed opening ledger."""

    path: Path
    ledger_sha256: str
    batch_sha256: str
    purpose: str
    operator: str
    opening_marker_path: Path
    _canonical_payload: str

    @property
    def payload(self) -> dict[str, object]:
        decoded: object = json.loads(self._canonical_payload)
        if not isinstance(decoded, dict):  # pragma: no cover - constructor invariant
            raise AuditRuntimeIntegrityError("verified ledger payload is not an object")
        return cast(dict[str, object], decoded)

    @property
    def plan(self) -> Mapping[str, object]:
        return _mapping(self.payload["plan"], "verified ledger plan")

    @property
    def members(self) -> Mapping[str, object]:
        return _mapping(self.payload["members"], "verified ledger members")


class _IndexedPhysicalDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, dataset: Dataset[tuple[Tensor, Tensor]]) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        signal, target = self.dataset[index]
        return torch.tensor(index, dtype=torch.int64), signal, target


@dataclass(slots=True)
class AuditMemberRuntime:
    """Loaded authoritative model plus its physical fold-10 data contract."""

    member_id: str
    architecture: str
    seed: int
    refit: RefitMember
    calibration: CalibrationMember
    decisions: CalibrationDecisionArtifact
    sealed_prediction: PredictionArtifact
    model: nn.Module
    physical_dataset: Dataset[tuple[Tensor, Tensor]]
    selected_manifest: pd.DataFrame
    normalization: NormalizationStats
    resolved_config: Mapping[str, object]
    checkpoint_sha256: str
    checkpoint_epoch: int
    settings: AuditInferenceSettings
    runtime: TrainingRuntime

    def __post_init__(self) -> None:
        if self.member_id != f"{self.architecture}-seed{self.seed}":
            raise AuditRuntimeIntegrityError("audit member identity is inconsistent")
        if self.settings.seed != self.seed:
            raise AuditRuntimeIntegrityError("audit inference seed differs from member seed")
        if str(self.runtime.device) != self.settings.device:
            raise AuditRuntimeIntegrityError(
                "selected audit device differs from the final-test plan"
            )
        if self.runtime.bf16_enabled is not self.settings.bf16:
            raise AuditRuntimeIntegrityError(
                "selected BF16 runtime differs from the final-test plan"
            )
        if len(cast(Sized, self.physical_dataset)) != len(self.selected_manifest):
            raise AuditRuntimeIntegrityError(
                "physical dataset size differs from the fold-10 manifest"
            )
        self._sealed_order()
        self.model.eval()

    def normalize_physical_batch(self, signals_mv: Tensor) -> Tensor:
        """Apply the frozen per-lead normalization to a physical CPU batch."""

        if not isinstance(signals_mv, Tensor) or signals_mv.ndim != 3:
            raise AuditRuntimeIntegrityError(
                "physical ECG batches must be tensors shaped [batch, 12, 1000]"
            )
        if signals_mv.device.type != "cpu":
            raise AuditRuntimeIntegrityError(
                "physical transforms and normalization must run on CPU"
            )
        expected_shape = (len(LEADS), self.normalization.provenance.samples_per_record)
        if tuple(signals_mv.shape[1:]) != expected_shape or signals_mv.shape[0] < 1:
            raise AuditRuntimeIntegrityError(
                f"physical ECG batch shape must be [batch, {expected_shape[0]}, "
                f"{expected_shape[1]}]"
            )
        physical = signals_mv.to(dtype=torch.float32).contiguous()
        if not torch.isfinite(physical).all().item():
            raise AuditRuntimeIntegrityError("physical ECG batch contains non-finite values")
        mean = torch.tensor(self.normalization.mean, dtype=torch.float32).view(1, -1, 1)
        std = torch.tensor(self.normalization.std, dtype=torch.float32).view(1, -1, 1)
        normalized = (physical - mean) / std
        if not torch.isfinite(normalized).all().item():
            raise AuditRuntimeIntegrityError(
                "physical ECG batch became non-finite after normalization"
            )
        return normalized.contiguous()

    def infer_logits(
        self,
        transform: PhysicalBatchTransform | None = None,
    ) -> AlignedAuditInference:
        """Run deterministic inference and align results to the sealed rows."""

        generator = _seed_inference(self.seed, self.runtime)
        loader = DataLoader(
            _IndexedPhysicalDataset(self.physical_dataset),
            batch_size=self.settings.batch_size,
            shuffle=False,
            num_workers=self.settings.num_workers,
            pin_memory=(
                self.settings.pin_memory and self.runtime.device.type == "cuda"
            ),
            persistent_workers=(
                self.settings.persistent_workers and self.settings.num_workers > 0
            ),
            worker_init_fn=(
                seed_dataloader_worker if self.settings.num_workers > 0 else None
            ),
            generator=generator,
            drop_last=False,
        )
        self.model.eval()
        positions: list[Tensor] = []
        logits: list[Tensor] = []
        targets: list[Tensor] = []
        with torch.inference_mode():
            for raw_positions, raw_signals, raw_targets in loader:
                self._validate_batch(raw_positions, raw_signals, raw_targets)
                batch_positions = raw_positions.to(device="cpu", dtype=torch.int64)
                signals_mv = raw_signals.detach().to(
                    device="cpu", dtype=torch.float32
                )
                if transform is not None:
                    ecg_ids = self._batch_ecg_ids(batch_positions)
                    transformed = transform(signals_mv.clone(), ecg_ids)
                    if not isinstance(transformed, Tensor):
                        raise AuditRuntimeIntegrityError(
                            "physical batch transform must return a tensor"
                        )
                    if transformed.shape != signals_mv.shape:
                        raise AuditRuntimeIntegrityError(
                            "physical batch transform changed the ECG shape"
                        )
                    signals_mv = transformed
                normalized = self.normalize_physical_batch(signals_mv)
                device_signals = normalized.to(
                    device=self.runtime.device,
                    dtype=torch.float32,
                    non_blocking=self.runtime.device.type == "cuda",
                )
                with torch.autocast(
                    device_type=self.runtime.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.runtime.bf16_enabled,
                ):
                    raw_logits = self.model(device_signals)
                if not isinstance(raw_logits, Tensor) or (
                    raw_logits.shape != raw_targets.shape
                ):
                    raise AuditRuntimeIntegrityError(
                        "authoritative model logits do not align with targets"
                    )
                batch_logits = raw_logits.detach().to(
                    device="cpu", dtype=torch.float32
                )
                if not torch.isfinite(batch_logits).all().item():
                    raise AuditRuntimeIntegrityError(
                        "authoritative model produced non-finite logits"
                    )
                positions.append(batch_positions)
                logits.append(batch_logits)
                targets.append(raw_targets.to(device="cpu", dtype=torch.float32))
        if not positions:
            raise AuditRuntimeIntegrityError("physical fold-10 loader produced no rows")
        all_positions = torch.cat(positions).numpy()
        expected_positions = np.arange(len(self.selected_manifest), dtype=np.int64)
        if not np.array_equal(all_positions, expected_positions):
            raise AuditRuntimeIntegrityError(
                "physical fold-10 loader order is incomplete or nondeterministic"
            )
        all_targets = torch.cat(targets).numpy().astype(np.int8, copy=False)
        manifest_targets = self.selected_manifest.loc[
            :, list(TARGET_COLUMNS)
        ].to_numpy(dtype=np.int8)
        if not np.array_equal(all_targets, manifest_targets):
            raise AuditRuntimeIntegrityError(
                "physical dataset targets differ from the selected manifest"
            )
        all_logits = torch.cat(logits).numpy().astype(np.float64, copy=False)
        order = self._sealed_order()
        aligned_targets = all_targets[order]
        if not np.array_equal(aligned_targets, self.sealed_prediction.targets):
            raise AuditRuntimeIntegrityError(
                "runtime targets differ from the sealed prediction artifact"
            )
        return AlignedAuditInference(
            member_id=self.member_id,
            ecg_id=_readonly_identifier(self.sealed_prediction.ecg_id),
            patient_id=_readonly_identifier(self.sealed_prediction.patient_id),
            strat_fold=_readonly_int8(self.sealed_prediction.strat_fold),
            targets=_readonly_int8(aligned_targets),
            raw_logits=_readonly_float64(all_logits[order]),
        )

    def assert_clean_logit_equivalence(self) -> CleanLogitEquivalence:
        """Fail closed unless clean inference is bit-exact to sealed logits."""

        observed = self.infer_logits()
        expected = self.sealed_prediction.raw_logits
        if not np.array_equal(observed.raw_logits, expected):
            absolute = np.abs(observed.raw_logits - expected)
            mismatches = int(np.count_nonzero(observed.raw_logits != expected))
            raise CleanLogitMismatchError(
                self.member_id,
                mismatch_count=mismatches,
                maximum_absolute_error=float(np.max(absolute)),
                mean_absolute_error=float(np.mean(absolute)),
            )
        artifact_hash = self.sealed_prediction.integrity_sha256
        if artifact_hash is None:  # pragma: no cover - loaded artifact invariant
            raise AuditRuntimeIntegrityError("sealed prediction is not integrity-bound")
        return CleanLogitEquivalence(
            member_id=self.member_id,
            record_count=self.sealed_prediction.n_samples,
            logit_count=int(expected.size),
            sealed_prediction_sha256=artifact_hash,
        )

    def _validate_batch(
        self,
        positions: Tensor,
        signals: Tensor,
        targets: Tensor,
    ) -> None:
        if positions.ndim != 1 or signals.ndim != 3 or targets.ndim != 2:
            raise AuditRuntimeIntegrityError("physical loader returned invalid ranks")
        if (
            positions.shape[0] != signals.shape[0]
            or signals.shape[0] != targets.shape[0]
            or targets.shape[1] != len(TARGET_COLUMNS)
        ):
            raise AuditRuntimeIntegrityError("physical loader batch does not align")
        if not torch.isfinite(signals).all().item():
            raise AuditRuntimeIntegrityError("physical loader returned non-finite ECGs")
        if not torch.isfinite(targets).all().item() or not torch.all(
            (targets == 0.0) | (targets == 1.0)
        ).item():
            raise AuditRuntimeIntegrityError(
                "physical loader targets must be finite and binary"
            )

    def _batch_ecg_ids(self, positions: Tensor) -> IdentifierArray:
        indices = positions.numpy().astype(np.int64, copy=False)
        values = self.selected_manifest.iloc[indices]["ecg_id"].to_numpy(copy=True)
        return _readonly_identifier(values)

    def _sealed_order(self) -> NDArray[np.int64]:
        required = {"ecg_id", "patient_id", "strat_fold", *TARGET_COLUMNS}
        missing = sorted(required.difference(self.selected_manifest.columns))
        if missing:
            raise AuditRuntimeIntegrityError(
                f"selected fold-10 manifest is missing columns: {missing}"
            )
        selected_ids = self.selected_manifest["ecg_id"]
        if selected_ids.duplicated().any():
            raise AuditRuntimeIntegrityError("selected fold-10 ECG IDs are not unique")
        indexer = pd.Index(selected_ids).get_indexer(self.sealed_prediction.ecg_id)
        order = np.asarray(indexer, dtype=np.int64)
        if (order < 0).any() or len(order) != len(self.selected_manifest):
            raise AuditRuntimeIntegrityError(
                "sealed prediction ECG IDs differ from the physical manifest"
            )
        selected_ecg = selected_ids.to_numpy()[order]
        selected_patient = self.selected_manifest["patient_id"].to_numpy()[order]
        selected_fold = self.selected_manifest["strat_fold"].to_numpy(
            dtype=np.int8
        )[order]
        if not np.array_equal(selected_ecg, self.sealed_prediction.ecg_id):
            raise AuditRuntimeIntegrityError("sealed ECG identity alignment failed")
        if not np.array_equal(selected_patient, self.sealed_prediction.patient_id):
            raise AuditRuntimeIntegrityError("sealed patient identity alignment failed")
        if not np.array_equal(selected_fold, self.sealed_prediction.strat_fold):
            raise AuditRuntimeIntegrityError("sealed fold identity alignment failed")
        return cast(NDArray[np.int64], order)


@dataclass(frozen=True, slots=True)
class CompletedAuditRuntime:
    """Exact-six verified post-evaluation runtime and its clean gate evidence."""

    protocol: ExperimentProtocol
    final_evaluation_spec: FinalEvaluationSpec
    refit_bundle: RefitBundle
    calibration_bundle: CalibrationBundle
    ledger: VerifiedCompletedLedger
    test_access: FinalTestAccessToken
    members: tuple[AuditMemberRuntime, ...]
    clean_equivalence: tuple[CleanLogitEquivalence, ...]

    def __post_init__(self) -> None:
        member_ids = tuple(member.member_id for member in self.members)
        evidence_ids = tuple(item.member_id for item in self.clean_equivalence)
        if member_ids != EXPECTED_AUDIT_MEMBER_IDS or evidence_ids != member_ids:
            raise AuditRuntimeIntegrityError(
                "completed audit runtime is not the ordered exact-six release"
            )

    def member(self, member_id: str) -> AuditMemberRuntime:
        """Return one member by its immutable release identity."""

        for member in self.members:
            if member.member_id == member_id:
                return member
        raise KeyError(f"unknown audit member {member_id!r}")

    def assert_clean_logit_equivalence(
        self,
    ) -> tuple[CleanLogitEquivalence, ...]:
        """Repeat the exact-six clean gate with the loaded authoritative models."""

        return tuple(member.assert_clean_logit_equivalence() for member in self.members)


def load_completed_audit_runtime(
    *,
    protocol: ExperimentProtocol,
    final_evaluation_spec_path: str | Path,
    refit_bundle_path: str | Path,
    calibration_bundle_path: str | Path,
    ledger_path: str | Path | None = None,
) -> CompletedAuditRuntime:
    """Load and clean-gate the complete sealed six-member fold-10 release."""

    if not isinstance(protocol, ExperimentProtocol):
        raise TypeError("protocol must be an ExperimentProtocol")
    try:
        final_spec = load_final_evaluation_spec(
            final_evaluation_spec_path,
            protocol=protocol,
            verify_sources=True,
            # Post-evaluation code necessarily follows the frozen execution commit.
            verify_runtime=False,
        )
        refit_bundle = load_refit_bundle(
            refit_bundle_path, protocol=protocol, verify_sources=True
        )
        calibration_bundle = load_calibration_bundle(
            calibration_bundle_path,
            protocol=protocol,
            # Full source replay intentionally demands the earlier execution
            # Git revision.  The posthoc runtime verifies every member source
            # below without weakening their content bindings.
            verify_sources=False,
        )
        resolved_ledger_path = (
            _canonical_ledger_path(final_spec)
            if ledger_path is None
            else Path(ledger_path).resolve()
        )
        ledger = _load_verified_completed_ledger(
            resolved_ledger_path,
            protocol=protocol,
            final_spec=final_spec,
            refit_bundle=refit_bundle,
            calibration_bundle=calibration_bundle,
        )
        test_access = authorize_final_test_access(
            protocol,
            purpose=ledger.purpose,
            confirmation=FINAL_TEST_CONFIRMATION,
        )
        refits = {member.member_id: member for member in refit_bundle.members}
        calibrations = {
            member.member_id: member for member in calibration_bundle.members
        }
        plan_members = {
            _string(member["member_id"], "plan member_id"): member
            for member in _mapping_sequence(ledger.plan["members"], "plan members")
        }
        ledger_members = _mapping(ledger.members, "ledger members")
        loaded: list[AuditMemberRuntime] = []
        evidence: list[CleanLogitEquivalence] = []
        reference_prediction: PredictionArtifact | None = None
        for member_id in EXPECTED_AUDIT_MEMBER_IDS:
            runtime = _load_member_runtime(
                refit=refits[member_id],
                calibration=calibrations[member_id],
                planned=plan_members[member_id],
                ledger_state=_mapping(
                    ledger_members[member_id], f"ledger member {member_id}"
                ),
                protocol=protocol,
                test_access=test_access,
            )
            if reference_prediction is not None:
                assert_prediction_artifacts_aligned(
                    reference_prediction, runtime.sealed_prediction
                )
            else:
                reference_prediction = runtime.sealed_prediction
            member_evidence = runtime.assert_clean_logit_equivalence()
            loaded.append(runtime)
            evidence.append(member_evidence)
    except AuditRuntimeError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise AuditRuntimeIntegrityError(
            f"could not load the completed audit runtime: {error}"
        ) from error
    return CompletedAuditRuntime(
        protocol=protocol,
        final_evaluation_spec=final_spec,
        refit_bundle=refit_bundle,
        calibration_bundle=calibration_bundle,
        ledger=ledger,
        test_access=test_access,
        members=tuple(loaded),
        clean_equivalence=tuple(evidence),
    )


def _load_member_runtime(
    *,
    refit: RefitMember,
    calibration: CalibrationMember,
    planned: Mapping[str, object],
    ledger_state: Mapping[str, object],
    protocol: ExperimentProtocol,
    test_access: FinalTestAccessToken,
) -> AuditMemberRuntime:
    _validate_member_bindings(refit, calibration, planned)
    _validate_calibration_source_files(calibration)
    calibration_prediction = load_prediction_artifact(
        calibration.prediction_path,
        protocol=protocol,
        expected_config_hash=refit.resolved_config_hash,
        expected_manifest_hash=refit.manifest_sha256,
    )
    decision = load_calibration_decisions(calibration.decision_path, protocol=protocol)
    _validate_decision(
        decision,
        refit=refit,
        calibration=calibration,
        calibration_prediction=calibration_prediction,
    )
    prediction_path = Path(
        _string(planned["final_prediction_path"], "final prediction path")
    ).resolve()
    prediction = load_prediction_artifact(
        prediction_path,
        protocol=protocol,
        test_access=test_access,
        expected_config_hash=refit.resolved_config_hash,
        expected_manifest_hash=refit.manifest_sha256,
    )
    _validate_prediction(
        prediction,
        prediction_path=prediction_path,
        refit=refit,
        calibration=calibration,
        planned=planned,
        ledger_state=ledger_state,
    )
    inference = _mapping(planned["inference"], "planned inference")
    settings = AuditInferenceSettings(
        batch_size=_integer(inference["batch_size"], "inference.batch_size", minimum=1),
        num_workers=_integer(
            inference["num_workers"], "inference.num_workers", minimum=0
        ),
        device=_string(inference["device"], "inference.device"),
        bf16=_boolean(inference["bf16"], "inference.bf16"),
        seed=refit.seed,
    )
    request = PredictionExportRequest(
        checkpoint_path=refit.final_checkpoint_path,
        resolved_config_path=refit.resolved_config_path,
        output_path=prediction_path,
        fold_role=FoldRole.FINAL_TEST,
        run_metadata_path=refit.metadata_path,
        refit_completion_path=refit.completion_path,
        manifest_path=refit.manifest_path,
        normalization_path=refit.normalization_path,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
        persistent_workers=settings.persistent_workers,
        device=settings.device,
        bf16=settings.bf16,
    )
    run = _load_resolved_run(request.resolved_config_path)
    _validate_requested_lineage(request, run)
    if (
        run.config_hash != refit.resolved_config_hash
        or run.run_name != refit.run_name
        or run.seed != refit.seed
    ):
        raise AuditRuntimeIntegrityError(
            f"resolved config identity differs for {refit.member_id}"
        )
    expected_folds = protocol.folds_for(FoldRole.FINAL_TEST, test_access=test_access)
    inputs = _validate_inputs(
        request,
        run=run,
        protocol=protocol,
        expected_folds=expected_folds,
    )
    if (
        _prefixed(inputs.manifest_hash) != refit.manifest_sha256
        or _prefixed(inputs.normalization_hash) != refit.normalization_sha256
    ):
        raise AuditRuntimeIntegrityError(
            f"validated data sources differ for {refit.member_id}"
        )
    model, checkpoint_hash, checkpoint_epoch = _load_inference_model(
        request,
        run=run,
        protocol=protocol,
        inputs=inputs,
    )
    if _prefixed(checkpoint_hash) != refit.final_checkpoint_sha256:
        raise AuditRuntimeIntegrityError(
            f"loaded checkpoint hash differs for {refit.member_id}"
        )
    runtime = select_device(settings.device, enable_bf16=settings.bf16)
    if str(runtime.device) != settings.device or runtime.bf16_enabled is not settings.bf16:
        raise AuditRuntimeIntegrityError(
            f"runtime cannot reproduce the sealed settings for {refit.member_id}"
        )
    model = model.to(runtime.device)
    physical_dataset = PTBXLDataset(
        inputs.selected_manifest,
        run.dataset_root,
        folds=expected_folds,
        normalization=None,
        protocol=protocol,
        test_access=test_access,
    )
    if len(physical_dataset) != len(inputs.selected_manifest):
        raise AuditRuntimeIntegrityError(
            f"physical dataset size differs for {refit.member_id}"
        )
    return AuditMemberRuntime(
        member_id=refit.member_id,
        architecture=refit.architecture,
        seed=refit.seed,
        refit=refit,
        calibration=calibration,
        decisions=decision,
        sealed_prediction=prediction,
        model=model,
        physical_dataset=physical_dataset,
        selected_manifest=inputs.selected_manifest.copy(),
        normalization=inputs.normalization,
        resolved_config=MappingProxyType(dict(run.config)),
        checkpoint_sha256=_prefixed(checkpoint_hash),
        checkpoint_epoch=checkpoint_epoch,
        settings=settings,
        runtime=runtime,
    )


def _load_verified_completed_ledger(
    path: Path,
    *,
    protocol: ExperimentProtocol,
    final_spec: FinalEvaluationSpec,
    refit_bundle: RefitBundle,
    calibration_bundle: CalibrationBundle,
) -> VerifiedCompletedLedger:
    root = _read_json(path, "completed final-test opening ledger")
    _exact_keys(
        root,
        {
            "schema_version",
            "artifact_type",
            "plan",
            "opening",
            "state",
            "members",
            "outputs",
            "events",
            "updated_at_utc",
            "ledger_sha256",
        },
        "completed final-test opening ledger",
    )
    if root["schema_version"] != FINAL_LEDGER_SCHEMA_VERSION:
        raise AuditRuntimeIntegrityError("unsupported final-test ledger schema")
    if root["artifact_type"] != FINAL_LEDGER_TYPE:
        raise AuditRuntimeIntegrityError("unexpected final-test ledger type")
    ledger_hash = _prefixed_hash(root["ledger_sha256"], "ledger_sha256")
    unhashed_ledger = dict(root)
    del unhashed_ledger["ledger_sha256"]
    if canonical_sha256(unhashed_ledger) != ledger_hash:
        raise AuditRuntimeIntegrityError("final-test ledger self-hash mismatch")
    if root["state"] != "complete":
        raise AuditRuntimeIntegrityError("final-test ledger is not complete")
    plan = _mapping(root["plan"], "final-test plan")
    _validate_plan(
        plan,
        protocol=protocol,
        final_spec=final_spec,
        refit_bundle=refit_bundle,
        calibration_bundle=calibration_bundle,
    )
    batch_sha256 = _prefixed_hash(plan["batch_sha256"], "batch_sha256")
    opening = _mapping(root["opening"], "ledger opening")
    _exact_keys(
        opening,
        {
            "purpose",
            "operator",
            "confirmation_sha256",
            "opening_intent_sha256",
            "created_at_utc",
            "ledger_precedes_fold10_access",
        },
        "ledger opening",
    )
    purpose = _string(opening["purpose"], "opening purpose")
    operator = _string(opening["operator"], "opening operator")
    created = _string(opening["created_at_utc"], "opening created_at_utc")
    if opening["ledger_precedes_fold10_access"] is not True:
        raise AuditRuntimeIntegrityError("ledger does not precede fold-10 access")
    expected_confirmation = hashlib.sha256(
        FINAL_TEST_CONFIRMATION.encode("utf-8")
    ).hexdigest()
    if _raw_hash(opening["confirmation_sha256"], "confirmation_sha256") != (
        expected_confirmation
    ):
        raise AuditRuntimeIntegrityError("ledger confirmation does not authorize fold 10")
    opening_intent = _prefixed_hash(
        opening["opening_intent_sha256"], "opening_intent_sha256"
    )
    expected_intent = canonical_sha256(
        {
            "batch_sha256": batch_sha256,
            "ledger_path": str(path.resolve()),
            "purpose": purpose,
            "operator": operator,
            "confirmation_sha256": expected_confirmation,
            "created_at_utc": created,
            "ledger_precedes_fold10_access": True,
        }
    )
    if opening_intent != expected_intent:
        raise AuditRuntimeIntegrityError("ledger opening intent is invalid")
    plan_members = {
        _string(member["member_id"], "plan member_id"): member
        for member in _mapping_sequence(plan["members"], "plan members")
    }
    member_states = _mapping(root["members"], "ledger members")
    if tuple(sorted(member_states)) != tuple(sorted(EXPECTED_AUDIT_MEMBER_IDS)):
        raise AuditRuntimeIntegrityError("ledger member grid is not exact-six")
    for member_id in EXPECTED_AUDIT_MEMBER_IDS:
        _validate_completed_member_state(
            _mapping(member_states[member_id], f"ledger member {member_id}"),
            planned=plan_members[member_id],
            member_id=member_id,
        )
    events = _sequence(root["events"], "ledger events")
    if not events:
        raise AuditRuntimeIntegrityError("completed ledger has no events")
    for index, raw_event in enumerate(events):
        event = _mapping(raw_event, f"ledger event {index}")
        if event.get("sequence") != index:
            raise AuditRuntimeIntegrityError("ledger event sequence is not contiguous")
    last_event = _mapping(events[-1], "terminal ledger event")
    if last_event.get("event") != "exact_six_member_final_batch_complete":
        raise AuditRuntimeIntegrityError("ledger has no terminal completion event")
    _validate_completed_outputs(_mapping(root["outputs"], "ledger outputs"))
    marker_path = Path(
        _string(plan["opening_marker_path"], "opening marker path")
    ).resolve()
    canonical_ledger_path = _ledger_path_from_marker(marker_path)
    if path.resolve() != canonical_ledger_path:
        raise AuditRuntimeIntegrityError("ledger path is not canonical for its marker")
    _validate_opening_marker(
        marker_path,
        ledger_path=path,
        plan=plan,
        batch_sha256=batch_sha256,
        created_at_utc=created,
        opening_intent_sha256=opening_intent,
    )
    canonical = json.dumps(
        root,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return VerifiedCompletedLedger(
        path=path.resolve(),
        ledger_sha256=ledger_hash,
        batch_sha256=batch_sha256,
        purpose=purpose,
        operator=operator,
        opening_marker_path=marker_path,
        _canonical_payload=canonical,
    )


def _validate_plan(
    plan: Mapping[str, object],
    *,
    protocol: ExperimentProtocol,
    final_spec: FinalEvaluationSpec,
    refit_bundle: RefitBundle,
    calibration_bundle: CalibrationBundle,
) -> None:
    _exact_keys(
        plan,
        {
            "schema_version",
            "artifact_type",
            "protocol_hash",
            "refit_bundle_sha256",
            "calibration_bundle_sha256",
            "manifest_sha256",
            "normalization_sha256",
            "label_order",
            "final_evaluation_spec",
            "opening_marker_path",
            "settings",
            "members",
            "member_count",
            "final_folds",
            "retuning_allowed",
            "batch_sha256",
        },
        "final-test plan",
    )
    if plan["schema_version"] != FINAL_BATCH_PLAN_SCHEMA_VERSION:
        raise AuditRuntimeIntegrityError("unsupported final-test plan schema")
    if plan["artifact_type"] != FINAL_BATCH_PLAN_TYPE:
        raise AuditRuntimeIntegrityError("unexpected final-test plan type")
    if plan["member_count"] != 6 or plan["retuning_allowed"] is not False:
        raise AuditRuntimeIntegrityError("final-test plan cardinality or tuning changed")
    if _integer_tuple(plan["final_folds"], "final_folds") != FINAL_TEST_FOLDS:
        raise AuditRuntimeIntegrityError("final-test plan must contain fold 10 only")
    batch_hash = _prefixed_hash(plan["batch_sha256"], "batch_sha256")
    unhashed_plan = dict(plan)
    del unhashed_plan["batch_sha256"]
    if canonical_sha256(unhashed_plan) != batch_hash:
        raise AuditRuntimeIntegrityError("final-test plan self-hash mismatch")
    refit_hash = _required_bundle_hash(refit_bundle.artifact_sha256, "refit bundle")
    calibration_hash = _required_bundle_hash(
        calibration_bundle.artifact_sha256, "calibration bundle"
    )
    expected_scalars: dict[str, object] = {
        "protocol_hash": protocol.protocol_hash,
        "refit_bundle_sha256": refit_hash,
        "calibration_bundle_sha256": calibration_hash,
        "manifest_sha256": refit_bundle.manifest_sha256,
        "normalization_sha256": refit_bundle.normalization_sha256,
    }
    drift = [field for field, expected in expected_scalars.items() if plan[field] != expected]
    if drift:
        raise AuditRuntimeIntegrityError(
            "final-test plan differs from verified release: " + ", ".join(drift)
        )
    if calibration_bundle.refit_bundle_sha256 != refit_hash:
        raise AuditRuntimeIntegrityError("calibration bundle binds another refit bundle")
    if (
        calibration_bundle.protocol_hash != refit_bundle.protocol_hash
        or calibration_bundle.manifest_sha256 != refit_bundle.manifest_sha256
        or calibration_bundle.normalization_sha256 != refit_bundle.normalization_sha256
        or calibration_bundle.label_order != refit_bundle.label_order
    ):
        raise AuditRuntimeIntegrityError("verified release bundles do not align")
    if _string_tuple(plan["label_order"], "label_order") != LABEL_ORDER:
        raise AuditRuntimeIntegrityError("final-test label order changed")
    spec_binding = _mapping(plan["final_evaluation_spec"], "final spec binding")
    _exact_keys(
        spec_binding,
        {"path", "file_sha256", "artifact_sha256"},
        "final spec binding",
    )
    if final_spec.path is None:  # pragma: no cover - loaded spec invariant
        raise AuditRuntimeIntegrityError("final-evaluation spec has no source path")
    if (
        Path(_string(spec_binding["path"], "final spec path")).resolve()
        != final_spec.path
        or spec_binding["artifact_sha256"] != final_spec.artifact_sha256
        or spec_binding["file_sha256"] != _prefixed(sha256_file(final_spec.path))
    ):
        raise AuditRuntimeIntegrityError("final-test plan binds another specification")
    refits = {member.member_id: member for member in refit_bundle.members}
    calibrations = {member.member_id: member for member in calibration_bundle.members}
    if set(refits) != set(EXPECTED_AUDIT_MEMBER_IDS) or set(calibrations) != set(
        EXPECTED_AUDIT_MEMBER_IDS
    ):
        raise AuditRuntimeIntegrityError("verified release bundles are not exact-six")
    plan_members = _mapping_sequence(plan["members"], "plan members")
    if len(plan_members) != 6:
        raise AuditRuntimeIntegrityError("final-test plan is not exact-six")
    observed: list[str] = []
    for member in plan_members:
        member_id = _string(member["member_id"], "plan member_id")
        if member_id not in refits or member_id in observed:
            raise AuditRuntimeIntegrityError("final-test plan member grid is invalid")
        observed.append(member_id)
        _validate_member_bindings(refits[member_id], calibrations[member_id], member)
    if tuple(observed) != EXPECTED_AUDIT_MEMBER_IDS:
        raise AuditRuntimeIntegrityError("final-test plan member order changed")
    _validate_settings(_mapping(plan["settings"], "final-test settings"))


def _validate_member_bindings(
    refit: RefitMember,
    calibration: CalibrationMember,
    planned: Mapping[str, object],
) -> None:
    _exact_keys(
        planned,
        {
            "member_id",
            "architecture",
            "seed",
            "model_name",
            "refit_lineage_sha256",
            "checkpoint_sha256",
            "resolved_config_hash",
            "calibration_decision_sha256",
            "fold9_prediction_sha256",
            "inference",
            "final_prediction_path",
            "final_report_path",
        },
        f"plan member {refit.member_id}",
    )
    expected: dict[str, object] = {
        "member_id": refit.member_id,
        "architecture": refit.architecture,
        "seed": refit.seed,
        "model_name": refit.run_name,
        "refit_lineage_sha256": refit.lineage_sha256,
        "checkpoint_sha256": refit.final_checkpoint_sha256,
        "resolved_config_hash": refit.resolved_config_hash,
        "calibration_decision_sha256": calibration.decision_artifact_sha256,
        "fold9_prediction_sha256": calibration.prediction_artifact_sha256,
    }
    drift = [field for field, value in expected.items() if planned[field] != value]
    if drift:
        raise AuditRuntimeIntegrityError(
            f"planned member {refit.member_id} differs: " + ", ".join(drift)
        )
    pair_expected: dict[str, object] = {
        "member_id": refit.member_id,
        "architecture": refit.architecture,
        "seed": refit.seed,
        "model_name": refit.run_name,
        "refit_lineage_sha256": refit.lineage_sha256,
        "checkpoint_path": refit.final_checkpoint_path,
        "checkpoint_sha256": refit.final_checkpoint_sha256,
        "resolved_config_path": refit.resolved_config_path,
        "resolved_config_hash": refit.resolved_config_hash,
        "resolved_config_file_sha256": refit.resolved_config_file_sha256,
        "normalization_path": refit.normalization_path,
        "normalization_sha256": refit.normalization_sha256,
    }
    pair_observed: dict[str, object] = {
        "member_id": calibration.member_id,
        "architecture": calibration.architecture,
        "seed": calibration.seed,
        "model_name": calibration.model_name,
        "refit_lineage_sha256": calibration.refit_lineage_sha256,
        "checkpoint_path": calibration.checkpoint_path,
        "checkpoint_sha256": calibration.checkpoint_sha256,
        "resolved_config_path": calibration.resolved_config_path,
        "resolved_config_hash": calibration.resolved_config_hash,
        "resolved_config_file_sha256": calibration.resolved_config_file_sha256,
        "normalization_path": calibration.normalization_path,
        "normalization_sha256": calibration.normalization_sha256,
    }
    pair_drift = [
        field for field, value in pair_expected.items() if pair_observed[field] != value
    ]
    if pair_drift:
        raise AuditRuntimeIntegrityError(
            f"refit/calibration member {refit.member_id} differs: "
            + ", ".join(pair_drift)
        )
    inference = _mapping(planned["inference"], "planned inference")
    _exact_keys(
        inference,
        {"batch_size", "num_workers", "device", "bf16"},
        "planned inference",
    )
    AuditInferenceSettings(
        batch_size=_integer(inference["batch_size"], "batch_size", minimum=1),
        num_workers=_integer(inference["num_workers"], "num_workers", minimum=0),
        device=_string(inference["device"], "device"),
        bf16=_boolean(inference["bf16"], "bf16"),
        seed=refit.seed,
    )
    if Path(_string(planned["final_prediction_path"], "prediction path")).suffix != ".npz":
        raise AuditRuntimeIntegrityError("planned prediction path must end in .npz")
    if Path(_string(planned["final_report_path"], "report path")).suffix != ".json":
        raise AuditRuntimeIntegrityError("planned report path must end in .json")


def _validate_completed_member_state(
    state: Mapping[str, object],
    *,
    planned: Mapping[str, object],
    member_id: str,
) -> None:
    _exact_keys(
        state,
        {
            "state",
            "final_prediction_path",
            "final_prediction_artifact_sha256",
            "final_prediction_file_sha256",
            "final_prediction_sidecar_sha256",
            "final_report_path",
            "final_report_sha256",
        },
        f"ledger member {member_id}",
    )
    if state["state"] != "report_saved":
        raise AuditRuntimeIntegrityError(f"ledger member {member_id} is incomplete")
    if (
        state["final_prediction_path"] != planned["final_prediction_path"]
        or state["final_report_path"] != planned["final_report_path"]
    ):
        raise AuditRuntimeIntegrityError(f"ledger paths differ for {member_id}")
    _prefixed_hash(
        state["final_prediction_artifact_sha256"],
        f"{member_id} final prediction artifact hash",
    )
    _raw_hash(
        state["final_prediction_file_sha256"],
        f"{member_id} final prediction file hash",
    )
    _raw_hash(
        state["final_prediction_sidecar_sha256"],
        f"{member_id} final prediction sidecar hash",
    )
    _prefixed_hash(
        state["final_report_sha256"], f"{member_id} final report hash"
    )


def _validate_prediction(
    prediction: PredictionArtifact,
    *,
    prediction_path: Path,
    refit: RefitMember,
    calibration: CalibrationMember,
    planned: Mapping[str, object],
    ledger_state: Mapping[str, object],
) -> None:
    expected: dict[str, object] = {
        "model_name": refit.run_name,
        "model_seed": refit.seed,
        "config_hash": refit.resolved_config_hash,
        "manifest_hash": refit.manifest_sha256,
        "label_order": LABEL_ORDER,
        "fold_role": FoldRole.FINAL_TEST,
        "folds": FINAL_TEST_FOLDS,
        "integrity_sha256": ledger_state["final_prediction_artifact_sha256"],
    }
    observed: dict[str, object] = {
        "model_name": prediction.model_name,
        "model_seed": prediction.model_seed,
        "config_hash": prediction.config_hash,
        "manifest_hash": prediction.manifest_hash,
        "label_order": prediction.label_order,
        "fold_role": prediction.fold_role,
        "folds": prediction.folds,
        "integrity_sha256": prediction.integrity_sha256,
    }
    drift = [field for field, value in expected.items() if observed[field] != value]
    if drift:
        raise AuditRuntimeIntegrityError(
            f"sealed prediction {refit.member_id} differs: " + ", ".join(drift)
        )
    if prediction.integrity_sha256 is None:  # pragma: no cover - loaded invariant
        raise AuditRuntimeIntegrityError("sealed prediction has no artifact hash")
    if prediction.integrity_sha256 != ledger_state["final_prediction_artifact_sha256"]:
        raise AuditRuntimeIntegrityError("ledger prediction artifact hash differs")
    if sha256_file(prediction_path) != ledger_state["final_prediction_file_sha256"]:
        raise AuditRuntimeIntegrityError("ledger prediction file hash differs")
    sidecar = prediction_path.with_suffix(".json")
    if sha256_file(sidecar) != ledger_state["final_prediction_sidecar_sha256"]:
        raise AuditRuntimeIntegrityError("ledger prediction sidecar hash differs")
    metadata = prediction.extra_metadata
    inference = _mapping(planned["inference"], "planned inference")
    metadata_expected: dict[str, object] = {
        "lineage": "frozen_refit",
        "checkpoint_path": str(refit.final_checkpoint_path.resolve()),
        "checkpoint_sha256": refit.final_checkpoint_sha256.removeprefix("sha256:"),
        "resolved_config_path": str(refit.resolved_config_path.resolve()),
        "normalization_sha256": refit.normalization_sha256.removeprefix("sha256:"),
        "inference_device": inference["device"],
        "inference_bf16": inference["bf16"],
        "inference_batch_size": inference["batch_size"],
        "inference_num_workers": inference["num_workers"],
        "refit_completion_sha256": refit.completion_sha256,
    }
    metadata_drift = [
        field for field, value in metadata_expected.items() if metadata.get(field) != value
    ]
    if metadata_drift:
        raise AuditRuntimeIntegrityError(
            f"sealed prediction metadata differs for {refit.member_id}: "
            + ", ".join(metadata_drift)
        )
    if calibration.model_name != prediction.model_name:
        raise AuditRuntimeIntegrityError("calibration/final prediction model differs")


def _validate_decision(
    decision: CalibrationDecisionArtifact,
    *,
    refit: RefitMember,
    calibration: CalibrationMember,
    calibration_prediction: PredictionArtifact,
) -> None:
    expected: dict[str, object] = {
        "model_name": refit.run_name,
        "model_seed": refit.seed,
        "protocol_hash": refit.protocol_hash,
        "config_hash": refit.resolved_config_hash,
        "manifest_hash": refit.manifest_sha256,
        "label_order": LABEL_ORDER,
        "integrity_sha256": calibration.decision_artifact_sha256,
        "temperature": calibration.temperature,
        "thresholds": calibration.thresholds,
        "source_prediction_sha256": calibration.prediction_artifact_sha256,
        "source_alignment_sha256": calibration.prediction_alignment_sha256,
    }
    observed: dict[str, object] = {
        "model_name": decision.model_name,
        "model_seed": decision.model_seed,
        "protocol_hash": decision.protocol_hash,
        "config_hash": decision.config_hash,
        "manifest_hash": decision.manifest_hash,
        "label_order": decision.label_order,
        "integrity_sha256": decision.integrity_sha256,
        "temperature": decision.temperature_scaling.temperature,
        "thresholds": decision.threshold_optimization.thresholds,
        "source_prediction_sha256": decision.source_prediction_sha256,
        "source_alignment_sha256": decision.source_alignment_sha256,
    }
    drift = [field for field, value in expected.items() if observed[field] != value]
    if drift:
        raise AuditRuntimeIntegrityError(
            f"calibration decisions differ for {refit.member_id}: " + ", ".join(drift)
        )
    if (
        calibration_prediction.integrity_sha256
        != calibration.prediction_artifact_sha256
        or calibration_prediction.alignment_sha256
        != calibration.prediction_alignment_sha256
        or calibration_prediction.fold_role is not FoldRole.CALIBRATION
        or calibration_prediction.folds != (9,)
    ):
        raise AuditRuntimeIntegrityError(
            f"fold-9 calibration prediction differs for {refit.member_id}"
        )
    observed_gates = tuple(gate.to_dict() for gate in decision.coverage_gates)
    expected_gates = tuple(dict(gate) for gate in calibration.entropy_gates)
    if observed_gates != expected_gates:
        raise AuditRuntimeIntegrityError(
            f"calibration coverage gates differ for {refit.member_id}"
        )


def _validate_calibration_source_files(calibration: CalibrationMember) -> None:
    expected_prefixed: tuple[tuple[Path, str, str], ...] = (
        (
            calibration.checkpoint_path,
            calibration.checkpoint_sha256,
            "checkpoint",
        ),
        (
            calibration.resolved_config_path,
            calibration.resolved_config_file_sha256,
            "resolved config",
        ),
        (
            calibration.normalization_path,
            calibration.normalization_sha256,
            "normalization",
        ),
    )
    for path, expected, context in expected_prefixed:
        if _prefixed(sha256_file(path)) != expected:
            raise AuditRuntimeIntegrityError(
                f"calibration {context} changed for {calibration.member_id}"
            )
    expected_raw: tuple[tuple[Path, str, str], ...] = (
        (
            calibration.prediction_path,
            calibration.prediction_npz_sha256,
            "fold-9 prediction",
        ),
        (
            calibration.prediction_sidecar_path,
            calibration.prediction_sidecar_sha256,
            "fold-9 prediction sidecar",
        ),
        (
            calibration.decision_path,
            calibration.decision_file_sha256,
            "calibration decision",
        ),
    )
    for path, expected, context in expected_raw:
        if sha256_file(path) != expected:
            raise AuditRuntimeIntegrityError(
                f"{context} changed for {calibration.member_id}"
            )


def _validate_settings(settings: Mapping[str, object]) -> None:
    _exact_keys(
        settings,
        {"output_directory", "subgroups", "inference", "evaluation", "retuning_allowed"},
        "final-test settings",
    )
    if settings["retuning_allowed"] is not False:
        raise AuditRuntimeIntegrityError("final-test settings allow retuning")
    _string(settings["output_directory"], "settings.output_directory")
    subgroups = _mapping(settings["subgroups"], "settings.subgroups")
    _exact_keys(subgroups, {"path", "sha256"}, "settings.subgroups")
    _string(subgroups["path"], "settings.subgroups.path")
    _raw_hash(subgroups["sha256"], "settings.subgroups.sha256")
    inference = _mapping(settings["inference"], "settings.inference")
    _exact_keys(
        inference,
        {"batch_size", "num_workers", "device", "bf16"},
        "settings.inference",
    )
    if inference["batch_size"] is not None or inference["num_workers"] is not None:
        raise AuditRuntimeIntegrityError("final-test settings contain loader overrides")
    _string(inference["device"], "settings.inference.device")
    _boolean(inference["bf16"], "settings.inference.bf16")
    evaluation = _mapping(settings["evaluation"], "settings.evaluation")
    _exact_keys(
        evaluation,
        {
            "bootstrap_resamples",
            "bootstrap_seed",
            "bootstrap_confidence",
            "bootstrap_minimum_valid",
            "minimum_group_samples",
            "minimum_group_patients",
            "ece_bins",
            "paired_bootstrap_seed_strategy",
        },
        "settings.evaluation",
    )
    if evaluation["paired_bootstrap_seed_strategy"] != "base_plus_model_seed":
        raise AuditRuntimeIntegrityError("paired bootstrap seed strategy changed")


def _validate_completed_outputs(outputs: Mapping[str, object]) -> None:
    expected = {
        "batch_summary_path",
        "batch_summary_sha256",
        "paired_manifest_path",
        "paired_manifest_sha256",
        *(
            f"architecture_{architecture}_{suffix}"
            for architecture in EXPECTED_ARCHITECTURES
            for suffix in ("path", "sha256")
        ),
    }
    _exact_keys(outputs, expected, "completed ledger outputs")
    for key, value in outputs.items():
        if key.endswith("_path"):
            path = Path(_string(value, key)).resolve()
            if not path.is_file():
                raise AuditRuntimeIntegrityError(f"completed output is missing: {path}")
        else:
            _prefixed_hash(value, key)


def _validate_opening_marker(
    path: Path,
    *,
    ledger_path: Path,
    plan: Mapping[str, object],
    batch_sha256: str,
    created_at_utc: str,
    opening_intent_sha256: str,
) -> None:
    marker = _read_json(path, "canonical fold-10 opening marker")
    _exact_keys(
        marker,
        {
            "schema_version",
            "artifact_type",
            "batch_sha256",
            "refit_bundle_sha256",
            "calibration_bundle_sha256",
            "ledger_path",
            "output_directory",
            "created_at_utc",
            "opening_intent_sha256",
            "marker_precedes_fold10_access",
            "marker_sha256",
        },
        "canonical fold-10 opening marker",
    )
    marker_hash = _prefixed_hash(marker["marker_sha256"], "marker_sha256")
    unhashed = dict(marker)
    del unhashed["marker_sha256"]
    if canonical_sha256(unhashed) != marker_hash:
        raise AuditRuntimeIntegrityError("opening marker self-hash mismatch")
    settings = _mapping(plan["settings"], "plan settings")
    expected: dict[str, object] = {
        "schema_version": FINAL_OPENING_MARKER_SCHEMA_VERSION,
        "artifact_type": FINAL_OPENING_MARKER_TYPE,
        "batch_sha256": batch_sha256,
        "refit_bundle_sha256": plan["refit_bundle_sha256"],
        "calibration_bundle_sha256": plan["calibration_bundle_sha256"],
        "ledger_path": str(ledger_path.resolve()),
        "output_directory": str(
            Path(_string(settings["output_directory"], "output_directory")).resolve()
        ),
        "created_at_utc": created_at_utc,
        "opening_intent_sha256": opening_intent_sha256,
        "marker_precedes_fold10_access": True,
    }
    drift = [field for field, value in expected.items() if marker.get(field) != value]
    if drift:
        raise AuditRuntimeIntegrityError(
            "opening marker differs from completed ledger: " + ", ".join(drift)
        )


def _canonical_ledger_path(final_spec: FinalEvaluationSpec) -> Path:
    if final_spec.path is None:  # pragma: no cover - loaded spec invariant
        raise AuditRuntimeIntegrityError("final-evaluation spec has no source path")
    release_root = final_spec.path.resolve().parent.parent
    return (
        release_root
        / ".final-test-openings"
        / (
            final_spec.artifact_sha256.removeprefix("sha256:")
            + ".opening-ledger.json"
        )
    ).resolve()


def _ledger_path_from_marker(marker: Path) -> Path:
    suffix = ".opening.json"
    if not marker.name.endswith(suffix):
        raise AuditRuntimeIntegrityError("opening marker filename is not canonical")
    return marker.with_name(
        marker.name[: -len(suffix)] + ".opening-ledger.json"
    ).resolve()


def _read_json(path: Path, context: str) -> Mapping[str, object]:
    source = path.resolve()
    if not source.is_file():
        raise AuditRuntimeIntegrityError(f"{context} is missing: {source}")
    if source.stat().st_size > 100_000_000:
        raise AuditRuntimeIntegrityError(f"{context} is unreasonably large")
    try:
        decoded: object = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuditRuntimeIntegrityError(f"could not decode {context}: {error}") from error
    return _mapping(decoded, context)


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise AuditRuntimeIntegrityError(f"{context} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _sequence(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise AuditRuntimeIntegrityError(f"{context} must be a sequence")
    return cast(Sequence[object], value)


def _mapping_sequence(
    value: object, context: str
) -> tuple[Mapping[str, object], ...]:
    return tuple(_mapping(item, context) for item in _sequence(value, context))


def _exact_keys(
    value: Mapping[str, object], expected: set[str], context: str
) -> None:
    if set(value) != expected:
        missing = sorted(expected.difference(value))
        unexpected = sorted(set(value).difference(expected))
        raise AuditRuntimeIntegrityError(
            f"{context} keys differ; missing={missing}, unexpected={unexpected}"
        )


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuditRuntimeIntegrityError(f"{context} must be a non-empty string")
    return value


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise AuditRuntimeIntegrityError(f"{context} must be boolean")
    return value


def _integer(value: object, context: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AuditRuntimeIntegrityError(
            f"{context} must be an integer >= {minimum}"
        )
    return value


def _integer_tuple(value: object, context: str) -> tuple[int, ...]:
    return tuple(_integer(item, context, minimum=0) for item in _sequence(value, context))


def _string_tuple(value: object, context: str) -> tuple[str, ...]:
    return tuple(_string(item, context) for item in _sequence(value, context))


def _prefixed_hash(value: object, context: str) -> str:
    text = _string(value, context)
    digest = text.removeprefix("sha256:")
    if (
        not text.startswith("sha256:")
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise AuditRuntimeIntegrityError(
            f"{context} must be a prefixed lower-case SHA-256"
        )
    return text


def _raw_hash(value: object, context: str) -> str:
    text = _string(value, context)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise AuditRuntimeIntegrityError(
            f"{context} must be an unprefixed lower-case SHA-256"
        )
    return text


def _prefixed(raw_hash: str) -> str:
    return "sha256:" + _raw_hash(raw_hash, "source SHA-256")


def _required_bundle_hash(value: str | None, context: str) -> str:
    if value is None:
        raise AuditRuntimeIntegrityError(f"{context} is not integrity-bound")
    return _prefixed_hash(value, f"{context} artifact_sha256")


def _readonly_identifier(value: object) -> IdentifierArray:
    array = np.array(value, copy=True)
    if array.dtype.kind in {"i", "u"}:
        result = array.astype(np.int64, copy=False)
    elif array.dtype.kind in {"U", "S"}:
        result = array.astype(np.str_, copy=False)
    else:
        raise AuditRuntimeIntegrityError("ECG identifiers must be integer or string")
    result.setflags(write=False)
    return cast(IdentifierArray, result)


def _readonly_int8(value: object) -> NDArray[np.int8]:
    result = np.array(value, dtype=np.int8, copy=True)
    result.setflags(write=False)
    return result


def _readonly_float64(value: object) -> NDArray[np.float64]:
    result = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise AuditRuntimeIntegrityError("audit logits contain non-finite values")
    result.setflags(write=False)
    return result


__all__ = [
    "EXPECTED_AUDIT_MEMBER_IDS",
    "AlignedAuditInference",
    "AuditInferenceSettings",
    "AuditMemberRuntime",
    "AuditRuntimeError",
    "AuditRuntimeIntegrityError",
    "CleanLogitEquivalence",
    "CleanLogitMismatchError",
    "CompletedAuditRuntime",
    "PhysicalBatchTransform",
    "VerifiedCompletedLedger",
    "load_completed_audit_runtime",
]
