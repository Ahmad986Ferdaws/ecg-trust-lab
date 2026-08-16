"""Frozen SPH external-transport inference over the sealed exact-six release.

The module deliberately separates private row identity from identifier-free public
predictions.  SPH is never used to fit normalization, temperatures, thresholds,
abstention gates, or model parameters.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import tarfile
from collections.abc import Callable, Mapping, Sequence, Sized
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ecg_trust.audit_artifacts import save_audit_array_artifact
from ecg_trust.audit_runtime import (
    EXPECTED_AUDIT_MEMBER_IDS,
    AuditMemberRuntime,
    CompletedAuditRuntime,
    load_completed_audit_runtime,
)
from ecg_trust.constants import LEADS, TARGET_COLUMNS
from ecg_trust.data.sph import (
    SPH_AMBIGUOUS_PRIMARY_CODES,
    SPH_HDF5_DATASET_KEY,
    SPH_NATIVE_FREQUENCY_HZ,
    SPH_NATIVE_SAMPLES,
    SPH_SIGNAL_UNIT,
    SPH_SUPERCLASS_CODES,
    SPH_TARGET_FREQUENCY_HZ,
    SPH_TARGET_SAMPLES,
    SPHExternalTransportDataset,
    build_sph_transport_manifest,
    select_sph_exact_10s_records,
    select_sph_transport_records,
)
from ecg_trust.evaluation import stable_sigmoid
from ecg_trust.post_analysis import mean_normalized_binary_entropy
from ecg_trust.protocol import LABEL_ORDER, load_protocol
from ecg_trust.release_gates import canonical_sha256, sha256_file, write_new_hashed_json
from ecg_trust.sph_transport_metrics import (
    evaluate_sph_transport,
    paired_sph_transport_differences,
)
from ecg_trust.training import seed_dataloader_worker

SPH_TRANSPORT_SCHEMA_VERSION = 1
SPH_TRANSPORT_PROTOCOL_ID = "sph-external-transport-v1"
SPH_TRANSPORT_CONFIG = Path("configs/external_transport_sph_frozen.yaml")
SPH_TRANSPORT_PREDICTION_TYPE = "ecg_trust.sph_transport_predictions"
SPH_TRANSPORT_ALIGNMENT_TYPE = "ecg_trust.sph_transport_private_alignment"
SPH_TRANSPORT_QC_TYPE = "ecg_trust.sph_transport_signal_qc"
COHORT_ORDER = ("primary_mapped", "broad_exact10", "no_ambiguous_mapped")
EXPECTED_COHORT_COUNTS = {
    "primary_mapped": (15_698, 15_193),
    "broad_exact10": (18_842, 18_157),
    "no_ambiguous_mapped": (15_563, 15_066),
}
EXPECTED_POSITIVE_RECORDS = {
    "primary_mapped": (11_172, 138, 3_030, 1_510, 113),
    "broad_exact10": (11_172, 138, 3_030, 1_510, 113),
    "no_ambiguous_mapped": (11_172, 131, 2_981, 1_470, 64),
}
EXPECTED_POSITIVE_PATIENTS = {
    "primary_mapped": (10_874, 131, 2_947, 1_453, 110),
    "broad_exact10": (10_874, 131, 2_947, 1_453, 110),
    "no_ambiguous_mapped": (10_874, 124, 2_899, 1_417, 63),
}
EXPECTED_ECE_BINS = 15
EXPECTED_BOOTSTRAP_RESAMPLES = 1_000
EXPECTED_BOOTSTRAP_CONFIDENCE = 0.95
EXPECTED_BOOTSTRAP_MINIMUM_VALID = 500
EXPECTED_BOOTSTRAP_BASE_SEED = 20_260_816
EXPECTED_PUBLIC_ROW_PERMUTATION_SEED = 2_026_081_601
EXPECTED_COHORT_SEED_OFFSETS = {
    "primary_mapped": 0,
    "broad_exact10": 100_000,
    "no_ambiguous_mapped": 200_000,
}
EXPECTED_GATE_TARGETS = (1.0, 0.9, 0.8, 0.7, 0.5)
_SOURCE_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9])[AS][0-9]{5}(?![0-9])")

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int8]
BoolArray = NDArray[np.bool_]
Progress = Callable[[str], None]


class SPHTransportError(RuntimeError):
    """Raised when the frozen transport study cannot proceed safely."""


class SPHTransportIntegrityError(SPHTransportError):
    """Raised when a source, cohort, runtime, or artifact differs from protocol."""


@dataclass(frozen=True, slots=True)
class SPHTransportSpec:
    """Validated paths and settings from the frozen preregistration."""

    path: Path
    project_root: Path
    file_sha256: str
    metadata_path: Path
    code_dictionary_path: Path
    rule_path: Path
    records_archive_path: Path
    records_dir: Path
    normalization_path: Path
    refit_bundle_path: Path
    calibration_bundle_path: Path
    final_evaluation_spec_path: Path
    opening_ledger_path: Path
    protocol_path: Path
    output_root: Path
    public_row_permutation_seed: int
    _payload_json: str

    @property
    def payload(self) -> dict[str, object]:
        decoded: object = json.loads(self._payload_json)
        if not isinstance(decoded, dict):  # pragma: no cover - constructor invariant
            raise SPHTransportIntegrityError("frozen protocol payload is not an object")
        return cast(dict[str, object], decoded)


@dataclass(frozen=True, slots=True)
class SPHTransportCohorts:
    """Canonical broad inference rows plus preregistered post-inference masks."""

    manifest: pd.DataFrame
    ecg_ids: NDArray[np.str_]
    patient_ids: NDArray[np.str_]
    targets: IntArray
    masks: Mapping[str, BoolArray]
    alignment_sha256: str
    summaries: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class SPHTransportInference:
    """One physical-data pass through all six frozen models."""

    targets: IntArray
    raw_logits: Mapping[str, FloatArray]
    per_record_max_abs_mv: FloatArray
    per_record_lead_max_abs_mv: FloatArray
    physical_signal_sha256: str
    per_lead_qc: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class SPHFrozenMemberOutput:
    """Identifier-free predictions derived from one frozen member."""

    member_id: str
    architecture: str
    seed: int
    raw_logits: FloatArray
    raw_probabilities: FloatArray
    calibrated_probabilities: FloatArray
    predictions: BoolArray
    uncertainty: FloatArray
    gate_selected: BoolArray
    temperature: float
    thresholds: tuple[float, ...]
    entropy_gate_cutoffs: tuple[float, ...]
    entropy_gate_targets: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SPHTransportEvaluation:
    """All member/cohort reports and paired architecture comparisons."""

    member_outputs: Mapping[str, SPHFrozenMemberOutput]
    member_reports: Mapping[str, Mapping[str, object]]
    paired_reports: Mapping[str, Mapping[str, object]]
    architecture_summaries: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class SPHTransportRun:
    """Committed immutable output identity."""

    output_root: Path
    public_manifest_path: Path
    public_manifest_sha256: str
    private_alignment_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "output_root": str(self.output_root),
            "public_manifest_path": str(self.public_manifest_path),
            "public_manifest_sha256": self.public_manifest_sha256,
            "private_alignment_sha256": self.private_alignment_sha256,
        }


@dataclass(frozen=True, slots=True)
class SPHTransportAttempt:
    """Immutable reservation proving that an official attempt started."""

    output_root: Path
    marker_path: Path
    marker_sha256: str


class _IndexedDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, dataset: Dataset[tuple[Tensor, Tensor]]) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        signal, target = self.dataset[index]
        return torch.tensor(index, dtype=torch.int64), signal, target


def load_sph_transport_spec(path: str | Path = SPH_TRANSPORT_CONFIG) -> SPHTransportSpec:
    """Load and strictly validate the frozen SPH transport YAML and bound files."""

    source = Path(path).resolve()
    try:
        decoded: object = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise SPHTransportIntegrityError(f"could not load frozen SPH protocol: {error}") from error
    payload = _mapping(decoded, "frozen SPH protocol")
    if payload.get("schema_version") != SPH_TRANSPORT_SCHEMA_VERSION:
        raise SPHTransportIntegrityError("unsupported SPH protocol schema_version")
    if payload.get("protocol_id") != SPH_TRANSPORT_PROTOCOL_ID:
        raise SPHTransportIntegrityError("unexpected SPH transport protocol_id")
    if payload.get("status") != "frozen_before_first_inference":
        raise SPHTransportIntegrityError("SPH protocol was not frozen before inference")

    project_root = source.parent.parent.resolve()
    purpose = _mapping(payload.get("purpose"), "purpose")
    if (
        purpose.get("target_order") != list(LABEL_ORDER)
        or purpose.get("clinical_validation") is not False
    ):
        raise SPHTransportIntegrityError("SPH purpose/label contract differs")
    _validate_signal_and_label_contract(payload)
    _validate_evaluation_contract(payload)

    source_section = _mapping(payload.get("source"), "source")
    files = _mapping(source_section.get("files"), "source.files")
    metadata_path = _verify_file_binding(
        _mapping(files.get("metadata"), "metadata"), project_root, "metadata"
    )
    code_path = _verify_file_binding(
        _mapping(files.get("diagnostic_dictionary"), "diagnostic_dictionary"),
        project_root,
        "diagnostic_dictionary",
    )
    rule_path = _verify_file_binding(
        _mapping(files.get("translation_rules"), "translation_rules"),
        project_root,
        "translation_rules",
    )
    archive_binding = _mapping(files.get("records_archive"), "records_archive")
    archive_path = _verify_file_binding(archive_binding, project_root, "records_archive")
    records_dir = _resolve_bound_path(
        archive_binding.get("extracted_root"), project_root, "records_archive.extracted_root"
    )
    if not records_dir.is_dir():
        raise SPHTransportIntegrityError("SPH extracted records directory is missing")

    signal = _mapping(payload.get("signal_contract"), "signal_contract")
    normalization = _mapping(signal.get("normalization"), "normalization")
    normalization_path = _verify_path_hash_binding(
        normalization, project_root, "normalization", hash_key="sha256"
    )
    frozen_models = _mapping(payload.get("frozen_models"), "frozen_models")
    refit = _mapping(frozen_models.get("refit_bundle"), "refit_bundle")
    calibration = _mapping(frozen_models.get("calibration_bundle"), "calibration_bundle")
    refit_path = _verify_path_hash_binding(refit, project_root, "refit_bundle")
    calibration_path = _verify_path_hash_binding(calibration, project_root, "calibration_bundle")
    member_ids = tuple(_string_sequence(frozen_models.get("members"), "members"))
    if member_ids != EXPECTED_AUDIT_MEMBER_IDS:
        raise SPHTransportIntegrityError("frozen model grid is not the ordered exact six")
    lineage = _mapping(frozen_models.get("runtime_lineage"), "runtime_lineage")
    if lineage.get("required_loader") != "load_completed_audit_runtime":
        raise SPHTransportIntegrityError("frozen runtime loader differs")
    protocol_binding = _mapping(lineage.get("protocol"), "runtime_lineage.protocol")
    final_spec_binding = _mapping(
        lineage.get("final_evaluation_spec"), "runtime_lineage.final_evaluation_spec"
    )
    ledger_binding = _mapping(lineage.get("opening_ledger"), "runtime_lineage.opening_ledger")
    protocol_path = _verify_path_hash_binding(protocol_binding, project_root, "runtime protocol")
    final_spec_path = _verify_path_hash_binding(
        final_spec_binding, project_root, "final_evaluation_spec"
    )
    ledger_path = _verify_path_hash_binding(ledger_binding, project_root, "opening_ledger")
    loaded_protocol = load_protocol(protocol_path)
    if loaded_protocol.protocol_hash != _prefixed_hash(
        protocol_binding.get("protocol_hash"), "runtime protocol.protocol_hash"
    ):
        raise SPHTransportIntegrityError("canonical PTB protocol hash differs")
    outputs = _mapping(payload.get("outputs"), "outputs")
    if outputs.get("overwrite") is not False or outputs.get("if_populated") != "fail_closed":
        raise SPHTransportIntegrityError("SPH output immutability contract differs")
    output_root = _resolve_bound_path(outputs.get("root"), project_root, "outputs.root")
    privacy = _mapping(outputs.get("privacy"), "outputs.privacy")
    permutation_seed = _integer(
        privacy.get("public_row_permutation_seed"),
        "outputs.privacy.public_row_permutation_seed",
    )
    if (
        permutation_seed != EXPECTED_PUBLIC_ROW_PERMUTATION_SEED
        or privacy.get("public_row_identifiers_allowed") is not False
        or privacy.get("public_paths_or_codes_allowed") is not False
    ):
        raise SPHTransportIntegrityError("public row privacy/ordering contract differs")

    return SPHTransportSpec(
        path=source,
        project_root=project_root,
        file_sha256=_prefixed_file_sha256(source),
        metadata_path=metadata_path,
        code_dictionary_path=code_path,
        rule_path=rule_path,
        records_archive_path=archive_path,
        records_dir=records_dir,
        normalization_path=normalization_path,
        refit_bundle_path=refit_path,
        calibration_bundle_path=calibration_path,
        final_evaluation_spec_path=final_spec_path,
        opening_ledger_path=ledger_path,
        protocol_path=protocol_path,
        output_root=output_root,
        public_row_permutation_seed=permutation_seed,
        _payload_json=_canonical_json(payload),
    )


def prepare_sph_transport_cohorts(spec: SPHTransportSpec) -> SPHTransportCohorts:
    """Build and verify the broad inference universe and both posthoc masks."""

    manifest = build_sph_transport_manifest(spec.metadata_path, spec.code_dictionary_path)
    source = _mapping(spec.payload["source"], "source")
    if len(manifest) != _integer(source.get("official_records"), "official_records"):
        raise SPHTransportIntegrityError("SPH metadata record count differs from protocol")
    if int(manifest["patient_id"].nunique()) != _integer(
        source.get("official_patients"), "official_patients"
    ):
        raise SPHTransportIntegrityError("SPH metadata patient count differs from protocol")
    if int(manifest["norm_abnormal_conflict"].astype(bool).sum()) != 0:
        raise SPHTransportIntegrityError("exact-token NORM/direct-abnormal conflicts are nonzero")

    broad = select_sph_exact_10s_records(manifest)
    primary = select_sph_transport_records(manifest)
    no_ambiguous = select_sph_transport_records(manifest, exclude_ambiguous=True)
    broad_ids = pd.Index(broad["ecg_id"])
    if broad_ids.has_duplicates:
        raise SPHTransportIntegrityError("broad SPH ECG identities are not unique")
    masks: dict[str, BoolArray] = {
        "primary_mapped": np.asarray(broad_ids.isin(primary["ecg_id"]), dtype=np.bool_),
        "broad_exact10": np.ones(len(broad), dtype=np.bool_),
        "no_ambiguous_mapped": np.asarray(broad_ids.isin(no_ambiguous["ecg_id"]), dtype=np.bool_),
    }
    targets = broad.loc[:, list(TARGET_COLUMNS)].to_numpy(dtype=np.int8)
    patients = broad["patient_id"].astype(str).to_numpy(dtype=np.str_)
    ecg_ids = broad["ecg_id"].astype(str).to_numpy(dtype=np.str_)
    summaries: dict[str, Mapping[str, object]] = {}
    cohort_protocol = _mapping(spec.payload["cohorts"], "cohorts")
    for name in COHORT_ORDER:
        mask = masks[name]
        expected_records, expected_patients = EXPECTED_COHORT_COUNTS[name]
        observed_records = int(mask.sum())
        observed_patients = int(np.unique(patients[mask]).size)
        positives = tuple(int(value) for value in targets[mask].sum(axis=0))
        positive_patients = tuple(
            int(np.unique(patients[mask][targets[mask, index] == 1]).size)
            for index in range(len(LABEL_ORDER))
        )
        if (observed_records, observed_patients) != (expected_records, expected_patients):
            raise SPHTransportIntegrityError(f"{name} record/patient count differs")
        if positives != EXPECTED_POSITIVE_RECORDS[name]:
            raise SPHTransportIntegrityError(f"{name} positive counts differ")
        if positive_patients != EXPECTED_POSITIVE_PATIENTS[name]:
            raise SPHTransportIntegrityError(f"{name} positive patient counts differ")
        frozen = _mapping(cohort_protocol.get(name), f"cohorts.{name}")
        frozen_positive_records = _mapping(
            frozen.get("positive_records"), f"cohorts.{name}.positive_records"
        )
        frozen_positive_patients = _mapping(
            frozen.get("positive_patients"), f"cohorts.{name}.positive_patients"
        )
        if (
            frozen.get("records") != observed_records
            or frozen.get("patients") != observed_patients
            or tuple(frozen_positive_records.get(label) for label in LABEL_ORDER) != positives
            or tuple(frozen_positive_patients.get(label) for label in LABEL_ORDER)
            != positive_patients
        ):
            raise SPHTransportIntegrityError(f"{name} frozen counts differ")
        summaries[name] = MappingProxyType(
            {
                "records": observed_records,
                "patients": observed_patients,
                "positive_records": dict(zip(LABEL_ORDER, positives, strict=True)),
                "positive_patients": dict(zip(LABEL_ORDER, positive_patients, strict=True)),
                "all_zero_rows": int(np.sum(targets[mask].sum(axis=1) == 0)),
            }
        )
    if not np.all(masks["no_ambiguous_mapped"] <= masks["primary_mapped"]):
        raise SPHTransportIntegrityError("no-ambiguous cohort is not a primary subset")

    record_names = {path.name for path in spec.records_dir.glob("*.h5") if path.is_file()}
    expected_names = set(manifest["record_path"].astype(str))
    if record_names != expected_names:
        raise SPHTransportIntegrityError("extracted HDF5 record set differs from metadata")
    alignment_payload = {
        "ecg_id": ecg_ids.tolist(),
        "patient_id": patients.tolist(),
        "targets": targets.tolist(),
        "primary_mapped": masks["primary_mapped"].tolist(),
        "no_ambiguous_mapped": masks["no_ambiguous_mapped"].tolist(),
    }
    return SPHTransportCohorts(
        manifest=broad.reset_index(drop=True),
        ecg_ids=_readonly(ecg_ids),
        patient_ids=_readonly(patients),
        targets=cast(IntArray, _readonly(targets)),
        masks=MappingProxyType(
            {name: cast(BoolArray, _readonly(mask)) for name, mask in masks.items()}
        ),
        alignment_sha256=canonical_sha256(alignment_payload),
        summaries=MappingProxyType(summaries),
    )


def load_sph_completed_runtime(spec: SPHTransportSpec) -> CompletedAuditRuntime:
    """Load the exact-six sealed runtime; this repeats its clean fold-10 gate."""

    protocol = load_protocol(spec.protocol_path)
    return load_completed_audit_runtime(
        protocol=protocol,
        final_evaluation_spec_path=spec.final_evaluation_spec_path,
        refit_bundle_path=spec.refit_bundle_path,
        calibration_bundle_path=spec.calibration_bundle_path,
        ledger_path=spec.opening_ledger_path,
    )


def infer_sph_transport(
    runtime: CompletedAuditRuntime,
    dataset: Dataset[tuple[Tensor, Tensor]],
    *,
    expected_targets: NDArray[np.generic] | None = None,
) -> SPHTransportInference:
    """Read each SPH record once and fan every normalized batch through all six models."""

    members = tuple(runtime.members)
    if tuple(member.member_id for member in members) != EXPECTED_AUDIT_MEMBER_IDS:
        raise SPHTransportIntegrityError("runtime is not the ordered exact-six release")
    if len(cast(Sized, dataset)) < 1:
        raise SPHTransportIntegrityError("SPH inference dataset is empty")
    _validate_runtime_compatibility(members)
    reference = members[0]
    batch_size = min(member.settings.batch_size for member in members)
    num_workers = min(member.settings.num_workers for member in members)
    generator = torch.Generator().manual_seed(EXPECTED_BOOTSTRAP_BASE_SEED)
    loader = DataLoader(
        _IndexedDataset(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=reference.settings.pin_memory and reference.runtime.device.type == "cuda",
        persistent_workers=reference.settings.persistent_workers and num_workers > 0,
        worker_init_fn=seed_dataloader_worker if num_workers > 0 else None,
        generator=generator,
        drop_last=False,
    )
    count = len(cast(Sized, dataset))
    raw_logits = {
        member.member_id: np.empty((count, len(LABEL_ORDER)), dtype=np.float64)
        for member in members
    }
    targets = np.empty((count, len(LABEL_ORDER)), dtype=np.int8)
    record_max = np.empty(count, dtype=np.float64)
    lead_record_max = np.empty((count, len(LEADS)), dtype=np.float64)
    lead_sum = np.zeros(len(LEADS), dtype=np.float64)
    lead_sum_squares = np.zeros(len(LEADS), dtype=np.float64)
    signal_hasher = hashlib.sha256()
    observed_positions: list[NDArray[np.int64]] = []
    for member in members:
        member.model.eval()

    with torch.inference_mode():
        for raw_positions, raw_signals, raw_targets in loader:
            positions = raw_positions.detach().cpu().numpy().astype(np.int64, copy=False)
            if positions.ndim != 1 or not np.array_equal(
                positions, np.arange(positions[0], positions[0] + positions.size)
            ):
                raise SPHTransportIntegrityError("SPH DataLoader positions are not contiguous")
            signals = raw_signals.detach().to(device="cpu", dtype=torch.float32).contiguous()
            if tuple(signals.shape[1:]) != (len(LEADS), SPH_TARGET_SAMPLES):
                raise SPHTransportIntegrityError("SPH batch shape differs from [B,12,1000]")
            if not torch.isfinite(signals).all().item():
                raise SPHTransportIntegrityError("SPH batch contains non-finite values")
            target_batch = raw_targets.detach().cpu().numpy()
            if (
                target_batch.shape != (positions.size, len(LABEL_ORDER))
                or not np.isin(target_batch, (0.0, 1.0)).all()
            ):
                raise SPHTransportIntegrityError("SPH targets are not aligned finite binary rows")

            # All QC is measured in physical mV before the unchanged PTB normalization.
            physical = signals.numpy()
            absolute = np.abs(physical.astype(np.float64, copy=False))
            record_max[positions] = absolute.max(axis=(1, 2))
            lead_record_max[positions] = absolute.max(axis=2)
            lead_sum += physical.sum(axis=(0, 2), dtype=np.float64)
            lead_sum_squares += np.square(physical, dtype=np.float64).sum(
                axis=(0, 2), dtype=np.float64
            )
            signal_hasher.update(np.ascontiguousarray(physical, dtype="<f4").tobytes())
            targets[positions] = target_batch.astype(np.int8, copy=False)
            observed_positions.append(positions.copy())

            normalized = reference.normalize_physical_batch(signals)
            device_signals = normalized.to(
                device=reference.runtime.device,
                dtype=torch.float32,
                non_blocking=reference.runtime.device.type == "cuda",
            )
            for member in members:
                with torch.autocast(
                    device_type=member.runtime.device.type,
                    dtype=torch.bfloat16,
                    enabled=member.runtime.bf16_enabled,
                ):
                    logits = member.model(device_signals)
                if not isinstance(logits, Tensor) or logits.shape != raw_targets.shape:
                    raise SPHTransportIntegrityError(
                        f"{member.member_id} returned an invalid logit shape"
                    )
                values = logits.detach().to(device="cpu", dtype=torch.float32).numpy()
                if not np.isfinite(values).all():
                    raise SPHTransportIntegrityError(
                        f"{member.member_id} returned non-finite logits"
                    )
                raw_logits[member.member_id][positions] = values.astype(np.float64, copy=False)
    order = np.concatenate(observed_positions) if observed_positions else np.empty(0, dtype=int)
    if not np.array_equal(order, np.arange(count, dtype=np.int64)):
        raise SPHTransportIntegrityError("SPH inference pass is incomplete or reordered")
    if expected_targets is not None and not np.array_equal(
        targets, np.asarray(expected_targets, dtype=np.int8)
    ):
        raise SPHTransportIntegrityError("inference targets differ from frozen cohort targets")
    samples_per_lead = count * SPH_TARGET_SAMPLES
    means = lead_sum / samples_per_lead
    variances = np.maximum(lead_sum_squares / samples_per_lead - np.square(means), 0.0)
    lead_qc: dict[str, Mapping[str, object]] = {}
    for index, lead in enumerate(LEADS):
        lead_qc[lead] = MappingProxyType(
            {
                "sample_mean_mv": float(means[index]),
                "sample_std_mv": float(math.sqrt(variances[index])),
                "per_record_max_abs_mv_quantiles": {
                    "q50": float(np.quantile(lead_record_max[:, index], 0.50)),
                    "q95": float(np.quantile(lead_record_max[:, index], 0.95)),
                    "q99": float(np.quantile(lead_record_max[:, index], 0.99)),
                    "maximum": float(lead_record_max[:, index].max()),
                },
            }
        )
    return SPHTransportInference(
        targets=cast(IntArray, _readonly(targets)),
        raw_logits=MappingProxyType(
            {name: cast(FloatArray, _readonly(values)) for name, values in raw_logits.items()}
        ),
        per_record_max_abs_mv=cast(FloatArray, _readonly(record_max)),
        per_record_lead_max_abs_mv=cast(FloatArray, _readonly(lead_record_max)),
        physical_signal_sha256="sha256:" + signal_hasher.hexdigest(),
        per_lead_qc=MappingProxyType(lead_qc),
    )


def assert_frozen_sph_runtime(spec: SPHTransportSpec, runtime: CompletedAuditRuntime) -> None:
    """Require the exact CUDA/BF16/batching contract frozen for this study."""

    frozen = _mapping(spec.payload.get("runtime"), "runtime")
    expected = {
        "inference_passes": 1,
        "device": "cuda:0",
        "allow_auto_device_resolution": False,
        "allow_cpu_fallback": False,
        "precision": "bf16",
        "bf16_required": True,
        "batch_size": 128,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "shuffle": False,
        "drop_last": False,
        "out_of_memory_policy": "fail_closed",
    }
    if any(frozen.get(key) != value for key, value in expected.items()):
        raise SPHTransportIntegrityError("frozen SPH runtime settings differ")
    members = tuple(runtime.members)
    if tuple(member.member_id for member in members) != EXPECTED_AUDIT_MEMBER_IDS:
        raise SPHTransportIntegrityError("runtime is not the ordered exact-six release")
    if min(member.settings.batch_size for member in members) != 128:
        raise SPHTransportIntegrityError("sealed member plans do not support batch 128")
    for member in members:
        if (
            str(member.runtime.device) != "cuda:0"
            or member.runtime.bf16_enabled is not True
            or member.settings.num_workers != 4
            or member.settings.pin_memory is not True
            or member.settings.persistent_workers is not True
        ):
            raise SPHTransportIntegrityError(
                f"{member.member_id} runtime differs from frozen CUDA/BF16 contract"
            )


def apply_frozen_member_decisions(
    member: AuditMemberRuntime, raw_logits: NDArray[np.generic]
) -> SPHFrozenMemberOutput:
    """Apply only the member's sealed fold-9 temperature, thresholds, and gates."""

    logits = np.asarray(raw_logits, dtype=np.float64)
    decisions = member.decisions
    if decisions.label_order != LABEL_ORDER:
        raise SPHTransportIntegrityError("member decision label order differs")
    raw_probabilities = stable_sigmoid(logits)
    calibrated = decisions.temperature_scaling.predict_proba(logits, label_order=LABEL_ORDER)
    predictions = decisions.threshold_optimization.apply(calibrated, label_order=LABEL_ORDER)
    uncertainty = mean_normalized_binary_entropy(calibrated)
    gate_selected = np.column_stack(
        [uncertainty <= gate.maximum_entropy for gate in decisions.coverage_gates]
    ).astype(np.bool_, copy=False)
    return SPHFrozenMemberOutput(
        member_id=member.member_id,
        architecture=member.architecture,
        seed=member.seed,
        raw_logits=cast(FloatArray, _readonly(logits)),
        raw_probabilities=cast(FloatArray, _readonly(raw_probabilities)),
        calibrated_probabilities=cast(FloatArray, _readonly(calibrated)),
        predictions=cast(BoolArray, _readonly(predictions)),
        uncertainty=cast(FloatArray, _readonly(uncertainty)),
        gate_selected=cast(BoolArray, _readonly(gate_selected)),
        temperature=decisions.temperature_scaling.temperature,
        thresholds=decisions.threshold_optimization.thresholds,
        entropy_gate_cutoffs=tuple(gate.maximum_entropy for gate in decisions.coverage_gates),
        entropy_gate_targets=tuple(gate.target_coverage for gate in decisions.coverage_gates),
    )


def evaluate_all_sph_cohorts(
    spec: SPHTransportSpec,
    runtime: CompletedAuditRuntime,
    cohorts: SPHTransportCohorts,
    inference: SPHTransportInference,
) -> SPHTransportEvaluation:
    """Evaluate all six members on all three frozen cohort masks without fitting."""

    outputs = {
        member.member_id: apply_frozen_member_decisions(
            member, inference.raw_logits[member.member_id]
        )
        for member in runtime.members
    }
    reports: dict[str, Mapping[str, object]] = {}
    paired: dict[str, Mapping[str, object]] = {}
    for cohort_name in COHORT_ORDER:
        mask = cohorts.masks[cohort_name]
        seed_offset = EXPECTED_COHORT_SEED_OFFSETS[cohort_name]
        for member in runtime.members:
            output = outputs[member.member_id]
            key = f"{cohort_name}/{member.member_id}"
            reports[key] = MappingProxyType(
                evaluate_sph_transport(
                    cohorts.targets[mask],
                    output.raw_logits[mask],
                    cohorts.patient_ids[mask],
                    temperature=output.temperature,
                    thresholds=output.thresholds,
                    entropy_gates=[
                        {
                            "target_coverage": target,
                            "maximum_entropy": cutoff,
                        }
                        for target, cutoff in zip(
                            output.entropy_gate_targets,
                            output.entropy_gate_cutoffs,
                            strict=True,
                        )
                    ],
                    n_resamples=EXPECTED_BOOTSTRAP_RESAMPLES,
                    seed=EXPECTED_BOOTSTRAP_BASE_SEED + seed_offset + member.seed,
                    confidence_level=EXPECTED_BOOTSTRAP_CONFIDENCE,
                    minimum_valid_resamples=EXPECTED_BOOTSTRAP_MINIMUM_VALID,
                    label_order=LABEL_ORDER,
                    ece_bins=EXPECTED_ECE_BINS,
                )
            )
        for model_seed in (2026, 2027, 2028):
            resnet = outputs[f"resnet1d-seed{model_seed}"]
            transformer = outputs[f"ecg_transformer-seed{model_seed}"]
            key = f"{cohort_name}/seed{model_seed}"
            paired[key] = MappingProxyType(
                paired_sph_transport_differences(
                    cohorts.targets[mask],
                    resnet.raw_logits[mask],
                    transformer.raw_logits[mask],
                    cohorts.patient_ids[mask],
                    resnet_temperature=resnet.temperature,
                    transformer_temperature=transformer.temperature,
                    n_resamples=EXPECTED_BOOTSTRAP_RESAMPLES,
                    seed=EXPECTED_BOOTSTRAP_BASE_SEED + seed_offset + model_seed,
                    confidence_level=EXPECTED_BOOTSTRAP_CONFIDENCE,
                    minimum_valid_resamples=EXPECTED_BOOTSTRAP_MINIMUM_VALID,
                    label_order=LABEL_ORDER,
                    ece_bins=EXPECTED_ECE_BINS,
                )
            )
    summaries = _architecture_summaries(reports)
    return SPHTransportEvaluation(
        member_outputs=MappingProxyType(outputs),
        member_reports=MappingProxyType(reports),
        paired_reports=MappingProxyType(paired),
        architecture_summaries=MappingProxyType(summaries),
    )


def reserve_sph_transport_attempt(
    spec: SPHTransportSpec,
    runtime: CompletedAuditRuntime,
    *,
    source_inventory: Mapping[str, object],
    execution_state: Mapping[str, object],
) -> SPHTransportAttempt:
    """Reserve the official root before inference so failed attempts remain visible."""

    root = spec.output_root
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"SPH output root is populated; refusing attempt: {root}")
    root.mkdir(parents=True, exist_ok=True)
    private_root = root / "private"
    if private_root.exists():
        raise FileExistsError("empty output root unexpectedly contains private state")
    private_root.mkdir()
    marker_path, marker_sha = write_new_hashed_json(
        private_root / "attempt-start.json",
        {
            "schema_version": 1,
            "artifact_type": "ecg_trust.sph_transport_attempt_start",
            "protocol_sha256": spec.file_sha256,
            "member_ids": [member.member_id for member in runtime.members],
            "clean_equivalence": [item.to_dict() for item in runtime.clean_equivalence],
            "source_inventory_sha256": canonical_sha256(source_inventory),
            "execution_state": dict(execution_state),
            "state": "reserved_before_first_sph_model_inference",
        },
        hash_field="attempt_sha256",
    )
    return SPHTransportAttempt(
        output_root=root,
        marker_path=marker_path,
        marker_sha256=marker_sha,
    )


def _verify_reserved_attempt(spec: SPHTransportSpec, attempt: SPHTransportAttempt) -> None:
    root = spec.output_root.resolve()
    marker = (root / "private" / "attempt-start.json").resolve()
    if attempt.output_root.resolve() != root or attempt.marker_path.resolve() != marker:
        raise SPHTransportIntegrityError("attempt reservation belongs to another output root")
    if not marker.is_file() or {path.name for path in root.iterdir()} != {"private"}:
        raise SPHTransportIntegrityError("reserved output root contains unexpected state")
    private_files = {path.name for path in marker.parent.iterdir()}
    if private_files != {marker.name}:
        raise SPHTransportIntegrityError("reserved private root contains unexpected state")
    try:
        decoded: object = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SPHTransportIntegrityError(f"could not verify attempt marker: {error}") from error
    payload = _mapping(decoded, "attempt marker")
    stored = _prefixed_hash(payload.get("attempt_sha256"), "attempt_sha256")
    unhashed = dict(payload)
    del unhashed["attempt_sha256"]
    if (
        stored != attempt.marker_sha256
        or canonical_sha256(unhashed) != stored
        or payload.get("protocol_sha256") != spec.file_sha256
        or payload.get("state") != "reserved_before_first_sph_model_inference"
    ):
        raise SPHTransportIntegrityError("attempt marker integrity differs")


def record_sph_attempt_failure(attempt: SPHTransportAttempt, error: BaseException) -> Path:
    """Commit a non-overwriting private failure receipt and preserve partial output."""

    marker = attempt.marker_path
    if not marker.is_file():
        raise SPHTransportIntegrityError("cannot record failure without attempt marker")
    path, _ = write_new_hashed_json(
        marker.parent / "attempt-failure.json",
        {
            "schema_version": 1,
            "artifact_type": "ecg_trust.sph_transport_attempt_failure",
            "attempt_sha256": attempt.marker_sha256,
            "exception_type": f"{type(error).__module__}.{type(error).__qualname__}",
            "exception_message": str(error),
            "state": "failed_after_attempt_reservation",
            "retry_policy": "new_frozen_protocol_and_new_output_root",
        },
        hash_field="failure_sha256",
    )
    return path


def save_sph_transport_outputs(
    spec: SPHTransportSpec,
    runtime: CompletedAuditRuntime,
    cohorts: SPHTransportCohorts,
    inference: SPHTransportInference,
    evaluation: SPHTransportEvaluation,
    *,
    attempt: SPHTransportAttempt,
    source_inventory: Mapping[str, object] | None = None,
    execution_state: Mapping[str, object] | None = None,
) -> SPHTransportRun:
    """Commit private alignment and identifier-free public outputs without overwrite."""

    root = spec.output_root
    _verify_reserved_attempt(spec, attempt)
    private_root = root / "private"
    public_root = root / "public"
    if public_root.exists():
        raise FileExistsError("reserved SPH attempt already contains public output")
    public_root.mkdir()
    (public_root / "member_predictions").mkdir()
    (public_root / "member_reports").mkdir()
    (public_root / "paired_bootstrap_reports").mkdir()
    (public_root / "architecture_summaries").mkdir()
    shutil.copyfile(spec.path, private_root / "protocol.snapshot.yaml")

    inventory_path, _ = write_new_hashed_json(
        private_root / "source_inventory.json",
        {
            "schema_version": 1,
            "artifact_type": "ecg_trust.sph_transport_source_inventory",
            "protocol_sha256": spec.file_sha256,
            "inventory": dict(source_inventory or {"status": "synthetic_test"}),
        },
        hash_field="artifact_sha256",
    )

    publication_order = np.random.default_rng(spec.public_row_permutation_seed).permutation(
        len(cohorts.ecg_ids)
    )
    internal_to_public = np.empty(len(publication_order), dtype=np.int64)
    internal_to_public[publication_order] = np.arange(len(publication_order), dtype=np.int64)
    private_manifest = cohorts.manifest.copy()
    private_manifest["public_position"] = internal_to_public
    for name, mask in cohorts.masks.items():
        private_manifest[f"cohort_{name}"] = mask
    for column in ("primary_codes", "modifier_codes", "ambiguous_primary_codes"):
        if column in private_manifest:
            private_manifest[column] = private_manifest[column].map(
                lambda value: json.dumps(list(value), separators=(",", ":"))
            )
    parquet_path = private_root / "cohort_manifest.parquet"
    if parquet_path.exists():
        raise FileExistsError(f"private cohort manifest already exists: {parquet_path}")
    private_manifest.to_parquet(parquet_path, index=False)
    private_alignment_sha = _prefixed_file_sha256(parquet_path)
    save_audit_array_artifact(
        private_root / "cohort_alignment.npz",
        artifact_type=SPH_TRANSPORT_ALIGNMENT_TYPE,
        arrays={
            "ecg_id": cohorts.ecg_ids,
            "patient_id": cohorts.patient_ids,
            "public_position": internal_to_public,
            "targets": cohorts.targets,
            "primary_mapped": cohorts.masks["primary_mapped"],
            "no_ambiguous_mapped": cohorts.masks["no_ambiguous_mapped"],
        },
        metadata={
            "protocol_sha256": spec.file_sha256,
            "alignment_sha256": cohorts.alignment_sha256,
            "cohort_manifest_file_sha256": private_alignment_sha,
            "n_samples": len(cohorts.ecg_ids),
            "visibility": "local_private_gitignored",
        },
    )

    safe_masks = {
        name: np.asarray(mask[publication_order], dtype=np.bool_)
        for name, mask in cohorts.masks.items()
    }
    public_entries: list[dict[str, object]] = []
    qc_files = save_audit_array_artifact(
        public_root / "signal_qc.npz",
        artifact_type=SPH_TRANSPORT_QC_TYPE,
        arrays={
            "per_record_max_abs_mv": inference.per_record_max_abs_mv[publication_order],
            "per_record_lead_max_abs_mv": inference.per_record_lead_max_abs_mv[publication_order],
            **safe_masks,
        },
        metadata={
            "protocol_sha256": spec.file_sha256,
            "private_alignment_sha256": private_alignment_sha,
            "physical_signal_sha256": inference.physical_signal_sha256,
            "lead_order": list(LEADS),
            "per_lead_qc": {lead: dict(values) for lead, values in inference.per_lead_qc.items()},
            "outlier_policy": "retain_without_clipping_rejection_or_rescaling",
        },
    )
    public_entries.extend(_array_file_entries(qc_files, public_root))

    for member in runtime.members:
        output = evaluation.member_outputs[member.member_id]
        files = save_audit_array_artifact(
            public_root / "member_predictions" / f"{member.member_id}.npz",
            artifact_type=SPH_TRANSPORT_PREDICTION_TYPE,
            arrays={
                "targets": cohorts.targets[publication_order],
                "raw_logits": output.raw_logits[publication_order],
                "raw_probabilities": output.raw_probabilities[publication_order],
                "calibrated_probabilities": output.calibrated_probabilities[publication_order],
                "predictions": output.predictions[publication_order],
                "uncertainty": output.uncertainty[publication_order],
                "gate_selected": output.gate_selected[publication_order],
                **safe_masks,
            },
            metadata={
                "protocol_sha256": spec.file_sha256,
                "private_alignment_sha256": private_alignment_sha,
                "physical_signal_sha256": inference.physical_signal_sha256,
                "member_id": member.member_id,
                "architecture": member.architecture,
                "seed": member.seed,
                "checkpoint_sha256": member.checkpoint_sha256,
                "decision_sha256": member.decisions.integrity_sha256,
                "normalization_sha256": runtime.refit_bundle.normalization_sha256,
                "label_order": list(LABEL_ORDER),
                "temperature": output.temperature,
                "thresholds": list(output.thresholds),
                "entropy_gate_cutoffs": list(output.entropy_gate_cutoffs),
                "entropy_gate_targets": list(output.entropy_gate_targets),
                "sph_fitting_performed": False,
                "privacy_description": (
                    "identifier-omitted local publication candidate; deterministic "
                    "row ordering is not a confidentiality mechanism"
                ),
            },
        )
        public_entries.extend(_array_file_entries(files, public_root))

    for key, report in evaluation.member_reports.items():
        cohort_name, member_id = key.split("/", maxsplit=1)
        path, _ = write_new_hashed_json(
            public_root / "member_reports" / f"{cohort_name}__{member_id}.json",
            {
                "schema_version": 1,
                "artifact_type": "ecg_trust.sph_transport_member_report",
                "protocol_sha256": spec.file_sha256,
                "private_alignment_sha256": private_alignment_sha,
                "cohort": cohort_name,
                "member_id": member_id,
                "report": dict(report),
                "interpretation": "exploratory external transport stress test",
                "sph_fitting_performed": False,
                "clinical_validation": False,
            },
            hash_field="artifact_sha256",
        )
        public_entries.append(_file_entry(path, public_root))
    for key, report in evaluation.paired_reports.items():
        cohort_name, seed_name = key.split("/", maxsplit=1)
        path, _ = write_new_hashed_json(
            public_root / "paired_bootstrap_reports" / f"{cohort_name}__{seed_name}.json",
            {
                "schema_version": 1,
                "artifact_type": "ecg_trust.sph_transport_paired_report",
                "protocol_sha256": spec.file_sha256,
                "private_alignment_sha256": private_alignment_sha,
                "cohort": cohort_name,
                "seed_pair": seed_name,
                "direction": "ecg_transformer_minus_resnet1d",
                "report": dict(report),
            },
            hash_field="artifact_sha256",
        )
        public_entries.append(_file_entry(path, public_root))
    for key, summary in evaluation.architecture_summaries.items():
        path, _ = write_new_hashed_json(
            public_root / "architecture_summaries" / f"{key}.json",
            {
                "schema_version": 1,
                "artifact_type": "ecg_trust.sph_transport_architecture_summary",
                "protocol_sha256": spec.file_sha256,
                "private_alignment_sha256": private_alignment_sha,
                "summary": dict(summary),
            },
            hash_field="artifact_sha256",
        )
        public_entries.append(_file_entry(path, public_root))

    summary_path, _ = write_new_hashed_json(
        public_root / "cohort_summary.json",
        {
            "schema_version": 1,
            "artifact_type": "ecg_trust.sph_transport_cohort_summary",
            "protocol_sha256": spec.file_sha256,
            "private_alignment_sha256": private_alignment_sha,
            "cohorts": {name: dict(value) for name, value in cohorts.summaries.items()},
            "broad_all_zero_caveat": ("absent direct mapping is unknown, not a verified negative"),
            "norm_plus_direct_abnormal_conflicts": 0,
        },
        hash_field="artifact_sha256",
    )
    public_entries.append(_file_entry(summary_path, public_root))
    results_path = public_root / "FINAL_RESULTS.md"
    _write_new_text(
        results_path,
        _final_results_markdown(spec, cohorts, evaluation),
    )
    public_entries.append(_file_entry(results_path, public_root))
    public_entries = sorted(public_entries, key=lambda item: cast(str, item["file"]))
    manifest_path, manifest_sha = write_new_hashed_json(
        public_root / "manifest.json",
        {
            "schema_version": 1,
            "artifact_type": "ecg_trust.sph_transport_public_manifest",
            "protocol_sha256": spec.file_sha256,
            "private_alignment_sha256": private_alignment_sha,
            "physical_signal_sha256": inference.physical_signal_sha256,
            "identifier_fields_present": False,
            "automatic_publication": False,
            "privacy_description": (
                "identifier-omitted local publication candidate; deterministic row "
                "ordering is not a confidentiality mechanism"
            ),
            "files": public_entries,
        },
        hash_field="artifact_sha256",
    )
    assert_identifier_free_public_outputs(public_root)

    state = dict(execution_state or {"status": "synthetic_test"})
    _write_new_text(
        private_root / "RUN_LOG.md",
        (
            "# SPH external-transport run log\n\n"
            "This is an exploratory external transport stress test. No tuning or "
            "recalibration on SPH was performed. It is not clinical validation and is "
            "research only.\n\n"
            f"- Protocol SHA-256: `{spec.file_sha256}`\n"
            f"- Physical signal SHA-256: `{inference.physical_signal_sha256}`\n"
            f"- Private alignment SHA-256: `{private_alignment_sha}`\n"
            f"- Execution state: `{json.dumps(state, sort_keys=True)}`\n"
        ),
    )
    generated_before_manifest = sorted(path for path in root.rglob("*") if path.is_file())
    bound_inputs = _bound_input_entries(spec, require_all=source_inventory is not None)
    write_new_hashed_json(
        private_root / "derived_artifacts.manifest.json",
        {
            "schema_version": 1,
            "artifact_type": "ecg_trust.sph_transport_derived_manifest",
            "protocol_sha256": spec.file_sha256,
            "generated_files": [_file_entry(path, root) for path in generated_before_manifest],
            "bound_inputs": bound_inputs,
            "public_manifest_sha256": manifest_sha,
            "private_alignment_sha256": private_alignment_sha,
            "source_inventory_file_sha256": _prefixed_file_sha256(inventory_path),
        },
        hash_field="artifact_sha256",
    )
    return SPHTransportRun(
        output_root=root,
        public_manifest_path=manifest_path,
        public_manifest_sha256=manifest_sha,
        private_alignment_sha256=private_alignment_sha,
    )


def run_sph_transport(
    spec: SPHTransportSpec,
    *,
    progress: Progress | None = None,
) -> SPHTransportRun:
    """Execute the preregistered study without scientific command-line overrides."""

    emit = progress or (lambda _message: None)
    if spec.output_root.exists() and any(spec.output_root.iterdir()):
        raise FileExistsError(
            f"SPH output root is populated; use a new frozen protocol: {spec.output_root}"
        )
    emit("verifying clean Git execution revision")
    execution_state = verify_clean_git_execution(spec)
    emit("validating SPH metadata and frozen cohort masks")
    cohorts = prepare_sph_transport_cohorts(spec)
    emit("auditing every source archive member before inference")
    archive_audit = verify_sph_archive_safety(spec, cohorts)
    source_inventory = _source_inventory(spec, cohorts, archive_audit)
    emit("loading and clean-gating the sealed exact-six PTB runtime")
    runtime = load_sph_completed_runtime(spec)
    assert_frozen_sph_runtime(spec, runtime)
    dataset = SPHExternalTransportDataset(cohorts.manifest, spec.records_dir, allow_all_zero=True)
    emit("reserving immutable official-attempt output root")
    attempt = reserve_sph_transport_attempt(
        spec,
        runtime,
        source_inventory=source_inventory,
        execution_state=execution_state,
    )
    try:
        emit("running one broad exact-10-second SPH pass through all six frozen members")
        inference = infer_sph_transport(runtime, dataset, expected_targets=cohorts.targets)
        emit("evaluating primary and two sensitivity cohorts with frozen decisions")
        evaluation = evaluate_all_sph_cohorts(spec, runtime, cohorts, inference)
        emit("committing private alignment and identifier-free public artifacts")
        return save_sph_transport_outputs(
            spec,
            runtime,
            cohorts,
            inference,
            evaluation,
            attempt=attempt,
            source_inventory=source_inventory,
            execution_state=execution_state,
        )
    except BaseException as error:
        record_sph_attempt_failure(attempt, error)
        raise


def assert_identifier_free_public_outputs(public_root: str | Path) -> None:
    """Reject source record/patient identities, paths, or codes in public artifacts."""

    root = Path(public_root)
    forbidden_keys = {
        "ecg_id",
        "patient_id",
        "record_path",
        "aha_code",
        "raw_aha_codes",
        "primary_codes",
        "modifier_codes",
        "ambiguous_primary_codes",
        "age",
        "sex",
        "acquisition_date",
    }
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.casefold() == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                if forbidden_keys.intersection(archive.files):
                    raise SPHTransportIntegrityError("public NPZ contains private fields")
                if any(np.asarray(archive[name]).dtype.kind in "SUO" for name in archive.files):
                    raise SPHTransportIntegrityError("public NPZ contains string identities")
        elif path.suffix.casefold() == ".json":
            decoded: object = json.loads(path.read_text(encoding="utf-8"))
            _assert_public_json(decoded, forbidden_keys)
        elif path.suffix.casefold() in {".md", ".txt", ".csv", ".tsv"}:
            _assert_public_text(path.read_text(encoding="utf-8"), forbidden_keys)


def verify_clean_git_execution(spec: SPHTransportSpec) -> Mapping[str, object]:
    """Fail before inference unless the evaluator runs from a clean Git revision."""

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=spec.project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=spec.project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise SPHTransportIntegrityError(f"could not verify clean Git state: {error}") from error
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise SPHTransportIntegrityError("execution Git revision is invalid")
    if status.strip():
        raise SPHTransportIntegrityError("official SPH inference requires a clean Git worktree")
    parent = spec.payload.get("scientific_parent_git_revision")
    if not isinstance(parent, str) or len(parent) != 40:
        raise SPHTransportIntegrityError("scientific parent Git revision is invalid")
    return MappingProxyType(
        {
            "git_revision": revision,
            "git_worktree_clean": True,
            "scientific_parent_git_revision": parent,
        }
    )


def verify_sph_archive_safety(
    spec: SPHTransportSpec, cohorts: SPHTransportCohorts
) -> Mapping[str, object]:
    """Audit every tar member and bind the archive record set before inference."""

    # The broad manifest excludes non-10-second rows, so use the official metadata
    # record set for the archive identity audit.
    full_manifest = build_sph_transport_manifest(spec.metadata_path, spec.code_dictionary_path)
    expected_all = set(full_manifest["record_path"].astype(str))
    observed: dict[str, str] = {}
    content_tree = hashlib.sha256()
    member_count = 0
    try:
        # The official Figshare object is named ``records.tar.gz`` but v1 is a
        # plain tar stream.  Auto-detect the container instead of trusting its suffix.
        with tarfile.open(spec.records_archive_path, mode="r:*") as archive:
            for member in archive:
                member_count += 1
                raw_name = member.name
                if "\\" in raw_name:
                    raise SPHTransportIntegrityError("SPH archive contains a non-POSIX member path")
                posix = PurePosixPath(raw_name)
                windows = PureWindowsPath(raw_name)
                if (
                    posix.is_absolute()
                    or windows.is_absolute()
                    or windows.drive
                    or any(part in {"", ".", ".."} for part in posix.parts)
                    or member.issym()
                    or member.islnk()
                    or member.isdev()
                ):
                    raise SPHTransportIntegrityError("SPH archive contains an unsafe member")
                if member.isfile() and posix.suffix.casefold() == ".h5":
                    if posix.name in observed:
                        raise SPHTransportIntegrityError(
                            "SPH archive contains a duplicate HDF5 basename"
                        )
                    archived_handle = archive.extractfile(member)
                    if archived_handle is None:
                        raise SPHTransportIntegrityError("SPH archive HDF5 member cannot be read")
                    archived_digest = _sha256_handle(archived_handle)
                    extracted_path = spec.records_dir / posix.name
                    if (
                        not extracted_path.is_file()
                        or extracted_path.stat().st_size != member.size
                        or sha256_file(extracted_path) != archived_digest
                    ):
                        raise SPHTransportIntegrityError(
                            f"extracted SPH record differs from archive member {posix.name}"
                        )
                    observed[posix.name] = archived_digest
                    content_tree.update(f"{posix.name}:{archived_digest}\n".encode("ascii"))
    except SPHTransportIntegrityError:
        raise
    except (OSError, tarfile.TarError) as error:
        raise SPHTransportIntegrityError(f"could not audit SPH archive: {error}") from error
    if set(observed) != expected_all:
        raise SPHTransportIntegrityError("SPH archive record set differs from metadata")
    return MappingProxyType(
        {
            "archive_sha256": _prefixed_file_sha256(spec.records_archive_path),
            "tar_members_checked": member_count,
            "hdf5_files": len(observed),
            "extracted_content_tree_sha256": "sha256:" + content_tree.hexdigest(),
            "archive_member_content_match": True,
            "safe_paths": True,
            "metadata_record_set_match": True,
        }
    )


def _sha256_handle(handle: Any) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _source_inventory(
    spec: SPHTransportSpec,
    cohorts: SPHTransportCohorts,
    archive_audit: Mapping[str, object],
) -> Mapping[str, object]:
    return MappingProxyType(
        {
            "source_files": {
                "metadata": _bound_file_identity(spec.metadata_path),
                "diagnostic_dictionary": _bound_file_identity(spec.code_dictionary_path),
                "translation_rules": _bound_file_identity(spec.rule_path),
                "records_archive": _bound_file_identity(spec.records_archive_path),
            },
            "archive_audit": dict(archive_audit),
            "broad_alignment_sha256": cohorts.alignment_sha256,
            "cohorts": {name: dict(summary) for name, summary in cohorts.summaries.items()},
        }
    )


def _bound_file_identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _prefixed_file_sha256(path),
    }


def _bound_input_entries(spec: SPHTransportSpec, *, require_all: bool) -> list[dict[str, object]]:
    paths = (
        spec.path,
        spec.metadata_path,
        spec.code_dictionary_path,
        spec.rule_path,
        spec.records_archive_path,
        spec.normalization_path,
        spec.protocol_path,
        spec.final_evaluation_spec_path,
        spec.opening_ledger_path,
        spec.refit_bundle_path,
        spec.calibration_bundle_path,
    )
    if require_all and any(not path.is_file() for path in paths):
        raise SPHTransportIntegrityError("a bound input disappeared before manifesting")
    return [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _prefixed_file_sha256(path),
        }
        for path in paths
        if path.is_file()
    ]


def _write_new_text(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"immutable text artifact already exists: {path}")
    path.write_text(content, encoding="utf-8", newline="\n")


def _final_results_markdown(
    spec: SPHTransportSpec,
    cohorts: SPHTransportCohorts,
    evaluation: SPHTransportEvaluation,
) -> str:
    lines = [
        "# SPH exploratory external transport stress test",
        "",
        "No tuning or recalibration on SPH was performed. This is not clinical "
        "validation; all outputs are research only.",
        "",
        f"Frozen protocol: `{spec.file_sha256}`",
        "",
        "## Cohorts",
        "",
        "| Cohort | Records | Patients |",
        "|---|---:|---:|",
    ]
    for name in COHORT_ORDER:
        summary = cohorts.summaries[name]
        lines.append(f"| {name} | {summary['records']} | {summary['patients']} |")
    lines.extend(
        [
            "",
            "## Primary calibrated architecture results",
            "",
            "Mean +/- sample SD across the three frozen seeds.",
            "",
            "| Architecture | Macro AUROC | Macro AP | Macro Brier | Macro ECE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for architecture in ("resnet1d", "ecg_transformer"):
        values = [
            _architecture_metric_values(
                evaluation,
                cohort="primary_mapped",
                architecture=architecture,
                metric=metric,
            )
            for metric in ("roc_auc", "average_precision", "brier_score", "ece")
        ]
        lines.append(
            f"| {architecture} | " + " | ".join(_format_mean_sd(item) for item in values) + " |"
        )
    lines.extend(
        [
            "",
            "## Primary calibrated per-class results",
            "",
            "| Architecture | Label | AUROC | AP | Brier | ECE |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for architecture in ("resnet1d", "ecg_transformer"):
        for label in LABEL_ORDER:
            values = [
                _architecture_metric_values(
                    evaluation,
                    cohort="primary_mapped",
                    architecture=architecture,
                    metric=metric,
                    label=label,
                )
                for metric in ("roc_auc", "average_precision", "brier_score", "ece")
            ]
            lines.append(
                f"| {architecture} | {label} | "
                + " | ".join(_format_mean_sd(item) for item in values)
                + " |"
            )
    lines.extend(
        [
            "",
            "## Frozen entropy gates on the primary cohort",
            "",
            "Each target is the nominal PTB fold-9 coverage; observed SPH coverage can differ.",
            "",
            "| Target | Architecture | Observed coverage | Hamming risk | Exact-match accuracy |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for target_coverage in EXPECTED_GATE_TARGETS:
        for architecture in ("resnet1d", "ecg_transformer"):
            gate_values = [
                _architecture_gate_values(
                    evaluation,
                    architecture=architecture,
                    target_coverage=target_coverage,
                    field=field,
                )
                for field in (
                    "observed_coverage",
                    "hamming_risk",
                    "exact_match_accuracy",
                )
            ]
            lines.append(
                f"| {target_coverage:.1f} | {architecture} | "
                + " | ".join(_format_mean_sd(item) for item in gate_values)
                + " |"
            )
    lines.extend(
        [
            "",
            "## Paired Transformer-minus-ResNet primary differences",
            "",
            "Calibrated point estimates and 95% paired patient-cluster bootstrap CIs.",
            "",
            "| Seed | Metric | Estimate | 95% CI |",
            "|---:|---|---:|---:|",
        ]
    )
    for seed in (2026, 2027, 2028):
        paired = _mapping(evaluation.paired_reports[f"primary_mapped/seed{seed}"], "paired report")
        views = _mapping(paired.get("probability_views"), "paired probability_views")
        calibrated = _mapping(views.get("frozen_temperature_calibrated"), "paired calibrated")
        macro = _mapping(calibrated.get("macro"), "paired calibrated macro")
        for metric in ("roc_auc", "average_precision", "brier_score", "ece"):
            interval = _mapping(macro.get(metric), f"paired {metric}")
            lines.append(
                f"| {seed} | {metric} | {_format_optional(interval.get('estimate'))} | "
                f"[{_format_optional(interval.get('lower'))}, "
                f"{_format_optional(interval.get('upper'))}] |"
            )
    lines.extend(
        [
            "",
            "## Sensitivity summaries",
            "",
            "Calibrated macro mean +/- sample SD across frozen seeds.",
            "",
            "| Cohort | Architecture | AUROC | AP | Brier | ECE |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for cohort in ("broad_exact10", "no_ambiguous_mapped"):
        for architecture in ("resnet1d", "ecg_transformer"):
            values = [
                _architecture_metric_values(
                    evaluation,
                    cohort=cohort,
                    architecture=architecture,
                    metric=metric,
                )
                for metric in ("roc_auc", "average_precision", "brier_score", "ece")
            ]
            lines.append(
                f"| {cohort} | {architecture} | "
                + " | ".join(_format_mean_sd(item) for item in values)
                + " |"
            )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "The broad all-zero sensitivity treats diagnoses without a direct mapping "
            "as operationally all-zero; those absences are unknown, not verified negatives.",
            "",
            "SPH labels are a conservative cross-ontology mapping without expert "
            "adjudication. Rare MI and HYP labels can yield unstable or undefined "
            "bootstrap replicates. Frozen PTB coverage cutoffs need not achieve their "
            "nominal coverage after transport.",
            "",
            "Finite physical-amplitude extremes were retained without clipping, "
            "rejection, rescaling, model selection, or outcome-informed cleaning.",
            "",
        ]
    )
    return "\n".join(lines)


def _architecture_metric_values(
    evaluation: SPHTransportEvaluation,
    *,
    cohort: str,
    architecture: str,
    metric: str,
    label: str | None = None,
) -> list[float]:
    values: list[float] = []
    for seed in (2026, 2027, 2028):
        report = _mapping(
            evaluation.member_reports[f"{cohort}/{architecture}-seed{seed}"],
            "member report",
        )
        views = _mapping(report.get("probability_views"), "probability_views")
        calibrated = _mapping(views.get("frozen_temperature_calibrated"), "calibrated view")
        metrics = _mapping(calibrated.get("metrics"), "calibrated metrics")
        if label is None:
            selected = _mapping(metrics.get("macro"), "calibrated macro")
        else:
            per_label = metrics.get("per_label")
            if not isinstance(per_label, list):
                raise SPHTransportIntegrityError("per-label metrics must be a list")
            candidates = [
                _mapping(item, "per-label metric")
                for item in per_label
                if isinstance(item, Mapping) and item.get("label") == label
            ]
            if len(candidates) != 1:
                raise SPHTransportIntegrityError("per-label metric is missing or duplicated")
            selected = candidates[0]
        value = _optional_number(selected.get(metric))
        if value is not None:
            values.append(value)
    return values


def _architecture_gate_values(
    evaluation: SPHTransportEvaluation,
    *,
    architecture: str,
    target_coverage: float,
    field: str,
) -> list[float]:
    values: list[float] = []
    for seed in (2026, 2027, 2028):
        report = _mapping(
            evaluation.member_reports[f"primary_mapped/{architecture}-seed{seed}"],
            "member report",
        )
        gates = report.get("frozen_entropy_gates")
        if not isinstance(gates, list):
            raise SPHTransportIntegrityError("frozen entropy gates must be a list")
        selected = [
            _mapping(gate, "entropy gate")
            for gate in gates
            if isinstance(gate, Mapping) and gate.get("target_coverage") == target_coverage
        ]
        if len(selected) != 1:
            raise SPHTransportIntegrityError(
                f"nominal-{target_coverage:.1f} frozen gate is missing or duplicated"
            )
        value = _optional_number(selected[0].get(field))
        if value is not None:
            values.append(value)
    return values


def _format_mean_sd(values: Sequence[float]) -> str:
    if not values:
        return "NA"
    mean = float(np.mean(values))
    if len(values) < 2:
        return f"{mean:.6f} +/- NA"
    return f"{mean:.6f} +/- {float(np.std(values, ddof=1)):.6f}"


def _format_optional(value: object) -> str:
    number = _optional_number(value)
    return "NA" if number is None else f"{number:.6f}"


def _validate_signal_and_label_contract(payload: Mapping[str, object]) -> None:
    purpose = _mapping(payload.get("purpose"), "purpose")
    source = _mapping(payload.get("source"), "source")
    signal = _mapping(payload.get("signal_contract"), "signal_contract")
    resampling = _mapping(signal.get("resampling"), "resampling")
    expected_signal = {
        "source_sampling": source.get("sampling_rate_hz"),
        "source_leads": source.get("lead_order"),
        "key": signal.get("hdf5_dataset_key"),
        "shape": signal.get("source_shape"),
        "unit": signal.get("source_unit"),
        "target_rate": resampling.get("target_sampling_rate_hz"),
        "target_shape": resampling.get("target_shape"),
        "up": resampling.get("up"),
        "down": resampling.get("down"),
        "axis": resampling.get("axis"),
    }
    required_signal = {
        "source_sampling": SPH_NATIVE_FREQUENCY_HZ,
        "source_leads": list(LEADS),
        "key": SPH_HDF5_DATASET_KEY,
        "shape": [len(LEADS), SPH_NATIVE_SAMPLES],
        "unit": "millivolts" if SPH_SIGNAL_UNIT == "mV" else SPH_SIGNAL_UNIT,
        "target_rate": SPH_TARGET_FREQUENCY_HZ,
        "target_shape": [len(LEADS), SPH_TARGET_SAMPLES],
        "up": 1,
        "down": 5,
        "axis": 1,
    }
    if expected_signal != required_signal or purpose.get("target_order") != list(LABEL_ORDER):
        raise SPHTransportIntegrityError("SPH signal contract differs from adapter")
    label = _mapping(payload.get("label_contract"), "label_contract")
    direct_map = _mapping(label.get("direct_map"), "direct_map")
    observed_map = {
        name: frozenset(_integer_sequence(_mapping(direct_map.get(name), name).get("codes"), name))
        for name in LABEL_ORDER
    }
    if observed_map != dict(SPH_SUPERCLASS_CODES):
        raise SPHTransportIntegrityError("SPH direct label map differs from adapter")
    ambiguous_value = label.get("ambiguous_unmapped_primary_codes")
    if not isinstance(ambiguous_value, Mapping):
        raise SPHTransportIntegrityError("ambiguous code contract must be a mapping")
    ambiguous = frozenset(int(key) for key in ambiguous_value)
    if ambiguous != SPH_AMBIGUOUS_PRIMARY_CODES:
        raise SPHTransportIntegrityError("SPH ambiguous-code set differs from adapter")
    normalization = _mapping(signal.get("normalization"), "normalization")
    if any(normalization.get(key) is not False for key in ("refit_on_sph", "adapt_on_sph")) or any(
        signal.get(key) is not False
        for key in (
            "allow_transpose",
            "allow_lead_reordering",
            "allow_amplitude_rescaling",
        )
    ):
        raise SPHTransportIntegrityError("SPH preprocessing adaptation is not forbidden")


def _validate_evaluation_contract(payload: Mapping[str, object]) -> None:
    models = _mapping(payload.get("frozen_models"), "frozen_models")
    for key in (
        "tuning_on_sph",
        "model_selection_on_sph",
        "recalibration_on_sph",
        "threshold_fitting_on_sph",
        "gate_fitting_on_sph",
    ):
        if models.get(key) is not False:
            raise SPHTransportIntegrityError(f"{key} must be false")
    evaluation = _mapping(payload.get("evaluation"), "evaluation")
    bootstrap = _mapping(payload.get("bootstrap"), "bootstrap")
    if (
        evaluation.get("ece_bins") != EXPECTED_ECE_BINS
        or evaluation.get("sensitivities") != ["broad_exact10", "no_ambiguous_mapped"]
        or bootstrap.get("resamples") != EXPECTED_BOOTSTRAP_RESAMPLES
        or bootstrap.get("confidence") != EXPECTED_BOOTSTRAP_CONFIDENCE
        or bootstrap.get("minimum_valid_resamples") != EXPECTED_BOOTSTRAP_MINIMUM_VALID
        or bootstrap.get("base_seed") != EXPECTED_BOOTSTRAP_BASE_SEED
        or _mapping(bootstrap.get("cohort_seed_offsets"), "cohort_seed_offsets")
        != EXPECTED_COHORT_SEED_OFFSETS
    ):
        raise SPHTransportIntegrityError("SPH evaluation/bootstrap contract differs")
    runtime = _mapping(payload.get("runtime"), "runtime")
    expected_runtime = {
        "inference_passes": 1,
        "device": "cuda:0",
        "allow_auto_device_resolution": False,
        "allow_cpu_fallback": False,
        "precision": "bf16",
        "bf16_required": True,
        "batch_size": 128,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "shuffle": False,
        "drop_last": False,
        "failure_policy": "fail_closed_without_in_place_setting_changes",
        "out_of_memory_policy": "fail_closed",
    }
    if any(runtime.get(key) != value for key, value in expected_runtime.items()):
        raise SPHTransportIntegrityError("SPH frozen runtime contract differs")


def _validate_runtime_compatibility(members: Sequence[AuditMemberRuntime]) -> None:
    reference = members[0]
    reference_normalization = reference.normalization.to_dict()
    for member in members:
        if (
            member.runtime.device != reference.runtime.device
            or member.runtime.bf16_enabled != reference.runtime.bf16_enabled
            or member.normalization.to_dict() != reference_normalization
        ):
            raise SPHTransportIntegrityError(
                "exact-six runtime devices, BF16, or PTB normalization differ"
            )


def _architecture_summaries(
    reports: Mapping[str, Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    summaries: dict[str, Mapping[str, object]] = {}
    for cohort_name in COHORT_ORDER:
        for architecture in ("resnet1d", "ecg_transformer"):
            values: dict[str, list[float | None]] = {}
            members: list[str] = []
            for seed in (2026, 2027, 2028):
                member_id = f"{architecture}-seed{seed}"
                report = reports[f"{cohort_name}/{member_id}"]
                members.append(member_id)
                for name, value in _summary_scalars(report).items():
                    values.setdefault(name, []).append(value)
            statistics: dict[str, Mapping[str, object]] = {}
            for name, observed in sorted(values.items()):
                if len(observed) != 3:
                    raise SPHTransportIntegrityError(
                        f"architecture summary {name!r} does not retain all three seeds"
                    )
                valid = [value for value in observed if value is not None]
                statistics[name] = MappingProxyType(
                    {
                        "values": observed,
                        "mean": float(np.mean(valid)) if valid else None,
                        "sample_standard_deviation": (
                            float(np.std(valid, ddof=1)) if len(valid) > 1 else None
                        ),
                        "valid_members": len(valid),
                    }
                )
            summaries[f"{cohort_name}__{architecture}"] = MappingProxyType(
                {
                    "cohort": cohort_name,
                    "architecture": architecture,
                    "members": members,
                    "seeds": [2026, 2027, 2028],
                    "statistics": statistics,
                }
            )
    return summaries


def _summary_scalars(report: Mapping[str, object]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    probability_views = _mapping(report.get("probability_views"), "probability_views")
    for view in ("raw_sigmoid", "frozen_temperature_calibrated"):
        section = _mapping(probability_views.get(view), view)
        metrics = _mapping(section.get("metrics"), f"{view}.metrics")
        macro = _mapping(metrics.get("macro"), f"{view}.macro")
        for metric in ("roc_auc", "average_precision", "brier_score", "ece"):
            result[f"{view}.macro.{metric}"] = _optional_number(macro.get(metric))
        per_label = metrics.get("per_label")
        if not isinstance(per_label, list):
            raise SPHTransportIntegrityError(f"{view}.per_label must be a list")
        for label in LABEL_ORDER:
            selected = [
                _mapping(item, f"{view}.{label}")
                for item in per_label
                if isinstance(item, Mapping) and item.get("label") == label
            ]
            if len(selected) != 1:
                raise SPHTransportIntegrityError(
                    f"{view} per-label metric {label!r} is missing or duplicated"
                )
            for metric in (
                "prevalence",
                "roc_auc",
                "average_precision",
                "brier_score",
                "ece",
            ):
                result[f"{view}.per_label.{label}.{metric}"] = _optional_number(
                    selected[0].get(metric)
                )
    threshold = _mapping(report.get("frozen_threshold_decisions"), "frozen_threshold_decisions")
    result["frozen_threshold_decisions.hamming_risk"] = _optional_number(
        threshold.get("hamming_risk")
    )
    result["frozen_threshold_decisions.exact_match_accuracy"] = _optional_number(
        threshold.get("exact_match_accuracy")
    )
    gates = report.get("frozen_entropy_gates")
    if not isinstance(gates, list):
        raise SPHTransportIntegrityError("frozen entropy gates must be a list")
    observed_targets = tuple(
        _optional_number(_mapping(gate, "entropy gate").get("target_coverage")) for gate in gates
    )
    if observed_targets != EXPECTED_GATE_TARGETS:
        raise SPHTransportIntegrityError(
            "frozen entropy gates differ from targets 1.0, 0.9, 0.8, 0.7, 0.5"
        )
    for gate, target in zip(gates, EXPECTED_GATE_TARGETS, strict=True):
        selected_gate = _mapping(gate, "entropy gate")
        prefix = f"frozen_entropy_gates.{_gate_summary_key(target)}"
        for metric in (
            "observed_coverage",
            "hamming_risk",
            "exact_match_accuracy",
        ):
            result[f"{prefix}.{metric}"] = _optional_number(selected_gate.get(metric))
    return result


def _gate_summary_key(target: float) -> str:
    return f"target_{target:.1f}".replace(".", "p")


def _array_file_entries(files: Any, root: Path) -> list[dict[str, object]]:
    return [_file_entry(files.npz_path, root), _file_entry(files.json_path, root)]


def _file_entry(path: Path, root: Path) -> dict[str, object]:
    return {
        "file": path.resolve().relative_to(root.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _prefixed_file_sha256(path),
    }


def _assert_public_json(value: object, forbidden: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in forbidden:
                raise SPHTransportIntegrityError(f"public JSON contains private key {key!r}")
            _assert_public_json(item, forbidden)
    elif isinstance(value, list):
        for item in value:
            _assert_public_json(item, forbidden)
    elif isinstance(value, str):
        _assert_public_text(value, forbidden)


def _assert_public_text(value: str, forbidden: set[str]) -> None:
    normalized = value.replace("\\", "/")
    lowered = normalized.casefold()
    if (
        _SOURCE_ID_PATTERN.search(value) is not None
        or ":/" in normalized
        or normalized.casefold().endswith(".h5")
        or "data/raw/" in lowered
        or any(
            re.search(rf"(?<![a-z0-9]){re.escape(field)}(?![a-z0-9])", lowered) is not None
            for field in forbidden
        )
    ):
        raise SPHTransportIntegrityError("public text contains a private identity/path field")


def _verify_file_binding(binding: Mapping[str, object], project_root: Path, context: str) -> Path:
    path = _verify_path_hash_binding(binding, project_root, context, hash_key="local_sha256")
    expected_size = _integer(binding.get("size_bytes"), f"{context}.size_bytes")
    if path.stat().st_size != expected_size:
        raise SPHTransportIntegrityError(f"{context} file size differs")
    return path


def _verify_path_hash_binding(
    binding: Mapping[str, object],
    project_root: Path,
    context: str,
    *,
    hash_key: str = "file_sha256",
) -> Path:
    path = _resolve_bound_path(binding.get("path"), project_root, f"{context}.path")
    expected = _prefixed_hash(binding.get(hash_key), f"{context}.{hash_key}")
    if _prefixed_file_sha256(path) != expected:
        raise SPHTransportIntegrityError(f"{context} SHA-256 differs")
    return path


def _resolve_bound_path(value: object, project_root: Path, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SPHTransportIntegrityError(f"{context} must be a relative path")
    raw = Path(value)
    if raw.is_absolute():
        raise SPHTransportIntegrityError(f"{context} must remain project-relative")
    resolved = (project_root / raw).resolve()
    if not resolved.is_relative_to(project_root):
        raise SPHTransportIntegrityError(f"{context} escapes the project root")
    return resolved


def _prefixed_file_sha256(path: Path) -> str:
    return "sha256:" + sha256_file(path)


def _prefixed_hash(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise SPHTransportIntegrityError(f"{context} must be a prefixed SHA-256")
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise SPHTransportIntegrityError(f"{context} must be a prefixed SHA-256")
    return value


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise SPHTransportIntegrityError(f"{context} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SPHTransportIntegrityError(f"{context} must be an integer")
    return value


def _string_sequence(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SPHTransportIntegrityError(f"{context} must be a string list")
    return tuple(value)


def _integer_sequence(value: object, context: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise SPHTransportIntegrityError(f"{context} must be an integer list")
    return tuple(value)


def _optional_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SPHTransportIntegrityError("summary metric must be numeric or null")
    number = float(value)
    if not math.isfinite(number):
        raise SPHTransportIntegrityError("summary metric must be finite")
    return number


def _canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _readonly(value: NDArray[Any]) -> NDArray[Any]:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


__all__ = [
    "COHORT_ORDER",
    "EXPECTED_BOOTSTRAP_BASE_SEED",
    "EXPECTED_BOOTSTRAP_CONFIDENCE",
    "EXPECTED_BOOTSTRAP_MINIMUM_VALID",
    "EXPECTED_BOOTSTRAP_RESAMPLES",
    "EXPECTED_COHORT_COUNTS",
    "EXPECTED_GATE_TARGETS",
    "SPHFrozenMemberOutput",
    "SPHTransportAttempt",
    "SPHTransportCohorts",
    "SPHTransportError",
    "SPHTransportEvaluation",
    "SPHTransportInference",
    "SPHTransportIntegrityError",
    "SPHTransportRun",
    "SPHTransportSpec",
    "apply_frozen_member_decisions",
    "assert_identifier_free_public_outputs",
    "evaluate_all_sph_cohorts",
    "infer_sph_transport",
    "load_sph_completed_runtime",
    "load_sph_transport_spec",
    "prepare_sph_transport_cohorts",
    "record_sph_attempt_failure",
    "reserve_sph_transport_attempt",
    "run_sph_transport",
    "save_sph_transport_outputs",
]
