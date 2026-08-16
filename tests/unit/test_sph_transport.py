from __future__ import annotations

import hashlib
import io
import json
import math
import tarfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
import torch
import yaml  # type: ignore[import-untyped]
from torch import Tensor, nn
from torch.utils.data import Dataset

import ecg_trust.sph_transport as sph_transport_module
from ecg_trust.audit_artifacts import load_audit_array_artifact
from ecg_trust.audit_runtime import EXPECTED_AUDIT_MEMBER_IDS, CompletedAuditRuntime
from ecg_trust.constants import LEADS
from ecg_trust.protocol import LABEL_ORDER
from ecg_trust.sph_transport import (
    SPHFrozenMemberOutput,
    SPHTransportCohorts,
    SPHTransportEvaluation,
    SPHTransportInference,
    SPHTransportIntegrityError,
    SPHTransportSpec,
    _architecture_summaries,
    _bound_input_snapshot,
    _canonical_payload_sha256,
    _plain_json_mapping,
    _scientific_contract_projection,
    _source_inventory,
    _verify_bound_inputs_unchanged,
    apply_frozen_member_decisions,
    assert_identifier_free_public_outputs,
    infer_sph_transport,
    load_sph_inference_checkpoint,
    load_sph_transport_spec,
    record_sph_attempt_failure,
    reserve_sph_transport_attempt,
    run_sph_transport,
    save_sph_transport_outputs,
    verify_sph_archive_safety,
)
from scripts.sph_transport import build_parser


class _CountingDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, count: int) -> None:
        self.count = count
        self.reads = np.zeros(count, dtype=np.int64)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        self.reads[index] += 1
        signal = torch.full((12, 1000), float(index + 1), dtype=torch.float32)
        target = torch.tensor([(index + offset) % 2 for offset in range(5)], dtype=torch.float32)
        return signal, target


class _CountingModel(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset
        self.calls = 0

    def forward(self, signals: Tensor) -> Tensor:
        self.calls += 1
        base = signals.mean(dim=(1, 2)) + self.offset
        return torch.stack([base + index for index in range(5)], dim=1)


class _Normalization:
    def to_dict(self) -> dict[str, object]:
        return {"kind": "frozen-test-normalization"}


class _Temperature:
    temperature = 2.0

    def predict_proba(
        self, logits: np.ndarray[Any, Any], *, label_order: tuple[str, ...]
    ) -> np.ndarray[Any, Any]:
        assert label_order == LABEL_ORDER
        return 1.0 / (1.0 + np.exp(-np.asarray(logits) / self.temperature))


class _Thresholds:
    thresholds = (0.5, 0.5, 0.5, 0.5, 0.5)

    def apply(
        self, probabilities: np.ndarray[Any, Any], *, label_order: tuple[str, ...]
    ) -> np.ndarray[Any, Any]:
        assert label_order == LABEL_ORDER
        return np.asarray(probabilities) >= np.asarray(self.thresholds)[None, :]


def _fake_member(member_id: str, offset: float = 0.0) -> SimpleNamespace:
    architecture, raw_seed = member_id.split("-seed")
    model = _CountingModel(offset)
    decisions = SimpleNamespace(
        label_order=LABEL_ORDER,
        temperature_scaling=_Temperature(),
        threshold_optimization=_Thresholds(),
        coverage_gates=(
            SimpleNamespace(target_coverage=1.0, maximum_entropy=1.0),
            SimpleNamespace(target_coverage=0.5, maximum_entropy=0.5),
        ),
        integrity_sha256="sha256:" + "1" * 64,
    )
    return SimpleNamespace(
        member_id=member_id,
        architecture=architecture,
        seed=int(raw_seed),
        model=model,
        normalization=_Normalization(),
        runtime=SimpleNamespace(device=torch.device("cpu"), bf16_enabled=False),
        settings=SimpleNamespace(
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        ),
        decisions=decisions,
        checkpoint_sha256="sha256:" + "2" * 64,
        normalize_physical_batch=lambda signals: signals / 2.0,
    )


def _fake_runtime() -> SimpleNamespace:
    return SimpleNamespace(
        members=tuple(
            _fake_member(member_id, offset=float(index))
            for index, member_id in enumerate(EXPECTED_AUDIT_MEMBER_IDS)
        ),
        refit_bundle=SimpleNamespace(normalization_sha256="sha256:" + "3" * 64),
        clean_equivalence=tuple(
            SimpleNamespace(
                to_dict=lambda member_id=member_id: {
                    "member_id": member_id,
                    "exact": True,
                }
            )
            for member_id in EXPECTED_AUDIT_MEMBER_IDS
        ),
    )


def _output_spec(tmp_path: Path, output_name: str = "run") -> SPHTransportSpec:
    config = tmp_path / f"{output_name}-frozen.yaml"
    config.write_text("protocol_id: test\n", encoding="utf-8")
    config_sha256 = "sha256:" + hashlib.sha256(config.read_bytes()).hexdigest()
    spec = SPHTransportSpec(
        path=config,
        project_root=tmp_path,
        file_sha256=config_sha256,
        metadata_path=tmp_path / "metadata.csv",
        code_dictionary_path=tmp_path / "code.csv",
        rule_path=tmp_path / "rule.pdf",
        records_archive_path=tmp_path / "records.tar.gz",
        records_dir=tmp_path / "records",
        normalization_path=tmp_path / "normalization.json",
        refit_bundle_path=tmp_path / "refit.json",
        calibration_bundle_path=tmp_path / "calibration.json",
        final_evaluation_spec_path=tmp_path / "final.json",
        opening_ledger_path=tmp_path / "ledger.json",
        protocol_path=tmp_path / "protocol.yaml",
        output_root=tmp_path / output_name,
        public_row_permutation_seed=2_026_081_601,
        _payload_json="{}",
    )
    spec.records_dir.mkdir(exist_ok=True)
    for path in (
        spec.metadata_path,
        spec.code_dictionary_path,
        spec.rule_path,
        spec.records_archive_path,
        spec.normalization_path,
        spec.refit_bundle_path,
        spec.calibration_bundle_path,
        spec.final_evaluation_spec_path,
        spec.opening_ledger_path,
        spec.protocol_path,
    ):
        path.write_bytes(f"synthetic bound input: {path.name}\n".encode())
    return spec


def test_one_data_pass_fans_out_to_exact_six_and_preserves_alignment() -> None:
    dataset = _CountingDataset(7)
    expected_targets = np.asarray(
        [[(row + offset) % 2 for offset in range(5)] for row in range(7)],
        dtype=np.int8,
    )
    runtime = _fake_runtime()

    result = infer_sph_transport(
        cast(CompletedAuditRuntime, runtime),
        dataset,
        expected_targets=expected_targets,
    )

    assert np.array_equal(dataset.reads, np.ones(7, dtype=np.int64))
    assert np.array_equal(result.targets, expected_targets)
    assert tuple(result.raw_logits) == EXPECTED_AUDIT_MEMBER_IDS
    assert result.per_record_max_abs_mv.tolist() == pytest.approx(
        [float(index) for index in range(1, 8)]
    )
    assert result.physical_signal_sha256.startswith("sha256:")
    for member in runtime.members:
        assert member.model.calls == math.ceil(7 / 2)
        assert result.raw_logits[member.member_id].shape == (7, 5)


def test_frozen_member_policy_uses_threshold_equality_and_fixed_gate_cutoffs() -> None:
    member = _fake_member(EXPECTED_AUDIT_MEMBER_IDS[0])
    logits = np.zeros((3, 5), dtype=np.float64)

    result = apply_frozen_member_decisions(cast(Any, member), logits)

    assert result.predictions.all()  # sigmoid(0 / T) == frozen threshold 0.5
    assert result.gate_selected[:, 0].all()
    assert not result.gate_selected[:, 1].any()  # entropy 1.0 exceeds cutoff 0.5
    assert result.temperature == 2.0


def test_private_alignment_and_identifier_free_public_outputs_are_immutable(
    tmp_path: Path,
) -> None:
    count = 4
    spec = _output_spec(tmp_path)
    output_root = spec.output_root
    targets = np.asarray(
        [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [1, 1, 0, 0, 0], [0, 0, 1, 1, 0]],
        dtype=np.int8,
    )
    masks = {
        "primary_mapped": np.asarray([True, True, True, True]),
        "broad_exact10": np.asarray([True, True, True, True]),
        "no_ambiguous_mapped": np.asarray([True, False, True, True]),
    }
    cohorts = SPHTransportCohorts(
        manifest=pd.DataFrame(
            {
                "ecg_id": ["A00001", "A00002", "A00003", "A00004"],
                "patient_id": ["S00001", "S00002", "S00002", "S00003"],
                "record_path": [f"A0000{index}.h5" for index in range(1, 5)],
            }
        ),
        ecg_ids=np.asarray(["A00001", "A00002", "A00003", "A00004"]),
        patient_ids=np.asarray(["S00001", "S00002", "S00002", "S00003"]),
        targets=targets,
        masks=MappingProxyType(masks),
        alignment_sha256="sha256:" + "b" * 64,
        summaries=MappingProxyType(
            {
                name: MappingProxyType({"records": int(mask.sum()), "patients": 3})
                for name, mask in masks.items()
            }
        ),
    )
    inference = SPHTransportInference(
        targets=targets,
        raw_logits=MappingProxyType(
            {
                member_id: np.zeros((count, len(LABEL_ORDER)), dtype=np.float64)
                for member_id in EXPECTED_AUDIT_MEMBER_IDS
            }
        ),
        per_record_max_abs_mv=np.arange(1, count + 1, dtype=np.float64),
        per_record_lead_max_abs_mv=np.ones((count, 12), dtype=np.float64),
        physical_signal_sha256="sha256:" + "c" * 64,
        per_lead_qc=MappingProxyType(
            {lead: MappingProxyType({"sample_mean_mv": 0.0}) for lead in LEADS}
        ),
    )
    runtime = _fake_runtime()
    outputs: dict[str, SPHFrozenMemberOutput] = {}
    member_reports: dict[str, Mapping[str, object]] = {}
    metric_payload = {
        "macro": {
            "roc_auc": 0.7,
            "average_precision": 0.6,
            "brier_score": 0.2,
            "ece": 0.1,
        },
        "per_label": [
            {
                "label": label,
                "prevalence": 0.5,
                "roc_auc": 0.7,
                "average_precision": 0.6,
                "brier_score": 0.2,
                "ece": 0.1,
            }
            for label in LABEL_ORDER
        ],
    }
    for member in runtime.members:
        output = apply_frozen_member_decisions(
            cast(Any, member), np.zeros((count, 5), dtype=np.float64)
        )
        outputs[member.member_id] = output
        for cohort_name in masks:
            member_reports[f"{cohort_name}/{member.member_id}"] = MappingProxyType(
                {
                    "n_samples": int(masks[cohort_name].sum()),
                    "probability_views": {
                        "raw_sigmoid": {"metrics": metric_payload},
                        "frozen_temperature_calibrated": {"metrics": metric_payload},
                    },
                    "frozen_threshold_decisions": {
                        "hamming_risk": 0.25,
                        "exact_match_accuracy": 0.5,
                    },
                    "frozen_entropy_gates": [
                        {
                            "target_coverage": target,
                            "observed_coverage": target - 0.05,
                            "hamming_risk": 0.2,
                            "exact_match_accuracy": 0.4,
                        }
                        for target in (1.0, 0.9, 0.8, 0.7, 0.5)
                    ],
                }
            )
    interval = {"estimate": 0.01, "lower": -0.02, "upper": 0.04}
    paired_reports = {
        f"{cohort_name}/seed{seed}": MappingProxyType(
            {
                "probability_views": {
                    "frozen_temperature_calibrated": {
                        "macro": {
                            metric: interval
                            for metric in (
                                "roc_auc",
                                "average_precision",
                                "brier_score",
                                "ece",
                            )
                        }
                    }
                }
            }
        )
        for cohort_name in masks
        for seed in (2026, 2027, 2028)
    }
    evaluation = SPHTransportEvaluation(
        member_outputs=MappingProxyType(outputs),
        member_reports=MappingProxyType(member_reports),
        paired_reports=MappingProxyType(paired_reports),
        architecture_summaries=MappingProxyType(_architecture_summaries(member_reports)),
    )

    source_inventory = MappingProxyType({"status": "synthetic_test"})
    execution_state = MappingProxyType({"git_worktree_clean": np.bool_(True)})
    attempt = reserve_sph_transport_attempt(
        spec,
        cast(CompletedAuditRuntime, runtime),
        source_inventory=source_inventory,
        execution_state=execution_state,
    )
    committed = save_sph_transport_outputs(
        spec,
        cast(CompletedAuditRuntime, runtime),
        cohorts,
        inference,
        evaluation,
        attempt=attempt,
    )

    private = load_audit_array_artifact(output_root / "private" / "cohort_alignment.npz")
    public = load_audit_array_artifact(
        output_root / "public" / "member_predictions" / "resnet1d-seed2026.npz"
    )
    checkpoint = load_sph_inference_checkpoint(
        output_root / "private" / "inference_checkpoint.npz",
        spec,
        cast(CompletedAuditRuntime, runtime),
        cohorts,
        private_alignment_sha256=committed.private_alignment_sha256,
    )
    assert {"ecg_id", "patient_id"} <= set(private.arrays)
    assert "ecg_id" not in public.arrays and "patient_id" not in public.arrays
    assert tuple(checkpoint.raw_logits) == EXPECTED_AUDIT_MEMBER_IDS
    assert np.array_equal(checkpoint.targets, targets)
    assert committed.public_manifest_path.is_file()
    for relative in (
        "private/protocol.snapshot.yaml",
        "private/source_inventory.json",
        "private/bound_inputs.preinference.json",
        "private/cohort_manifest.parquet",
        "private/inference_checkpoint.npz",
        "private/inference_checkpoint.json",
        "private/RUN_LOG.md",
        "private/derived_artifacts.manifest.json",
        "public/cohort_summary.json",
        "public/FINAL_RESULTS.md",
        "public/member_predictions",
        "public/member_reports",
        "public/architecture_summaries",
        "public/paired_bootstrap_reports",
    ):
        assert (output_root / relative).exists()
    rendered = (output_root / "public" / "FINAL_RESULTS.md").read_text(encoding="utf-8")
    assert "Primary calibrated architecture results" in rendered
    assert "Paired Transformer-minus-ResNet" in rendered
    assert "| 1.0 | resnet1d |" in rendered
    assert "| 0.5 | ecg_transformer |" in rendered
    assert_identifier_free_public_outputs(output_root / "public")

    mutated_spec = _output_spec(tmp_path, "mutated-finalization")
    mutated_attempt = reserve_sph_transport_attempt(
        mutated_spec,
        cast(CompletedAuditRuntime, runtime),
        source_inventory=source_inventory,
        execution_state=execution_state,
    )
    mutated_prepared = sph_transport_module.prepare_sph_transport_outputs(
        mutated_spec,
        cast(CompletedAuditRuntime, runtime),
        cohorts,
        attempt=mutated_attempt,
        source_inventory=source_inventory,
    )
    mutated_spec.normalization_path.write_text("concurrently changed\n", encoding="utf-8")
    with pytest.raises(SPHTransportIntegrityError, match="changed after pre-inference"):
        save_sph_transport_outputs(
            mutated_spec,
            cast(CompletedAuditRuntime, runtime),
            cohorts,
            inference,
            evaluation,
            attempt=mutated_attempt,
            prepared=mutated_prepared,
        )
    assert not (mutated_spec.output_root / "private" / "derived_artifacts.manifest.json").exists()

    with pytest.raises(SPHTransportIntegrityError, match="unexpected state"):
        save_sph_transport_outputs(
            spec,
            cast(CompletedAuditRuntime, runtime),
            cohorts,
            inference,
            evaluation,
            attempt=attempt,
        )


def test_malformed_or_unfrozen_protocol_fails_before_source_access(tmp_path: Path) -> None:
    config = tmp_path / "configs" / "external_transport_sph_frozen_r2.yaml"
    config.parent.mkdir()
    config.write_text(
        "schema_version: 1\nprotocol_id: sph-external-transport-v1-r2\nstatus: draft\n",
        encoding="utf-8",
    )
    with pytest.raises(SPHTransportIntegrityError, match="not frozen"):
        load_sph_transport_spec(config)


def test_architecture_summaries_retain_per_label_threshold_and_all_gate_seed_values() -> None:
    reports: dict[str, Mapping[str, object]] = {}
    for cohort_name in ("primary_mapped", "broad_exact10", "no_ambiguous_mapped"):
        for architecture in ("resnet1d", "ecg_transformer"):
            for seed_index, seed in enumerate((2026, 2027, 2028)):
                metric_payload = {
                    "macro": {
                        "roc_auc": 0.70 + seed_index * 0.01,
                        "average_precision": 0.60,
                        "brier_score": 0.20,
                        "ece": 0.10,
                    },
                    "per_label": [
                        {
                            "label": label,
                            "prevalence": 0.50,
                            "roc_auc": 0.71 + seed_index * 0.01,
                            "average_precision": 0.61,
                            "brier_score": 0.21,
                            "ece": 0.11,
                        }
                        for label in LABEL_ORDER
                    ],
                }
                reports[f"{cohort_name}/{architecture}-seed{seed}"] = {
                    "probability_views": {
                        "raw_sigmoid": {"metrics": metric_payload},
                        "frozen_temperature_calibrated": {"metrics": metric_payload},
                    },
                    "frozen_threshold_decisions": {
                        "hamming_risk": 0.25,
                        "exact_match_accuracy": 0.50,
                    },
                    "frozen_entropy_gates": [
                        {
                            "target_coverage": target,
                            "observed_coverage": target - 0.01,
                            "hamming_risk": 0.20,
                            "exact_match_accuracy": 0.40,
                        }
                        for target in (1.0, 0.9, 0.8, 0.7, 0.5)
                    ],
                }

    summaries = _architecture_summaries(reports)
    summary = summaries["primary_mapped__resnet1d"]
    statistics = cast(Mapping[str, Mapping[str, object]], summary["statistics"])

    representative = statistics["raw_sigmoid.per_label.NORM.roc_auc"]
    assert representative["values"] == pytest.approx([0.71, 0.72, 0.73])
    assert representative["mean"] == pytest.approx(0.72)
    assert representative["sample_standard_deviation"] == pytest.approx(0.01)
    assert "frozen_threshold_decisions.hamming_risk" in statistics
    for target_key in ("target_1p0", "target_0p9", "target_0p8", "target_0p7", "target_0p5"):
        for metric in ("observed_coverage", "hamming_risk", "exact_match_accuracy"):
            key = f"frozen_entropy_gates.{target_key}.{metric}"
            assert key in statistics
            assert len(cast(list[object], statistics[key]["values"])) == 3


def test_attempt_reservation_and_failure_receipt_make_failed_run_visible(
    tmp_path: Path,
) -> None:
    spec = _output_spec(tmp_path, "failed-run")
    runtime = _fake_runtime()
    attempt = reserve_sph_transport_attempt(
        spec,
        cast(CompletedAuditRuntime, runtime),
        source_inventory={"archive_member_content_match": True},
        execution_state={"git_worktree_clean": True},
    )
    assert attempt.marker_path.is_file()
    marker = json.loads(attempt.marker_path.read_text(encoding="utf-8"))
    assert marker["bound_inputs_sha256"] == attempt.bound_inputs_sha256
    assert marker["bound_inputs"] == [dict(entry) for entry in attempt.bound_inputs]

    phase_state = {
        "sph_inference_started": True,
        "sph_inference_completed": False,
        "sph_predictions_generated": False,
        "inference_checkpoint_committed": False,
        "sph_metrics_started": False,
        "sph_metrics_completed": False,
        "output_commit_started": True,
        "output_commit_completed": False,
    }
    failure = record_sph_attempt_failure(
        attempt,
        RuntimeError("synthetic OOM"),
        failure_phase="sph_model_inference",
        phase_state=phase_state,
    )

    payload = json.loads(failure.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "ecg_trust.sph_transport_attempt_failure"
    assert payload["exception_type"] == "builtins.RuntimeError"
    assert payload["exception_message"] == "synthetic OOM"
    assert payload["failure_phase"] == "sph_model_inference"
    assert payload["phase_state"] == phase_state
    assert payload["protocol_sha256"] == spec.file_sha256
    assert payload["planned_attempt_sha256"] == attempt.marker_sha256
    assert payload["attempt_sha256"] == attempt.marker_sha256
    assert any(spec.output_root.iterdir())
    with pytest.raises(FileExistsError, match="permanently spent"):
        reserve_sph_transport_attempt(
            spec,
            cast(CompletedAuditRuntime, runtime),
            source_inventory={"archive_member_content_match": True},
            execution_state={"git_worktree_clean": True},
        )
    with pytest.raises(FileExistsError):
        record_sph_attempt_failure(
            attempt,
            RuntimeError("second failure"),
            failure_phase="sph_model_inference",
            phase_state=phase_state,
        )


def test_bound_input_mutation_fails_the_finalization_identity_gate(tmp_path: Path) -> None:
    spec = _output_spec(tmp_path, "mutated-bound-input")
    bound_inputs, bound_inputs_sha256 = _bound_input_snapshot(spec, require_all=True)
    prepared = SimpleNamespace(
        bound_inputs=bound_inputs,
        bound_inputs_sha256=bound_inputs_sha256,
    )
    spec.normalization_path.write_text("concurrently changed\n", encoding="utf-8")

    with pytest.raises(SPHTransportIntegrityError, match="changed after pre-inference"):
        _verify_bound_inputs_unchanged(spec, cast(Any, prepared))


def test_attempt_reservation_rejects_an_existing_empty_root(tmp_path: Path) -> None:
    spec = _output_spec(tmp_path, "empty-spent-root")
    spec.output_root.mkdir()

    with pytest.raises(FileExistsError, match="permanently spent"):
        reserve_sph_transport_attempt(
            spec,
            cast(CompletedAuditRuntime, _fake_runtime()),
            source_inventory={"status": "synthetic_test"},
            execution_state={"git_worktree_clean": True},
        )
    assert not any(spec.output_root.iterdir())


def test_concurrent_attempt_reservation_has_exactly_one_winner(tmp_path: Path) -> None:
    spec = _output_spec(tmp_path, "concurrent-root")
    runtime = cast(CompletedAuditRuntime, _fake_runtime())

    def reserve() -> object:
        try:
            return reserve_sph_transport_attempt(
                spec,
                runtime,
                source_inventory=MappingProxyType(
                    {"archive_audit": MappingProxyType({"safe_paths": True})}
                ),
                execution_state={"git_worktree_clean": True},
            )
        except FileExistsError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: reserve(), range(2)))

    assert sum(not isinstance(result, FileExistsError) for result in results) == 1
    assert sum(isinstance(result, FileExistsError) for result in results) == 1
    assert (spec.output_root / "private" / "attempt-start.json").is_file()


def test_production_shaped_source_inventory_and_nested_immutable_json_normalize(
    tmp_path: Path,
) -> None:
    spec = _output_spec(tmp_path, "production-shaped-inventory")
    for path in (
        spec.metadata_path,
        spec.code_dictionary_path,
        spec.rule_path,
        spec.records_archive_path,
    ):
        path.write_bytes(path.name.encode("ascii"))
    cohort_shell = SimpleNamespace(
        alignment_sha256="sha256:" + "a" * 64,
        summaries=MappingProxyType(
            {
                "primary_mapped": MappingProxyType(
                    {"records": np.int64(4), "positive_records": (1, 2, 3, 4, 5)}
                )
            }
        ),
    )
    inventory = _source_inventory(
        spec,
        cast(SPHTransportCohorts, cohort_shell),
        MappingProxyType(
            {
                "safe_paths": np.bool_(True),
                "quantiles": (np.float32(0.5), np.float64(1.25)),
            }
        ),
    )

    normalized = _plain_json_mapping(inventory, context="production source inventory")

    assert isinstance(inventory, MappingProxyType)
    assert normalized["archive_audit"] == {
        "safe_paths": True,
        "quantiles": [0.5, 1.25],
    }
    assert _canonical_payload_sha256(inventory, context="production source inventory").startswith(
        "sha256:"
    )
    attempt = reserve_sph_transport_attempt(
        spec,
        cast(CompletedAuditRuntime, _fake_runtime()),
        source_inventory=inventory,
        execution_state=MappingProxyType({"git": MappingProxyType({"clean": np.bool_(True)})}),
    )
    assert attempt.marker_path.is_file()


@pytest.mark.parametrize(
    "invalid",
    [Path("private.txt"), np.asarray([1.0]), float("nan"), float("inf"), float("-inf")],
)
def test_deep_json_normalizer_rejects_implicit_or_nonfinite_values(
    invalid: object,
) -> None:
    with pytest.raises(SPHTransportIntegrityError):
        _plain_json_mapping(
            MappingProxyType({"nested": MappingProxyType({"invalid": invalid})}),
            context="invalid protocol artifact",
        )


def test_invalid_marker_payload_is_rejected_before_output_root_creation(
    tmp_path: Path,
) -> None:
    spec = _output_spec(tmp_path, "prevalidation-failure")

    with pytest.raises(SPHTransportIntegrityError, match="non-finite"):
        reserve_sph_transport_attempt(
            spec,
            cast(CompletedAuditRuntime, _fake_runtime()),
            source_inventory=MappingProxyType(
                {"nested": MappingProxyType({"invalid": float("nan")})}
            ),
            execution_state={"git_worktree_clean": True},
        )

    assert not spec.output_root.exists()


def test_marker_commit_failure_spends_root_and_records_pre_inference_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _output_spec(tmp_path, "marker-io-failure")
    original_writer = sph_transport_module.write_new_hashed_json
    original = OSError("synthetic marker I/O failure")
    planned_sha256: str | None = None

    def fail_marker(
        path: str | Path,
        payload: Mapping[str, object],
        *,
        hash_field: str,
    ) -> tuple[Path, str]:
        nonlocal planned_sha256
        if Path(path).name == "attempt-start.json":
            planned_sha256 = _canonical_payload_sha256(payload, context="captured attempt marker")
            raise original
        return original_writer(path, payload, hash_field=hash_field)

    monkeypatch.setattr(sph_transport_module, "write_new_hashed_json", fail_marker)

    with pytest.raises(OSError) as caught:
        reserve_sph_transport_attempt(
            spec,
            cast(CompletedAuditRuntime, _fake_runtime()),
            source_inventory={"safe_paths": True},
            execution_state={"git_worktree_clean": True},
        )

    assert caught.value is original
    receipt = json.loads(
        (spec.output_root / "private" / "attempt-failure.json").read_text(encoding="utf-8")
    )
    assert receipt["failure_phase"] == "attempt_marker_commit"
    assert receipt["protocol_sha256"] == spec.file_sha256
    assert receipt["attempt_sha256"] is None
    assert receipt["planned_attempt_sha256"] == planned_sha256
    assert not any(cast(dict[str, bool], receipt["phase_state"]).values())
    assert spec.output_root.exists()


@pytest.mark.parametrize(
    ("failure_stage", "expected_phase", "expected_true_fields"),
    [
        (
            "prepare",
            "outcome_independent_output_commit",
            {"output_commit_started"},
        ),
        (
            "inference",
            "sph_model_inference",
            {"output_commit_started", "sph_inference_started"},
        ),
        (
            "checkpoint",
            "private_inference_checkpoint_commit",
            {
                "output_commit_started",
                "sph_inference_started",
                "sph_inference_completed",
                "sph_predictions_generated",
            },
        ),
        (
            "source_reverification",
            "post_inference_source_reverification",
            {
                "output_commit_started",
                "sph_inference_started",
                "sph_inference_completed",
                "sph_predictions_generated",
                "inference_checkpoint_committed",
            },
        ),
        (
            "metrics",
            "sph_metrics",
            {
                "output_commit_started",
                "sph_inference_started",
                "sph_inference_completed",
                "sph_predictions_generated",
                "inference_checkpoint_committed",
                "sph_metrics_started",
            },
        ),
        (
            "output",
            "final_output_commit",
            {
                "output_commit_started",
                "sph_inference_started",
                "sph_inference_completed",
                "sph_predictions_generated",
                "inference_checkpoint_committed",
                "sph_metrics_started",
                "sph_metrics_completed",
            },
        ),
    ],
)
def test_run_preserves_phase_accurate_failure_receipt_and_original_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    expected_phase: str,
    expected_true_fields: set[str],
) -> None:
    spec = _output_spec(tmp_path, f"phase-{failure_stage}")
    runtime = _fake_runtime()
    cohorts = SimpleNamespace(
        manifest=pd.DataFrame({"ecg_id": ["A00001"]}),
        targets=np.zeros((1, len(LABEL_ORDER)), dtype=np.int8),
    )
    inference = SimpleNamespace(raw_logits={})
    evaluation = SimpleNamespace()
    prepared = SimpleNamespace(private_alignment_sha256="sha256:" + "a" * 64)
    original = RuntimeError(f"original {failure_stage} failure")

    monkeypatch.setattr(
        "ecg_trust.sph_transport.verify_clean_git_execution",
        lambda _spec: MappingProxyType({"git_worktree_clean": True}),
    )
    monkeypatch.setattr(
        "ecg_trust.sph_transport.prepare_sph_transport_cohorts", lambda _spec: cohorts
    )
    archive_audit_calls = 0

    def audit_source(*_args: object) -> Mapping[str, object]:
        nonlocal archive_audit_calls
        archive_audit_calls += 1
        if failure_stage == "source_reverification" and archive_audit_calls == 2:
            raise original
        return MappingProxyType({"safe_paths": True})

    monkeypatch.setattr("ecg_trust.sph_transport.verify_sph_archive_safety", audit_source)
    monkeypatch.setattr(
        "ecg_trust.sph_transport._source_inventory",
        lambda _spec, _cohorts, _audit: MappingProxyType(
            {"archive_audit": MappingProxyType({"safe_paths": True})}
        ),
    )
    monkeypatch.setattr("ecg_trust.sph_transport.load_sph_completed_runtime", lambda _spec: runtime)
    monkeypatch.setattr(
        "ecg_trust.sph_transport.assert_frozen_sph_runtime",
        lambda _spec, _runtime: None,
    )
    monkeypatch.setattr(
        "ecg_trust.sph_transport.SPHExternalTransportDataset",
        lambda _manifest, _records, *, allow_all_zero: object(),
    )

    def prepare(*_args: object, **_kwargs: object) -> object:
        assert (spec.output_root / "private" / "attempt-start.json").is_file()
        if failure_stage == "prepare":
            raise original
        return prepared

    def infer(*_args: object, **_kwargs: object) -> object:
        assert (spec.output_root / "private" / "attempt-start.json").is_file()
        if failure_stage == "inference":
            raise original
        return inference

    def checkpoint(*_args: object, **_kwargs: object) -> None:
        if failure_stage == "checkpoint":
            raise original

    def evaluate(*_args: object, **_kwargs: object) -> object:
        if failure_stage == "metrics":
            raise original
        return evaluation

    def save(*_args: object, **_kwargs: object) -> object:
        if failure_stage == "output":
            raise original
        return SimpleNamespace()

    monkeypatch.setattr("ecg_trust.sph_transport.prepare_sph_transport_outputs", prepare)
    monkeypatch.setattr("ecg_trust.sph_transport.infer_sph_transport", infer)
    monkeypatch.setattr("ecg_trust.sph_transport.save_sph_inference_checkpoint", checkpoint)
    monkeypatch.setattr("ecg_trust.sph_transport.evaluate_all_sph_cohorts", evaluate)
    monkeypatch.setattr("ecg_trust.sph_transport.save_sph_transport_outputs", save)

    with pytest.raises(RuntimeError) as caught:
        run_sph_transport(spec)

    assert caught.value is original
    receipt = json.loads(
        (spec.output_root / "private" / "attempt-failure.json").read_text(encoding="utf-8")
    )
    assert receipt["failure_phase"] == expected_phase
    phase_state = cast(dict[str, bool], receipt["phase_state"])
    assert {name for name, value in phase_state.items() if value} == expected_true_fields


def test_failure_receipt_error_does_not_mask_original_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _output_spec(tmp_path, "receipt-write-failure")
    runtime = _fake_runtime()
    cohorts = SimpleNamespace(
        manifest=pd.DataFrame({"ecg_id": ["A00001"]}),
        targets=np.zeros((1, len(LABEL_ORDER)), dtype=np.int8),
    )
    original = RuntimeError("original inference failure")

    monkeypatch.setattr(
        "ecg_trust.sph_transport.verify_clean_git_execution", lambda _spec: {"clean": True}
    )
    monkeypatch.setattr(
        "ecg_trust.sph_transport.prepare_sph_transport_cohorts", lambda _spec: cohorts
    )
    monkeypatch.setattr(
        "ecg_trust.sph_transport.verify_sph_archive_safety",
        lambda _spec, _cohorts: {"safe_paths": True},
    )
    monkeypatch.setattr(
        "ecg_trust.sph_transport._source_inventory",
        lambda *_args: MappingProxyType({"safe_paths": True}),
    )
    monkeypatch.setattr("ecg_trust.sph_transport.load_sph_completed_runtime", lambda _spec: runtime)
    monkeypatch.setattr("ecg_trust.sph_transport.assert_frozen_sph_runtime", lambda *_args: None)
    monkeypatch.setattr(
        "ecg_trust.sph_transport.SPHExternalTransportDataset", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        "ecg_trust.sph_transport.prepare_sph_transport_outputs",
        lambda *_args, **_kwargs: SimpleNamespace(private_alignment_sha256="sha256:" + "a" * 64),
    )
    monkeypatch.setattr(
        "ecg_trust.sph_transport.infer_sph_transport",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(original),
    )
    monkeypatch.setattr(
        "ecg_trust.sph_transport.record_sph_attempt_failure",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("receipt disk failure")),
    )

    with pytest.raises(RuntimeError) as caught:
        run_sph_transport(spec)

    assert caught.value is original
    assert any("receipt disk failure" in note for note in getattr(original, "__notes__", []))


def test_public_privacy_scan_rejects_identity_in_markdown(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir()
    (public / "results.md").write_text(
        "A source row A24032 must never appear here.\n", encoding="utf-8"
    )
    with pytest.raises(SPHTransportIntegrityError, match="public text"):
        assert_identifier_free_public_outputs(public)


def test_public_privacy_scan_rejects_aha_code_field_in_text(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir()
    (public / "notes.txt").write_text("AHA_Code must stay private.\n", encoding="utf-8")
    with pytest.raises(SPHTransportIntegrityError, match="public text"):
        assert_identifier_free_public_outputs(public)


def test_public_privacy_scan_rejects_private_field_in_json_value(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir()
    (public / "notes.json").write_text('{"note":"do not emit AHA_Code values"}\n', encoding="utf-8")
    with pytest.raises(SPHTransportIntegrityError, match="public text"):
        assert_identifier_free_public_outputs(public)


def test_archive_audit_rejects_same_named_modified_extracted_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / "records.tar.gz"  # Official suffix may hold plain tar.
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    (records_dir / "A00001.h5").write_bytes(b"modified")
    with tarfile.open(archive_path, mode="w") as archive:
        payload = b"authoritative"
        member = tarfile.TarInfo("records/A00001.h5")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    monkeypatch.setattr(
        "ecg_trust.sph_transport.build_sph_transport_manifest",
        lambda _metadata, _codes: pd.DataFrame({"record_path": ["A00001.h5"]}),
    )
    spec = cast(
        SPHTransportSpec,
        SimpleNamespace(
            records_archive_path=archive_path,
            records_dir=records_dir,
            metadata_path=tmp_path / "metadata.csv",
            code_dictionary_path=tmp_path / "code.csv",
        ),
    )
    with pytest.raises(SPHTransportIntegrityError, match="differs from archive member"):
        verify_sph_archive_safety(spec, cast(Any, None))


def test_cli_exposes_no_scientific_overrides() -> None:
    parser = build_parser()
    destinations = {action.dest for action in parser._actions}  # noqa: SLF001
    assert destinations == {"help", "config"}
    args = parser.parse_args([])
    assert args.config.name == "external_transport_sph_frozen_r2.yaml"
    with pytest.raises(SystemExit):
        parser.parse_args(["--bootstrap-resamples", "2"])


def test_r2_protocol_preserves_the_v1_scientific_projection() -> None:
    project_root = Path(__file__).resolve().parents[2]
    v1_path = project_root / "configs" / "external_transport_sph_frozen.yaml"
    r2_path = project_root / "configs" / "external_transport_sph_frozen_r2.yaml"
    v1 = cast(dict[str, object], yaml.safe_load(v1_path.read_text(encoding="utf-8")))
    r2 = cast(dict[str, object], yaml.safe_load(r2_path.read_text(encoding="utf-8")))

    v1_projection = _scientific_contract_projection(v1)
    r2_projection = _scientific_contract_projection(r2)

    assert r2_projection == v1_projection
    assert r2["scientific_contract_sha256"] == _canonical_payload_sha256(
        v1_projection, context="v1 scientific projection"
    )
    supersession = cast(dict[str, object], r2["supersession"])
    assert supersession["superseded_execution_git_revision"] == (
        "724a510b03eb539eda2add6de359855f3ffaf2b5"
    )
    assert supersession["attempt_start_marker_created"] is False
    assert supersession["sph_model_inference_started"] is False
    assert supersession["ptb_fold10_clean_equivalence_inference_ran"] is True
    assert r2["protocol_id"] == "sph-external-transport-v1-r2"
