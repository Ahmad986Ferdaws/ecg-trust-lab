from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

import ecg_trust.decisioning as decisioning_module
from ecg_trust.calibration_cli import app
from ecg_trust.decisioning import (
    CalibrationDecisionArtifact,
    DecisioningError,
    DecisionIntegrityError,
    FinalReportProvenanceError,
    fit_calibration_decisions,
    generate_final_report,
    load_calibration_decisions,
    save_calibration_decisions,
    save_final_report,
    verify_final_report,
)
from ecg_trust.predictions import (
    PredictionArtifact,
    create_prediction_artifact,
    load_prediction_artifact,
    save_prediction_artifact,
)
from ecg_trust.protocol import (
    DEFAULT_PROTOCOL_PATH,
    FINAL_TEST_CONFIRMATION,
    ExperimentProtocol,
    FinalTestAccessError,
    FinalTestAccessToken,
    FoldRole,
    authorize_final_test_access,
)


def _hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _token(protocol: ExperimentProtocol) -> FinalTestAccessToken:
    return authorize_final_test_access(
        protocol,
        purpose="synthetic offline final-report test",
        confirmation=FINAL_TEST_CONFIRMATION,
    )


def _stored_prediction(
    tmp_path: Path,
    *,
    fold: int,
    role: FoldRole,
    protocol: ExperimentProtocol,
    token: FinalTestAccessToken | None = None,
    config_hash: str | None = None,
    model_name: str = "resnet1d",
) -> PredictionArtifact:
    n_samples = 20
    row = np.arange(n_samples)
    ecg_id = fold * 10_000 + row
    patient_id = fold * 1_000 + np.repeat(np.arange(n_samples // 2), 2)
    targets = ((row[:, None] + np.arange(5)[None, :]) % 2).astype(np.int8)
    logits = np.where(targets == 1, 2.5, -2.5).astype(np.float64)
    logits[::7] *= -1.0
    artifact = create_prediction_artifact(
        ecg_id=ecg_id,
        patient_id=patient_id,
        strat_fold=np.full(n_samples, fold),
        targets=targets,
        raw_logits=logits,
        model_name=model_name,
        model_seed=17,
        protocol=protocol,
        config_hash=config_hash or _hash("shared-model-config"),
        manifest_hash=_hash("ptbxl-manifest"),
        fold_role=role,
        created_at_utc="2026-08-08T12:00:00Z",
        test_access=token,
    )
    stored = save_prediction_artifact(
        artifact,
        tmp_path / f"fold-{fold}-{model_name}.npz",
        protocol=protocol,
        test_access=token,
    )
    return load_prediction_artifact(
        stored.npz_path,
        protocol=protocol,
        test_access=token,
    )


def _stored_decisions(
    tmp_path: Path,
    protocol: ExperimentProtocol,
) -> tuple[PredictionArtifact, CalibrationDecisionArtifact]:
    calibration = _stored_prediction(
        tmp_path,
        fold=9,
        role=FoldRole.CALIBRATION,
        protocol=protocol,
    )
    decisions = fit_calibration_decisions(
        calibration,
        protocol=protocol,
        coverage_targets=(1.0, 0.75, 0.5),
        created_at_utc="2026-08-08T13:00:00Z",
    )
    stored = save_calibration_decisions(decisions, tmp_path / "decisions.json")
    return calibration, load_calibration_decisions(stored.path, protocol=protocol)


def test_calibration_fits_only_integrity_bound_fold_9_and_freezes_gates(
    tmp_path: Path,
) -> None:
    protocol = ExperimentProtocol.canonical()
    calibration, loaded = _stored_decisions(tmp_path, protocol)

    assert loaded.integrity_sha256 is not None
    assert loaded.model_name == calibration.model_name
    assert loaded.config_hash == calibration.config_hash
    assert loaded.temperature_scaling.source_folds == (9,)
    assert loaded.threshold_optimization.source_folds == (9,)
    assert [gate.target_coverage for gate in loaded.coverage_gates] == [1.0, 0.75, 0.5]
    assert all(
        gate.calibration_coverage >= gate.target_coverage
        for gate in loaded.coverage_gates
    )
    assert loaded.coverage_gates[0].maximum_entropy == 1.0
    assert loaded.coverage_gates[0].calibration_coverage == 1.0
    json.dumps(loaded.to_payload(), allow_nan=False)

    fold8 = _stored_prediction(
        tmp_path,
        fold=8,
        role=FoldRole.MODEL_SELECTION,
        protocol=protocol,
    )
    with pytest.raises(DecisioningError, match="fold-9-only"):
        fit_calibration_decisions(fold8, protocol=protocol)

    in_memory = create_prediction_artifact(
        ecg_id=[1, 2],
        patient_id=[1, 2],
        strat_fold=[9, 9],
        targets=[[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]],
        raw_logits=[[-1, 1, -1, 1, -1], [1, -1, 1, -1, 1]],
        model_name="resnet1d",
        model_seed=17,
        protocol=protocol,
        config_hash=_hash("shared-model-config"),
        manifest_hash=_hash("ptbxl-manifest"),
        fold_role=FoldRole.CALIBRATION,
        created_at_utc="2026-08-08T12:00:00Z",
    )
    with pytest.raises(DecisioningError, match="integrity-bound"):
        fit_calibration_decisions(in_memory, protocol=protocol)


def test_calibration_json_integrity_and_no_overwrite(tmp_path: Path) -> None:
    protocol = ExperimentProtocol.canonical()
    calibration = _stored_prediction(
        tmp_path,
        fold=9,
        role=FoldRole.CALIBRATION,
        protocol=protocol,
    )
    decisions = fit_calibration_decisions(
        calibration, protocol=protocol, coverage_targets=(1.0, 0.5)
    )
    path = tmp_path / "calibration.json"
    saved = save_calibration_decisions(decisions, path)
    with pytest.raises(FileExistsError, match="already exists"):
        save_calibration_decisions(decisions, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["temperature_scaling"]["temperature"] = 9.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DecisionIntegrityError, match="SHA-256 mismatch"):
        load_calibration_decisions(saved.path, protocol=protocol)


def test_final_report_uses_frozen_decisions_without_calling_fit_functions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = ExperimentProtocol.canonical()
    _, decisions = _stored_decisions(tmp_path, protocol)
    token = _token(protocol)
    final_prediction = _stored_prediction(
        tmp_path,
        fold=10,
        role=FoldRole.FINAL_TEST,
        protocol=protocol,
        token=token,
    )

    def forbidden_fit(**_: object) -> object:
        raise AssertionError("final report attempted calibration fitting")

    monkeypatch.setattr(decisioning_module, "fit_temperature_scaling", forbidden_fit)
    monkeypatch.setattr(decisioning_module, "optimize_thresholds", forbidden_fit)
    subgroup_ids = final_prediction.ecg_id.copy()
    subgroups = {
        "sex": np.where(np.arange(final_prediction.n_samples) % 2 == 0, "F", "M"),
        "site": np.where(np.arange(final_prediction.n_samples) < 12, "A", "B"),
    }
    report = generate_final_report(
        decisions,
        final_prediction,
        protocol=protocol,
        test_access=token,
        subgroup_ecg_id=subgroup_ids,
        subgroups=subgroups,
        bootstrap_resamples=30,
        bootstrap_seed=11,
        bootstrap_minimum_valid=5,
        minimum_group_samples=2,
        minimum_group_patients=2,
        ece_bins=5,
        created_at_utc="2026-08-08T14:00:00Z",
    )

    assert report.applied_temperature == decisions.temperature_scaling.temperature
    assert report.applied_thresholds == decisions.threshold_optimization.thresholds
    assert report.patient_bootstrap.completed_resamples == 30
    assert report.metrics.n_samples == final_prediction.n_samples
    assert len(report.subgroup_audit.groups) == 4
    assert report.selective_prediction.uncertainty_method == "mean_normalized_binary_entropy"
    probabilities = decisions.temperature_scaling.predict_proba(final_prediction.raw_logits)
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    entropy = (
        -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped))
        / np.log(2.0)
    ).mean(axis=1)
    for gate, point in zip(
        decisions.coverage_gates,
        report.selective_prediction.coverage_points,
        strict=True,
    ):
        expected = tuple(int(index) for index in np.flatnonzero(entropy <= gate.maximum_entropy))
        assert point.selected_indices == expected
    json.dumps(report.to_payload(), allow_nan=False)


def test_final_report_requires_token_exact_provenance_and_subgroup_alignment(
    tmp_path: Path,
) -> None:
    protocol = ExperimentProtocol.canonical()
    _, decisions = _stored_decisions(tmp_path, protocol)
    token = _token(protocol)
    final_prediction = _stored_prediction(
        tmp_path,
        fold=10,
        role=FoldRole.FINAL_TEST,
        protocol=protocol,
        token=token,
    )
    subgroup = {"sex": np.asarray(["F"] * final_prediction.n_samples)}

    with pytest.raises(FinalTestAccessError, match="fold 10 is sealed"):
        generate_final_report(
            decisions,
            final_prediction,
            protocol=protocol,
            test_access=None,  # type: ignore[arg-type]
            subgroup_ecg_id=final_prediction.ecg_id,
            subgroups=subgroup,
            bootstrap_resamples=10,
        )
    with pytest.raises(FinalReportProvenanceError, match="ecg_id order"):
        generate_final_report(
            decisions,
            final_prediction,
            protocol=protocol,
            test_access=token,
            subgroup_ecg_id=final_prediction.ecg_id[::-1],
            subgroups=subgroup,
            bootstrap_resamples=10,
        )

    mismatched = _stored_prediction(
        tmp_path,
        fold=10,
        role=FoldRole.FINAL_TEST,
        protocol=protocol,
        token=token,
        config_hash=_hash("different-config"),
        model_name="different",
    )
    with pytest.raises(FinalReportProvenanceError, match="provenance mismatch"):
        generate_final_report(
            decisions,
            mismatched,
            protocol=protocol,
            test_access=token,
            subgroup_ecg_id=mismatched.ecg_id,
            subgroups={"sex": np.asarray(["F"] * mismatched.n_samples)},
            bootstrap_resamples=10,
        )


def test_final_report_is_integrity_bound_and_token_gated(tmp_path: Path) -> None:
    protocol = ExperimentProtocol.canonical()
    _, decisions = _stored_decisions(tmp_path, protocol)
    token = _token(protocol)
    final_prediction = _stored_prediction(
        tmp_path,
        fold=10,
        role=FoldRole.FINAL_TEST,
        protocol=protocol,
        token=token,
    )
    report = generate_final_report(
        decisions,
        final_prediction,
        protocol=protocol,
        test_access=token,
        subgroup_ecg_id=final_prediction.ecg_id,
        subgroups={"sex": np.where(np.arange(20) % 2, "M", "F")},
        bootstrap_resamples=20,
        bootstrap_minimum_valid=5,
        minimum_group_samples=2,
        minimum_group_patients=2,
    )
    with pytest.raises(FinalTestAccessError, match="fold 10 is sealed"):
        save_final_report(
            report,
            tmp_path / "unauthorized.json",
            protocol=protocol,
            test_access=None,  # type: ignore[arg-type]
        )
    saved = save_final_report(
        report,
        tmp_path / "final-report.json",
        protocol=protocol,
        test_access=token,
    )
    verified = verify_final_report(saved.path, protocol=protocol, test_access=token)
    assert verified["report_sha256"] == saved.sha256

    with pytest.raises(FinalTestAccessError, match="fold 10 is sealed"):
        verify_final_report(
            saved.path, protocol=protocol, test_access=None  # type: ignore[arg-type]
        )
    payload = json.loads(saved.path.read_text(encoding="utf-8"))
    payload["metrics"]["macro"]["roc_auc"] = 0.0
    saved.path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DecisionIntegrityError, match="SHA-256 mismatch"):
        verify_final_report(saved.path, protocol=protocol, test_access=token)


def test_cli_runs_synthetic_fold9_and_authorized_fold10_flow(tmp_path: Path) -> None:
    protocol = ExperimentProtocol.canonical()
    calibration = _stored_prediction(
        tmp_path,
        fold=9,
        role=FoldRole.CALIBRATION,
        protocol=protocol,
    )
    calibration_path = tmp_path / "fold-9-resnet1d.npz"
    assert calibration.integrity_sha256 is not None
    decision_path = tmp_path / "cli-decisions.json"
    runner = CliRunner()
    fit_result = runner.invoke(
        app,
        [
            "fit",
            "--predictions",
            str(calibration_path),
            "--protocol",
            str(DEFAULT_PROTOCOL_PATH),
            "--output",
            str(decision_path),
            "--coverage",
            "1.0",
            "--coverage",
            "0.5",
        ],
    )
    assert fit_result.exit_code == 0, fit_result.output
    assert decision_path.is_file()

    token = _token(protocol)
    final_prediction = _stored_prediction(
        tmp_path,
        fold=10,
        role=FoldRole.FINAL_TEST,
        protocol=protocol,
        token=token,
    )
    final_path = tmp_path / "fold-10-resnet1d.npz"
    subgroup_path = tmp_path / "subgroups.json"
    subgroup_path.write_text(
        json.dumps(
            {
                "ecg_id": final_prediction.ecg_id.tolist(),
                "attributes": {"sex": ["F", "M"] * 10},
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "cli-final-report.json"
    report_result = runner.invoke(
        app,
        [
            "final-report",
            "--decisions",
            str(decision_path),
            "--predictions",
            str(final_path),
            "--subgroups",
            str(subgroup_path),
            "--protocol",
            str(DEFAULT_PROTOCOL_PATH),
            "--output",
            str(report_path),
            "--final-test-purpose",
            "synthetic CLI test",
            "--final-test-confirmation",
            FINAL_TEST_CONFIRMATION,
            "--bootstrap-resamples",
            "10",
            "--minimum-valid-resamples",
            "2",
            "--minimum-group-samples",
            "2",
            "--minimum-group-patients",
            "2",
        ],
    )
    assert report_result.exit_code == 0, report_result.output
    assert report_path.is_file()
