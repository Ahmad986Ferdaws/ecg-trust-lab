from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import ecg_trust.release_gates as release_gates
from ecg_trust.decisioning import load_calibration_decisions
from ecg_trust.predictions import create_prediction_artifact, save_prediction_artifact
from ecg_trust.protocol import ExperimentProtocol, FoldRole
from ecg_trust.release_gates import (
    EXPECTED_ARCHITECTURES,
    EXPECTED_SEEDS,
    ReleaseGateError,
    ReleaseIntegrityError,
    canonical_sha256,
    create_refit_bundle,
    export_fold9_predictions,
    fit_calibration_bundle,
    load_calibration_bundle,
    load_refit_bundle,
    save_calibration_bundle,
    save_refit_bundle,
)


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path.resolve()


class _FakeFreeze:
    def __init__(
        self,
        *,
        path: Path,
        manifest_hash: str,
        normalization_hash: str,
        recipes: dict[tuple[str, int], dict[str, object]],
    ) -> None:
        self.path = path
        self.artifact_sha256 = _file_hash(path)
        self.comparison_id = "comparison-v1"
        self.payload: dict[str, object] = {
            "comparison_id": self.comparison_id,
            "protocol_hash": ExperimentProtocol.canonical().protocol_hash,
            "manifest_hash": manifest_hash,
            "normalization_hash": normalization_hash,
        }
        self.recipes = recipes

    def recipe_template(self, architecture: str, seed: int) -> dict[str, object]:
        return self.recipes[(architecture, seed)]


def _completion_fixture(
    tmp_path: Path,
    *,
    architecture: str,
    seed: int,
    manifest: Path,
    normalization: Path,
    freeze: Path,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    member_id = f"{architecture}-seed{seed}"
    run_dir = tmp_path / "runs" / member_id
    checkpoint = _write(run_dir / "final.ckpt", f"checkpoint:{member_id}")
    metadata = _write(run_dir / "refit_metadata.json", f'{{"member":"{member_id}"}}')
    protocol_file = _write(run_dir / "protocol.json", "{}")
    history = _write(run_dir / "refit_history.jsonl", '{"epoch":0}\n')
    source_checkpoint = _write(
        tmp_path / "development" / f"{member_id}.ckpt", f"source:{member_id}"
    )
    source_dir = tmp_path / "development" / member_id
    source_dir.mkdir(parents=True, exist_ok=True)
    member_plan = _write(source_dir / "member-plan.json", "{}")
    source_metadata = _write(source_dir / "run_metadata.json", "{}")
    source_resolved = _write(source_dir / "resolved_config.json", "{}")
    source_history = _write(source_dir / "history.jsonl", '{"epoch":0}\n')
    source_prediction = _write(source_dir / "fold8.npz", "prediction")
    source_prediction_json = _write(source_dir / "fold8.json", "{}")
    freeze_hash = _file_hash(freeze)
    manifest_hash = _file_hash(manifest)
    normalization_hash = _file_hash(normalization)
    source_config_hash = canonical_sha256({"development": member_id})
    source_receipt_body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "ecg_trust.multiseed_member_completion",
        "comparison_id": "comparison-v1",
        "architecture": architecture,
        "seed": seed,
        "status": "complete",
        "member_plan_path": str(member_plan),
        "member_plan_sha256": _file_hash(member_plan),
        "run_dir": str(source_dir.resolve()),
        "run_metadata_path": str(source_metadata),
        "run_metadata_sha256": _file_hash(source_metadata),
        "resolved_config_path": str(source_resolved),
        "resolved_config_sha256": _file_hash(source_resolved),
        "history_path": str(source_history),
        "history_sha256": _file_hash(source_history),
        "best_checkpoint_path": str(source_checkpoint),
        "best_checkpoint_sha256": _file_hash(source_checkpoint),
        "config_hash": source_config_hash,
        "protocol_hash": ExperimentProtocol.canonical().protocol_hash,
        "manifest_hash": manifest_hash,
        "normalization_sha256": normalization_hash,
        "best_epoch": 4,
        "best_validation_macro_auroc": 0.81,
        "completed_epochs": 10,
        "prediction_path": str(source_prediction),
        "prediction_npz_sha256": _file_hash(source_prediction),
        "prediction_json_path": str(source_prediction_json),
        "prediction_artifact_sha256": canonical_sha256(
            {"prediction": member_id}
        ),
    }
    source_receipt_payload = dict(source_receipt_body)
    source_receipt_payload["artifact_sha256"] = canonical_sha256(
        source_receipt_body
    )
    source_completion = source_dir / "member-completion.json"
    source_completion.write_text(
        json.dumps(source_receipt_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_completion = source_completion.resolve()
    source_completion_hash = _file_hash(source_completion)
    source: dict[str, object] = {
        "member_completion": str(source_completion),
        "member_completion_sha256": source_completion_hash,
        "manifest_sha256": manifest_hash,
        "normalization_sha256": normalization_hash,
        "run_metadata": str(source_metadata),
        "run_metadata_sha256": _file_hash(source_metadata),
        "resolved_config": str(source_resolved),
        "resolved_config_file_sha256": _file_hash(source_resolved),
        "resolved_config_hash": source_config_hash,
        "history": str(source_history),
        "history_sha256": _file_hash(source_history),
        "best_checkpoint": str(source_checkpoint),
        "best_checkpoint_sha256": _file_hash(source_checkpoint),
        "prediction": str(source_prediction),
        "prediction_npz_sha256": _file_hash(source_prediction),
        "prediction_json": str(source_prediction_json),
        "prediction_artifact_sha256": source_receipt_payload[
            "prediction_artifact_sha256"
        ],
        "best_epoch": 4,
        "best_validation_macro_auroc": 0.81,
    }
    frozen_epochs = 12 if architecture == "resnet1d" else 15
    model_identity: dict[str, object] = {
        "architecture": architecture,
        "preset": "smoke",
        "class": f"synthetic.{architecture}",
        "trainable_parameters": 123,
        "resolved_architecture_config": {"synthetic": True},
    }
    recipe_template: dict[str, object] = {
        "schema_version": 2,
        "run_kind": "post_sweep_frozen_refit",
        "freeze_artifact": "${FREEZE_ARTIFACT_PATH}",
        "freeze_artifact_sha256": "${FREEZE_ARTIFACT_SHA256}",
        "comparison_id": "comparison-v1",
        "architecture": architecture,
        "confirmation_seed": seed,
        "run_name": f"{architecture}_refit_folds1-8_seed{seed}",
        "initialization": "fresh",
        "folds": {"refit": list(range(1, 9)), "normalization": list(range(1, 8))},
        "data": {
            "manifest": str(manifest),
            "dataset_root": str(tmp_path / "records"),
            "normalization": str(normalization),
        },
        "source": source,
        "selection": {
            "objective": "fold8_uncalibrated_macro_roc_auc",
            "architecture_mean_macro_auroc": 0.81,
            "frozen_epochs": frozen_epochs,
            "epoch_budget_rule": (
                "max(warmup_epochs+1,median("
                "selected_zero_based_best_epoch+1_across_seeds))"
            ),
        },
        "model": {"architecture": architecture, "preset": "smoke"},
        "model_identity": model_identity,
        "loader": {
            "batch_size": 16,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        },
        "optimization": {
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "warmup_epochs": 1,
            "minimum_lr_ratio": 0.1,
            "gradient_clip_norm": 1.0,
            "scheduler": "warmup_cosine",
        },
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8},
        "runtime": {"seed": seed, "device": "cpu", "bf16": False},
        "output": {"root_dir": str(tmp_path / "runs")},
        "downstream_provenance": {
            "project_root": str(tmp_path),
            "code_revision": "synthetic",
            "dependency_lock_sha256": canonical_sha256({"lock": 1}),
        },
    }
    recipe_hash = canonical_sha256(recipe_template)
    recipe_template["recipe_sha256"] = recipe_hash
    materialized_recipe = json.loads(json.dumps(recipe_template))
    materialized_recipe["freeze_artifact"] = str(freeze)
    materialized_recipe["freeze_artifact_sha256"] = freeze_hash
    selection: dict[str, object] = {
        "checkpoint": str(source_checkpoint),
        "checkpoint_sha256": _file_hash(source_checkpoint).removeprefix("sha256:"),
        "checkpoint_config_hash": source_config_hash,
        "selected_epoch": 4,
        "selected_epoch_count": 5,
        "selected_macro_auroc": 0.81,
        "source_seed": seed,
        "member_completion_sha256": source_completion_hash,
        "freeze_artifact_sha256": freeze_hash,
        "recipe_sha256": recipe_hash,
    }
    config: dict[str, object] = dict(materialized_recipe)
    config["model"] = model_identity
    config["selection_provenance"] = selection
    config["freeze_binding"] = {
        "path": str(freeze),
        "artifact_sha256": freeze_hash,
        "comparison_id": "comparison-v1",
        "recipe_sha256": recipe_hash,
    }
    config["attempt_index"] = 0
    config["effective_data"] = {"refit_records": 100}
    config["checkpoint_roles"] = {
        "best_training_loss.ckpt": "diagnostic minimum training loss only",
        "last.ckpt": "crash-recovery state from the latest completed epoch",
        "final.ckpt": "authoritative frozen-epoch refit artifact",
    }
    config_hash = canonical_sha256(config)
    resolved = run_dir / "resolved_refit_config.json"
    resolved.write_text(
        json.dumps({"config_hash": config_hash, "config": config}),
        encoding="utf-8",
    )
    resolved = resolved.resolve()
    def entry(path: Path, *, with_config: bool = False) -> dict[str, object]:
        result: dict[str, object] = {"path": str(path), "sha256": _file_hash(path)}
        if with_config:
            result["config_hash"] = config_hash
        return result

    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "ecg_trust.refit_completion",
        "comparison_id": "comparison-v1",
        "architecture": architecture,
        "seed": seed,
        "status": "complete",
        "run_name": materialized_recipe["run_name"],
        "run_dir": str(run_dir.resolve()),
        "freeze_artifact_path": str(freeze),
        "freeze_artifact_sha256": freeze_hash,
        "recipe_sha256": recipe_hash,
        "source_member_completion_path": str(source_completion),
        "source_member_completion_sha256": source_completion_hash,
        "refit_folds": list(range(1, 9)),
        "normalization_folds": list(range(1, 8)),
        "frozen_epochs": frozen_epochs,
        "protocol_hash": ExperimentProtocol.canonical().protocol_hash,
        "manifest_hash": manifest_hash,
        "normalization_hash": normalization_hash,
        "selection_provenance": selection,
        "selection_lineage_sha256": canonical_sha256(selection),
        "files": {
            "final_checkpoint": entry(checkpoint),
            "resolved_config": entry(resolved, with_config=True),
            "metadata": entry(metadata),
            "protocol": entry(protocol_file),
            "history": entry(history),
            "manifest": entry(manifest),
            "normalization": entry(normalization),
            "source_checkpoint": entry(source_checkpoint),
        },
    }
    body["artifact_sha256"] = canonical_sha256(body)
    completion = run_dir / "refit_completion.json"
    completion.write_text(json.dumps(body), encoding="utf-8")
    return completion.resolve(), body, recipe_template


@pytest.fixture
def synthetic_completions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[Path], dict[str, dict[str, object]], _FakeFreeze]:
    manifest = _write(tmp_path / "manifest.parquet", "shared manifest")
    normalization = _write(tmp_path / "normalization.json", "shared normalization")
    freeze = _write(tmp_path / "freeze.json", "shared freeze")
    completions: list[Path] = []
    payloads: dict[str, dict[str, object]] = {}
    recipes: dict[tuple[str, int], dict[str, object]] = {}
    for architecture in EXPECTED_ARCHITECTURES:
        for seed in EXPECTED_SEEDS:
            path, payload, recipe = _completion_fixture(
                tmp_path,
                architecture=architecture,
                seed=seed,
                manifest=manifest,
                normalization=normalization,
                freeze=freeze,
            )
            completions.append(path)
            payloads[str(path)] = payload
            recipes[(architecture, seed)] = recipe

    fake_freeze = _FakeFreeze(
        path=freeze,
        manifest_hash=_file_hash(manifest),
        normalization_hash=_file_hash(normalization),
        recipes=recipes,
    )

    def fake_completion(
        path: str | Path,
        *,
        protocol: ExperimentProtocol,
        verify_sources: bool = True,
    ) -> dict[str, object]:
        del protocol, verify_sources
        return payloads[str(Path(path).resolve())]

    monkeypatch.setattr(release_gates, "load_refit_completion", fake_completion)
    monkeypatch.setattr(
        release_gates,
        "load_multiseed_freeze",
        lambda path, protocol, verify_sources=True: fake_freeze,
    )
    return completions, payloads, fake_freeze


def test_refit_bundle_requires_and_binds_exact_six_receipts(
    tmp_path: Path,
    synthetic_completions: tuple[
        list[Path], dict[str, dict[str, object]], _FakeFreeze
    ],
) -> None:
    completions, _, _ = synthetic_completions
    protocol = ExperimentProtocol.canonical()
    with pytest.raises(ReleaseGateError, match="exactly six"):
        create_refit_bundle(completions[:-1], protocol=protocol)

    bundle = create_refit_bundle(
        list(reversed(completions)),
        protocol=protocol,
        created_at_utc="2026-08-08T12:00:00+00:00",
    )
    path, digest = save_refit_bundle(bundle, tmp_path / "refit-bundle.json")
    loaded = load_refit_bundle(path, protocol=protocol, verify_sources=False)

    assert loaded.artifact_sha256 == digest
    assert len(loaded.members) == 6
    assert {member.completion_sha256 for member in loaded.members} == {
        str(payload["artifact_sha256"])
        for _, payload in synthetic_completions[1].items()
    }
    assert {
        member.frozen_epochs
        for member in loaded.members
        if member.architecture == "resnet1d"
    } == {12}


def test_refit_bundle_tamper_is_detected(
    tmp_path: Path,
    synthetic_completions: tuple[
        list[Path], dict[str, dict[str, object]], _FakeFreeze
    ],
) -> None:
    completions, _, _ = synthetic_completions
    protocol = ExperimentProtocol.canonical()
    bundle = create_refit_bundle(completions, protocol=protocol)
    path, _ = save_refit_bundle(bundle, tmp_path / "refit-bundle.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["members"][0]["seed"] = 9999
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReleaseIntegrityError, match="artifact hash"):
        load_refit_bundle(path, protocol=protocol, verify_sources=False)


@pytest.mark.parametrize(
    ("file_key", "receipt_key", "replacement"),
    (
        ("manifest", "manifest_hash", "drifted manifest"),
        ("normalization", "normalization_hash", "drifted normalization"),
    ),
)
def test_consistent_data_receipt_drift_cannot_move_freeze_root(
    synthetic_completions: tuple[
        list[Path], dict[str, dict[str, object]], _FakeFreeze
    ],
    file_key: str,
    receipt_key: str,
    replacement: str,
) -> None:
    completions, payloads, _ = synthetic_completions
    first_files = payloads[str(completions[0])]["files"]
    assert isinstance(first_files, dict)
    first_entry = first_files[file_key]
    assert isinstance(first_entry, dict)
    shared_path = Path(str(first_entry["path"]))
    shared_path.write_text(replacement, encoding="utf-8")
    drifted_hash = _file_hash(shared_path)
    for payload in payloads.values():
        files = payload["files"]
        assert isinstance(files, dict)
        entry = files[file_key]
        assert isinstance(entry, dict)
        entry["sha256"] = drifted_hash
        payload[receipt_key] = drifted_hash
        payload.pop("artifact_sha256")
        payload["artifact_sha256"] = canonical_sha256(payload)

    with pytest.raises(ReleaseIntegrityError, match="freeze root"):
        create_refit_bundle(completions, protocol=ExperimentProtocol.canonical())


def test_self_valid_source_receipt_semantic_drift_is_rejected(
    synthetic_completions: tuple[
        list[Path], dict[str, dict[str, object]], _FakeFreeze
    ],
) -> None:
    completions, payloads, freeze = synthetic_completions
    payload = payloads[str(completions[0])]
    architecture = str(payload["architecture"])
    raw_seed = payload["seed"]
    assert isinstance(raw_seed, int) and not isinstance(raw_seed, bool)
    seed = raw_seed
    source_path = Path(str(payload["source_member_completion_path"]))
    source_receipt = json.loads(source_path.read_text(encoding="utf-8"))
    source_receipt["best_epoch"] = 6
    source_receipt.pop("artifact_sha256")
    source_receipt["artifact_sha256"] = canonical_sha256(source_receipt)
    source_path.write_text(
        json.dumps(source_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    drifted_file_hash = _file_hash(source_path)
    payload["source_member_completion_sha256"] = drifted_file_hash
    recipe = freeze.recipes[(architecture, seed)]
    recipe_source = recipe["source"]
    assert isinstance(recipe_source, dict)
    recipe_source["member_completion_sha256"] = drifted_file_hash
    recipe.pop("recipe_sha256")
    recipe["recipe_sha256"] = canonical_sha256(recipe)
    payload["recipe_sha256"] = recipe["recipe_sha256"]
    payload.pop("artifact_sha256")
    payload["artifact_sha256"] = canonical_sha256(payload)

    with pytest.raises(ReleaseIntegrityError, match="best_epoch.*freeze recipe"):
        create_refit_bundle(completions, protocol=ExperimentProtocol.canonical())


def test_fold9_export_rechecks_bound_receipts_before_exporter_call(
    tmp_path: Path,
    synthetic_completions: tuple[
        list[Path], dict[str, dict[str, object]], _FakeFreeze
    ],
) -> None:
    completions, payloads, _ = synthetic_completions
    protocol = ExperimentProtocol.canonical()
    bundle = create_refit_bundle(completions, protocol=protocol)
    path, _ = save_refit_bundle(bundle, tmp_path / "refit-bundle.json")
    first = payloads[str(completions[0])]
    source_path = Path(str(first["source_member_completion_path"]))
    source_path.write_text("tampered", encoding="utf-8")
    called = False

    def forbidden_exporter(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("exporter must not be called")

    with pytest.raises(ReleaseIntegrityError, match="source member completion"):
        export_fold9_predictions(
            path,
            tmp_path / "fold9",
            protocol=protocol,
            exporter=forbidden_exporter,  # type: ignore[arg-type]
        )
    assert called is False


def test_calibration_batch_rejects_non_fold9_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_completions: tuple[
        list[Path], dict[str, dict[str, object]], _FakeFreeze
    ],
) -> None:
    completions, _, _ = synthetic_completions
    protocol = ExperimentProtocol.canonical()
    unsigned = create_refit_bundle(completions, protocol=protocol)
    bundle_path, _ = save_refit_bundle(unsigned, tmp_path / "refit-bundle.json")
    bundle = load_refit_bundle(bundle_path, protocol=protocol, verify_sources=False)
    first = bundle.members[0]
    development = create_prediction_artifact(
        ecg_id=np.asarray([1, 2]),
        patient_id=np.asarray([10, 20]),
        strat_fold=np.asarray([8, 8]),
        targets=np.asarray([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]),
        raw_logits=np.zeros((2, 5)),
        model_name=first.run_name,
        model_seed=first.seed,
        protocol=protocol,
        config_hash=first.resolved_config_hash,
        manifest_hash=first.manifest_sha256,
        fold_role=FoldRole.MODEL_SELECTION,
    )
    prediction_path = tmp_path / "fold8.npz"
    save_prediction_artifact(development, prediction_path, protocol=protocol)
    paths = {member.member_id: prediction_path for member in bundle.members}
    monkeypatch.setattr(release_gates, "load_refit_bundle", lambda *args, **kwargs: bundle)

    with pytest.raises(ReleaseGateError, match="fold-9"):
        fit_calibration_bundle(
            bundle_path,
            paths,
            tmp_path / "decisions",
            protocol=protocol,
        )


def test_calibration_reload_recomputes_decision_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_completions: tuple[
        list[Path], dict[str, dict[str, object]], _FakeFreeze
    ],
) -> None:
    completions, _, _ = synthetic_completions
    protocol = ExperimentProtocol.canonical()
    refit = create_refit_bundle(completions, protocol=protocol)
    refit_path, _ = save_refit_bundle(refit, tmp_path / "refit-bundle.json")
    refit = load_refit_bundle(refit_path, protocol=protocol, verify_sources=False)
    prediction_paths: dict[str, Path] = {}
    targets = np.asarray(
        [
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
        ],
        dtype=np.int8,
    )
    for index, member in enumerate(refit.members):
        prediction = create_prediction_artifact(
            ecg_id=np.asarray([1, 2, 3, 4]),
            patient_id=np.asarray([11, 12, 13, 14]),
            strat_fold=np.asarray([9, 9, 9, 9]),
            targets=targets,
            raw_logits=np.asarray(
                [[-1.0, 1.0, -0.5, 0.5, -0.2], [1.0, -1.0, 0.5, -0.5, 0.2]]
                * 2
            )
            + index * 0.01,
            model_name=member.run_name,
            model_seed=member.seed,
            protocol=protocol,
            config_hash=member.resolved_config_hash,
            manifest_hash=member.manifest_sha256,
            fold_role=FoldRole.CALIBRATION,
            extra_metadata={
                "lineage": "frozen_refit",
                "checkpoint_sha256": member.final_checkpoint_sha256,
                "normalization_sha256": member.normalization_sha256,
            },
        )
        prediction_path = tmp_path / "fold9" / f"{member.member_id}.npz"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        save_prediction_artifact(prediction, prediction_path, protocol=protocol)
        prediction_paths[member.member_id] = prediction_path
    calibration = fit_calibration_bundle(
        refit_path,
        prediction_paths,
        tmp_path / "decisions",
        protocol=protocol,
        coverage_targets=(1.0, 0.5),
    )
    calibration_path, _ = save_calibration_bundle(
        calibration, tmp_path / "calibration-bundle.json"
    )
    load_calibration_bundle(calibration_path, protocol=protocol, verify_sources=True)

    first = calibration.members[0]
    original = load_calibration_decisions(first.decision_path, protocol=protocol)
    drifted = replace(
        original,
        temperature_scaling=replace(
            original.temperature_scaling,
            temperature=original.temperature_scaling.temperature + 0.25,
        ),
    )
    real_loader = load_calibration_decisions

    def semantic_drift(path: str | Path, *, protocol: ExperimentProtocol) -> object:
        if Path(path).resolve() == first.decision_path.resolve():
            return drifted
        return real_loader(path, protocol=protocol)

    monkeypatch.setattr(release_gates, "load_calibration_decisions", semantic_drift)
    with pytest.raises(ReleaseIntegrityError, match="temperature differs"):
        load_calibration_bundle(
            calibration_path,
            protocol=protocol,
            verify_sources=True,
        )
