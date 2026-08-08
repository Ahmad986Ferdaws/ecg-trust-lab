from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from ecg_trust.final_batch import (
    FinalBatchSettings,
    authorize_ledgered_final_test,
    build_final_batch_plan,
    create_final_opening_ledger,
    load_final_opening_ledger,
    open_or_resume_final_batch,
)
from ecg_trust.protocol import FINAL_TEST_CONFIRMATION, ExperimentProtocol
from ecg_trust.release_gates import (
    EXPECTED_ARCHITECTURES,
    EXPECTED_SEEDS,
    CalibrationBundle,
    CalibrationMember,
    RefitBundle,
    RefitMember,
    ReleaseGateError,
    ReleaseIntegrityError,
    ReleaseStateError,
    materialize_demo_policy_payload,
)


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _raw_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _bundles(tmp_path: Path) -> tuple[RefitBundle, CalibrationBundle]:
    protocol = ExperimentProtocol.canonical()
    manifest_hash = _hash("manifest")
    normalization_hash = _hash("normalization")
    refit_members: list[RefitMember] = []
    calibration_members: list[CalibrationMember] = []
    for architecture in EXPECTED_ARCHITECTURES:
        for seed in EXPECTED_SEEDS:
            member_id = f"{architecture}-seed{seed}"
            selection = {
                "freeze_artifact_sha256": _hash("freeze"),
                "recipe_sha256": _hash("recipe:" + member_id),
                "member_completion_sha256": _hash("development:" + member_id),
            }
            refit_lineage = _hash("refit-lineage:" + member_id)
            checkpoint_hash = _hash("checkpoint:" + member_id)
            config_hash = _hash("config:" + member_id)
            dummy = tmp_path / member_id
            refit_members.append(
                RefitMember(
                    member_id=member_id,
                    comparison_id="comparison-v1",
                    architecture=architecture,
                    seed=seed,
                    run_name=f"{member_id}-refit",
                    run_dir=dummy,
                    completion_path=dummy / "refit_completion.json",
                    completion_sha256=_hash("completion:" + member_id),
                    freeze_artifact_path=tmp_path / "freeze.json",
                    freeze_artifact_sha256=_hash("freeze"),
                    recipe_sha256=_hash("recipe:" + member_id),
                    source_member_completion_path=dummy / "member_completion.json",
                    source_member_completion_sha256=_hash(
                        "development:" + member_id
                    ),
                    final_checkpoint_path=dummy / "final.ckpt",
                    final_checkpoint_sha256=checkpoint_hash,
                    resolved_config_path=dummy / "resolved_refit_config.json",
                    resolved_config_file_sha256=_hash("resolved-file:" + member_id),
                    resolved_config_hash=config_hash,
                    metadata_path=dummy / "refit_metadata.json",
                    metadata_sha256=_hash("metadata:" + member_id),
                    protocol_path=dummy / "protocol.json",
                    protocol_file_sha256=_hash("protocol-file:" + member_id),
                    history_path=dummy / "refit_history.jsonl",
                    history_sha256=_hash("history:" + member_id),
                    protocol_hash=protocol.protocol_hash,
                    manifest_path=tmp_path / "manifest.parquet",
                    manifest_sha256=manifest_hash,
                    normalization_path=tmp_path / "normalization.json",
                    normalization_sha256=normalization_hash,
                    source_checkpoint_path=dummy / "source.ckpt",
                    source_checkpoint_sha256=_hash("source-checkpoint:" + member_id),
                    frozen_epochs=12 if architecture == "resnet1d" else 15,
                    selection_provenance=selection,
                    selection_lineage_sha256=_hash("selection:" + member_id),
                    lineage_sha256=refit_lineage,
                )
            )
            calibration_members.append(
                CalibrationMember(
                    member_id=member_id,
                    architecture=architecture,
                    seed=seed,
                    model_name=f"{member_id}-refit",
                    refit_lineage_sha256=refit_lineage,
                    checkpoint_path=dummy / "final.ckpt",
                    resolved_config_hash=config_hash,
                    checkpoint_sha256=checkpoint_hash,
                    resolved_config_path=dummy / "resolved_refit_config.json",
                    resolved_config_file_sha256=_hash(
                        "resolved-file:" + member_id
                    ),
                    normalization_path=tmp_path / "normalization.json",
                    normalization_sha256=normalization_hash,
                    prediction_path=dummy / "fold9.npz",
                    prediction_sidecar_path=dummy / "fold9.json",
                    prediction_npz_sha256=_raw_hash("prediction:" + member_id),
                    prediction_sidecar_sha256=_raw_hash("sidecar:" + member_id),
                    prediction_artifact_sha256=_hash("prediction-artifact:" + member_id),
                    prediction_alignment_sha256=_hash("alignment"),
                    decision_path=dummy / "decision.json",
                    decision_file_sha256=_raw_hash("decision-file:" + member_id),
                    decision_artifact_sha256=_hash("decision:" + member_id),
                    temperature=1.0 + seed / 100_000,
                    thresholds=(0.1, 0.2, 0.3, 0.4, 0.5),
                    entropy_gates=(
                        {
                            "target_coverage": 0.8,
                            "maximum_entropy": 0.5,
                            "calibration_coverage": 0.8,
                            "selected_count": 80,
                            "calibration_count": 100,
                        },
                    ),
                    independent_fit_sha256=_hash("fit:" + member_id),
                )
            )
    refit_bundle = RefitBundle(
        protocol_hash=protocol.protocol_hash,
        manifest_sha256=manifest_hash,
        normalization_sha256=normalization_hash,
        label_order=protocol.label_order,
        members=tuple(refit_members),
        created_at_utc="2026-08-08T12:00:00+00:00",
        artifact_sha256=_hash("refit-bundle"),
    )
    calibration_bundle = CalibrationBundle(
        refit_bundle_sha256=_hash("refit-bundle"),
        protocol_hash=protocol.protocol_hash,
        manifest_sha256=manifest_hash,
        normalization_sha256=normalization_hash,
        label_order=protocol.label_order,
        members=tuple(calibration_members),
        created_at_utc="2026-08-08T13:00:00+00:00",
        artifact_sha256=_hash("calibration-bundle"),
    )
    return refit_bundle, calibration_bundle


def _settings(tmp_path: Path, *, output_name: str = "final") -> FinalBatchSettings:
    subgroup = tmp_path / "subgroups.json"
    if not subgroup.exists():
        subgroup.write_text(
            json.dumps({"ecg_id": [1], "attributes": {"sex": ["female"]}}),
            encoding="utf-8",
        )
    return FinalBatchSettings.create(
        output_directory=tmp_path / output_name,
        subgroup_path=subgroup,
        bootstrap_resamples=10,
        bootstrap_minimum_valid=5,
    )


def test_exact_six_plan_is_frozen_and_contains_no_retuning(
    tmp_path: Path,
) -> None:
    refits, calibrations = _bundles(tmp_path)
    plan = build_final_batch_plan(refits, calibrations, _settings(tmp_path))

    assert len(plan.members) == 6
    assert plan.to_payload()["retuning_allowed"] is False
    assert plan.settings.to_payload()["retuning_allowed"] is False
    assert {
        (member["architecture"], member["seed"]) for member in plan.members
    } == {
        (architecture, seed)
        for architecture in EXPECTED_ARCHITECTURES
        for seed in EXPECTED_SEEDS
    }

    incomplete = replace(refits, members=refits.members[:-1])
    with pytest.raises(ReleaseGateError, match="exact six"):
        build_final_batch_plan(incomplete, calibrations, _settings(tmp_path))

    policy = materialize_demo_policy_payload(
        calibrations, "resnet1d-seed2026", target_coverage=0.8
    )
    provenance = policy["provenance"]
    assert isinstance(provenance, dict)
    assert provenance["calibration_folds"] == [9]
    assert not str(provenance["checkpoint_sha256"]).startswith("sha256:")


def test_ledger_must_exist_before_final_authorization(tmp_path: Path) -> None:
    protocol = ExperimentProtocol.canonical()
    refits, calibrations = _bundles(tmp_path)
    plan = build_final_batch_plan(refits, calibrations, _settings(tmp_path))
    ledger_path = tmp_path / "opening-ledger.json"

    with pytest.raises(ReleaseIntegrityError, match="missing"):
        authorize_ledgered_final_test(
            ledger_path,
            plan,
            protocol=protocol,
            purpose="one-time preregistered final evaluation",
            confirmation=FINAL_TEST_CONFIRMATION,
        )

    ledger = create_final_opening_ledger(
        ledger_path,
        plan,
        purpose="one-time preregistered final evaluation",
        operator="synthetic-test",
        confirmation=FINAL_TEST_CONFIRMATION,
        created_at_utc="2026-08-08T14:00:00+00:00",
    )
    assert ledger_path.is_file()
    assert ledger.events[0]["event"] == "ledger_created_before_fold10_access"
    token = authorize_ledgered_final_test(
        ledger_path,
        plan,
        protocol=protocol,
        purpose="one-time preregistered final evaluation",
        confirmation=FINAL_TEST_CONFIRMATION,
    )
    assert token.purpose == "one-time preregistered final evaluation"


def test_opening_is_one_time_and_resume_requires_identical_batch(tmp_path: Path) -> None:
    protocol = ExperimentProtocol.canonical()
    refits, calibrations = _bundles(tmp_path)
    plan = build_final_batch_plan(refits, calibrations, _settings(tmp_path))
    ledger_path = tmp_path / "opening-ledger.json"
    common = {
        "protocol": protocol,
        "purpose": "one-time preregistered final evaluation",
        "operator": "synthetic-test",
        "confirmation": FINAL_TEST_CONFIRMATION,
    }
    created = open_or_resume_final_batch(
        ledger_path, plan, resume=False, **common
    )
    resumed = open_or_resume_final_batch(
        ledger_path, plan, resume=True, **common
    )
    assert resumed.ledger_sha256 == created.ledger_sha256

    with pytest.raises(ReleaseStateError, match="already exists"):
        open_or_resume_final_batch(ledger_path, plan, resume=False, **common)
    changed_plan = build_final_batch_plan(
        refits, calibrations, _settings(tmp_path, output_name="different-output")
    )
    with pytest.raises(ReleaseStateError, match="differs"):
        open_or_resume_final_batch(
            ledger_path, changed_plan, resume=True, **common
        )


def test_opening_ledger_tamper_is_detected(tmp_path: Path) -> None:
    protocol = ExperimentProtocol.canonical()
    refits, calibrations = _bundles(tmp_path)
    plan = build_final_batch_plan(refits, calibrations, _settings(tmp_path))
    ledger_path = tmp_path / "opening-ledger.json"
    create_final_opening_ledger(
        ledger_path,
        plan,
        purpose="one-time preregistered final evaluation",
        operator="synthetic-test",
        confirmation=FINAL_TEST_CONFIRMATION,
    )
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    payload["opening"]["operator"] = "tampered"
    ledger_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReleaseIntegrityError, match="ledger hash"):
        load_final_opening_ledger(ledger_path, protocol=protocol)


def test_canonical_marker_blocks_second_ledger_and_output_choice(
    tmp_path: Path,
) -> None:
    protocol = ExperimentProtocol.canonical()
    refits, calibrations = _bundles(tmp_path)
    first = build_final_batch_plan(refits, calibrations, _settings(tmp_path))
    common = {
        "protocol": protocol,
        "purpose": "one-time preregistered final evaluation",
        "operator": "synthetic-test",
        "confirmation": FINAL_TEST_CONFIRMATION,
    }
    open_or_resume_final_batch(
        tmp_path / "first-ledger.json", first, resume=False, **common
    )
    second = build_final_batch_plan(
        refits,
        calibrations,
        _settings(tmp_path, output_name="alternate-final-output"),
    )

    assert second.opening_marker_path == first.opening_marker_path
    with pytest.raises(ReleaseStateError, match="already opened"):
        open_or_resume_final_batch(
            tmp_path / "second-ledger.json", second, resume=False, **common
        )


def test_opening_rejects_final_artifacts_that_predate_ledger(tmp_path: Path) -> None:
    refits, calibrations = _bundles(tmp_path)
    plan = build_final_batch_plan(refits, calibrations, _settings(tmp_path))
    first_prediction = Path(str(plan.members[0]["final_prediction_path"]))
    first_prediction.parent.mkdir(parents=True, exist_ok=True)
    first_prediction.write_bytes(b"predates opening")

    with pytest.raises(ReleaseStateError, match="pre-existing"):
        create_final_opening_ledger(
            tmp_path / "opening-ledger.json",
            plan,
            purpose="pre-existing artifact defense",
            operator="synthetic-test",
            confirmation=FINAL_TEST_CONFIRMATION,
        )
    assert not plan.opening_marker_path.exists()
