from __future__ import annotations

import hashlib
import json
import os
import socket
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import ecg_trust.final_batch as final_batch
from ecg_trust.final_batch import (
    FinalBatchSettings,
    authorize_ledgered_final_test,
    build_final_batch_plan,
    create_final_opening_ledger,
    load_final_opening_ledger,
    open_or_resume_final_batch,
)
from ecg_trust.protocol import (
    FINAL_TEST_CONFIRMATION,
    ExperimentProtocol,
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
    ReleaseGateError,
    ReleaseIntegrityError,
    ReleaseStateError,
    materialize_demo_policy_payload,
)


@pytest.fixture(autouse=True)
def _stub_verified_final_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    def load(path: Path, *, protocol_hash: str) -> dict[str, object]:
        del protocol_hash
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        assert isinstance(value, dict)
        return value

    monkeypatch.setattr(final_batch, "_load_verified_final_evaluation_spec", load)


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _raw_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _canonical_hash(payload: dict[str, object]) -> str:
    serialized = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


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
            dummy = tmp_path / member_id
            dummy.mkdir(parents=True, exist_ok=True)
            resolved_config = {"loader": {"batch_size": 64, "num_workers": 0}}
            config_hash = _canonical_hash(resolved_config)
            resolved_path = dummy / "resolved_refit_config.json"
            resolved_path.write_text(
                json.dumps(
                    {"config_hash": config_hash, "config": resolved_config},
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            resolved_file_hash = "sha256:" + hashlib.sha256(
                resolved_path.read_bytes()
            ).hexdigest()
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
                    resolved_config_path=resolved_path,
                    resolved_config_file_sha256=resolved_file_hash,
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
                    resolved_config_path=resolved_path,
                    resolved_config_file_sha256=resolved_file_hash,
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
    subgroup_path = tmp_path / "subgroups.json"
    if not subgroup_path.exists():
        subgroup_path.write_text(
            json.dumps({"ecg_id": [1], "attributes": {"sex": ["female"]}}),
            encoding="utf-8",
        )
    deviations_path = tmp_path / "PROTOCOL_DEVIATIONS.md"
    deviations_path.write_text("Synthetic protocol deviations.\n", encoding="utf-8")
    spec_body: dict[str, object] = {
        "refit_bundle": {
            "artifact_sha256": refit_bundle.artifact_sha256,
            "manifest_sha256": refit_bundle.manifest_sha256,
        },
        "subgroup_artifact": {
            "path": str(subgroup_path.resolve()),
            "file_sha256": "sha256:"
            + hashlib.sha256(subgroup_path.read_bytes()).hexdigest(),
        },
        "protocol_deviations": {
            "path": str(deviations_path.resolve()),
            "file_sha256": "sha256:"
            + hashlib.sha256(deviations_path.read_bytes()).hexdigest(),
            "required_in_final_reporting": True,
        },
        "final_evaluation": {
            "final_folds": [10],
            "patient_resampling": "patient_cluster_percentile_bootstrap",
            "bootstrap_resamples": 10,
            "bootstrap_base_seed": 20_260_808,
            "bootstrap_confidence": 0.95,
            "bootstrap_minimum_valid": 5,
            "bootstrap_seed_strategy": "base_plus_model_seed",
            "ece_bins": 15,
            "minimum_group_samples": 30,
            "minimum_group_patients": 20,
            "retuning_allowed": False,
        },
        "runtime_envelope": {
            "project_root": str(tmp_path.resolve()),
            "hardware": {
                "requested_device": "cuda:0",
                "resolved_device": "cuda:0",
                "bf16_supported": True,
            }
        },
    }
    spec_payload = dict(spec_body)
    spec_payload["artifact_sha256"] = _canonical_hash(spec_body)
    spec_path = tmp_path / "final-evaluation-spec.json"
    spec_path.write_text(json.dumps(spec_payload, sort_keys=True), encoding="utf-8")
    spec_binding = {
        "path": str(spec_path.resolve()),
        "file_sha256": "sha256:"
        + hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "artifact_sha256": spec_payload["artifact_sha256"],
    }
    calibration_bundle = CalibrationBundle(
        refit_bundle_sha256=_hash("refit-bundle"),
        protocol_hash=protocol.protocol_hash,
        manifest_sha256=manifest_hash,
        normalization_sha256=normalization_hash,
        label_order=protocol.label_order,
        members=tuple(calibration_members),
        created_at_utc="2026-08-08T13:00:00+00:00",
        artifact_sha256=_hash("calibration-bundle"),
        stage_provenance={"final_evaluation_spec": spec_binding},
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
        device="cuda:0",
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
    ledger_path = final_batch.canonical_final_ledger_path(plan)

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
    ledger_path = final_batch.canonical_final_ledger_path(plan)
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
    ledger_path = final_batch.canonical_final_ledger_path(plan)
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
        final_batch.canonical_final_ledger_path(first),
        first,
        resume=False,
        **common,
    )
    second = build_final_batch_plan(
        refits,
        calibrations,
        _settings(tmp_path, output_name="alternate-final-output"),
    )

    assert second.opening_marker_path == first.opening_marker_path
    with pytest.raises(ReleaseStateError, match="already exists|already opened"):
        open_or_resume_final_batch(
            final_batch.canonical_final_ledger_path(second),
            second,
            resume=False,
            **common,
        )


def test_opening_rejects_final_artifacts_that_predate_ledger(tmp_path: Path) -> None:
    refits, calibrations = _bundles(tmp_path)
    plan = build_final_batch_plan(refits, calibrations, _settings(tmp_path))
    first_prediction = Path(str(plan.members[0]["final_prediction_path"]))
    first_prediction.parent.mkdir(parents=True, exist_ok=True)
    first_prediction.write_bytes(b"predates opening")

    with pytest.raises(ReleaseStateError, match="pre-existing"):
        create_final_opening_ledger(
            final_batch.canonical_final_ledger_path(plan),
            plan,
            purpose="pre-existing artifact defense",
            operator="synthetic-test",
            confirmation=FINAL_TEST_CONFIRMATION,
        )
    assert not plan.opening_marker_path.exists()


def test_resume_recovers_crash_after_pending_ledger_before_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = ExperimentProtocol.canonical()
    refits, calibrations = _bundles(tmp_path)
    plan = build_final_batch_plan(refits, calibrations, _settings(tmp_path))
    ledger_path = final_batch.canonical_final_ledger_path(plan)
    common = {
        "protocol": protocol,
        "purpose": "one-time preregistered final evaluation",
        "operator": "synthetic-test",
        "confirmation": FINAL_TEST_CONFIRMATION,
    }
    real_ensure = final_batch._ensure_canonical_opening_marker

    def crash_before_marker(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("injected crash before marker")

    monkeypatch.setattr(
        final_batch, "_ensure_canonical_opening_marker", crash_before_marker
    )
    with pytest.raises(RuntimeError, match="before marker"):
        open_or_resume_final_batch(
            ledger_path, plan, resume=False, **common  # type: ignore[arg-type]
        )

    pending = load_final_opening_ledger(ledger_path, protocol=protocol)
    assert pending.state == "opening_pending"
    assert not plan.opening_marker_path.exists()
    with pytest.raises(ReleaseStateError, match="not durably open"):
        authorize_ledgered_final_test(
            ledger_path,
            plan,
            protocol=protocol,
            purpose=str(common["purpose"]),
            confirmation=FINAL_TEST_CONFIRMATION,
        )
    with pytest.raises(ReleaseStateError, match="already exists"):
        open_or_resume_final_batch(
            ledger_path, plan, resume=False, **common  # type: ignore[arg-type]
        )

    changed = build_final_batch_plan(
        refits, calibrations, _settings(tmp_path, output_name="changed-final")
    )
    monkeypatch.setattr(final_batch, "_ensure_canonical_opening_marker", real_ensure)
    with pytest.raises(ReleaseStateError, match="differs"):
        open_or_resume_final_batch(
            ledger_path, changed, resume=True, **common  # type: ignore[arg-type]
        )
    assert not plan.opening_marker_path.exists()

    unauthorized = Path(str(plan.members[0]["final_prediction_path"]))
    unauthorized.parent.mkdir(parents=True, exist_ok=True)
    unauthorized.write_bytes(b"appeared while opening was pending")
    with pytest.raises(ReleaseStateError, match="pending opening"):
        open_or_resume_final_batch(
            ledger_path, plan, resume=True, **common  # type: ignore[arg-type]
        )
    assert not plan.opening_marker_path.exists()
    assert load_final_opening_ledger(ledger_path, protocol=protocol).state == (
        "opening_pending"
    )
    unauthorized.unlink()

    recovered = open_or_resume_final_batch(
        ledger_path, plan, resume=True, **common  # type: ignore[arg-type]
    )
    assert recovered.state == "open"
    assert plan.opening_marker_path.is_file()
    assert [event["event"] for event in recovered.events] == [
        "ledger_created_before_fold10_access",
        "canonical_opening_marker_committed",
    ]


def test_resume_recovers_crash_after_marker_before_open_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = ExperimentProtocol.canonical()
    refits, calibrations = _bundles(tmp_path)
    plan = build_final_batch_plan(refits, calibrations, _settings(tmp_path))
    ledger_path = final_batch.canonical_final_ledger_path(plan)
    common = {
        "protocol": protocol,
        "purpose": "one-time preregistered final evaluation",
        "operator": "synthetic-test",
        "confirmation": FINAL_TEST_CONFIRMATION,
    }
    real_transition = final_batch._transition_pending_opening_to_open

    def crash_after_marker(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("injected crash after marker")

    monkeypatch.setattr(
        final_batch, "_transition_pending_opening_to_open", crash_after_marker
    )
    with pytest.raises(RuntimeError, match="after marker"):
        open_or_resume_final_batch(
            ledger_path, plan, resume=False, **common  # type: ignore[arg-type]
        )

    pending = load_final_opening_ledger(ledger_path, protocol=protocol)
    assert pending.state == "opening_pending"
    assert plan.opening_marker_path.is_file()

    monkeypatch.setattr(
        final_batch, "_transition_pending_opening_to_open", real_transition
    )
    recovered = open_or_resume_final_batch(
        ledger_path, plan, resume=True, **common  # type: ignore[arg-type]
    )
    assert recovered.state == "open"
    token = authorize_ledgered_final_test(
        ledger_path,
        plan,
        protocol=protocol,
        purpose=str(common["purpose"]),
        confirmation=FINAL_TEST_CONFIRMATION,
    )
    assert token.purpose == common["purpose"]


def test_pending_opening_cannot_be_bypassed_with_another_ledger_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = ExperimentProtocol.canonical()
    refits, calibrations = _bundles(tmp_path)
    plan = build_final_batch_plan(refits, calibrations, _settings(tmp_path))
    ledger_path = final_batch.canonical_final_ledger_path(plan)

    monkeypatch.setattr(
        final_batch,
        "_ensure_canonical_opening_marker",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("injected pre-marker crash")
        ),
    )
    with pytest.raises(RuntimeError, match="pre-marker"):
        open_or_resume_final_batch(
            ledger_path,
            plan,
            protocol=protocol,
            purpose="canonical reservation test",
            operator="synthetic-test",
            confirmation=FINAL_TEST_CONFIRMATION,
            resume=False,
        )
    alternate = tmp_path / "alternate-opening-ledger.json"
    with pytest.raises(ReleaseStateError, match="canonical release ledger"):
        open_or_resume_final_batch(
            alternate,
            plan,
            protocol=protocol,
            purpose="canonical reservation test",
            operator="synthetic-test",
            confirmation=FINAL_TEST_CONFIRMATION,
            resume=False,
        )
    assert ledger_path.is_file()
    assert not alternate.exists()
    assert not plan.opening_marker_path.exists()


def test_marker_rejects_rehashed_opening_intent_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = ExperimentProtocol.canonical()
    refits, calibrations = _bundles(tmp_path)
    plan = build_final_batch_plan(refits, calibrations, _settings(tmp_path))
    ledger_path = final_batch.canonical_final_ledger_path(plan)
    real_transition = final_batch._transition_pending_opening_to_open
    monkeypatch.setattr(
        final_batch,
        "_transition_pending_opening_to_open",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("injected post-marker crash")
        ),
    )
    with pytest.raises(RuntimeError, match="post-marker"):
        open_or_resume_final_batch(
            ledger_path,
            plan,
            protocol=protocol,
            purpose="immutable opening intent",
            operator="original-operator",
            confirmation=FINAL_TEST_CONFIRMATION,
            resume=False,
        )
    monkeypatch.setattr(
        final_batch, "_transition_pending_opening_to_open", real_transition
    )
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    opening = payload["opening"]
    opening["operator"] = "drifted-operator"
    intent = final_batch._opening_intent_sha256(
        plan=plan,
        ledger_path=ledger_path,
        purpose=opening["purpose"],
        operator=opening["operator"],
        confirmation_sha256=opening["confirmation_sha256"],
        created_at_utc=opening["created_at_utc"],
    )
    opening["opening_intent_sha256"] = intent
    payload["events"][0]["opening_intent_sha256"] = intent
    unhashed = dict(payload)
    del unhashed["ledger_sha256"]
    payload["ledger_sha256"] = _canonical_hash(unhashed)
    ledger_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ReleaseIntegrityError, match="opening marker differs"):
        load_final_opening_ledger(ledger_path, protocol=protocol)


def test_spec_identity_uses_one_registry_across_copied_spec_paths(
    tmp_path: Path,
) -> None:
    refits, calibrations = _bundles(tmp_path)
    first = build_final_batch_plan(refits, calibrations, _settings(tmp_path))
    provenance = dict(calibrations.stage_provenance or {})
    binding = dict(provenance["final_evaluation_spec"])
    original = Path(str(binding["path"]))
    copied = tmp_path / "copied" / original.name
    copied.parent.mkdir(parents=True)
    copied.write_bytes(original.read_bytes())
    binding["path"] = str(copied.resolve())
    provenance["final_evaluation_spec"] = binding
    copied_calibration = replace(calibrations, stage_provenance=provenance)
    second = build_final_batch_plan(
        refits,
        copied_calibration,
        _settings(tmp_path, output_name="alternate-final"),
    )

    assert first.opening_marker_path == second.opening_marker_path
    assert final_batch.canonical_final_ledger_path(first) == (
        final_batch.canonical_final_ledger_path(second)
    )


def _writer_lock_payload(*, pid: int, nonce: str) -> dict[str, object]:
    return {
        "schema_version": final_batch.FINAL_LEDGER_LOCK_SCHEMA_VERSION,
        "artifact_type": final_batch.FINAL_LEDGER_LOCK_TYPE,
        "host": socket.gethostname(),
        "pid": pid,
        "process_started_at": (
            final_batch.psutil.Process(pid).create_time()
            if pid == os.getpid()
            else 1.0
        ),
        "nonce": nonce,
    }


def test_ledger_writer_lock_rejects_live_owner(tmp_path: Path) -> None:
    ledger_path = tmp_path / "opening-ledger.json"
    lock_path = ledger_path.with_name(ledger_path.name + ".writer.lock")
    lock_path.write_text(
        json.dumps(_writer_lock_payload(pid=os.getpid(), nonce="live-owner")),
        encoding="utf-8",
    )

    with (
        pytest.raises(ReleaseStateError, match="active writer"),
        final_batch._ledger_writer_lock(ledger_path),
    ):
        raise AssertionError("a live owner must not be displaced")
    assert lock_path.exists()


def test_ledger_writer_lock_recovers_stale_owner_and_cleans_exact_nonce(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger_path = tmp_path / "opening-ledger.json"
    lock_path = ledger_path.with_name(ledger_path.name + ".writer.lock")
    lock_path.write_text(
        json.dumps(_writer_lock_payload(pid=123_456_789, nonce="stale-owner")),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        final_batch, "_process_identity_is_alive", lambda pid, started: False
    )

    with final_batch._ledger_writer_lock(ledger_path):
        replacement = json.loads(lock_path.read_text(encoding="utf-8"))
        assert replacement["pid"] == os.getpid()
        assert replacement["nonce"] != "stale-owner"
    assert not lock_path.exists()


def test_ledger_writer_lock_does_not_remove_replacement_owner(tmp_path: Path) -> None:
    ledger_path = tmp_path / "opening-ledger.json"
    lock_path = ledger_path.with_name(ledger_path.name + ".writer.lock")
    with final_batch._ledger_writer_lock(ledger_path):
        lock_path.write_text(
            json.dumps(_writer_lock_payload(pid=os.getpid(), nonce="replacement")),
            encoding="utf-8",
        )
    assert json.loads(lock_path.read_text(encoding="utf-8"))["nonce"] == "replacement"


@pytest.mark.parametrize("orphan_suffix", [".npz", ".json"])
def test_planned_partial_prediction_is_quarantined_before_safe_reexport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    orphan_suffix: str,
) -> None:
    protocol = ExperimentProtocol.canonical()
    refits, calibrations = _bundles(tmp_path)
    refit = refits.members[0]
    calibration = calibrations.members[0]
    prediction_path = tmp_path / "final" / "member.fold10.npz"
    prediction_path.parent.mkdir(parents=True)
    orphan_path = prediction_path.with_suffix(orphan_suffix)
    orphan_path.write_bytes(b"preserve this partial file")
    exported: list[Path] = []
    loaded_prediction = object()

    def fake_exporter(request: SimpleNamespace, **kwargs: object) -> object:
        del kwargs
        output = Path(request.output_path)
        exported.append(output)
        output.write_bytes(b"new complete npz")
        output.with_suffix(".json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(
            files=SimpleNamespace(
                npz_path=output,
                json_path=output.with_suffix(".json"),
                npz_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            ),
            lineage="frozen_refit",
            fold_role=FoldRole.FINAL_TEST,
            folds=(10,),
            model_name=refit.run_name,
            model_seed=refit.seed,
            checkpoint_sha256=refit.final_checkpoint_sha256,
            config_hash=refit.resolved_config_hash,
            manifest_hash=refit.manifest_sha256,
            normalization_sha256=refit.normalization_sha256,
            device="cuda:0",
            bf16_enabled=True,
        )

    monkeypatch.setattr(
        final_batch,
        "load_prediction_artifact",
        lambda *args, **kwargs: loaded_prediction,
    )
    monkeypatch.setattr(
        final_batch, "_validate_final_prediction", lambda *args, **kwargs: None
    )
    token = authorize_final_test_access(
        protocol,
        purpose="synthetic final recovery test",
        confirmation=FINAL_TEST_CONFIRMATION,
    )

    result = final_batch._export_or_resume_prediction(
        refit,
        calibration,
        prediction_path,
        state={"state": "planned"},
        settings=_settings(tmp_path),
        protocol=protocol,
        test_access=token,
        exporter=fake_exporter,  # type: ignore[arg-type]
    )

    assert result is loaded_prediction
    assert exported == [prediction_path]
    assert prediction_path.is_file()
    assert prediction_path.with_suffix(".json").is_file()
    quarantined = list(
        (
            prediction_path.parent
            / final_batch.FINAL_PREDICTION_ORPHAN_DIRECTORY
            / refit.member_id
        ).glob("*.orphan")
    )
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"preserve this partial file"


def test_complete_or_ledger_recorded_prediction_is_never_quarantined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = ExperimentProtocol.canonical()
    refits, calibrations = _bundles(tmp_path)
    refit = refits.members[0]
    calibration = calibrations.members[0]
    prediction_path = tmp_path / "final" / "member.fold10.npz"
    prediction_path.parent.mkdir(parents=True)
    prediction_path.write_bytes(b"complete npz")
    prediction_path.with_suffix(".json").write_text("{}", encoding="utf-8")
    loaded_prediction = object()

    def forbidden_exporter(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("a complete prediction pair must be adopted")

    monkeypatch.setattr(
        final_batch,
        "load_prediction_artifact",
        lambda *args, **kwargs: loaded_prediction,
    )
    monkeypatch.setattr(
        final_batch, "_validate_final_prediction", lambda *args, **kwargs: None
    )
    token = authorize_final_test_access(
        protocol,
        purpose="synthetic final recovery test",
        confirmation=FINAL_TEST_CONFIRMATION,
    )
    result = final_batch._export_or_resume_prediction(
        refit,
        calibration,
        prediction_path,
        state={"state": "planned"},
        settings=_settings(tmp_path),
        protocol=protocol,
        test_access=token,
        exporter=forbidden_exporter,  # type: ignore[arg-type]
    )
    assert result is loaded_prediction
    assert not (
        prediction_path.parent / final_batch.FINAL_PREDICTION_ORPHAN_DIRECTORY
    ).exists()

    prediction_path.with_suffix(".json").unlink()
    monkeypatch.setattr(
        final_batch,
        "load_prediction_artifact",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ReleaseIntegrityError("recorded sidecar is missing")
        ),
    )
    with pytest.raises(ReleaseIntegrityError, match="recorded sidecar"):
        final_batch._export_or_resume_prediction(
            refit,
            calibration,
            prediction_path,
            state={"state": "prediction_saved"},
            settings=_settings(tmp_path),
            protocol=protocol,
            test_access=token,
            exporter=forbidden_exporter,  # type: ignore[arg-type]
        )
    assert prediction_path.read_bytes() == b"complete npz"
    assert not (
        prediction_path.parent / final_batch.FINAL_PREDICTION_ORPHAN_DIRECTORY
    ).exists()


@pytest.mark.parametrize("crash_phase", ["after_link", "after_unlink"])
def test_orphan_evidence_is_reconciled_after_hard_crash_boundaries(
    tmp_path: Path, crash_phase: str
) -> None:
    protocol = ExperimentProtocol.canonical()
    refits, calibrations = _bundles(tmp_path)
    settings = _settings(tmp_path)
    plan = build_final_batch_plan(refits, calibrations, settings)
    destination = final_batch.canonical_final_ledger_path(plan)
    ledger = open_or_resume_final_batch(
        destination,
        plan,
        protocol=protocol,
        purpose="orphan reconciliation regression",
        operator="synthetic-test",
        confirmation=FINAL_TEST_CONFIRMATION,
        resume=False,
    )
    refit = refits.members[0]
    prediction_path = Path(str(plan.members[0]["final_prediction_path"]))
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_bytes(b"crash-window fragment")
    digest = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    quarantine = (
        prediction_path.parent
        / final_batch.FINAL_PREDICTION_ORPHAN_DIRECTORY
        / refit.member_id
    )
    if crash_phase == "after_link":
        quarantine.mkdir(parents=True)
        os.link(
            prediction_path,
            quarantine / f"{prediction_path.name}.{digest}.orphan",
        )
    else:
        final_batch._quarantine_planned_partial_prediction(refit, prediction_path)

    reconciled = final_batch._reconcile_planned_orphan_evidence(
        destination,
        ledger,
        refit=refit,
        prediction_path=prediction_path,
    )
    events = [
        event
        for event in reconciled.events
        if event["event"] == "partial_final_prediction_quarantined"
    ]
    assert len(events) == 1
    assert events[0]["file_sha256"] == digest
    assert not prediction_path.exists()
    assert len(list(quarantine.glob("*.orphan"))) == 1

    repeated = final_batch._reconcile_planned_orphan_evidence(
        destination,
        reconciled,
        refit=refit,
        prediction_path=prediction_path,
    )
    assert repeated.ledger_sha256 == reconciled.ledger_sha256
    assert len(repeated.events) == len(reconciled.events)


def test_complete_run_resume_is_read_only_and_missing_output_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = ExperimentProtocol.canonical()
    refits, calibrations = _bundles(tmp_path)
    settings = _settings(tmp_path)
    plan = build_final_batch_plan(refits, calibrations, settings)
    destination = final_batch.canonical_final_ledger_path(plan)
    ledger = open_or_resume_final_batch(
        destination,
        plan,
        protocol=protocol,
        purpose="completed resume regression",
        operator="synthetic-test",
        confirmation=FINAL_TEST_CONFIRMATION,
        resume=False,
    )
    report_hash = _hash("synthetic-final-report")
    states = {
        member_id: {
            **dict(state),
            "state": "report_saved",
            "final_prediction_artifact_sha256": _hash(
                "prediction:" + member_id
            ),
            "final_prediction_file_sha256": _raw_hash("npz:" + member_id),
            "final_prediction_sidecar_sha256": _raw_hash("json:" + member_id),
            "final_report_sha256": report_hash,
        }
        for member_id, state in ledger.members.items()
    }
    ready = replace(ledger, members=states)
    preregistration = final_batch._publication_preregistration(plan)

    def save_artifact(path: Path, body: dict[str, object]) -> str:
        payload = dict(body)
        digest = _canonical_hash(payload)
        payload["artifact_sha256"] = digest
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return digest

    architecture_bindings: dict[str, dict[str, object]] = {}
    outputs: dict[str, object] = {}
    for architecture in EXPECTED_ARCHITECTURES:
        path = settings.output_directory / f"{architecture}.architecture-summary.json"
        digest = save_artifact(
            path,
            {
                "batch_sha256": plan.batch_sha256,
                "preregistration": preregistration,
                "architecture": architecture,
            },
        )
        outputs[f"architecture_{architecture}_path"] = str(path.resolve())
        outputs[f"architecture_{architecture}_sha256"] = digest
        architecture_bindings[architecture] = {
            "path": str(path.resolve()),
            "artifact_sha256": digest,
        }
    entries: list[dict[str, object]] = []
    for seed in EXPECTED_SEEDS:
        path = settings.output_directory / f"paired-seed{seed}.bootstrap.json"
        digest = save_artifact(
            path,
            {
                "batch_sha256": plan.batch_sha256,
                "preregistration": preregistration,
                "seed": seed,
            },
        )
        entries.append(
            {
                "seed": seed,
                "path": str(path.resolve()),
                "artifact_sha256": digest,
                "direction": "ecg_transformer_minus_resnet1d",
                "alignment_sha256": _hash(f"alignment:{seed}"),
            }
        )
    paired_path = settings.output_directory / "paired-patient-bootstrap.manifest.json"
    paired_hash = save_artifact(
        paired_path,
        {
            "batch_sha256": plan.batch_sha256,
            "preregistration": preregistration,
            "entries": entries,
        },
    )
    summary_path = settings.output_directory / "final-batch-summary.json"
    summary_hash = save_artifact(
        summary_path,
        {
            "batch_sha256": plan.batch_sha256,
            "preregistration": preregistration,
            "architecture_reports": architecture_bindings,
            "paired_bootstrap_manifest": {
                "path": str(paired_path.resolve()),
                "artifact_sha256": paired_hash,
            },
        },
    )
    outputs.update(
        {
            "paired_manifest_path": str(paired_path.resolve()),
            "paired_manifest_sha256": paired_hash,
            "batch_summary_path": str(summary_path.resolve()),
            "batch_summary_sha256": summary_hash,
        }
    )
    completed = final_batch._complete_ledger(destination, ready, outputs)
    before = destination.read_bytes()
    export_calls: list[object] = []

    def forbidden_exporter(*args: object, **kwargs: object) -> object:
        export_calls.append((args, kwargs))
        raise AssertionError("complete resume must not export predictions")

    monkeypatch.setattr(
        final_batch,
        "_export_or_resume_prediction",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        final_batch,
        "verify_final_report",
        lambda *args, **kwargs: {"report_sha256": report_hash},
    )
    monkeypatch.setattr(final_batch, "_validate_member_report", lambda *a, **k: None)

    result = final_batch._run_final_batch_locked(
        destination=destination,
        plan=plan,
        refit_bundle=refits,
        calibration_bundle=calibrations,
        settings=settings,
        protocol=protocol,
        purpose=completed.purpose,
        operator=completed.operator,
        confirmation=FINAL_TEST_CONFIRMATION,
        resume=True,
        exporter=forbidden_exporter,
        subgroup_ids=np.asarray([1], dtype=np.int64),
        subgroups={"sex": np.asarray(["female"], dtype=object)},
    )
    assert result.batch_sha256 == plan.batch_sha256
    assert destination.read_bytes() == before
    assert export_calls == []

    summary_path.unlink()
    with pytest.raises(ReleaseIntegrityError, match="missing or unreadable"):
        final_batch._run_final_batch_locked(
            destination=destination,
            plan=plan,
            refit_bundle=refits,
            calibration_bundle=calibrations,
            settings=settings,
            protocol=protocol,
            purpose=completed.purpose,
            operator=completed.operator,
            confirmation=FINAL_TEST_CONFIRMATION,
            resume=True,
            exporter=forbidden_exporter,
            subgroup_ids=np.asarray([1], dtype=np.int64),
            subgroups={"sex": np.asarray(["female"], dtype=object)},
        )
    assert not summary_path.exists()
    assert export_calls == []
