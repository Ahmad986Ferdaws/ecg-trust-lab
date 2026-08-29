from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import ecg_trust.ood_completion.pipeline as pipeline
from ecg_trust.ood_completion.cohorts import load_ood_cohorts
from ecg_trust.ood_completion.models import load_ood_completion_failure_bytes
from ecg_trust.ood_completion.pipeline import (
    OODCompletionConfigError,
    OODCompletionExecutionError,
    OODCompletionIntegrityError,
    load_ood_completion_config,
    verify_ood_inputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "trust_sentinel_ood_completion_v1.yaml"
SOURCE_RESULT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "trust_sentinel"
    / "source_calibration_v1"
    / "source-calibration-result.json"
)


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _claim_state(character: str = "a") -> pipeline._OneShotClaimState:
    return pipeline._OneShotClaimState(owner_nonce=character * 64)


def test_official_config_loads_as_one_exact_byte_frozen_contract() -> None:
    config = load_ood_completion_config(CONFIG_PATH)

    assert config.file_sha256 == (
        "sha256:5d12a71e8cd11350580a6d88b3656ca416392bedd3209d558ba116a90d536070"
    )
    assert config.reference_identity_sha256 == (
        "sha256:4aec3498193b962a0f9434e2032f5050e6f7daf4a8ddb44f87f54721efb72ae8"
    )
    assert config.fold9_identity_sha256 == (
        "sha256:f5b06b01ca347e33068b128d93a0ed6bc3cd0e1f2e85f931cae5b35834612707"
    )
    assert config.expected_counts.reference.records == 17_084
    assert config.expected_counts.threshold_fit.records == 834
    assert config.expected_counts.source_validation.records == 465
    assert config.patient_split_salt == "trust-sentinel-v1"
    assert config.selected_record_count == 18_383
    assert config.selected_file_count == 36_766
    assert config.expected_cudnn_version == 92_000


def test_any_config_byte_change_is_rejected_before_execution(tmp_path: Path) -> None:
    changed = tmp_path / "config.yaml"
    changed.write_bytes(CONFIG_PATH.read_bytes() + b"\n")

    with pytest.raises(OODCompletionConfigError, match="frozen v1 bytes"):
        load_ood_completion_config(changed)


@pytest.mark.skipif(
    not SOURCE_RESULT_PATH.is_file(),
    reason="private source calibration evidence is not available in this checkout",
)
def test_real_preflight_is_metric_blind_and_leaves_source_v1_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_before = SOURCE_RESULT_PATH.read_bytes()
    config = load_ood_completion_config(CONFIG_PATH)
    original_config_loader = pipeline.load_source_calibration_config
    original_result_loader = pipeline.load_source_calibration_result_bytes
    original_input_verifier = pipeline.verify_source_inputs

    def forbidden_decode(*_args: object, **_kwargs: object) -> object:
        raise AssertionError(
            "source calibration semantics were decoded before detector seal"
        )

    monkeypatch.setattr(pipeline, "load_source_calibration_config", forbidden_decode)
    monkeypatch.setattr(pipeline, "load_source_calibration_result_bytes", forbidden_decode)
    monkeypatch.setattr(pipeline, "verify_source_inputs", forbidden_decode)
    inputs = verify_ood_inputs(config, project_root=PROJECT_ROOT)
    cohorts = load_ood_cohorts(
        inputs.paths["dataset_manifest"],
        patient_split_salt=config.patient_split_salt,
        expected_counts=config.expected_counts,
    )

    subsets = pipeline._waveform_subsets(cohorts, inputs)
    provenance = pipeline._lineage_provenance(
        config=config,
        inputs=inputs,
        code_revision="1" * 40,
        reference_subset=subsets[0],
        source_subset=subsets[3],
        selected_subset=subsets[4],
    )

    assert config.patient_split_salt == "trust-sentinel-v1"
    assert cohorts.reference_sha256 == config.reference_identity_sha256
    assert cohorts.full_fold9_sha256 == config.fold9_identity_sha256
    assert len(cohorts.full_fold9_records) == 2146
    assert [subset.record_count for subset in subsets] == [17_084, 834, 465, 1299, 18_383]
    assert [subset.file_count for subset in subsets] == [34_168, 1668, 930, 2598, 36_766]
    assert provenance.raw_selected_inventory_sha256 == (
        "sha256:edf9b22f57d44d9291915a44be414422649f7224f9b9b878c8c1ba3ab7b72e86"
    )

    monkeypatch.setattr(pipeline, "load_source_calibration_config", original_config_loader)
    monkeypatch.setattr(pipeline, "load_source_calibration_result_bytes", original_result_loader)
    monkeypatch.setattr(pipeline, "verify_source_inputs", original_input_verifier)
    monkeypatch.setattr(pipeline, "_verify_sealed_policy_proof", lambda _proof: object())
    source_artifacts = pipeline._load_postseal_source_artifacts(
        sealed_policy=object(),  # type: ignore[arg-type]
        config=config,
        inputs=inputs,
        cohorts=cohorts,
    )

    assert source_artifacts.assignment_sha256 == (
        "sha256:87992206fcbfc2b091d8f8dd08998a5d9bae3d55a2d2056f1ab674a316b0675b"
    )
    assert source_artifacts.result.open_world.status == "PENDING"
    assert SOURCE_RESULT_PATH.read_bytes() == source_before
    assert _file_sha256(SOURCE_RESULT_PATH) == (
        "sha256:8bae3acdebac42504167afc7bb7d2051b7ac2c48019aa429ed6544f14a59f38f"
    )


def test_atomic_new_artifacts_and_output_roots_never_overwrite(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    pipeline._atomic_write_new(artifact, b"first\n")
    with pytest.raises(OODCompletionExecutionError, match="already exists"):
        pipeline._atomic_write_new(artifact, b"second\n")
    assert artifact.read_bytes() == b"first\n"

    staging = tmp_path / ".result.staging-test"
    staging.mkdir()
    (staging / "result.json").write_bytes(b"evidence\n")
    output = tmp_path / "result"
    pipeline._commit_staged_directory(staging, output)
    assert (output / "result.json").read_bytes() == b"evidence\n"
    replacement = tmp_path / ".result.staging-second"
    replacement.mkdir()
    with pytest.raises(OODCompletionExecutionError, match="already exists"):
        pipeline._commit_staged_directory(replacement, output)


def test_terminal_success_write_uses_only_an_adjacent_temporary_name(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result"
    output.mkdir()
    (output / "evidence.json").write_bytes(b"evidence\n")

    pipeline._atomic_write_terminal_success(output, b"success\n")

    assert {path.name for path in output.iterdir()} == {
        "evidence.json",
        pipeline.OOD_COMPLETION_SUCCESS_FILENAME,
    }
    assert not tuple(tmp_path.glob(".result.success-manifest-*.tmp"))
    with pytest.raises(OODCompletionExecutionError, match="already exists"):
        pipeline._atomic_write_terminal_success(output, b"replacement\n")


def test_terminal_success_cannot_be_revoked_by_post_publication_sync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result"
    output.mkdir()
    monkeypatch.setattr(
        pipeline,
        "_fsync_directory",
        lambda _path: (_ for _ in ()).throw(OSError("simulated sync failure")),
    )

    pipeline._atomic_write_terminal_success(output, b"success\n")

    assert (output / pipeline.OOD_COMPLETION_SUCCESS_FILENAME).read_bytes() == b"success\n"


def test_validation_extraction_requires_a_sealed_policy_proof() -> None:
    with pytest.raises(TypeError, match="sealed distribution-policy proof"):
        pipeline._extract_validation_after_policy_seal(
            sealed_policy=object(),  # type: ignore[arg-type]
            cohort=object(),  # type: ignore[arg-type]
            config=object(),  # type: ignore[arg-type]
            inputs=object(),  # type: ignore[arg-type]
            normalization=object(),  # type: ignore[arg-type]
            model=object(),  # type: ignore[arg-type]
            runtime=object(),  # type: ignore[arg-type]
            staging_root=Path("unused"),
        )


def test_source_metric_decode_requires_a_sealed_policy_proof() -> None:
    with pytest.raises(TypeError, match="sealed distribution-policy proof"):
        pipeline._load_postseal_source_artifacts(
            sealed_policy=object(),  # type: ignore[arg-type]
            config=object(),  # type: ignore[arg-type]
            inputs=object(),  # type: ignore[arg-type]
            cohorts=object(),  # type: ignore[arg-type]
        )


def test_fixed_one_shot_claim_is_atomic_and_cannot_be_reacquired(tmp_path: Path) -> None:
    output = tmp_path / "result"
    claim = pipeline._validation_access_claim_path(output)
    state = _claim_state()

    assert state.published_by_this_process is False
    claim_sha256 = pipeline._claim_validation_access(claim, claim_state=state)

    assert claim.read_bytes() == state.claim_bytes
    assert state.published_by_this_process is True
    assert claim_sha256 == pipeline._sha256_bytes(state.claim_bytes)
    loser = _claim_state("b")
    with pytest.raises(OODCompletionExecutionError, match="already exists"):
        pipeline._claim_validation_access(claim, claim_state=loser)
    assert loser.published_by_this_process is False


def test_marked_crash_staging_blocks_any_retry(tmp_path: Path) -> None:
    output = tmp_path / "result"
    staging = tmp_path / ".result.staging-crash"
    staging.mkdir()
    pipeline._mark_validation_access_armed(
        staging,
        validation_claim_file_sha256="sha256:" + "a" * 64,
        owner_nonce="b" * 64,
    )

    with pytest.raises(OODCompletionExecutionError, match="retry is forbidden"):
        pipeline._assert_no_marked_staging_retry(output)


def test_post_validation_failure_commits_evidence_and_sanitized_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result"
    config = SimpleNamespace(
        output_root="result",
        frozen_at_utc=datetime(2026, 8, 29, tzinfo=UTC),
        file_sha256="sha256:" + "a" * 64,
    )
    inputs = SimpleNamespace(project_root=tmp_path)
    monkeypatch.setattr(pipeline, "load_ood_completion_config", lambda _: config)
    monkeypatch.setattr(pipeline, "verify_ood_inputs", lambda *_args, **_kwargs: inputs)
    monkeypatch.setattr(pipeline, "_assert_clean_code_revision", lambda *_args: None)

    def fail_after_validation_access(**kwargs: object) -> object:
        staging = kwargs["staging_root"]
        claim = kwargs["validation_claim_path"]
        claim_state = kwargs["validation_claim_state"]
        assert isinstance(staging, Path)
        assert isinstance(claim, Path)
        assert isinstance(claim_state, pipeline._OneShotClaimState)
        claim_sha256 = pipeline._sha256_bytes(claim_state.claim_bytes)
        pipeline._mark_validation_access_armed(
            staging,
            validation_claim_file_sha256=claim_sha256,
            owner_nonce=claim_state.owner_nonce,
        )
        observed_claim_sha256 = pipeline._claim_validation_access(
            claim,
            claim_state=claim_state,
        )
        assert observed_claim_sha256 == claim_sha256
        raise OODCompletionExecutionError("simulated post-C failure")

    monkeypatch.setattr(pipeline, "_execute_staged", fail_after_validation_access)

    with pytest.raises(OODCompletionExecutionError, match="simulated post-C failure"):
        pipeline.prepare_ood_completion(
            config_path=tmp_path / "config.yaml",
            project_root=tmp_path,
            code_revision="1" * 40,
        )

    assert output.is_dir()
    assert (output / pipeline._VALIDATION_ACCESS_MARKER_FILENAME).is_file()
    receipt = load_ood_completion_failure_bytes(
        (output / "failure-receipt.json").read_bytes()
    )
    assert receipt.status == "FAILED"
    assert receipt.retry_requires_new_output_root is True
    assert pipeline._validation_access_claim_path(output).is_file()


def test_post_claim_failure_commits_prearmed_marker_and_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result"
    config = SimpleNamespace(
        output_root="result",
        frozen_at_utc=datetime(2026, 8, 29, tzinfo=UTC),
        file_sha256="sha256:" + "a" * 64,
    )
    inputs = SimpleNamespace(project_root=tmp_path)
    monkeypatch.setattr(pipeline, "load_ood_completion_config", lambda _: config)
    monkeypatch.setattr(pipeline, "verify_ood_inputs", lambda *_args, **_kwargs: inputs)
    monkeypatch.setattr(pipeline, "_assert_clean_code_revision", lambda *_args: None)

    def fail_after_claim(**kwargs: object) -> object:
        staging = kwargs["staging_root"]
        claim = kwargs["validation_claim_path"]
        claim_state = kwargs["validation_claim_state"]
        assert isinstance(staging, Path)
        assert isinstance(claim, Path)
        assert isinstance(claim_state, pipeline._OneShotClaimState)
        pipeline._mark_validation_access_armed(
            staging,
            validation_claim_file_sha256=pipeline._sha256_bytes(claim_state.claim_bytes),
            owner_nonce=claim_state.owner_nonce,
        )
        pipeline._claim_validation_access(claim, claim_state=claim_state)
        raise OODCompletionExecutionError("simulated post-claim failure")

    monkeypatch.setattr(pipeline, "_execute_staged", fail_after_claim)

    with pytest.raises(OODCompletionExecutionError, match="simulated post-claim failure"):
        pipeline.prepare_ood_completion(
            config_path=tmp_path / "config.yaml",
            project_root=tmp_path,
            code_revision="1" * 40,
        )

    assert output.is_dir()
    assert pipeline._validation_access_claim_path(output).is_file()
    marker = output / pipeline._VALIDATION_ACCESS_MARKER_FILENAME
    assert marker.is_file()
    receipt = load_ood_completion_failure_bytes(
        (output / "failure-receipt.json").read_bytes()
    )
    assert receipt.retry_requires_new_output_root is True
    assert not tuple(tmp_path.glob(".result.staging-*"))


def test_claim_readback_failure_still_commits_marker_and_failure_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result"
    claim_path = pipeline._validation_access_claim_path(output)
    config = SimpleNamespace(
        output_root="result",
        frozen_at_utc=datetime(2026, 8, 29, tzinfo=UTC),
        file_sha256="sha256:" + "a" * 64,
    )
    inputs = SimpleNamespace(project_root=tmp_path)
    monkeypatch.setattr(pipeline, "load_ood_completion_config", lambda _: config)
    monkeypatch.setattr(pipeline, "verify_ood_inputs", lambda *_args, **_kwargs: inputs)
    monkeypatch.setattr(pipeline, "_assert_clean_code_revision", lambda *_args: None)
    original_read = pipeline._read_bounded_file_snapshot
    fail_readback_once = True

    def fail_first_published_claim_read(
        path: Path,
        *,
        maximum_bytes: int,
        context: str,
    ) -> bytes:
        nonlocal fail_readback_once
        if path == claim_path and path.exists() and fail_readback_once:
            fail_readback_once = False
            raise OODCompletionIntegrityError("simulated claim readback failure")
        return original_read(path, maximum_bytes=maximum_bytes, context=context)

    monkeypatch.setattr(
        pipeline,
        "_read_bounded_file_snapshot",
        fail_first_published_claim_read,
    )

    def fail_during_claim_readback(**kwargs: object) -> object:
        staging = kwargs["staging_root"]
        claim_state = kwargs["validation_claim_state"]
        assert isinstance(staging, Path)
        assert isinstance(claim_state, pipeline._OneShotClaimState)
        pipeline._mark_validation_access_armed(
            staging,
            validation_claim_file_sha256=pipeline._sha256_bytes(claim_state.claim_bytes),
            owner_nonce=claim_state.owner_nonce,
        )
        pipeline._claim_validation_access(claim_path, claim_state=claim_state)
        raise AssertionError("unreachable")

    monkeypatch.setattr(pipeline, "_execute_staged", fail_during_claim_readback)

    with pytest.raises(OODCompletionExecutionError, match="could not be verified"):
        pipeline.prepare_ood_completion(
            config_path=tmp_path / "config.yaml",
            project_root=tmp_path,
            code_revision="1" * 40,
        )

    assert output.is_dir()
    assert (output / pipeline._VALIDATION_ACCESS_MARKER_FILENAME).is_file()
    receipt = load_ood_completion_failure_bytes(
        (output / pipeline.OOD_COMPLETION_FAILURE_FILENAME).read_bytes()
    )
    assert receipt.failure_code.value == "OUTPUT_COMMIT_FAILED"
    assert receipt.retry_requires_new_output_root is True
    assert claim_path.is_file()


def test_preclaim_armed_crash_is_preserved_without_publishing_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result"
    config = SimpleNamespace(
        output_root="result",
        frozen_at_utc=datetime(2026, 8, 29, tzinfo=UTC),
        file_sha256="sha256:" + "a" * 64,
    )
    inputs = SimpleNamespace(project_root=tmp_path)
    monkeypatch.setattr(pipeline, "load_ood_completion_config", lambda _: config)
    monkeypatch.setattr(pipeline, "verify_ood_inputs", lambda *_args, **_kwargs: inputs)
    monkeypatch.setattr(pipeline, "_assert_clean_code_revision", lambda *_args: None)

    def fail_after_arming(**kwargs: object) -> object:
        staging = kwargs["staging_root"]
        claim_state = kwargs["validation_claim_state"]
        assert isinstance(staging, Path)
        assert isinstance(claim_state, pipeline._OneShotClaimState)
        pipeline._mark_validation_access_armed(
            staging,
            validation_claim_file_sha256=pipeline._sha256_bytes(claim_state.claim_bytes),
            owner_nonce=claim_state.owner_nonce,
        )
        raise OODCompletionExecutionError("simulated pre-claim crash")

    monkeypatch.setattr(pipeline, "_execute_staged", fail_after_arming)

    with pytest.raises(OODCompletionExecutionError, match="simulated pre-claim crash"):
        pipeline.prepare_ood_completion(
            config_path=tmp_path / "config.yaml",
            project_root=tmp_path,
            code_revision="1" * 40,
        )

    assert not output.exists()
    assert not pipeline._validation_access_claim_path(output).exists()
    retained = tuple(tmp_path.glob(".result.staging-*"))
    assert len(retained) == 1
    assert (retained[0] / pipeline._VALIDATION_ACCESS_MARKER_FILENAME).is_file()


def test_concurrent_claim_loser_removes_only_its_preclaim_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result"
    config = SimpleNamespace(
        output_root="result",
        frozen_at_utc=datetime(2026, 8, 29, tzinfo=UTC),
        file_sha256="sha256:" + "a" * 64,
    )
    inputs = SimpleNamespace(project_root=tmp_path)
    monkeypatch.setattr(pipeline, "load_ood_completion_config", lambda _: config)
    monkeypatch.setattr(pipeline, "verify_ood_inputs", lambda *_args, **_kwargs: inputs)
    monkeypatch.setattr(pipeline, "_assert_clean_code_revision", lambda *_args: None)

    winner = _claim_state("b")

    def lose_claim_race(**kwargs: object) -> object:
        staging = kwargs["staging_root"]
        claim_path = kwargs["validation_claim_path"]
        contender = kwargs["validation_claim_state"]
        assert isinstance(staging, Path)
        assert isinstance(claim_path, Path)
        assert isinstance(contender, pipeline._OneShotClaimState)
        pipeline._mark_validation_access_armed(
            staging,
            validation_claim_file_sha256=pipeline._sha256_bytes(contender.claim_bytes),
            owner_nonce=contender.owner_nonce,
        )
        pipeline._claim_validation_access(claim_path, claim_state=winner)
        pipeline._claim_validation_access(claim_path, claim_state=contender)
        raise AssertionError("unreachable")

    monkeypatch.setattr(pipeline, "_execute_staged", lose_claim_race)

    with pytest.raises(OODCompletionExecutionError, match="already exists"):
        pipeline.prepare_ood_completion(
            config_path=tmp_path / "config.yaml",
            project_root=tmp_path,
            code_revision="1" * 40,
        )

    assert not output.exists()
    claim = pipeline._validation_access_claim_path(output)
    assert claim.read_bytes() == winner.claim_bytes
    assert not tuple(tmp_path.glob(".result.staging-*"))


def test_partial_output_commit_is_reported_as_already_committed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / ".result.staging-partial"
    staging.mkdir()
    output = tmp_path / "result"

    def fail_sync(_path: Path) -> None:
        raise OSError("simulated directory sync failure")

    monkeypatch.setattr(pipeline, "_fsync_directory", fail_sync)
    with pytest.raises(pipeline._OODOutputCommitError) as failure:
        pipeline._commit_staged_directory(staging, output)

    assert failure.value.output_root_committed is True
    assert output.is_dir()
    assert not staging.exists()


def test_direct_pipeline_revision_binding_rejects_forged_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "verify_clean_git_revision", lambda _: "2" * 40)

    with pytest.raises(pipeline.OODCompletionIntegrityError, match="differs from Git HEAD"):
        pipeline._assert_clean_code_revision(tmp_path, "1" * 40)


def test_bundle_tree_and_external_claim_marker_are_exact(tmp_path: Path) -> None:
    output = tmp_path / "ood_completion_v1"
    (output / "private").mkdir(parents=True)
    claim = pipeline._validation_access_claim_path(output)
    state = _claim_state()
    claim_sha256 = pipeline._sha256_bytes(state.claim_bytes)
    pipeline._mark_validation_access_armed(
        output,
        validation_claim_file_sha256=claim_sha256,
        owner_nonce=state.owner_nonce,
    )
    observed_claim_sha256 = pipeline._claim_validation_access(
        claim,
        claim_state=state,
    )
    assert observed_claim_sha256 == claim_sha256
    for relative_path in pipeline._BUNDLE_MEMBER_PATHS:
        destination = output.joinpath(*relative_path.split("/"))
        if relative_path == pipeline._VALIDATION_ACCESS_MARKER_FILENAME:
            continue
        else:
            destination.write_bytes(b"evidence\n")

    pipeline._assert_exact_bundle_tree(output, include_success_manifest=False)
    assert pipeline._verify_validation_access_claim_and_marker(output)[0] == claim_sha256

    (output / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(OODCompletionIntegrityError, match="exact file inventory"):
        pipeline._assert_exact_bundle_tree(output, include_success_manifest=False)


def test_failure_receipt_forbids_bundle_use_even_with_success_manifest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "ood_completion_v1"
    (output / "private").mkdir(parents=True)
    for relative_path in (*pipeline._BUNDLE_MEMBER_PATHS, "success-manifest.json"):
        destination = output.joinpath(*relative_path.split("/"))
        destination.write_bytes(b"evidence\n")
    (output / "failure-receipt.json").write_bytes(b"failure\n")

    with pytest.raises(OODCompletionIntegrityError, match="failure receipt"):
        pipeline._assert_exact_bundle_tree(output, include_success_manifest=True)


def test_failure_receipt_codes_preserve_execution_stage() -> None:
    assert pipeline._failure_code(
        pipeline.OODDeterminismError("repeat mismatch")
    ).value == "DETERMINISM_FAILED"
    assert pipeline._failure_code(
        pipeline._OODFitError("fit failed")
    ).value == "FIT_FAILED"
    assert pipeline._failure_code(
        pipeline._OODValidationError("validation failed")
    ).value == "VALIDATION_FAILED"


def test_project_path_rejects_lexical_symlink_before_resolution(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"bound input")
    symbolic = tmp_path / "symbolic.bin"
    try:
        symbolic.symlink_to(target)
    except OSError:
        pytest.skip("creating a symbolic link is not permitted on this Windows host")

    with pytest.raises(OODCompletionIntegrityError, match="link or junction"):
        pipeline._resolve_project_path(tmp_path, "symbolic.bin", require_file=True)
