from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import pytest
import yaml  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "trust_sentinel_ood_completion_v1.yaml"


def _load() -> dict[str, Any]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cohort_identity_sha256(frame: pd.DataFrame) -> str:
    required = ["ecg_id", "patient_id", "strat_fold", "record_path"]
    assert list(frame.columns) == required
    assert not frame.isna().any().any()
    ordered = frame.sort_values("ecg_id", kind="stable").reset_index(drop=True)
    ecg_ids = [int(value) for value in ordered["ecg_id"]]
    assert ecg_ids == sorted(ecg_ids)
    assert len(ecg_ids) == len(set(ecg_ids))

    records: list[dict[str, int | str]] = []
    for ecg_id, patient_id, strat_fold, raw_path in ordered.itertuples(
        index=False, name=None
    ):
        assert not isinstance(ecg_id, bool)
        assert not isinstance(patient_id, bool)
        assert not isinstance(strat_fold, bool)
        path = str(raw_path)
        parsed = PurePosixPath(path)
        assert "\\" not in path
        assert not parsed.is_absolute()
        assert parsed.as_posix() == path
        assert not any(part in {"", ".", ".."} for part in parsed.parts)
        records.append(
            {
                "ecg_id": int(ecg_id),
                "patient_id": int(patient_id),
                "record_path": path,
                "strat_fold": int(strat_fold),
            }
        )
    payload = {
        "algorithm": "ordered_role_input_identity_v1",
        "records": records,
        "schema_version": 1,
    }
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    domain = b"ecg_trust.ordered_role_input_identity.v1\x00"
    return "sha256:" + hashlib.sha256(domain + canonical).hexdigest()


def _official_checksum_subset_sha256(pairs: list[tuple[str, str]]) -> str:
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_path, raw_digest in pairs:
        path = str(raw_path)
        parsed = PurePosixPath(path)
        assert "\\" not in path
        assert not parsed.is_absolute()
        assert parsed.as_posix() == path
        assert not any(part in {"", ".", ".."} for part in parsed.parts)
        assert path not in seen
        seen.add(path)
        assert len(raw_digest) == 64
        assert raw_digest == raw_digest.lower()
        assert all(character in "0123456789abcdef" for character in raw_digest)
        normalized.append({"relative_path": path, "sha256": raw_digest})
    files = sorted(normalized, key=lambda item: item["relative_path"].encode("utf-8"))
    payload = {
        "algorithm": "official_checksum_subset_v1",
        "files": files,
        "schema_version": 1,
    }
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    domain = b"ecg_trust.official_checksum_subset.v1\x00"
    return "sha256:" + hashlib.sha256(domain + canonical).hexdigest()


EXPECTED_FILE_BINDINGS = {
    "source_calibration_result": (
        "artifacts/trust_sentinel/source_calibration_v1/source-calibration-result.json",
        "8bae3acdebac42504167afc7bb7d2051b7ac2c48019aa429ed6544f14a59f38f",
    ),
    "source_calibration_config": (
        "configs/trust_sentinel_source_calibration_v1.yaml",
        "3dbef163757807c442276b80631e0c83a6c07b241c62974fc64ba91bbedb8178",
    ),
    "dataset_manifest": (
        "data/manifests/ptbxl_superclasses_v1.0.3.parquet",
        "563a2b715cc6f6657b04c2f67d813fd7c30a696210740f97c55a070f157579a0",
    ),
    "official_dataset_checksums": (
        "data/raw/ptb-xl/1.0.3/SHA256SUMS.txt",
        "b7224b92b341511ec3ceb13dc6652079b2c36a06504bcb49506f157f51dc695d",
    ),
    "refit_completion": (
        "runs/refit/ptbxl_matched_equal_budget_v1/"
        "resnet1d_refit_folds1-8_seed2026/attempt00/refit_completion.json",
        "ca6957bdaec738e771f66829b8b0994256cd82538bb902ea69f75aee4570d1ca",
    ),
    "checkpoint": (
        "runs/refit/ptbxl_matched_equal_budget_v1/"
        "resnet1d_refit_folds1-8_seed2026/attempt00/final.ckpt",
        "d3b8a19ab891db34afa6039179edab9847a8812e466a65c4cb408df12b402b35",
    ),
    "resolved_config": (
        "runs/refit/ptbxl_matched_equal_budget_v1/"
        "resnet1d_refit_folds1-8_seed2026/attempt00/resolved_refit_config.json",
        "d00643dadc1c27a241da5c100bccd45f314fe0b94b16c9e3ce9d88ed22656d49",
    ),
    "normalization": (
        "artifacts/preprocessing/ptbxl_v1.0.3_train_folds_1-7_normalization.json",
        "4a6cb489098361d8221403c14871c242672c346975af3a07f731ceac97264363",
    ),
    "historical_demo_policy": (
        "artifacts/demo/ptbxl_matched_equal_budget_v1/"
        "resnet1d-seed2026.coverage80.demo-policy.json",
        "539d9e7dfc84edc49ab285775cdd0f6e93b2f5bb804c6fe7be7d00bc2aff4d42",
    ),
    "demo_binding": (
        "artifacts/demo/ptbxl_matched_equal_budget_v1/"
        "resnet1d-seed2026.coverage80.demo-binding.json",
        "cea592df221a2821c5f29cec5f8e0175aabe997da6836f22ab28acb36dbf3b14",
    ),
    "experiment_protocol": (
        "configs/protocol.yaml",
        "d630ccb99569513082ccaaafe1b0117f5fe1567a505c600add9c0a79b64c51c8",
    ),
    "dependency_lock": (
        "uv.lock",
        "73234642bb4982a0e2cb4e25774f8d82e7a51a650033faec2297d45032da2d07",
    ),
    "project_manifest": (
        "pyproject.toml",
        "e1de755829678d588784bbcc34becc8c031c742d9b3f05458e76e67f577da3cd",
    ),
}


def test_protocol_is_a_separate_frozen_preregistration() -> None:
    payload = _load()

    assert payload["schema_version"] == 1
    assert payload["protocol_id"] == "trust-sentinel-ood-completion-v1"
    assert payload["status"] == "frozen_pre_execution"
    assert payload["frozen_at_utc"].isoformat() == "2026-08-29T08:53:21+00:00"
    assert payload["research_only"] is True
    assert payload["immutability"] == {
        "source_calibration_v1_must_remain_pending": True,
        "source_calibration_v1_mutation": "forbidden",
        "protocol_changes_after_embedding_access": "forbidden",
        "unfavorable_results_must_be_retained": True,
    }


def test_runtime_binds_exact_environment_versions() -> None:
    runtime = _load()["runtime"]

    assert runtime["python_version"] == "3.12.13"
    assert runtime["nvidia_driver_version"] == "596.49"
    assert runtime["cudnn_version"] == "9.20.0"
    assert runtime["cudnn_version_api_integer"] == 92_000


def test_all_input_identities_are_frozen_to_exact_paths_and_hashes() -> None:
    bindings = _load()["bindings"]

    for name, (path, digest) in EXPECTED_FILE_BINDINGS.items():
        assert bindings[name]["path"] == path
        assert bindings[name]["file_sha256"] == digest

    assert bindings["source_calibration_result"]["artifact_sha256"] == (
        "sha256:b9063fd2965b194806f9e544f3ea6390cc19bc8a93b27d3e88a674bf0aa7c839"
    )
    assert bindings["refit_completion"]["artifact_sha256"] == (
        "sha256:a5947701b8d70cddc68031be739eba38e0b8910fd4c9d105e151e9f98d8b0853"
    )
    assert bindings["resolved_config"]["inner_config_sha256"] == (
        "sha256:003125474caa877585e609b7b248727aa3ecaf7c716d8c249966ff4b9188e71e"
    )
    assert bindings["experiment_protocol"]["protocol_sha256"] == (
        "sha256:ebfdb588615bfa22eedc6d936d7b0155a33702878cbe0258ebb84aaa88567e09"
    )
    assert bindings["experiment_protocol"]["resolved_file_sha256"] == (
        "242bc6b1e37e264a9225702b31eca092d4edfd430d1cb014d348ceca92878fa2"
    )


def test_available_local_bindings_match_the_frozen_bytes() -> None:
    bindings = _load()["bindings"]

    for name, (relative_path, _) in EXPECTED_FILE_BINDINGS.items():
        source = PROJECT_ROOT / relative_path
        if source.is_file():
            assert _sha256(source) == bindings[name]["file_sha256"]
    resolved_protocol = PROJECT_ROOT / bindings["experiment_protocol"]["resolved_path"]
    if resolved_protocol.is_file():
        assert _sha256(resolved_protocol) == bindings["experiment_protocol"][
            "resolved_file_sha256"
        ]


def test_bound_json_artifact_identities_match_their_contents() -> None:
    bindings = _load()["bindings"]
    checks = (
        ("source_calibration_result", "artifact_sha256"),
        ("refit_completion", "artifact_sha256"),
    )
    for binding_name, field in checks:
        path = PROJECT_ROOT / bindings[binding_name]["path"]
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload[field] == bindings[binding_name][field]


def test_roles_are_exact_patient_disjoint_and_have_no_observed_test_source() -> None:
    payload = _load()
    roles = payload["roles"]

    assert roles["records_are_weighted_equally_for_fit_and_threshold"] is True
    assert roles["patients_must_be_disjoint_across_reference_threshold_and_validation"] is True
    assert roles["reference"] == {
        "artifact_role": "REFERENCE",
        "source": "ptbxl_task_manifest",
        "folds": [1, 2, 3, 4, 5, 6, 7, 8],
        "records": 17_084,
        "patients": 14_823,
        "allowed_use": "embedding_mean_and_covariance_only",
    }
    assert roles["threshold_fit"]["artifact_role"] == "THRESHOLD_FIT"
    assert roles["threshold_fit"]["records"] == 834
    assert roles["threshold_fit"]["patients"] == 757
    assert roles["source_validation"]["artifact_role"] == "SOURCE_VALIDATION"
    assert roles["source_validation"]["records"] == 465
    assert roles["source_validation"]["patients"] == 409
    assert roles["source_validation"]["tuning_allowed"] is False
    assert roles["source_validation"]["inclusion"] == (
        "all_canonical_records_no_quality_filtering"
    )
    assert roles["excluded"] == [
        "ptbxl_fold9_decision_fit",
        "ptbxl_fold10",
        "sph",
        "future_external_observed_sites",
        "future_external_lockbox_sites",
    ]


def test_cohort_identities_recompute_from_exact_domain_separated_bytes() -> None:
    payload = _load()
    identity = payload["cohort_identity"]
    manifest_path = PROJECT_ROOT / payload["bindings"]["dataset_manifest"]["path"]
    if not manifest_path.is_file():
        pytest.skip("private PTB-XL task manifest is unavailable")
    frame = pd.read_parquet(
        manifest_path,
        columns=["ecg_id", "patient_id", "strat_fold", "record_path"],
    )

    assert identity["domain_prefix_utf8"] == "ecg_trust.ordered_role_input_identity.v1"
    assert identity["domain_terminator_hex"] == "00"
    assert identity["json_serialization"] == {
        "allow_nan": False,
        "ensure_ascii": True,
        "separators": [",", ":"],
        "sort_keys": True,
        "trailing_newline": False,
    }
    reference = frame.loc[frame["strat_fold"].between(1, 8)].reset_index(drop=True)
    fold9 = frame.loc[frame["strat_fold"].eq(9)].reset_index(drop=True)
    assert len(reference) == 17_084
    assert len(fold9) == 2_146
    assert _cohort_identity_sha256(reference) == identity["reference_sha256"]
    assert _cohort_identity_sha256(fold9) == identity["fold9_sha256"]


def test_official_checksum_subset_encoding_is_exact_and_execution_filled() -> None:
    subset = _load()["official_checksum_subset"]

    assert subset["selection_roles"] == ["REFERENCE", "THRESHOLD_FIT", "SOURCE_VALIDATION"]
    assert subset["excluded_role"] == "ptbxl_fold9_decision_fit"
    assert subset["file_extensions"] == [".dat", ".hea"]
    assert subset["expected_selected_records"] == 18_383
    assert subset["expected_selected_files"] == 36_766
    assert subset["domain_prefix_utf8"] == "ecg_trust.official_checksum_subset.v1"
    assert subset["domain_terminator_hex"] == "00"
    assert subset["pair_order"] == "ascending_relative_path_utf8_byte_order"
    assert subset["json_serialization"] == {
        "allow_nan": False,
        "ensure_ascii": True,
        "separators": [",", ":"],
        "sort_keys": True,
        "trailing_newline": False,
    }
    assert subset["observed_sha256"] is None
    assert subset["observed_digest_timing"] == "execution_after_input_verification"

    pairs = [
        ("records100/00000/00001_lr.hea", "f" * 64),
        ("records100/00000/00001_lr.dat", "0" * 64),
    ]
    expected = "sha256:74e360f66d91281da0bd0215fac11b18f7c2e465de36ac86c683499f8a5bb315"
    assert _official_checksum_subset_sha256(pairs) == expected
    assert _official_checksum_subset_sha256(list(reversed(pairs))) == expected


def test_embedding_runtime_and_two_pass_reproducibility_are_frozen() -> None:
    payload = _load()
    embedding = payload["embedding"]
    runtime = payload["runtime"]

    assert embedding["method"] == "frozen_resnet_preclassifier_global_average_pool"
    assert embedding["dimension"] == 512
    assert embedding["output_dtype"] == "float32"
    assert runtime["device"] == "cuda:0"
    assert runtime["input_dtype"] == "float32"
    assert runtime["model_parameter_dtype"] == "float32"
    assert runtime["embedding_dtype"] == "float32"
    assert runtime["autocast"] is False
    assert runtime["tf32"] is False
    assert runtime["torch_compile"] is False
    assert runtime["deterministic_algorithms"] is True
    assert runtime["batch_size"] == 128
    assert runtime["num_workers"] == 4
    assert runtime["shuffle"] is False
    assert runtime["drop_last"] is False
    assert runtime["full_embedding_passes_per_role"] == 2
    assert runtime["pass_comparison"] == "exact_alignment_and_embedding_tensor_sha256"
    assert runtime["pass_mismatch_policy"] == "fail_closed"


def test_detector_threshold_and_validation_statistics_are_frozen() -> None:
    payload = _load()
    detector = payload["detector"]
    threshold = detector["threshold"]
    evaluation = payload["evaluation"]
    bootstrap = evaluation["bootstrap"]

    assert detector["fit_math_device"] == "cpu"
    assert detector["fit_math_dtype"] == "float64"
    assert detector["shrinkage"] == 0.1
    assert detector["ridge"] == 0.000001
    assert threshold["unit"] == "record"
    assert threshold["sample_count"] == 834
    assert threshold["inlier_coverage"] == 0.95
    assert threshold["expected_rank"] == 794
    assert threshold["unsupported_comparison"] == "score_strictly_greater_than_threshold"
    assert threshold["tie_rule"] == "retain_all_threshold_ties"
    assert evaluation["preseal_access"] == {
        "allowed": [
            "preregistered_label_free_identity",
            "official_checksum_metadata",
        ],
        "forbidden": [
            "waveform_bytes",
            "waveform_decoding",
            "embeddings",
            "scores",
            "metrics",
        ],
        "purpose": "partition_alignment_and_provenance_integrity_only",
    }
    assert evaluation["detector_must_be_sealed_before_forbidden_access"] is True
    assert evaluation["tuning_allowed"] is False
    assert evaluation["source_record_support_coverage"] == {
        "required_public_field": True,
        "formula": "accepted_records_divided_by_records",
        "identity": "one_minus_source_record_false_rejection_rate",
    }
    assert bootstrap == {
        "unit": "patient_cluster",
        "method": "percentile",
        "resamples": 10_000,
        "seed": 20_260_829,
        "random_generator": "numpy.random.Generator_PCG64",
        "patient_order": "ascending_numeric_patient_id",
        "draw_rule": "draw_n_patients_with_replacement_per_resample",
        "records_per_drawn_patient": "all_records_repeated_once_per_patient_draw",
        "record_order_within_patient": "ascending_ecg_id",
        "confidence": 0.95,
        "two_sided_quantiles": [0.025, 0.975],
        "one_sided_upper_quantile": 0.95,
        "resampled_statistic": "record_weighted_false_rejection_rate",
        "percentile_function": "numpy.quantile",
        "quantile_method": "linear",
        "nonfinite_or_empty_resample_policy": "fail_closed",
    }
    assert evaluation["research_bundle_eligibility"] == {
        "metric": "source_record_false_rejection_rate_one_sided_95_upper_bound",
        "comparison": "less_than_or_equal_to",
        "maximum": 0.05,
        "failure_policy": "completed_evidence_not_research_bundle_eligible",
        "eligible_status": "SOURCE_SUPPORT_GATE_COMPLETE",
        "ineligible_status": "SOURCE_SUPPORT_GATE_TARGET_MISSED",
        "status_claim_boundary": "source_support_only_not_validated_ood_or_release",
    }
    assert evaluation["ood_positive_metrics"] == {
        "semantic_ood_recall": "NOT_EVALUATED",
        "severe_ood_recall": "NOT_EVALUATED",
        "ood_auroc": "NOT_EVALUATED",
        "ood_average_precision": "NOT_EVALUATED",
        "unseen_site_or_device_performance": "NOT_EVALUATED",
    }


def test_private_outputs_and_claims_fail_closed() -> None:
    payload = _load()
    artifacts = payload["artifacts"]
    execution = payload["execution"]
    claims = payload["claims"]

    assert artifacts["output_root"] == "artifacts/trust_sentinel/ood_completion_v1"
    assert artifacts["output_root_must_be_absent"] is True
    assert artifacts["output_root_directory_commit_precedes_finalization"] is True
    assert artifacts["output_root_immutable_after_success_manifest"] is True
    assert artifacts["post_directory_commit_existing_file_mutation"] == "forbidden"
    assert artifacts["post_directory_commit_allowed_new_files"] == [
        "success-manifest.json",
        "failure-receipt.json",
    ]
    assert artifacts["private_embeddings"]["public"] is False
    assert artifacts["private_embeddings"]["retain_locally"] is True
    assert artifacts["private_embeddings"]["exact_npz_keys"] == [
        "ecg_id",
        "patient_id",
        "strat_fold",
        "embedding",
    ]
    assert set(artifacts["private_embeddings"]["roles"]) == {
        "REFERENCE",
        "THRESHOLD_FIT",
        "SOURCE_VALIDATION",
    }
    assert artifacts["aggregate_result"]["public_fields"] == "aggregate_only"
    assert artifacts["aggregate_result"][
        "raw_ids_embeddings_scores_logits_probabilities_waveforms_or_paths"
    ] == "forbidden"
    assert artifacts["automatic_publication"] is False
    assert execution["validation_access_before_detector_seal"] == (
        "preregistered_label_free_identity_and_official_checksum_metadata_only"
    )
    assert execution["skipped_records"] == "forbidden"
    assert execution["automatic_release"] is False
    assert claims["allowed_scope"] == "retrospective_ptbxl_source_domain_development_only"
    assert "validated_ood_detection" in claims["forbidden"]
    assert "complete_vnext_release" in claims["forbidden"]


def test_source_validation_is_one_shot_and_success_manifest_is_last() -> None:
    payload = _load()
    artifacts = payload["artifacts"]
    one_shot = artifacts["source_validation_one_shot"]
    external = one_shot["external_claim"]
    marker = one_shot["retained_staging_marker"]

    assert external == {
        "basename": ".ood_completion_v1.source-validation-one-shot-claim.json",
        "path": (
            "artifacts/trust_sentinel/"
            ".ood_completion_v1.source-validation-one-shot-claim.json"
        ),
        "location": "adjacent_to_output_root",
        "creation": "atomic_create_new_no_overwrite",
        "timing": (
            "after_durable_staging_marker_before_any_source_validation_waveform_byte_access"
        ),
        "retention": "permanent",
        "concurrent_or_retry_access": "forbidden",
        "sanitized": True,
    }
    assert marker == {
        "path": "source-validation-access-armed.json",
        "creation": "atomic_create_new_no_overwrite",
        "timing": (
            "durably_staged_before_external_claim_and_before_any_source_validation_waveform_byte_access"
        ),
        "retention": "success_or_post_claim_failure",
        "sanitized": True,
        "state": "SOURCE_VALIDATION_ACCESS_ARMED",
        "contains_owner_nonce": True,
        "binds_external_claim_file_sha256": True,
    }
    assert one_shot["pre_claim_marker_only_crash"] == {
        "proves_source_validation_access": False,
        "automatic_retry_or_cleanup": "forbidden",
        "resolution": "manual_forensic_review_under_a_new_protocol",
    }
    assert one_shot["sanitized_content_forbids"] == (
        "raw_ids_embeddings_scores_logits_probabilities_waveforms_or_paths"
    )

    failure = artifacts["post_claim_or_post_validation_failure"]
    assert failure == {
        "preserve_existing_evidence": True,
        "external_claim_retention": "permanent",
        "staging_marker_retention": "required",
        "failure_receipt": "emit_atomically_when_possible",
        "success_manifest": "forbidden",
        "retry_or_resume": "forbidden",
    }

    manifest = artifacts["success_manifest"]
    expected_inventory = [
        "distribution-policy.json",
        "ood-completion-result.json",
        "private/reference-embeddings.json",
        "private/reference-embeddings.npz",
        "private/source-validation-embeddings.json",
        "private/source-validation-embeddings.npz",
        "private/threshold-fit-embeddings.json",
        "private/threshold-fit-embeddings.npz",
        "source-validation-access-armed.json",
    ]
    assert expected_inventory == sorted(
        expected_inventory,
        key=lambda value: value.encode("utf-8"),
    )
    assert manifest["path"] == "success-manifest.json"
    assert manifest["schema_version"] == 1
    assert manifest["artifact_type"] == "ecg_trust.ood_completion_success_manifest"
    assert manifest["protocol_id"] == "trust-sentinel-ood-completion-v1"
    assert manifest["status"] == "SUCCESS"
    assert manifest["inventory_order"] == "ascending_relative_path_utf8_byte_order"
    assert manifest["inventory_entry_fields"] == [
        "relative_path",
        "file_sha256",
        "size_bytes",
    ]
    assert manifest["exact_inventory"] == expected_inventory
    assert manifest["manifest_excluded_from_own_inventory"] is True
    assert manifest["self_hash_field"] == "artifact_sha256"
    assert manifest["self_hash_algorithm"] == (
        "sha256_over_canonical_json_excluding_artifact_sha256"
    )
    assert manifest["canonical_json"] == {
        "allow_nan": False,
        "ensure_ascii": True,
        "separators": [",", ":"],
        "sort_keys": True,
        "trailing_newline_in_hash": False,
        "file_trailing_newline": True,
    }
    assert manifest["creation"] == "atomic_create_new_no_overwrite"
    assert manifest["write_order"] == "last_after_post_commit_verification"

    assert artifacts["bundle_verification"] == {
        "require_success_manifest": True,
        "require_external_one_shot_claim": True,
        "reject_failure_receipt": True,
        "expected_output_files": "exact_success_manifest_inventory_plus_manifest_itself",
        "reject_missing_expected_artifact": True,
        "reject_extra_artifact": True,
        "reject_file_size_or_sha256_mismatch": True,
        "reject_external_claim_sha256_mismatch_with_marker": True,
        "reject_manifest_self_hash_mismatch": True,
    }
    execution = payload["execution"]
    assert execution["concurrent_or_retry_source_validation_access"] == "forbidden"
    assert execution["success_manifest_must_be_final_output_write"] is True
    assert execution["bundle_use_requires_success_manifest_verification"] is True
