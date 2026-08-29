from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "trust_sentinel_vnext.yaml"


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


def test_vnext_binds_immutable_baseline_files_by_sha256() -> None:
    payload = _load()
    baseline = payload["baseline"]

    assert baseline["immutable"] is True
    for key in ("ptbxl_protocol", "sph_transport_protocol", "public_result_snapshot"):
        binding = baseline[key]
        source = PROJECT_ROOT / binding["path"]
        assert source.is_file()
        assert _sha256(source) == binding["sha256"]


def test_vnext_decision_contract_is_fail_closed_and_predictions_are_last() -> None:
    payload = _load()
    decision = payload["decision_contract"]

    assert decision["state_order"] == [
        "INVALID_INPUT",
        "REACQUIRE",
        "UNSUPPORTED_INPUT",
        "ABSTAIN",
        "PREDICTION_ALLOWED",
    ]
    assert decision["predictions_exposed_only_for"] == "PREDICTION_ALLOWED"
    assert decision["component_failure_policy"] == "fail_closed"


def test_vnext_preserves_observed_cohorts_and_requires_a_new_lockbox() -> None:
    payload = _load()
    roles = payload["data_roles"]
    future = roles["future_external_sources"]

    assert roles["ptbxl_fold_10"] == "previously_observed_baseline_only"
    assert roles["sph"] == "previously_observed_transport_only"
    assert future["exclude_overlapping_ptb_and_ptbxl_records"] is True
    assert future["shared_ontology_requires_cardiology_review"] is True
    assert future["reserve_at_least_one_untouched_site_lockbox"] is True


def test_vnext_forbids_target_fitting_automatic_retraining_and_public_rows() -> None:
    payload = _load()

    assert payload["open_world"]["target_site_fitting"] == ("forbidden_in_zero_adaptation_track")
    assert payload["release"]["automatic_model_retraining"] == "forbidden"
    assert payload["release"]["public_waveform_or_row_level_artifacts"] == "forbidden"
    assert payload["claims"]["allowed_scope"] == ("retrospective_research_and_education_only")
