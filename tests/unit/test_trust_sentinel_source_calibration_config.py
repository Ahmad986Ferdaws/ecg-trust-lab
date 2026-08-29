from __future__ import annotations

from pathlib import Path

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "trust_sentinel_source_calibration_v1.yaml"


def _load() -> dict[str, object]:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_source_calibration_roles_are_patient_disjoint_and_exclude_observed_tests() -> None:
    payload = _load()
    split = payload["patient_split"]
    assert isinstance(split, dict)
    assert split["unit"] == "patient_id"
    assert split["ranges"] == {
        "decision_fit": "[0.0,0.4)",
        "conformal_and_ood_threshold_fit": "[0.4,0.8)",
        "source_validation": "[0.8,1.0)",
    }
    assert payload["forbidden_fit_or_selection_sources"] == [
        "ptbxl_fold10",
        "sph",
        "future_external_observed_sites",
        "future_external_lockbox_sites",
    ]


def test_expected_split_counts_are_complete_and_preserve_all_five_labels() -> None:
    payload = _load()
    split = payload["patient_split"]
    assert isinstance(split, dict)
    expected = split["expected"]
    assert isinstance(expected, dict)
    assert sum(item["records"] for item in expected.values()) == 2_146
    assert all(item["patients"] > 0 for item in expected.values())
    assert all(
        set(item["positive_records"]) == {"NORM", "MI", "STTC", "CD", "HYP"}
        for item in expected.values()
    )
    assert (
        min(count for item in expected.values() for count in item["positive_records"].values())
        >= 20
    )


def test_conformal_and_ood_thresholds_have_separate_frozen_roles() -> None:
    payload = _load()
    decision_fit = payload["decision_fit"]
    conformal = payload["conformal"]
    open_world = payload["open_world"]
    assert isinstance(decision_fit, dict)
    assert isinstance(conformal, dict)
    assert isinstance(open_world, dict)
    assert decision_fit["classification_threshold_tie_rule"] == (
        "maximum_f1_then_closest_to_0.5_then_higher_threshold"
    )
    assert conformal["fit_role"] == "conformal_and_ood_threshold_fit"
    assert conformal["coverage_scope"] == "labelwise_marginal_under_exchangeability"
    assert conformal["individual_certainty_guarantee"] is False
    assert open_world["reference_role"] == "ptbxl_folds_1_to_8_training_reference"
    assert open_world["threshold_role"] == "conformal_and_ood_threshold_fit"
    assert open_world["target_site_fitting"] == "forbidden"


def test_execution_is_fail_closed_and_local_only() -> None:
    execution = _load()["execution"]
    assert isinstance(execution, dict)
    assert execution == {
        "require_clean_committed_revision": True,
        "require_verified_input_hashes": True,
        "output_root": "artifacts/trust_sentinel/source_calibration_v1",
        "output_root_must_be_absent": True,
        "automatic_publication": False,
        "raw_ids_or_row_arrays_public": False,
    }
