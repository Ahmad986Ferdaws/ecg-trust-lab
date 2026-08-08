from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import yaml

from ecg_trust.protocol import MODEL_SELECTION_FOLDS, TRAIN_FOLDS
from ecg_trust.sweep_config import (
    SWEEP_CONFIG_SCHEMA_VERSION,
    EqualBudgetSweepPair,
    SweepConfig,
    SweepConfigError,
    load_equal_budget_pair,
    load_sweep_config,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _paths() -> tuple[Path, Path]:
    root = _project_root()
    return (
        root / "configs" / "sweep_resnet_equal_budget.yaml",
        root / "configs" / "sweep_transformer_equal_budget.yaml",
    )


def _pair() -> EqualBudgetSweepPair:
    return load_equal_budget_pair(_paths(), base_dir=_project_root())


def _raw_sweep(path: Path) -> dict[str, object]:
    decoded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(decoded, dict)
    return cast(dict[str, object], decoded)


def test_bundled_pair_freezes_the_v2_equal_budget_contract() -> None:
    pair = _pair()

    assert pair.resnet.schema_version == SWEEP_CONFIG_SCHEMA_VERSION == 2
    assert pair.resnet.architecture == "resnet1d"
    assert pair.transformer.architecture == "ecg_transformer"
    assert pair.resnet.base_experiment.model.preset == "matched_capacity"
    assert pair.transformer.base_experiment.model.preset == "matched_capacity"
    for field in (
        "budget",
        "candidate_design",
        "objective",
        "seed_policy",
        "tie_break",
        "failure_policy",
        "search_space",
        "storage",
    ):
        assert getattr(pair.resnet, field) == getattr(pair.transformer, field)

    for config in pair.configs:
        assert config.budget.complete_candidates == 12
        assert config.budget.max_epochs == 30
        assert config.candidate_design.algorithm == "scipy_qmc_latin_hypercube"
        assert config.candidate_design.version == 1
        assert config.objective.name == "fold8_uncalibrated_macro_roc_auc"
        assert config.objective.pruning == "none"
        assert config.objective.required_label_count == 5
        assert config.seed_policy.kind == "fixed_across_candidates"
        assert config.seed_policy.experiment_seed == 2026
        assert config.failure_policy.retry_same_candidate is True
        assert config.failure_policy.failed_attempts_consume_budget is False
        assert config.failure_policy.interrupted_attempts_mark_failed is True
        assert config.base_experiment.train_folds == TRAIN_FOLDS
        assert config.base_experiment.validation_folds == MODEL_SELECTION_FOLDS


def test_paths_resolve_from_project_root_and_hashes_are_stable() -> None:
    root = _project_root()
    resnet_path, _ = _paths()
    first = load_sweep_config(resnet_path, base_dir=root)
    second = load_sweep_config(resnet_path, base_dir=root)

    assert first.base_experiment_path == root / "configs" / "train_resnet_matched.yaml"
    assert first.storage.sqlite_path == (
        root / "runs" / "sweeps" / first.comparison_id / "optuna.sqlite3"
    )
    assert first.storage.output_root == root / "runs" / "sweeps" / first.comparison_id
    assert first.config_hash == second.config_hash
    assert first.config_hash.startswith("sha256:")


@pytest.mark.parametrize(
    "field",
    [
        "budget",
        "candidate_design",
        "objective",
        "seed_policy",
        "tie_break",
        "failure_policy",
    ],
)
def test_pair_rejects_any_protocol_policy_mismatch(field: str) -> None:
    pair = _pair()
    original = getattr(pair.transformer, field)
    dataclass_field = next(iter(original.__dataclass_fields__))
    current = getattr(original, dataclass_field)
    replacement: object
    if isinstance(current, bool):
        replacement = not current
    elif isinstance(current, int):
        replacement = current + 1
    elif isinstance(current, tuple):
        replacement = tuple(reversed(current))
    else:
        replacement = f"{current}_changed"
    changed_policy = replace(original, **{dataclass_field: replacement})
    changed_transformer = replace(pair.transformer, **{field: changed_policy})

    with pytest.raises(SweepConfigError, match="paired sweep contract differs"):
        EqualBudgetSweepPair.create([pair.resnet, changed_transformer])


def test_pair_rejects_search_storage_and_study_identity_mismatches() -> None:
    pair = _pair()
    changed_space = replace(
        pair.transformer.search_space,
        batch_size=(32, 64, 256),
    )
    with pytest.raises(SweepConfigError, match="search space"):
        EqualBudgetSweepPair.create(
            [pair.resnet, replace(pair.transformer, search_space=changed_space)]
        )

    changed_storage = replace(
        pair.transformer.storage,
        sqlite_path=pair.transformer.storage.sqlite_path.with_name("other.sqlite3"),
    )
    with pytest.raises(SweepConfigError, match="storage"):
        EqualBudgetSweepPair.create(
            [pair.resnet, replace(pair.transformer, storage=changed_storage)]
        )

    with pytest.raises(SweepConfigError, match="distinct study names"):
        EqualBudgetSweepPair.create(
            [pair.resnet, replace(pair.transformer, study_name=pair.resnet.study_name)]
        )


@pytest.mark.parametrize(
    ("section", "field", "replacement", "message"),
    [
        ("budget", "complete_candidates", 11, "exactly 12"),
        ("budget", "max_epochs", 31, "exactly 30"),
        ("objective", "pruning", "median", "pruning"),
        ("objective", "required_label_count", 4, "must be 5"),
        ("seed_policy", "kind", "per_candidate", "fixed_across_candidates"),
        ("candidate_design", "version", 2, "version must be 1"),
        ("failure_policy", "retry_same_candidate", False, "retry"),
        ("failure_policy", "failed_attempts_consume_budget", True, "must not consume"),
    ],
)
def test_sweep_rejects_protocol_weakening(
    section: str,
    field: str,
    replacement: object,
    message: str,
) -> None:
    payload = _raw_sweep(_paths()[0])
    nested = cast(dict[str, object], payload[section])
    nested[field] = replacement

    with pytest.raises(SweepConfigError, match=message):
        SweepConfig.from_mapping(payload, base_dir=_project_root())


def test_sweep_rejects_smoke_preset_and_fold_9_base_configs(tmp_path: Path) -> None:
    root = _project_root()
    resnet_path, _ = _paths()
    smoke_sweep = _raw_sweep(resnet_path)
    smoke_sweep["base_experiment"] = "configs/train_smoke.yaml"
    with pytest.raises(SweepConfigError, match="matched_capacity"):
        SweepConfig.from_mapping(smoke_sweep, base_dir=root)

    decoded: object = yaml.safe_load(
        (root / "configs" / "train_resnet_matched.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(decoded, dict)
    contaminated = copy.deepcopy(cast(dict[str, object], decoded))
    folds = cast(dict[str, object], contaminated["folds"])
    folds["model_selection"] = [8, 9]
    contaminated_path = tmp_path / "contaminated_train.yaml"
    contaminated_path.write_text(yaml.safe_dump(contaminated), encoding="utf-8")

    contaminated_sweep = _raw_sweep(resnet_path)
    contaminated_sweep["base_experiment"] = str(contaminated_path)
    with pytest.raises(ValueError, match="immutable"):
        SweepConfig.from_mapping(contaminated_sweep, base_dir=root)


def test_search_space_requires_three_choices_and_warmup_below_horizon() -> None:
    payload = _raw_sweep(_paths()[0])
    search = cast(dict[str, object], payload["search_space"])
    search["batch_size"] = [64, 128]
    with pytest.raises(SweepConfigError, match="3 choices"):
        SweepConfig.from_mapping(payload, base_dir=_project_root())

    payload = _raw_sweep(_paths()[0])
    search = cast(dict[str, object], payload["search_space"])
    search["warmup_epochs"] = [2, 5, 30]
    with pytest.raises(SweepConfigError, match="below budget.max_epochs"):
        SweepConfig.from_mapping(payload, base_dir=_project_root())
