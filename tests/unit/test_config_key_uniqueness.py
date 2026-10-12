from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ecg_trust.experiment_config import ExperimentConfigError, load_experiment_config
from ecg_trust.protocol import ProtocolValidationError, load_protocol
from ecg_trust.refit_config import RefitConfigError, load_refit_config
from ecg_trust.sweep_config import SweepConfigError, load_sweep_config
from tests.unit.test_experiment_runner import _config_payload
from tests.unit.test_refit_runner import _payload as _refit_payload

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("equal_values", [False, True])
def test_development_config_refuses_duplicate_nested_controls(
    tmp_path: Path, equal_values: bool
) -> None:
    source = yaml.safe_dump(_config_payload(tmp_path), sort_keys=False)
    replacement = "  epochs: 2\n  epochs: " + ("2" if equal_values else "3")
    source = source.replace("  epochs: 2", replacement)
    path = tmp_path / "development.yaml"
    path.write_text(source)
    with pytest.raises(ExperimentConfigError, match="duplicate"):
        load_experiment_config(path)


def test_protocol_refuses_duplicate_top_level_declaration(tmp_path: Path) -> None:
    source = (ROOT / "configs/protocol.yaml").read_text()
    data = yaml.safe_load(source)
    key = next(iter(data))
    path = tmp_path / "protocol.yaml"
    path.write_text(source + "\n" + yaml.safe_dump({key: data[key]}))
    with pytest.raises(ProtocolValidationError, match="duplicate"):
        load_protocol(path)


def test_sweep_refuses_duplicate_budget_mapping(tmp_path: Path) -> None:
    source = (ROOT / "configs/sweep_resnet_equal_budget.yaml").read_text()
    data = yaml.safe_load(source)
    path = tmp_path / "sweep.yaml"
    path.write_text(source + "\n" + yaml.safe_dump({"budget": data["budget"]}))
    with pytest.raises(SweepConfigError, match="duplicate"):
        load_sweep_config(path, base_dir=ROOT)


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_refit_refuses_duplicate_training_declaration(tmp_path: Path, suffix: str) -> None:
    data = _refit_payload(tmp_path)
    if suffix == ".yaml":
        source = yaml.safe_dump(data) + "schema_version: 1\n"
    else:
        source = json.dumps(data)[:-1] + ', "schema_version": 1}'
    path = tmp_path / ("refit" + suffix)
    path.write_text(source)
    with pytest.raises(RefitConfigError, match="duplicate"):
        load_refit_config(path)


@pytest.mark.parametrize(
    "source",
    [
        "name: first\nname: second\n",
        "root: {key: first, key: second}",
        'root: {"key": first, "k\\u0065y": second}',
        "1: first\ntrue: second\n",
        "root: {<<: {key: first, key: second}}",
        "base: &base {key: first, key: second}\nroot: {<<: *base}",
        "base: &base {key: first}\nroot: {<<: *base, <<: *base}",
    ],
)
def test_yaml_rejects_native_key_collisions_before_merge_expansion(source: str) -> None:
    from ecg_trust._config_loading import load_yaml_unique_keys

    with pytest.raises(yaml.YAMLError, match="duplicate"):
        load_yaml_unique_keys(source)


@pytest.mark.parametrize(
    "source",
    [
        "base: &base {key: inherited, other: 1}\nroot: {<<: *base, key: explicit}\ncopy: *base",
        "first: &first {key: first}\nsecond: &second {key: second}\nroot: {<<: [*first, *second]}",
        'root: {<<: {key: inherited}, "<<": literal}',
        "=: literal\nkey: value",
        "flags: [yes, no, true, false]\nnumber: 0xB\nempty: null",
    ],
)
def test_valid_yaml_merges_aliases_and_scalar_resolution_are_preserved(source: str) -> None:
    from ecg_trust._config_loading import load_yaml_unique_keys

    assert load_yaml_unique_keys(source) == yaml.safe_load(source)


def test_duplicate_checks_do_not_modify_global_safe_loader() -> None:
    from ecg_trust._config_loading import load_yaml_unique_keys

    with pytest.raises(yaml.YAMLError):
        load_yaml_unique_keys("name: first\nname: second")
    assert yaml.safe_load("name: first\nname: second") == {"name": "second"}


@pytest.mark.parametrize(
    "source",
    [
        "!!python/object/apply:builtins.print ['must not execute']",
        "? [unhashable, key]\n: value",
        "root: [unfinished",
        "one: 1\n---\ntwo: 2",
    ],
)
def test_safe_loader_refusals_are_preserved(source: str) -> None:
    from ecg_trust._config_loading import load_yaml_unique_keys

    with pytest.raises(yaml.YAMLError):
        load_yaml_unique_keys(source)


@pytest.mark.parametrize("source", ['{"root":{"x":1,"x":2}}', '{"x":1,"\\u0078":2}'])
def test_json_rejects_nested_and_escaped_duplicate_keys(source: str) -> None:
    from ecg_trust._config_loading import DuplicateConfigKeyError, load_json_unique_keys

    with pytest.raises(DuplicateConfigKeyError):
        load_json_unique_keys(source)


def test_all_committed_yaml_configs_decode_to_the_same_values() -> None:
    from ecg_trust._config_loading import load_yaml_unique_keys

    paths = sorted((ROOT / "configs").glob("*.yaml"))
    assert paths
    for path in paths:
        source = path.read_text()
        assert load_yaml_unique_keys(source) == yaml.safe_load(source), path.name


def test_transport_config_rejects_duplicates_before_bound_file_checks(tmp_path: Path) -> None:
    from ecg_trust.sph_transport import SPHTransportIntegrityError, load_sph_transport_spec

    source = (ROOT / "configs/external_transport_sph_frozen_r2.yaml").read_text()
    data = yaml.safe_load(source)
    path = tmp_path / "transport.yaml"
    path.write_text(source + "\n" + yaml.safe_dump({"schema_version": data["schema_version"]}))
    with pytest.raises(SPHTransportIntegrityError, match="duplicate"):
        load_sph_transport_spec(path)


def test_protocol_and_training_config_hashes_match_previous_decoding(monkeypatch) -> None:
    from ecg_trust import experiment_config, protocol, refit_config, sweep_config

    cases = [
        (
            experiment_config,
            load_experiment_config,
            "train_smoke.yaml",
            "config_hash",
            {"base_dir": ROOT},
        ),
        (
            sweep_config,
            load_sweep_config,
            "sweep_resnet_equal_budget.yaml",
            "config_hash",
            {"base_dir": ROOT},
        ),
        (
            refit_config,
            load_refit_config,
            "refit_resnet_frozen.yaml",
            "config_hash",
            {"base_dir": ROOT},
        ),
        (protocol, load_protocol, "protocol.yaml", "protocol_hash", {}),
    ]
    for module, loader, filename, hash_attribute, kwargs in cases:
        path = ROOT / "configs" / filename
        current = loader(path, **kwargs)
        with monkeypatch.context() as context:
            context.setattr(module, "load_yaml_unique_keys", yaml.safe_load)
            previous = loader(path, **kwargs)
        assert getattr(current, hash_attribute) == getattr(previous, hash_attribute)
