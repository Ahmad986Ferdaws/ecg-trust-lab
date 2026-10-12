from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pytest

from scripts import compare_slices


def _document():
    return {
        "schema_version": 1,
        "reference": [
            {"group": "large", "count": 100, "value": 0.8},
            {"group": "small", "count": 40, "value": 0.8},
        ],
        "candidate": [
            {"group": "large", "count": 100, "value": 0.8},
            {"group": "small", "count": 40, "value": 0.8},
        ],
    }


def _args(path):
    return [
        str(path),
        "--metric",
        "retention",
        "--direction",
        "higher",
        "--max-overall-drop",
        "0.02",
        "--max-slice-drop",
        "0.05",
        "--min-count",
        "30",
    ]


def _write(tmp_path, document):
    path = tmp_path / "groups.json"
    path.write_text(json.dumps(document))
    return path


def test_success_binds_exact_input_and_explicit_controls(tmp_path, capsys):
    path = _write(tmp_path, _document())
    assert compare_slices.main(_args(path)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["result"]["passed"] is True
    assert report["input_sha256"] == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["parameters"]["min_count"] == 30
    assert report["group_counts"][1] == {"group": "small", "reference": 40, "candidate": 40}


def test_subgroup_regression_exits_one_and_keeps_the_explanation(tmp_path):
    document = _document()
    document["candidate"][1]["value"] = 0.7
    path = _write(tmp_path, document)
    result = subprocess.run(
        [sys.executable, "scripts/compare_slices.py", *_args(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert not result.stderr
    report = json.loads(result.stdout)
    assert report["result"]["slices"][1]["status"] == "fail"


def test_insufficient_policy_is_explicit_and_reported(tmp_path, capsys):
    document = _document()
    document["candidate"][1]["count"] = 2
    path = _write(tmp_path, document)
    assert compare_slices.main(_args(path)) == 1
    assert compare_slices.main([*_args(path), "--insufficient-policy", "report"]) == 0
    assert '"insufficient"' in capsys.readouterr().out


def test_lower_is_better_preserves_metric_direction(tmp_path, capsys):
    document = _document()
    document["candidate"][0]["value"] = 0.7
    path = _write(tmp_path, document)
    args = _args(path)
    args[args.index("higher")] = "lower"
    assert compare_slices.main(args) == 0
    assert json.loads(capsys.readouterr().out)["parameters"]["higher_is_better"] is False


@pytest.mark.parametrize("mutation", ["extra", "boolean_version", "boolean_count", "missing_group"])
def test_invalid_schema_or_groups_produce_no_report(tmp_path, capsys, mutation):
    document = _document()
    if mutation == "extra":
        document["unrecognized"] = "private marker"
    elif mutation == "boolean_version":
        document["schema_version"] = True
    elif mutation == "boolean_count":
        document["candidate"][0]["count"] = True
    else:
        document["candidate"][0]["group"] = "private marker"
    assert compare_slices.main(_args(_write(tmp_path, document))) == 2
    output = capsys.readouterr()
    assert not output.out
    assert "private marker" not in output.err


@pytest.mark.parametrize(
    "payload", [b'{"schema_version":1,"schema_version":1}', b'{"x":NaN}', b"\xff"]
)
def test_duplicate_nonfinite_or_non_utf8_json_is_rejected(tmp_path, capsys, payload):
    path = tmp_path / "bad.json"
    path.write_bytes(payload)
    assert compare_slices.main(_args(path)) == 2
    assert not capsys.readouterr().out


def test_oversized_file_is_rejected(tmp_path, capsys):
    path = tmp_path / "large.json"
    path.write_bytes(b" " * (compare_slices.MAX_INPUT_BYTES + 1))
    assert compare_slices.main(_args(path)) == 2
    assert not capsys.readouterr().out


def test_nonfinite_computed_drop_does_not_publish_invalid_json(tmp_path, capsys):
    document = _document()
    document["reference"][0]["value"] = -1e308
    document["candidate"][0]["value"] = 1e308
    assert compare_slices.main(_args(_write(tmp_path, document))) == 2
    assert not capsys.readouterr().out


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO contract is POSIX-only")
def test_fifo_does_not_block_waiting_for_a_writer(tmp_path):
    path = tmp_path / "pipe"
    os.mkfifo(path)
    result = subprocess.run(
        [sys.executable, "scripts/compare_slices.py", *_args(path)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2
    assert not result.stdout
