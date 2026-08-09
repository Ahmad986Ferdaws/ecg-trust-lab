from __future__ import annotations

import copy
from pathlib import Path
from typing import cast

import numpy as np
import pytest

import ecg_trust.post_evaluation as post
from ecg_trust.protocol import LABEL_ORDER, ExperimentProtocol
from scripts.freeze_post_evaluation import build_parser


def _hash(character: str = "a") -> str:
    return "sha256:" + character * 64


def _cohort() -> dict[str, object]:
    count = 300
    ecg_id = np.arange(1, count + 1, dtype=np.int64)
    patient_id = np.arange(10_001, 10_001 + count, dtype=np.int64)
    rows = np.arange(count, dtype=np.int64)
    targets = np.stack(
        [((rows // (2**index)) % 2).astype(np.int8) for index in range(len(LABEL_ORDER))],
        axis=1,
    )
    return post._explanation_cohort(
        ecg_id,
        patient_id,
        targets,
        alignment_sha256=_hash("b"),
    )


def _valid_payload(project_root: Path) -> dict[str, object]:
    protocol = ExperimentProtocol.canonical()
    comparison_id = "ptbxl_matched_equal_budget_v1"
    output = post._output_contract(project_root, comparison_id)
    root = Path(cast(str, output["root"]))
    members: list[dict[str, object]] = []
    for architecture in post.EXPECTED_ARCHITECTURES:
        for seed in post.EXPECTED_SEEDS:
            member_id = f"{architecture}-seed{seed}"
            members.append(
                {
                    "member_id": member_id,
                    "architecture": architecture,
                    "seed": seed,
                    "model_name": f"{member_id}-refit",
                    "prediction": {
                        "npz_path": str(project_root / f"{member_id}.npz"),
                        "npz_file_sha256": _hash("1"),
                        "sidecar_path": str(project_root / f"{member_id}.json"),
                        "sidecar_file_sha256": _hash("2"),
                        "artifact_sha256": _hash("3"),
                        "alignment_sha256": _hash("4"),
                        "record_count": 100,
                    },
                    "final_report": {
                        "path": str(project_root / f"{member_id}.report.json"),
                        "file_sha256": _hash("5"),
                        "artifact_sha256": _hash("6"),
                    },
                    "checkpoint": {
                        "path": str(project_root / f"{member_id}.ckpt"),
                        "file_sha256": _hash("7"),
                    },
                    "resolved_config": {
                        "path": str(project_root / f"{member_id}.config.json"),
                        "file_sha256": _hash("8"),
                        "config_hash": _hash("9"),
                    },
                    "calibration_decision": {
                        "path": str(project_root / f"{member_id}.decisions.json"),
                        "file_sha256": _hash("a"),
                        "artifact_sha256": _hash("b"),
                    },
                    "refit_lineage_sha256": _hash("c"),
                }
            )
    demo_member = members[0]
    body: dict[str, object] = {
        "schema_version": post.POST_EVALUATION_SPEC_SCHEMA_VERSION,
        "artifact_type": post.POST_EVALUATION_SPEC_TYPE,
        "protocol": {
            "protocol_hash": protocol.protocol_hash,
            "comparison_id": comparison_id,
            "manifest_sha256": _hash("d"),
            "normalization_sha256": _hash("e"),
            "label_order": list(LABEL_ORDER),
            "final_folds": [10],
        },
        "analysis_runtime": {
            "project_root": str(project_root.resolve()),
            "git_revision": "f" * 40,
            "clean": True,
        },
        "sealed_evaluation": {
            "final_batch_summary": {
                "path": str(project_root / "summary.json"),
                "file_sha256": _hash("1"),
                "artifact_sha256": _hash("2"),
                "batch_sha256": _hash("3"),
            },
            "opening_ledger": {
                "path": str(project_root / "ledger.json"),
                "file_sha256": _hash("4"),
                "ledger_sha256": _hash("5"),
                "state": "complete",
                "purpose": "synthetic final evaluation",
                "operator": "tester",
                "batch_sha256": _hash("3"),
                "opening_marker_path": str(project_root / "opening.json"),
                "opening_marker_file_sha256": _hash("6"),
                "opening_marker_sha256": _hash("7"),
                "terminal_event": "exact_six_member_final_batch_complete",
            },
            "final_evaluation_spec": {
                "path": str(project_root / "final-spec.json"),
                "file_sha256": _hash("7"),
                "artifact_sha256": _hash("8"),
                "evaluation_git_revision": "e" * 40,
            },
            "protocol_deviations": {
                "path": str(project_root / "PROTOCOL_DEVIATIONS.md"),
                "file_sha256": _hash("9"),
                "required_in_all_reporting": True,
            },
            "refit_bundle": {
                "path": str(project_root / "refit.json"),
                "file_sha256": _hash("a"),
                "artifact_sha256": _hash("b"),
                "protocol_hash": protocol.protocol_hash,
                "manifest_sha256": _hash("d"),
                "normalization_sha256": _hash("e"),
                "member_count": 6,
            },
            "calibration_bundle": {
                "path": str(project_root / "calibration.json"),
                "file_sha256": _hash("c"),
                "artifact_sha256": _hash("d"),
                "protocol_hash": protocol.protocol_hash,
                "manifest_sha256": _hash("d"),
                "normalization_sha256": _hash("e"),
                "member_count": 6,
            },
        },
        "members": members,
        "aggregate_outputs": {
            "architecture_summaries": [
                {
                    "architecture": architecture,
                    "path": str(project_root / f"{architecture}.json"),
                    "file_sha256": _hash("1"),
                    "artifact_sha256": _hash("2"),
                }
                for architecture in post.EXPECTED_ARCHITECTURES
            ],
            "paired_manifest": {
                "path": str(project_root / "paired.json"),
                "file_sha256": _hash("3"),
                "artifact_sha256": _hash("4"),
            },
            "paired_reports": [
                {
                    "seed": seed,
                    "alignment_sha256": _hash("5"),
                    "path": str(project_root / f"paired-{seed}.json"),
                    "file_sha256": _hash("6"),
                    "artifact_sha256": _hash("7"),
                }
                for seed in post.EXPECTED_SEEDS
            ],
        },
        "audit_protocols": {
            "robustness": post._robustness_protocol(),
            "explanations": {
                "cohort": _cohort(),
                "settings": post._explanation_settings(),
            },
            "demo": {
                "member_id": post.DEMO_MEMBER_ID,
                "architecture": "resnet1d",
                "seed": 2026,
                "target_coverage": post.DEMO_TARGET_COVERAGE,
                "entropy_method": "mean_normalized_binary_entropy",
                "gate": {"target_coverage": post.DEMO_TARGET_COVERAGE},
                "checkpoint": demo_member["checkpoint"],
                "resolved_config": demo_member["resolved_config"],
                "calibration_decision": demo_member["calibration_decision"],
                "selection_basis": ("fixed_operational_demo_default_not_fold10_model_selection"),
                "retuning_allowed": False,
            },
        },
        "output_contract": output,
    }
    payload = dict(body)
    payload["artifact_sha256"] = post.canonical_sha256(body)
    assert root == project_root / "runs" / "post_evaluation" / comparison_id
    return payload


def _saved_v1_spec(project_root: Path) -> post.PostEvaluationSpec:
    payload = _valid_payload(project_root)
    spec = post._spec_from_payload(payload, path=None)
    return post.save_post_evaluation_spec(
        spec,
        spec.output_root / post.POST_EVALUATION_FILENAME,
    )


def _v2_payload(
    old: post.PostEvaluationSpec,
    *,
    replacement_revision: int = 2,
) -> dict[str, object]:
    if old.path is None:
        raise AssertionError("saved v1 fixture must have a path")
    old_payload = old.payload
    body = copy.deepcopy(old_payload)
    del body["artifact_sha256"]
    body["schema_version"] = post.POST_EVALUATION_SUPERSESSION_SCHEMA_VERSION
    runtime = cast(dict[str, object], body["analysis_runtime"])
    runtime["git_revision"] = "e" * 40
    protocol = cast(dict[str, object], body["protocol"])
    project_root = Path(cast(str, runtime["project_root"]))
    body["output_contract"] = post._output_contract(
        project_root,
        cast(str, protocol["comparison_id"]),
        audit_revision=replacement_revision,
    )
    _, supersession = post._build_supersession_binding(
        old.path,
        reason=post.SUPERSESSION_REASON_DECIMAL_CASE_PATH_COLLISION,
    )
    body["supersession"] = supersession
    return {**body, "artifact_sha256": post.canonical_sha256(body)}


def test_canonical_robustness_matrix_has_exact_41_cases() -> None:
    protocol = post._robustness_protocol()
    cases = cast(list[dict[str, object]], protocol["severity_matrix"])
    assert len(cases) == 41
    assert cases[0] == {"case_id": "clean", "corruption": "clean", "parameters": {}}
    assert len({case["case_id"] for case in cases}) == 41
    assert protocol["clean_baseline_gate"] == {
        "required_before_corruptions": True,
        "comparison": "np.array_equal_against_sealed_raw_logits",
        "maximum_absolute_logit_error": 0.0,
        "failure_policy": "reject_all_corruption_results",
    }


def test_explanation_cohort_is_balanced_unique_and_deterministic() -> None:
    first = _cohort()
    second = _cohort()
    assert first == second
    records = cast(list[dict[str, object]], first["records"])
    assert len(records) == 60
    assert len({record["ecg_id"] for record in records}) == 60
    assert len({record["patient_id"] for record in records}) == 60
    cells: dict[tuple[object, object], int] = {}
    for record in records:
        cell = (record["target_label"], record["target_status"])
        cells[cell] = cells.get(cell, 0) + 1
        assert (
            cast(list[int], record["target_bits"])[cast(int, record["target_index"])]
            == (record["target_value"])
        )
    assert set(cells.values()) == {6}
    assert len(cells) == 10


def test_self_hashed_spec_round_trip_and_no_overwrite(tmp_path: Path) -> None:
    project_root = (tmp_path / "repo").resolve()
    project_root.mkdir()
    payload = _valid_payload(project_root)
    spec = post._spec_from_payload(payload, path=None)
    destination = spec.output_root / post.POST_EVALUATION_FILENAME
    saved = post.save_post_evaluation_spec(spec, destination)
    loaded = post.load_post_evaluation_spec(
        destination,
        protocol=ExperimentProtocol.canonical(),
        verify_sources=False,
        verify_git=False,
    )
    assert loaded.to_payload() == saved.to_payload()
    assert saved.path == destination.resolve()
    with pytest.raises(FileExistsError, match="already exists"):
        post.save_post_evaluation_spec(spec, destination)


def test_v2_supersession_coexists_and_preserves_release_bindings(tmp_path: Path) -> None:
    project_root = (tmp_path / "repo").resolve()
    project_root.mkdir()
    old = _saved_v1_spec(project_root)
    if old.path is None:
        raise AssertionError("saved v1 fixture must have a path")
    old_bytes = old.path.read_bytes()
    replacement = post._spec_from_payload(_v2_payload(old), path=None)
    destination = replacement.output_root / post.POST_EVALUATION_FILENAME
    post.save_post_evaluation_spec(replacement, destination)
    loaded = post.load_post_evaluation_spec(
        destination,
        protocol=ExperimentProtocol.canonical(),
        verify_sources=False,
        verify_git=False,
    )

    assert old.path.read_bytes() == old_bytes
    assert old.path.is_file() and destination.is_file()
    assert old.output_root != loaded.output_root
    assert loaded.output_root.name == "ptbxl_matched_equal_budget_v1__audit-r2"
    assert loaded.payload["schema_version"] == 2
    for key in (
        "protocol",
        "sealed_evaluation",
        "members",
        "aggregate_outputs",
        "audit_protocols",
    ):
        assert loaded.payload[key] == old.payload[key]
    with pytest.raises(FileExistsError, match="already exists"):
        post.save_post_evaluation_spec(replacement, destination)


@pytest.mark.parametrize("mutation", ["missing", "tampered"])
def test_v2_rejects_missing_or_tampered_superseded_spec(
    tmp_path: Path,
    mutation: str,
) -> None:
    project_root = (tmp_path / "repo").resolve()
    project_root.mkdir()
    old = _saved_v1_spec(project_root)
    payload = _v2_payload(old)
    if old.path is None:
        raise AssertionError("saved v1 fixture must have a path")
    if mutation == "missing":
        old.path.unlink()
    else:
        old.path.write_bytes(old.path.read_bytes() + b"tampered\n")

    with pytest.raises(post.PostEvaluationIntegrityError, match="missing|changed"):
        post._spec_from_payload(payload, path=None)


@pytest.mark.parametrize("field", ["path", "output_root"])
def test_v2_rejects_noncanonical_superseded_path_spelling(
    tmp_path: Path,
    field: str,
) -> None:
    project_root = (tmp_path / "repo").resolve()
    project_root.mkdir()
    old = _saved_v1_spec(project_root)
    payload = _v2_payload(old)
    supersession = cast(dict[str, object], payload["supersession"])
    binding = cast(dict[str, object], supersession["superseded_spec"])
    canonical = Path(cast(str, binding[field]))
    binding[field] = (
        str(canonical / "unused" / "..")
        if field == "output_root"
        else str(canonical.parent / "unused" / ".." / canonical.name)
    )
    unhashed = dict(payload)
    del unhashed["artifact_sha256"]
    payload["artifact_sha256"] = post.canonical_sha256(unhashed)

    with pytest.raises(post.PostEvaluationIntegrityError, match="canonical absolute spelling"):
        post._spec_from_payload(payload, path=None)


def test_v2_recomputes_the_complete_superseded_output_tree(tmp_path: Path) -> None:
    project_root = (tmp_path / "repo").resolve()
    project_root.mkdir()
    old = _saved_v1_spec(project_root)
    replacement = post._spec_from_payload(_v2_payload(old), path=None)
    destination = replacement.output_root / post.POST_EVALUATION_FILENAME
    post.save_post_evaluation_spec(replacement, destination)
    added = old.output_root / "late-derived-file.json"
    added.write_text("{}\n", encoding="utf-8")

    with pytest.raises(post.PostEvaluationIntegrityError, match="output tree changed"):
        post.load_post_evaluation_spec(
            destination,
            protocol=ExperimentProtocol.canonical(),
            verify_sources=False,
            verify_git=False,
        )


@pytest.mark.parametrize(
    "section",
    [
        "protocol",
        "sealed_evaluation",
        "members",
        "aggregate_outputs",
        "audit_protocols",
    ],
)
def test_v2_rejects_rehashed_release_facing_section_drift(
    tmp_path: Path,
    section: str,
) -> None:
    project_root = (tmp_path / "repo").resolve()
    project_root.mkdir()
    old = _saved_v1_spec(project_root)
    payload = _v2_payload(old)
    if section == "protocol":
        cast(dict[str, object], payload[section])["manifest_sha256"] = _hash("0")
    elif section == "sealed_evaluation":
        sealed = cast(dict[str, object], payload[section])
        cast(dict[str, object], sealed["final_batch_summary"])["file_sha256"] = _hash("0")
    elif section == "members":
        cast(list[dict[str, object]], payload[section])[0]["model_name"] = "changed"
    elif section == "aggregate_outputs":
        aggregate = cast(dict[str, object], payload[section])
        cast(list[dict[str, object]], aggregate["architecture_summaries"])[0]["file_sha256"] = (
            _hash("0")
        )
    else:
        protocols = cast(dict[str, object], payload[section])
        cast(dict[str, object], protocols["demo"])["selection_basis"] = "changed"
    unhashed = dict(payload)
    del unhashed["artifact_sha256"]
    payload["artifact_sha256"] = post.canonical_sha256(unhashed)

    with pytest.raises(post.PostEvaluationIntegrityError, match=f"audit {section} differs"):
        post._spec_from_payload(payload, path=None)


@pytest.mark.parametrize("mutation", ["unsorted", "duplicate", "edited"])
def test_v2_rejects_noncanonical_or_edited_tree_entries(
    tmp_path: Path,
    mutation: str,
) -> None:
    project_root = (tmp_path / "repo").resolve()
    project_root.mkdir()
    old = _saved_v1_spec(project_root)
    (old.output_root / "z-last.txt").write_text("bound\n", encoding="utf-8")
    payload = _v2_payload(old)
    supersession = cast(dict[str, object], payload["supersession"])
    snapshot = cast(dict[str, object], supersession["output_tree"])
    files = cast(list[dict[str, object]], snapshot["files"])
    if mutation == "unsorted":
        files.reverse()
    elif mutation == "duplicate":
        files.append(copy.deepcopy(files[-1]))
        snapshot["file_count"] = len(files)
    else:
        files[0]["size_bytes"] = cast(int, files[0]["size_bytes"]) + 1
    snapshot["tree_sha256"] = post.canonical_sha256({"files": files})
    unhashed = dict(payload)
    del unhashed["artifact_sha256"]
    payload["artifact_sha256"] = post.canonical_sha256(unhashed)

    with pytest.raises(
        post.PostEvaluationIntegrityError,
        match="uniquely sorted|output tree changed",
    ):
        post._spec_from_payload(payload, path=None)


def test_superseded_tree_rejects_a_junction_before_descent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = (tmp_path / "repo").resolve()
    project_root.mkdir()
    old = _saved_v1_spec(project_root)
    linked = old.output_root / "linked"
    linked.mkdir()
    monkeypatch.setattr(post, "_is_junction", lambda path: path.name == "linked")

    with pytest.raises(post.PostEvaluationIntegrityError, match="link or junction"):
        post._output_tree_snapshot(old.output_root)


def test_v2_rejects_reused_or_skipped_revision_roots(tmp_path: Path) -> None:
    project_root = (tmp_path / "repo").resolve()
    project_root.mkdir()
    old = _saved_v1_spec(project_root)

    reused = _v2_payload(old)
    reused["output_contract"] = post._output_contract(
        project_root,
        "ptbxl_matched_equal_budget_v1",
    )
    unhashed = dict(reused)
    del unhashed["artifact_sha256"]
    reused["artifact_sha256"] = post.canonical_sha256(unhashed)
    with pytest.raises(post.PostEvaluationIntegrityError, match="schema and output-root"):
        post._spec_from_payload(reused, path=None)

    skipped = _v2_payload(old, replacement_revision=999)
    with pytest.raises(post.PostEvaluationIntegrityError, match="next distinct sibling"):
        post._spec_from_payload(skipped, path=None)


def test_v2_rejects_supersession_at_the_same_git_revision(tmp_path: Path) -> None:
    project_root = (tmp_path / "repo").resolve()
    project_root.mkdir()
    old = _saved_v1_spec(project_root)
    payload = _v2_payload(old)
    runtime = cast(dict[str, object], payload["analysis_runtime"])
    runtime["git_revision"] = cast(dict[str, object], old.payload["analysis_runtime"])[
        "git_revision"
    ]
    unhashed = dict(payload)
    del unhashed["artifact_sha256"]
    payload["artifact_sha256"] = post.canonical_sha256(unhashed)

    with pytest.raises(post.PostEvaluationIntegrityError, match="different committed Git"):
        post._spec_from_payload(payload, path=None)


@pytest.mark.parametrize(
    "relative_manifest",
    [Path("robustness/manifest.json"), Path("derived_artifacts.manifest.json")],
)
def test_v2_requires_both_incomplete_manifests_to_be_absent(
    tmp_path: Path,
    relative_manifest: Path,
) -> None:
    project_root = (tmp_path / "repo").resolve()
    project_root.mkdir()
    old = _saved_v1_spec(project_root)
    manifest = old.output_root / relative_manifest
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}\n", encoding="utf-8")

    with pytest.raises(post.PostEvaluationIntegrityError, match="must be absent"):
        post._build_supersession_binding(
            cast(Path, old.path),
            reason=post.SUPERSESSION_REASON_DECIMAL_CASE_PATH_COLLISION,
        )


def test_rehashed_policy_or_output_root_drift_is_rejected(tmp_path: Path) -> None:
    project_root = (tmp_path / "repo").resolve()
    project_root.mkdir()
    payload = _valid_payload(project_root)
    protocols = cast(dict[str, object], payload["audit_protocols"])
    robustness = cast(dict[str, object], protocols["robustness"])
    robustness["random_seed"] = 1
    unhashed = dict(payload)
    del unhashed["artifact_sha256"]
    payload["artifact_sha256"] = post.canonical_sha256(unhashed)
    with pytest.raises(post.PostEvaluationIntegrityError, match="severity matrix"):
        post._spec_from_payload(payload, path=None)

    escaped = _valid_payload(project_root)
    escaped["output_contract"] = post._output_contract(project_root, "different")
    unhashed = dict(escaped)
    del unhashed["artifact_sha256"]
    escaped["artifact_sha256"] = post.canonical_sha256(unhashed)
    with pytest.raises(post.PostEvaluationIntegrityError, match="output contract"):
        post._spec_from_payload(escaped, path=None)


def test_clean_git_capture_rejects_dirty_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = tmp_path.resolve()
    monkeypatch.setattr(
        post,
        "_run_git",
        lambda root, *args: (
            str(root)
            if args == ("rev-parse", "--show-toplevel")
            else " M source.py"
            if args[0] == "status"
            else "a" * 40
        ),
    )
    with pytest.raises(post.PostEvaluationError, match="clean committed"):
        post._capture_clean_git(project_root)


def test_cli_defaults_remain_v1_and_supersession_is_explicit() -> None:
    args = build_parser().parse_args([])
    assert args.output is None
    assert args.output_root is None
    assert args.supersedes_spec is None
    assert args.supersession_reason is None
    assert args.opening_ledger is None

    replacement = build_parser().parse_args(
        [
            "--supersedes-spec",
            "old/audit_spec.json",
            "--supersession-reason",
            post.SUPERSESSION_REASON_DECIMAL_CASE_PATH_COLLISION,
        ]
    )
    assert replacement.supersedes_spec == Path("old/audit_spec.json")
    assert replacement.supersession_reason == post.SUPERSESSION_REASON_DECIMAL_CASE_PATH_COLLISION
