from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.build_trust_sentinel_ood_v2_inventory as cli
from ecg_trust.constants import LEADS
from ecg_trust.ood_v2 import inventory as inventory_module
from ecg_trust.ood_v2.inventory import (
    ExternalInventoryError,
    load_external_inventory,
)


def _write_wfdb_pair(root: Path, record_ref: str, marker: bytes) -> None:
    base = root / record_ref
    base.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{base}.hea").write_bytes(b"header-" + marker)
    Path(f"{base}.dat").write_bytes(b"data-" + marker)


def _fixture_inputs(tmp_path: Path) -> cli.InventoryInputPaths:
    challenge_root = tmp_path / "challenge"
    zzu_root = tmp_path / "zzu"
    _write_wfdb_pair(challenge_root, "c001", b"c001")
    _write_wfdb_pair(challenge_root, "c002", b"c002")
    _write_wfdb_pair(zzu_root, "P00/P00001/P00001_E01", b"z001")
    _write_wfdb_pair(zzu_root, "P00/P00002/P00002_E01", b"z002")
    _write_wfdb_pair(zzu_root, "P00/P00003/P00003_E01", b"z003")

    metadata = tmp_path / "metadata"
    metadata.mkdir()
    records = metadata / "RECORDS"
    acceptable = metadata / "RECORDS-acceptable"
    unacceptable = metadata / "RECORDS-unacceptable"
    zzu_metadata = metadata / "zzu.csv"
    records.write_text("c001\nc002\n", encoding="utf-8")
    acceptable.write_text("c001\n", encoding="utf-8")
    unacceptable.write_text("c002\n", encoding="utf-8")
    zzu_metadata.write_text(
        "Filename,ECG_ID,Patient_ID,Sampling_point,Lead,Note\n"
        'P00/P00001/P00001_E01,P00001_E01,P00001,5500,12,"quoted, value"\n'
        "P00/P00002/P00002_E01,P00002_E01,P00002,4500,12,plain\n"
        "P00/P00003/P00003_E01,P00003_E01,P00003,5000,9,plain\n",
        encoding="utf-8",
    )
    return cli.InventoryInputPaths(
        challenge_root=challenge_root,
        challenge_records=records,
        challenge_acceptable=acceptable,
        challenge_unacceptable=unacceptable,
        zzu_root=zzu_root,
        zzu_metadata=zzu_metadata,
    )


@pytest.fixture
def header_only_wfdb(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    observed: list[str] = []

    def fake_rdheader(record_base: str) -> SimpleNamespace:
        observed.append(record_base)
        name = Path(record_base).name
        data_file_names = [f"{name}.dat"] * 12
        if name.startswith("c"):
            return SimpleNamespace(
                fs=500.0,
                sig_len=5_000,
                n_sig=12,
                sig_name=list(LEADS),
                units=["mV"] * 12,
                file_name=data_file_names,
            )
        if name == "P00002_E01":
            return SimpleNamespace(
                fs=500.0,
                sig_len=4_500,
                n_sig=12,
                sig_name=list(LEADS),
                units=["mV"] * 12,
                file_name=data_file_names,
            )
        if name == "P00003_E01":
            return SimpleNamespace(
                fs=500.0,
                sig_len=5_000,
                n_sig=9,
                sig_name=list(LEADS[:9]),
                units=["mV"] * 9,
                file_name=[f"{name}.dat"] * 9,
            )
        return SimpleNamespace(
            fs=500.0,
            sig_len=5_500,
            n_sig=12,
            sig_name=list(LEADS),
            units=["mV"] * 12,
            file_name=data_file_names,
        )

    monkeypatch.setattr(inventory_module.wfdb, "rdheader", fake_rdheader)
    monkeypatch.setattr(
        inventory_module.wfdb,
        "rdrecord",
        lambda *args, **kwargs: pytest.fail("inventory CLI must never decode a waveform"),
    )
    return observed


def _expectations() -> cli.InventoryExpectations:
    return cli.InventoryExpectations(
        challenge_records=2,
        zzu_records=3,
        zzu_patients=3,
        zzu_twelve_lead_records=2,
        zzu_nine_lead_records=1,
    )


def test_builds_header_only_private_inventory_and_aggregate_projection(
    tmp_path: Path,
    header_only_wfdb: list[str],
) -> None:
    inputs = _fixture_inputs(tmp_path)

    artifacts = cli.build_inventory_artifacts(
        inputs,
        zzu_schema=cli.ZZUMetadataSchema(),
        expectations=_expectations(),
    )

    assert artifacts.inventory.record_count == 3
    assert artifacts.challenge_record_count == 2
    assert artifacts.zzu_candidate_record_count == 3
    assert artifacts.zzu_patient_count == 3
    assert artifacts.zzu_build_summary.selected_record_count == 1
    assert artifacts.zzu_build_summary.excluded_record_count == 2
    assert artifacts.zzu_build_summary.exclusion_counts == {
        "pediatric_12_lead_flag_false": 1,
        "sampling_frequency_not_500_hz": 0,
        "duration_under_10_seconds": 1,
        "lead_count_not_12": 0,
        "noncanonical_lead_set": 0,
    }
    assert cli.verify_public_projection(artifacts.public_projection).startswith("sha256:")
    encoded = json.dumps(artifacts.public_projection, sort_keys=True)
    for forbidden in (
        "c001",
        "c002",
        "z001",
        "z002-short",
        "z003-nine",
        "zzu_pecg_v1:p001",
        cli.CHALLENGE_PRIVATE_SITE,
        cli.ZZU_PRIVATE_SITE,
    ):
        assert forbidden not in encoded
    assert header_only_wfdb


def test_writes_create_new_artifacts_then_reloads_and_reverifies(
    tmp_path: Path,
    header_only_wfdb: list[str],
) -> None:
    inputs = _fixture_inputs(tmp_path)
    artifacts = cli.build_inventory_artifacts(
        inputs,
        zzu_schema=cli.ZZUMetadataSchema(),
        expectations=_expectations(),
    )
    private_output = tmp_path / "outputs" / "private-inventory.json"
    public_output = tmp_path / "outputs" / "public-projection.json"

    same_output = tmp_path / "outputs" / "same.json"
    with pytest.raises(cli.InventoryCLIError, match="must be distinct"):
        cli.write_inventory_artifacts(
            artifacts,
            inputs=inputs,
            private_output=same_output,
            public_output=same_output,
        )
    assert not same_output.exists()

    result = cli.write_inventory_artifacts(
        artifacts,
        inputs=inputs,
        private_output=private_output,
        public_output=public_output,
    )

    assert result.inventory_sha256 == artifacts.inventory.inventory_sha256
    assert result.public_projection_sha256 == artifacts.public_projection_sha256
    assert result.challenge_record_count == 2
    assert result.zzu_selected_record_count == 1
    assert result.zzu_excluded_record_count == 2
    assert load_external_inventory(private_output) == artifacts.inventory
    assert private_output.read_bytes() == artifacts.inventory.to_canonical_json_bytes()
    assert public_output.read_bytes().endswith(b"\n")
    with pytest.raises(cli.InventoryCLIError, match="already exists"):
        cli.write_inventory_artifacts(
            artifacts,
            inputs=inputs,
            private_output=private_output,
            public_output=public_output,
        )
    assert header_only_wfdb


def test_zzu_metadata_schema_is_explicit_and_preserves_full_mapping(tmp_path: Path) -> None:
    metadata = tmp_path / "zzu.tsv"
    metadata.write_text(
        "rid\teid\tpid\tpoints\tleads\nP00/P00001/P00001_E01\tP00001_E01\tP00001\t6000\t12\n",
        encoding="utf-8",
    )

    candidates = cli.read_zzu_candidates(
        metadata,
        schema=cli.ZZUMetadataSchema(
            record_column="rid",
            ecg_id_column="eid",
            patient_column="pid",
            lead_count_column="leads",
            sampling_point_column="points",
            delimiter="\t",
        ),
    )

    assert len(candidates) == 1
    assert candidates[0].record_ref == "P00/P00001/P00001_E01"
    assert candidates[0].ecg_id == "P00001_E01"
    assert candidates[0].patient_key == "zzu_pecg_v1:P00001"
    assert candidates[0].declared_lead_count == 12
    assert candidates[0].pediatric_12_lead is True
    assert candidates[0].declared_sample_count == 6_000


@pytest.mark.parametrize(
    "payload",
    [
        "Filename,ECG_ID,Patient_ID,Patient_ID,Sampling_point\n"
        "P00/P00001/P00001_E01,P00001_E01,P00001,12,5000\n",
        "Filename,ECG_ID,Patient_ID,Lead,Sampling_point\n"
        "P00/P00001/P00001_E01,P00001_E01,P00001,8,5000\n",
        "Filename,ECG_ID,Patient_ID,Lead,Sampling_point\n"
        "P00/P00001/P00001_E01,P00001_E01,P00001,12,5000\n"
        "P00/P00001/P00001_E01,P00001_E01,P00001,12,5000\n",
    ],
)
def test_metadata_duplicates_and_noncanonical_counts_fail_closed(
    tmp_path: Path,
    payload: str,
) -> None:
    metadata = tmp_path / "zzu.csv"
    metadata.write_text(payload, encoding="utf-8")
    with pytest.raises(cli.InventoryCLIError):
        cli.read_zzu_candidates(metadata, schema=cli.ZZUMetadataSchema())


def test_public_projection_tamper_is_rejected(
    tmp_path: Path,
    header_only_wfdb: list[str],
) -> None:
    artifacts = cli.build_inventory_artifacts(
        _fixture_inputs(tmp_path),
        zzu_schema=cli.ZZUMetadataSchema(),
        expectations=_expectations(),
    )
    tampered = dict(artifacts.public_projection)
    tampered["challenge_record_count"] = 99

    with pytest.raises(cli.InventoryCLIError):
        cli.verify_public_projection(tampered)

    nested_tamper = copy.deepcopy(artifacts.public_projection)
    nested_inventory = nested_tamper["inventory"]
    assert isinstance(nested_inventory, dict)
    groups = nested_inventory["groups"]
    assert isinstance(groups, list)
    groups[0]["record_ref"] = "private-record"
    outer_body = dict(nested_tamper)
    del outer_body["projection_sha256"]
    nested_tamper["projection_sha256"] = (
        "sha256:"
        + hashlib.sha256(
            cli.PUBLIC_ARTIFACT_DOMAIN + cli._canonical_json_bytes(outer_body)
        ).hexdigest()
    )
    with pytest.raises(cli.InventoryCLIError, match="allowlist"):
        cli.verify_public_projection(nested_tamper)
    assert header_only_wfdb


def test_main_prints_only_aggregate_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = cli.InventoryCLIResult(
        inventory_sha256="sha256:" + "a" * 64,
        public_projection_sha256="sha256:" + "b" * 64,
        challenge_record_count=1_000,
        zzu_candidate_record_count=14_190,
        zzu_selected_record_count=12_000,
        zzu_excluded_record_count=2_190,
    )
    monkeypatch.setattr(cli, "run_inventory_build", lambda arguments: result)
    private_path = tmp_path / "private-name.json"
    public_path = tmp_path / "public-name.json"

    status = cli.main(
        [
            "--implementation-revision",
            "a" * 40,
            "--challenge-root",
            str(tmp_path / "challenge-secret"),
            "--challenge-archive",
            str(tmp_path / "challenge-archive-secret"),
            "--challenge-records",
            str(tmp_path / "RECORDS-secret"),
            "--challenge-acceptable",
            str(tmp_path / "acceptable-secret"),
            "--challenge-unacceptable",
            str(tmp_path / "unacceptable-secret"),
            "--zzu-root",
            str(tmp_path / "zzu-secret"),
            "--zzu-archive-z01",
            str(tmp_path / "zzu-archive-z01-secret"),
            "--zzu-archive-zip",
            str(tmp_path / "zzu-archive-zip-secret"),
            "--seven-zip-executable",
            str(tmp_path / "seven-zip-secret"),
            "--zzu-metadata",
            str(tmp_path / "metadata-secret.csv"),
            "--private-output",
            str(private_path),
            "--public-output",
            str(public_path),
        ]
    )
    captured = capsys.readouterr()
    output = json.loads(captured.out)

    assert status == 0
    assert captured.err == ""
    assert output["status"] == "OOD_V2_INVENTORY_FROZEN"
    assert output["challenge_record_count"] == 1_000
    assert output["zzu_candidate_record_count"] == 14_190
    assert output["zzu_selected_record_count"] == 12_000
    assert "secret" not in captured.out
    assert "private-name" not in captured.out
    assert "public-name" not in captured.out
    assert "expected_challenge_records" not in vars(
        cli._parser().parse_args(
            [
                "--implementation-revision",
                "a" * 40,
                "--challenge-root",
                "c",
                "--challenge-archive",
                "ca",
                "--challenge-records",
                "r",
                "--challenge-acceptable",
                "a",
                "--challenge-unacceptable",
                "u",
                "--zzu-root",
                "z",
                "--zzu-archive-z01",
                "z1",
                "--zzu-archive-zip",
                "zz",
                "--seven-zip-executable",
                "7z",
                "--zzu-metadata",
                "m",
                "--private-output",
                "p",
                "--public-output",
                "q",
            ]
        )
    )


def test_main_failure_message_is_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "run_inventory_build",
        lambda arguments: (_ for _ in ()).throw(
            ExternalInventoryError("private patient p001 at C:/secret")
        ),
    )

    status = cli.main(
        [
            "--implementation-revision",
            "a" * 40,
            "--challenge-root",
            str(tmp_path),
            "--challenge-archive",
            str(tmp_path / "ca"),
            "--challenge-records",
            str(tmp_path / "r"),
            "--challenge-acceptable",
            str(tmp_path / "a"),
            "--challenge-unacceptable",
            str(tmp_path / "u"),
            "--zzu-root",
            str(tmp_path),
            "--zzu-archive-z01",
            str(tmp_path / "z01"),
            "--zzu-archive-zip",
            str(tmp_path / "zip"),
            "--seven-zip-executable",
            str(tmp_path / "7z"),
            "--zzu-metadata",
            str(tmp_path / "m"),
            "--private-output",
            str(tmp_path / "private"),
            "--public-output",
            str(tmp_path / "public"),
        ]
    )
    captured = capsys.readouterr()

    assert status == 1
    assert captured.out == ""
    assert captured.err == (
        "OOD_V2_INVENTORY_FAILED: inspect private local inputs and immutable output state.\n"
    )
    assert "p001" not in captured.err
    assert "secret" not in captured.err


def test_production_inventory_outputs_require_isolation_and_exact_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    artifacts = root / "artifacts" / "trust_sentinel"
    artifacts.mkdir(parents=True)
    private_output = root / cli.SUCCESSOR_PRIVATE_INVENTORY_PATH
    public_output = root / cli.SUCCESSOR_PUBLIC_PROJECTION_PATH

    with pytest.raises(cli.InventoryCLIError, match="requires the isolated launcher"):
        cli._verify_production_output_destinations(private_output, public_output)

    monkeypatch.setattr(cli, "_ISOLATED_RUNTIME_ACTIVE", True)
    monkeypatch.setattr(
        cli,
        "_project_layout",
        lambda: (
            root / "scripts" / "inventory.py",
            root,
            root / ".venv" / "Lib" / "site-packages",
            root / "src",
        ),
    )
    cli._verify_production_output_destinations(private_output, public_output)

    with pytest.raises(cli.InventoryCLIError, match="exact successor namespace"):
        cli._verify_production_output_destinations(
            root / "artifacts" / "private" / "inventory.json",
            public_output,
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows junction contract")
def test_production_inventory_outputs_reject_intermediate_junction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    trust_root = root / "artifacts" / "trust_sentinel"
    redirected = root / "redirected"
    trust_root.mkdir(parents=True)
    redirected.mkdir()
    junction = trust_root / "ood_external_v2_1_preflight"
    completed = subprocess.run(
        [
            r"C:\Windows\System32\cmd.exe",
            "/d",
            "/c",
            "mklink",
            "/J",
            os.fspath(junction),
            os.fspath(redirected),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert junction.is_junction()

    monkeypatch.setattr(cli, "_ISOLATED_RUNTIME_ACTIVE", True)
    monkeypatch.setattr(
        cli,
        "_project_layout",
        lambda: (
            root / "scripts" / "inventory.py",
            root,
            root / ".venv" / "Lib" / "site-packages",
            root / "src",
        ),
    )
    try:
        with pytest.raises(cli.InventoryCLIError, match="indirect component"):
            cli._verify_production_output_destinations(
                root / cli.SUCCESSOR_PRIVATE_INVENTORY_PATH,
                root / cli.SUCCESSOR_PUBLIC_PROJECTION_PATH,
            )
    finally:
        os.rmdir(junction)


def test_inventory_builder_preflight_runs_before_any_raw_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = Path("configs/parent.yaml")
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "_project_layout",
        lambda: (
            tmp_path / "scripts" / "inventory.py",
            tmp_path,
            tmp_path / ".venv" / "Lib" / "site-packages",
            tmp_path / "src",
        ),
    )

    def refuse_preflight(
        parent_path: Path,
        project_root: Path,
        implementation_revision: str,
    ) -> None:
        observed.update(
            parent_path=parent_path,
            project_root=project_root,
            implementation_revision=implementation_revision,
        )
        raise cli.InventoryCLIError("preflight refused")

    monkeypatch.setattr(cli, "verify_inventory_builder_preflight", refuse_preflight)
    monkeypatch.setattr(
        cli,
        "build_inventory_artifacts",
        lambda *_args, **_kwargs: pytest.fail("raw inventory build ran before preflight"),
    )

    with pytest.raises(cli.InventoryCLIError, match="preflight refused"):
        cli.run_inventory_build(
            SimpleNamespace(
                parent=parent,
                implementation_revision="a" * 40,
            )
        )

    assert observed == {
        "parent_path": parent,
        "project_root": tmp_path,
        "implementation_revision": "a" * 40,
    }


def test_inventory_builder_postflight_closes_exact_in_memory_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    preflight = object()
    private_artifact_sha256 = "sha256:" + "1" * 64
    public_artifact_sha256 = "sha256:" + "2" * 64
    private_bytes = b"private-canonical\n"
    public_projection = {"projection_sha256": public_artifact_sha256}
    public_bytes = cli._canonical_json_bytes(public_projection) + b"\n"
    artifacts = SimpleNamespace(
        inventory=SimpleNamespace(
            inventory_sha256=private_artifact_sha256,
            to_canonical_json_bytes=lambda: private_bytes,
        ),
        public_projection=public_projection,
    )
    result = cli.InventoryCLIResult(
        inventory_sha256=private_artifact_sha256,
        public_projection_sha256=public_artifact_sha256,
        challenge_record_count=1_000,
        zzu_candidate_record_count=14_190,
        zzu_selected_record_count=12_328,
        zzu_excluded_record_count=1_862,
    )
    monkeypatch.setattr(
        cli,
        "_project_layout",
        lambda: (
            tmp_path / "scripts" / "inventory.py",
            tmp_path,
            tmp_path / ".venv" / "Lib" / "site-packages",
            tmp_path / "src",
        ),
    )
    monkeypatch.setattr(
        cli,
        "verify_inventory_builder_preflight",
        lambda *_args: calls.append("preflight") or preflight,
    )
    monkeypatch.setattr(
        cli,
        "_verify_production_output_destinations",
        lambda *_args: calls.append("destinations"),
    )
    monkeypatch.setattr(
        cli,
        "build_inventory_artifacts",
        lambda *_args, **_kwargs: calls.append("build") or artifacts,
    )
    monkeypatch.setattr(
        cli,
        "write_inventory_artifacts",
        lambda *_args, **_kwargs: calls.append("write") or result,
    )
    observed_postflight: dict[str, object] = {}

    def postflight(boundary: object, **kwargs: object) -> None:
        calls.append("postflight")
        observed_postflight.update(boundary=boundary, **kwargs)

    monkeypatch.setattr(cli, "verify_inventory_builder_postflight", postflight)
    arguments = SimpleNamespace(
        parent=Path("configs/parent.yaml"),
        implementation_revision="a" * 40,
        challenge_root=tmp_path / "challenge",
        challenge_archive=tmp_path / "challenge.tar.gz",
        challenge_records=tmp_path / "RECORDS",
        challenge_acceptable=tmp_path / "RECORDS-acceptable",
        challenge_unacceptable=tmp_path / "RECORDS-unacceptable",
        zzu_root=tmp_path / "zzu",
        zzu_archive_z01=tmp_path / "zzu.z01",
        zzu_archive_zip=tmp_path / "zzu.zip",
        seven_zip_executable=tmp_path / "7z.exe",
        zzu_metadata=tmp_path / "attributes.csv",
        zzu_record_column="Filename",
        zzu_ecg_id_column="ECG_ID",
        zzu_patient_column="Patient_ID",
        zzu_lead_count_column="Lead",
        zzu_sampling_point_column="Sampling_point",
        zzu_delimiter="comma",
        private_output=tmp_path / "private.json",
        public_output=tmp_path / "public.json",
    )

    assert cli.run_inventory_build(arguments) is result
    assert calls == ["preflight", "destinations", "build", "write", "postflight"]
    assert observed_postflight == {
        "boundary": preflight,
        "parent_path": arguments.parent,
        "project_root": tmp_path,
        "implementation_revision": "a" * 40,
        "inventory_path": arguments.private_output,
        "public_projection_path": arguments.public_output,
        "expected_inventory_file_sha256": ("sha256:" + hashlib.sha256(private_bytes).hexdigest()),
        "expected_inventory_sha256": private_artifact_sha256,
        "expected_public_projection_file_sha256": (
            "sha256:" + hashlib.sha256(public_bytes).hexdigest()
        ),
        "expected_public_projection_artifact_sha256": public_artifact_sha256,
    }


def test_cli_archive_closure_requires_exact_full_release_role_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    challenge_root = tmp_path / "set-a"
    challenge_root.mkdir()
    records = challenge_root / "RECORDS"
    acceptable = challenge_root / "RECORDS-acceptable"
    unacceptable = challenge_root / "RECORDS-unacceptable"
    for path in (records, acceptable, unacceptable):
        path.write_text("c001\n", encoding="utf-8")
    inputs = cli.InventoryInputPaths(
        challenge_root=challenge_root,
        challenge_records=records,
        challenge_acceptable=acceptable,
        challenge_unacceptable=unacceptable,
        zzu_root=tmp_path / "Child_ecg",
        zzu_metadata=tmp_path / "attributes.csv",
        challenge_archive=tmp_path / "set-a.tar.gz",
        zzu_archive_z01=tmp_path / "Child_ecg.z01",
        zzu_archive_zip=tmp_path / "Child_ecg.zip",
        seven_zip_executable=tmp_path / "7z.exe",
    )
    challenge_record = SimpleNamespace(record_ref="c001")
    candidate = SimpleNamespace(record_ref="P00/P00001/P00001_E01")
    challenge_members = tuple(
        SimpleNamespace(extracted_relative_path=path, role=role)
        for path, role in (
            ("c001.hea", "wfdb_header"),
            ("c001.dat", "wfdb_data"),
            ("c001.txt", "ignored_release_file"),
            ("HEADER.shtml", "ignored_release_file"),
            ("RECORDS", "quality_reference"),
            ("RECORDS-acceptable", "quality_reference"),
            ("RECORDS-unacceptable", "quality_reference"),
        )
    )
    zzu_members = (
        SimpleNamespace(
            extracted_relative_path="P00/P00001/P00001_E01.hea",
            role="wfdb_header",
        ),
        SimpleNamespace(
            extracted_relative_path="P00/P00001/P00001_E01.dat",
            role="wfdb_data",
        ),
    )
    challenge_closure = SimpleNamespace(
        dataset=cli.CHALLENGE_2011_DATASET,
        members=challenge_members,
    )
    zzu_closure = SimpleNamespace(dataset=cli.ZZU_PEDIATRIC_DATASET, members=zzu_members)
    monkeypatch.setattr(
        cli,
        "build_challenge_tar_extraction_closure",
        lambda *args, **kwargs: challenge_closure,
    )
    monkeypatch.setattr(
        cli,
        "build_zzu_split_zip_extraction_closure",
        lambda *args, **kwargs: zzu_closure,
    )

    closures = cli._build_archive_closures(inputs, (challenge_record,), (candidate,))
    assert closures == (challenge_closure, zzu_closure)

    challenge_closure.members = challenge_members[:-1]
    with pytest.raises(cli.InventoryCLIError, match="full official release"):
        cli._build_archive_closures(inputs, (challenge_record,), (candidate,))
