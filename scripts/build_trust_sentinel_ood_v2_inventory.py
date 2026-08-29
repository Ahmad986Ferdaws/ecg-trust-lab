#!/usr/bin/env python3
"""Freeze the header-only external input inventory for Trust Sentinel OOD v2.

This command is deliberately pre-inference.  It reads official text/CSV
metadata, WFDB headers, and raw file bytes for hashing.  It never decodes a
waveform, runs quality logic, loads a model, or calculates a score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import cast

_ISOLATED_CHILD_FLAG = "--_ecg-trust-ood-v2-isolated-child"
_RUNTIME_ROOT_PREFIX = ".ood_external_v2_1.runtime-"
_HANDOFF_FILENAME = ".parent-handoff"
_FROZEN_WINDOWS_DIRECTORY = Path(r"C:\Windows")
_FROZEN_PROGRAM_FILES_DIRECTORY = Path(r"C:\Program Files")
_FROZEN_PROGRAM_FILES_X86_DIRECTORY = Path(r"C:\Program Files (x86)")
_FROZEN_PROGRAM_DATA_DIRECTORY = Path(r"C:\ProgramData")
_ISOLATED_RUNTIME_ACTIVE = False


def _is_indirect(path: Path) -> bool:
    junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(junction is not None and junction())


def _direct_path(path: Path, *, directory: bool) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    for component in (lexical, *lexical.parents):
        if _is_indirect(component):
            raise RuntimeError("isolated launcher path is indirect")
    resolved = lexical.resolve(strict=True)
    expected_kind = resolved.is_dir() if directory else resolved.is_file()
    if resolved != lexical or not expected_kind:
        raise RuntimeError("isolated launcher path is unavailable")
    return resolved


def _project_layout() -> tuple[Path, Path, Path, Path]:
    script = _direct_path(Path(__file__), directory=False)
    root = _direct_path(script.parent.parent, directory=True)
    executable = _direct_path(root / ".venv" / "Scripts" / "python.exe", directory=False)
    if _direct_path(Path(sys.executable), directory=False) != executable:
        raise RuntimeError("isolated launcher requires the project Python executable")
    site_packages = _direct_path(
        root / ".venv" / "Lib" / "site-packages",
        directory=True,
    )
    project_src = _direct_path(root / "src", directory=True)
    return script, root, site_packages, project_src


def _sanitized_runtime_environment(runtime_root: Path) -> dict[str, str]:
    windows = _direct_path(_FROZEN_WINDOWS_DIRECTORY, directory=True)
    system32 = _direct_path(windows / "System32", directory=True)
    program_files = _direct_path(_FROZEN_PROGRAM_FILES_DIRECTORY, directory=True)
    program_files_x86 = _direct_path(
        _FROZEN_PROGRAM_FILES_X86_DIRECTORY,
        directory=True,
    )
    program_data = _direct_path(_FROZEN_PROGRAM_DATA_DIRECTORY, directory=True)
    temporary = _direct_path(runtime_root / "temp", directory=True)
    profile = _direct_path(runtime_root / "home", directory=True)
    roaming = _direct_path(profile / "AppData" / "Roaming", directory=True)
    local = _direct_path(profile / "AppData" / "Local", directory=True)
    return {
        "APPDATA": os.fspath(roaming),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "COMSPEC": os.fspath(_direct_path(system32 / "cmd.exe", directory=False)),
        "CUDA_CACHE_DISABLE": "1",
        "LOCALAPPDATA": os.fspath(local),
        "NUMBER_OF_PROCESSORS": str(os.cpu_count() or 1),
        "OS": "Windows_NT",
        "PATH": os.fspath(system32),
        "PATHEXT": ".COM;.EXE;.BAT;.CMD",
        "PROCESSOR_ARCHITECTURE": "AMD64",
        "PROGRAMDATA": os.fspath(program_data),
        "PROGRAMFILES": os.fspath(program_files),
        "PROGRAMFILES(X86)": os.fspath(program_files_x86),
        "PROGRAMW6432": os.fspath(program_files),
        "SYSTEMDRIVE": windows.drive,
        "SYSTEMROOT": os.fspath(windows),
        "TEMP": os.fspath(temporary),
        "TMP": os.fspath(temporary),
        "TORCHINDUCTOR_CACHE_DIR": os.fspath(temporary),
        "USERPROFILE": os.fspath(profile),
        "WINDIR": os.fspath(windows),
    }


def _remove_empty_runtime_root(runtime_root: Path) -> None:
    if {entry.name for entry in runtime_root.iterdir()} != {"pycache", "temp", "home"}:
        raise RuntimeError("isolated runtime root has unexpected contents")
    home = _direct_path(runtime_root / "home", directory=True)
    app_data = _direct_path(home / "AppData", directory=True)
    if {entry.name for entry in home.iterdir()} != {"AppData"} or {
        entry.name for entry in app_data.iterdir()
    } != {"Roaming", "Local"}:
        raise RuntimeError("isolated runtime profile has unexpected contents")
    for relative in (
        "pycache",
        "temp",
        "home/AppData/Roaming",
        "home/AppData/Local",
    ):
        directory = _direct_path(runtime_root / Path(relative), directory=True)
        if any(directory.iterdir()):
            raise RuntimeError("isolated runtime directory contains unexpected files")
    for relative in (
        "home/AppData/Roaming",
        "home/AppData/Local",
        "home/AppData",
        "home",
        "temp",
        "pycache",
    ):
        (runtime_root / Path(relative)).rmdir()
    runtime_root.rmdir()


def _write_parent_handoff(path: Path, *, token: str) -> None:
    payload = f"{token}\n".encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _consume_parent_handoff(runtime_root: Path, *, token: str) -> None:
    if len(token) != 64 or any(character not in "0123456789abcdef" for character in token):
        raise RuntimeError("isolated child handoff token is malformed")
    expected_runtime_root = runtime_root.parent / f"{_RUNTIME_ROOT_PREFIX}{token}"
    if runtime_root != expected_runtime_root:
        raise RuntimeError("isolated runtime root is not bound to the handoff token")

    cache = _direct_path(runtime_root / "pycache", directory=True)
    temporary = _direct_path(runtime_root / "temp", directory=True)
    home = _direct_path(runtime_root / "home", directory=True)
    app_data = _direct_path(home / "AppData", directory=True)
    roaming = _direct_path(app_data / "Roaming", directory=True)
    local = _direct_path(app_data / "Local", directory=True)
    handoff = _direct_path(runtime_root / _HANDOFF_FILENAME, directory=False)
    if {entry.name for entry in runtime_root.iterdir()} != {
        "pycache",
        "temp",
        "home",
        _HANDOFF_FILENAME,
    }:
        raise RuntimeError("isolated runtime root has an unexpected layout")
    if {entry.name for entry in home.iterdir()} != {"AppData"} or {
        entry.name for entry in app_data.iterdir()
    } != {"Roaming", "Local"}:
        raise RuntimeError("isolated runtime profile has an unexpected layout")
    for directory in (cache, temporary, roaming, local):
        if any(directory.iterdir()):
            raise RuntimeError("isolated runtime directory was not empty at bootstrap")

    expected_payload = f"{token}\n".encode("ascii")
    payload = handoff.read_bytes()
    if not secrets.compare_digest(payload, expected_payload):
        raise RuntimeError("isolated child handoff proof is invalid")
    handoff.unlink()


def _relaunch_isolated(arguments: Sequence[str]) -> int:
    if _ISOLATED_CHILD_FLAG in arguments:
        raise RuntimeError("isolated child marker cannot be supplied directly")
    script, root, _, _ = _project_layout()
    cache_parent = _direct_path(root / "artifacts" / "trust_sentinel", directory=True)
    token = secrets.token_hex(32)
    runtime_root = cache_parent / f"{_RUNTIME_ROOT_PREFIX}{token}"
    runtime_root.mkdir(mode=0o700, parents=False, exist_ok=False)
    cache = runtime_root / "pycache"
    temporary = runtime_root / "temp"
    roaming = runtime_root / "home" / "AppData" / "Roaming"
    local = runtime_root / "home" / "AppData" / "Local"
    cache.mkdir()
    temporary.mkdir()
    roaming.mkdir(parents=True)
    local.mkdir()
    _write_parent_handoff(runtime_root / _HANDOFF_FILENAME, token=token)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-X",
                f"pycache_prefix={cache}",
                os.fspath(script),
                _ISOLATED_CHILD_FLAG,
                token,
                *arguments,
            ],
            check=False,
            cwd=root,
            env=_sanitized_runtime_environment(runtime_root),
        )
        _remove_empty_runtime_root(runtime_root)
        return completed.returncode
    finally:
        if runtime_root.exists():
            with suppress(OSError):
                _remove_empty_runtime_root(runtime_root)


def _enter_isolated_runtime() -> None:
    global _ISOLATED_RUNTIME_ACTIVE
    script, root, site_packages, project_src = _project_layout()
    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None)
    if not isinstance(main_file, str) or _direct_path(Path(main_file), directory=False) != script:
        raise RuntimeError("isolated launcher requires the exact bound __main__ script")
    if (
        sys.argv.count(_ISOLATED_CHILD_FLAG) != 1
        or len(sys.argv) < 3
        or sys.argv[1] != _ISOLATED_CHILD_FLAG
    ):
        raise RuntimeError("isolated child marker is missing or misplaced")
    token = sys.argv[2]
    del sys.argv[1:3]
    if (
        sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.no_user_site != 1
        or not sys.dont_write_bytecode
        or not isinstance(sys.pycache_prefix, str)
    ):
        raise RuntimeError("isolated Python flags are incomplete")
    cache = _direct_path(Path(sys.pycache_prefix), directory=True)
    runtime_root = cache.parent
    cache_parent = _direct_path(root / "artifacts" / "trust_sentinel", directory=True)
    if _direct_path(runtime_root, directory=True).parent != cache_parent:
        raise RuntimeError("isolated runtime root is outside the bound artifacts directory")
    if _direct_path(Path.cwd(), directory=True) != root:
        raise RuntimeError("isolated launcher working directory is not the project root")
    _consume_parent_handoff(runtime_root, token=token)
    environment = _sanitized_runtime_environment(runtime_root)
    if {name.upper(): value for name, value in os.environ.items()} != environment:
        raise RuntimeError("isolated child environment differs from the parent contract")
    os.environ.clear()
    os.environ.update(environment)
    sys.path.extend((os.fspath(site_packages), os.fspath(project_src)))
    _ISOLATED_RUNTIME_ACTIVE = True


if __name__ == "__main__":
    try:
        safe_flags = (
            sys.flags.isolated == 1
            and sys.flags.no_site == 1
            and sys.flags.no_user_site == 1
            and sys.dont_write_bytecode
        )
        if not safe_flags:
            raise SystemExit("isolated launcher requires Python -I -S -B")
        if sys.pycache_prefix is None:
            raise SystemExit(_relaunch_isolated(tuple(sys.argv[1:])))
        _enter_isolated_runtime()
    except (OSError, RuntimeError, ValueError):
        raise SystemExit("isolated launcher refused an invalid runtime contract") from None

from ecg_trust.ood_v2.inventory import (  # noqa: E402
    BUILD_SUMMARY_DOMAIN,
    CHALLENGE_2011_DATASET,
    CHALLENGE_2011_VERSION,
    INVENTORY_SCHEMA_VERSION,
    PUBLIC_PROJECTION_DOMAIN,
    PUBLIC_PROJECTION_KIND,
    ZZU_PEDIATRIC_DATASET,
    ZZU_PEDIATRIC_VERSION,
    ArchiveExtractionClosure,
    ExternalInventoryBuildSummary,
    ExternalInventoryError,
    ExternalInventoryRecord,
    ExternalWaveformInventory,
    SevenZipToolBinding,
    ZZUPediatricCandidate,
    build_challenge_tar_extraction_closure,
    build_external_inventory,
    build_zzu_split_zip_extraction_closure,
    external_inventory_public_projection,
    inventory_challenge_2011_record,
    load_external_inventory,
    parse_challenge_2011_quality_lists,
    parse_zzu_pediatric_attributes_csv,
    select_zzu_pediatric_inventory_records,
    validate_challenge_2011_set_a_inventory,
    verify_external_inventory,
    verify_wfdb_candidate_file_set,
)
from ecg_trust.ood_v2.pipeline import (  # noqa: E402
    verify_inventory_builder_postflight,
    verify_inventory_builder_preflight,
)

EXPECTED_CHALLENGE_RECORDS = 1_000
EXPECTED_ZZU_RECORDS = 14_190
EXPECTED_ZZU_PATIENTS = 11_643
EXPECTED_ZZU_TWELVE_LEAD_RECORDS = 12_334
EXPECTED_ZZU_NINE_LEAD_RECORDS = 1_856

CHALLENGE_PRIVATE_SITE = "PhysioNet Challenge 2011 Set A"
CHALLENGE_SITE_ALIAS = "challenge-set-a"
ZZU_PRIVATE_SITE = "Zhengzhou University pediatric ECG"
ZZU_SITE_ALIAS = "zzu-pecg"

PUBLIC_ARTIFACT_KIND = "ecg_trust.ood_v2.inventory_publication"
PUBLIC_ARTIFACT_SCHEMA_VERSION = 1
PUBLIC_ARTIFACT_DOMAIN = b"ecg_trust.ood_v2.inventory_publication.v1\x00"
MAX_METADATA_BYTES = 64 * 1024 * 1024
SUCCESSOR_PRIVATE_INVENTORY_PATH = Path(
    "artifacts/trust_sentinel/ood_external_v2_1_preflight/private/external-waveform-inventory.json"
)
SUCCESSOR_PUBLIC_PROJECTION_PATH = Path(
    "artifacts/trust_sentinel/ood_external_v2_1_preflight/public/external-inventory-summary.json"
)


class InventoryCLIError(ValueError):
    """Raised when official inputs or immutable outputs violate the CLI contract."""


@dataclass(frozen=True, slots=True)
class InventoryExpectations:
    """Non-CLI production counts; tests may inject a complete miniature fixture."""

    challenge_records: int = EXPECTED_CHALLENGE_RECORDS
    zzu_records: int = EXPECTED_ZZU_RECORDS
    zzu_patients: int = EXPECTED_ZZU_PATIENTS
    zzu_twelve_lead_records: int = EXPECTED_ZZU_TWELVE_LEAD_RECORDS
    zzu_nine_lead_records: int = EXPECTED_ZZU_NINE_LEAD_RECORDS

    def __post_init__(self) -> None:
        for field, value in (
            ("challenge_records", self.challenge_records),
            ("zzu_records", self.zzu_records),
            ("zzu_patients", self.zzu_patients),
            ("zzu_twelve_lead_records", self.zzu_twelve_lead_records),
            ("zzu_nine_lead_records", self.zzu_nine_lead_records),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise InventoryCLIError(f"{field} must be a positive integer")
        if self.zzu_twelve_lead_records + self.zzu_nine_lead_records != self.zzu_records:
            raise InventoryCLIError("ZZU lead-count strata must sum to the total record count")
        if self.zzu_patients > self.zzu_records:
            raise InventoryCLIError("ZZU patient count cannot exceed its record count")


@dataclass(frozen=True, slots=True)
class InventoryInputPaths:
    challenge_root: Path
    challenge_records: Path
    challenge_acceptable: Path
    challenge_unacceptable: Path
    zzu_root: Path
    zzu_metadata: Path
    challenge_archive: Path | None = None
    zzu_archive_z01: Path | None = None
    zzu_archive_zip: Path | None = None
    seven_zip_executable: Path | None = None


@dataclass(frozen=True, slots=True)
class ZZUMetadataSchema:
    record_column: str = "Filename"
    ecg_id_column: str = "ECG_ID"
    patient_column: str = "Patient_ID"
    lead_count_column: str = "Lead"
    sampling_point_column: str = "Sampling_point"
    delimiter: str = ","

    def __post_init__(self) -> None:
        columns = (
            self.record_column,
            self.ecg_id_column,
            self.patient_column,
            self.lead_count_column,
            self.sampling_point_column,
        )
        if any(
            not isinstance(column, str) or not column or column != column.strip()
            for column in columns
        ):
            raise InventoryCLIError("ZZU metadata columns must be canonical non-empty text")
        if len(set(columns)) != len(columns):
            raise InventoryCLIError("ZZU metadata columns must be distinct")
        if self.delimiter not in {",", "\t", ";"}:
            raise InventoryCLIError("ZZU metadata delimiter is unsupported")


@dataclass(frozen=True, slots=True)
class BuiltInventoryArtifacts:
    inventory: ExternalWaveformInventory
    public_projection: Mapping[str, object]
    zzu_build_summary: ExternalInventoryBuildSummary
    challenge_record_count: int
    zzu_candidate_record_count: int
    zzu_patient_count: int

    @property
    def public_projection_sha256(self) -> str:
        value = self.public_projection.get("projection_sha256")
        if not isinstance(value, str):
            raise InventoryCLIError("public projection has no SHA-256 identity")
        return value


@dataclass(frozen=True, slots=True)
class InventoryCLIResult:
    inventory_sha256: str
    public_projection_sha256: str
    challenge_record_count: int
    zzu_candidate_record_count: int
    zzu_selected_record_count: int
    zzu_excluded_record_count: int


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _read_bounded_utf8(path: Path, *, label: str) -> str:
    if not isinstance(path, Path):
        raise TypeError(f"{label} path must be pathlib.Path")
    if path.is_symlink():
        raise InventoryCLIError(f"{label} must not be a symlink")
    if not path.is_file():
        raise InventoryCLIError(f"{label} is missing")
    before = path.stat()
    if before.st_size < 1 or before.st_size > MAX_METADATA_BYTES:
        raise InventoryCLIError(f"{label} has an invalid byte size")
    with path.open("rb") as handle:
        raw = handle.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise InventoryCLIError(f"{label} exceeds the metadata byte limit")
    after = path.stat()
    if path.is_symlink() or (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise InventoryCLIError(f"{label} changed while it was being read")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise InventoryCLIError(f"{label} must be valid UTF-8 text") from error


def read_zzu_candidates(
    metadata_path: Path,
    *,
    schema: ZZUMetadataSchema,
) -> tuple[ZZUPediatricCandidate, ...]:
    """Read the exact official Filename/ECG_ID/Patient_ID eligibility mapping."""

    if not isinstance(schema, ZZUMetadataSchema):
        raise TypeError("schema must be ZZUMetadataSchema")
    text = _read_bounded_utf8(metadata_path, label="ZZU metadata")
    try:
        return parse_zzu_pediatric_attributes_csv(
            text,
            site=ZZU_PRIVATE_SITE,
            site_alias=ZZU_SITE_ALIAS,
            dataset_version=ZZU_PEDIATRIC_VERSION,
            record_column=schema.record_column,
            ecg_id_column=schema.ecg_id_column,
            patient_column=schema.patient_column,
            lead_count_column=schema.lead_count_column,
            sampling_point_column=schema.sampling_point_column,
            delimiter=schema.delimiter,
        )
    except ExternalInventoryError as error:
        raise InventoryCLIError(
            "ZZU metadata failed the strict official mapping contract"
        ) from error


def _validate_zzu_metadata_counts(
    candidates: Sequence[ZZUPediatricCandidate],
    expectations: InventoryExpectations,
) -> None:
    if len(candidates) != expectations.zzu_records:
        raise InventoryCLIError(
            f"expected {expectations.zzu_records} ZZU records, found {len(candidates)}"
        )
    record_refs = [candidate.record_ref for candidate in candidates]
    if len(record_refs) != len(set(record_refs)):
        raise InventoryCLIError("ZZU metadata contains duplicate record identities")
    patients = {candidate.patient_key for candidate in candidates}
    if len(patients) != expectations.zzu_patients:
        raise InventoryCLIError(
            f"expected {expectations.zzu_patients} ZZU patients, found {len(patients)}"
        )
    twelve_lead = sum(candidate.pediatric_12_lead for candidate in candidates)
    if twelve_lead != expectations.zzu_twelve_lead_records:
        raise InventoryCLIError(
            "ZZU twelve-lead metadata count differs from the frozen upstream count"
        )
    if len(candidates) - twelve_lead != expectations.zzu_nine_lead_records:
        raise InventoryCLIError(
            "ZZU nine-lead metadata count differs from the frozen upstream count"
        )


def _challenge_records(
    inputs: InventoryInputPaths,
    *,
    expectations: InventoryExpectations,
) -> tuple[ExternalInventoryRecord, ...]:
    all_text = _read_bounded_utf8(inputs.challenge_records, label="Challenge RECORDS")
    acceptable_text = _read_bounded_utf8(
        inputs.challenge_acceptable, label="Challenge acceptable list"
    )
    unacceptable_text = _read_bounded_utf8(
        inputs.challenge_unacceptable, label="Challenge unacceptable list"
    )
    labels = parse_challenge_2011_quality_lists(
        all_text,
        acceptable_text,
        unacceptable_text,
        expected_record_count=expectations.challenge_records,
    )
    verify_wfdb_candidate_file_set(inputs.challenge_root, tuple(labels))
    records = tuple(
        inventory_challenge_2011_record(
            inputs.challenge_root,
            dataset_version=CHALLENGE_2011_VERSION,
            site=CHALLENGE_PRIVATE_SITE,
            site_alias=CHALLENGE_SITE_ALIAS,
            record_ref=record_ref,
            quality_label=quality_label,
        )
        for record_ref, quality_label in labels.items()
    )
    inventory = build_external_inventory(records)
    validate_challenge_2011_set_a_inventory(
        inventory,
        expected_record_count=expectations.challenge_records,
        expected_quality_by_record=labels,
    )
    return records


def _public_projection(
    inventory: ExternalWaveformInventory,
    zzu_summary: ExternalInventoryBuildSummary,
    *,
    challenge_record_count: int,
    zzu_patient_count: int,
) -> dict[str, object]:
    body: dict[str, object] = {
        "challenge_record_count": challenge_record_count,
        "inventory": external_inventory_public_projection(inventory),
        "kind": PUBLIC_ARTIFACT_KIND,
        "schema_version": PUBLIC_ARTIFACT_SCHEMA_VERSION,
        "zzu_candidate_patient_count": zzu_patient_count,
        "zzu_inventory_build_summary": zzu_summary.to_dict(),
    }
    digest = hashlib.sha256(PUBLIC_ARTIFACT_DOMAIN + _canonical_json_bytes(body)).hexdigest()
    body["projection_sha256"] = f"sha256:{digest}"
    return body


def verify_public_projection(payload: Mapping[str, object]) -> str:
    """Verify the aggregate publication allowlist and its canonical identity."""

    expected_keys = {
        "challenge_record_count",
        "inventory",
        "kind",
        "projection_sha256",
        "schema_version",
        "zzu_candidate_patient_count",
        "zzu_inventory_build_summary",
    }
    if set(payload) != expected_keys:
        raise InventoryCLIError("public projection fields differ from the aggregate allowlist")
    if payload["kind"] != PUBLIC_ARTIFACT_KIND:
        raise InventoryCLIError("public projection kind is unsupported")
    if payload["schema_version"] != PUBLIC_ARTIFACT_SCHEMA_VERSION:
        raise InventoryCLIError("public projection schema version is unsupported")
    _verify_nested_inventory_projection(payload["inventory"])
    _verify_zzu_build_summary_projection(payload["zzu_inventory_build_summary"])
    challenge_count = _nonnegative_int(payload["challenge_record_count"], "challenge_record_count")
    _nonnegative_int(payload["zzu_candidate_patient_count"], "zzu_candidate_patient_count")
    inventory_projection = cast(Mapping[str, object], payload["inventory"])
    groups = cast(list[dict[str, object]], inventory_projection["groups"])
    challenge_group = next(group for group in groups if group["dataset"] == CHALLENGE_2011_DATASET)
    zzu_group = next(group for group in groups if group["dataset"] == ZZU_PEDIATRIC_DATASET)
    if challenge_group["record_count"] != challenge_count:
        raise InventoryCLIError("public Challenge aggregate count is inconsistent")
    zzu_summary = cast(Mapping[str, object], payload["zzu_inventory_build_summary"])
    if zzu_group["record_count"] != zzu_summary["selected_record_count"]:
        raise InventoryCLIError("public ZZU selected aggregate count is inconsistent")
    raw_hash = payload["projection_sha256"]
    if not isinstance(raw_hash, str):
        raise InventoryCLIError("public projection SHA-256 identity must be text")
    body = dict(payload)
    del body["projection_sha256"]
    expected = (
        "sha256:" + hashlib.sha256(PUBLIC_ARTIFACT_DOMAIN + _canonical_json_bytes(body)).hexdigest()
    )
    if raw_hash != expected:
        raise InventoryCLIError("public projection self-hash does not match")
    return expected


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InventoryCLIError(f"public {field} must be a non-negative integer")
    return value


def _verify_nested_inventory_projection(value: object) -> None:
    if not isinstance(value, Mapping):
        raise InventoryCLIError("public inventory projection must be an object")
    projection = cast(Mapping[str, object], value)
    expected = {
        "archive_closures",
        "group_count",
        "groups",
        "kind",
        "projection_sha256",
        "record_count",
        "schema_version",
        "source_inventory_sha256",
    }
    if set(projection) != expected:
        raise InventoryCLIError("nested inventory projection fields differ from its allowlist")
    if projection["kind"] != PUBLIC_PROJECTION_KIND:
        raise InventoryCLIError("nested inventory projection kind is unsupported")
    if projection["schema_version"] != INVENTORY_SCHEMA_VERSION:
        raise InventoryCLIError("nested inventory projection schema is unsupported")
    if projection["group_count"] != 2:
        raise InventoryCLIError("public inventory projection must contain exactly two groups")
    groups_value = projection["groups"]
    if not isinstance(groups_value, list) or len(groups_value) != 2:
        raise InventoryCLIError("public inventory groups must be a two-item array")
    expected_group_keys = {
        "challenge_quality_counts",
        "dataset",
        "dataset_version",
        "duration_seconds_max",
        "duration_seconds_min",
        "known_patient_count",
        "lead_count_distribution",
        "missing_patient_key_records",
        "pediatric_12_lead_counts",
        "raw_bytes_total",
        "record_count",
        "sampling_frequencies_hz",
        "site_alias",
        "source_role",
    }
    expected_sources = {
        CHALLENGE_2011_DATASET: (CHALLENGE_2011_VERSION, CHALLENGE_SITE_ALIAS),
        ZZU_PEDIATRIC_DATASET: (ZZU_PEDIATRIC_VERSION, ZZU_SITE_ALIAS),
    }
    observed_sources: set[str] = set()
    total_records = 0
    for raw_group in groups_value:
        if not isinstance(raw_group, dict) or set(raw_group) != expected_group_keys:
            raise InventoryCLIError("public inventory group fields differ from the allowlist")
        group = cast(dict[str, object], raw_group)
        dataset = group["dataset"]
        if not isinstance(dataset, str) or dataset not in expected_sources:
            raise InventoryCLIError("public inventory group dataset is unsupported")
        if dataset in observed_sources:
            raise InventoryCLIError("public inventory dataset groups must be unique")
        observed_sources.add(dataset)
        expected_version, expected_alias = expected_sources[dataset]
        if group["dataset_version"] != expected_version or group["site_alias"] != expected_alias:
            raise InventoryCLIError("public inventory source version or alias is unexpected")
        if group["source_role"] != "confirmation_lockbox":
            raise InventoryCLIError("public inventory source role is unexpected")
        record_count = _nonnegative_int(group["record_count"], "group record_count")
        total_records += record_count
        quality_counts = group["challenge_quality_counts"]
        if not isinstance(quality_counts, Mapping) or set(quality_counts) != {
            "acceptable",
            "indeterminate",
            "unacceptable",
        }:
            raise InventoryCLIError("public Challenge quality counts are not allowlisted")
        pediatric_counts = group["pediatric_12_lead_counts"]
        if not isinstance(pediatric_counts, Mapping) or set(pediatric_counts) != {
            "false",
            "true",
        }:
            raise InventoryCLIError("public pediatric lead counts are not allowlisted")
    if observed_sources != set(expected_sources):
        raise InventoryCLIError("public inventory projection is missing a frozen dataset")
    _verify_public_archive_closure_summaries(projection["archive_closures"])
    if _nonnegative_int(projection["record_count"], "record_count") != total_records:
        raise InventoryCLIError("public inventory record counts are inconsistent")
    raw_hash = projection["projection_sha256"]
    if not isinstance(raw_hash, str):
        raise InventoryCLIError("nested inventory projection hash must be text")
    body = dict(projection)
    del body["projection_sha256"]
    expected_hash = (
        "sha256:"
        + hashlib.sha256(PUBLIC_PROJECTION_DOMAIN + _canonical_json_bytes(body)).hexdigest()
    )
    if raw_hash != expected_hash:
        raise InventoryCLIError("nested inventory projection self-hash does not match")


def _verify_public_archive_closure_summaries(value: object) -> None:
    if not isinstance(value, list):
        raise InventoryCLIError("public archive closures must be an array")
    if len(value) not in {0, 2}:
        raise InventoryCLIError("public archive closures must be empty or cover both datasets")
    expected_keys = {
        "archive_bytes_total",
        "archive_file_count",
        "archive_format",
        "closure_sha256",
        "dataset",
        "member_bytes_total",
        "member_count",
        "member_role_counts",
        "tool_binding",
    }
    observed_datasets: set[str] = set()
    for raw_summary in value:
        if not isinstance(raw_summary, Mapping) or set(raw_summary) != expected_keys:
            raise InventoryCLIError("public archive closure fields differ from the allowlist")
        summary = cast(Mapping[str, object], raw_summary)
        dataset = summary["dataset"]
        if not isinstance(dataset, str) or dataset not in {
            CHALLENGE_2011_DATASET,
            ZZU_PEDIATRIC_DATASET,
        }:
            raise InventoryCLIError("public archive closure dataset is unsupported")
        if dataset in observed_datasets:
            raise InventoryCLIError("public archive closure datasets must be unique")
        observed_datasets.add(dataset)
        archive_file_count = _nonnegative_int(summary["archive_file_count"], "archive_file_count")
        _nonnegative_int(summary["archive_bytes_total"], "archive_bytes_total")
        member_count = _nonnegative_int(summary["member_count"], "member_count")
        _nonnegative_int(summary["member_bytes_total"], "member_bytes_total")
        if archive_file_count < 1 or member_count < 1:
            raise InventoryCLIError("public archive closure counts must be positive")
        roles = summary["member_role_counts"]
        if not isinstance(roles, Mapping) or set(roles) != {
            "ignored_release_file",
            "quality_reference",
            "wfdb_data",
            "wfdb_header",
        }:
            raise InventoryCLIError("public archive member roles differ from the allowlist")
        role_total = sum(
            _nonnegative_int(count, f"member role {role}") for role, count in roles.items()
        )
        if role_total != member_count:
            raise InventoryCLIError("public archive member role counts are inconsistent")
        closure_hash = summary["closure_sha256"]
        if (
            not isinstance(closure_hash, str)
            or not closure_hash.startswith("sha256:")
            or len(closure_hash) != 71
            or any(
                character not in "0123456789abcdef"
                for character in closure_hash.removeprefix("sha256:")
            )
        ):
            raise InventoryCLIError("public archive closure hash is invalid")
        if dataset == CHALLENGE_2011_DATASET:
            if (
                summary["archive_format"] != "tar_gzip"
                or archive_file_count != 1
                or summary["tool_binding"] is not None
            ):
                raise InventoryCLIError("public Challenge archive closure is inconsistent")
        else:
            if summary["archive_format"] != "split_zip_7zip" or archive_file_count != 2:
                raise InventoryCLIError("public ZZU archive closure is inconsistent")
            raw_tool = summary["tool_binding"]
            if not isinstance(raw_tool, Mapping):
                raise InventoryCLIError("public ZZU archive closure omits the tool binding")
            try:
                SevenZipToolBinding.from_dict(cast(Mapping[str, object], raw_tool))
            except ExternalInventoryError as error:
                raise InventoryCLIError("public 7-Zip tool binding is invalid") from error
    if value and observed_datasets != {CHALLENGE_2011_DATASET, ZZU_PEDIATRIC_DATASET}:
        raise InventoryCLIError("public archive closures must cover both datasets")


def _verify_zzu_build_summary_projection(value: object) -> None:
    if not isinstance(value, Mapping):
        raise InventoryCLIError("public ZZU build summary must be an object")
    summary = cast(Mapping[str, object], value)
    expected = {
        "candidate_record_count",
        "dataset",
        "exclusion_counts",
        "selected_record_count",
        "summary_sha256",
    }
    if set(summary) != expected or summary["dataset"] != ZZU_PEDIATRIC_DATASET:
        raise InventoryCLIError("public ZZU build summary differs from its allowlist")
    exclusion_value = summary["exclusion_counts"]
    if not isinstance(exclusion_value, Mapping) or set(exclusion_value) != {
        "duration_under_10_seconds",
        "lead_count_not_12",
        "noncanonical_lead_set",
        "pediatric_12_lead_flag_false",
        "sampling_frequency_not_500_hz",
    }:
        raise InventoryCLIError("public ZZU exclusion reasons differ from the allowlist")
    exclusions = {
        str(reason): _nonnegative_int(count, f"exclusion {reason}")
        for reason, count in exclusion_value.items()
    }
    candidates = _nonnegative_int(summary["candidate_record_count"], "candidate_record_count")
    selected = _nonnegative_int(summary["selected_record_count"], "selected_record_count")
    if selected + sum(exclusions.values()) != candidates:
        raise InventoryCLIError("public ZZU selected and excluded counts are inconsistent")
    raw_hash = summary["summary_sha256"]
    if not isinstance(raw_hash, str):
        raise InventoryCLIError("public ZZU build summary hash must be text")
    body = dict(summary)
    del body["summary_sha256"]
    expected_hash = (
        "sha256:" + hashlib.sha256(BUILD_SUMMARY_DOMAIN + _canonical_json_bytes(body)).hexdigest()
    )
    if raw_hash != expected_hash:
        raise InventoryCLIError("public ZZU build summary self-hash does not match")


def _relative_input_inside_root(path: Path, root: Path, *, expected_name: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise InventoryCLIError("Challenge reference list is unavailable")
    try:
        resolved_root = root.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
        relative = resolved_path.relative_to(resolved_root).as_posix()
    except (OSError, ValueError) as error:
        raise InventoryCLIError(
            "Challenge reference lists must be inside the exact extraction root"
        ) from error
    if relative != expected_name:
        raise InventoryCLIError("Challenge reference list path is not the frozen leaf")
    return relative


def _build_archive_closures(
    inputs: InventoryInputPaths,
    challenge: Sequence[ExternalInventoryRecord],
    candidates: Sequence[ZZUPediatricCandidate],
) -> tuple[ArchiveExtractionClosure, ...]:
    optional_paths = (
        inputs.challenge_archive,
        inputs.zzu_archive_z01,
        inputs.zzu_archive_zip,
        inputs.seven_zip_executable,
    )
    if all(path is None for path in optional_paths):
        return ()
    if any(path is None for path in optional_paths):
        raise InventoryCLIError("archive closure inputs must be supplied as one exact set")
    challenge_archive = cast(Path, inputs.challenge_archive)
    zzu_archive_z01 = cast(Path, inputs.zzu_archive_z01)
    zzu_archive_zip = cast(Path, inputs.zzu_archive_zip)
    seven_zip_executable = cast(Path, inputs.seven_zip_executable)
    challenge_required: list[str] = []
    for record in challenge:
        challenge_required.extend((f"{record.record_ref}.hea", f"{record.record_ref}.dat"))
    challenge_required.extend(
        (
            _relative_input_inside_root(
                inputs.challenge_records,
                inputs.challenge_root,
                expected_name="RECORDS",
            ),
            _relative_input_inside_root(
                inputs.challenge_acceptable,
                inputs.challenge_root,
                expected_name="RECORDS-acceptable",
            ),
            _relative_input_inside_root(
                inputs.challenge_unacceptable,
                inputs.challenge_root,
                expected_name="RECORDS-unacceptable",
            ),
        )
    )
    challenge_closure = build_challenge_tar_extraction_closure(
        challenge_archive,
        inputs.challenge_root,
        expected_required_relative_paths=tuple(challenge_required),
        archive_root_prefix="set-a",
    )
    zzu_required = tuple(
        path
        for candidate in candidates
        for path in (f"{candidate.record_ref}.hea", f"{candidate.record_ref}.dat")
    )
    zzu_closure = build_zzu_split_zip_extraction_closure(
        zzu_archive_z01,
        zzu_archive_zip,
        inputs.zzu_root,
        seven_zip_executable,
        expected_required_relative_paths=zzu_required,
        archive_root_prefix="Child_ecg",
    )
    challenge_roles = Counter(member.role for member in challenge_closure.members)
    expected_challenge_roles = {
        "ignored_release_file": len(challenge) + 1,
        "quality_reference": 3,
        "wfdb_data": len(challenge),
        "wfdb_header": len(challenge),
    }
    if challenge_roles != expected_challenge_roles:
        raise InventoryCLIError(
            "Challenge closure does not contain the exact full official release membership"
        )
    challenge_paths = {member.extracted_relative_path for member in challenge_closure.members}
    expected_ignored_paths = {
        *(f"{record.record_ref}.txt" for record in challenge),
        "HEADER.shtml",
    }
    observed_ignored_paths = {
        member.extracted_relative_path
        for member in challenge_closure.members
        if member.role == "ignored_release_file"
    }
    if observed_ignored_paths != expected_ignored_paths:
        raise InventoryCLIError(
            "Challenge closure ignored-file membership differs from the official release"
        )
    if not set(challenge_required).issubset(challenge_paths):
        raise InventoryCLIError("Challenge closure lost a required operational member")
    zzu_roles = Counter(member.role for member in zzu_closure.members)
    if zzu_roles != {
        "wfdb_data": len(candidates),
        "wfdb_header": len(candidates),
    }:
        raise InventoryCLIError(
            "ZZU closure does not contain the exact full candidate waveform membership"
        )
    return tuple(sorted((challenge_closure, zzu_closure), key=lambda value: value.dataset))


def build_inventory_artifacts(
    inputs: InventoryInputPaths,
    *,
    zzu_schema: ZZUMetadataSchema,
    expectations: InventoryExpectations | None = None,
) -> BuiltInventoryArtifacts:
    """Build and verify both selected cohorts without decoding any waveform."""

    if not isinstance(inputs, InventoryInputPaths):
        raise TypeError("inputs must be InventoryInputPaths")
    if not isinstance(zzu_schema, ZZUMetadataSchema):
        raise TypeError("zzu_schema must be ZZUMetadataSchema")
    resolved_expectations = expectations or InventoryExpectations()
    if not isinstance(resolved_expectations, InventoryExpectations):
        raise TypeError("expectations must be InventoryExpectations")
    challenge = _challenge_records(inputs, expectations=resolved_expectations)
    candidates = read_zzu_candidates(inputs.zzu_metadata, schema=zzu_schema)
    _validate_zzu_metadata_counts(candidates, resolved_expectations)
    selected_zzu, zzu_summary = select_zzu_pediatric_inventory_records(inputs.zzu_root, candidates)
    if not selected_zzu:
        raise InventoryCLIError("no ZZU record satisfies the frozen input contract")
    if zzu_summary.candidate_record_count != resolved_expectations.zzu_records:
        raise InventoryCLIError("ZZU build summary lost candidate records")

    archive_closures = _build_archive_closures(inputs, challenge, candidates)
    inventory = build_external_inventory(
        (*challenge, *selected_zzu),
        archive_closures=archive_closures,
    )
    challenge_inventory = build_external_inventory(challenge)
    zzu_inventory = build_external_inventory(selected_zzu)
    verify_external_inventory(inputs.challenge_root, challenge_inventory)
    verify_external_inventory(inputs.zzu_root, zzu_inventory)
    patient_count = len({candidate.patient_key for candidate in candidates})
    projection = _public_projection(
        inventory,
        zzu_summary,
        challenge_record_count=len(challenge),
        zzu_patient_count=patient_count,
    )
    verify_public_projection(projection)
    return BuiltInventoryArtifacts(
        inventory=inventory,
        public_projection=projection,
        zzu_build_summary=zzu_summary,
        challenge_record_count=len(challenge),
        zzu_candidate_record_count=len(candidates),
        zzu_patient_count=patient_count,
    )


def _reject_output_path(path: Path) -> None:
    _assert_direct_existing_ancestry(path)
    if _is_indirect(path) or path.exists():
        raise InventoryCLIError("immutable output path already exists")
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if _direct_path(parent, directory=True) != parent:
        raise InventoryCLIError("immutable output parent is invalid")


def _assert_direct_existing_ancestry(path: Path) -> None:
    lexical = Path(os.path.abspath(os.fspath(path)))
    components = (lexical, *lexical.parents)
    for component in components:
        if _is_indirect(component):
            raise InventoryCLIError("immutable output path traverses an indirect component")
    existing = next((component for component in components if component.exists()), None)
    if existing is None:
        raise InventoryCLIError("immutable output ancestry is unavailable")
    try:
        resolved = existing.resolve(strict=True)
    except OSError as error:
        raise InventoryCLIError("immutable output ancestry is unavailable") from error
    if resolved != Path(os.path.abspath(os.fspath(existing))):
        raise InventoryCLIError("immutable output ancestry resolves indirectly")


def _commit_new_outputs(outputs: Mapping[Path, tuple[bytes, int]]) -> None:
    """Commit exact bytes with create-new hard links and clean transaction debris."""

    if not outputs:
        raise InventoryCLIError("no immutable outputs were provided")
    canonical_paths = [Path(os.path.abspath(os.fspath(path))) for path in outputs]
    if len(canonical_paths) != len(set(canonical_paths)):
        raise InventoryCLIError("immutable output paths must be distinct")
    for path in canonical_paths:
        _reject_output_path(path)
    parent_identities = {
        path.parent: (path.parent.stat().st_dev, path.parent.stat().st_ino)
        for path in canonical_paths
    }

    temporary_by_destination: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        for original_path, (payload, mode) in outputs.items():
            destination = Path(os.path.abspath(os.fspath(original_path)))
            if (
                _direct_path(destination.parent, directory=True) != destination.parent
                or (
                    destination.parent.stat().st_dev,
                    destination.parent.stat().st_ino,
                )
                != parent_identities[destination.parent]
            ):
                raise InventoryCLIError("immutable output parent changed during transaction")
            descriptor, raw_temp = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
            )
            temporary = Path(raw_temp)
            temporary_by_destination[destination] = temporary
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temporary, mode)
            except BaseException:
                if temporary.exists():
                    temporary.unlink()
                raise
        for destination, temporary in temporary_by_destination.items():
            if (
                _direct_path(destination.parent, directory=True) != destination.parent
                or (
                    destination.parent.stat().st_dev,
                    destination.parent.stat().st_ino,
                )
                != parent_identities[destination.parent]
            ):
                raise InventoryCLIError("immutable output parent changed before commit")
            os.link(temporary, destination)
            committed.append(destination)
        for destination, (payload, _) in zip(canonical_paths, outputs.values(), strict=True):
            if _direct_path(destination, directory=False).read_bytes() != payload:
                raise InventoryCLIError("immutable output bytes changed during commit")
    except BaseException as error:
        cleanup_failed = False
        for destination in committed:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                cleanup_failed = True
        if cleanup_failed:
            raise InventoryCLIError(
                "immutable output transaction failed and requires private cleanup"
            ) from error
        raise
    finally:
        for temporary in temporary_by_destination.values():
            temporary.unlink(missing_ok=True)


def write_inventory_artifacts(
    artifacts: BuiltInventoryArtifacts,
    *,
    inputs: InventoryInputPaths,
    private_output: Path,
    public_output: Path,
) -> InventoryCLIResult:
    """Create, reload, and byte-verify immutable private and public artifacts."""

    if not isinstance(artifacts, BuiltInventoryArtifacts):
        raise TypeError("artifacts must be BuiltInventoryArtifacts")
    private_absolute = Path(os.path.abspath(os.fspath(private_output)))
    public_absolute = Path(os.path.abspath(os.fspath(public_output)))
    if private_absolute == public_absolute:
        raise InventoryCLIError("private and public output paths must be distinct")
    _verify_inventory_roots(artifacts.inventory, inputs)
    private_bytes = artifacts.inventory.to_canonical_json_bytes()
    public_bytes = _canonical_json_bytes(artifacts.public_projection) + b"\n"
    _commit_new_outputs(
        {
            private_output: (private_bytes, 0o600),
            public_output: (public_bytes, 0o644),
        }
    )

    loaded = load_external_inventory(private_output)
    if loaded != artifacts.inventory:
        raise InventoryCLIError("reloaded private inventory differs from its frozen value")
    _verify_inventory_roots(loaded, inputs)
    try:
        decoded: object = json.loads(public_output.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InventoryCLIError("could not reload the public projection") from error
    if not isinstance(decoded, dict):
        raise InventoryCLIError("public projection must be a JSON object")
    public_mapping = cast(dict[str, object], decoded)
    public_hash = verify_public_projection(public_mapping)
    if public_output.read_bytes() != public_bytes:
        raise InventoryCLIError("public projection is not in exact canonical form")
    return InventoryCLIResult(
        inventory_sha256=loaded.inventory_sha256,
        public_projection_sha256=public_hash,
        challenge_record_count=artifacts.challenge_record_count,
        zzu_candidate_record_count=artifacts.zzu_candidate_record_count,
        zzu_selected_record_count=artifacts.zzu_build_summary.selected_record_count,
        zzu_excluded_record_count=artifacts.zzu_build_summary.excluded_record_count,
    )


def _verify_inventory_roots(
    inventory: ExternalWaveformInventory,
    inputs: InventoryInputPaths,
) -> None:
    challenge = tuple(
        record for record in inventory.records if record.dataset == CHALLENGE_2011_DATASET
    )
    zzu = tuple(record for record in inventory.records if record.dataset == ZZU_PEDIATRIC_DATASET)
    if not challenge or not zzu or len(challenge) + len(zzu) != inventory.record_count:
        raise InventoryCLIError("private inventory has an unexpected dataset partition")
    verify_external_inventory(inputs.challenge_root, build_external_inventory(challenge))
    verify_external_inventory(inputs.zzu_root, build_external_inventory(zzu))


def _delimiter(value: str) -> str:
    mapping = {"comma": ",", "tab": "\t", "semicolon": ";"}
    try:
        return mapping[value]
    except KeyError as error:
        raise InventoryCLIError("unsupported ZZU metadata delimiter") from error


def _verify_production_output_destinations(
    private_output: Path,
    public_output: Path,
) -> None:
    """Bind writes to the successor namespace; child freeze proves Git hygiene."""

    if not _ISOLATED_RUNTIME_ACTIVE:
        raise InventoryCLIError("production inventory build requires the isolated launcher")
    _, root, _, _ = _project_layout()
    expected_private = root / SUCCESSOR_PRIVATE_INVENTORY_PATH
    expected_public = root / SUCCESSOR_PUBLIC_PROJECTION_PATH
    requested_private = Path(os.path.abspath(os.fspath(private_output)))
    requested_public = Path(os.path.abspath(os.fspath(public_output)))
    if (requested_private, requested_public) != (expected_private, expected_public):
        raise InventoryCLIError("inventory outputs must use the exact successor namespace")
    for destination in (requested_private, requested_public):
        _assert_direct_existing_ancestry(destination)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path("configs/trust_sentinel_ood_external_v2_1.yaml"),
    )
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--challenge-root", type=Path, required=True)
    parser.add_argument("--challenge-archive", type=Path, required=True)
    parser.add_argument("--challenge-records", type=Path, required=True)
    parser.add_argument("--challenge-acceptable", type=Path, required=True)
    parser.add_argument("--challenge-unacceptable", type=Path, required=True)
    parser.add_argument("--zzu-root", type=Path, required=True)
    parser.add_argument("--zzu-archive-z01", type=Path, required=True)
    parser.add_argument("--zzu-archive-zip", type=Path, required=True)
    parser.add_argument("--seven-zip-executable", type=Path, required=True)
    parser.add_argument("--zzu-metadata", type=Path, required=True)
    parser.add_argument("--zzu-record-column", default="Filename")
    parser.add_argument("--zzu-ecg-id-column", default="ECG_ID")
    parser.add_argument("--zzu-patient-column", default="Patient_ID")
    parser.add_argument("--zzu-lead-count-column", default="Lead")
    parser.add_argument("--zzu-sampling-point-column", default="Sampling_point")
    parser.add_argument(
        "--zzu-delimiter",
        choices=("comma", "tab", "semicolon"),
        default="comma",
    )
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    return parser


def run_inventory_build(arguments: argparse.Namespace) -> InventoryCLIResult:
    _, project_root, _, _ = _project_layout()
    preflight = verify_inventory_builder_preflight(
        arguments.parent,
        project_root,
        arguments.implementation_revision,
    )
    _verify_production_output_destinations(
        arguments.private_output,
        arguments.public_output,
    )
    inputs = InventoryInputPaths(
        challenge_root=arguments.challenge_root,
        challenge_archive=arguments.challenge_archive,
        challenge_records=arguments.challenge_records,
        challenge_acceptable=arguments.challenge_acceptable,
        challenge_unacceptable=arguments.challenge_unacceptable,
        zzu_root=arguments.zzu_root,
        zzu_archive_z01=arguments.zzu_archive_z01,
        zzu_archive_zip=arguments.zzu_archive_zip,
        seven_zip_executable=arguments.seven_zip_executable,
        zzu_metadata=arguments.zzu_metadata,
    )
    schema = ZZUMetadataSchema(
        record_column=arguments.zzu_record_column,
        ecg_id_column=arguments.zzu_ecg_id_column,
        patient_column=arguments.zzu_patient_column,
        lead_count_column=arguments.zzu_lead_count_column,
        sampling_point_column=arguments.zzu_sampling_point_column,
        delimiter=_delimiter(arguments.zzu_delimiter),
    )
    artifacts = build_inventory_artifacts(inputs, zzu_schema=schema)
    private_bytes = artifacts.inventory.to_canonical_json_bytes()
    public_bytes = _canonical_json_bytes(artifacts.public_projection) + b"\n"
    result = write_inventory_artifacts(
        artifacts,
        inputs=inputs,
        private_output=arguments.private_output,
        public_output=arguments.public_output,
    )
    public_artifact_sha256 = artifacts.public_projection.get("projection_sha256")
    if not isinstance(public_artifact_sha256, str):
        raise InventoryCLIError("public projection has no in-memory artifact identity")
    verify_inventory_builder_postflight(
        preflight,
        parent_path=arguments.parent,
        project_root=project_root,
        implementation_revision=arguments.implementation_revision,
        inventory_path=arguments.private_output,
        public_projection_path=arguments.public_output,
        expected_inventory_file_sha256=("sha256:" + hashlib.sha256(private_bytes).hexdigest()),
        expected_inventory_sha256=artifacts.inventory.inventory_sha256,
        expected_public_projection_file_sha256=(
            "sha256:" + hashlib.sha256(public_bytes).hexdigest()
        ),
        expected_public_projection_artifact_sha256=public_artifact_sha256,
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = run_inventory_build(arguments)
    except Exception:
        print(
            "OOD_V2_INVENTORY_FAILED: inspect private local inputs and immutable output state.",
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "challenge_record_count": result.challenge_record_count,
                "inventory_sha256": result.inventory_sha256,
                "public_projection_sha256": result.public_projection_sha256,
                "status": "OOD_V2_INVENTORY_FROZEN",
                "zzu_candidate_record_count": result.zzu_candidate_record_count,
                "zzu_excluded_record_count": result.zzu_excluded_record_count,
                "zzu_selected_record_count": result.zzu_selected_record_count,
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
