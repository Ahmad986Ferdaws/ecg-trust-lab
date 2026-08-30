#!/usr/bin/env python3
"""Freeze X11 external OOD v2.1 child metadata (original v2 is hard-refused)."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import secrets
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import suppress
from ctypes import wintypes
from pathlib import Path

_ISOLATED_CHILD_FLAG = "--_ecg-trust-ood-v2-isolated-child"
_RUNTIME_ROOT_PREFIX = ".ood_external_v2_1.runtime-"
_HANDOFF_FILENAME = ".parent-handoff"
_CHILD_RESULT_HANDOFF_FILENAME = ".child-result"
_MAX_CHILD_RESULT_HANDOFF_BYTES = 4096
_GCM_COMMANDLINE_SENTINEL_DIRECTORY = "system-commandline-sentinel-files"
_FROZEN_WINDOWS_DIRECTORY = Path(r"C:\Windows")
_FROZEN_PROGRAM_FILES_DIRECTORY = Path(r"C:\Program Files")
_FROZEN_PROGRAM_FILES_X86_DIRECTORY = Path(r"C:\Program Files (x86)")
_FROZEN_PROGRAM_DATA_DIRECTORY = Path(r"C:\ProgramData")

_WINDOWS_DELETE = 0x00010000
_WINDOWS_FILE_LIST_DIRECTORY = 0x00000001
_WINDOWS_FILE_READ_ATTRIBUTES = 0x00000080
_WINDOWS_FILE_SHARE_READ = 0x00000001
_WINDOWS_FILE_SHARE_WRITE = 0x00000002
_WINDOWS_FILE_SHARE_DELETE = 0x00000004
_WINDOWS_OPEN_EXISTING = 3
_WINDOWS_FILE_ATTRIBUTE_DIRECTORY = 0x00000010
_WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
_WINDOWS_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_WINDOWS_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_WINDOWS_FILE_DISPOSITION_INFO_CLASS = 4
_WINDOWS_FILE_ATTRIBUTE_TAG_INFO_CLASS = 9
_WINDOWS_FILE_ID_INFO_CLASS = 18
_WINDOWS_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

_EXPECTED_CHILD_FREEZE_PREFLIGHT_STAGES = (
    "parent_lineage",
    "runtime_environment",
    "git_source_provenance",
    "x8_inventory_evidence",
    "decision_and_runtime_bindings",
    "namespace_and_timestamp",
    "closing_control_state",
)
_EXPECTED_CHILD_FREEZE_ATTEMPT_STAGES = (
    "authorization_publication",
    "raw_source_binding_verification",
    "challenge_archive_closure",
    "zzu_tool_resolution",
    "zzu_archive_listing",
    "zzu_archive_test",
    "zzu_evaluated_tree_snapshot",
    "zzu_isolated_extraction",
    "zzu_archive_comparison",
    "decision_and_child_materialization",
    "prepublication_control_reverification",
    "child_publication",
    "child_reload_and_postflight",
)
_EXPECTED_CHILD_FREEZE_FAILURE_REASONS = (
    "STAGE_REFUSED",
    "DESTINATION_PREEXISTED",
    "PUBLICATION_FAILED_BEFORE_VISIBILITY",
    "PUBLICATION_FAILED_AFTER_VISIBILITY",
    "POSTPUBLICATION_RELOAD_REFUSED",
    "UNEXPECTED_INTERNAL_FAILURE",
)
_EXPECTED_CHILD_FREEZE_OUTPUT_STATES = (
    "NONE",
    "VISIBLE_EXACT_DURABILITY_UNCONFIRMED",
    "DURABLE_EXACT",
    "PRESENT_UNVERIFIABLE",
)
_EXPECTED_CHILD_FREEZE_AUTHORIZATION_STATES = (
    "NOT_CONSUMED",
    "CONSUMED",
    "UNVERIFIABLE",
)
_EXPECTED_CHILD_FREEZE_CLEANUP_STATES = (
    "NOT_REACHED",
    "CLEAN",
    "FAILED",
)
_CHILD_FREEZE_LAUNCHER_STAGES = (
    "isolated_child_terminalization",
    "launcher_cleanup",
)
_CHILD_FREEZE_STAGE_SCOPES = ("PREFLIGHT", "ATTEMPT", "LAUNCHER")
_CHILD_FREEZE_TERMINAL_STATUSES = (
    "OOD_EXTERNAL_V2_CHILD_PREFLIGHT_VERIFIED",
    "OOD_EXTERNAL_V2_CHILD_PREFLIGHT_REFUSED",
    "OOD_EXTERNAL_V2_CHILD_FROZEN",
    "OOD_EXTERNAL_V2_CHILD_FREEZE_FAILED",
)
_CHILD_FREEZE_TERMINAL_FIELDS = frozenset(
    {
        "authorization_state",
        "child_publication_witnessed",
        "child_visibility_witnessed",
        "cleanup_state",
        "failure_reason",
        "failure_receipt_written",
        "official_source_content_accessed",
        "output_state",
        "retry_authorized",
        "stage",
        "stage_ordinal",
        "stage_scope",
        "status",
    }
)
_ISOLATED_RUNTIME_ACTIVE = False
_ISOLATED_RUNTIME_ROOT: Path | None = None


class _WindowsFileAttributeTagInfo(ctypes.Structure):
    _fields_ = [
        ("file_attributes", wintypes.DWORD),
        ("reparse_tag", wintypes.DWORD),
    ]


class _WindowsFileId128(ctypes.Structure):
    _fields_ = [("identifier", ctypes.c_ubyte * 16)]


class _WindowsFileIdInfo(ctypes.Structure):
    _fields_ = [
        ("volume_serial_number", ctypes.c_uint64),
        ("file_id", _WindowsFileId128),
    ]


class _WindowsFileDispositionInfo(ctypes.Structure):
    _fields_ = [("delete_file", ctypes.c_ubyte)]


_WINDOWS_CREATE_FILE_W = None
_WINDOWS_GET_FILE_INFORMATION_BY_HANDLE_EX = None
_WINDOWS_SET_FILE_INFORMATION_BY_HANDLE = None
_WINDOWS_CLOSE_HANDLE = None
if os.name == "nt":
    _WINDOWS_KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _WINDOWS_CREATE_FILE_W = _WINDOWS_KERNEL32.CreateFileW
    _WINDOWS_CREATE_FILE_W.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    _WINDOWS_CREATE_FILE_W.restype = wintypes.HANDLE
    _WINDOWS_GET_FILE_INFORMATION_BY_HANDLE_EX = _WINDOWS_KERNEL32.GetFileInformationByHandleEx
    _WINDOWS_GET_FILE_INFORMATION_BY_HANDLE_EX.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    _WINDOWS_GET_FILE_INFORMATION_BY_HANDLE_EX.restype = wintypes.BOOL
    _WINDOWS_SET_FILE_INFORMATION_BY_HANDLE = _WINDOWS_KERNEL32.SetFileInformationByHandle
    _WINDOWS_SET_FILE_INFORMATION_BY_HANDLE.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    _WINDOWS_SET_FILE_INFORMATION_BY_HANDLE.restype = wintypes.BOOL
    _WINDOWS_CLOSE_HANDLE = _WINDOWS_KERNEL32.CloseHandle
    _WINDOWS_CLOSE_HANDLE.argtypes = [wintypes.HANDLE]
    _WINDOWS_CLOSE_HANDLE.restype = wintypes.BOOL


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


def _windows_directory_handle_state(handle: int) -> tuple[int, int, int, bytes]:
    if _WINDOWS_GET_FILE_INFORMATION_BY_HANDLE_EX is None:
        raise RuntimeError("isolated runtime GCM cleanup requires Windows handle APIs")
    attribute_information = _WindowsFileAttributeTagInfo()
    if not _WINDOWS_GET_FILE_INFORMATION_BY_HANDLE_EX(
        handle,
        _WINDOWS_FILE_ATTRIBUTE_TAG_INFO_CLASS,
        ctypes.byref(attribute_information),
        ctypes.sizeof(attribute_information),
    ):
        error = ctypes.WinError(ctypes.get_last_error())
        raise RuntimeError("isolated runtime GCM handle attribute query failed") from error
    identity_information = _WindowsFileIdInfo()
    if not _WINDOWS_GET_FILE_INFORMATION_BY_HANDLE_EX(
        handle,
        _WINDOWS_FILE_ID_INFO_CLASS,
        ctypes.byref(identity_information),
        ctypes.sizeof(identity_information),
    ):
        error = ctypes.WinError(ctypes.get_last_error())
        raise RuntimeError("isolated runtime GCM handle identity query failed") from error
    return (
        int(attribute_information.file_attributes),
        int(attribute_information.reparse_tag),
        int(identity_information.volume_serial_number),
        bytes(identity_information.file_id.identifier),
    )


def _remove_empty_gcm_commandline_sentinel(temporary: Path) -> None:
    entries = tuple(temporary.iterdir())
    if not entries:
        return
    expected = temporary / _GCM_COMMANDLINE_SENTINEL_DIRECTORY
    if len(entries) != 1 or entries[0].name != _GCM_COMMANDLINE_SENTINEL_DIRECTORY:
        raise RuntimeError("isolated runtime temp has unexpected contents")
    _direct_path(expected, directory=True)
    if (
        _WINDOWS_CREATE_FILE_W is None
        or _WINDOWS_SET_FILE_INFORMATION_BY_HANDLE is None
        or _WINDOWS_CLOSE_HANDLE is None
    ):
        raise RuntimeError("isolated runtime GCM cleanup requires Windows handle APIs")
    main_handle = _WINDOWS_CREATE_FILE_W(
        os.fspath(expected),
        _WINDOWS_DELETE | _WINDOWS_FILE_LIST_DIRECTORY | _WINDOWS_FILE_READ_ATTRIBUTES,
        _WINDOWS_FILE_SHARE_READ,
        None,
        _WINDOWS_OPEN_EXISTING,
        _WINDOWS_FILE_FLAG_BACKUP_SEMANTICS | _WINDOWS_FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if main_handle is None or main_handle == _WINDOWS_INVALID_HANDLE_VALUE:
        error = ctypes.WinError(ctypes.get_last_error())
        raise RuntimeError("isolated runtime GCM main handle open failed") from error
    try:
        initial_state = _windows_directory_handle_state(main_handle)
        if (
            not initial_state[0] & _WINDOWS_FILE_ATTRIBUTE_DIRECTORY
            or initial_state[0] & _WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT
            or initial_state[1] != 0
        ):
            raise RuntimeError("isolated runtime GCM handle is not a direct directory")
        temporary = _direct_path(temporary, directory=True)
        locked_entries = tuple(temporary.iterdir())
        if (
            len(locked_entries) != 1
            or locked_entries[0].name != _GCM_COMMANDLINE_SENTINEL_DIRECTORY
        ):
            raise RuntimeError("isolated runtime temp changed during locked cleanup")
        sentinel = _direct_path(expected, directory=True)
        if any(sentinel.iterdir()):
            raise RuntimeError("isolated runtime GCM sentinel is not empty")

        witness_handle = _WINDOWS_CREATE_FILE_W(
            os.fspath(expected),
            _WINDOWS_FILE_LIST_DIRECTORY | _WINDOWS_FILE_READ_ATTRIBUTES,
            _WINDOWS_FILE_SHARE_READ | _WINDOWS_FILE_SHARE_WRITE | _WINDOWS_FILE_SHARE_DELETE,
            None,
            _WINDOWS_OPEN_EXISTING,
            _WINDOWS_FILE_FLAG_BACKUP_SEMANTICS | _WINDOWS_FILE_FLAG_OPEN_REPARSE_POINT,
            None,
        )
        if witness_handle is None or witness_handle == _WINDOWS_INVALID_HANDLE_VALUE:
            error = ctypes.WinError(ctypes.get_last_error())
            raise RuntimeError("isolated runtime GCM witness handle open failed") from error
        try:
            witness_state = _windows_directory_handle_state(witness_handle)
        finally:
            if not _WINDOWS_CLOSE_HANDLE(witness_handle):
                error = ctypes.WinError(ctypes.get_last_error())
                raise RuntimeError("isolated runtime GCM witness handle close failed") from error
        if witness_state != initial_state:
            raise RuntimeError("isolated runtime GCM pathname identity changed")
        final_state = _windows_directory_handle_state(main_handle)
        if final_state != initial_state:
            raise RuntimeError("isolated runtime GCM handle identity changed")
        disposition = _WindowsFileDispositionInfo(delete_file=True)
        if not _WINDOWS_SET_FILE_INFORMATION_BY_HANDLE(
            main_handle,
            _WINDOWS_FILE_DISPOSITION_INFO_CLASS,
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            raise RuntimeError("isolated runtime GCM handle deletion failed") from error
    finally:
        if not _WINDOWS_CLOSE_HANDLE(main_handle):
            error = ctypes.WinError(ctypes.get_last_error())
            raise RuntimeError("isolated runtime GCM main handle close failed") from error
    temporary = _direct_path(temporary, directory=True)
    if _is_indirect(expected):
        raise RuntimeError("isolated runtime GCM path became indirect after cleanup")
    try:
        expected.lstat()
    except FileNotFoundError:
        pass
    except OSError as error:
        raise RuntimeError("isolated runtime GCM path verification failed") from error
    else:
        raise RuntimeError("isolated runtime GCM path remains after cleanup")
    if any(temporary.iterdir()):
        raise RuntimeError("isolated runtime temp changed during cleanup")


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
        "home/AppData/Roaming",
        "home/AppData/Local",
    ):
        directory = _direct_path(runtime_root / Path(relative), directory=True)
        if any(directory.iterdir()):
            raise RuntimeError("isolated runtime directory contains unexpected files")
    temporary = _direct_path(runtime_root / "temp", directory=True)
    _remove_empty_gcm_commandline_sentinel(temporary)
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


def _terminal_stage_contract(scope: str) -> tuple[str, ...]:
    if scope == "PREFLIGHT":
        return _EXPECTED_CHILD_FREEZE_PREFLIGHT_STAGES
    if scope == "ATTEMPT":
        return _EXPECTED_CHILD_FREEZE_ATTEMPT_STAGES
    if scope == "LAUNCHER":
        return _CHILD_FREEZE_LAUNCHER_STAGES
    raise RuntimeError("child freeze terminal stage scope is not allowlisted")


def _validate_terminal_report(report: object) -> dict[str, object]:
    if not isinstance(report, Mapping) or set(report) != _CHILD_FREEZE_TERMINAL_FIELDS:
        raise RuntimeError("child freeze terminal report fields differ")
    payload = dict(report)
    status = payload["status"]
    if not isinstance(status, str) or status not in _CHILD_FREEZE_TERMINAL_STATUSES:
        raise RuntimeError("child freeze terminal status is not allowlisted")
    scope = payload["stage_scope"]
    if not isinstance(scope, str) or scope not in _CHILD_FREEZE_STAGE_SCOPES:
        raise RuntimeError("child freeze terminal stage scope is not allowlisted")
    stage = payload["stage"]
    stages = _terminal_stage_contract(scope)
    ordinal = payload["stage_ordinal"]
    if (
        not isinstance(stage, str)
        or stage not in stages
        or isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or ordinal != stages.index(stage)
    ):
        raise RuntimeError("child freeze terminal stage evidence differs")
    authorization_state = payload["authorization_state"]
    if (
        not isinstance(authorization_state, str)
        or authorization_state not in _EXPECTED_CHILD_FREEZE_AUTHORIZATION_STATES
    ):
        raise RuntimeError("child freeze authorization state is not allowlisted")
    cleanup_state = payload["cleanup_state"]
    if (
        not isinstance(cleanup_state, str)
        or cleanup_state not in _EXPECTED_CHILD_FREEZE_CLEANUP_STATES
    ):
        raise RuntimeError("child freeze cleanup state is not allowlisted")
    output_state = payload["output_state"]
    if (
        not isinstance(output_state, str)
        or output_state not in _EXPECTED_CHILD_FREEZE_OUTPUT_STATES
    ):
        raise RuntimeError("child freeze output state is not allowlisted")
    reason = payload["failure_reason"]
    if reason is not None and (
        not isinstance(reason, str) or reason not in _EXPECTED_CHILD_FREEZE_FAILURE_REASONS
    ):
        raise RuntimeError("child freeze failure reason is not allowlisted")
    for field in (
        "child_publication_witnessed",
        "child_visibility_witnessed",
        "failure_receipt_written",
        "official_source_content_accessed",
    ):
        value = payload[field]
        if value is not None and type(value) is not bool:
            raise RuntimeError("child freeze terminal Boolean evidence differs")
    if type(payload["retry_authorized"]) is not bool:
        raise RuntimeError("child freeze retry state differs")
    if payload["child_publication_witnessed"] is True and (
        payload["child_visibility_witnessed"] is not True
    ):
        raise RuntimeError("child publication visibility evidence is inconsistent")
    if payload["failure_receipt_written"] is True and authorization_state != "CONSUMED":
        raise RuntimeError("child failure receipt authorization evidence is inconsistent")
    if payload["retry_authorized"] and (
        authorization_state != "NOT_CONSUMED"
        or output_state != "NONE"
        or cleanup_state == "FAILED"
    ):
        raise RuntimeError("child freeze retry evidence is inconsistent")
    if cleanup_state == "FAILED" and (
        scope != "LAUNCHER" or stage != "launcher_cleanup"
    ):
        raise RuntimeError("child freeze cleanup failure stage differs")
    if status == "OOD_EXTERNAL_V2_CHILD_FROZEN" and (
        scope != "ATTEMPT"
        or stage != _EXPECTED_CHILD_FREEZE_ATTEMPT_STAGES[-1]
        or authorization_state != "CONSUMED"
        or output_state != "DURABLE_EXACT"
        or payload["child_visibility_witnessed"] is not True
        or payload["child_publication_witnessed"] is not True
        or reason is not None
    ):
        raise RuntimeError("child freeze success evidence differs")
    if status in {
        "OOD_EXTERNAL_V2_CHILD_PREFLIGHT_VERIFIED",
        "OOD_EXTERNAL_V2_CHILD_PREFLIGHT_REFUSED",
    } and (scope != "PREFLIGHT" or authorization_state != "NOT_CONSUMED"):
        raise RuntimeError("child freeze preflight evidence differs")
    if status == "OOD_EXTERNAL_V2_CHILD_FREEZE_FAILED" and reason is None:
        raise RuntimeError("child freeze failure reason is missing")
    return payload


def _terminal_json(report: object) -> str:
    return json.dumps(
        _validate_terminal_report(report),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _terminal_handoff_bytes(*, exit_code: int, report: object) -> bytes:
    if (
        isinstance(exit_code, bool)
        or not isinstance(exit_code, int)
        or exit_code not in {0, 2, 3}
    ):
        raise RuntimeError("child freeze terminal exit code is not allowlisted")
    payload = {
        "exit_code": exit_code,
        "report": _validate_terminal_report(report),
    }
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _write_child_result_handoff(
    runtime_root: Path,
    *,
    exit_code: int,
    report: object,
) -> None:
    root = _direct_path(runtime_root, directory=True)
    path = root / _CHILD_RESULT_HANDOFF_FILENAME
    if _is_indirect(path):
        raise RuntimeError("isolated child result handoff path is indirect")
    payload = _terminal_handoff_bytes(exit_code=exit_code, report=report)
    if len(payload) > _MAX_CHILD_RESULT_HANDOFF_BYTES:
        raise RuntimeError("isolated child result handoff is too large")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _consume_child_result_handoff(runtime_root: Path) -> tuple[int, dict[str, object]]:
    root = _direct_path(runtime_root, directory=True)
    path = _direct_path(root / _CHILD_RESULT_HANDOFF_FILENAME, directory=False)
    try:
        if {entry.name for entry in root.iterdir()} != {
            "pycache",
            "temp",
            "home",
            _CHILD_RESULT_HANDOFF_FILENAME,
        }:
            raise RuntimeError("isolated child result handoff layout differs")
        if path.stat().st_size > _MAX_CHILD_RESULT_HANDOFF_BYTES:
            raise RuntimeError("isolated child result handoff is too large")
        observed = path.read_bytes()
        if not observed or len(observed) > _MAX_CHILD_RESULT_HANDOFF_BYTES:
            raise RuntimeError("isolated child result handoff size differs")
        try:
            decoded = json.loads(observed.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("isolated child result handoff is malformed") from error
        if not isinstance(decoded, Mapping) or set(decoded) != {"exit_code", "report"}:
            raise RuntimeError("isolated child result handoff fields differ")
        exit_code = decoded["exit_code"]
        report = _validate_terminal_report(decoded["report"])
        expected = _terminal_handoff_bytes(exit_code=exit_code, report=report)
        if not secrets.compare_digest(observed, expected):
            raise RuntimeError("isolated child result handoff is noncanonical")
        return exit_code, report
    finally:
        path.unlink()


def _terminalization_failure_report(
    report: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload = (
        dict(report)
        if report is not None
        else {
            "authorization_state": "UNVERIFIABLE",
            "child_publication_witnessed": None,
            "child_visibility_witnessed": None,
            "cleanup_state": "NOT_REACHED",
            "failure_reason": "UNEXPECTED_INTERNAL_FAILURE",
            "failure_receipt_written": None,
            "official_source_content_accessed": None,
            "output_state": "PRESENT_UNVERIFIABLE",
            "retry_authorized": False,
            "stage": "isolated_child_terminalization",
            "stage_ordinal": 0,
            "stage_scope": "LAUNCHER",
            "status": "OOD_EXTERNAL_V2_CHILD_FREEZE_FAILED",
        }
    )
    payload.update(
        {
            "cleanup_state": "NOT_REACHED",
            "failure_reason": "UNEXPECTED_INTERNAL_FAILURE",
            "retry_authorized": False,
            "stage": "isolated_child_terminalization",
            "stage_ordinal": 0,
            "stage_scope": "LAUNCHER",
            "status": "OOD_EXTERNAL_V2_CHILD_FREEZE_FAILED",
        }
    )
    return _validate_terminal_report(payload)


def _with_cleanup_state(
    report: Mapping[str, object],
    *,
    cleanup_state: str,
) -> dict[str, object]:
    payload = dict(report)
    payload["cleanup_state"] = cleanup_state
    if cleanup_state == "FAILED":
        payload.update(
            {
                "failure_reason": "UNEXPECTED_INTERNAL_FAILURE",
                "retry_authorized": False,
                "stage": "launcher_cleanup",
                "stage_ordinal": 1,
                "stage_scope": "LAUNCHER",
                "status": "OOD_EXTERNAL_V2_CHILD_FREEZE_FAILED",
            }
        )
    return _validate_terminal_report(payload)


def _emit_terminal_report(report: Mapping[str, object], *, exit_code: int) -> None:
    destination = sys.stdout if exit_code == 0 else sys.stderr
    print(_terminal_json(report), file=destination)


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
    passthrough_help = any(argument in {"-h", "--help"} for argument in arguments)
    report: dict[str, object] | None = None
    exit_code = 2
    try:
        command = [
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
        ]
        if passthrough_help:
            completed = subprocess.run(
                command,
                check=False,
                cwd=root,
                env=_sanitized_runtime_environment(runtime_root),
            )
        else:
            completed = subprocess.run(
                command,
                check=False,
                cwd=root,
                env=_sanitized_runtime_environment(runtime_root),
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )
    except BaseException:
        with suppress(OSError, RuntimeError, ValueError):
            _consume_parent_handoff(runtime_root, token=token)
        report = _terminalization_failure_report()
    else:
        exit_code = completed.returncode
        if not passthrough_help:
            try:
                child_exit_code, child_report = _consume_child_result_handoff(runtime_root)
            except BaseException:
                report = _terminalization_failure_report()
                exit_code = 2
            else:
                report = child_report
                if child_exit_code != completed.returncode:
                    report = _terminalization_failure_report(child_report)
                    exit_code = 2
                else:
                    exit_code = child_exit_code

    try:
        _remove_empty_runtime_root(runtime_root)
    except BaseException:
        if report is None:
            report = {
                "authorization_state": "NOT_CONSUMED",
                "child_publication_witnessed": False,
                "child_visibility_witnessed": False,
                "cleanup_state": "NOT_REACHED",
                "failure_reason": "UNEXPECTED_INTERNAL_FAILURE",
                "failure_receipt_written": False,
                "official_source_content_accessed": False,
                "output_state": "NONE",
                "retry_authorized": False,
                "stage": "launcher_cleanup",
                "stage_ordinal": 1,
                "stage_scope": "LAUNCHER",
                "status": "OOD_EXTERNAL_V2_CHILD_FREEZE_FAILED",
            }
        report = _with_cleanup_state(report, cleanup_state="FAILED")
        _emit_terminal_report(report, exit_code=2)
        return 2

    if passthrough_help:
        return exit_code
    if report is None:
        report = _terminalization_failure_report()
        exit_code = 2
    report = _with_cleanup_state(report, cleanup_state="CLEAN")
    _emit_terminal_report(report, exit_code=exit_code)
    return exit_code


def _enter_isolated_runtime() -> None:
    global _ISOLATED_RUNTIME_ACTIVE, _ISOLATED_RUNTIME_ROOT
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
    _ISOLATED_RUNTIME_ROOT = runtime_root


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

from ecg_trust.ood_v2.pipeline import (  # noqa: E402
    CHILD_FREEZE_ATTEMPT_STAGES,
    CHILD_FREEZE_AUTHORIZATION_STATES,
    CHILD_FREEZE_CLEANUP_STATES,
    CHILD_FREEZE_FAILURE_REASONS,
    CHILD_FREEZE_OUTPUT_STATES,
    CHILD_FREEZE_PREFLIGHT_STAGES,
    ChildFreezeAttemptError,
    ChildFreezePreflightStageError,
    freeze_external_v2_child_contract,
    verify_child_freeze_preflight,
)

if CHILD_FREEZE_PREFLIGHT_STAGES != _EXPECTED_CHILD_FREEZE_PREFLIGHT_STAGES:
    raise RuntimeError("child freeze preflight stage contract differs")
if CHILD_FREEZE_ATTEMPT_STAGES != _EXPECTED_CHILD_FREEZE_ATTEMPT_STAGES:
    raise RuntimeError("child freeze attempt stage contract differs")
if CHILD_FREEZE_FAILURE_REASONS != _EXPECTED_CHILD_FREEZE_FAILURE_REASONS:
    raise RuntimeError("child freeze failure reason contract differs")
if CHILD_FREEZE_OUTPUT_STATES != _EXPECTED_CHILD_FREEZE_OUTPUT_STATES:
    raise RuntimeError("child freeze output state contract differs")
if CHILD_FREEZE_AUTHORIZATION_STATES != _EXPECTED_CHILD_FREEZE_AUTHORIZATION_STATES:
    raise RuntimeError("child freeze authorization state contract differs")
if CHILD_FREEZE_CLEANUP_STATES != _EXPECTED_CHILD_FREEZE_CLEANUP_STATES:
    raise RuntimeError("child freeze cleanup state contract differs")


class _ChildFreezeStageTracker:
    """Accept only exact forward transitions through the frozen attempt stages."""

    def __init__(self) -> None:
        self._position = -1
        self._stage: str | None = None

    @property
    def stage(self) -> str | None:
        return self._stage

    @property
    def complete(self) -> bool:
        return self._position == len(CHILD_FREEZE_ATTEMPT_STAGES) - 1

    def transition(self, stage: str) -> None:
        if stage not in CHILD_FREEZE_ATTEMPT_STAGES:
            raise RuntimeError("child freeze stage is not allowlisted")
        position = CHILD_FREEZE_ATTEMPT_STAGES.index(stage)
        if position != self._position + 1:
            raise RuntimeError("child freeze stage order changed")
        self._position = position
        self._stage = stage


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify repeatable X11 child-freeze controls without durable writes",
    )
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path("configs/trust_sentinel_ood_external_v2_1.yaml"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path(
            "artifacts/trust_sentinel/ood_external_v2_1_preflight/private/"
            "external-waveform-inventory.json"
        ),
    )
    parser.add_argument(
        "--public-projection",
        type=Path,
        default=Path(
            "artifacts/trust_sentinel/ood_external_v2_1_preflight/public/"
            "external-inventory-summary.json"
        ),
    )
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--frozen-at-utc", required=True)
    parser.add_argument("--challenge-root", type=Path, required=True)
    parser.add_argument("--zzu-root", type=Path, required=True)
    parser.add_argument("--challenge-records", type=int, required=True)
    parser.add_argument("--zzu-records", type=int, required=True)
    parser.add_argument("--zzu-patients", type=int, required=True)
    parser.add_argument("--selected-records-total", type=int, required=True)
    parser.add_argument("--seven-zip-executable", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/trust_sentinel_ood_external_v2_1_execution.json"),
    )
    return parser


def _call_child_freeze_preflight(arguments: argparse.Namespace) -> None:
    verify_child_freeze_preflight(
        parent_path=arguments.parent,
        project_root=arguments.project_root,
        inventory_path=arguments.inventory,
        public_projection_path=arguments.public_projection,
        implementation_revision=arguments.implementation_revision,
        frozen_at_utc=arguments.frozen_at_utc,
        challenge_root=arguments.challenge_root,
        zzu_root=arguments.zzu_root,
        challenge_records=arguments.challenge_records,
        zzu_records=arguments.zzu_records,
        zzu_patients=arguments.zzu_patients,
        selected_records_total=arguments.selected_records_total,
        output_path=arguments.output,
        seven_zip_executable=arguments.seven_zip_executable,
    )


def _finish_terminal(report: Mapping[str, object], *, exit_code: int) -> int:
    validated = _validate_terminal_report(report)
    if _ISOLATED_RUNTIME_ACTIVE:
        if _ISOLATED_RUNTIME_ROOT is None:
            raise RuntimeError("isolated child result root is unavailable")
        _write_child_result_handoff(
            _ISOLATED_RUNTIME_ROOT,
            exit_code=exit_code,
            report=validated,
        )
    else:
        _emit_terminal_report(validated, exit_code=exit_code)
    return exit_code


def _preflight_report(*, stage: str, verified: bool) -> dict[str, object]:
    if stage not in CHILD_FREEZE_PREFLIGHT_STAGES:
        raise RuntimeError("child freeze preflight report stage is not allowlisted")
    return _validate_terminal_report(
        {
            "authorization_state": "NOT_CONSUMED",
            "child_publication_witnessed": False,
            "child_visibility_witnessed": False,
            "cleanup_state": "NOT_REACHED",
            "failure_reason": None if verified else "STAGE_REFUSED",
            "failure_receipt_written": False,
            "official_source_content_accessed": False,
            "output_state": "NONE" if verified else "PRESENT_UNVERIFIABLE",
            "retry_authorized": verified,
            "stage": stage,
            "stage_ordinal": CHILD_FREEZE_PREFLIGHT_STAGES.index(stage),
            "stage_scope": "PREFLIGHT",
            "status": (
                "OOD_EXTERNAL_V2_CHILD_PREFLIGHT_VERIFIED"
                if verified
                else "OOD_EXTERNAL_V2_CHILD_PREFLIGHT_REFUSED"
            ),
        }
    )


def _attempt_failure_report(
    error: ChildFreezeAttemptError,
    *,
    tracker: _ChildFreezeStageTracker,
    official_source_content_accessed: bool,
    child_visibility_witnessed: bool,
    child_publication_witnessed: bool,
) -> dict[str, object]:
    if (
        tracker.stage != error.stage
        or error.authorization_consumed is not True
        or error.official_source_content_accessed
        is not official_source_content_accessed
    ):
        return _terminalization_failure_report()
    return _validate_terminal_report(
        {
            "authorization_state": "CONSUMED",
            "child_publication_witnessed": child_publication_witnessed,
            "child_visibility_witnessed": child_visibility_witnessed,
            "cleanup_state": "NOT_REACHED",
            "failure_reason": error.reason,
            "failure_receipt_written": error.failure_receipt_written,
            "official_source_content_accessed": error.official_source_content_accessed,
            "output_state": error.output_state,
            "retry_authorized": False,
            "stage": error.stage,
            "stage_ordinal": CHILD_FREEZE_ATTEMPT_STAGES.index(error.stage),
            "stage_scope": "ATTEMPT",
            "status": "OOD_EXTERNAL_V2_CHILD_FREEZE_FAILED",
        }
    )


def _untyped_failure_report(
    *,
    tracker: _ChildFreezeStageTracker,
    official_source_content_accessed: bool,
    child_visibility_witnessed: bool,
    child_publication_witnessed: bool,
) -> dict[str, object]:
    if tracker.stage is None:
        return _terminalization_failure_report()
    return _validate_terminal_report(
        {
            "authorization_state": "UNVERIFIABLE",
            "child_publication_witnessed": child_publication_witnessed,
            "child_visibility_witnessed": child_visibility_witnessed,
            "cleanup_state": "NOT_REACHED",
            "failure_reason": "UNEXPECTED_INTERNAL_FAILURE",
            "failure_receipt_written": None,
            "official_source_content_accessed": official_source_content_accessed,
            "output_state": "PRESENT_UNVERIFIABLE",
            "retry_authorized": False,
            "stage": tracker.stage,
            "stage_ordinal": CHILD_FREEZE_ATTEMPT_STAGES.index(tracker.stage),
            "stage_scope": "ATTEMPT",
            "status": "OOD_EXTERNAL_V2_CHILD_FREEZE_FAILED",
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.preflight_only:
        try:
            _call_child_freeze_preflight(arguments)
        except ChildFreezePreflightStageError as error:
            return _finish_terminal(
                _preflight_report(stage=error.stage, verified=False),
                exit_code=3,
            )
        except Exception:
            return _finish_terminal(_terminalization_failure_report(), exit_code=3)
        return _finish_terminal(
            _preflight_report(
                stage=CHILD_FREEZE_PREFLIGHT_STAGES[-1],
                verified=True,
            ),
            exit_code=0,
        )

    tracker = _ChildFreezeStageTracker()
    official_source_content_accessed = False
    child_visibility_witnessed = False
    child_publication_witnessed = False

    def mark_official_source_content_accessed() -> None:
        nonlocal official_source_content_accessed
        official_source_content_accessed = True

    def mark_child_visible() -> None:
        nonlocal child_visibility_witnessed
        child_visibility_witnessed = True

    def mark_child_published() -> None:
        nonlocal child_publication_witnessed
        child_publication_witnessed = True

    try:
        freeze_external_v2_child_contract(
            parent_path=arguments.parent,
            project_root=arguments.project_root,
            inventory_path=arguments.inventory,
            public_projection_path=arguments.public_projection,
            implementation_revision=arguments.implementation_revision,
            frozen_at_utc=arguments.frozen_at_utc,
            challenge_root=arguments.challenge_root,
            zzu_root=arguments.zzu_root,
            challenge_records=arguments.challenge_records,
            zzu_records=arguments.zzu_records,
            zzu_patients=arguments.zzu_patients,
            selected_records_total=arguments.selected_records_total,
            output_path=arguments.output,
            seven_zip_executable=arguments.seven_zip_executable,
            stage_callback=tracker.transition,
            source_access_witness=mark_official_source_content_accessed,
            child_visibility_witness=mark_child_visible,
            child_publication_witness=mark_child_published,
        )
        if (
            not tracker.complete
            or not official_source_content_accessed
            or not child_visibility_witnessed
            or not child_publication_witnessed
        ):
            raise RuntimeError("child freeze success witness contract differs")
    except ChildFreezePreflightStageError as error:
        return _finish_terminal(
            _preflight_report(stage=error.stage, verified=False),
            exit_code=3,
        )
    except ChildFreezeAttemptError as error:
        return _finish_terminal(
            _attempt_failure_report(
                error,
                tracker=tracker,
                official_source_content_accessed=official_source_content_accessed,
                child_visibility_witnessed=child_visibility_witnessed,
                child_publication_witnessed=child_publication_witnessed,
            ),
            exit_code=2,
        )
    except Exception:
        return _finish_terminal(
            _untyped_failure_report(
                tracker=tracker,
                official_source_content_accessed=official_source_content_accessed,
                child_visibility_witnessed=child_visibility_witnessed,
                child_publication_witnessed=child_publication_witnessed,
            ),
            exit_code=2,
        )
    return _finish_terminal(
        {
            "authorization_state": "CONSUMED",
            "child_publication_witnessed": True,
            "child_visibility_witnessed": True,
            "cleanup_state": "NOT_REACHED",
            "failure_reason": None,
            "failure_receipt_written": False,
            "official_source_content_accessed": True,
            "output_state": "DURABLE_EXACT",
            "retry_authorized": False,
            "stage": CHILD_FREEZE_ATTEMPT_STAGES[-1],
            "stage_ordinal": len(CHILD_FREEZE_ATTEMPT_STAGES) - 1,
            "stage_scope": "ATTEMPT",
            "status": "OOD_EXTERNAL_V2_CHILD_FROZEN",
        },
        exit_code=0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
