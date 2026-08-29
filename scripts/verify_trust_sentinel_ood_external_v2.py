#!/usr/bin/env python3
"""Verify external OOD v2 metadata or an immutable completed bundle."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

_ISOLATED_CHILD_FLAG = "--_ecg-trust-ood-v2-isolated-child"
_RUNTIME_ROOT_PREFIX = ".ood_external_v2_1.runtime-"
_HANDOFF_FILENAME = ".parent-handoff"
_FROZEN_WINDOWS_DIRECTORY = Path(r"C:\Windows")
_FROZEN_PROGRAM_FILES_DIRECTORY = Path(r"C:\Program Files")
_FROZEN_PROGRAM_FILES_X86_DIRECTORY = Path(r"C:\Program Files (x86)")
_FROZEN_PROGRAM_DATA_DIRECTORY = Path(r"C:\ProgramData")


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

from ecg_trust.ood_v2.bundle import verify_external_v2_bundle  # noqa: E402
from ecg_trust.ood_v2.pipeline import (  # noqa: E402
    EXPECTED_PARENT_CONFIG_SHA256,
    OODExternalV2ParentConfig,
    SuccessorParentPreflight,
    verify_external_v2_metadata,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path("configs/trust_sentinel_ood_external_v2_1.yaml"),
    )
    parser.add_argument("--child", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--code-revision")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seven-zip-executable", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.output_root is not None:
            verified = verify_external_v2_bundle(
                arguments.output_root,
                project_root=arguments.project_root,
                seven_zip_executable=arguments.seven_zip_executable,
            )
            payload = {
                "result_artifact_sha256": verified.result.artifact_sha256,
                "status": "OOD_EXTERNAL_V2_BUNDLE_VERIFIED",
            }
        else:
            verified_metadata = verify_external_v2_metadata(
                parent_path=arguments.parent,
                child_path=arguments.child,
                project_root=arguments.project_root,
                code_revision=arguments.code_revision,
                seven_zip_executable=arguments.seven_zip_executable,
            )
            if isinstance(verified_metadata, OODExternalV2ParentConfig):
                payload = {
                    "parent_config_file_sha256": EXPECTED_PARENT_CONFIG_SHA256,
                    "status": "PRE_INFERENCE_PROTOCOL_INFEASIBLE",
                }
            elif isinstance(verified_metadata, SuccessorParentPreflight):
                payload = {
                    "parent_config_file_sha256": verified_metadata.file_sha256,
                    "status": "SUCCESSOR_PARENT_PREFLIGHT_VERIFIED_NOT_EXECUTABLE",
                }
            else:
                payload = {
                    "child_contract_file_sha256": verified_metadata.child.file_sha256,
                    "status": "OOD_EXTERNAL_V2_METADATA_VERIFIED",
                }
    except Exception:
        print(
            json.dumps(
                {"status": "OOD_EXTERNAL_V2_VERIFICATION_FAILED"},
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
