from __future__ import annotations

import ctypes
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Protocol

import pytest

import scripts.build_trust_sentinel_ood_v2_inventory as inventory_cli
import scripts.evaluate_trust_sentinel_ood_external_v2 as evaluate_cli
import scripts.freeze_trust_sentinel_ood_external_v2 as freeze_cli
import scripts.verify_trust_sentinel_ood_external_v2 as verify_cli


class _LauncherModule(Protocol):
    __file__: str
    _GCM_COMMANDLINE_SENTINEL_DIRECTORY: str
    _ISOLATED_CHILD_FLAG: str
    _HANDOFF_FILENAME: str
    _RUNTIME_ROOT_PREFIX: str
    _WINDOWS_CLOSE_HANDLE: object
    _WINDOWS_CREATE_FILE_W: object
    _WINDOWS_FILE_ATTRIBUTE_DIRECTORY: int
    _WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT: int
    _WINDOWS_FILE_ATTRIBUTE_TAG_INFO_CLASS: int
    _WINDOWS_FILE_DISPOSITION_INFO_CLASS: int
    _WINDOWS_FILE_FLAG_BACKUP_SEMANTICS: int
    _WINDOWS_FILE_FLAG_OPEN_REPARSE_POINT: int
    _WINDOWS_FILE_ID_INFO_CLASS: int
    _WINDOWS_FILE_LIST_DIRECTORY: int
    _WINDOWS_FILE_READ_ATTRIBUTES: int
    _WINDOWS_FILE_SHARE_DELETE: int
    _WINDOWS_FILE_SHARE_READ: int
    _WINDOWS_FILE_SHARE_WRITE: int
    _WINDOWS_GET_FILE_INFORMATION_BY_HANDLE_EX: object
    _WINDOWS_INVALID_HANDLE_VALUE: int
    _WINDOWS_OPEN_EXISTING: int
    _WINDOWS_SET_FILE_INFORMATION_BY_HANDLE: object
    secrets: ModuleType
    subprocess: ModuleType

    def _sanitized_runtime_environment(self, runtime_root: Path) -> dict[str, str]: ...

    def _relaunch_isolated(self, arguments: tuple[str, ...]) -> int: ...

    def _remove_empty_gcm_commandline_sentinel(self, temporary: Path) -> None: ...

    def _remove_empty_runtime_root(self, runtime_root: Path) -> None: ...

    def _windows_directory_handle_state(
        self,
        handle: int,
    ) -> tuple[int, int, int, bytes]: ...


def _make_runtime_root(parent: Path, *, name: str) -> Path:
    runtime_root = parent / name
    (runtime_root / "pycache").mkdir(parents=True)
    (runtime_root / "temp").mkdir()
    (runtime_root / "home" / "AppData" / "Roaming").mkdir(parents=True)
    (runtime_root / "home" / "AppData" / "Local").mkdir()
    return runtime_root


def _freeze_arguments(*, preflight_only: bool = False) -> list[str]:
    arguments = [
        "--implementation-revision",
        "a" * 40,
        "--frozen-at-utc",
        "2026-08-30T10:00:00Z",
        "--challenge-root",
        "challenge",
        "--zzu-root",
        "zzu",
        "--challenge-records",
        "1000",
        "--zzu-records",
        "12328",
        "--zzu-patients",
        "10350",
        "--selected-records-total",
        "13328",
        "--seven-zip-executable",
        "7z",
    ]
    if preflight_only:
        arguments.insert(0, "--preflight-only")
    return arguments


def _successful_freeze_report(*, cleanup_state: str = "NOT_REACHED") -> dict[str, object]:
    return {
        "authorization_state": "CONSUMED",
        "child_publication_witnessed": True,
        "child_visibility_witnessed": True,
        "cleanup_state": cleanup_state,
        "failure_reason": None,
        "failure_receipt_written": False,
        "official_source_content_accessed": True,
        "output_state": "DURABLE_EXACT",
        "retry_authorized": False,
        "stage": freeze_cli.CHILD_FREEZE_ATTEMPT_STAGES[-1],
        "stage_ordinal": len(freeze_cli.CHILD_FREEZE_ATTEMPT_STAGES) - 1,
        "stage_scope": "ATTEMPT",
        "status": "OOD_EXTERNAL_V2_CHILD_FROZEN",
    }


@pytest.mark.parametrize(
    "module",
    [inventory_cli, freeze_cli, evaluate_cli, verify_cli],
)
def test_runtime_launcher_sanitizes_code_injection_environment(
    module: _LauncherModule,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dangerous = {
        "COVERAGE_PROCESS_START": "attacker.py",
        "CUBLAS_WORKSPACE_CONFIG": ":16:8",
        "CUDA_INJECTION64_PATH": r"C:\attacker.dll",
        "CUDA_CACHE_DISABLE": "0",
        "CUDA_VISIBLE_DEVICES": "7",
        "CUDNN_LOGDEST_DBG": r"C:\secret.log",
        "LD_PRELOAD": "/tmp/attacker.so",
        "NVIDIA_TF32_OVERRIDE": "1",
        "OMP_NUM_THREADS": "999",
        "PYTHONPATH": r"C:\attacker",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_LOGS": "+all",
        "TORCHINDUCTOR_CACHE_DIR": r"C:\attacker-cache",
    }
    for name, value in dangerous.items():
        monkeypatch.setenv(name, value)
    runtime_root = _make_runtime_root(tmp_path, name="runtime")

    environment = module._sanitized_runtime_environment(runtime_root)

    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert environment["CUDA_CACHE_DISABLE"] == "1"
    assert environment["PATH"].casefold() == os.fspath(Path(r"C:\Windows") / "System32").casefold()
    assert environment["PROGRAMFILES"] == r"C:\Program Files"
    assert environment["PROGRAMFILES(X86)"] == r"C:\Program Files (x86)"
    assert environment["PROGRAMW6432"] == r"C:\Program Files"
    assert environment["TEMP"] == os.fspath(runtime_root / "temp")
    assert environment["TMP"] == environment["TEMP"]
    assert environment["TORCHINDUCTOR_CACHE_DIR"] == environment["TEMP"]
    assert environment["USERPROFILE"] == os.fspath(runtime_root / "home")
    assert environment["APPDATA"] == os.fspath(runtime_root / "home" / "AppData" / "Roaming")
    assert environment["LOCALAPPDATA"] == os.fspath(runtime_root / "home" / "AppData" / "Local")
    assert all(
        name not in environment
        for name in dangerous
        if name
        not in {
            "CUBLAS_WORKSPACE_CONFIG",
            "CUDA_CACHE_DISABLE",
            "TORCHINDUCTOR_CACHE_DIR",
        }
    )


@pytest.mark.parametrize(
    "module",
    [inventory_cli, freeze_cli, evaluate_cli, verify_cli],
)
def test_runtime_launcher_constructs_exact_isolated_child_and_cleans_cache(
    module: _LauncherModule,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    cache_parent = root / "artifacts" / "trust_sentinel"
    site_packages = root / ".venv" / "Lib" / "site-packages"
    project_src = root / "src"
    script = root / "scripts" / "entry.py"
    cache_parent.mkdir(parents=True)
    site_packages.mkdir(parents=True)
    project_src.mkdir()
    script.parent.mkdir()
    script.write_text("# bound\n", encoding="utf-8")
    marker = "a" * 64
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        module,
        "_project_layout",
        lambda: (script, root, site_packages, project_src),
    )
    monkeypatch.setattr(module.secrets, "token_hex", lambda _: marker)

    def fake_run(
        arguments: list[str],
        *,
        check: bool,
        cwd: Path,
        env: dict[str, str],
    ) -> SimpleNamespace:
        observed.update(arguments=arguments, check=check, cwd=cwd, environment=env)
        runtime_root = cache_parent / f"{module._RUNTIME_ROOT_PREFIX}{marker}"
        cache = runtime_root / "pycache"
        assert cache.is_dir()
        assert tuple(cache.iterdir()) == ()
        assert set(path.name for path in runtime_root.iterdir()) == {
            "pycache",
            "temp",
            "home",
            module._HANDOFF_FILENAME,
        }
        assert (runtime_root / module._HANDOFF_FILENAME).read_bytes() == (
            f"{marker}\n".encode("ascii")
        )
        (runtime_root / module._HANDOFF_FILENAME).unlink()
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module._relaunch_isolated(("--help",)) == 7
    arguments = observed["arguments"]
    assert isinstance(arguments, list)
    assert arguments[1:6] == [
        "-I",
        "-S",
        "-B",
        "-X",
        f"pycache_prefix={cache_parent / f'{module._RUNTIME_ROOT_PREFIX}{marker}' / 'pycache'}",
    ]
    assert arguments[6:] == [
        os.fspath(script),
        module._ISOLATED_CHILD_FLAG,
        marker,
        "--help",
    ]
    assert observed["check"] is False
    assert observed["cwd"] == root
    assert not (cache_parent / f"{module._RUNTIME_ROOT_PREFIX}{marker}").exists()


def test_child_result_handoff_is_bounded_canonical_and_consumed(
    tmp_path: Path,
) -> None:
    runtime_root = _make_runtime_root(tmp_path, name="runtime")
    report = freeze_cli._preflight_report(
        stage=freeze_cli.CHILD_FREEZE_PREFLIGHT_STAGES[-1],
        verified=True,
    )

    freeze_cli._write_child_result_handoff(
        runtime_root,
        exit_code=0,
        report=report,
    )
    result_path = runtime_root / freeze_cli._CHILD_RESULT_HANDOFF_FILENAME
    assert result_path.stat().st_size <= freeze_cli._MAX_CHILD_RESULT_HANDOFF_BYTES

    exit_code, observed = freeze_cli._consume_child_result_handoff(runtime_root)

    assert exit_code == 0
    assert observed == report
    assert not result_path.exists()
    freeze_cli._remove_empty_runtime_root(runtime_root)


def test_child_result_handoff_rejects_noncanonical_bytes_without_disclosure(
    tmp_path: Path,
) -> None:
    runtime_root = _make_runtime_root(tmp_path, name="runtime")
    report = freeze_cli._preflight_report(
        stage=freeze_cli.CHILD_FREEZE_PREFLIGHT_STAGES[-1],
        verified=True,
    )
    payload = freeze_cli._terminal_handoff_bytes(exit_code=0, report=report)
    result_path = runtime_root / freeze_cli._CHILD_RESULT_HANDOFF_FILENAME
    result_path.write_bytes(payload + b"\n")

    with pytest.raises(RuntimeError, match="noncanonical"):
        freeze_cli._consume_child_result_handoff(runtime_root)

    assert not result_path.exists()
    freeze_cli._remove_empty_runtime_root(runtime_root)


def test_child_freeze_launcher_suppresses_process_output_and_reports_after_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "project"
    cache_parent = root / "artifacts" / "trust_sentinel"
    site_packages = root / ".venv" / "Lib" / "site-packages"
    project_src = root / "src"
    script = root / "scripts" / "entry.py"
    cache_parent.mkdir(parents=True)
    site_packages.mkdir(parents=True)
    project_src.mkdir()
    script.parent.mkdir()
    script.write_text("# bound\n", encoding="utf-8")
    token = "1" * 64
    runtime_root = cache_parent / f"{freeze_cli._RUNTIME_ROOT_PREFIX}{token}"
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        freeze_cli,
        "_project_layout",
        lambda: (script, root, site_packages, project_src),
    )
    monkeypatch.setattr(freeze_cli.secrets, "token_hex", lambda _: token)
    monkeypatch.setattr(freeze_cli, "_sanitized_runtime_environment", lambda _: {})

    def fake_run(
        arguments: list[str],
        *,
        check: bool,
        cwd: Path,
        env: dict[str, str],
        stderr: object,
        stdout: object,
    ) -> SimpleNamespace:
        observed.update(
            arguments=arguments,
            check=check,
            cwd=cwd,
            env=env,
            stderr=stderr,
            stdout=stdout,
        )
        freeze_cli._consume_parent_handoff(runtime_root, token=token)
        freeze_cli._write_child_result_handoff(
            runtime_root,
            exit_code=0,
            report=_successful_freeze_report(),
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(freeze_cli.subprocess, "run", fake_run)

    assert freeze_cli._relaunch_isolated(("--production",)) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    assert json.loads(captured.out) == _successful_freeze_report(cleanup_state="CLEAN")
    assert observed["stderr"] is subprocess.DEVNULL
    assert observed["stdout"] is subprocess.DEVNULL
    assert not runtime_root.exists()


def test_child_freeze_launcher_cleanup_failure_preserves_child_truth_and_never_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "project"
    cache_parent = root / "artifacts" / "trust_sentinel"
    site_packages = root / ".venv" / "Lib" / "site-packages"
    project_src = root / "src"
    script = root / "scripts" / "entry.py"
    cache_parent.mkdir(parents=True)
    site_packages.mkdir(parents=True)
    project_src.mkdir()
    script.parent.mkdir()
    script.write_text("# bound\n", encoding="utf-8")
    token = "2" * 64
    runtime_root = cache_parent / f"{freeze_cli._RUNTIME_ROOT_PREFIX}{token}"

    monkeypatch.setattr(
        freeze_cli,
        "_project_layout",
        lambda: (script, root, site_packages, project_src),
    )
    monkeypatch.setattr(freeze_cli.secrets, "token_hex", lambda _: token)
    monkeypatch.setattr(freeze_cli, "_sanitized_runtime_environment", lambda _: {})

    def fake_run(
        _arguments: list[str],
        *,
        check: bool,
        cwd: Path,
        env: dict[str, str],
        stderr: object,
        stdout: object,
    ) -> SimpleNamespace:
        del check, cwd, env, stderr, stdout
        freeze_cli._consume_parent_handoff(runtime_root, token=token)
        freeze_cli._write_child_result_handoff(
            runtime_root,
            exit_code=0,
            report=_successful_freeze_report(),
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(freeze_cli.subprocess, "run", fake_run)
    original_cleanup = freeze_cli._remove_empty_runtime_root
    cleanup_calls = 0

    def fail_cleanup(_: Path) -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        raise RuntimeError(r"secret C:\private\patient-123")

    monkeypatch.setattr(freeze_cli, "_remove_empty_runtime_root", fail_cleanup)

    assert freeze_cli._relaunch_isolated(("--production",)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    report = json.loads(captured.err)
    assert report["authorization_state"] == "CONSUMED"
    assert report["child_visibility_witnessed"] is True
    assert report["child_publication_witnessed"] is True
    assert report["output_state"] == "DURABLE_EXACT"
    assert report["cleanup_state"] == "FAILED"
    assert report["stage"] == "launcher_cleanup"
    assert report["failure_reason"] == "UNEXPECTED_INTERNAL_FAILURE"
    assert report["retry_authorized"] is False
    assert cleanup_calls == 1
    assert runtime_root.is_dir()
    assert "secret" not in captured.err
    assert "private" not in captured.err
    assert "patient" not in captured.err
    original_cleanup(runtime_root)


def test_child_freeze_launcher_missing_handoff_is_unverifiable_and_cleanup_is_truthful(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "project"
    cache_parent = root / "artifacts" / "trust_sentinel"
    site_packages = root / ".venv" / "Lib" / "site-packages"
    project_src = root / "src"
    script = root / "scripts" / "entry.py"
    cache_parent.mkdir(parents=True)
    site_packages.mkdir(parents=True)
    project_src.mkdir()
    script.parent.mkdir()
    script.write_text("# bound\n", encoding="utf-8")
    token = "3" * 64
    runtime_root = cache_parent / f"{freeze_cli._RUNTIME_ROOT_PREFIX}{token}"

    monkeypatch.setattr(
        freeze_cli,
        "_project_layout",
        lambda: (script, root, site_packages, project_src),
    )
    monkeypatch.setattr(freeze_cli.secrets, "token_hex", lambda _: token)
    monkeypatch.setattr(freeze_cli, "_sanitized_runtime_environment", lambda _: {})

    def fake_run(
        _arguments: list[str],
        *,
        check: bool,
        cwd: Path,
        env: dict[str, str],
        stderr: object,
        stdout: object,
    ) -> SimpleNamespace:
        del check, cwd, env, stderr, stdout
        freeze_cli._consume_parent_handoff(runtime_root, token=token)
        return SimpleNamespace(returncode=99)

    monkeypatch.setattr(freeze_cli.subprocess, "run", fake_run)

    assert freeze_cli._relaunch_isolated(("--production",)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    report = json.loads(captured.err)
    assert report["authorization_state"] == "UNVERIFIABLE"
    assert report["official_source_content_accessed"] is None
    assert report["failure_receipt_written"] is None
    assert report["output_state"] == "PRESENT_UNVERIFIABLE"
    assert report["cleanup_state"] == "CLEAN"
    assert report["stage"] == "isolated_child_terminalization"
    assert report["retry_authorized"] is False
    assert not runtime_root.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows handle deletion contract")
@pytest.mark.parametrize(
    "module",
    [inventory_cli, freeze_cli, evaluate_cli, verify_cli],
)
def test_runtime_launcher_accepts_exact_empty_gcm_sentinel_after_child_success(
    module: _LauncherModule,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    cache_parent = root / "artifacts" / "trust_sentinel"
    site_packages = root / ".venv" / "Lib" / "site-packages"
    project_src = root / "src"
    script = root / "scripts" / "entry.py"
    cache_parent.mkdir(parents=True)
    site_packages.mkdir(parents=True)
    project_src.mkdir()
    script.parent.mkdir()
    script.write_text("# bound\n", encoding="utf-8")
    marker = "d" * 64
    runtime_root = cache_parent / f"{module._RUNTIME_ROOT_PREFIX}{marker}"
    protected_paths = (
        root / "data" / "raw" / "external-ood",
        cache_parent / ".ood_external_v2_1.x4-inventory-build-attempt.json",
        cache_parent / ".ood_external_v2_1.x5-inventory-build-attempt.json",
        cache_parent / ".ood_external_v2_1.x6-inventory-build-attempt.json",
        cache_parent / ".ood_external_v2_1.one-shot-claim.json",
        cache_parent / "ood_external_v2_1",
        cache_parent / "ood_external_v2_1_preflight/private/external-waveform-inventory.json",
        cache_parent / "ood_external_v2_1_preflight/public/external-inventory-summary.json",
        root / "configs" / "trust_sentinel_ood_external_v2_1_execution.json",
    )
    sibling_canary = cache_parent / "launcher-cleanup-canary.txt"
    sibling_canary.write_bytes(b"must remain")
    assert all(not path.exists() for path in protected_paths)

    monkeypatch.setattr(
        module,
        "_project_layout",
        lambda: (script, root, site_packages, project_src),
    )
    monkeypatch.setattr(module.secrets, "token_hex", lambda _: marker)

    def fake_run(
        arguments: list[str],
        *,
        check: bool,
        cwd: Path,
        env: dict[str, str],
    ) -> SimpleNamespace:
        del arguments
        assert check is False
        assert cwd == root
        assert env["TEMP"] == os.fspath(runtime_root / "temp")
        assert env["TMP"] == env["TEMP"]
        (runtime_root / module._HANDOFF_FILENAME).unlink()
        sentinel = runtime_root / "temp" / module._GCM_COMMANDLINE_SENTINEL_DIRECTORY
        sentinel.mkdir()
        assert tuple(sentinel.iterdir()) == ()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module._relaunch_isolated(("--help",)) == 0
    assert not runtime_root.exists()
    assert sibling_canary.read_bytes() == b"must remain"
    assert all(not path.exists() for path in protected_paths)
    assert not tuple(cache_parent.glob(".*-inventory-build-attempt.json"))


@pytest.mark.parametrize(
    "module",
    [inventory_cli, freeze_cli, evaluate_cli, verify_cli],
)
@pytest.mark.parametrize(
    "invalid_state",
    (
        "exact_name_file",
        "nonempty_file",
        "nonempty_directory",
        "alternate_name",
        "case_variant",
        "sibling_entry",
    ),
)
def test_runtime_cleanup_rejects_every_nonexact_gcm_temp_state(
    module: _LauncherModule,
    invalid_state: str,
    tmp_path: Path,
) -> None:
    runtime_root = _make_runtime_root(tmp_path, name="runtime")
    temporary = runtime_root / "temp"
    sentinel = temporary / module._GCM_COMMANDLINE_SENTINEL_DIRECTORY
    if invalid_state == "exact_name_file":
        sentinel.write_bytes(b"")
    elif invalid_state == "nonempty_file":
        sentinel.mkdir()
        (sentinel / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    elif invalid_state == "nonempty_directory":
        (sentinel / "unexpected").mkdir(parents=True)
    elif invalid_state == "alternate_name":
        (temporary / "system-commandline-sentinel-file").mkdir()
    elif invalid_state == "case_variant":
        (temporary / "System-Commandline-Sentinel-Files").mkdir()
    else:
        sentinel.mkdir()
        (temporary / "unexpected.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(RuntimeError):
        module._remove_empty_runtime_root(runtime_root)

    assert runtime_root.is_dir()
    assert any(temporary.iterdir())


@pytest.mark.parametrize(
    "module",
    [inventory_cli, freeze_cli, evaluate_cli, verify_cli],
)
def test_runtime_cleanup_rejects_gcm_sentinel_symlink_without_touching_target(
    module: _LauncherModule,
    tmp_path: Path,
) -> None:
    runtime_root = _make_runtime_root(tmp_path / "owned", name="runtime")
    target = tmp_path / "external-target"
    target.mkdir()
    sentinel = runtime_root / "temp" / module._GCM_COMMANDLINE_SENTINEL_DIRECTORY
    try:
        sentinel.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")

    with pytest.raises(RuntimeError, match="indirect"):
        module._remove_empty_runtime_root(runtime_root)

    assert sentinel.is_symlink()
    assert target.is_dir()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction contract")
@pytest.mark.parametrize(
    "module",
    [inventory_cli, freeze_cli, evaluate_cli, verify_cli],
)
def test_runtime_cleanup_rejects_gcm_sentinel_junction_without_touching_target(
    module: _LauncherModule,
    tmp_path: Path,
) -> None:
    runtime_root = _make_runtime_root(tmp_path / "owned", name="runtime")
    target = tmp_path / "external-target"
    target.mkdir()
    sentinel = runtime_root / "temp" / module._GCM_COMMANDLINE_SENTINEL_DIRECTORY
    completed = subprocess.run(
        [
            r"C:\Windows\System32\cmd.exe",
            "/d",
            "/c",
            "mklink",
            "/J",
            os.fspath(sentinel),
            os.fspath(target),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert sentinel.is_junction()
    try:
        with pytest.raises(RuntimeError, match="indirect"):
            module._remove_empty_runtime_root(runtime_root)
        assert sentinel.is_junction()
        assert target.is_dir()
    finally:
        if sentinel.is_junction():
            os.rmdir(sentinel)


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-sharing contract")
@pytest.mark.parametrize(
    "module",
    [inventory_cli, freeze_cli, evaluate_cli, verify_cli],
)
@pytest.mark.parametrize("replacement_kind", ("direct_directory", "junction"))
def test_runtime_cleanup_blocks_final_gap_directory_and_junction_swaps(
    module: _LauncherModule,
    replacement_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = _make_runtime_root(tmp_path / "owned", name="runtime")
    temporary = runtime_root / "temp"
    sentinel = temporary / module._GCM_COMMANDLINE_SENTINEL_DIRECTORY
    sentinel.mkdir()
    replacement = tmp_path / f"replacement-{replacement_kind}"
    target = tmp_path / "external-target"
    target.mkdir()
    canary = target / "canary.txt"
    canary.write_bytes(b"must remain")
    if replacement_kind == "direct_directory":
        replacement.mkdir()
    else:
        completed = subprocess.run(
            [
                r"C:\Windows\System32\cmd.exe",
                "/d",
                "/c",
                "mklink",
                "/J",
                os.fspath(replacement),
                os.fspath(target),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert replacement.is_junction()
    displaced = tmp_path / f"displaced-{replacement_kind}"
    original_set_information = module._WINDOWS_SET_FILE_INFORMATION_BY_HANDLE
    assert callable(original_set_information)
    blocked_renames: list[OSError] = []

    def race_at_disposition(*arguments: object) -> object:
        try:
            sentinel.rename(displaced)
        except OSError as error:
            blocked_renames.append(error)
        else:
            replacement.rename(sentinel)
        return original_set_information(*arguments)

    monkeypatch.setattr(
        module,
        "_WINDOWS_SET_FILE_INFORMATION_BY_HANDLE",
        race_at_disposition,
    )
    try:
        module._remove_empty_gcm_commandline_sentinel(temporary)

        assert blocked_renames
        assert not sentinel.exists()
        assert not displaced.exists()
        assert replacement.is_dir()
        assert canary.read_bytes() == b"must remain"
    finally:
        for candidate in (replacement, sentinel):
            if candidate.is_junction():
                os.rmdir(candidate)


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-bound emptiness contract")
@pytest.mark.parametrize(
    "module",
    [inventory_cli, freeze_cli, evaluate_cli, verify_cli],
)
def test_runtime_cleanup_fails_closed_if_sentinel_gains_content_at_disposition(
    module: _LauncherModule,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = _make_runtime_root(tmp_path, name="runtime")
    sentinel = runtime_root / "temp" / module._GCM_COMMANDLINE_SENTINEL_DIRECTORY
    sentinel.mkdir()
    late = sentinel / "late.txt"
    original_set_information = module._WINDOWS_SET_FILE_INFORMATION_BY_HANDLE
    assert callable(original_set_information)

    def add_content_at_disposition(*arguments: object) -> object:
        late.write_text("late", encoding="utf-8")
        return original_set_information(*arguments)

    monkeypatch.setattr(
        module,
        "_WINDOWS_SET_FILE_INFORMATION_BY_HANDLE",
        add_content_at_disposition,
    )

    with pytest.raises(RuntimeError, match="handle deletion failed"):
        module._remove_empty_runtime_root(runtime_root)

    assert late.read_text(encoding="utf-8") == "late"
    assert runtime_root.is_dir()


@pytest.mark.skipif(os.name != "nt", reason="Windows handle identity contract")
@pytest.mark.parametrize(
    "module",
    [inventory_cli, freeze_cli, evaluate_cli, verify_cli],
)
@pytest.mark.parametrize(
    ("changed_query", "message"),
    ((2, "pathname identity changed"), (3, "handle identity changed")),
)
def test_runtime_cleanup_rejects_witness_and_final_main_identity_changes(
    module: _LauncherModule,
    changed_query: int,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = _make_runtime_root(tmp_path, name="runtime")
    sentinel = runtime_root / "temp" / module._GCM_COMMANDLINE_SENTINEL_DIRECTORY
    sentinel.mkdir()
    original_state = module._windows_directory_handle_state
    query_count = 0

    def drifting_state(handle: int) -> tuple[int, int, int, bytes]:
        nonlocal query_count
        state = original_state(handle)
        query_count += 1
        if query_count != changed_query:
            return state
        changed_identifier = bytes((state[3][0] ^ 1,)) + state[3][1:]
        return state[0], state[1], state[2], changed_identifier

    monkeypatch.setattr(module, "_windows_directory_handle_state", drifting_state)

    with pytest.raises(RuntimeError, match=message):
        module._remove_empty_runtime_root(runtime_root)

    assert sentinel.is_dir()
    assert runtime_root.is_dir()


@pytest.mark.skipif(os.name != "nt", reason="Windows handle API contract")
@pytest.mark.parametrize(
    "failed_api",
    ("CreateFileW", "GetFileInformationByHandleEx", "SetFileInformationByHandle", "CloseHandle"),
)
def test_runtime_cleanup_fails_closed_on_each_windows_api_failure(
    failed_api: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = evaluate_cli
    runtime_root = _make_runtime_root(tmp_path, name="runtime")
    sentinel = runtime_root / "temp" / module._GCM_COMMANDLINE_SENTINEL_DIRECTORY
    sentinel.mkdir()

    if failed_api == "CreateFileW":

        def fail_create_file(*arguments: object) -> int:
            del arguments
            ctypes.set_last_error(5)
            return module._WINDOWS_INVALID_HANDLE_VALUE

        monkeypatch.setattr(module, "_WINDOWS_CREATE_FILE_W", fail_create_file)
        message = "main handle open failed"
    elif failed_api == "GetFileInformationByHandleEx":

        def fail_information_query(*arguments: object) -> int:
            del arguments
            ctypes.set_last_error(5)
            return 0

        monkeypatch.setattr(
            module,
            "_WINDOWS_GET_FILE_INFORMATION_BY_HANDLE_EX",
            fail_information_query,
        )
        message = "handle attribute query failed"
    elif failed_api == "SetFileInformationByHandle":

        def fail_disposition(*arguments: object) -> int:
            del arguments
            ctypes.set_last_error(5)
            return 0

        monkeypatch.setattr(
            module,
            "_WINDOWS_SET_FILE_INFORMATION_BY_HANDLE",
            fail_disposition,
        )
        message = "handle deletion failed"
    else:
        original_close = module._WINDOWS_CLOSE_HANDLE
        assert callable(original_close)
        close_count = 0

        def fail_witness_close(handle: int) -> object:
            nonlocal close_count
            result = original_close(handle)
            close_count += 1
            if close_count == 1:
                ctypes.set_last_error(5)
                return 0
            return result

        monkeypatch.setattr(module, "_WINDOWS_CLOSE_HANDLE", fail_witness_close)
        message = "witness handle close failed"

    with pytest.raises(RuntimeError, match=message):
        module._remove_empty_runtime_root(runtime_root)

    assert sentinel.is_dir()
    assert runtime_root.is_dir()


@pytest.mark.skipif(os.name != "nt", reason="Windows post-close contract")
@pytest.mark.parametrize(
    "module",
    [inventory_cli, freeze_cli, evaluate_cli, verify_cli],
)
def test_runtime_cleanup_requires_temp_empty_after_main_handle_close(
    module: _LauncherModule,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = _make_runtime_root(tmp_path, name="runtime")
    temporary = runtime_root / "temp"
    sentinel = temporary / module._GCM_COMMANDLINE_SENTINEL_DIRECTORY
    sentinel.mkdir()
    late = temporary / "late-sibling.txt"
    original_close = module._WINDOWS_CLOSE_HANDLE
    assert callable(original_close)
    close_count = 0

    def inject_after_main_close(handle: int) -> object:
        nonlocal close_count
        result = original_close(handle)
        close_count += 1
        if close_count == 2:
            late.write_text("late", encoding="utf-8")
        return result

    monkeypatch.setattr(module, "_WINDOWS_CLOSE_HANDLE", inject_after_main_close)

    with pytest.raises(RuntimeError, match="temp changed"):
        module._remove_empty_runtime_root(runtime_root)

    assert late.read_text(encoding="utf-8") == "late"
    assert runtime_root.is_dir()


def test_all_runtime_launchers_share_handle_bound_nonrecursive_gcm_cleanup() -> None:
    modules: tuple[_LauncherModule, ...] = (
        inventory_cli,
        freeze_cli,
        evaluate_cli,
        verify_cli,
    )
    state_sources = {
        inspect.getsource(module._windows_directory_handle_state) for module in modules
    }
    assert state_sources and len(state_sources) == 1
    helper_sources = {
        inspect.getsource(module._remove_empty_gcm_commandline_sentinel) for module in modules
    }
    assert helper_sources and len(helper_sources) == 1
    helper_source = next(iter(helper_sources))
    for required in (
        "_WINDOWS_CREATE_FILE_W(",
        "_WINDOWS_GET_FILE_INFORMATION_BY_HANDLE_EX",
        "_WINDOWS_SET_FILE_INFORMATION_BY_HANDLE(",
        "_WindowsFileDispositionInfo(delete_file=True)",
        "_WINDOWS_CLOSE_HANDLE(",
        "expected.lstat()",
    ):
        assert required in helper_source or required in next(iter(state_sources))
    for forbidden in (
        "rmdir(",
        "rmtree",
        ".unlink(",
        "os.remove",
        ".glob(",
        "os.walk",
    ):
        assert forbidden not in helper_source
    for module in modules:
        assert module._GCM_COMMANDLINE_SENTINEL_DIRECTORY == "system-commandline-sentinel-files"
        assert module._WINDOWS_FILE_ATTRIBUTE_TAG_INFO_CLASS == 9
        assert module._WINDOWS_FILE_ID_INFO_CLASS == 18
        assert module._WINDOWS_FILE_DISPOSITION_INFO_CLASS == 4
        assert ctypes.sizeof(module._WindowsFileDispositionInfo) == 1
        assert module._WINDOWS_FILE_SHARE_READ == 1
        assert module._WINDOWS_FILE_SHARE_WRITE == 2
        assert module._WINDOWS_FILE_SHARE_DELETE == 4
        assert module._WINDOWS_OPEN_EXISTING == 3
        assert (
            module._WINDOWS_FILE_FLAG_BACKUP_SEMANTICS
            | module._WINDOWS_FILE_FLAG_OPEN_REPARSE_POINT
        ) == 0x02200000
        assert module._WINDOWS_FILE_ATTRIBUTE_DIRECTORY == 0x10
        assert module._WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT == 0x400
        child_source = inspect.getsource(module._consume_parent_handoff)
        assert "for directory in (cache, temporary, roaming, local)" in child_source
        assert "any(directory.iterdir())" in child_source
        assert "_remove_empty_gcm_commandline_sentinel" not in child_source


@pytest.mark.parametrize(
    "module",
    [inventory_cli, evaluate_cli, verify_cli],
)
def test_runtime_launcher_refuses_nonempty_scratch_after_child(
    module: _LauncherModule,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    cache_parent = root / "artifacts" / "trust_sentinel"
    site_packages = root / ".venv" / "Lib" / "site-packages"
    project_src = root / "src"
    script = root / "scripts" / "entry.py"
    cache_parent.mkdir(parents=True)
    site_packages.mkdir(parents=True)
    project_src.mkdir()
    script.parent.mkdir()
    script.write_text("# bound\n", encoding="utf-8")
    marker = "b" * 64
    runtime_root = cache_parent / f"{module._RUNTIME_ROOT_PREFIX}{marker}"

    monkeypatch.setattr(
        module,
        "_project_layout",
        lambda: (script, root, site_packages, project_src),
    )
    monkeypatch.setattr(module.secrets, "token_hex", lambda _: marker)

    def fake_run(
        arguments: list[str],
        *,
        check: bool,
        cwd: Path,
        env: dict[str, str],
    ) -> SimpleNamespace:
        del arguments, check, cwd, env
        (runtime_root / module._HANDOFF_FILENAME).unlink()
        (runtime_root / "temp" / "unexpected.txt").write_text("unexpected", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="unexpected contents"):
        module._relaunch_isolated(("--help",))

    (runtime_root / "temp" / "unexpected.txt").unlink()
    module._remove_empty_runtime_root(runtime_root)


@pytest.mark.parametrize(
    "module",
    [inventory_cli, evaluate_cli, verify_cli],
)
def test_runtime_launcher_never_retries_failed_post_child_cleanup(
    module: _LauncherModule,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    cache_parent = root / "artifacts" / "trust_sentinel"
    site_packages = root / ".venv" / "Lib" / "site-packages"
    project_src = root / "src"
    script = root / "scripts" / "entry.py"
    cache_parent.mkdir(parents=True)
    site_packages.mkdir(parents=True)
    project_src.mkdir()
    script.parent.mkdir()
    script.write_text("# bound\n", encoding="utf-8")
    marker = "e" * 64
    runtime_root = cache_parent / f"{module._RUNTIME_ROOT_PREFIX}{marker}"

    monkeypatch.setattr(
        module,
        "_project_layout",
        lambda: (script, root, site_packages, project_src),
    )
    monkeypatch.setattr(module.secrets, "token_hex", lambda _: marker)

    def fake_run(
        arguments: list[str],
        *,
        check: bool,
        cwd: Path,
        env: dict[str, str],
    ) -> SimpleNamespace:
        del arguments, check, cwd, env
        (runtime_root / module._HANDOFF_FILENAME).unlink()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    original_cleanup = module._remove_empty_runtime_root
    cleanup_calls = 0

    def fail_once(path: Path) -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        if cleanup_calls == 1:
            raise RuntimeError("one-shot cleanup failure")
        original_cleanup(path)

    monkeypatch.setattr(module, "_remove_empty_runtime_root", fail_once)

    with pytest.raises(RuntimeError, match="one-shot cleanup failure"):
        module._relaunch_isolated(("--help",))

    assert cleanup_calls == 1
    assert runtime_root.is_dir()
    original_cleanup(runtime_root)


@pytest.mark.parametrize(
    "module",
    [inventory_cli, freeze_cli, evaluate_cli, verify_cli],
)
def test_runtime_launcher_rejects_user_supplied_child_marker(
    module: _LauncherModule,
) -> None:
    marker = module._ISOLATED_CHILD_FLAG
    with pytest.raises(RuntimeError, match="cannot be supplied"):
        module._relaunch_isolated((marker,))


@pytest.mark.parametrize(
    "script_name",
    [
        "build_trust_sentinel_ood_v2_inventory.py",
        "freeze_trust_sentinel_ood_external_v2.py",
        "evaluate_trust_sentinel_ood_external_v2.py",
        "verify_trust_sentinel_ood_external_v2.py",
    ],
)
def test_real_launcher_help_reexecutes_isolated_and_leaves_no_runtime_root(
    script_name: str,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    executable = project_root / ".venv" / "Scripts" / "python.exe"
    assert Path(sys.executable).resolve() == executable.resolve()
    runtime_parent = project_root / "artifacts" / "trust_sentinel"
    before = {path.name for path in runtime_parent.glob(".ood_external_v2_1.runtime-*")}

    completed = subprocess.run(
        [
            os.fspath(executable),
            "-I",
            "-S",
            "-B",
            os.fspath(project_root / "scripts" / script_name),
            "--help",
        ],
        check=False,
        capture_output=True,
        cwd=project_root,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
    after = {path.name for path in runtime_parent.glob(".ood_external_v2_1.runtime-*")}
    assert after == before


@pytest.mark.parametrize(
    "script_name",
    [
        "build_trust_sentinel_ood_v2_inventory.py",
        "freeze_trust_sentinel_ood_external_v2.py",
        "evaluate_trust_sentinel_ood_external_v2.py",
        "verify_trust_sentinel_ood_external_v2.py",
    ],
)
def test_real_launcher_rejects_ambient_python_startup(script_name: str) -> None:
    project_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            os.fspath(project_root / "scripts" / script_name),
            "--help",
        ],
        check=False,
        capture_output=True,
        cwd=project_root,
        text=True,
    )

    assert completed.returncode != 0
    assert "isolated launcher requires Python -I -S -B" in completed.stderr


@pytest.mark.parametrize(
    "module",
    [inventory_cli, freeze_cli, evaluate_cli, verify_cli],
)
def test_real_launcher_rejects_forged_runtime_outside_bound_artifacts(
    module: _LauncherModule,
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    script = Path(module.__file__).resolve()
    token = "c" * 64
    runtime_root = _make_runtime_root(
        tmp_path,
        name=f"{module._RUNTIME_ROOT_PREFIX}{token}",
    )
    (runtime_root / module._HANDOFF_FILENAME).write_bytes(f"{token}\n".encode("ascii"))

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-X",
            f"pycache_prefix={runtime_root / 'pycache'}",
            os.fspath(script),
            module._ISOLATED_CHILD_FLAG,
            token,
            "--help",
        ],
        check=False,
        capture_output=True,
        cwd=project_root,
        env=module._sanitized_runtime_environment(runtime_root),
        text=True,
    )

    assert completed.returncode != 0
    assert completed.stderr.strip() == "isolated launcher refused an invalid runtime contract"
    assert os.fspath(runtime_root) not in completed.stderr


def test_successor_cli_defaults_are_atomic_and_have_no_scientific_overrides() -> None:
    evaluate = evaluate_cli._parser().parse_args(
        [
            "--code-revision",
            "a" * 40,
            "--seven-zip-executable",
            "7z",
        ]
    )
    assert evaluate.parent == Path("configs/trust_sentinel_ood_external_v2_1.yaml")
    assert evaluate.child == Path("configs/trust_sentinel_ood_external_v2_1_execution.json")
    assert set(vars(evaluate)) == {
        "child",
        "code_revision",
        "parent",
        "project_root",
        "seven_zip_executable",
    }

    freeze = freeze_cli._parser().parse_args(
        [
            "--implementation-revision",
            "a" * 40,
            "--frozen-at-utc",
            "2026-08-29T00:00:00Z",
            "--challenge-root",
            "challenge",
            "--zzu-root",
            "zzu",
            "--challenge-records",
            "1000",
            "--zzu-records",
            "12328",
            "--zzu-patients",
            "10350",
            "--selected-records-total",
            "13328",
            "--seven-zip-executable",
            "7z",
        ]
    )
    assert freeze.parent == Path("configs/trust_sentinel_ood_external_v2_1.yaml")
    assert freeze.inventory == Path(
        "artifacts/trust_sentinel/ood_external_v2_1_preflight/private/"
        "external-waveform-inventory.json"
    )
    assert freeze.public_projection == Path(
        "artifacts/trust_sentinel/ood_external_v2_1_preflight/public/"
        "external-inventory-summary.json"
    )
    assert freeze.output == Path("configs/trust_sentinel_ood_external_v2_1_execution.json")
    assert freeze.preflight_only is False


def test_child_freeze_preflight_only_is_repeatable_and_canonical(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    def verify(**kwargs: object) -> object:
        observed.update(kwargs)
        return object()

    monkeypatch.setattr(freeze_cli, "verify_child_freeze_preflight", verify)
    monkeypatch.setattr(
        freeze_cli,
        "freeze_external_v2_child_contract",
        lambda **_: pytest.fail("preflight-only invoked production freeze"),
    )

    assert freeze_cli.main(_freeze_arguments(preflight_only=True)) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    assert json.loads(captured.out) == {
        "authorization_state": "NOT_CONSUMED",
        "child_publication_witnessed": False,
        "child_visibility_witnessed": False,
        "cleanup_state": "NOT_REACHED",
        "failure_reason": None,
        "failure_receipt_written": False,
        "official_source_content_accessed": False,
        "output_state": "NONE",
        "retry_authorized": True,
        "stage": "closing_control_state",
        "stage_ordinal": 5,
        "stage_scope": "PREFLIGHT",
        "status": "OOD_EXTERNAL_V2_CHILD_PREFLIGHT_VERIFIED",
    }
    assert observed["frozen_at_utc"] == "2026-08-30T10:00:00Z"
    assert observed["seven_zip_executable"] == Path("7z")


def test_child_freeze_preflight_refusal_discloses_only_allowlisted_stage(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def refuse(**_: object) -> None:
        raise freeze_cli.ChildFreezePreflightStageError(
            "x8_inventory_evidence"
        ) from RuntimeError(r"secret C:\private\patient-123")

    monkeypatch.setattr(freeze_cli, "verify_child_freeze_preflight", refuse)

    assert freeze_cli.main(_freeze_arguments(preflight_only=True)) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert json.loads(captured.err) == {
        "authorization_state": "NOT_CONSUMED",
        "child_publication_witnessed": False,
        "child_visibility_witnessed": False,
        "cleanup_state": "NOT_REACHED",
        "failure_reason": "STAGE_REFUSED",
        "failure_receipt_written": False,
        "official_source_content_accessed": False,
        "output_state": "PRESENT_UNVERIFIABLE",
        "retry_authorized": False,
        "stage": "x8_inventory_evidence",
        "stage_ordinal": 3,
        "stage_scope": "PREFLIGHT",
        "status": "OOD_EXTERNAL_V2_CHILD_PREFLIGHT_REFUSED",
    }
    assert "secret" not in captured.err
    assert "private" not in captured.err
    assert "patient" not in captured.err


def test_child_freeze_cli_requires_exact_stage_order_and_witnesses_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def freeze(**kwargs: object) -> object:
        stage_callback = kwargs["stage_callback"]
        source_witness = kwargs["source_access_witness"]
        visibility_witness = kwargs["child_visibility_witness"]
        publication_witness = kwargs["child_publication_witness"]
        assert callable(stage_callback)
        assert callable(source_witness)
        assert callable(visibility_witness)
        assert callable(publication_witness)
        for stage in freeze_cli.CHILD_FREEZE_ATTEMPT_STAGES:
            stage_callback(stage)
            if stage == "raw_source_binding_verification":
                source_witness()
            if stage == "child_publication":
                visibility_witness()
                publication_witness()
        return object()

    monkeypatch.setattr(freeze_cli, "freeze_external_v2_child_contract", freeze)

    assert freeze_cli.main(_freeze_arguments()) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    assert json.loads(captured.out) == _successful_freeze_report()

    tracker = freeze_cli._ChildFreezeStageTracker()
    with pytest.raises(RuntimeError, match="order changed"):
        tracker.transition(freeze_cli.CHILD_FREEZE_ATTEMPT_STAGES[1])
    tracker.transition(freeze_cli.CHILD_FREEZE_ATTEMPT_STAGES[0])
    with pytest.raises(RuntimeError, match="order changed"):
        tracker.transition(freeze_cli.CHILD_FREEZE_ATTEMPT_STAGES[0])
    with pytest.raises(RuntimeError, match="not allowlisted"):
        tracker.transition("secret-stage")


def test_child_freeze_attempt_failure_preserves_sanitized_publication_truth(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def freeze(**kwargs: object) -> None:
        stage_callback = kwargs["stage_callback"]
        source_witness = kwargs["source_access_witness"]
        visibility_witness = kwargs["child_visibility_witness"]
        assert callable(stage_callback)
        assert callable(source_witness)
        assert callable(visibility_witness)
        for stage in freeze_cli.CHILD_FREEZE_ATTEMPT_STAGES[:12]:
            stage_callback(stage)
            if stage == "raw_source_binding_verification":
                source_witness()
            if stage == "child_publication":
                visibility_witness()
        raise freeze_cli.ChildFreezeAttemptError(
            stage="child_publication",
            reason="PUBLICATION_FAILED_AFTER_VISIBILITY",
            output_state="VISIBLE_EXACT_DURABILITY_UNCONFIRMED",
            official_source_content_accessed=True,
            failure_receipt_written=True,
        ) from RuntimeError(r"secret C:\private\patient-123")

    monkeypatch.setattr(freeze_cli, "freeze_external_v2_child_contract", freeze)

    assert freeze_cli.main(_freeze_arguments()) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert json.loads(captured.err) == {
        "authorization_state": "CONSUMED",
        "child_publication_witnessed": False,
        "child_visibility_witnessed": True,
        "cleanup_state": "NOT_REACHED",
        "failure_reason": "PUBLICATION_FAILED_AFTER_VISIBILITY",
        "failure_receipt_written": True,
        "official_source_content_accessed": True,
        "output_state": "VISIBLE_EXACT_DURABILITY_UNCONFIRMED",
        "retry_authorized": False,
        "stage": "child_publication",
        "stage_ordinal": 11,
        "stage_scope": "ATTEMPT",
        "status": "OOD_EXTERNAL_V2_CHILD_FREEZE_FAILED",
    }
    assert "secret" not in captured.err
    assert "private" not in captured.err
    assert "patient" not in captured.err


def test_child_freeze_untyped_failure_never_infers_authorization_or_leaks(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def freeze(**kwargs: object) -> None:
        stage_callback = kwargs["stage_callback"]
        assert callable(stage_callback)
        stage_callback("authorization_publication")
        raise RuntimeError(r"secret C:\private\patient-123")

    monkeypatch.setattr(freeze_cli, "freeze_external_v2_child_contract", freeze)

    assert freeze_cli.main(_freeze_arguments()) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    report = json.loads(captured.err)
    assert report["authorization_state"] == "UNVERIFIABLE"
    assert report["failure_reason"] == "UNEXPECTED_INTERNAL_FAILURE"
    assert report["output_state"] == "PRESENT_UNVERIFIABLE"
    assert report["retry_authorized"] is False
    assert report["stage"] == "authorization_publication"
    assert report["failure_receipt_written"] is None
    assert "secret" not in captured.err
    assert "private" not in captured.err
    assert "patient" not in captured.err


@pytest.mark.parametrize(
    ("module", "attribute", "arguments", "status"),
    [
        (
            evaluate_cli,
            "prepare_ood_external_v2",
            ["--code-revision", "a" * 40, "--seven-zip-executable", "7z"],
            "OOD_EXTERNAL_V2_EXECUTION_FAILED",
        ),
    ],
)
def test_mutating_clis_fail_closed_without_leaking_exception_or_path(
    module: object,
    attribute: str,
    arguments: list[str],
    status: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(**_: object) -> None:
        raise RuntimeError(r"secret C:\\private\\patient-123")

    monkeypatch.setattr(module, attribute, fail)
    assert module.main(arguments) == 2  # type: ignore[attr-defined]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {"status": status}
    assert "private" not in captured.err
    assert "patient" not in captured.err


def test_verify_cli_reports_successor_preflight_without_private_fields(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeSuccessorPreflight:
        file_sha256 = "sha256:" + "b" * 64
        private_patient_identifier = "must-not-leak"

    monkeypatch.setattr(verify_cli, "SuccessorParentPreflight", FakeSuccessorPreflight)
    monkeypatch.setattr(
        verify_cli,
        "verify_external_v2_metadata",
        lambda **_: FakeSuccessorPreflight(),
    )

    assert verify_cli.main(["--seven-zip-executable", "7z"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "parent_config_file_sha256": "sha256:" + "b" * 64,
        "status": "SUCCESSOR_PARENT_PREFLIGHT_VERIFIED_NOT_EXECUTABLE",
    }
    assert "patient" not in captured.out


def test_verify_cli_binds_terminal_bundle_to_live_project_and_seven_zip(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    def verify(output_root: Path, **kwargs: object) -> SimpleNamespace:
        observed.update(output_root=output_root, **kwargs)
        return SimpleNamespace(result=SimpleNamespace(artifact_sha256="sha256:" + "d" * 64))

    monkeypatch.setattr(verify_cli, "verify_external_v2_bundle", verify)

    assert (
        verify_cli.main(
            [
                "--project-root",
                "project",
                "--output-root",
                "output",
                "--seven-zip-executable",
                "7z.exe",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "result_artifact_sha256": "sha256:" + "d" * 64,
        "status": "OOD_EXTERNAL_V2_BUNDLE_VERIFIED",
    }
    assert observed == {
        "output_root": Path("output"),
        "project_root": Path("project"),
        "seven_zip_executable": Path("7z.exe"),
    }


def test_evaluate_cli_prints_only_aggregate_terminal_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private = SimpleNamespace(patient_key="must-not-leak")
    result = SimpleNamespace(
        artifact_sha256="sha256:" + "c" * 64,
        private=private,
        status=SimpleNamespace(value="EXTERNAL_OOD_TARGET_MISSED"),
    )
    monkeypatch.setattr(evaluate_cli, "prepare_ood_external_v2", lambda **_: result)

    assert evaluate_cli.main(["--code-revision", "a" * 40, "--seven-zip-executable", "7z"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "artifact_sha256": "sha256:" + "c" * 64,
        "status": "EXTERNAL_OOD_TARGET_MISSED",
    }
    assert "patient" not in captured.out
