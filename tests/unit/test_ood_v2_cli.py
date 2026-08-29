from __future__ import annotations

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
    _ISOLATED_CHILD_FLAG: str
    _HANDOFF_FILENAME: str
    _RUNTIME_ROOT_PREFIX: str
    secrets: ModuleType
    subprocess: ModuleType

    def _sanitized_runtime_environment(self, runtime_root: Path) -> dict[str, str]: ...

    def _relaunch_isolated(self, arguments: tuple[str, ...]) -> int: ...

    def _remove_empty_runtime_root(self, runtime_root: Path) -> None: ...


def _make_runtime_root(parent: Path, *, name: str) -> Path:
    runtime_root = parent / name
    (runtime_root / "pycache").mkdir(parents=True)
    (runtime_root / "temp").mkdir()
    (runtime_root / "home" / "AppData" / "Roaming").mkdir(parents=True)
    (runtime_root / "home" / "AppData" / "Local").mkdir()
    return runtime_root


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


@pytest.mark.parametrize(
    "module",
    [inventory_cli, freeze_cli, evaluate_cli, verify_cli],
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

    with pytest.raises(RuntimeError, match="unexpected files"):
        module._relaunch_isolated(("--help",))

    (runtime_root / "temp" / "unexpected.txt").unlink()
    module._remove_empty_runtime_root(runtime_root)


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


@pytest.mark.parametrize(
    ("module", "attribute", "arguments", "status"),
    [
        (
            evaluate_cli,
            "prepare_ood_external_v2",
            ["--code-revision", "a" * 40, "--seven-zip-executable", "7z"],
            "OOD_EXTERNAL_V2_EXECUTION_FAILED",
        ),
        (
            freeze_cli,
            "freeze_external_v2_child_contract",
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
            ],
            "OOD_EXTERNAL_V2_CHILD_FREEZE_FAILED",
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
