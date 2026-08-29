# ruff: noqa: E402 -- the isolated smoke must bootstrap frozen import roots first.
from __future__ import annotations

import importlib.util
import inspect
import json
import multiprocessing
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, cast

# The opt-in Windows worker smoke executes this test module directly under
# ``-I -S -B``.  Match the production launcher's two explicit import roots
# before importing any third-party or project module.  Spawned workers receive
# this exact sys.path from their parent, so the membership guards avoid drift.
_SMOKE_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if sys.flags.no_site:
    for _SMOKE_IMPORT_ROOT in (
        _SMOKE_PROJECT_ROOT / ".venv" / "Lib" / "site-packages",
        _SMOKE_PROJECT_ROOT / "src",
    ):
        _SMOKE_IMPORT_TEXT = os.fspath(_SMOKE_IMPORT_ROOT)
        if _SMOKE_IMPORT_TEXT not in sys.path:
            sys.path.append(_SMOKE_IMPORT_TEXT)

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.utils.data import Dataset, get_worker_info

from ecg_trust.conformal import BinaryDecision
from ecg_trust.constants import LEADS, SUPERCLASSES
from ecg_trust.models.resnet1d import ResNet1D, ResNet1DConfig
from ecg_trust.ood_completion import runtime as completion_runtime
from ecg_trust.ood_v2 import bundle as bundle_module
from ecg_trust.ood_v2 import inventory as inventory_module
from ecg_trust.ood_v2 import pipeline
from ecg_trust.ood_v2.adapters import TARGET_SAMPLES, ExternalECGAdapterError
from ecg_trust.ood_v2.bundle import (
    ACCESS_MARKER_FILENAME,
    FAILURE_RECEIPT_FILENAME,
    canonical_json_bytes,
    canonical_sha256,
    sha256_bytes,
    sha256_file,
)

_DIGEST_A = "sha256:" + "a" * 64
_DIGEST_B = "sha256:" + "b" * 64
_DIGEST_C = "sha256:" + "c" * 64
_SEALED_SOURCE_ASSIGNMENT_SHA256 = (
    "sha256:87992206fcbfc2b091d8f8dd08998a5d9bae3d55a2d2056f1ab674a316b0675b"
)

_ISOLATED_ENTRYPOINTS = (
    "scripts/build_trust_sentinel_ood_v2_inventory.py",
    "scripts/evaluate_trust_sentinel_ood_external_v2.py",
    "scripts/freeze_trust_sentinel_ood_external_v2.py",
    "scripts/verify_trust_sentinel_ood_external_v2.py",
)

_WINDOWS_DATALOADER_SMOKE_FLAG = "--_ecg-trust-windows-dataloader-smoke"
_WINDOWS_DATALOADER_SMOKE_ENV = "ECG_TRUST_RUN_WINDOWS_DATALOADER_SMOKE"
_WORKER_PROBE_FIELDS = 11
_SMOKE_FORWARD_CALLS = 0


class _WindowsWorkerProbeDataset(Dataset[tuple[Tensor, Tensor]]):
    """Synthetic ECGs that retain per-worker runtime observations out of band."""

    def __init__(
        self,
        *,
        records: int,
        observation_root: Path,
        fail_index: int | None = None,
    ) -> None:
        self.records = records
        self.observation_root = observation_root
        self.fail_index = fail_index
        self.worker_calls: dict[int, int] = {}

    def __len__(self) -> int:
        return self.records

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        worker = get_worker_info()
        if worker is None:
            raise RuntimeError("synthetic smoke requires a DataLoader worker")
        worker_id = int(worker.id)
        call_number = self.worker_calls.get(worker_id, 0)
        self.worker_calls[worker_id] = call_number + 1
        observation = {
            "call_number": call_number,
            "cwd": os.fspath(Path.cwd()),
            "environment": {name.upper(): value for name, value in os.environ.items()},
            "flags": {
                "dont_write_bytecode": bool(sys.dont_write_bytecode),
                "isolated": int(sys.flags.isolated),
                "no_site": int(sys.flags.no_site),
                "no_user_site": int(sys.flags.no_user_site),
            },
            "index": index,
            "pid": os.getpid(),
            "pycache_prefix": sys.pycache_prefix,
            "sys_path": list(sys.path),
            "worker_id": worker_id,
        }
        if call_number in {0, self.records // 4} or index == self.fail_index:
            observation_path = self.observation_root / (
                f"worker-{worker_id}-call-{call_number:04d}-index-{index:04d}.json"
            )
            observation_path.write_text(
                json.dumps(observation, ensure_ascii=True, sort_keys=True),
                encoding="utf-8",
            )
        if index == self.fail_index:
            raise RuntimeError("synthetic mid-pass worker failure")

        signal = torch.zeros((len(LEADS), 64), dtype=torch.float32)
        values = (
            float(index),
            float(worker_id),
            float(os.getpid()),
            float(sys.flags.isolated),
            float(sys.flags.no_site),
            float(sys.flags.no_user_site),
            float(bool(sys.dont_write_bytecode)),
            float(Path.cwd() == _SMOKE_PROJECT_ROOT),
            float(os.environ.get("CUDA_CACHE_DISABLE") == "1"),
            float(sys.pycache_prefix is None),
            float(len(sys.path)),
        )
        signal[0, :_WORKER_PROBE_FIELDS] = torch.tensor(values, dtype=torch.float32)
        return signal, torch.tensor(index, dtype=torch.int64)


def _synthetic_probe_embedding(_model: ResNet1D, signals: Tensor) -> Tensor:
    global _SMOKE_FORWARD_CALLS
    _SMOKE_FORWARD_CALLS += 1
    embeddings = torch.zeros(
        (int(signals.shape[0]), 512),
        dtype=torch.float32,
        device=signals.device,
    )
    embeddings[:, :_WORKER_PROBE_FIELDS] = signals[
        :, 0, :_WORKER_PROBE_FIELDS
    ]
    return embeddings


def _git_result(
    arguments: tuple[str, ...],
    stdout: str,
    *,
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(arguments, returncode, stdout, "")


def _load_isolated_entrypoint(relative_path: str) -> Any:
    source = Path(__file__).parents[2] / relative_path
    module_name = f"_protocol_closure_{source.stem}"
    spec = importlib.util.spec_from_file_location(
        module_name,
        source,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {source}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            del sys.modules[module_name]
        else:
            sys.modules[module_name] = previous
    return module


def _runtime_scratch_root(project_root: Path, token: str) -> Path:
    runtime_root = (
        project_root
        / "artifacts"
        / "trust_sentinel"
        / f".ood_external_v2_1.runtime-{token}"
    )
    (runtime_root / "pycache").mkdir(parents=True)
    (runtime_root / "temp").mkdir()
    (runtime_root / "home" / "AppData" / "Roaming").mkdir(parents=True)
    (runtime_root / "home" / "AppData" / "Local").mkdir()
    return runtime_root


def _private_row(
    *,
    quality_status: str = "reacquire",
    route: str = "REACQUIRE",
    canonical_signal_sha256: str = _DIGEST_B,
    quality_report: dict[str, object] | None = None,
) -> pipeline._PrivateRecordEvidence:
    return pipeline._PrivateRecordEvidence(
        dataset="synthetic-dataset",
        record_ref="synthetic-record",
        patient_key=None,
        challenge_quality_label="acceptable",
        adapter_provenance_sha256=_DIGEST_A,
        adapter_source_sample_count=5_000,
        adapter_raw_physical_units=("mV",) * len(LEADS),
        canonical_signal_sha256=canonical_signal_sha256,
        quality_report_sha256=_DIGEST_C,
        quality_report=quality_report,
        quality_status=quality_status,
        quality_reason_codes=() if quality_status == "pass" else ("synthetic_reason",),
        route=route,
        distribution_score=0.0 if quality_status == "pass" else None,
        entropy=0.0 if quality_status == "pass" else None,
        entropy_accepted=True if quality_status == "pass" else None,
        conformal_decisions=("not_supported",) * len(SUPERCLASSES)
        if quality_status == "pass"
        else None,
        all_conformal_decisions_singleton=True if quality_status == "pass" else None,
    )


def _remote_git_runner(
    revision: str,
    *,
    live_stdout: str | None = None,
    backup_tag_type: str = "commit",
    backup_tag_ancestor_returncode: int = 0,
    observed: list[tuple[str, ...]] | None = None,
) -> Any:
    responses = {
        ("remote",): f"{pipeline.EXPECTED_GIT_REMOTE_NAME}\n",
        (
            "remote",
            "get-url",
            "--all",
            pipeline.EXPECTED_GIT_REMOTE_NAME,
        ): f"{pipeline.EXPECTED_GIT_REMOTE_URL}\n",
        (
            "remote",
            "get-url",
            "--push",
            "--all",
            pipeline.EXPECTED_GIT_REMOTE_NAME,
        ): f"{pipeline.EXPECTED_GIT_REMOTE_URL}\n",
        (
            "rev-parse",
            "--verify",
            pipeline.EXPECTED_GIT_REMOTE_MAIN_REF,
        ): f"{revision}\n",
        (
            "ls-remote",
            "--symref",
            pipeline.EXPECTED_GIT_REMOTE_URL,
        ): live_stdout
        if live_stdout is not None
        else (
            "ref: refs/heads/main\tHEAD\n"
            f"{revision}\tHEAD\n"
            f"{revision}\trefs/heads/main\n"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REVISION}\t"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REF}\n"
        ),
        (
            "cat-file",
            "-t",
            pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REVISION,
        ): f"{backup_tag_type}\n",
        (
            "merge-base",
            "--is-ancestor",
            pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REVISION,
            revision,
        ): "",
    }

    def runner(
        _root: Path,
        *arguments: str,
        allow_empty: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        is_ancestor_check = arguments[:2] == ("merge-base", "--is-ancestor")
        assert allow_empty is is_ancestor_check
        if observed is not None:
            observed.append(arguments)
        if arguments not in responses:
            raise AssertionError(f"unexpected Git invocation: {arguments!r}")
        return _git_result(
            arguments,
            responses[arguments],
            returncode=(
                backup_tag_ancestor_returncode if is_ancestor_check else 0
            ),
        )

    return runner


@pytest.mark.parametrize("relative_path", _ISOLATED_ENTRYPOINTS)
def test_isolated_entrypoint_parent_handoff_binds_token_cwd_and_fresh_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relative_path: str,
) -> None:
    module = _load_isolated_entrypoint(relative_path)
    script = tmp_path / Path(relative_path).name
    script.write_bytes(b"# synthetic bound entrypoint\n")
    site_packages = tmp_path / ".venv" / "Lib" / "site-packages"
    project_src = tmp_path / "src"
    site_packages.mkdir(parents=True)
    project_src.mkdir()
    (tmp_path / "artifacts" / "trust_sentinel").mkdir(parents=True)
    token = "a" * 64
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        module,
        "_project_layout",
        lambda: (script, tmp_path, site_packages, project_src),
    )
    monkeypatch.setattr(module.secrets, "token_hex", lambda size: token if size == 32 else "")
    monkeypatch.setattr(
        module,
        "_sanitized_runtime_environment",
        lambda _runtime_root: {"SYNTHETIC_SAFE_ENV": "1"},
    )

    def fake_run(
        command: list[str],
        *,
        check: bool,
        cwd: Path,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[bytes]:
        runtime_root = (
            tmp_path
            / "artifacts"
            / "trust_sentinel"
            / f".ood_external_v2_1.runtime-{token}"
        )
        handoff = runtime_root / ".parent-handoff"
        assert handoff.read_bytes() == f"{token}\n".encode("ascii")
        handoff.unlink()
        observed.update(command=command, check=check, cwd=cwd, env=env)
        return subprocess.CompletedProcess(command, 0, None, None)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module._relaunch_isolated(("--help",)) == 0
    assert observed == {
        "command": [
            module.sys.executable,
            "-I",
            "-S",
            "-B",
            "-X",
            "pycache_prefix="
            + os.fspath(
                tmp_path
                / "artifacts"
                / "trust_sentinel"
                / f".ood_external_v2_1.runtime-{token}"
                / "pycache"
            ),
            os.fspath(script),
            "--_ecg-trust-ood-v2-isolated-child",
            token,
            "--help",
        ],
        "check": False,
        "cwd": tmp_path,
        "env": {"SYNTHETIC_SAFE_ENV": "1"},
    }
    assert not (
        tmp_path
        / "artifacts"
        / "trust_sentinel"
        / f".ood_external_v2_1.runtime-{token}"
    ).exists()


def test_isolated_child_handoff_is_root_bound_and_single_use(tmp_path: Path) -> None:
    module = _load_isolated_entrypoint(_ISOLATED_ENTRYPOINTS[0])
    token = "b" * 64
    runtime_root = _runtime_scratch_root(tmp_path, token)
    module._write_parent_handoff(runtime_root / ".parent-handoff", token=token)

    with pytest.raises(RuntimeError, match="bound to the handoff token"):
        module._consume_parent_handoff(runtime_root, token="c" * 64)
    module._consume_parent_handoff(runtime_root, token=token)
    assert {entry.name for entry in runtime_root.iterdir()} == {
        "home",
        "pycache",
        "temp",
    }
    with pytest.raises(FileNotFoundError):
        module._consume_parent_handoff(runtime_root, token=token)


@pytest.mark.parametrize("relative_path", _ISOLATED_ENTRYPOINTS)
def test_isolated_entrypoint_forged_or_consumed_handoff_has_path_free_public_error(
    tmp_path: Path,
    relative_path: str,
) -> None:
    project_root = Path(__file__).parents[2]
    script = project_root / relative_path
    token = {
        _ISOLATED_ENTRYPOINTS[0]: "c" * 64,
        _ISOLATED_ENTRYPOINTS[1]: "d" * 64,
        _ISOLATED_ENTRYPOINTS[2]: "e" * 64,
        _ISOLATED_ENTRYPOINTS[3]: "f" * 64,
    }[relative_path]
    nonexistent_cache = (
        project_root
        / "artifacts"
        / "trust_sentinel"
        / f".ood_external_v2_1.runtime-{token}"
        / "pycache"
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-X",
            f"pycache_prefix={nonexistent_cache}",
            os.fspath(script),
            "--_ecg-trust-ood-v2-isolated-child",
            token,
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert completed.stderr.strip() == (
        "isolated launcher refused an invalid runtime contract"
    )
    assert os.fspath(project_root) not in completed.stderr
    assert token not in completed.stderr


def test_runtime_environment_binding_is_path_neutral_across_fresh_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first_root = _runtime_scratch_root(tmp_path, "1" * 64)
    second_root = _runtime_scratch_root(tmp_path, "2" * 64)

    def environment_for(runtime_root: Path) -> dict[str, str]:
        home = runtime_root / "home"
        return {
            "APPDATA": os.fspath(home / "AppData" / "Roaming"),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_CACHE_DISABLE": "1",
            "COMSPEC": r"C:\Windows\System32\cmd.exe",
            "LOCALAPPDATA": os.fspath(home / "AppData" / "Local"),
            "NUMBER_OF_PROCESSORS": "16",
            "OS": "Windows_NT",
            "PATH": r"C:\Windows\System32",
            "PATHEXT": ".COM;.EXE;.BAT;.CMD",
            "PROCESSOR_ARCHITECTURE": "AMD64",
            "PROGRAMDATA": r"C:\ProgramData",
            "PROGRAMFILES": r"C:\Program Files",
            "PROGRAMFILES(X86)": r"C:\Program Files (x86)",
            "PROGRAMW6432": r"C:\Program Files",
            "SYSTEMDRIVE": "C:",
            "SYSTEMROOT": r"C:\Windows",
            "TEMP": os.fspath(runtime_root / "temp"),
            "TMP": os.fspath(runtime_root / "temp"),
            "TORCHINDUCTOR_CACHE_DIR": os.fspath(runtime_root / "temp"),
            "USERPROFILE": os.fspath(home),
            "WINDIR": r"C:\Windows",
        }

    environment = environment_for(first_root)
    monkeypatch.setattr(os, "environ", environment)
    first = pipeline._frozen_runtime_environment_material(
        first_root,
        project_root=tmp_path,
    )
    environment.clear()
    environment.update(environment_for(second_root))
    second = pipeline._frozen_runtime_environment_material(
        second_root,
        project_root=tmp_path,
    )

    assert first == second
    assert dict(first)["APPDATA"] == "<runtime_roaming>"
    assert dict(first)["TEMP"] == "<runtime_temp>"
    assert dict(first)["TORCHINDUCTOR_CACHE_DIR"] == "<runtime_temp>"
    assert os.fspath(first_root) not in repr(first)
    assert os.fspath(second_root) not in repr(second)


@pytest.mark.parametrize(
    ("relative_path", "kind"),
    (
        ("linked/nested/synthetic.json", "file"),
        ("linked/nested", "directory"),
        ("linked/new-output.json", "write"),
    ),
)
def test_project_relative_resolution_rejects_an_indirect_intermediate_component(
    tmp_path: Path,
    relative_path: str,
    kind: str,
) -> None:
    project_root = tmp_path / "project"
    actual = project_root / "actual"
    nested = actual / "nested"
    nested.mkdir(parents=True)
    (nested / "synthetic.json").write_bytes(b"{}\n")
    linked = project_root / "linked"
    try:
        linked.symlink_to(actual, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")

    keyword_arguments = {
        "require_file": kind == "file",
        "require_directory": kind == "directory",
    }
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="indirect filesystem component",
    ):
        pipeline._resolve_project_relative(
            project_root,
            relative_path,
            **keyword_arguments,
        )


def test_bundle_tree_snapshot_rejects_an_indirect_parent_component(
    tmp_path: Path,
) -> None:
    actual_parent = tmp_path / "actual-parent"
    output_root = actual_parent / "synthetic-output"
    output_root.mkdir(parents=True)
    for relative_path in bundle_module.BUNDLE_MEMBER_PATHS:
        member = output_root.joinpath(*Path(relative_path).parts)
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_bytes(b"synthetic manifest-covered member\n")
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(actual_parent, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")

    with pytest.raises(
        bundle_module.ExternalV2BundleError,
        match="indirect|link|junction",
    ):
        bundle_module._exact_tree_snapshot(
            linked_parent / "synthetic-output",
            include_success_manifest=False,
        )


def test_seven_zip_runner_excludes_sibling_plugins_and_unbound_dll_search_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "installed-seven-zip"
    source_root.mkdir()
    executable = source_root / "7z.exe"
    executable.write_bytes(b"synthetic-seven-zip-executable")
    (source_root / "7z.dll").write_bytes(b"synthetic-bound-seven-zip-library")
    (source_root / "adversarial.dll").write_bytes(b"must-not-be-discoverable")
    codecs = source_root / "Codecs"
    codecs.mkdir()
    (codecs / "adversarial-codec.dll").write_bytes(b"must-not-be-copied")
    observed: dict[str, object] = {}

    class SyntheticProcess:
        returncode = 0

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

        def kill(self) -> None:
            raise AssertionError("a successful synthetic process must not be killed")

    def fake_popen(
        command: list[str],
        *,
        stdout: Any,
        stderr: Any,
        cwd: Path,
        env: dict[str, str],
        close_fds: bool,
        shell: bool,
    ) -> SyntheticProcess:
        del stderr
        replica_executable = Path(command[0])
        tool_root = replica_executable.parent
        assert replica_executable != executable
        assert {entry.name for entry in tool_root.iterdir()} == {"7z.exe", "7z.dll"}
        assert (tool_root / "7z.exe").read_bytes() == executable.read_bytes()
        assert (tool_root / "7z.dll").read_bytes() == (
            source_root / "7z.dll"
        ).read_bytes()
        assert cwd.parent == tool_root.parent
        assert cwd != tool_root
        assert tuple(cwd.iterdir()) == ()
        assert set(env) == {
            "COMSPEC",
            "PATHEXT",
            "PATH",
            "SystemDrive",
            "SystemRoot",
            "TEMP",
            "TMP",
            "WINDIR",
        }
        search_paths = tuple(env["PATH"].split(os.pathsep))
        assert search_paths[0] == os.fspath(tool_root)
        assert os.fspath(source_root) not in search_paths
        assert env["TEMP"] == env["TMP"] == os.fspath(cwd)
        assert close_fds is True
        assert shell is False
        stdout.write(b"synthetic isolated output\n")
        stdout.flush()
        observed.update(command=command, cwd=cwd, env=env)
        return SyntheticProcess()

    monkeypatch.setattr(vars(inventory_module)["subprocess"], "Popen", fake_popen)

    assert inventory_module._run_seven_zip(executable, ("i", "-sccUTF-8")) == (
        "synthetic isolated output\n"
    )
    assert observed


def test_terminal_transaction_rechecks_runtime_scratch_after_bundle_reread() -> None:
    source = inspect.getsource(pipeline.prepare_ood_external_v2)
    ownership_definition_offset = source.index("def verify_terminal_ownership")
    terminal_write_offset = source.index(
        "_atomic_write_terminal_success",
        ownership_definition_offset,
    )
    ownership_definition = source[ownership_definition_offset:terminal_write_offset]
    assert "_verify_runtime_scratch_empty(inputs.project_root)" in ownership_definition

    reread_offset = source.index("reloaded = verify_external_v2_bundle")
    final_ownership_offset = source.index(
        "verify_terminal_ownership()",
        reread_offset,
    )
    return_offset = source.index("return reloaded.result", reread_offset)

    assert reread_offset < final_ownership_offset < return_offset


def test_postclaim_failure_receipt_does_not_require_clean_runtime_scratch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "owned-output"
    output_root.mkdir()
    marker_bytes = b'{"synthetic":"owned-marker"}\n'
    (output_root / ACCESS_MARKER_FILENAME).write_bytes(marker_bytes)
    identity = pipeline._owned_directory_identity(output_root)
    inputs = cast(
        Any,
        SimpleNamespace(
            child=SimpleNamespace(file_sha256=_DIGEST_A),
            inventory=SimpleNamespace(inventory_sha256=_DIGEST_B),
            parent=SimpleNamespace(file_sha256=_DIGEST_C),
            project_root=tmp_path,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_verify_runtime_scratch_empty",
        lambda _root: pytest.fail(
            "dirty runtime scratch must not prevent retention of its failure receipt"
        ),
    )

    pipeline._retain_postclaim_failure(
        staging=tmp_path / "unused-staging",
        output_root=output_root,
        inputs=inputs,
        code_revision="1" * 40,
        error=pipeline.OODExternalV2IntegrityError("synthetic scratch violation"),
        output_root_owned=True,
        terminal_manifest_visible=True,
        external_claim_file_sha256=_DIGEST_A,
        owner_nonce="1" * 64,
        expected_directory_identity=identity,
        expected_marker_bytes=marker_bytes,
        expected_parent_identity=pipeline._owned_directory_identity(output_root.parent),
    )

    receipt = json.loads(
        (output_root / FAILURE_RECEIPT_FILENAME).read_text(encoding="ascii")
    )
    assert receipt["status"] == "FAILED"
    assert receipt["terminal_state"] == "AMBIGUOUS_TERMINAL_COMMIT"


def test_output_commit_rejects_persistently_substituted_namespace_parent(
    tmp_path: Path,
) -> None:
    namespace = tmp_path / "namespace"
    staging = namespace / ".synthetic.staging-owned"
    staging.mkdir(parents=True)
    marker_bytes = b'{"synthetic":"owned-marker"}\n'
    (staging / ACCESS_MARKER_FILENAME).write_bytes(marker_bytes)
    namespace_identity = pipeline._owned_directory_identity(namespace)
    staging_identity = pipeline._owned_directory_identity(staging)
    moved_namespace = tmp_path / "moved-namespace"
    namespace.rename(moved_namespace)
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    try:
        namespace.symlink_to(attacker, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")

    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="indirect filesystem component",
    ):
        pipeline._commit_staged_directory(
            namespace / staging.name,
            namespace / "synthetic-output",
            expected_directory_identity=staging_identity,
            expected_marker_bytes=marker_bytes,
            expected_parent_identity=namespace_identity,
        )

    assert not (attacker / "synthetic-output").exists()
    assert (moved_namespace / staging.name / ACCESS_MARKER_FILENAME).read_bytes() == (
        marker_bytes
    )


def test_terminal_semantic_verifier_requires_and_routes_live_source_closure(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="requires the live project and 7-Zip tool",
    ):
        pipeline.verify_private_external_v2_bundle_semantics(
            tmp_path,
            result=cast(Any, None),
            inventory=cast(Any, None),
            parent_config_file_sha256=_DIGEST_A,
            child_contract_file_sha256=_DIGEST_B,
            project_root=None,
            seven_zip_executable=None,
        )

    semantic_source = inspect.getsource(
        pipeline.verify_private_external_v2_bundle_semantics
    )
    live_inputs_offset = semantic_source.index(
        "live_inputs = verify_external_v2_inputs"
    )
    row_load_offset = semantic_source.index("rows = _load_private_record_evidence")
    raw_replay_offset = semantic_source.index("_verify_raw_to_canonical_replay")
    assert live_inputs_offset < row_load_offset < raw_replay_offset

    bundle_source = inspect.getsource(bundle_module._verify_external_v2_bundle_members)
    semantic_call_offset = bundle_source.index(
        "verify_private_external_v2_bundle_semantics("
    )
    semantic_call = bundle_source[semantic_call_offset:]
    assert "project_root=project_root" in semantic_call
    assert "seven_zip_executable=seven_zip_executable" in semantic_call


def test_live_remote_query_requires_exact_main_and_backup_tag_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    revision = "1" * 40
    observed: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        pipeline,
        "_run_git",
        _remote_git_runner(revision, observed=observed),
    )

    pipeline._verify_git_remote_state(tmp_path, expected_revision=revision)

    assert observed == [
        ("remote",),
        ("remote", "get-url", "--all", pipeline.EXPECTED_GIT_REMOTE_NAME),
        (
            "remote",
            "get-url",
            "--push",
            "--all",
            pipeline.EXPECTED_GIT_REMOTE_NAME,
        ),
        ("rev-parse", "--verify", pipeline.EXPECTED_GIT_REMOTE_MAIN_REF),
        (
            "ls-remote",
            "--symref",
            pipeline.EXPECTED_GIT_REMOTE_URL,
        ),
        (
            "cat-file",
            "-t",
            pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REVISION,
        ),
        (
            "merge-base",
            "--is-ancestor",
            pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REVISION,
            revision,
        ),
    ]


@pytest.mark.parametrize(
    ("backup_tag_type", "ancestor_returncode"),
    (("tag", 0), ("commit", 1)),
)
def test_live_remote_query_rejects_noncommit_or_nonancestor_backup_tag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backup_tag_type: str,
    ancestor_returncode: int,
) -> None:
    revision = "1" * 40
    monkeypatch.setattr(
        pipeline,
        "_run_git",
        _remote_git_runner(
            revision,
            backup_tag_type=backup_tag_type,
            backup_tag_ancestor_returncode=ancestor_returncode,
        ),
    )

    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="exact pushed frozen revision",
    ):
        pipeline._verify_git_remote_state(tmp_path, expected_revision=revision)


def test_clean_git_revision_rejects_skip_worktree_and_assume_unchanged_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    revision = "7" * 40
    observed: list[tuple[str, ...]] = []

    def clean_runner(
        _root: Path,
        *arguments: str,
        allow_empty: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        assert allow_empty is False
        observed.append(arguments)
        responses = {
            ("ls-files", "-t"): "H tracked.py\n",
            ("ls-files", "-v"): "H tracked.py\n",
            ("status", "--porcelain=v1", "--untracked-files=all"): "",
            ("rev-parse", "HEAD"): revision + "\n",
        }
        return _git_result(arguments, responses[arguments])

    monkeypatch.setattr(pipeline, "_run_git", clean_runner)
    assert pipeline._verify_clean_git_revision(tmp_path) == revision
    assert observed == [
        ("ls-files", "-t"),
        ("ls-files", "-v"),
        ("status", "--porcelain=v1", "--untracked-files=all"),
        ("rev-parse", "HEAD"),
    ]

    for target_arguments, forged_line in (
        (("ls-files", "-t"), "S tracked.py\n"),
        (("ls-files", "-v"), "h tracked.py\n"),
    ):
        def forged_runner(
            _root: Path,
            *arguments: str,
            allow_empty: bool = False,
            _target_arguments: tuple[str, ...] = target_arguments,
            _forged_line: str = forged_line,
        ) -> subprocess.CompletedProcess[str]:
            assert allow_empty is False
            stdout = (
                _forged_line if arguments == _target_arguments else "H tracked.py\n"
            )
            return _git_result(arguments, stdout)

        monkeypatch.setattr(pipeline, "_run_git", forged_runner)
        with pytest.raises(
            pipeline.OODExternalV2IntegrityError,
            match="skip-worktree or assume-unchanged",
        ):
            pipeline._verify_clean_git_revision(tmp_path)


def test_execution_revision_rejects_merge_even_when_x_to_y_count_is_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    implementation_revision = "7" * 40
    execution_revision = "8" * 40
    extra_parent = "6" * 40
    child = cast(
        Any,
        SimpleNamespace(implementation_revision=implementation_revision),
    )
    monkeypatch.setattr(
        pipeline,
        "_verify_clean_git_revision",
        lambda _root: execution_revision,
    )
    monkeypatch.setattr(
        pipeline,
        "_verify_successor_amendment_revision",
        lambda *_args, **_kwargs: None,
    )

    def git_runner(
        _root: Path,
        *arguments: str,
        allow_empty: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        if arguments[:2] == ("merge-base", "--is-ancestor"):
            assert allow_empty is True
            return _git_result(arguments, "")
        assert allow_empty is False
        if arguments[:2] == ("rev-list", "--count"):
            return _git_result(arguments, "1\n")
        if arguments[:2] == ("rev-list", "--parents"):
            return _git_result(
                arguments,
                f"{execution_revision} {implementation_revision} {extra_parent}\n",
            )
        raise AssertionError(f"unexpected Git invocation: {arguments!r}")

    monkeypatch.setattr(pipeline, "_run_git", git_runner)
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="exactly one child-freeze commit",
    ):
        pipeline._verify_revision_boundary(
            tmp_path,
            child=child,
            execution_revision=execution_revision,
        )


@pytest.mark.parametrize(
    "live_stdout",
    [
        (
            "ref: refs/heads/main\tHEAD\n"
            f"{'1' * 40}\tHEAD\n"
            f"{'1' * 40}\trefs/heads/main\n"
        ),
        (
            "ref: refs/heads/main\tHEAD\n"
            f"{'2' * 40}\tHEAD\n"
            f"{'2' * 40}\trefs/heads/main\n"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REVISION}\t"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REF}\n"
        ),
        (
            "ref: refs/heads/main\tHEAD\n"
            f"{'1' * 40} HEAD\n"
            f"{'1' * 40}\trefs/heads/main\n"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REVISION}\t"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REF}\n"
        ),
        (
            "ref: refs/heads/main\tHEAD\n"
            f"{'1' * 40}\tHEAD\n"
            f"{'1' * 40}\trefs/heads/main\n"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REVISION}\t"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REF}\n"
            f"{'1' * 40}\trefs/tags/main\n"
        ),
        (
            "ref: refs/heads/not-main\tHEAD\n"
            f"{'1' * 40}\tHEAD\n"
            f"{'1' * 40}\trefs/heads/not-main\n"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REVISION}\t"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REF}\n"
        ),
        (
            "ref: refs/heads/main\tHEAD\n"
            f"{'1' * 40}\tHEAD\n"
            f"{'1' * 40}\trefs/heads/main\n"
            f"{'3' * 40}\t{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REF}\n"
        ),
        (
            "ref: refs/heads/main\tHEAD\r\n"
            f"{'1' * 40}\tHEAD\r\n"
            f"{'1' * 40}\trefs/heads/main\r\n"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REVISION}\t"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REF}\r\n"
        ),
        (
            "ref: refs/heads/main\tHEAD\n"
            f"{'1' * 40}\tHEAD\n"
            f"{'1' * 40}\trefs/heads/main\n\n"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REVISION}\t"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REF}\n"
        ),
        (
            "ref: refs/heads/main\tHEAD\n"
            f"{'1' * 40}\tHEAD\n"
            f"{'1' * 40}\trefs/heads/main\n"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REVISION}\t"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REF}"
        ),
        (
            "ref: refs/heads/main\tHEAD\n"
            f"{'1' * 40}\tHEAD\n"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REVISION}\t"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REF}\n"
            f"{'1' * 40}\trefs/heads/main\n"
        ),
        (
            "ref: refs/heads/main\tHEAD\n"
            f"{'1' * 40}\tHEAD\n"
            f"{'1' * 40}\trefs/heads/main\n"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REVISION}\t"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REF}\n"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REVISION}\t"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REF}\n"
        ),
        (
            "ref: refs/heads/main\tHEAD\n"
            f"{'1' * 40}\tHEAD\n"
            f"{'1' * 40}\trefs/heads/main\n"
            f"{'4' * 40}\t{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REF}\n"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REVISION}\t"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REF}^{{}}\n"
        ),
        (
            "ref: refs/heads/main\tHEAD\n"
            f"{'1' * 40}\tHEAD\n"
            f"{'1' * 40}\trefs/heads/main\n"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REVISION}\t"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REF}\n"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REVISION}\t"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REF}^{{}}\n"
        ),
    ],
)
def test_live_remote_query_rejects_forged_or_ambiguous_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    live_stdout: str,
) -> None:
    revision = "1" * 40
    monkeypatch.setattr(
        pipeline,
        "_run_git",
        _remote_git_runner(revision, live_stdout=live_stdout),
    )

    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="exact pushed frozen revision",
    ):
        pipeline._verify_git_remote_state(tmp_path, expected_revision=revision)


def test_private_history_query_covers_every_forbidden_path_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[tuple[str, ...]] = []
    assert pipeline.FORBIDDEN_GIT_HISTORY_PATHS == (
        ":(glob)data/raw/external-ood/**",
        ":(glob)artifacts/trust_sentinel/ood_external_v2_preflight/private/**",
        ":(glob)artifacts/trust_sentinel/ood_external_v2_1_preflight/private/**",
        pipeline.PREDECESSOR_OUTPUT_PATH,
        pipeline.SUCCESSOR_OUTPUT_PATH,
        pipeline.PREDECESSOR_CLAIM_PATH,
        pipeline.SUCCESSOR_CLAIM_PATH,
        ":(glob)artifacts/trust_sentinel/.ood_external_v2.staging-*/**",
        ":(glob)artifacts/trust_sentinel/.ood_external_v2_1.staging-*/**",
        ":(glob)artifacts/trust_sentinel/.ood_external_v2_1.runtime-*/**",
    )

    def empty_history(
        _root: Path,
        *arguments: str,
        allow_empty: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        assert allow_empty is False
        observed.append(arguments)
        return _git_result(arguments, "")

    monkeypatch.setattr(pipeline, "_run_git", empty_history)
    pipeline._verify_private_history_absent(tmp_path)
    assert observed == [
        (
            "log",
            "--full-history",
            "--all",
            "--reflog",
            "--format=%H",
            "--",
            *pipeline.FORBIDDEN_GIT_HISTORY_PATHS,
        )
    ]

    def nonempty_history(
        _root: Path,
        *arguments: str,
        allow_empty: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        assert allow_empty is False
        return _git_result(arguments, "f" * 40 + "\n")

    monkeypatch.setattr(pipeline, "_run_git", nonempty_history)
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="appear in Git history",
    ):
        pipeline._verify_private_history_absent(tmp_path)


def test_bound_git_environment_discards_inherited_git_controls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "mingw64" / "bin" / "git.exe"
    system_root = tmp_path / "synthetic-windows"
    inherited_controls = (
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CONFIG_KEY_0",
        "GIT_CONFIG_SYSTEM",
        "GIT_CONFIG_VALUE_0",
        "GIT_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_PROXY_COMMAND",
        "GIT_REPLACE_REF_BASE",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "GIT_WORK_TREE",
    )
    for name in inherited_controls:
        monkeypatch.setenv(name, "synthetic-attacker-control")
    monkeypatch.setenv("PATH", os.fspath(tmp_path / "synthetic-attacker-bin"))
    monkeypatch.setenv("SYSTEMROOT", os.fspath(system_root))
    monkeypatch.setenv("WINDIR", os.fspath(system_root))
    monkeypatch.setenv("COMSPEC", os.fspath(system_root / "System32" / "cmd.exe"))
    monkeypatch.setenv("TEMP", os.fspath(tmp_path / "temp"))
    monkeypatch.setenv("TMP", os.fspath(tmp_path / "tmp"))

    environment = pipeline._sanitized_git_environment(executable)

    assert not set(inherited_controls).intersection(environment)
    assert environment["GIT_CONFIG_COUNT"] == "0"
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_OPTIONAL_LOCKS"] == "0"
    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    assert environment["PATH"] == os.pathsep.join(
        (
            os.fspath(executable.parent),
            os.fspath(executable.parent.parent / "libexec" / "git-core"),
            os.fspath(system_root / "System32"),
        )
    )


def test_bound_git_execution_uses_real_executable_and_repository_controls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launcher = tmp_path / "cmd" / "git.exe"
    executable = tmp_path / "mingw64" / "bin" / "git.exe"
    install_root = tmp_path
    controls_checked: list[Path] = []
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        pipeline,
        "_git_executable_paths",
        lambda: (launcher, executable, install_root),
    )
    monkeypatch.setattr(
        pipeline,
        "_verify_git_runtime_tree_before_provenance",
        lambda: None,
    )
    monkeypatch.setattr(
        pipeline,
        "_verify_git_repository_controls",
        lambda root: controls_checked.append(root),
    )
    monkeypatch.setattr(
        pipeline,
        "_sanitized_git_environment",
        lambda value: {"PATH": os.fspath(value.parent)},
    )

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        check: bool,
        capture_output: bool,
        env: dict[str, str],
        timeout: int,
    ) -> subprocess.CompletedProcess[bytes]:
        observed.update(
            command=command,
            cwd=cwd,
            check=check,
            capture_output=capture_output,
            env=env,
            timeout=timeout,
        )
        return subprocess.CompletedProcess(command, 0, b"synthetic\n", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    completed = pipeline._execute_bound_git(tmp_path, "status", "--porcelain=v1")

    assert completed.stdout == b"synthetic\n"
    assert controls_checked == [tmp_path]
    assert observed["command"] == [
        os.fspath(executable),
        "--no-pager",
        "--no-replace-objects",
        f"--git-dir={tmp_path / '.git'}",
        f"--work-tree={tmp_path}",
        "-c",
        "core.hooksPath=NUL",
        "-c",
        "protocol.file.allow=never",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-c",
        "core.ignoreStat=false",
        "-c",
        "core.preloadIndex=false",
        "-c",
        "core.checkStat=default",
        "-c",
        "core.trustctime=true",
        "-c",
        "core.sparseCheckout=false",
        "-c",
        "core.sparseCheckoutCone=false",
        "-c",
        "extensions.worktreeConfig=false",
        "status",
        "--porcelain=v1",
    ]
    assert observed["cwd"] == tmp_path
    assert observed["env"] == {"PATH": os.fspath(executable.parent)}


def test_tracked_head_blob_requires_index_membership_hash_and_worktree_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    revision = "3" * 40
    relative_path = "configs/synthetic-parent.yaml"
    payload = b"protocol: synthetic\n"
    working = tmp_path / relative_path
    working.parent.mkdir(parents=True)
    working.write_bytes(payload)
    observed: list[tuple[tuple[str, ...], bool]] = []
    observed_blobs: list[tuple[str, ...]] = []

    def exact_head(
        _root: Path,
        *arguments: str,
        allow_empty: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        observed.append((arguments, allow_empty))
        if arguments[:2] == ("ls-files", "--"):
            return _git_result(arguments, relative_path + "\n")
        raise AssertionError(f"unexpected Git invocation: {arguments!r}")

    monkeypatch.setattr(pipeline, "_run_git", exact_head)
    def exact_blob(_root: Path, *arguments: str) -> bytes:
        observed_blobs.append(arguments)
        return payload

    monkeypatch.setattr(pipeline, "_run_git_bytes", exact_blob)
    pipeline._verify_tracked_head_blob(
        tmp_path,
        revision=revision,
        relative_path=relative_path,
        expected_file_sha256=sha256_bytes(payload),
    )
    assert observed == [
        (("ls-files", "--", relative_path), True),
    ]
    assert observed_blobs == [("show", f"{revision}:{relative_path}")]

    working.write_bytes(b"protocol: worktree-tamper\n")
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="working bytes differ from the exact Git HEAD blob",
    ):
        pipeline._verify_tracked_head_blob(
            tmp_path,
            revision=revision,
            relative_path=relative_path,
            expected_file_sha256=sha256_bytes(payload),
        )


def test_revision_bound_file_requires_exact_worktree_and_execution_blob(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    revision = "6" * 40
    relative_path = "configs/synthetic-child.json"
    payload = b'{"synthetic":"child"}\n'
    source = tmp_path / relative_path
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    observed: list[tuple[str, ...]] = []

    def exact_blob(_root: Path, *arguments: str, maximum_bytes: int = 8_000_000) -> bytes:
        assert maximum_bytes == 8_000_000
        observed.append(arguments)
        return payload

    monkeypatch.setattr(pipeline, "_run_git_bytes", exact_blob)
    pipeline._verify_revision_bound_file(
        tmp_path,
        revision=revision,
        relative_path=relative_path,
        expected_file_sha256=sha256_bytes(payload),
        context="synthetic child",
    )
    assert observed == [("show", f"{revision}:{relative_path}")]

    monkeypatch.setattr(
        pipeline,
        "_run_git_bytes",
        lambda *_args, **_kwargs: b'{"synthetic":"forged-y"}\n',
    )
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="differs from its exact execution Git blob",
    ):
        pipeline._verify_revision_bound_file(
            tmp_path,
            revision=revision,
            relative_path=relative_path,
            expected_file_sha256=sha256_bytes(payload),
            context="synthetic child",
        )


def test_project_source_tree_requires_exact_implementation_and_execution_blobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_package = tmp_path / "src" / "ecg_trust"
    source_package.mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (source_package / "__init__.py").write_bytes(b'"""synthetic package."""\n')
    (source_package / "runtime.py").write_bytes(b"VALUE = 1\n")
    for relative_entrypoint in pipeline.PROJECT_OPERATIONAL_ENTRYPOINTS:
        tmp_path.joinpath(*Path(relative_entrypoint).parts).write_bytes(
            b"raise SystemExit(0)\n"
        )
    binding = pipeline._build_project_source_tree(tmp_path)
    tracked_paths = tuple(item.relative_path for item in binding.files)
    implementation_revision = "4" * 40
    execution_revision = "5" * 40

    def tracked_git(
        _root: Path,
        *arguments: str,
        allow_empty: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        assert allow_empty is False
        assert arguments == (
            "ls-files",
            "--",
            pipeline.PROJECT_SOURCE_ROOT,
            *pipeline.PROJECT_OPERATIONAL_ENTRYPOINTS,
        )
        return _git_result(arguments, "".join(f"{path}\n" for path in tracked_paths))

    tampered_revision: str | None = None

    def git_blob(
        _root: Path,
        *arguments: str,
        maximum_bytes: int = 8_000_000,
    ) -> bytes:
        assert maximum_bytes == 8_000_000
        assert len(arguments) == 2 and arguments[0] == "show"
        revision, relative_path = arguments[1].split(":", maxsplit=1)
        assert revision in {implementation_revision, execution_revision}
        if revision == tampered_revision and relative_path.endswith("runtime.py"):
            return b"VALUE = 2\n"
        return tmp_path.joinpath(*Path(relative_path).parts).read_bytes()

    monkeypatch.setattr(pipeline, "_run_git", tracked_git)
    monkeypatch.setattr(pipeline, "_run_git_bytes", git_blob)
    pipeline._verify_project_source_tree_at_revisions(
        tmp_path,
        binding,
        implementation_revision=implementation_revision,
        execution_revision=execution_revision,
    )

    tampered_revision = execution_revision
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="exact frozen Git blob",
    ):
        pipeline._verify_project_source_tree_at_revisions(
            tmp_path,
            binding,
            implementation_revision=implementation_revision,
            execution_revision=execution_revision,
        )


def test_normalization_copy_is_byte_exact_and_bound_by_routing_hash(
    tmp_path: Path,
) -> None:
    policy_bytes = b'{"policy":"synthetic"}\n'
    checkpoint_bytes = b"synthetic-checkpoint"
    resolved_bytes = b'{"resolved":"synthetic"}\n'
    normalization_bytes = b'{"normalization":"synthetic"}\n'
    policy_path = tmp_path / "policy.json"
    checkpoint_path = tmp_path / "model.ckpt"
    resolved_path = tmp_path / "resolved.json"
    normalization_path = tmp_path / "normalization.json"
    policy_path.write_bytes(policy_bytes)
    checkpoint_path.write_bytes(checkpoint_bytes)
    resolved_path.write_bytes(resolved_bytes)
    normalization_path.write_bytes(normalization_bytes)
    private = tmp_path / "private"
    private.mkdir()

    inputs = cast(
        Any,
        SimpleNamespace(
            project_root=tmp_path,
            parent=SimpleNamespace(
                v1_distribution_policy=SimpleNamespace(
                    relative_path="policy.json",
                    file_sha256=sha256_bytes(policy_bytes),
                ),
                challenge_bootstrap_seed=11,
                confidence_level=0.95,
                bootstrap_resamples=1_000,
                zzu_bootstrap_seed=12,
                threshold=1.5,
                resolved_config_sha256=_DIGEST_A,
            ),
            checkpoint_path=checkpoint_path,
            resolved_config_path=resolved_path,
            normalization_path=normalization_path,
            routing=SimpleNamespace(
                conformal=SimpleNamespace(to_dict=lambda: {"synthetic": True}),
                demo_policy_file_sha256=_DIGEST_A,
                maximum_entropy=0.75,
                source_calibration_file_sha256=_DIGEST_B,
                source_calibration_result=SimpleNamespace(artifact_sha256=_DIGEST_C),
                temperature=1.0,
            ),
            inventory=SimpleNamespace(inventory_sha256=_DIGEST_B),
            v1=SimpleNamespace(policy=SimpleNamespace(artifact_sha256=_DIGEST_C)),
        ),
    )
    evaluated = cast(Any, SimpleNamespace(model_state_before_sha256=_DIGEST_A))

    pipeline._write_private_routing_contract(
        private,
        inputs=inputs,
        evaluated=evaluated,
    )

    copied = private / "frozen-normalization.json"
    routing = cast(
        dict[str, object],
        json.loads((private / "routing-contract.json").read_text(encoding="ascii")),
    )
    assert copied.read_bytes() == normalization_bytes
    normalization_hash = sha256_bytes(normalization_bytes)
    assert routing["normalization_file_sha256"] == normalization_hash
    assert routing["normalization_file_sha256"] == sha256_file(copied)

    copied.write_bytes(b"tampered-normalization")
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="normalization hash differs from routing contract",
    ):
        pipeline._load_private_normalization(
            copied,
            expected_file_sha256=normalization_hash,
        )


def _raw_replay_inputs() -> Any:
    record = SimpleNamespace(dataset="synthetic-dataset")
    return cast(
        Any,
        SimpleNamespace(
            inventory=SimpleNamespace(records=(record,)),
            dataset_roots={"synthetic-dataset": Path("synthetic-root")},
        ),
    )


def _mock_raw_replay_storage(
    monkeypatch: pytest.MonkeyPatch,
    stored_signal: np.ndarray[Any, np.dtype[np.float32]],
) -> None:
    monkeypatch.setattr(
        pipeline,
        "_verify_canonical_signal_sidecar",
        lambda *_args, **_kwargs: (object(),),
    )
    monkeypatch.setattr(
        pipeline,
        "_load_canonical_signal_shard",
        lambda *_args, **_kwargs: {0: stored_signal},
    )
    monkeypatch.setattr(
        pipeline,
        "resolve_inventory_record_base",
        lambda *_args, **_kwargs: Path("synthetic-record-base"),
    )
    monkeypatch.setattr(
        pipeline,
        "_verify_adapter_against_inventory",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        pipeline,
        "_verify_raw_source_files_unchanged",
        lambda *_args, **_kwargs: None,
    )


def test_raw_adapter_replay_rejects_array_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapted_signal = np.zeros((len(LEADS), TARGET_SAMPLES), dtype=np.float32)
    stored_signal = adapted_signal.copy()
    stored_signal[0, 0] = np.float32(1.0)
    row = _private_row(
        quality_status="pass",
        route="UNSUPPORTED_INPUT",
        canonical_signal_sha256=pipeline._tensor_sha256(adapted_signal),
    )
    adapted = SimpleNamespace(
        signal_mv=adapted_signal,
        provenance_sha256=row.adapter_provenance_sha256,
    )
    _mock_raw_replay_storage(monkeypatch, stored_signal)
    monkeypatch.setattr(pipeline, "_adapter_for_record", lambda *_args: adapted)

    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="raw-source adapter replay differs",
    ):
        pipeline._verify_raw_to_canonical_replay(
            tmp_path,
            inputs=_raw_replay_inputs(),
            expected_records=(row,),
        )


def test_raw_adapter_replay_rejects_adapter_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stored_signal = np.zeros((len(LEADS), TARGET_SAMPLES), dtype=np.float32)
    row = _private_row(
        quality_status="pass",
        route="UNSUPPORTED_INPUT",
        canonical_signal_sha256=pipeline._tensor_sha256(stored_signal),
    )
    _mock_raw_replay_storage(monkeypatch, stored_signal)

    def fail_adapter(*_args: object) -> None:
        raise ExternalECGAdapterError("synthetic adapter failure")

    monkeypatch.setattr(pipeline, "_adapter_for_record", fail_adapter)
    with pytest.raises(ExternalECGAdapterError, match="synthetic adapter failure"):
        pipeline._verify_raw_to_canonical_replay(
            tmp_path,
            inputs=_raw_replay_inputs(),
            expected_records=(row,),
        )


def test_raw_adapter_replay_rejects_completed_adapter_failure_row_before_source_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stored_signal = np.zeros((len(LEADS), TARGET_SAMPLES), dtype=np.float32)
    row = replace(_private_row(), adapter_provenance_sha256=None)
    _mock_raw_replay_storage(monkeypatch, stored_signal)
    monkeypatch.setattr(
        pipeline,
        "_adapter_for_record",
        lambda *_args: pytest.fail("adapter must not run for a terminal failure row"),
    )

    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="completed bundle contains an adapter-failure row",
    ):
        pipeline._verify_raw_to_canonical_replay(
            tmp_path,
            inputs=_raw_replay_inputs(),
            expected_records=(row,),
        )


def test_full_backbone_replay_rejects_first_repeated_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    signal = np.zeros((len(LEADS), TARGET_SAMPLES), dtype=np.float32)
    row = _private_row(
        quality_status="pass",
        route="UNSUPPORTED_INPUT",
        canonical_signal_sha256=pipeline._tensor_sha256(signal),
    )
    monkeypatch.setattr(
        pipeline,
        "_verify_canonical_signal_sidecar",
        lambda *_args, **_kwargs: (object(),),
    )
    monkeypatch.setattr(
        pipeline,
        "_load_canonical_signal_shard",
        lambda *_args, **_kwargs: {0: signal},
    )
    monkeypatch.setattr(pipeline, "model_state_sha256", lambda _model: _DIGEST_A)
    first = np.zeros((1, 512), dtype=np.float32)
    repeated = first.copy()
    repeated[0, 0] = np.float32(1.0)
    monkeypatch.setattr(
        pipeline,
        "extract_embeddings_twice",
        lambda *_args, **_kwargs: SimpleNamespace(first=first, repeated=repeated),
    )

    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="full-model CUDA embedding replay is nondeterministic",
    ):
        pipeline._replay_quality_pass_embeddings(
            tmp_path,
            inputs=cast(Any, SimpleNamespace(inventory=SimpleNamespace(records=(object(),)))),
            expected_records=(row,),
            model=cast(Any, object()),
            normalization=cast(
                Any,
                SimpleNamespace(mean=(0.0,) * len(LEADS), std=(1.0,) * len(LEADS)),
            ),
            runtime=cast(Any, object()),
        )


def test_full_backbone_zero_pass_does_not_touch_signal_or_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    row = _private_row()
    for name in (
        "_verify_canonical_signal_sidecar",
        "_load_canonical_signal_shard",
        "model_state_sha256",
        "extract_embeddings_twice",
    ):
        monkeypatch.setattr(
            pipeline,
            name,
            lambda *_args, _name=name, **_kwargs: pytest.fail(
                f"zero-pass replay called {_name}"
            ),
        )

    first, repeated = pipeline._replay_quality_pass_embeddings(
        tmp_path,
        inputs=cast(Any, object()),
        expected_records=(row,),
        model=cast(Any, object()),
        normalization=cast(Any, object()),
        runtime=cast(Any, object()),
    )

    assert first.shape == (0, 512)
    assert repeated.shape == (0, 512)
    assert first.dtype == np.dtype(np.float32)
    assert repeated.dtype == np.dtype(np.float32)
    assert first is not repeated


def test_embedding_bundle_binds_each_stored_pass_to_its_matching_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    private.mkdir()
    embeddings = np.zeros((1, 512), dtype=np.float32)
    logits = np.zeros((1, len(SUPERCLASSES)), dtype=np.float64)
    probabilities = np.full((1, len(SUPERCLASSES)), 0.5, dtype=np.float64)
    scores = np.zeros((1,), dtype=np.float64)
    npz_path = private / "quality-pass-embeddings.npz"
    np.savez(
        npz_path,
        dataset=np.asarray(["synthetic-dataset"], dtype=np.str_),
        embedding_first=embeddings,
        embedding_repeated=embeddings.copy(),
        logits_first=logits,
        logits_repeated=logits.copy(),
        patient_key=np.asarray([""], dtype=np.str_),
        probabilities=probabilities,
        record_ref=np.asarray(["synthetic-record"], dtype=np.str_),
        score=scores,
    )
    sidecar: dict[str, object] = {
        "artifact_sha256": _DIGEST_A,
        "artifact_type": pipeline.PRIVATE_EMBEDDING_ARTIFACT_TYPE,
        "embedding_dimension": 512,
        "embedding_dtype": "float32",
        "embedding_tensor_sha256": pipeline._tensor_sha256(embeddings),
        "first_logits_tensor_sha256": pipeline._tensor_sha256(logits),
        "inventory_sha256": _DIGEST_A,
        "logits_dtype": "float64",
        "model_state_after_sha256": _DIGEST_B,
        "model_state_before_sha256": _DIGEST_B,
        "model_state_unchanged": True,
        "npz_file_sha256": sha256_file(npz_path),
        "probabilities_dtype": "float64",
        "probabilities_tensor_sha256": pipeline._tensor_sha256(probabilities),
        "protocol_id": pipeline.PROTOCOL_ID,
        "quality_pass_records": 1,
        "repeated_embedding_tensor_sha256": pipeline._tensor_sha256(embeddings),
        "repeated_logits_tensor_sha256": pipeline._tensor_sha256(logits),
        "repeat_verified": True,
        "schema_version": 1,
        "score_dtype": "float64",
        "score_tensor_sha256": pipeline._tensor_sha256(scores),
    }
    monkeypatch.setattr(
        pipeline,
        "_load_private_sidecar",
        lambda *_args, **_kwargs: sidecar,
    )
    detector = SimpleNamespace(
        score=lambda values: np.zeros((len(values),), dtype=np.float64)
    )
    conformal = SimpleNamespace(
        predict=lambda _values: SimpleNamespace(
            decisions=(
                (BinaryDecision.NOT_SUPPORTED,) * len(SUPERCLASSES),
            )
        )
    )
    inputs = cast(
        Any,
        SimpleNamespace(
            inventory=SimpleNamespace(inventory_sha256=_DIGEST_A),
            v1=SimpleNamespace(policy=SimpleNamespace(to_detector=lambda: detector)),
            routing=SimpleNamespace(
                temperature=1.0,
                conformal=conformal,
                maximum_entropy=1.0,
            ),
        ),
    )
    row = _private_row(
        quality_status="pass",
        route="UNSUPPORTED_INPUT",
    )
    replayed_second = embeddings.copy()
    replayed_second[0, 0] = np.float32(1.0)

    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="stored embeddings differ from exact full-model CUDA replay",
    ):
        pipeline._verify_embedding_bundle_semantics(
            private,
            inputs=inputs,
            quality_pass_rows=(row,),
            replayed_embeddings=(embeddings.copy(), replayed_second),
        )


def _record_evidence_inputs() -> Any:
    inventory_record = SimpleNamespace(
        dataset="synthetic-dataset",
        record_ref="synthetic-record",
        patient_key=None,
        challenge_quality_label="acceptable",
    )
    return cast(
        Any,
        SimpleNamespace(
            inventory=SimpleNamespace(
                records=(inventory_record,),
                inventory_sha256=_DIGEST_A,
            ),
            parent=SimpleNamespace(file_sha256=_DIGEST_B, threshold=1.5),
            child=SimpleNamespace(file_sha256=_DIGEST_C),
            routing=SimpleNamespace(
                demo_policy_file_sha256=_DIGEST_A,
                source_calibration_file_sha256=_DIGEST_B,
            ),
        ),
    )


def _write_record_evidence(
    path: Path,
    *,
    row: dict[str, object],
) -> None:
    body: dict[str, object] = {
        "artifact_type": pipeline.PRIVATE_EVIDENCE_ARTIFACT_TYPE,
        "child_contract_file_sha256": _DIGEST_C,
        "decision_bindings": {
            "demo_policy_file_sha256": _DIGEST_A,
            "source_calibration_file_sha256": _DIGEST_B,
        },
        "inventory_sha256": _DIGEST_A,
        "parent_config_file_sha256": _DIGEST_B,
        "protocol_id": pipeline.PROTOCOL_ID,
        "record_count": 1,
        "records": [row],
        "route_counts": {
            route: int(route == "REACQUIRE") for route in pipeline.FROZEN_ROUTE_ORDER
        },
        "schema_version": 1,
        "threshold": 1.5,
    }
    body["artifact_sha256"] = canonical_sha256(body)
    path.write_bytes(canonical_json_bytes(body))


def test_record_index_requires_explicit_null_and_loader_rejects_non_null_body(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    row = _private_row(quality_report={"private": "shard-only"})
    serialized = pipeline._private_record_index_dict(row)
    assert set(serialized) == set(row.to_dict()) | {"quality_report"}
    assert serialized["quality_report"] is None
    assert serialized["quality_report_sha256"] == row.quality_report_sha256

    monkeypatch.setattr(
        pipeline,
        "_verify_quality_audit_shards",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        pipeline,
        "_verify_private_route_semantics",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        pipeline,
        "_verify_private_adapter_semantics",
        lambda *_args, **_kwargs: None,
    )
    null_path = tmp_path / "record-evidence-null.json"
    _write_record_evidence(null_path, row=serialized)
    loaded = pipeline._load_private_record_evidence(
        null_path,
        inputs=_record_evidence_inputs(),
    )
    assert len(loaded) == 1
    assert loaded[0].quality_report is None
    assert loaded[0].quality_report_sha256 == row.quality_report_sha256

    non_null = dict(serialized)
    non_null["quality_report"] = {"private": "must-not-be-in-index"}
    non_null_path = tmp_path / "record-evidence-non-null.json"
    _write_record_evidence(non_null_path, row=non_null)
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="must store only the sharded quality-report hash",
    ):
        pipeline._load_private_record_evidence(
            non_null_path,
            inputs=_record_evidence_inputs(),
        )


def test_historical_source_gate_copies_exact_sealed_assignment_identity() -> None:
    source_validation = SimpleNamespace(
        source_assignment_sha256=_SEALED_SOURCE_ASSIGNMENT_SHA256,
        records=10,
        patients=5,
        rejected_records=1,
        accepted_records=9,
        record_false_rejection_rate=0.1,
        source_record_support_coverage=0.9,
        maximum_allowed_record_false_rejection_rate=0.05,
        cluster_bootstrap=SimpleNamespace(
            seed=7,
            replicates=1_000,
            two_sided_lower=0.05,
            two_sided_upper=0.2,
            one_sided_upper=0.15,
        ),
    )

    gate = pipeline._historical_source_gate(
        cast(Any, SimpleNamespace(source_validation=source_validation))
    )

    assert gate.cohort_key == "sealed-v1-source-validation"
    assert gate.cohort_manifest_sha256 == _SEALED_SOURCE_ASSIGNMENT_SHA256
    assert gate.interval.records == source_validation.records
    assert gate.interval.event_count == source_validation.rejected_records


def _worker_observations(root: Path) -> list[dict[str, object]]:
    observations: list[dict[str, object]] = []
    for path in sorted(root.glob("worker-*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise AssertionError("worker observation was not a JSON object")
        observations.append(cast(dict[str, object], value))
    return observations


def _runtime_scratch_snapshot(runtime_root: Path) -> dict[str, list[str]]:
    return {
        relative: sorted(entry.name for entry in (runtime_root / relative).iterdir())
        for relative in (
            "pycache",
            "temp",
            "home/AppData/Roaming",
            "home/AppData/Local",
        )
    }


def _run_windows_dataloader_smoke(
    *,
    runtime_root: Path,
    report_path: Path,
    observation_root: Path,
) -> int:
    """Engineering-only subprocess body for the real four-worker CUDA path."""

    global _SMOKE_FORWARD_CALLS
    if sys.platform != "win32":
        raise AssertionError("Windows DataLoader smoke requires Windows")
    if (
        sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.no_user_site != 1
        or not sys.dont_write_bytecode
    ):
        raise AssertionError("engineering smoke Python flags are not isolated")
    project_root = Path(__file__).resolve().parents[2]
    if Path.cwd().resolve(strict=True) != project_root:
        raise AssertionError("engineering smoke cwd differs from the project root")
    expected_environment = {
        name.upper(): value for name, value in os.environ.items()
    }
    expected_sys_path = list(sys.path)
    success_observations = observation_root / "success"
    failure_observations = observation_root / "failure"
    success_observations.mkdir(parents=True)
    failure_observations.mkdir()

    runtime = completion_runtime.configure_deterministic_cuda(
        expected_device_name="NVIDIA GeForce RTX 5070 Ti Laptop GPU",
        expected_compute_capability=(12, 0),
        expected_python_version="3.12.13",
        expected_torch_version="2.13.0+cu130",
        expected_cuda_runtime="13.0",
        expected_cudnn_version=92_000,
        expected_nvidia_driver_version="596.49",
        nvidia_smi_executable=(
            Path(os.environ["SYSTEMROOT"]) / "System32" / "nvidia-smi.exe"
        ),
    )
    model = ResNet1D(
        ResNet1DConfig(
            stage_channels=(512,),
            blocks_per_stage=(1,),
            block_dropout=0.0,
            classifier_dropout=0.0,
        )
    )
    model.forward_embedding = MethodType(  # type: ignore[method-assign]
        _synthetic_probe_embedding,
        model,
    )
    prepared = completion_runtime.prepare_resnet_for_embedding(model, runtime=runtime)

    loader_calls: list[dict[str, object]] = []
    completion_runtime_namespace = vars(completion_runtime)
    original_loader = cast(Any, completion_runtime_namespace["DataLoader"])

    def recording_loader(*args: object, **kwargs: object) -> object:
        loader_calls.append(
            {
                "batch_size": kwargs.get("batch_size"),
                "drop_last": kwargs.get("drop_last"),
                "num_workers": kwargs.get("num_workers"),
                "persistent_workers": kwargs.get("persistent_workers"),
                "pin_memory": kwargs.get("pin_memory"),
                "shuffle": kwargs.get("shuffle"),
            }
        )
        return original_loader(*args, **kwargs)

    if multiprocessing.active_children():
        raise AssertionError("engineering smoke started with active child processes")
    completion_runtime_namespace["DataLoader"] = recording_loader
    retained_error: RuntimeError | None = None
    try:
        _SMOKE_FORWARD_CALLS = 0
        success = completion_runtime.extract_embeddings_twice(
            prepared,
            _WindowsWorkerProbeDataset(
                records=512,
                observation_root=success_observations,
            ),
            runtime=runtime,
        )
        success_forward_calls = _SMOKE_FORWARD_CALLS
        success_children = [child.pid for child in multiprocessing.active_children()]

        failure_start_calls = _SMOKE_FORWARD_CALLS
        try:
            completion_runtime.extract_embeddings_twice(
                prepared,
                _WindowsWorkerProbeDataset(
                    records=130,
                    observation_root=failure_observations,
                    fail_index=129,
                ),
                runtime=runtime,
            )
        except RuntimeError as error:
            retained_error = error
        failure_forward_calls = _SMOKE_FORWARD_CALLS - failure_start_calls
        failure_children = [child.pid for child in multiprocessing.active_children()]
    finally:
        completion_runtime_namespace["DataLoader"] = original_loader

    if loader_calls != [
        {
            "batch_size": 128,
            "drop_last": False,
            "num_workers": 4,
            "persistent_workers": True,
            "pin_memory": True,
            "shuffle": False,
        },
        {
            "batch_size": 128,
            "drop_last": False,
            "num_workers": 4,
            "persistent_workers": True,
            "pin_memory": True,
            "shuffle": False,
        },
    ]:
        raise AssertionError("production DataLoader arguments differ from the smoke")
    if success_forward_calls != 8:
        raise AssertionError("success smoke did not execute eight ordered batches")
    if success_children:
        raise AssertionError("success smoke retained active DataLoader children")
    if (
        retained_error is None
        or "synthetic mid-pass worker failure" not in str(retained_error)
        or retained_error.__traceback__ is None
    ):
        raise AssertionError("worker failure traceback was not retained by the caller")
    if failure_forward_calls != 1:
        raise AssertionError("worker failure was not injected after the first batch")
    if failure_children:
        raise AssertionError("failure smoke retained active DataLoader children")

    first_probe = success.first[:, :_WORKER_PROBE_FIELDS]
    repeated_probe = success.repeated[:, :_WORKER_PROBE_FIELDS]
    if not np.array_equal(first_probe, repeated_probe):
        raise AssertionError("worker probe order differs between persistent passes")
    expected_order = np.arange(512, dtype=np.int64)
    if not np.array_equal(first_probe[:, 0].astype(np.int64), expected_order):
        raise AssertionError("production DataLoader did not retain record order")
    if set(first_probe[:, 1].astype(int).tolist()) != {0, 1, 2, 3}:
        raise AssertionError("production DataLoader did not use all four workers")
    if not np.all(first_probe[:, 3:10] == 1.0):
        raise AssertionError("worker flags, cwd, environment, or cache probe failed")
    worker_records = _worker_observations(success_observations)
    if len(worker_records) != 8:
        raise AssertionError("persistent workers did not emit both pass observations")
    for observation in worker_records:
        if observation.get("cwd") != os.fspath(project_root):
            raise AssertionError("worker cwd differs from the production root")
        if observation.get("environment") != expected_environment:
            raise AssertionError("worker environment differs from the sanitized parent")
        if observation.get("sys_path") != expected_sys_path:
            raise AssertionError("worker sys.path differs from the isolated parent")
        if observation.get("flags") != {
            "dont_write_bytecode": True,
            "isolated": 1,
            "no_site": 1,
            "no_user_site": 1,
        }:
            raise AssertionError("worker Python flags differ from -I -S -B")
        if observation.get("pycache_prefix") is not None:
            raise AssertionError("spawned worker unexpectedly retained pycache_prefix")

    if not np.array_equal(first_probe[:, 2], repeated_probe[:, 2]):
        raise AssertionError("second pass did not reuse the persistent worker processes")
    persistent_worker_pids = {
        int(worker_id): int(first_probe[first_probe[:, 1] == worker_id, 2][0])
        for worker_id in range(4)
    }

    scratch = _runtime_scratch_snapshot(runtime_root)
    if any(scratch.values()):
        raise AssertionError("isolated runtime scratch was not empty after worker cleanup")
    report = {
        "failure": {
            "active_children_after_return": failure_children,
            "forward_batches_before_failure": failure_forward_calls,
            "retained_traceback": retained_error.__traceback__ is not None,
        },
        "loader_calls": loader_calls,
        "runtime_scratch": scratch,
        "success": {
            "active_children_after_return": success_children,
            "extraction_passes": 2,
            "first_order_exact": True,
            "forward_batches": success_forward_calls,
            "persistent_worker_pids": persistent_worker_pids,
            "records": int(first_probe.shape[0]),
            "repeated_order_exact": True,
            "worker_ids": sorted(set(first_probe[:, 1].astype(int).tolist())),
        },
        "worker_runtime": {
            "cwd": os.fspath(project_root),
            "environment": expected_environment,
            "flags": worker_records[0]["flags"],
            "pycache_prefix": worker_records[0]["pycache_prefix"],
            "sys_path": expected_sys_path,
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


@pytest.mark.skipif(
    sys.platform != "win32"
    or os.environ.get(_WINDOWS_DATALOADER_SMOKE_ENV) != "1",
    reason="opt-in pre-freeze Windows CUDA engineering smoke",
)
def test_windows_production_dataloader_success_and_failure_cleanup(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    runtime_root = _runtime_scratch_root(tmp_path, "f" * 64)
    report_path = tmp_path / "windows-dataloader-smoke-report.json"
    observation_root = tmp_path / "worker-observations"
    launcher = _load_isolated_entrypoint(
        "scripts/evaluate_trust_sentinel_ood_external_v2.py"
    )
    environment = launcher._sanitized_runtime_environment(runtime_root)
    completed = subprocess.run(
        [
            os.fspath(Path(sys.executable).resolve(strict=True)),
            "-I",
            "-S",
            "-B",
            "-X",
            f"pycache_prefix={runtime_root / 'pycache'}",
            os.fspath(Path(__file__).resolve(strict=True)),
            _WINDOWS_DATALOADER_SMOKE_FLAG,
            os.fspath(runtime_root),
            os.fspath(report_path),
            os.fspath(observation_root),
        ],
        check=False,
        capture_output=True,
        cwd=project_root,
        encoding="utf-8",
        env=environment,
        errors="strict",
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["loader_calls"] == [
        {
            "batch_size": 128,
            "drop_last": False,
            "num_workers": 4,
            "persistent_workers": True,
            "pin_memory": True,
            "shuffle": False,
        },
        {
            "batch_size": 128,
            "drop_last": False,
            "num_workers": 4,
            "persistent_workers": True,
            "pin_memory": True,
            "shuffle": False,
        },
    ]
    assert report["success"]["records"] == 512
    assert report["success"]["first_order_exact"] is True
    assert report["success"]["repeated_order_exact"] is True
    assert report["success"]["worker_ids"] == [0, 1, 2, 3]
    assert report["success"]["extraction_passes"] == 2
    assert report["success"]["forward_batches"] == 8
    assert report["success"]["active_children_after_return"] == []
    assert report["failure"]["forward_batches_before_failure"] == 1
    assert report["failure"]["retained_traceback"] is True
    assert report["failure"]["active_children_after_return"] == []
    assert report["runtime_scratch"] == {
        "home/AppData/Local": [],
        "home/AppData/Roaming": [],
        "pycache": [],
        "temp": [],
    }
    assert report["worker_runtime"]["flags"] == {
        "dont_write_bytecode": True,
        "isolated": 1,
        "no_site": 1,
        "no_user_site": 1,
    }
    assert report["worker_runtime"]["pycache_prefix"] is None
    launcher._remove_empty_runtime_root(runtime_root)
    assert not runtime_root.exists()


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != _WINDOWS_DATALOADER_SMOKE_FLAG:
        raise SystemExit("engineering smoke invocation is invalid")
    multiprocessing.freeze_support()
    raise SystemExit(
        _run_windows_dataloader_smoke(
            runtime_root=Path(sys.argv[2]),
            report_path=Path(sys.argv[3]),
            observation_root=Path(sys.argv[4]),
        )
    )
