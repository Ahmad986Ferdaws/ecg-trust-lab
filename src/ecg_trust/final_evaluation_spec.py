"""Immutable preregistration for calibration and one-time final evaluation.

The specification is created before fold-9 inference.  It binds the sealed
refit grid, label-free subgroup metadata, disclosed deviations, scientific
settings, report contracts, and the clean committed CUDA runtime that is
authorized to execute the remaining protocol.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from ecg_trust import __version__
from ecg_trust.protocol import FINAL_TEST_FOLDS, ExperimentProtocol, load_protocol
from ecg_trust.subgroup_artifact import (
    SUBGROUP_ATTRIBUTES,
    SubgroupArtifact,
    SubgroupArtifactError,
    load_subgroup_artifact,
)

if TYPE_CHECKING:
    from ecg_trust.release_gates import RefitBundle

FINAL_EVALUATION_SPEC_SCHEMA_VERSION = 1
FINAL_EVALUATION_SPEC_TYPE = "ecg_trust.final_evaluation_specification"
CANONICAL_COVERAGE_TARGETS: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.5)
EXPECTED_ARCHITECTURES: tuple[str, ...] = ("resnet1d", "ecg_transformer")
EXPECTED_SEEDS: tuple[int, ...] = (2026, 2027, 2028)
CORE_RUNTIME_DISTRIBUTIONS: tuple[str, ...] = (
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "wfdb",
    "pyarrow",
)
CANONICAL_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


class FinalEvaluationSpecError(ValueError):
    """Raised when a requested final-evaluation freeze is not canonical."""


class FinalEvaluationSpecIntegrityError(FinalEvaluationSpecError):
    """Raised when a stored specification or bound source has changed."""


@dataclass(frozen=True, slots=True)
class FinalEvaluationSpec:
    """A validated, self-hashed final-evaluation specification."""

    path: Path | None
    artifact_sha256: str
    _canonical_payload: str

    @property
    def payload(self) -> dict[str, object]:
        decoded: object = json.loads(self._canonical_payload)
        if not isinstance(decoded, dict):  # pragma: no cover - constructor invariant
            raise FinalEvaluationSpecIntegrityError("stored spec payload is not an object")
        return cast(dict[str, object], decoded)

    @property
    def protocol_hash(self) -> str:
        return cast(str, _mapping(self.payload["protocol"], "protocol")["protocol_hash"])

    @property
    def refit_bundle_sha256(self) -> str:
        return cast(
            str,
            _mapping(self.payload["refit_bundle"], "refit_bundle")["artifact_sha256"],
        )

    @property
    def subgroup_artifact_sha256(self) -> str:
        return cast(
            str,
            _mapping(self.payload["subgroup_artifact"], "subgroup_artifact")[
                "artifact_sha256"
            ],
        )

    @property
    def manifest_sha256(self) -> str:
        return cast(
            str,
            _mapping(self.payload["refit_bundle"], "refit_bundle")["manifest_sha256"],
        )

    @property
    def requested_device(self) -> str:
        runtime = _mapping(self.payload["runtime_envelope"], "runtime_envelope")
        hardware = _mapping(runtime["hardware"], "runtime_envelope.hardware")
        return cast(str, hardware["requested_device"])

    def to_payload(self) -> dict[str, object]:
        return self.payload


def canonical_sha256(payload: Mapping[str, object]) -> str:
    """Return a prefixed SHA-256 over finite canonical JSON."""

    try:
        serialized = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise FinalEvaluationSpecError("specification must be finite JSON") from error
    return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise FinalEvaluationSpecIntegrityError(f"required source is missing: {source}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise FinalEvaluationSpecIntegrityError(
            f"could not hash required source {source}: {error}"
        ) from error
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path, context: str) -> Mapping[str, object]:
    if not path.is_file():
        raise FinalEvaluationSpecIntegrityError(f"{context} is missing: {path}")
    if path.stat().st_size > 100_000_000:
        raise FinalEvaluationSpecIntegrityError(f"{context} is unreasonably large")
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FinalEvaluationSpecIntegrityError(
            f"could not decode {context}: {error}"
        ) from error
    return _mapping(decoded, context)


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise FinalEvaluationSpecIntegrityError(
            f"{context} must be a string-keyed mapping"
        )
    return cast(Mapping[str, object], value)


def _sequence(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise FinalEvaluationSpecIntegrityError(f"{context} must be a sequence")
    return cast(Sequence[object], value)


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FinalEvaluationSpecIntegrityError(f"{context} must be a non-empty string")
    return value


def _hash(value: object, context: str) -> str:
    text = _string(value, context)
    digest = text.removeprefix("sha256:")
    if (
        not text.startswith("sha256:")
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise FinalEvaluationSpecIntegrityError(
            f"{context} must be a prefixed lower-case SHA-256"
        )
    return text


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FinalEvaluationSpecIntegrityError(
            f"{context} must be an integer >= {minimum}"
        )
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], context: str) -> None:
    if set(value) != expected:
        missing = sorted(expected.difference(value))
        unknown = sorted(set(value).difference(expected))
        raise FinalEvaluationSpecIntegrityError(
            f"{context} keys are not canonical; missing={missing}, unknown={unknown}"
        )


def _calibration_policy() -> dict[str, object]:
    return {
        "coverage_targets": list(CANONICAL_COVERAGE_TARGETS),
        "temperature_scope": "one_global_temperature_per_member",
        "threshold_scope": "five_per_label_thresholds_per_member",
        "entropy_method": "mean_normalized_binary_entropy",
        "independent_member_fits": 6,
        "source_folds": [9],
        "retuning_after_freeze": False,
    }


def _evaluation_policy() -> dict[str, object]:
    return {
        "final_folds": list(FINAL_TEST_FOLDS),
        "patient_resampling": "patient_cluster_percentile_bootstrap",
        "bootstrap_resamples": 1_000,
        "bootstrap_base_seed": 20_260_808,
        "bootstrap_confidence": 0.95,
        "bootstrap_minimum_valid": 500,
        "bootstrap_seed_strategy": "base_plus_model_seed",
        "ece_bins": 15,
        "minimum_group_samples": 30,
        "minimum_group_patients": 20,
        "retuning_allowed": False,
    }


def _comparison_policy() -> dict[str, object]:
    return {
        "paired_model_seed_comparison": {
            "architectures": list(EXPECTED_ARCHITECTURES),
            "seeds": list(EXPECTED_SEEDS),
            "pairing": "within_seed_same_fold10_patients",
            "alignment": "exact_prediction_artifact_alignment",
            "direction": "ecg_transformer_minus_resnet1d",
            "patient_resampling": "paired_patient_cluster_percentile_bootstrap",
        },
        "architecture_aggregation": {
            "architectures": list(EXPECTED_ARCHITECTURES),
            "seeds": list(EXPECTED_SEEDS),
            "metrics": ["roc_auc", "average_precision", "brier_score", "ece"],
            "statistics": [
                "values",
                "mean",
                "sample_standard_deviation",
                "median",
                "minimum",
                "maximum",
                "valid_seeds",
            ],
            "interpretation": "descriptive_summary_across_three_preregistered_seeds",
            "missing_metric_policy": "finite_values_only_with_valid_seed_count",
        },
    }


def _canonical_subgroup_definitions() -> dict[str, object]:
    return {
        "sex": {
            "source_column": "sex",
            "mapping": {"0": "male", "1": "female"},
            "missing_group": "unknown",
            "source_semantics": "PTB-XL metadata: male=0, female=1",
        },
        "age_band": {
            "source_column": "age",
            "bands": [
                {"group": "<40", "minimum_inclusive": 0, "maximum_inclusive": 39},
                {"group": "40-59", "minimum_inclusive": 40, "maximum_inclusive": 59},
                {"group": "60-79", "minimum_inclusive": 60, "maximum_inclusive": 79},
                {"group": "80+", "minimum_inclusive": 80, "maximum_inclusive": 120},
            ],
            "censored_sentinel": {
                "value": 300,
                "meaning": "age greater than 89 years",
                "group": "80+",
            },
            "missing_group": "unknown",
        },
    }


def _report_contract() -> dict[str, object]:
    return {
        "schemas": {
            "member_final_report": {
                "schema_version": 1,
                "report_type": "ecg_trust.final_evaluation_report",
            },
            "architecture_aggregate": {
                "schema_version": 1,
                "artifact_type": "ecg_trust.final_architecture_aggregate",
            },
            "paired_model_report": {
                "schema_version": 1,
                "artifact_type": "ecg_trust.paired_patient_bootstrap_report",
            },
            "paired_bootstrap_manifest": {
                "schema_version": 1,
                "artifact_type": "ecg_trust.paired_patient_bootstrap_manifest",
            },
            "final_batch_summary": {
                "schema_version": 1,
                "artifact_type": "ecg_trust.final_batch_summary",
            },
        },
        "protocol_deviations_must_be_cited": True,
        "retuning_allowed": False,
        "retuning_performed": False,
    }


def _run_git(project_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={project_root}", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise FinalEvaluationSpecError(f"Git state is unavailable: {error}") from error
    return result.stdout.strip()


def _normalized_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).casefold()


def _capture_installed_environment() -> dict[str, object]:
    """Hash all installed distribution versions and expose core package versions."""

    entries: list[dict[str, str]] = []
    versions: dict[str, set[str]] = {}
    for distribution in importlib.metadata.distributions():
        metadata = distribution.metadata
        raw_name = metadata.get("Name")
        raw_version = metadata.get("Version")
        if (
            not isinstance(raw_name, str)
            or not raw_name.strip()
            or not isinstance(raw_version, str)
            or not raw_version.strip()
        ):
            # Some managed environments deny METADATA reads while still exposing
            # the standardized ``name-version.dist-info`` directory.  Bind that
            # identity rather than silently dropping the installed distribution.
            distribution_path = Path(str(getattr(distribution, "_path", "")))
            filename = distribution_path.name
            if filename.casefold().endswith(".dist-info"):
                filename = filename[: -len(".dist-info")]
            elif filename.casefold().endswith(".egg-info"):
                filename = filename[: -len(".egg-info")]
            if "-" not in filename:
                raise FinalEvaluationSpecError(
                    "an installed Python distribution has no bindable identity"
                )
            raw_name, raw_version = filename.rsplit("-", 1)
        name = _normalized_distribution_name(raw_name.strip())
        version = raw_version.strip()
        if not name or not version:
            raise FinalEvaluationSpecError(
                "an installed Python distribution has no bindable identity"
            )
        entries.append({"name": name, "version": version})
        versions.setdefault(name, set()).add(version)
    entries.sort(key=lambda entry: (entry["name"], entry["version"]))
    if not entries:
        raise FinalEvaluationSpecError("installed Python environment is unavailable")
    conflicts = sorted(name for name, values in versions.items() if len(values) != 1)
    if conflicts:
        raise FinalEvaluationSpecError(
            f"installed Python distributions have conflicting versions: {conflicts}"
        )
    missing = [name for name in CORE_RUNTIME_DISTRIBUTIONS if name not in versions]
    if missing:
        raise FinalEvaluationSpecError(
            f"required runtime distributions are unavailable: {missing}"
        )
    core = {
        name: next(iter(versions[name])) for name in CORE_RUNTIME_DISTRIBUTIONS
    }
    return {
        "distribution_count": len(entries),
        "distributions_sha256": canonical_sha256({"distributions": entries}),
        "core_packages": core,
    }


def _capture_git_envelope(project_root: Path) -> dict[str, object]:
    root_text = _run_git(project_root, "rev-parse", "--show-toplevel")
    if not root_text:
        raise FinalEvaluationSpecError("Git repository root is unavailable")
    git_root = Path(root_text).resolve()
    if git_root != project_root:
        raise FinalEvaluationSpecError(
            "project_root must be the exact Git repository root"
        )
    revision = _run_git(project_root, "rev-parse", "HEAD").casefold()
    if len(revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise FinalEvaluationSpecError("Git revision is unavailable or invalid")
    status = _run_git(
        project_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise FinalEvaluationSpecError(
            "final-evaluation specification requires a clean committed worktree"
        )
    return {"revision": revision, "dirty": False}


def _configure_deterministic_runtime() -> None:
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace != CANONICAL_CUBLAS_WORKSPACE_CONFIG:
        raise FinalEvaluationSpecError(
            "CUBLAS_WORKSPACE_CONFIG must be :4096:8 before CUDA initialization"
        )
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _capture_execution_policy() -> dict[str, object]:
    policy: dict[str, object] = {
        "allow_device_auto": False,
        "require_cuda": True,
        "bf16_requested": True,
        "bf16_required": True,
        "autocast_dtype": "torch.bfloat16",
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    static = dict(policy)
    static.pop("cuda_visible_devices")
    if static != {
        "allow_device_auto": False,
        "require_cuda": True,
        "bf16_requested": True,
        "bf16_required": True,
        "autocast_dtype": "torch.bfloat16",
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "cublas_workspace_config": CANONICAL_CUBLAS_WORKSPACE_CONFIG,
    }:
        raise FinalEvaluationSpecError(
            "deterministic CUDA/TF32 execution policy is not active"
        )
    visible_devices = policy["cuda_visible_devices"]
    if visible_devices is not None and (
        not isinstance(visible_devices, str) or not visible_devices.strip()
    ):
        raise FinalEvaluationSpecError("CUDA_VISIBLE_DEVICES is invalid")
    return policy


def _selected_device_supports_bf16(index: int) -> bool:
    try:
        with torch.cuda.device(index):
            if torch.cuda.current_device() != index:
                raise FinalEvaluationSpecError(
                    "selected CUDA context does not match the requested device"
                )
            return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except (RuntimeError, ValueError) as error:
        raise FinalEvaluationSpecError(
            "could not evaluate BF16 on the selected CUDA device"
        ) from error


def _capture_nvidia_driver(gpu_uuid: str) -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise FinalEvaluationSpecError(
            f"NVIDIA driver state is unavailable: {error}"
        ) from error
    matches: list[str] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0].casefold() == gpu_uuid.casefold():
            matches.append(parts[1])
    if len(matches) != 1 or re.fullmatch(r"\d+(?:\.\d+)+", matches[0]) is None:
        raise FinalEvaluationSpecError(
            "NVIDIA driver version for the selected GPU is unavailable"
        )
    return matches[0]


def _capture_runtime_envelope(
    project_root: Path, requested_device: str
) -> dict[str, object]:
    _configure_deterministic_runtime()
    if not isinstance(requested_device, str) or not requested_device.strip():
        raise FinalEvaluationSpecError("device must be a non-empty explicit CUDA device")
    normalized = requested_device.strip().casefold()
    if normalized == "auto":
        raise FinalEvaluationSpecError("device=auto is forbidden for the final freeze")
    try:
        device = torch.device(normalized)
    except (RuntimeError, ValueError) as error:
        raise FinalEvaluationSpecError(f"invalid CUDA device {requested_device!r}") from error
    if device.type != "cuda" or device.index is None:
        raise FinalEvaluationSpecError(
            "the final runtime must use an indexed CUDA device such as cuda:0"
        )
    if not torch.cuda.is_available():
        raise FinalEvaluationSpecError("CUDA is unavailable")
    index = device.index
    if index < 0 or index >= torch.cuda.device_count():
        raise FinalEvaluationSpecError("requested CUDA device index is unavailable")
    if not _selected_device_supports_bf16(index):
        raise FinalEvaluationSpecError("the selected CUDA runtime must support BF16")
    cuda_runtime = torch.version.cuda
    cudnn = torch.backends.cudnn.version()  # type: ignore[no-untyped-call]
    if not isinstance(cuda_runtime, str) or not cuda_runtime.strip():
        raise FinalEvaluationSpecError("PyTorch CUDA runtime version is unavailable")
    if isinstance(cudnn, bool) or not isinstance(cudnn, int) or cudnn <= 0:
        raise FinalEvaluationSpecError("cuDNN version is unavailable")
    properties = torch.cuda.get_device_properties(index)
    device_name = torch.cuda.get_device_name(index)
    if not device_name.strip():
        raise FinalEvaluationSpecError("CUDA device name is unavailable")
    raw_uuid = str(getattr(properties, "uuid", "")).strip()
    if raw_uuid.casefold().startswith("gpu-"):
        raw_uuid = raw_uuid[4:]
    try:
        gpu_uuid = f"GPU-{uuid.UUID(raw_uuid)}"
    except (AttributeError, ValueError) as error:
        raise FinalEvaluationSpecError("CUDA device UUID is unavailable") from error
    pci_fields: dict[str, int] = {}
    for field in ("pci_domain_id", "pci_bus_id", "pci_device_id"):
        value = getattr(properties, field, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise FinalEvaluationSpecError(f"CUDA {field} is unavailable")
        pci_fields[field] = value
    nvidia_driver = _capture_nvidia_driver(gpu_uuid)
    lock_path = project_root / "uv.lock"
    return {
        "project_root": str(project_root),
        "git": _capture_git_envelope(project_root),
        "dependency_lock": {
            "path": str(lock_path.resolve()),
            "sha256": _file_sha256(lock_path),
        },
        "software": {
            "python": platform.python_version(),
            "ecg_trust": __version__,
            "torch": str(torch.__version__),
            "cuda_runtime": cuda_runtime,
            "cudnn": cudnn,
            "nvidia_driver": nvidia_driver,
            "installed_environment": _capture_installed_environment(),
        },
        "hardware": {
            "requested_device": normalized,
            "resolved_device": f"cuda:{index}",
            "device_name": device_name,
            "device_uuid": gpu_uuid,
            **pci_fields,
            "device_capability": list(torch.cuda.get_device_capability(index)),
            "total_memory_bytes": properties.total_memory,
            "bf16_supported": True,
        },
        "policy": _capture_execution_policy(),
    }


def _validate_runtime_envelope(value: object) -> Mapping[str, object]:
    root = _mapping(value, "runtime_envelope")
    _exact_keys(
        root,
        {"project_root", "git", "dependency_lock", "software", "hardware", "policy"},
        "runtime_envelope",
    )
    Path(_string(root["project_root"], "runtime_envelope.project_root"))
    git = _mapping(root["git"], "runtime_envelope.git")
    _exact_keys(git, {"revision", "dirty"}, "runtime_envelope.git")
    revision = _string(git["revision"], "runtime_envelope.git.revision")
    if len(revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise FinalEvaluationSpecIntegrityError("runtime Git revision is invalid")
    if git["dirty"] is not False:
        raise FinalEvaluationSpecIntegrityError("runtime Git worktree must be clean")
    lock = _mapping(root["dependency_lock"], "runtime_envelope.dependency_lock")
    _exact_keys(lock, {"path", "sha256"}, "runtime_envelope.dependency_lock")
    _string(lock["path"], "runtime dependency lock path")
    _hash(lock["sha256"], "runtime dependency lock hash")
    software = _mapping(root["software"], "runtime_envelope.software")
    _exact_keys(
        software,
        {
            "python",
            "ecg_trust",
            "torch",
            "cuda_runtime",
            "cudnn",
            "nvidia_driver",
            "installed_environment",
        },
        "runtime_envelope.software",
    )
    for key in ("python", "ecg_trust", "torch", "cuda_runtime"):
        _string(software[key], f"runtime software {key}")
    _integer(software["cudnn"], "runtime software cudnn", minimum=1)
    driver = _string(software["nvidia_driver"], "runtime NVIDIA driver")
    if re.fullmatch(r"\d+(?:\.\d+)+", driver) is None:
        raise FinalEvaluationSpecIntegrityError("runtime NVIDIA driver is invalid")
    installed = _mapping(
        software["installed_environment"], "runtime installed_environment"
    )
    _exact_keys(
        installed,
        {"distribution_count", "distributions_sha256", "core_packages"},
        "runtime installed_environment",
    )
    _integer(
        installed["distribution_count"],
        "runtime distribution_count",
        minimum=len(CORE_RUNTIME_DISTRIBUTIONS),
    )
    _hash(installed["distributions_sha256"], "runtime distributions hash")
    core = _mapping(installed["core_packages"], "runtime core_packages")
    _exact_keys(core, set(CORE_RUNTIME_DISTRIBUTIONS), "runtime core_packages")
    for name in CORE_RUNTIME_DISTRIBUTIONS:
        _string(core[name], f"runtime core package {name}")
    hardware = _mapping(root["hardware"], "runtime_envelope.hardware")
    _exact_keys(
        hardware,
        {
            "requested_device",
            "resolved_device",
            "device_name",
            "device_uuid",
            "pci_domain_id",
            "pci_bus_id",
            "pci_device_id",
            "device_capability",
            "total_memory_bytes",
            "bf16_supported",
        },
        "runtime_envelope.hardware",
    )
    requested = _string(hardware["requested_device"], "requested_device").casefold()
    resolved = _string(hardware["resolved_device"], "resolved_device").casefold()
    try:
        requested_device = torch.device(requested)
    except (RuntimeError, ValueError) as error:
        raise FinalEvaluationSpecIntegrityError(
            "runtime requested device is invalid"
        ) from error
    if requested_device.type != "cuda" or requested_device.index is None:
        raise FinalEvaluationSpecIntegrityError(
            "runtime device must be indexed CUDA such as cuda:0"
        )
    if not resolved.startswith("cuda:"):
        raise FinalEvaluationSpecIntegrityError("resolved device must name a CUDA index")
    if requested != resolved:
        raise FinalEvaluationSpecIntegrityError(
            "requested and resolved CUDA devices must be identical"
        )
    _string(hardware["device_name"], "runtime CUDA device name")
    device_uuid = _string(hardware["device_uuid"], "runtime CUDA device UUID")
    if not device_uuid.startswith("GPU-"):
        raise FinalEvaluationSpecIntegrityError("runtime CUDA device UUID is invalid")
    try:
        canonical_uuid = str(uuid.UUID(device_uuid.removeprefix("GPU-")))
    except ValueError as error:
        raise FinalEvaluationSpecIntegrityError(
            "runtime CUDA device UUID is invalid"
        ) from error
    if device_uuid != f"GPU-{canonical_uuid}":
        raise FinalEvaluationSpecIntegrityError(
            "runtime CUDA device UUID is not canonical"
        )
    for field in ("pci_domain_id", "pci_bus_id", "pci_device_id"):
        _integer(hardware[field], f"runtime CUDA {field}")
    capability = _sequence(hardware["device_capability"], "device_capability")
    if len(capability) != 2:
        raise FinalEvaluationSpecIntegrityError("CUDA capability must have two integers")
    for index, component in enumerate(capability):
        _integer(component, f"CUDA capability {index}")
    _integer(hardware["total_memory_bytes"], "CUDA total memory", minimum=1)
    if hardware["bf16_supported"] is not True:
        raise FinalEvaluationSpecIntegrityError("runtime must support BF16")
    policy = _mapping(root["policy"], "runtime_envelope.policy")
    expected_policy: dict[str, object] = {
        "allow_device_auto": False,
        "require_cuda": True,
        "bf16_requested": True,
        "bf16_required": True,
        "autocast_dtype": "torch.bfloat16",
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "cublas_workspace_config": CANONICAL_CUBLAS_WORKSPACE_CONFIG,
    }
    _exact_keys(
        policy,
        {*expected_policy, "cuda_visible_devices"},
        "runtime_envelope.policy",
    )
    visible_devices = policy["cuda_visible_devices"]
    if visible_devices is not None and (
        not isinstance(visible_devices, str) or not visible_devices.strip()
    ):
        raise FinalEvaluationSpecIntegrityError(
            "runtime CUDA_VISIBLE_DEVICES is invalid"
        )
    static_policy = dict(policy)
    static_policy.pop("cuda_visible_devices")
    if static_policy != expected_policy:
        raise FinalEvaluationSpecIntegrityError("CUDA/BF16 policy is not canonical")
    return root


def _verify_protocol_source(
    protocol_path: Path, protocol: ExperimentProtocol
) -> dict[str, object]:
    try:
        loaded = load_protocol(protocol_path)
    except (OSError, RuntimeError, ValueError) as error:
        raise FinalEvaluationSpecIntegrityError(
            f"could not verify protocol source: {error}"
        ) from error
    if loaded.to_resolved_dict() != protocol.to_resolved_dict():
        raise FinalEvaluationSpecIntegrityError(
            "protocol source differs from the supplied protocol"
        )
    return {
        "source_path": str(protocol_path.resolve()),
        "source_file_sha256": _file_sha256(protocol_path),
        "protocol_hash": protocol.protocol_hash,
        "dataset_name": protocol.dataset_name,
        "dataset_version": protocol.dataset_version,
        "resolved": protocol.to_resolved_dict(),
    }


def _load_refit(path: Path, protocol: ExperimentProtocol) -> RefitBundle:
    # Lazy import keeps this module safe for release_gates to import later.
    from ecg_trust.release_gates import load_refit_bundle

    try:
        return load_refit_bundle(path, protocol=protocol, verify_sources=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise FinalEvaluationSpecIntegrityError(
            f"could not verify sealed refit bundle: {error}"
        ) from error


def _refit_binding(path: Path, bundle: RefitBundle) -> dict[str, object]:
    if bundle.artifact_sha256 is None:
        raise FinalEvaluationSpecIntegrityError("refit bundle is not self-hashed")
    if len(bundle.members) != 6:
        raise FinalEvaluationSpecIntegrityError("refit bundle must contain six members")
    return {
        "path": str(path.resolve()),
        "file_sha256": _file_sha256(path),
        "artifact_sha256": bundle.artifact_sha256,
        "protocol_hash": bundle.protocol_hash,
        "manifest_sha256": bundle.manifest_sha256,
        "normalization_sha256": bundle.normalization_sha256,
        "label_order": list(bundle.label_order),
        "member_count": len(bundle.members),
    }


def _load_subgroups(
    path: Path,
    *,
    protocol: ExperimentProtocol,
    expected_manifest_sha256: str,
) -> SubgroupArtifact:
    try:
        return load_subgroup_artifact(
            path,
            protocol=protocol,
            expected_manifest_sha256=expected_manifest_sha256,
            verify_source=True,
        )
    except (OSError, SubgroupArtifactError, RuntimeError, ValueError) as error:
        raise FinalEvaluationSpecIntegrityError(
            f"could not verify subgroup artifact: {error}"
        ) from error


def _subgroup_binding(path: Path, artifact: SubgroupArtifact) -> dict[str, object]:
    if artifact.artifact_sha256 is None:
        raise FinalEvaluationSpecIntegrityError("subgroup artifact is not self-hashed")
    payload = artifact.to_payload()
    definitions = _mapping(payload["definitions"], "subgroup definitions")
    summary = _mapping(payload["summary"], "subgroup summary")
    if dict(definitions) != _canonical_subgroup_definitions():
        raise FinalEvaluationSpecIntegrityError(
            "verified subgroup definitions are not canonical"
        )
    return {
        "path": str(path.resolve()),
        "file_sha256": _file_sha256(path),
        "artifact_sha256": artifact.artifact_sha256,
        "manifest_sha256": artifact.manifest_sha256,
        "folds": list(FINAL_TEST_FOLDS),
        "attributes": list(SUBGROUP_ATTRIBUTES),
        "definitions": dict(definitions),
        "counts": dict(summary),
        "diagnostic_target_columns_read": False,
    }


def _deviation_binding(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "file_sha256": _file_sha256(path),
        "required_in_final_reporting": True,
    }


def _validate_payload(root: Mapping[str, object]) -> None:
    _exact_keys(
        root,
        {
            "schema_version",
            "artifact_type",
            "protocol",
            "refit_bundle",
            "subgroup_artifact",
            "protocol_deviations",
            "calibration_policy",
            "final_evaluation",
            "comparison_policy",
            "report_contract",
            "runtime_envelope",
            "artifact_sha256",
        },
        "final evaluation specification",
    )
    if root["schema_version"] != FINAL_EVALUATION_SPEC_SCHEMA_VERSION:
        raise FinalEvaluationSpecIntegrityError("unsupported specification schema")
    if root["artifact_type"] != FINAL_EVALUATION_SPEC_TYPE:
        raise FinalEvaluationSpecIntegrityError("unexpected specification type")

    protocol = _mapping(root["protocol"], "protocol")
    _exact_keys(
        protocol,
        {
            "source_path",
            "source_file_sha256",
            "protocol_hash",
            "dataset_name",
            "dataset_version",
            "resolved",
        },
        "protocol",
    )
    _string(protocol["source_path"], "protocol.source_path")
    _hash(protocol["source_file_sha256"], "protocol.source_file_sha256")
    _hash(protocol["protocol_hash"], "protocol.protocol_hash")
    _string(protocol["dataset_name"], "protocol.dataset_name")
    _string(protocol["dataset_version"], "protocol.dataset_version")
    _mapping(protocol["resolved"], "protocol.resolved")

    refit = _mapping(root["refit_bundle"], "refit_bundle")
    _exact_keys(
        refit,
        {
            "path",
            "file_sha256",
            "artifact_sha256",
            "protocol_hash",
            "manifest_sha256",
            "normalization_sha256",
            "label_order",
            "member_count",
        },
        "refit_bundle",
    )
    _string(refit["path"], "refit_bundle.path")
    for field in (
        "file_sha256",
        "artifact_sha256",
        "protocol_hash",
        "manifest_sha256",
        "normalization_sha256",
    ):
        _hash(refit[field], f"refit_bundle.{field}")
    if _integer(refit["member_count"], "refit member_count", minimum=1) != 6:
        raise FinalEvaluationSpecIntegrityError("refit member_count must be six")
    labels = _sequence(refit["label_order"], "refit label_order")
    if not labels or not all(isinstance(label, str) and label for label in labels):
        raise FinalEvaluationSpecIntegrityError("refit label order is invalid")
    if refit["protocol_hash"] != protocol["protocol_hash"]:
        raise FinalEvaluationSpecIntegrityError("refit/protocol hashes differ")
    resolved_protocol = _mapping(protocol["resolved"], "protocol.resolved")
    task = _mapping(resolved_protocol.get("task"), "protocol.resolved.task")
    resolved_labels = list(
        _sequence(task.get("label_order"), "protocol resolved label_order")
    )
    if list(labels) != resolved_labels:
        raise FinalEvaluationSpecIntegrityError("refit/protocol label orders differ")

    subgroup = _mapping(root["subgroup_artifact"], "subgroup_artifact")
    _exact_keys(
        subgroup,
        {
            "path",
            "file_sha256",
            "artifact_sha256",
            "manifest_sha256",
            "folds",
            "attributes",
            "definitions",
            "counts",
            "diagnostic_target_columns_read",
        },
        "subgroup_artifact",
    )
    _string(subgroup["path"], "subgroup_artifact.path")
    for field in ("file_sha256", "artifact_sha256", "manifest_sha256"):
        _hash(subgroup[field], f"subgroup_artifact.{field}")
    if list(_sequence(subgroup["folds"], "subgroup folds")) != list(FINAL_TEST_FOLDS):
        raise FinalEvaluationSpecIntegrityError("subgroups must remain fold 10 only")
    if list(_sequence(subgroup["attributes"], "subgroup attributes")) != list(
        SUBGROUP_ATTRIBUTES
    ):
        raise FinalEvaluationSpecIntegrityError("subgroup attributes are not canonical")
    definitions = _mapping(subgroup["definitions"], "subgroup definitions")
    if dict(definitions) != _canonical_subgroup_definitions():
        raise FinalEvaluationSpecIntegrityError("subgroup definitions are not canonical")
    counts = _mapping(subgroup["counts"], "subgroup counts")
    _exact_keys(counts, {"record_count", "patient_count", "groups"}, "subgroup counts")
    record_count = _integer(counts["record_count"], "subgroup record_count", minimum=1)
    patient_count = _integer(
        counts["patient_count"], "subgroup patient_count", minimum=1
    )
    raw_groups = _sequence(counts["groups"], "subgroup groups")
    expected_groups = [
        *(('sex', group) for group in ("male", "female", "unknown")),
        *(('age_band', group) for group in ("<40", "40-59", "60-79", "80+", "unknown")),
    ]
    if len(raw_groups) != len(expected_groups):
        raise FinalEvaluationSpecIntegrityError("subgroup count grid is not canonical")
    observed_groups: list[tuple[str, str]] = []
    totals = {"sex": 0, "age_band": 0}
    patient_totals = {"sex": 0, "age_band": 0}
    for raw_group in raw_groups:
        group = _mapping(raw_group, "subgroup group count")
        _exact_keys(
            group,
            {"attribute", "group", "records", "patients"},
            "subgroup group count",
        )
        attribute = _string(group["attribute"], "subgroup count attribute")
        name = _string(group["group"], "subgroup count group")
        if attribute not in totals:
            raise FinalEvaluationSpecIntegrityError("subgroup count attribute is invalid")
        records = _integer(group["records"], "subgroup group records")
        patients = _integer(group["patients"], "subgroup group patients")
        if records > record_count or patients > patient_count or patients > records:
            raise FinalEvaluationSpecIntegrityError("subgroup group counts are impossible")
        observed_groups.append((attribute, name))
        totals[attribute] += records
        patient_totals[attribute] += patients
    if observed_groups != expected_groups or any(
        totals[attribute] != record_count
        or patient_totals[attribute] != patient_count
        for attribute in totals
    ):
        raise FinalEvaluationSpecIntegrityError("subgroup count grid is not canonical")
    if subgroup["diagnostic_target_columns_read"] is not False:
        raise FinalEvaluationSpecIntegrityError("subgroup freeze must remain label-free")
    if subgroup["manifest_sha256"] != refit["manifest_sha256"]:
        raise FinalEvaluationSpecIntegrityError("subgroup/refit manifest hashes differ")

    deviation = _mapping(root["protocol_deviations"], "protocol_deviations")
    _exact_keys(
        deviation,
        {"path", "file_sha256", "required_in_final_reporting"},
        "protocol_deviations",
    )
    _string(deviation["path"], "protocol_deviations.path")
    _hash(deviation["file_sha256"], "protocol_deviations.file_sha256")
    if deviation["required_in_final_reporting"] is not True:
        raise FinalEvaluationSpecIntegrityError("protocol deviations must be reported")

    if dict(_mapping(root["calibration_policy"], "calibration_policy")) != (
        _calibration_policy()
    ):
        raise FinalEvaluationSpecIntegrityError("calibration policy is not canonical")
    if dict(_mapping(root["final_evaluation"], "final_evaluation")) != (
        _evaluation_policy()
    ):
        raise FinalEvaluationSpecIntegrityError("final evaluation policy is not canonical")
    if dict(_mapping(root["comparison_policy"], "comparison_policy")) != (
        _comparison_policy()
    ):
        raise FinalEvaluationSpecIntegrityError("comparison policy is not canonical")
    if dict(_mapping(root["report_contract"], "report_contract")) != _report_contract():
        raise FinalEvaluationSpecIntegrityError("report contract is not canonical")
    _validate_runtime_envelope(root["runtime_envelope"])

    stored_hash = _hash(root["artifact_sha256"], "artifact_sha256")
    unhashed = dict(root)
    del unhashed["artifact_sha256"]
    if canonical_sha256(unhashed) != stored_hash:
        raise FinalEvaluationSpecIntegrityError("specification self-hash mismatch")


def _spec_from_payload(
    payload: Mapping[str, object], *, path: Path | None
) -> FinalEvaluationSpec:
    _validate_payload(payload)
    try:
        canonical = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:  # pragma: no cover - validation invariant
        raise FinalEvaluationSpecIntegrityError("spec payload is not finite JSON") from error
    return FinalEvaluationSpec(
        path=path.resolve() if path is not None else None,
        artifact_sha256=cast(str, payload["artifact_sha256"]),
        _canonical_payload=canonical,
    )


def create_final_evaluation_spec(
    *,
    protocol: ExperimentProtocol,
    protocol_path: str | Path,
    refit_bundle_path: str | Path,
    subgroup_artifact_path: str | Path,
    protocol_deviations_path: str | Path,
    project_root: str | Path,
    device: str,
) -> FinalEvaluationSpec:
    """Verify all pre-fold-9 sources and create a deterministic specification."""

    if not isinstance(protocol, ExperimentProtocol):
        raise TypeError("protocol must be an ExperimentProtocol")
    protocol_source = Path(protocol_path).resolve()
    refit_path = Path(refit_bundle_path).resolve()
    subgroup_path = Path(subgroup_artifact_path).resolve()
    deviation_path = Path(protocol_deviations_path).resolve()
    resolved_project = Path(project_root).resolve()
    protocol_binding = _verify_protocol_source(protocol_source, protocol)
    refit_bundle = _load_refit(refit_path, protocol)
    refit_binding = _refit_binding(refit_path, refit_bundle)
    if refit_bundle.protocol_hash != protocol.protocol_hash:
        raise FinalEvaluationSpecIntegrityError("refit bundle protocol differs")
    subgroup = _load_subgroups(
        subgroup_path,
        protocol=protocol,
        expected_manifest_sha256=refit_bundle.manifest_sha256,
    )
    subgroup_binding = _subgroup_binding(subgroup_path, subgroup)
    runtime = _capture_runtime_envelope(resolved_project, device)
    body: dict[str, object] = {
        "schema_version": FINAL_EVALUATION_SPEC_SCHEMA_VERSION,
        "artifact_type": FINAL_EVALUATION_SPEC_TYPE,
        "protocol": protocol_binding,
        "refit_bundle": refit_binding,
        "subgroup_artifact": subgroup_binding,
        "protocol_deviations": _deviation_binding(deviation_path),
        "calibration_policy": _calibration_policy(),
        "final_evaluation": _evaluation_policy(),
        "comparison_policy": _comparison_policy(),
        "report_contract": _report_contract(),
        "runtime_envelope": runtime,
    }
    payload = dict(body)
    payload["artifact_sha256"] = canonical_sha256(body)
    return _spec_from_payload(payload, path=None)


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.suffix.casefold() != ".json":
        raise FinalEvaluationSpecError("specification path must end in .json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable specification already exists: {path}")
    try:
        serialized = (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    except (TypeError, ValueError) as error:
        raise FinalEvaluationSpecError("specification must be finite JSON") from error
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"immutable specification already exists: {path}"
            ) from error
    finally:
        with suppress(OSError):
            temporary.unlink()


def save_final_evaluation_spec(
    spec: FinalEvaluationSpec, path: str | Path
) -> FinalEvaluationSpec:
    """Save a new specification without overwrite and return its bound path."""

    if not isinstance(spec, FinalEvaluationSpec):
        raise TypeError("spec must be a FinalEvaluationSpec")
    payload = spec.to_payload()
    _validate_payload(payload)
    destination = Path(path).resolve()
    _write_new_json(destination, payload)
    return _spec_from_payload(payload, path=destination)


def _verify_bound_sources(
    spec: FinalEvaluationSpec,
    *,
    protocol: ExperimentProtocol,
) -> None:
    payload = spec.payload
    protocol_binding = _mapping(payload["protocol"], "protocol")
    protocol_path = Path(_string(protocol_binding["source_path"], "protocol path"))
    if _verify_protocol_source(protocol_path, protocol) != dict(protocol_binding):
        raise FinalEvaluationSpecIntegrityError("protocol source changed after freeze")

    refit_binding = _mapping(payload["refit_bundle"], "refit_bundle")
    refit_path = Path(_string(refit_binding["path"], "refit bundle path"))
    refit = _load_refit(refit_path, protocol)
    if _refit_binding(refit_path, refit) != dict(refit_binding):
        raise FinalEvaluationSpecIntegrityError("refit bundle changed after freeze")

    subgroup_binding = _mapping(payload["subgroup_artifact"], "subgroup_artifact")
    subgroup_path = Path(_string(subgroup_binding["path"], "subgroup path"))
    subgroup = _load_subgroups(
        subgroup_path,
        protocol=protocol,
        expected_manifest_sha256=refit.manifest_sha256,
    )
    if _subgroup_binding(subgroup_path, subgroup) != dict(subgroup_binding):
        raise FinalEvaluationSpecIntegrityError("subgroup artifact changed after freeze")

    deviation = _mapping(payload["protocol_deviations"], "protocol_deviations")
    deviation_path = Path(_string(deviation["path"], "protocol deviations path"))
    if _deviation_binding(deviation_path) != dict(deviation):
        raise FinalEvaluationSpecIntegrityError("protocol deviations changed after freeze")


def load_final_evaluation_spec(
    path: str | Path,
    *,
    protocol: ExperimentProtocol,
    verify_sources: bool = True,
    verify_runtime: bool = True,
) -> FinalEvaluationSpec:
    """Load a strict specification and optionally reverify every bound input."""

    source = Path(path).resolve()
    spec = _spec_from_payload(_read_json(source, "final evaluation specification"), path=source)
    protocol_binding = _mapping(spec.payload["protocol"], "protocol")
    if protocol_binding["protocol_hash"] != protocol.protocol_hash or protocol_binding[
        "resolved"
    ] != protocol.to_resolved_dict():
        raise FinalEvaluationSpecIntegrityError(
            "specification protocol differs from the supplied protocol"
        )
    if verify_sources:
        _verify_bound_sources(spec, protocol=protocol)
    if verify_runtime:
        frozen_runtime = _mapping(spec.payload["runtime_envelope"], "runtime_envelope")
        project_root = Path(
            _string(frozen_runtime["project_root"], "runtime project_root")
        )
        hardware = _mapping(frozen_runtime["hardware"], "runtime hardware")
        requested_device = _string(hardware["requested_device"], "requested_device")
        current = _capture_runtime_envelope(project_root, requested_device)
        if current != dict(frozen_runtime):
            raise FinalEvaluationSpecIntegrityError(
                "current committed CUDA runtime differs from the frozen envelope"
            )
    return spec


def freeze_final_evaluation_spec(
    output_path: str | Path,
    *,
    protocol: ExperimentProtocol,
    protocol_path: str | Path,
    refit_bundle_path: str | Path,
    subgroup_artifact_path: str | Path,
    protocol_deviations_path: str | Path,
    project_root: str | Path,
    device: str,
) -> FinalEvaluationSpec:
    """Create and atomically freeze one specification without overwriting."""

    created = create_final_evaluation_spec(
        protocol=protocol,
        protocol_path=protocol_path,
        refit_bundle_path=refit_bundle_path,
        subgroup_artifact_path=subgroup_artifact_path,
        protocol_deviations_path=protocol_deviations_path,
        project_root=project_root,
        device=device,
    )
    saved = save_final_evaluation_spec(created, output_path)
    return load_final_evaluation_spec(
        cast(Path, saved.path),
        protocol=protocol,
        verify_sources=True,
        # The clean committed runtime was captured immediately before the write.
        # Writing an untracked output inside the repository may itself make Git
        # dirty, so probing again here could strand a valid immutable artifact.
        verify_runtime=False,
    )


__all__ = [
    "CANONICAL_COVERAGE_TARGETS",
    "FINAL_EVALUATION_SPEC_SCHEMA_VERSION",
    "FINAL_EVALUATION_SPEC_TYPE",
    "FinalEvaluationSpec",
    "FinalEvaluationSpecError",
    "FinalEvaluationSpecIntegrityError",
    "canonical_sha256",
    "create_final_evaluation_spec",
    "freeze_final_evaluation_spec",
    "load_final_evaluation_spec",
    "save_final_evaluation_spec",
]
