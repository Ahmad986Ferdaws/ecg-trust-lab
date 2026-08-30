"""One-shot external OOD v2 execution against the exact sealed v1 policy.

This module deliberately has no fitting, threshold, method-selection, or target
adaptation API.  The only model score accepted by the execution path is the
score produced by the byte-bound v1 distribution policy.  The authoritative v1
whole-bundle verifier touches private v1 bytes solely to establish integrity;
this module never exposes, subsets, scores, or analyzes those private bytes.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from collections import Counter
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType, ModuleType
from typing import Any, Final, cast
from urllib.parse import urlsplit

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from ecg_trust.conformal import BinaryDecision, LabelwiseBinaryConformal
from ecg_trust.constants import LEADS, SUPERCLASSES, TARGET_COLUMNS
from ecg_trust.data.dataset import NormalizationStats
from ecg_trust.demo_backend import FrozenDecisionPolicy
from ecg_trust.experiment_config import ModelConfig
from ecg_trust.experiment_runner import build_experiment_model
from ecg_trust.foundation.adapter import model_state_sha256
from ecg_trust.models.resnet1d import ResNet1D
from ecg_trust.ood_completion.models import (
    DistributionPolicy,
    OODCompletionResult,
    OODCompletionSuccessManifest,
    load_distribution_policy_bytes,
    load_ood_completion_result_bytes,
)
from ecg_trust.ood_completion.pipeline import verify_ood_completion_bundle
from ecg_trust.ood_completion.runtime import (
    DeterministicCUDARuntime,
    configure_deterministic_cuda,
    extract_embeddings_twice,
    prepare_resnet_for_embedding,
)
from ecg_trust.ood_v2.adapters import (
    ADAPTER_VERSION,
    PHYSICAL_UNITS,
    RESAMPLE_PADTYPE,
    RESAMPLE_WINDOW,
    TARGET_FREQUENCY_HZ,
    TARGET_SAMPLES,
    WINDOW_SECONDS,
    AdapterProvenance,
    CanonicalExternalSignal,
    ExternalECGAdapterError,
    load_challenge_2011_signal,
    load_zzu_pediatric_signal,
)
from ecg_trust.ood_v2.bundle import (
    ACCESS_CLAIM_ARTIFACT_TYPE,
    ACCESS_MARKER_ARTIFACT_TYPE,
    ACCESS_MARKER_FILENAME,
    CANONICAL_SIGNAL_MEMBER_MAX_BYTES,
    CANONICAL_SIGNAL_NPZ_PATH,
    CANONICAL_SIGNAL_SHARD_COUNT,
    CANONICAL_SIGNAL_SHARD_RECORDS,
    CANONICAL_SIGNAL_SIDECAR_PATH,
    FAILURE_RECEIPT_FILENAME,
    QUALITY_AUDIT_EXPECTED_RECORDS,
    QUALITY_AUDIT_INDEX_PATH,
    QUALITY_AUDIT_SHARD_COUNT,
    QUALITY_AUDIT_SHARD_MAX_BYTES,
    QUALITY_AUDIT_SHARD_PATHS,
    QUALITY_AUDIT_SHARD_RECORDS,
    SUCCESS_MANIFEST_FILENAME,
    build_success_manifest,
    canonical_json_bytes,
    canonical_sha256,
    preverify_external_v2_bundle,
    sha256_bytes,
    sha256_file,
    verify_external_v2_bundle,
)
from ecg_trust.ood_v2.inventory import (
    CHALLENGE_2011_DATASET,
    CHALLENGE_2011_VERSION,
    ZZU_PEDIATRIC_DATASET,
    ZZU_PEDIATRIC_VERSION,
    ArchiveExtractionClosure,
    ExternalInventoryRecord,
    ExternalWaveformInventory,
    SevenZipToolBinding,
    build_external_inventory,
    external_inventory_public_projection,
    load_external_inventory,
    parse_challenge_2011_quality_lists,
    parse_zzu_pediatric_attributes_csv,
    resolve_inventory_record_base,
    select_zzu_pediatric_inventory_records,
    validate_challenge_2011_set_a_inventory,
    verify_challenge_tar_extraction_closure,
    verify_external_inventory,
    verify_seven_zip_tool_binding,
    verify_wfdb_candidate_file_set,
    verify_zzu_split_zip_extraction_closure,
)
from ecg_trust.ood_v2.models import (
    OOD_V2_ARTIFACT_TYPE,
    OOD_V2_RESULT_FILENAME,
    AggregateRouteCounts,
    EvidenceRequirements,
    ExternalCohortRole,
    ExternalOODHardGates,
    HistoricalSourceBootstrapInterval,
    OODAxis,
    OODV2IntegritySummary,
    OODV2Result,
    OODV2ResultBody,
    OODV2Status,
    ResamplingUnit,
    SourceGateSummary,
    load_ood_v2_result_bytes,
    ood_v2_result_json_bytes,
    seal_ood_v2_result,
)
from ecg_trust.ood_v2.statistics import (
    evaluate_external_ood_gate,
    evaluate_technical_quality_gate,
)
from ecg_trust.open_world import normalized_bernoulli_entropy
from ecg_trust.quality.signal_quality import (
    DEFAULT_SIGNAL_QUALITY_CONFIG,
    QualityStatus,
    ReasonCode,
    SignalMetadata,
    SignalQualityReport,
    assess_signal_quality,
)
from ecg_trust.source_calibration.models import SourceCalibrationResult
from ecg_trust.source_calibration.pipeline import load_source_calibration_result_bytes

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
Int64Array = NDArray[np.int64]

ORIGINAL_PROTOCOL_ID: Final = "trust-sentinel-ood-external-v2-parent"
PROTOCOL_ID: Final = "trust-sentinel-ood-external-v2-1-parent"
PARENT_CONFIG_DEFAULT: Final = "configs/trust_sentinel_ood_external_v2.yaml"
SUCCESSOR_PARENT_CONFIG_PATH: Final = "configs/trust_sentinel_ood_external_v2_1.yaml"
SUCCESSOR_CHILD_CONFIG_PATH: Final = (
    "configs/trust_sentinel_ood_external_v2_1_execution.json"
)
SUCCESSOR_PRIVATE_INVENTORY_PATH: Final = (
    "artifacts/trust_sentinel/ood_external_v2_1_preflight/private/"
    "external-waveform-inventory.json"
)
SUCCESSOR_PUBLIC_PROJECTION_PATH: Final = (
    "artifacts/trust_sentinel/ood_external_v2_1_preflight/public/"
    "external-inventory-summary.json"
)
EXPECTED_PARENT_CONFIG_SHA256: Final = (
    "sha256:3aacb31be939d1a2bea96bb29f193d60a4b54c38a40a1a7e2a490cfe60c3b0d9"
)
EXPECTED_SUCCESSOR_PARENT_CONFIG_SHA256: Final[str | None] = (
    "sha256:2b6696d07c1fbab1e31eccb3d8d48fdc6251d12301df6ca604b8af1d02b7dd10"
)
SUCCESSOR_PROTOCOL_ID: Final = PROTOCOL_ID
PREDECESSOR_TERMINATION_PATH: Final = (
    "configs/trust_sentinel_ood_external_v2_termination.yaml"
)
PREDECESSOR_TERMINATION_FILE_SHA256: Final = (
    "sha256:289736acc200be025075c0f4094ac6d1719b4e50d55dfcf3b628b287158e240a"
)
PREDECESSOR_TERMINATION_NOTE_PATH: Final = (
    "docs/TRUST_SENTINEL_OOD_EXTERNAL_V2_INFEASIBILITY.md"
)
PREDECESSOR_TERMINATION_NOTE_FILE_SHA256: Final = (
    "sha256:5d16d2ebddf15acc8a5915ba274ba565bc8a35ac9b3bcdf99ccd9a03a6eedc89"
)
PREDECESSOR_PREFLIGHT_PRIVATE_PATH: Final = (
    "artifacts/trust_sentinel/ood_external_v2_preflight/private/"
    "external-waveform-inventory.json"
)
PREDECESSOR_PREFLIGHT_PRIVATE_FILE_SHA256: Final = (
    "sha256:01b33b992c3e9a777eb253571f35907e8aa99e3d36b34a19a3c374e7732aef13"
)
PREDECESSOR_PREFLIGHT_PUBLIC_PATH: Final = (
    "artifacts/trust_sentinel/ood_external_v2_preflight/public/"
    "external-inventory-summary.json"
)
PREDECESSOR_PREFLIGHT_PUBLIC_FILE_SHA256: Final = (
    "sha256:8de32fce76e73fd00958878e74250c9117cdae7e3f4d9f453f2231fcf16a5814"
)
PREDECESSOR_OUTPUT_PATH: Final = "artifacts/trust_sentinel/ood_external_v2"
PREDECESSOR_CLAIM_PATH: Final = (
    "artifacts/trust_sentinel/.ood_external_v2.one-shot-claim.json"
)
SUCCESSOR_OUTPUT_PATH: Final = "artifacts/trust_sentinel/ood_external_v2_1"
SUCCESSOR_CLAIM_PATH: Final = (
    "artifacts/trust_sentinel/.ood_external_v2_1.one-shot-claim.json"
)
HISTORICAL_X4_INVENTORY_BUILDER_ATTEMPT_PATH: Final = (
    "artifacts/trust_sentinel/.ood_external_v2_1.x4-inventory-build-attempt.json"
)
HISTORICAL_X5_INVENTORY_BUILDER_ATTEMPT_PATH: Final = (
    "artifacts/trust_sentinel/.ood_external_v2_1.x5-inventory-build-attempt.json"
)
HISTORICAL_X6_INVENTORY_BUILDER_ATTEMPT_PATH: Final = (
    "artifacts/trust_sentinel/.ood_external_v2_1.x6-inventory-build-attempt.json"
)
HISTORICAL_X6_INVENTORY_BUILDER_ATTEMPT_FILE_SHA256: Final = (
    "sha256:4e3e968a2dc9f0c7f552bc05f8d70ef6afc99d97b5b81a60c2920e064efbe9e8"
)
HISTORICAL_X6_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_SHA256: Final = (
    "sha256:88fb0a119f5c550f352cc2dca6f181567e0dd660449eb0ddd3c6247a7884cf93"
)
HISTORICAL_X7_INVENTORY_BUILDER_ATTEMPT_PATH: Final = (
    "artifacts/trust_sentinel/.ood_external_v2_1.x7-inventory-build-attempt.json"
)
HISTORICAL_X7_INVENTORY_BUILDER_ATTEMPT_FILE_SHA256: Final = (
    "sha256:8255b58e5c63a4e18ae2a0b7715109106e4f1a949cb68204b66cde9f1fd4af01"
)
HISTORICAL_X7_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_SHA256: Final = (
    "sha256:ec01f1554da7733a4d298161a3d67818dc1edd2fefb663760277070359354830"
)
HISTORICAL_X7_INVENTORY_BUILDER_FAILURE_PATH: Final = (
    "artifacts/trust_sentinel/.ood_external_v2_1.x7-inventory-build-failure.json"
)
HISTORICAL_X7_INVENTORY_BUILDER_FAILURE_FILE_SHA256: Final = (
    "sha256:af6995828daad64a6606dfd1875a2ced6daa9ac390e328152488270f5dcffac6"
)
HISTORICAL_X7_INVENTORY_BUILDER_FAILURE_ARTIFACT_SHA256: Final = (
    "sha256:02c2d212a1ff4108c9dd10bd67095a94727d5aabb68e0c1e94a2ed9d4304d7d3"
)
HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_PATH: Final = (
    "artifacts/trust_sentinel/.ood_external_v2_1.x8-inventory-build-attempt.json"
)
HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_FILE_SHA256: Final = (
    "sha256:3b9f418f63bc9d868f338af50f8c98635a32d701070faccec6c38363d2067aeb"
)
HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_SHA256: Final = (
    "sha256:58d5c5785d30f5708ee65b387ba623aa0d6d6dd382fd51dd95c0b9690763f2b6"
)
HISTORICAL_X8_INVENTORY_BUILDER_PROJECT_SOURCE_TREE_SHA256: Final = (
    "sha256:2d28663c1af1e10ccb46b91b30a3715fecd3911e17020d24243ba2ddd04f1976"
)
HISTORICAL_X8_PRIVATE_INVENTORY_FILE_SHA256: Final = (
    "sha256:424e0b09f3cb700e95ca202af7e42bc676ec903b74ac2cfaff3769444038f59c"
)
HISTORICAL_X8_PRIVATE_INVENTORY_ARTIFACT_SHA256: Final = (
    "sha256:c58f46fcfc34b711a0c847fcdc48fe0c1b92b8bcb31d9d94fdc950169eb1cb24"
)
HISTORICAL_X8_PUBLIC_PROJECTION_FILE_SHA256: Final = (
    "sha256:02ad715dc9db3b92e3a4a6e32193d07f8c1f37c3de5668186eb6790bc75fcdcc"
)
HISTORICAL_X8_PUBLIC_PROJECTION_ARTIFACT_SHA256: Final = (
    "sha256:e7d91f13000dc5b5fa5bd874f72ef4ba9e81e13446acd2d4673efb7e77dcace6"
)
# The inventory builder is permanently retired after the successful X8 run.
# This compatibility name remains because the child schema historically used
# it; every X9/Y verifier treats the target as immutable X8 evidence.
SUCCESSOR_INVENTORY_BUILDER_ATTEMPT_PATH: Final = (
    HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_PATH
)
SUCCESSOR_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_TYPE: Final = (
    "ecg_trust.ood_external_v2_1_inventory_builder_attempt"
)
HISTORICAL_X8_INVENTORY_BUILDER_FAILURE_PATH: Final = (
    "artifacts/trust_sentinel/.ood_external_v2_1.x8-inventory-build-failure.json"
)
SUCCESSOR_INVENTORY_BUILDER_FAILURE_PATH: Final = (
    HISTORICAL_X8_INVENTORY_BUILDER_FAILURE_PATH
)
SUCCESSOR_INVENTORY_BUILDER_FAILURE_ARTIFACT_TYPE: Final = (
    "ecg_trust.ood_external_v2_1_inventory_builder_failure"
)
HISTORICAL_X9_CHILD_FREEZE_ATTEMPT_PATH: Final = (
    "artifacts/trust_sentinel/.ood_external_v2_1.x9-child-freeze-attempt.json"
)
HISTORICAL_X9_CHILD_FREEZE_ATTEMPT_FILE_SHA256: Final = (
    "sha256:507eb238000c6a2485e58801d255ae2efae00b828b1ab93ee5a14de4d941b9bb"
)
HISTORICAL_X9_CHILD_FREEZE_ATTEMPT_ARTIFACT_SHA256: Final = (
    "sha256:46f1eb12f6dd145ead167a05b4b445353074fea8362961040063fc7b209872a2"
)
HISTORICAL_X9_CHILD_FREEZE_PROJECT_SOURCE_TREE_SHA256: Final = (
    "sha256:98b4ff7a7bfe30cf228494ddc14b6f40e6d4f1b542b8401ba951868d6a6e90b9"
)
HISTORICAL_X9_PYTHON_ENVIRONMENT_SHA256: Final = (
    "sha256:d834e2cf3e6cf1ec7fbf09607cb6fb8b5a05824dfdfba15445e2e5dad74c9188"
)
HISTORICAL_X9_CHILD_FROZEN_AT_UTC: Final = "2026-08-30T10:48:55Z"
HISTORICAL_X9_CHILD_FREEZE_FAILURE_PATH: Final = (
    "artifacts/trust_sentinel/.ood_external_v2_1.x9-child-freeze-failure.json"
)
HISTORICAL_X9_CHILD_FREEZE_FAILURE_FILE_SHA256: Final = (
    "sha256:c17e1271c2c799ef4816dcb9c41557f0455054562e1c9cf253b3897b4a75e296"
)
HISTORICAL_X9_CHILD_FREEZE_FAILURE_ARTIFACT_SHA256: Final = (
    "sha256:eaf06d21140c7f69f251ab4c288920dc7b9f2331939f4cae7c24aca389ab8aa2"
)
SUCCESSOR_CHILD_FREEZE_ATTEMPT_PATH: Final = (
    "artifacts/trust_sentinel/.ood_external_v2_1.x10-child-freeze-attempt.json"
)
SUCCESSOR_CHILD_FREEZE_ATTEMPT_ARTIFACT_TYPE: Final = (
    "ecg_trust.ood_external_v2_1_child_freeze_attempt"
)
SUCCESSOR_CHILD_FREEZE_FAILURE_PATH: Final = (
    "artifacts/trust_sentinel/.ood_external_v2_1.x10-child-freeze-failure.json"
)
SUCCESSOR_CHILD_FREEZE_FAILURE_ARTIFACT_TYPE: Final = (
    "ecg_trust.ood_external_v2_1_child_freeze_failure"
)
FORBIDDEN_GIT_HISTORY_PATHS: Final[tuple[str, ...]] = (
    ":(glob)data/raw/external-ood/**",
    ":(glob)artifacts/trust_sentinel/ood_external_v2_preflight/private/**",
    ":(glob)artifacts/trust_sentinel/ood_external_v2_1_preflight/private/**",
    PREDECESSOR_OUTPUT_PATH,
    SUCCESSOR_OUTPUT_PATH,
    PREDECESSOR_CLAIM_PATH,
    SUCCESSOR_CLAIM_PATH,
    HISTORICAL_X4_INVENTORY_BUILDER_ATTEMPT_PATH,
    HISTORICAL_X5_INVENTORY_BUILDER_ATTEMPT_PATH,
    HISTORICAL_X6_INVENTORY_BUILDER_ATTEMPT_PATH,
    HISTORICAL_X7_INVENTORY_BUILDER_ATTEMPT_PATH,
    HISTORICAL_X7_INVENTORY_BUILDER_FAILURE_PATH,
    SUCCESSOR_INVENTORY_BUILDER_ATTEMPT_PATH,
    SUCCESSOR_INVENTORY_BUILDER_FAILURE_PATH,
    HISTORICAL_X9_CHILD_FREEZE_ATTEMPT_PATH,
    HISTORICAL_X9_CHILD_FREEZE_FAILURE_PATH,
    SUCCESSOR_CHILD_FREEZE_ATTEMPT_PATH,
    SUCCESSOR_CHILD_FREEZE_FAILURE_PATH,
    ":(glob)artifacts/trust_sentinel/.ood_external_v2.staging-*/**",
    ":(glob)artifacts/trust_sentinel/.ood_external_v2_1.staging-*/**",
    ":(glob)artifacts/trust_sentinel/.ood_external_v2_1.runtime-*/**",
)
EXPECTED_SOURCE_CALIBRATION_FILE_SHA256: Final = (
    "sha256:8bae3acdebac42504167afc7bb7d2051b7ac2c48019aa429ed6544f14a59f38f"
)
EXPECTED_SOURCE_CALIBRATION_PATH: Final = (
    "artifacts/trust_sentinel/source_calibration_v1/"
    "source-calibration-result.json"
)
EXPECTED_DEMO_POLICY_FILE_SHA256: Final = (
    "sha256:539d9e7dfc84edc49ab285775cdd0f6e93b2f5bb804c6fe7be7d00bc2aff4d42"
)
EXPECTED_DEMO_POLICY_PATH: Final = (
    "artifacts/demo/ptbxl_matched_equal_budget_v1/"
    "resnet1d-seed2026.coverage80.demo-policy.json"
)
EXPECTED_DISTRIBUTION_POLICY_ARTIFACT_SHA256: Final = (
    "sha256:d544c28ad18b764e3e30cc316b092a41d75125a8334f1d41ed58c31ec37568db"
)
EXPECTED_SOURCE_CALIBRATION_ARTIFACT_SHA256: Final = (
    "sha256:b9063fd2965b194806f9e544f3ea6390cc19bc8a93b27d3e88a674bf0aa7c839"
)
EXPECTED_DISTRIBUTION_THRESHOLD: Final = 270.9668613705653
EXPECTED_TEMPERATURE: Final = 1.319620052379425
EXPECTED_ENTROPY_MAXIMUM: Final = 0.5975748221759414
EXPECTED_CONFORMAL_THRESHOLDS: Final[tuple[float, ...]] = (
    0.5943209280399379,
    0.5720188933250413,
    0.5936070307854548,
    0.5512516601251809,
    0.48857772684700007,
)
EXPECTED_CHECKPOINT_FILE_SHA256: Final = (
    "sha256:d3b8a19ab891db34afa6039179edab9847a8812e466a65c4cb408df12b402b35"
)
EXPECTED_RESOLVED_CONFIG_FILE_SHA256: Final = (
    "sha256:d00643dadc1c27a241da5c100bccd45f314fe0b94b16c9e3ce9d88ed22656d49"
)
EXPECTED_RESOLVED_CONFIG_SHA256: Final = (
    "sha256:003125474caa877585e609b7b248727aa3ecaf7c716d8c249966ff4b9188e71e"
)
CHILD_CONTRACT_ARTIFACT_TYPE: Final = "ecg_trust.ood_external_v2_1_child_contract"
PRIVATE_EVIDENCE_ARTIFACT_TYPE: Final = "ecg_trust.ood_external_v2_1_record_evidence"
PRIVATE_EMBEDDING_ARTIFACT_TYPE: Final = "ecg_trust.ood_external_v2_1_embeddings"
PRIVATE_BOOTSTRAP_ARTIFACT_TYPE: Final = "ecg_trust.ood_external_v2_1_bootstrap_replicates"
PRIVATE_QUALITY_AUDIT_ARTIFACT_TYPE: Final = (
    "ecg_trust.ood_external_v2_1_quality_audit_shard"
)
PRIVATE_QUALITY_AUDIT_INDEX_ARTIFACT_TYPE: Final = (
    "ecg_trust.ood_external_v2_1_quality_audit_index"
)
PRIVATE_CANONICAL_SIGNAL_ARTIFACT_TYPE: Final = (
    "ecg_trust.ood_external_v2_1_canonical_signals"
)
PRIVATE_ROUTING_CONTRACT_ARTIFACT_TYPE: Final = (
    "ecg_trust.ood_external_v2_1_routing_contract"
)
_QUALITY_REPORT_DOMAIN: Final = b"ecg_trust.ood_external_v2_1.quality_report.v1\x00"
FAILURE_ARTIFACT_TYPE: Final = "ecg_trust.ood_external_v2_1_failure"

_CONFIG_MAX_BYTES = 1_000_000
_CHILD_MAX_BYTES = 2_000_000
_BOUND_MAX_BYTES = 1_000_000_000
_V1_RESULT_MAX_BYTES = 2_000_000
_V1_POLICY_MAX_BYTES = 16_000_000
_V1_SUCCESS_MAX_BYTES = 2_000_000
_ACCESS_RECORD_MAX_BYTES = 16_384
_PRIVATE_JSON_MAX_BYTES = 64_000_000
_PRIVATE_NPZ_MAX_BYTES = 2_000_000_000
_PRIVATE_NPZ_MEMBER_MAX_BYTES = 256_000_000
_PRIVATE_NPZ_MEMBER_COUNT_MAX = 1_024
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MD5 = re.compile(r"md5:[0-9a-f]{32}\Z")
_GIT_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_OWNER_NONCE = re.compile(r"[0-9a-f]{64}\Z")

REQUIRED_RUNTIME_BINDING_PATHS: Final[tuple[str, ...]] = (
    "scripts/evaluate_trust_sentinel_ood_external_v2.py",
    "src/ecg_trust/conformal/multilabel.py",
    "src/ecg_trust/data/dataset.py",
    "src/ecg_trust/demo_backend.py",
    "src/ecg_trust/experiment_config.py",
    "src/ecg_trust/experiment_runner.py",
    "src/ecg_trust/foundation/adapter.py",
    "src/ecg_trust/models/resnet1d.py",
    "src/ecg_trust/ood_completion/models.py",
    "src/ecg_trust/ood_completion/pipeline.py",
    "src/ecg_trust/ood_completion/runtime.py",
    "src/ecg_trust/ood_v2/adapters.py",
    "src/ecg_trust/ood_v2/bundle.py",
    "src/ecg_trust/ood_v2/inventory.py",
    "src/ecg_trust/ood_v2/models.py",
    "src/ecg_trust/ood_v2/pipeline.py",
    "src/ecg_trust/ood_v2/statistics.py",
    "src/ecg_trust/open_world/scores.py",
    "src/ecg_trust/quality/signal_quality.py",
    "src/ecg_trust/source_calibration/models.py",
    "src/ecg_trust/source_calibration/pipeline.py",
)
PROJECT_SOURCE_ROOT: Final = "src/ecg_trust"
PROJECT_EVALUATION_ENTRYPOINT: Final = (
    "scripts/evaluate_trust_sentinel_ood_external_v2.py"
)
PROJECT_OPERATIONAL_ENTRYPOINTS: Final[tuple[str, ...]] = (
    "scripts/build_trust_sentinel_ood_v2_inventory.py",
    PROJECT_EVALUATION_ENTRYPOINT,
    "scripts/freeze_trust_sentinel_ood_external_v2.py",
    "scripts/verify_trust_sentinel_ood_external_v2.py",
)
RUNTIME_FILESYSTEM_TREE_KINDS: Final[tuple[str, str, str]] = (
    "cpython_base_runtime",
    "venv_site_packages",
    "git_mingw64_runtime",
)
EXPECTED_GIT_VERSION: Final = "git version 2.53.0.windows.2"
EXPECTED_GIT_INSTALL_ROOT: Final = r"C:\Program Files\Git"
EXPECTED_GIT_LAUNCHER_NAME: Final = "git.exe"
EXPECTED_GIT_LAUNCHER_SIZE_BYTES: Final = 46_464
EXPECTED_GIT_LAUNCHER_SHA256: Final = (
    "sha256:37c5725818d602e951ba2563b870d62763322956b73373da4c33a0b566a80bc9"
)
EXPECTED_GIT_EXECUTABLE_NAME: Final = "git.exe"
EXPECTED_GIT_EXECUTABLE_SIZE_BYTES: Final = 4_344_192
EXPECTED_GIT_EXECUTABLE_SHA256: Final = (
    "sha256:c39b1b4f7a57935bbeadf246dc2466316619453a6a9da77c4a9c6bd6d8fb21d3"
)
EXPECTED_GIT_CREDENTIAL_MANAGER_NAME: Final = "git-credential-manager.exe"
GCM_SYSTEM_COMMANDLINE_SENTINEL_DIRECTORY_NAME: Final = (
    "system-commandline-sentinel-files"
)
EXPECTED_GIT_CREDENTIAL_MANAGER_SIZE_BYTES: Final = 133_192
EXPECTED_GIT_CREDENTIAL_MANAGER_SHA256: Final = (
    "sha256:b7f0e61535b7bab81ea11126ecf1e7ad4486426df69921a78a680dc40bae2c12"
)
EXPECTED_GIT_CREDENTIAL_MANAGER_VERSION: Final = (
    "2.7.3+5fa7116896c82164996a609accd1c5ad90fe730a"
)
EXPECTED_GIT_CREDENTIAL_MANAGER_VERSION_STDOUT: Final = (
    EXPECTED_GIT_CREDENTIAL_MANAGER_VERSION + "\r\n"
).encode("ascii")
EXPECTED_GIT_RUNTIME_FILE_COUNT: Final = 4_565
EXPECTED_GIT_RUNTIME_DIRECTORY_COUNT: Final = 194
EXPECTED_GIT_RUNTIME_TOTAL_BYTES: Final = 224_959_003
EXPECTED_GIT_RUNTIME_TREE_SHA256: Final = (
    "sha256:086bd1898a3859d59d4c7184f1039d73cdf75c07de76f70fc375495ed922d9e2"
)
_GIT_RUNTIME_TREE_VERIFIED = False
EXPECTED_NVIDIA_DRIVER_VERSION: Final = "596.49"
EXPECTED_NVIDIA_DRIVER_FILES: Final[
    Mapping[str, tuple[int, str]]
] = MappingProxyType(
    {
        "nvidia-smi.exe": (
            1_624_808,
            "sha256:74348eb0bee800304ef5214d1fe8e643d7220ef8585a4e60c564fc24a06d3939",
        ),
        "nvml.dll": (
            1_391_848,
            "sha256:046b733c849261658cd318aab6e26fec94f330a9981a1dc4a30617dfea862673",
        ),
        "nvcuda.dll": (
            4_466_920,
            "sha256:ec9942ff94bcf2a6714531932720d0d36bd1f362df768af9ae21f2388c08ef7c",
        ),
    }
)
EXPECTED_HOST_SECURITY_NATIVE_MODULES: Final[
    Mapping[str, tuple[int, str]]
] = MappingProxyType(
    {
        r"C:\Program Files\Norton\Suite\aswAMSI.dll": (
            1_007_544,
            "sha256:5cb95df5fe2800f297c223fff08c710a0409c4c88b89f07b60ef33d2e2e2704c",
        ),
        r"C:\ProgramData\Microsoft\Windows Defender\Platform\4.18.26070.9-0\MpOav.dll": (
            673_816,
            "sha256:2d5e72b81c236db1fd30978e2ad6a20d241945090b90f2cc2a36993469dc144f",
        ),
    }
)
ALLOWED_FROZEN_MODULE_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "importlib._bootstrap": "_frozen_importlib",
        "importlib._bootstrap_external": "_frozen_importlib_external",
        "os.path": "ntpath",
    }
)
EXPECTED_RUNTIME_SYS_PATH_LAYOUT: Final[tuple[str, ...]] = (
    "cpython_zip",
    "cpython_dlls",
    "cpython_stdlib",
    "cpython_base",
    "venv_site_packages",
    "project_src",
)
EXPECTED_PYTHON_BASE_ALIAS_NAME: Final = "cpython-3.12-windows-x86_64-none"
EXPECTED_PYTHON_BASE_TARGET_NAME: Final = "cpython-3.12.13-windows-x86_64-none"
FORBIDDEN_CODE_ENVIRONMENT_VARIABLES: Final[tuple[str, ...]] = (
    "COVERAGE_PROCESS_START",
    "COVERAGE_RCFILE",
    "PYTHONHOME",
    "PYTHONINSPECT",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
)
FORBIDDEN_BOOTSTRAP_MODULES: Final[tuple[str, ...]] = (
    "_editable_impl_ecg_trust",
    "_virtualenv",
    "sitecustomize",
    "usercustomize",
)
ALLOWED_RUNTIME_ENVIRONMENT_VARIABLES: Final[frozenset[str]] = frozenset(
    {
        "ALLUSERSPROFILE",
        "APPDATA",
        "COMMONPROGRAMFILES",
        "COMMONPROGRAMFILES(X86)",
        "COMMONPROGRAMW6432",
        "COMSPEC",
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_CACHE_DISABLE",
        "DRIVERDATA",
        "HOMEDRIVE",
        "HOMEPATH",
        "LOCALAPPDATA",
        "NUMBER_OF_PROCESSORS",
        "OS",
        "PATHEXT",
        "PATH",
        "PROCESSOR_ARCHITECTURE",
        "PROCESSOR_IDENTIFIER",
        "PROCESSOR_LEVEL",
        "PROCESSOR_REVISION",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "PROGRAMW6432",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TORCHINDUCTOR_CACHE_DIR",
        "USERDOMAIN",
        "USERDOMAIN_ROAMINGPROFILE",
        "USERNAME",
        "USERPROFILE",
        "WINDIR",
    }
)

REQUIRED_RAW_SOURCE_BINDING_KEYS: Final[tuple[str, ...]] = (
    "challenge_archive",
    "challenge_records",
    "challenge_records_acceptable",
    "challenge_records_unacceptable",
    "zzu_archive_z01",
    "zzu_archive_zip",
    "zzu_attributes_dictionary",
    "zzu_disease_code",
    "zzu_ecg_code",
    "zzu_example_notebook",
)

EXPECTED_DATASET_ROOTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        CHALLENGE_2011_DATASET: (
            "data/raw/external-ood/challenge-2011-v1.0.0/extracted/set-a"
        ),
        ZZU_PEDIATRIC_DATASET: (
            "data/raw/external-ood/zzu-pecg-v1/extracted/Child_ecg"
        ),
    }
)

EXPECTED_RAW_SOURCE_PATHS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "challenge_archive": (
            "data/raw/external-ood/challenge-2011-v1.0.0/set-a.tar.gz"
        ),
        "challenge_records": (
            "data/raw/external-ood/challenge-2011-v1.0.0/extracted/set-a/RECORDS"
        ),
        "challenge_records_acceptable": (
            "data/raw/external-ood/challenge-2011-v1.0.0/extracted/set-a/"
            "RECORDS-acceptable"
        ),
        "challenge_records_unacceptable": (
            "data/raw/external-ood/challenge-2011-v1.0.0/extracted/set-a/"
            "RECORDS-unacceptable"
        ),
        "zzu_archive_z01": "data/raw/external-ood/zzu-pecg-v1/Child_ecg.z01",
        "zzu_archive_zip": "data/raw/external-ood/zzu-pecg-v1/Child_ecg.zip",
        "zzu_attributes_dictionary": (
            "data/raw/external-ood/zzu-pecg-v1/AttributesDictionary.csv"
        ),
        "zzu_disease_code": "data/raw/external-ood/zzu-pecg-v1/DiseaseCode.csv",
        "zzu_ecg_code": "data/raw/external-ood/zzu-pecg-v1/ECGCode.csv",
        "zzu_example_notebook": (
            "data/raw/external-ood/zzu-pecg-v1/ExampleReadingCode.ipynb"
        ),
    }
)

EXPECTED_SUCCESSOR_INVENTORY_COUNTS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "challenge_records": 1_000,
        "total_records": 13_328,
        "zzu_candidate_patients": 11_643,
        "zzu_candidate_records": 14_190,
        "zzu_nine_lead_records": 1_856,
        "zzu_patients": 10_350,
        "zzu_records": 12_328,
        "zzu_twelve_lead_records": 12_334,
    }
)
EXPECTED_SUCCESSOR_ZZU_EXCLUSION_COUNTS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "duration_under_10_seconds": 6,
        "lead_count_not_12": 0,
        "noncanonical_lead_set": 0,
        "pediatric_12_lead_flag_false": 1_856,
        "sampling_frequency_not_500_hz": 0,
    }
)
INVENTORY_BUILDER_RAW_SOURCE_KEYS: Final[tuple[str, ...]] = (
    "challenge_archive",
    "challenge_records",
    "challenge_records_acceptable",
    "challenge_records_unacceptable",
    "zzu_archive_z01",
    "zzu_archive_zip",
    "zzu_attributes_dictionary",
)

# The exact frozen v2 parent is retained as a transparent pre-inference
# infeasibility.  Its canonical-lead wording forbids the case-only augmented
# lead aliases present in the already inventoried ZZU headers.  No execution
# against these parent bytes may consume the permanent external-access claim.
FROZEN_V2_PREINFERENCE_INFEASIBILITY: Final = (
    "the frozen v2 parent forbids AVR/AVL/AVF to aVR/aVL/aVF lead-name "
    "canonicalization required by the selected ZZU source"
)
EXPECTED_SCIENTIFIC_PACKAGE_VERSIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "PyYAML": "6.0.3",
        "numpy": "2.5.1",
        "pydantic": "2.13.4",
        "pydantic-core": "2.46.4",
        "scipy": "1.18.0",
        "torch": "2.13.0+cu130",
        "wfdb": "4.3.1",
    }
)
EXPECTED_SCIENTIFIC_PACKAGE_IMPORT_ROOTS: Final[Mapping[str, tuple[str, ...]]] = (
    MappingProxyType(
        {
            "PyYAML": ("yaml", "_yaml"),
            "numpy": ("numpy", "numpy.libs"),
            "pydantic": ("pydantic",),
            "pydantic-core": ("pydantic_core",),
            "scipy": ("scipy", "scipy.libs"),
            "torch": ("torch", "functorch", "torchgen"),
            "wfdb": ("wfdb",),
        }
    )
)
EXPECTED_GIT_REMOTE_NAME: Final = "origin"
EXPECTED_GIT_REMOTE_URL: Final = (
    "https://github.com/Ahmad986Ferdaws/ecg-trust-lab.git"
)
EXPECTED_GIT_REMOTE_MAIN_REF: Final = "refs/remotes/origin/main"
EXPECTED_GIT_REMOTE_BACKUP_TAG_REF: Final = (
    "refs/tags/private-evidence-backup-v1-2026-08-29"
)
EXPECTED_GIT_REMOTE_BACKUP_TAG_REVISION: Final = (
    "a88ef86e8e0b28dd6f162cda88e16b4159d195d8"
)
PRIVATE_REMOTE_GIT_CONFIG: Final[tuple[str, ...]] = (
    "credential.helper=",
    "credential.helper=manager",
    "credential.interactive=false",
    "credential.guiPrompt=false",
    "credential.allowUnsafeRemotes=false",
    "credential.credentialStore=wincredman",
    "credential.namespace=git",
    "credential.useHttpPath=false",
    "credential.username=Ahmad986Ferdaws",
    "credential.https://github.com.provider=github",
    "credential.trace=false",
    "credential.traceSecrets=false",
    "credential.traceMsAuth=false",
    "credential.debug=false",
    "http.followRedirects=false",
    "http.sslVerify=true",
)
PRIVATE_REMOTE_ANONYMOUS_GIT_CONFIG: Final[tuple[str, ...]] = (
    "credential.helper=",
    "credential.interactive=false",
    "credential.guiPrompt=false",
    "credential.allowUnsafeRemotes=false",
    "http.followRedirects=false",
    "http.sslVerify=true",
)
EXPECTED_PRIVATE_REMOTE_ANONYMOUS_STDERR: Final = (
    b"fatal: unable to get password from user\n"
)
PRIVATE_REMOTE_GCM_ENVIRONMENT: Final[Mapping[str, str]] = MappingProxyType(
    {
        "GCM_ALLOW_UNSAFE_REMOTES": "0",
        "GCM_CREDENTIAL_STORE": "wincredman",
        "GCM_DEBUG": "0",
        "GCM_GUI_PROMPT": "0",
        "GCM_INTERACTIVE": "0",
        "GCM_NAMESPACE": "git",
        "GCM_PROVIDER": "github",
        "GCM_TRACE": "0",
        "GCM_TRACE_MSAUTH": "0",
        "GCM_TRACE_SECRETS": "0",
    }
)
PRIVATE_REMOTE_FORBIDDEN_ENVIRONMENT_KEYS: Final[tuple[str, ...]] = (
    "ALL_PROXY",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GIT_ASKPASS",
    "GIT_CONFIG_PARAMETERS",
    "GIT_SSH",
    "GIT_SSH_COMMAND",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "SSH_ASKPASS",
)
PRIVATE_REMOTE_STDOUT_LIMIT_BYTES: Final = 4_096
PRIVATE_REMOTE_STDERR_LIMIT_BYTES: Final = 4_096
GCM_VERSION_STDOUT_LIMIT_BYTES: Final = 256
GCM_VERSION_STDERR_LIMIT_BYTES: Final = 256
PRIVATE_REMOTE_TIMEOUT_SECONDS: Final = 60.0
GCM_VERSION_TIMEOUT_SECONDS: Final = 30.0
WINDOWS_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: Final = 0x00002000
WINDOWS_PROCESS_ATTRIBUTE_HANDLE_LIST: Final = 0x00020002
WINDOWS_PROCESS_ATTRIBUTE_JOB_LIST: Final = 0x0002000D
WINDOWS_EXTENDED_STARTUPINFO_PRESENT: Final = 0x00080000
WINDOWS_CREATE_UNICODE_ENVIRONMENT: Final = 0x00000400
WINDOWS_CREATE_NO_WINDOW: Final = 0x08000000
WINDOWS_PRIVATE_PROCESS_CLEANUP_TIMEOUT_SECONDS: Final = 10.0
FIRST_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION: Final = (
    "85b55d0f358e12052b23c8afa7468f2285342181"
)
FIRST_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256: Final = (
    "sha256:b60c757c5da69ec0a0929c5d503d434302b406fd35c0d6237aaf23f3ea243f98"
)
SECOND_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION: Final = (
    "b5727c47dc719a8ec3d51deacad9936fd9df2a50"
)
SECOND_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256: Final = (
    "sha256:33da72f63106f7783ff63bf40f33ca94d55dac457d05bdc551a26fa72d11fac0"
)
THIRD_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION: Final = (
    "6b6ddfd0e26c2c65265e7c128bafb3a13c0bf9a6"
)
THIRD_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256: Final = (
    "sha256:9b0358be1d4a12ca1771c57d8387c1b332bbef5698e01d3da2707f59157a586c"
)
FOURTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION: Final = (
    "6b04c5c6308cfddd9a3b2b06f1ebbe24acc961e9"
)
FOURTH_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256: Final = (
    "sha256:ac3653cd3a83d8d963531e54566487749c0faf03b5bc816ae66bdbde7f21927c"
)
FIFTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION: Final = (
    "ff7c821e8b01e48e7e96fc29ddcec6e515286ddb"
)
FIFTH_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256: Final = (
    "sha256:d4c3145985219fd65c9a5a4800773427cecd1f099b9e7ab75958596b7a995c61"
)
SIXTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION: Final = (
    "62f18d2ab4a20d8b588e97d8b6f93b95387996ca"
)
SIXTH_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256: Final = (
    "sha256:5457ef7e773825523446d15e4f9f688f7c7006364c7843cd2d624dc2514fe11a"
)
SEVENTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION: Final = (
    "207fd1568697adb56991baeccee29ded38d3caf1"
)
SEVENTH_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256: Final = (
    "sha256:1da505b37d64dec804f147fa8cfd43a5029fe2ee7d92d1666177d490ea7016e1"
)
EIGHTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION: Final = (
    "f0e91538bf374ca7bb2579a4689f4e70648776f2"
)
EIGHTH_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256: Final = (
    "sha256:725619c224569fb13fbe2a5d4a79b5a84fd607f6bc12ae34ef74aecb8db73c93"
)
EIGHTH_FROZEN_SUCCESSOR_PARENT_FROZEN_AT_UTC: Final = "2026-08-30T07:48:31Z"
NINTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION: Final = (
    "06005b87557bb39455cf5fea1f48fb1c0633da9a"
)
NINTH_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256: Final = (
    "sha256:790251613a94052a88eb9f8598021d9bb3684707cd0a865c74b62dcb76c8417b"
)
NINTH_FROZEN_SUCCESSOR_PARENT_FROZEN_AT_UTC: Final = "2026-08-30T09:49:39Z"
SUCCESSOR_AMENDMENT_MODIFIED_PATHS: Final[tuple[str, ...]] = (
    "configs/trust_sentinel_ood_external_v2_1.yaml",
    "docs/TRUST_SENTINEL_OOD_EXTERNAL_V2_1_PROTOCOL.md",
    "src/ecg_trust/ood_v2/models.py",
    "src/ecg_trust/ood_v2/pipeline.py",
    "tests/unit/test_ood_v2_models.py",
    "tests/unit/test_ood_v2_pipeline.py",
    "tests/unit/test_ood_v2_protocol_closure.py",
)
SUCCESSOR_PRIVATE_REMOTE_AMENDMENT_MODIFIED_PATHS: Final[tuple[str, ...]] = (
    "configs/trust_sentinel_ood_external_v2_1.yaml",
    "docs/TRUST_SENTINEL_OOD_EXTERNAL_V2_1_PROTOCOL.md",
    "src/ecg_trust/ood_v2/models.py",
    "src/ecg_trust/ood_v2/pipeline.py",
    "tests/unit/test_ood_v2_models.py",
    "tests/unit/test_ood_v2_pipeline.py",
    "tests/unit/test_ood_v2_protocol_closure.py",
)
SUCCESSOR_INVENTORY_BUILDER_AMENDMENT_MODIFIED_PATHS: Final[tuple[str, ...]] = (
    "configs/trust_sentinel_ood_external_v2_1.yaml",
    "docs/TRUST_SENTINEL_OOD_EXTERNAL_V2_1_PROTOCOL.md",
    "scripts/build_trust_sentinel_ood_v2_inventory.py",
    "src/ecg_trust/ood_v2/inventory.py",
    "src/ecg_trust/ood_v2/models.py",
    "src/ecg_trust/ood_v2/pipeline.py",
    "tests/unit/test_ood_v2_inventory.py",
    "tests/unit/test_ood_v2_inventory_cli.py",
    "tests/unit/test_ood_v2_models.py",
    "tests/unit/test_ood_v2_pipeline.py",
    "tests/unit/test_ood_v2_protocol_closure.py",
)
SUCCESSOR_RUNTIME_PREFLIGHT_AMENDMENT_MODIFIED_PATHS: Final[tuple[str, ...]] = (
    "configs/trust_sentinel_ood_external_v2_1.yaml",
    "docs/TRUST_SENTINEL_OOD_EXTERNAL_V2_1_PROTOCOL.md",
    "scripts/build_trust_sentinel_ood_v2_inventory.py",
    "src/ecg_trust/ood_v2/models.py",
    "src/ecg_trust/ood_v2/pipeline.py",
    "tests/unit/test_ood_v2_inventory_cli.py",
    "tests/unit/test_ood_v2_models.py",
    "tests/unit/test_ood_v2_pipeline.py",
    "tests/unit/test_ood_v2_protocol_closure.py",
)
SUCCESSOR_GCM_SCRATCH_AMENDMENT_MODIFIED_PATHS: Final[tuple[str, ...]] = (
    "configs/trust_sentinel_ood_external_v2_1.yaml",
    "docs/TRUST_SENTINEL_OOD_EXTERNAL_V2_1_PROTOCOL.md",
    "scripts/build_trust_sentinel_ood_v2_inventory.py",
    "scripts/evaluate_trust_sentinel_ood_external_v2.py",
    "scripts/freeze_trust_sentinel_ood_external_v2.py",
    "scripts/verify_trust_sentinel_ood_external_v2.py",
    "src/ecg_trust/ood_v2/models.py",
    "src/ecg_trust/ood_v2/pipeline.py",
    "tests/unit/test_ood_v2_cli.py",
    "tests/unit/test_ood_v2_inventory_cli.py",
    "tests/unit/test_ood_v2_models.py",
    "tests/unit/test_ood_v2_pipeline.py",
    "tests/unit/test_ood_v2_protocol_closure.py",
)
SUCCESSOR_INVENTORY_FAILURE_AMENDMENT_MODIFIED_PATHS: Final[tuple[str, ...]] = (
    "configs/trust_sentinel_ood_external_v2_1.yaml",
    "docs/TRUST_SENTINEL_OOD_EXTERNAL_V2_1_PROTOCOL.md",
    "scripts/build_trust_sentinel_ood_v2_inventory.py",
    "src/ecg_trust/ood_v2/inventory.py",
    "src/ecg_trust/ood_v2/models.py",
    "src/ecg_trust/ood_v2/pipeline.py",
    "tests/unit/test_ood_v2_inventory.py",
    "tests/unit/test_ood_v2_inventory_cli.py",
    "tests/unit/test_ood_v2_models.py",
    "tests/unit/test_ood_v2_pipeline.py",
    "tests/unit/test_ood_v2_protocol_closure.py",
)
SUCCESSOR_ARCHIVE_OPERAND_AMENDMENT_MODIFIED_PATHS: Final[tuple[str, ...]] = (
    "configs/trust_sentinel_ood_external_v2_1.yaml",
    "docs/TRUST_SENTINEL_OOD_EXTERNAL_V2_1_PROTOCOL.md",
    "src/ecg_trust/ood_v2/inventory.py",
    "src/ecg_trust/ood_v2/models.py",
    "src/ecg_trust/ood_v2/pipeline.py",
    "tests/unit/test_ood_v2_inventory.py",
    "tests/unit/test_ood_v2_models.py",
    "tests/unit/test_ood_v2_pipeline.py",
    "tests/unit/test_ood_v2_protocol_closure.py",
)
SUCCESSOR_CHILD_FREEZE_AMENDMENT_MODIFIED_PATHS: Final[tuple[str, ...]] = (
    "configs/trust_sentinel_ood_external_v2_1.yaml",
    "docs/TRUST_SENTINEL_OOD_EXTERNAL_V2_1_PROTOCOL.md",
    "scripts/freeze_trust_sentinel_ood_external_v2.py",
    "src/ecg_trust/ood_v2/models.py",
    "src/ecg_trust/ood_v2/pipeline.py",
    "tests/unit/test_ood_v2_cli.py",
    "tests/unit/test_ood_v2_models.py",
    "tests/unit/test_ood_v2_pipeline.py",
    "tests/unit/test_ood_v2_protocol_closure.py",
)
SUCCESSOR_CHILD_FREEZE_DECISION_BINDING_AMENDMENT_MODIFIED_PATHS: Final[
    tuple[str, ...]
] = SUCCESSOR_CHILD_FREEZE_AMENDMENT_MODIFIED_PATHS
_ARCHIVE_MEMBER_ROLES: Final[tuple[str, ...]] = (
    "ignored_release_file",
    "quality_reference",
    "wfdb_data",
    "wfdb_header",
)
FROZEN_ROUTE_ORDER: Final[tuple[str, ...]] = (
    "INVALID_INPUT",
    "REACQUIRE",
    "UNSUPPORTED_INPUT",
    "ABSTAIN",
    "PREDICTION_ALLOWED",
)


class OODExternalV2ConfigError(ValueError):
    """Raised when the parent or child preregistration is invalid."""


class OODExternalV2IntegrityError(RuntimeError):
    """Raised when immutable inputs or evidence fail verification."""


class OODExternalV2ExecutionError(RuntimeError):
    """Raised when a one-shot v2 execution cannot complete safely."""


INVENTORY_BUILDER_PREFLIGHT_STAGES: Final[tuple[str, ...]] = (
    "parent_lineage",
    "runtime_environment",
    "git_source_provenance",
    "namespace_state",
    "closing_control_state",
)
INVENTORY_BUILDER_ATTEMPT_STAGES: Final[tuple[str, ...]] = (
    "authorization_publication",
    "raw_source_binding_verification",
    "expectation_materialization",
    "challenge_inventory",
    "zzu_metadata_parse_and_counts",
    "zzu_header_selection_and_counts",
    "challenge_archive_closure",
    "zzu_tool_resolution",
    "zzu_archive_listing",
    "zzu_archive_test",
    "zzu_evaluated_tree_snapshot",
    "zzu_isolated_extraction",
    "zzu_archive_comparison",
    "archive_closure_role_validation",
    "inventory_assembly_and_reverification",
    "public_projection_build_and_verify",
    "canonical_serialization",
    "precommit_inventory_reverify",
    "output_transaction",
    "output_reload_and_verify",
    "postflight",
)
INVENTORY_BUILDER_OUTPUT_STATES: Final[tuple[str, ...]] = (
    "NONE",
    "PRIVATE_ONLY",
    "PUBLIC_ONLY",
    "BOTH",
    "UNVERIFIABLE",
)
CHILD_FREEZE_PREFLIGHT_STAGES: Final[tuple[str, ...]] = (
    "parent_lineage",
    "runtime_environment",
    "git_source_provenance",
    "x8_inventory_evidence",
    "decision_and_runtime_bindings",
    "namespace_and_timestamp",
    "closing_control_state",
)
CHILD_FREEZE_ATTEMPT_STAGES: Final[tuple[str, ...]] = (
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
CHILD_FREEZE_FAILURE_REASONS: Final[tuple[str, ...]] = (
    "STAGE_REFUSED",
    "DESTINATION_PREEXISTED",
    "PUBLICATION_FAILED_BEFORE_VISIBILITY",
    "PUBLICATION_FAILED_AFTER_VISIBILITY",
    "POSTPUBLICATION_RELOAD_REFUSED",
    "UNEXPECTED_INTERNAL_FAILURE",
)
CHILD_FREEZE_OUTPUT_STATES: Final[tuple[str, ...]] = (
    "NONE",
    "VISIBLE_EXACT_DURABILITY_UNCONFIRMED",
    "DURABLE_EXACT",
    "PRESENT_UNVERIFIABLE",
)
CHILD_FREEZE_AUTHORIZATION_STATES: Final[tuple[str, ...]] = (
    "NOT_CONSUMED",
    "CONSUMED",
    "UNVERIFIABLE",
)
CHILD_FREEZE_CLEANUP_STATES: Final[tuple[str, ...]] = (
    "NOT_REACHED",
    "CLEAN",
    "FAILED",
)


class InventoryBuilderPreflightStageError(OODExternalV2IntegrityError):
    """Sanitized, controls-only inventory-preflight stage refusal."""

    def __init__(self, stage: str) -> None:
        if stage not in INVENTORY_BUILDER_PREFLIGHT_STAGES:
            raise ValueError("inventory builder preflight stage is not allowlisted")
        self.stage = stage
        super().__init__(f"inventory builder preflight refused at stage {stage}")


class ChildFreezePreflightStageError(OODExternalV2IntegrityError):
    """Sanitized, controls-only X10 child-freeze preflight refusal."""

    def __init__(self, stage: str) -> None:
        if stage not in CHILD_FREEZE_PREFLIGHT_STAGES:
            raise ValueError("child freeze preflight stage is not allowlisted")
        self.stage = stage
        super().__init__(f"child freeze preflight refused at stage {stage}")


class ChildFreezeAttemptError(OODExternalV2ExecutionError):
    """Sanitized terminal refusal after the X10 marker became visible."""

    def __init__(
        self,
        *,
        stage: str,
        reason: str,
        output_state: str,
        official_source_content_accessed: bool,
        failure_receipt_written: bool,
    ) -> None:
        if stage not in CHILD_FREEZE_ATTEMPT_STAGES:
            raise ValueError("child freeze attempt stage is not allowlisted")
        if reason not in CHILD_FREEZE_FAILURE_REASONS:
            raise ValueError("child freeze failure reason is not allowlisted")
        if output_state not in CHILD_FREEZE_OUTPUT_STATES:
            raise ValueError("child freeze output state is not allowlisted")
        if type(official_source_content_accessed) is not bool:
            raise TypeError("official_source_content_accessed must be bool")
        if type(failure_receipt_written) is not bool:
            raise TypeError("failure_receipt_written must be bool")
        self.authorization_consumed = True
        self.stage = stage
        self.reason = reason
        self.output_state = output_state
        self.official_source_content_accessed = official_source_content_accessed
        self.failure_receipt_written = failure_receipt_written
        super().__init__("X10 child freeze failed after authorization consumption")


@dataclass(frozen=True, slots=True)
class BoundFile:
    relative_path: str
    file_sha256: str
    artifact_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class RawSourceBinding:
    relative_path: str
    file_sha256: str
    size_bytes: int
    official_md5: str | None


@dataclass(frozen=True, slots=True)
class RuntimeEnvironmentBinding:
    python_implementation: str
    python_version: str
    python_executable_file_sha256: str
    python_executable_size_bytes: int
    python_base_alias_name: str
    python_base_target_name: str
    python_environment_sha256: str
    numpy_version: str
    scipy_version: str
    wfdb_version: str
    package_trees: tuple[RuntimePackageTreeBinding, ...]
    python_base_tree: RuntimeFilesystemTreeBinding
    site_packages_tree: RuntimeFilesystemTreeBinding
    pyvenv_config_file_sha256: str
    pyvenv_config_size_bytes: int
    isolated_mode: bool
    no_site: bool
    dont_write_bytecode: bool
    user_site_disabled: bool
    pycache_prefix_verified_empty: bool
    sys_path_layout: tuple[str, ...]
    git_tool: GitToolBinding
    nvidia_driver_tool: NvidiaDriverToolBinding


@dataclass(frozen=True, slots=True)
class RuntimePackageTreeBinding:
    distribution: str
    version: str
    import_roots: tuple[str, ...]
    file_count: int
    total_bytes: int
    tree_sha256: str


@dataclass(frozen=True, slots=True)
class RuntimeFilesystemTreeBinding:
    tree_kind: str
    file_count: int
    directory_count: int
    total_bytes: int
    tree_sha256: str


@dataclass(frozen=True, slots=True)
class GitToolBinding:
    version: str
    launcher_name: str
    launcher_size_bytes: int
    launcher_sha256: str
    executable_name: str
    executable_size_bytes: int
    executable_sha256: str
    runtime_tree: RuntimeFilesystemTreeBinding


@dataclass(frozen=True, slots=True)
class NvidiaDriverToolBinding:
    driver_version: str
    nvidia_smi_name: str
    nvidia_smi_size_bytes: int
    nvidia_smi_sha256: str
    nvml_name: str
    nvml_size_bytes: int
    nvml_sha256: str
    nvcuda_name: str
    nvcuda_size_bytes: int
    nvcuda_sha256: str


@dataclass(frozen=True, slots=True)
class ProjectSourceFileBinding:
    relative_path: str
    size_bytes: int
    file_sha256: str


@dataclass(frozen=True, slots=True)
class ProjectSourceTreeBinding:
    files: tuple[ProjectSourceFileBinding, ...]
    file_count: int
    total_bytes: int
    tree_sha256: str


@dataclass(frozen=True, slots=True)
class SuccessorInventoryCountBinding:
    challenge_records: int
    zzu_candidate_records: int
    zzu_candidate_patients: int
    zzu_twelve_lead_records: int
    zzu_nine_lead_records: int
    zzu_records: int
    zzu_patients: int
    total_records: int
    zzu_exclusion_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        values = {
            "challenge_records": self.challenge_records,
            "total_records": self.total_records,
            "zzu_candidate_patients": self.zzu_candidate_patients,
            "zzu_candidate_records": self.zzu_candidate_records,
            "zzu_nine_lead_records": self.zzu_nine_lead_records,
            "zzu_patients": self.zzu_patients,
            "zzu_records": self.zzu_records,
            "zzu_twelve_lead_records": self.zzu_twelve_lead_records,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in values.values()
        ):
            raise OODExternalV2ConfigError(
                "successor inventory counts must be positive integers"
            )
        exclusions = dict(self.zzu_exclusion_counts)
        if set(exclusions) != set(EXPECTED_SUCCESSOR_ZZU_EXCLUSION_COUNTS) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in exclusions.values()
        ):
            raise OODExternalV2ConfigError(
                "successor ZZU exclusion counts differ from the frozen reasons"
            )
        if (
            self.challenge_records + self.zzu_records != self.total_records
            or self.zzu_records + sum(exclusions.values())
            != self.zzu_candidate_records
            or self.zzu_twelve_lead_records + self.zzu_nine_lead_records
            != self.zzu_candidate_records
            or exclusions["pediatric_12_lead_flag_false"]
            != self.zzu_nine_lead_records
            or (
                self.zzu_records
                + exclusions["duration_under_10_seconds"]
                + exclusions["lead_count_not_12"]
                + exclusions["noncanonical_lead_set"]
                + exclusions["sampling_frequency_not_500_hz"]
                != self.zzu_twelve_lead_records
            )
            or self.zzu_patients > self.zzu_records
            or self.zzu_candidate_patients > self.zzu_candidate_records
        ):
            raise OODExternalV2ConfigError(
                "successor inventory count accounting is inconsistent"
            )
        object.__setattr__(self, "zzu_exclusion_counts", MappingProxyType(exclusions))


@dataclass(frozen=True, slots=True)
class OODExternalV2ParentConfig:
    path: Path
    file_sha256: str
    status: str
    v1_result: BoundFile
    v1_success_manifest: BoundFile
    v1_distribution_policy: BoundFile
    checkpoint: BoundFile
    resolved_config: BoundFile
    resolved_config_sha256: str
    normalization: BoundFile
    quality_implementation: BoundFile
    quality_config_version: str
    dependency_lock: BoundFile
    project_manifest: BoundFile
    threshold: float
    challenge_expected_records: int
    bootstrap_resamples: int
    challenge_bootstrap_seed: int
    zzu_bootstrap_seed: int
    confidence_level: float
    challenge_group3_minimum: float
    challenge_group1_minimum: float
    challenge_distribution_minimum: float
    zzu_distribution_minimum: float
    output_root: str
    claim_path: str
    raw_source_bindings: Mapping[str, RawSourceBinding] | None
    seven_zip_tool_binding: SevenZipToolBinding | None
    inventory_counts: SuccessorInventoryCountBinding | None


@dataclass(frozen=True, slots=True)
class SuccessorParentPreflight:
    path: Path
    file_sha256: str
    status: str
    raw_source_bindings: Mapping[str, RawSourceBinding]
    seven_zip_tool_binding: SevenZipToolBinding
    inventory_counts: SuccessorInventoryCountBinding


@dataclass(frozen=True, slots=True)
class InventoryBuilderPreflight:
    """Path-free proof that metadata-only inventory construction may begin."""

    status: str
    parent_config_file_sha256: str
    implementation_revision: str
    project_source_tree_sha256: str
    python_environment_sha256: str
    git_runtime_tree_sha256: str
    raw_source_bindings: Mapping[str, RawSourceBinding]
    seven_zip_tool_binding: SevenZipToolBinding
    inventory_counts: SuccessorInventoryCountBinding


@dataclass(frozen=True, slots=True)
class InventoryBuilderPostflight:
    """Aggregate-only proof that the builder wrote exactly its intended files."""

    status: str
    preflight: InventoryBuilderPreflight
    inventory_file_sha256: str
    inventory_sha256: str
    public_projection_file_sha256: str
    public_projection_artifact_sha256: str


@dataclass(frozen=True, slots=True)
class ChildFreezePreflight:
    """Content-free proof that the single X10 child-freeze attempt may begin."""

    status: str
    parent: OODExternalV2ParentConfig
    project_root: Path
    implementation_revision: str
    project_source_tree: ProjectSourceTreeBinding
    runtime_environment: RuntimeEnvironmentBinding
    decision_bindings: Mapping[str, BoundFile]
    runtime_bindings: Mapping[str, str]
    frozen_at_utc: datetime
    inventory_path: Path
    public_projection_path: Path
    challenge_root: Path
    zzu_root: Path
    declared_counts: tuple[int, int, int, int]
    output_path: Path
    output_parent_identity: _OwnedDirectoryIdentity
    protocol_artifact_parent_identity: _OwnedDirectoryIdentity
    seven_zip_executable: Path


@dataclass(frozen=True, slots=True)
class ArchiveClosureSummaryBinding:
    dataset: str
    archive_format: str
    archive_file_count: int
    archive_bytes_total: int
    member_count: int
    member_bytes_total: int
    member_role_counts: tuple[int, int, int, int]
    closure_sha256: str
    tool_binding: SevenZipToolBinding | None


@dataclass(frozen=True, slots=True)
class InventoryBinding:
    relative_path: str
    file_sha256: str
    inventory_sha256: str
    selected_records_total: int
    challenge_records: int
    zzu_records: int
    zzu_patients: int
    archive_closures: tuple[ArchiveClosureSummaryBinding, ...]


@dataclass(frozen=True, slots=True)
class OODExternalV2ChildContract:
    path: Path
    file_sha256: str
    artifact_sha256: str
    frozen_at_utc: datetime
    parent_config_file_sha256: str
    implementation_revision: str
    inventory: InventoryBinding
    dataset_roots: Mapping[str, str]
    decision_bindings: Mapping[str, BoundFile]
    raw_source_bindings: Mapping[str, RawSourceBinding]
    inventory_builder_attempt: BoundFile
    child_freeze_attempt: BoundFile
    runtime_environment: RuntimeEnvironmentBinding
    runtime_bindings: Mapping[str, str]
    project_source_tree: ProjectSourceTreeBinding
    public_inventory_projection: BoundFile | None
    output_root: str


@dataclass(frozen=True, slots=True)
class VerifiedV1PublicEvidence:
    result: OODCompletionResult
    policy: DistributionPolicy
    success_manifest: OODCompletionSuccessManifest
    claim_file_sha256: str
    snapshots: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class VerifiedExternalV2Inputs:
    project_root: Path
    parent: OODExternalV2ParentConfig
    child: OODExternalV2ChildContract
    inventory: ExternalWaveformInventory
    inventory_path: Path
    dataset_roots: Mapping[str, Path]
    raw_source_paths: Mapping[str, Path]
    v1: VerifiedV1PublicEvidence
    checkpoint_path: Path
    resolved_config_path: Path
    normalization_path: Path
    routing: FrozenRoutingComponents


@dataclass(frozen=True, slots=True)
class FrozenRoutingComponents:
    source_calibration_result: SourceCalibrationResult
    historical_demo_policy: FrozenDecisionPolicy
    conformal: LabelwiseBinaryConformal
    temperature: float
    maximum_entropy: float
    source_calibration_file_sha256: str
    demo_policy_file_sha256: str


@dataclass(frozen=True, slots=True)
class _PrivateRecordEvidence:
    dataset: str
    record_ref: str
    patient_key: str | None
    challenge_quality_label: str | None
    adapter_provenance_sha256: str | None
    adapter_source_sample_count: int | None
    adapter_raw_physical_units: tuple[str, ...] | None
    canonical_signal_sha256: str | None
    quality_report_sha256: str | None
    quality_report: dict[str, object] | None
    quality_status: str
    quality_reason_codes: tuple[str, ...]
    route: str
    distribution_score: float | None
    entropy: float | None
    entropy_accepted: bool | None
    conformal_decisions: tuple[str, ...] | None
    all_conformal_decisions_singleton: bool | None

    def to_dict(self) -> dict[str, object]:
        return {
            "adapter_provenance_sha256": self.adapter_provenance_sha256,
            "adapter_raw_physical_units": (
                None
                if self.adapter_raw_physical_units is None
                else list(self.adapter_raw_physical_units)
            ),
            "adapter_source_sample_count": self.adapter_source_sample_count,
            "canonical_signal_sha256": self.canonical_signal_sha256,
            "challenge_quality_label": self.challenge_quality_label,
            "dataset": self.dataset,
            "distribution_score": self.distribution_score,
            "entropy": self.entropy,
            "entropy_accepted": self.entropy_accepted,
            "conformal_decisions": (
                None
                if self.conformal_decisions is None
                else list(self.conformal_decisions)
            ),
            "all_conformal_decisions_singleton": (
                self.all_conformal_decisions_singleton
            ),
            "patient_key": self.patient_key,
            "quality_reason_codes": list(self.quality_reason_codes),
            "quality_report_sha256": self.quality_report_sha256,
            "quality_status": self.quality_status,
            "record_ref": self.record_ref,
            "route": self.route,
        }


@dataclass(frozen=True, slots=True)
class _EvaluatedExternalRecords:
    records: tuple[_PrivateRecordEvidence, ...]
    adapter_success_inventory_indices: Int64Array
    canonical_signals: Float32Array
    quality_pass_inventory_indices: Int64Array
    embeddings: Float32Array
    repeated_embeddings: Float32Array
    repeated_embedding_sha256: str
    embedding_sha256: str
    scores: Float64Array
    logits: Float64Array
    repeated_logits: Float64Array
    probabilities: Float64Array
    first_logits_sha256: str
    repeated_logits_sha256: str
    probabilities_sha256: str
    model_state_before_sha256: str
    model_state_after_sha256: str
    model_state_unchanged: bool


@dataclass(frozen=True, slots=True)
class _EndpointEvidence:
    external_cohorts: tuple[object, ...]
    technical_quality_endpoints: tuple[object, ...]
    bootstrap_replicates: Mapping[str, Float64Array]
    challenge_group3_prediction_allowed_count: int
    route_counts: Mapping[str, int]


class _NormalizedSignalDataset(Dataset[tuple[Tensor, Tensor]]):
    """In-memory quality-passing signals normalized by the frozen PTB statistics."""

    def __init__(self, signals: Float32Array, normalization: NormalizationStats) -> None:
        values = np.asarray(signals)
        if (
            values.ndim != 3
            or values.shape[0] == 0
            or values.shape[1:] != (len(LEADS), 1000)
            or values.dtype != np.dtype(np.float32)
            or not np.isfinite(values).all()
        ):
            raise OODExternalV2ExecutionError("quality-passing signal matrix is invalid")
        self._signals = np.ascontiguousarray(values, dtype=np.float32)
        # Reuse the exact bound PTB dataset's Torch-float32 arithmetic.  A
        # NumPy reimplementation is not accepted because rounding at a strict
        # downstream threshold can differ.
        self._mean = torch.tensor(normalization.mean, dtype=torch.float32).unsqueeze(1)
        self._std = torch.tensor(normalization.std, dtype=torch.float32).unsqueeze(1)
        self._target = torch.zeros(len(SUPERCLASSES), dtype=torch.float32)

    def __len__(self) -> int:
        return int(self._signals.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        signal = torch.from_numpy(self._signals[index])
        normalized = (signal - self._mean) / self._std
        if not torch.isfinite(normalized).all().item():
            raise OODExternalV2ExecutionError(
                "external signal became nonfinite after normalization"
            )
        return normalized.contiguous(), self._target


def load_parent_config(path: str | Path) -> OODExternalV2ParentConfig:
    """Load execution-critical values from the exact parent YAML."""

    source = Path(os.path.abspath(os.fspath(path)))
    raw = _read_bounded(source, _CONFIG_MAX_BYTES, "parent protocol")
    file_sha256 = sha256_bytes(raw)
    if file_sha256 != EXPECTED_PARENT_CONFIG_SHA256:
        raise OODExternalV2ConfigError("parent protocol differs from frozen bytes")
    try:
        text = raw.decode("utf-8")
        _reject_duplicate_yaml_keys(text)
        decoded: object = yaml.safe_load(text)
    except (UnicodeError, yaml.YAMLError) as error:
        raise OODExternalV2ConfigError("parent protocol is not valid UTF-8 YAML") from error
    root = _mapping(decoded, "parent protocol")
    if (
        root.get("schema_version") != 1
        or root.get("protocol_id") != ORIGINAL_PROTOCOL_ID
        or root.get("status") != "frozen_parent_preregistration_pre_download"
        or root.get("research_only") is not True
    ):
        raise OODExternalV2ConfigError("parent protocol identity or status is invalid")

    design = _mapping(root.get("design_history"), "design_history")
    immutable = _mapping(root.get("immutability"), "immutability")
    if (
        design.get("written_after_v1_result_was_observed") is not True
        or design.get("independent_of_v1_claim") is not False
        or immutable.get("v1_output_must_remain_byte_identical") is not True
        or immutable.get("v1_distribution_policy_reused_exactly") is not True
        or immutable.get("v1_detector_refit") != "forbidden"
        or immutable.get("v1_threshold_change") != "forbidden"
        or immutable.get("external_target_fitting_or_adaptation") != "forbidden"
    ):
        raise OODExternalV2ConfigError("parent immutability boundary was weakened")
    c_boundary = _mapping(design.get("v1_c_influence"), "design_history.v1_c_influence")
    forbidden_c = c_boundary.get("forbidden")
    required_c_forbidden = {
        "waveform_access",
        "embedding_access",
        "row_identity_access",
        "score_access",
        "subgroup_or_error_analysis",
        "detector_fit",
        "threshold_fit",
        "method_selection",
        "operating_point_selection",
    }
    if not isinstance(forbidden_c, list) or not required_c_forbidden.issubset(forbidden_c):
        raise OODExternalV2ConfigError("sealed v1 C prohibition is incomplete")

    bindings = _mapping(root.get("bindings"), "bindings")
    v1_bundle = _mapping(bindings.get("v1_completion_bundle"), "v1 completion bundle")
    result = _bound_file(v1_bundle.get("result"), "v1 result", require_artifact=True)
    success = _bound_file(
        v1_bundle.get("success_manifest"),
        "v1 success manifest",
        require_artifact=True,
    )
    policy = _bound_file(
        v1_bundle.get("distribution_policy"),
        "v1 distribution policy",
        require_artifact=True,
    )
    policy_mapping = _mapping(v1_bundle.get("distribution_policy"), "v1 policy")
    threshold = _exact_float(policy_mapping.get("threshold"), "v1 threshold")
    if (
        threshold != 270.9668613705653
        or policy_mapping.get("method") != "shrinkage_mahalanobis_embedding_distance"
        or policy_mapping.get("threshold_comparison")
        != "score_strictly_greater_than_threshold"
    ):
        raise OODExternalV2ConfigError("v1 distribution policy contract changed")

    checkpoint = _bound_file(bindings.get("v1_checkpoint"), "checkpoint")
    resolved = _bound_file(bindings.get("v1_resolved_config"), "resolved config")
    resolved_mapping = _mapping(bindings.get("v1_resolved_config"), "resolved config")
    normalization = _bound_file(bindings.get("normalization"), "normalization")
    quality = _bound_file(
        bindings.get("signal_quality_implementation"),
        "quality implementation",
    )
    quality_mapping = _mapping(
        bindings.get("signal_quality_implementation"),
        "quality implementation",
    )
    dependency_lock = _bound_file(bindings.get("dependency_lock"), "dependency lock")
    project_manifest = _bound_file(bindings.get("project_manifest"), "project manifest")

    external = _mapping(root.get("external_sources"), "external_sources")
    challenge = _mapping(
        external.get(CHALLENGE_2011_DATASET),
        "Challenge 2011 source",
    )
    zzu = _mapping(external.get(ZZU_PEDIATRIC_DATASET), "ZZU source")
    if (
        challenge.get("version") != CHALLENGE_2011_VERSION
        or challenge.get("license_spdx") != "ODC-By-1.0"
        or zzu.get("version") != 1
        or zzu.get("license_spdx") != "CC-BY-4.0"
    ):
        raise OODExternalV2ConfigError("external source version or license changed")

    evaluation = _mapping(root.get("evaluation"), "evaluation")
    primary = _mapping(evaluation.get("primary_endpoints"), "primary endpoints")
    multiplicity = _mapping(evaluation.get("multiplicity"), "multiplicity")
    bootstrap = _mapping(evaluation.get("bootstrap"), "bootstrap")
    if (
        multiplicity.get("family") != "exact_four_co_primary_endpoints"
        or multiplicity.get("method") != "bonferroni"
        or multiplicity.get("all_four_endpoints_must_pass") is not True
        or _exact_float(multiplicity.get("required_one_sided_confidence_level"), "confidence")
        != 0.9875
    ):
        raise OODExternalV2ConfigError("co-primary multiplicity contract changed")
    challenge_bootstrap = _mapping(bootstrap.get("challenge"), "Challenge bootstrap")
    zzu_bootstrap = _mapping(bootstrap.get("zzu"), "ZZU bootstrap")
    if (
        challenge_bootstrap.get("unit") != "record"
        or zzu_bootstrap.get("unit") != "patient_cluster"
    ):
        raise OODExternalV2ConfigError("bootstrap resampling units changed")

    one_shot = _mapping(root.get("one_shot_external_access"), "one-shot access")
    claim = _mapping(one_shot.get("external_claim"), "external claim")
    return OODExternalV2ParentConfig(
        path=source,
        file_sha256=file_sha256,
        status=cast(str, root["status"]),
        v1_result=result,
        v1_success_manifest=success,
        v1_distribution_policy=policy,
        checkpoint=checkpoint,
        resolved_config=resolved,
        resolved_config_sha256=_digest(
            resolved_mapping.get("inner_config_sha256"),
            "resolved config logical hash",
        ),
        normalization=normalization,
        quality_implementation=quality,
        quality_config_version=_exact_string(
            quality_mapping.get("config_version"),
            "canonical-12x1000-mv-v1",
            "quality config version",
        ),
        dependency_lock=dependency_lock,
        project_manifest=project_manifest,
        threshold=threshold,
        challenge_expected_records=_exact_integer(
            challenge.get("expected_records"),
            1000,
            "Challenge expected records",
        ),
        bootstrap_resamples=_exact_integer(
            bootstrap.get("resamples"),
            10_000,
            "bootstrap resamples",
        ),
        challenge_bootstrap_seed=_exact_integer(
            challenge_bootstrap.get("seed"),
            20_260_901,
            "Challenge bootstrap seed",
        ),
        zzu_bootstrap_seed=_exact_integer(
            zzu_bootstrap.get("seed"),
            20_260_902,
            "ZZU bootstrap seed",
        ),
        confidence_level=0.9875,
        challenge_group3_minimum=_endpoint_minimum(
            primary,
            "challenge_group3_technical_block_sensitivity",
            0.95,
        ),
        challenge_group1_minimum=_endpoint_minimum(
            primary,
            "challenge_group1_quality_pass_rate",
            0.90,
        ),
        challenge_distribution_minimum=_endpoint_minimum(
            primary,
            "challenge_external_distribution_recall",
            0.90,
        ),
        zzu_distribution_minimum=_endpoint_minimum(
            primary,
            "zzu_external_distribution_recall",
            0.90,
        ),
        output_root=_relative_path(one_shot.get("output_root"), "output root"),
        claim_path=_relative_path(claim.get("path"), "external claim path"),
        # The preserved original was frozen before download and therefore has
        # no authoritative raw-source byte table.  Its execution is refused.
        # The v2.1 loader must populate this from successor-parent bytes.
        raw_source_bindings=None,
        seven_zip_tool_binding=None,
        inventory_counts=None,
    )


def _successor_inventory_count_binding(
    payload: Mapping[str, object],
) -> SuccessorInventoryCountBinding:
    external_sources = _mapping(
        payload.get("external_sources"),
        "successor external sources",
    )
    challenge = _mapping(
        external_sources.get(CHALLENGE_2011_DATASET),
        "successor Challenge source",
    )
    zzu = _mapping(
        external_sources.get(ZZU_PEDIATRIC_DATASET),
        "successor ZZU source",
    )
    upstream = _mapping(zzu.get("upstream_counts"), "successor ZZU upstream counts")
    observed = _mapping(
        zzu.get("observed_metadata_only_counts"),
        "successor ZZU observed metadata counts",
    )
    inventory_contract = _mapping(
        payload.get("successor_inventory_contract"),
        "successor inventory contract",
    )
    selected = _mapping(
        inventory_contract.get("exact_selected_counts"),
        "successor exact selected counts",
    )
    exclusions = _mapping(
        inventory_contract.get("exact_zzu_exclusion_counts"),
        "successor exact ZZU exclusion counts",
    )
    count_invariants = _mapping(
        inventory_contract.get("count_invariants"),
        "successor inventory count invariants",
    )
    expected_upstream = {
        "nine_lead_records": 1_856,
        "patients_total": 11_643,
        "records_total": 14_190,
        "twelve_lead_records": 12_334,
    }
    expected_observed = {
        "candidate_patients": 11_643,
        "candidate_records": 14_190,
        "excluded_non_12_lead": 1_856,
        "excluded_other": 0,
        "excluded_under_10_seconds": 6,
        "selected_patients": 10_350,
        "selected_records": 12_328,
    }
    expected_selected = {
        "challenge_records": 1_000,
        "total_records": 13_328,
        "zzu_patients": 10_350,
        "zzu_records": 12_328,
    }
    if (
        challenge.get("expected_records") != 1_000
        or dict(upstream) != expected_upstream
        or dict(observed) != expected_observed
        or dict(selected) != expected_selected
        or dict(exclusions) != dict(EXPECTED_SUCCESSOR_ZZU_EXCLUSION_COUNTS)
        or count_invariants
        != {
            "challenge_plus_zzu_selected_equals_total_records": True,
            "pediatric_12_lead_flag_false_equals_nine_lead_records": True,
            "zzu_selected_patients_not_greater_than_zzu_selected_records": True,
            "zzu_selected_plus_exclusions_equals_candidate_records": True,
            "zzu_selected_plus_nonflag_exclusions_equals_twelve_lead_records": True,
            "zzu_twelve_lead_plus_nine_lead_equals_candidate_records": True,
        }
    ):
        raise OODExternalV2ConfigError(
            "successor inventory counts differ from the frozen metadata contract"
        )
    binding = SuccessorInventoryCountBinding(
        challenge_records=1_000,
        zzu_candidate_records=14_190,
        zzu_candidate_patients=11_643,
        zzu_twelve_lead_records=12_334,
        zzu_nine_lead_records=1_856,
        zzu_records=12_328,
        zzu_patients=10_350,
        total_records=13_328,
        zzu_exclusion_counts=cast(Mapping[str, int], exclusions),
    )
    if {
        "challenge_records": binding.challenge_records,
        "total_records": binding.total_records,
        "zzu_candidate_patients": binding.zzu_candidate_patients,
        "zzu_candidate_records": binding.zzu_candidate_records,
        "zzu_nine_lead_records": binding.zzu_nine_lead_records,
        "zzu_patients": binding.zzu_patients,
        "zzu_records": binding.zzu_records,
        "zzu_twelve_lead_records": binding.zzu_twelve_lead_records,
    } != dict(EXPECTED_SUCCESSOR_INVENTORY_COUNTS):
        raise OODExternalV2ConfigError("successor inventory count binding differs")
    return binding


def verify_successor_parent_preflight(
    path: str | Path,
    *,
    project_root: str | Path,
) -> SuccessorParentPreflight:
    """Verify the draft/frozen v2.1 lineage without enabling execution."""

    root_path = _strict_project_root(project_root)
    source = Path(os.path.abspath(os.fspath(path)))
    expected_source = root_path.joinpath(
        *PurePosixPath(SUCCESSOR_PARENT_CONFIG_PATH).parts
    )
    if source != expected_source or _is_indirect(source) or not source.is_file():
        raise OODExternalV2ConfigError(
            "successor parent must use its exact canonical project path"
        )
    raw = _read_bounded(source, _CONFIG_MAX_BYTES, "successor parent protocol")
    try:
        text = raw.decode("utf-8")
        _reject_duplicate_yaml_keys(text)
        decoded: object = yaml.safe_load(text)
    except (UnicodeError, yaml.YAMLError) as error:
        raise OODExternalV2ConfigError(
            "successor parent is not valid unique-key UTF-8 YAML"
        ) from error
    payload = _mapping(decoded, "successor parent")
    status = payload.get("status")
    frozen_at = payload.get("frozen_at_utc")
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_id") != SUCCESSOR_PROTOCOL_ID
        or payload.get("research_only") is not True
        or status
        not in {
            "draft_successor_preregistration_pre_waveform",
            "frozen_parent_preregistration_pre_waveform",
        }
        or (
            status == "draft_successor_preregistration_pre_waveform"
            and frozen_at is not None
        )
        or (
            status == "frozen_parent_preregistration_pre_waveform"
            and not isinstance(frozen_at, str)
        )
    ):
        raise OODExternalV2ConfigError("successor parent identity or freeze state differs")

    design = _mapping(payload.get("design_history"), "successor design history")
    predecessor = _mapping(design.get("predecessor"), "successor predecessor")
    if predecessor != {
        "config_file_sha256": EXPECTED_PARENT_CONFIG_SHA256,
        "config_path": PARENT_CONFIG_DEFAULT,
        "original_claim_or_output_created": False,
        "protocol_id": ORIGINAL_PROTOCOL_ID,
        "status": "PRE_INFERENCE_PROTOCOL_INFEASIBLE",
        "termination_reason": "exact_ZZU_augmented_lead_name_case_mismatch",
        "waveform_or_model_access_occurred": False,
    }:
        raise OODExternalV2ConfigError("successor predecessor declaration differs")
    amendment = _mapping(
        design.get("pre_inventory_remote_preflight"),
        "successor pre-inventory amendment",
    )
    amendment_revision = _mapping(
        amendment.get("amendment_revision_contract"),
        "successor amendment revision contract",
    )
    expected_amendment = {
        "amendment": (
            "allow_only_the_exact_pinned_v1_backup_tag_and_still_forbid_every_other_ref"
        ),
        "amendment_revision_contract": {
            "commit_count_after_first_frozen_revision": 1,
            "exact_modified_paths": list(SUCCESSOR_AMENDMENT_MODIFIED_PATHS),
            "sole_parent": FIRST_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
            "status_for_every_path": "modified",
        },
        "first_frozen_at_utc": "2026-08-29T22:03:25Z",
        "first_frozen_implementation_revision": (
            FIRST_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION
        ),
        "first_frozen_parent_config_file_sha256": (
            FIRST_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256
        ),
        "new_successor_archive_header_metadata_or_waveform_byte_read": False,
        "outcome": "refused_before_new_successor_external_raw_or_waveform_access",
        "preserved_predecessor_metadata_inventory_evidence_reread": True,
        "reason": "exact_preexisting_encrypted_v1_backup_tag_was_not_declared",
        "successor_inventory_claim_or_output_created": False,
        "new_successor_waveform_trained_checkpoint_or_inference_access_occurred": (
            False
        ),
    }
    if amendment != expected_amendment or amendment_revision != (
        expected_amendment["amendment_revision_contract"]
    ):
        raise OODExternalV2ConfigError(
            "successor pre-inventory amendment declaration differs"
        )
    private_remote_amendment = _mapping(
        design.get("private_remote_authentication_preflight"),
        "successor private-remote authentication amendment",
    )
    expected_private_remote_amendment = {
        "amendment": (
            "pin_bound_gcm_wincred_noninteractive_authentication_for_exact_private_remote_only"
        ),
        "amendment_revision_contract": {
            "commit_count_after_predecessor_amendment_revision": 1,
            "exact_modified_paths": list(
                SUCCESSOR_PRIVATE_REMOTE_AMENDMENT_MODIFIED_PATHS
            ),
            "sole_parent": SECOND_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
            "status_for_every_path": "modified",
        },
        "authenticated_git_metadata_feasibility_probes_observed": True,
        "credential_observation_boundary": {
            "authorization_scope_runtime_verifiable": False,
            "captured_subprocess_streams": (
                "streamed_with_fixed_caps_without_"
                "logging_hashing_evidence_or_error_disclosure"
            ),
            "credential_store_contents": "intentionally_unbound_and_excluded",
            "git_and_gcm_process_memory_and_https_exchange": (
                "necessarily_contains_secret"
            ),
            "python_launcher_supplied_secret": "forbidden",
        },
        "git_runtime_and_project_source_metadata_reread": True,
        "new_successor_external_source_archive_header_record_metadata_or_waveform_byte_read": (
            False
        ),
        "new_successor_trained_checkpoint_or_inference_access_occurred": False,
        "observed_git_metadata": {
            "anonymous_exact_git_query_denied": True,
            "exact_expected_four_ref_lines": True,
            "release_asset_fetched_or_authenticated": False,
            "repository_visibility": "private",
        },
        "outcome": "refused_before_new_successor_external_source_access",
        "predecessor_amended_frozen_at_utc": "2026-08-29T22:49:01Z",
        "predecessor_amended_implementation_revision": (
            SECOND_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION
        ),
        "predecessor_amended_parent_config_file_sha256": (
            SECOND_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256
        ),
        "preserved_predecessor_metadata_inventory_evidence_reread": True,
        "reasons": [
            "restricted_execution_network_transport_unavailable",
            "private_remote_required_explicit_credential_helper_under_sanitized_git",
        ],
        "successor_inventory_claim_or_output_created": False,
    }
    if private_remote_amendment != expected_private_remote_amendment:
        raise OODExternalV2ConfigError(
            "successor private-remote authentication amendment declaration differs"
        )
    inventory_builder_amendment = _mapping(
        design.get("x3_inventory_builder_preflight"),
        "successor inventory-builder amendment",
    )
    expected_inventory_builder_amendment = {
        "amendment": (
            "pin_exact_Git_install_root_add_single_durable_inventory_build_"
            "authorization_and_harden_inventory_contracts"
        ),
        "amendment_revision_contract": {
            "commit_count_after_predecessor_private_auth_revision": 1,
            "exact_modified_paths": list(
                SUCCESSOR_INVENTORY_BUILDER_AMENDMENT_MODIFIED_PATHS
            ),
            "sole_parent": THIRD_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
            "status_for_every_path": "modified",
        },
        "new_successor_external_source_archive_header_record_metadata_or_"
        "waveform_byte_read": False,
        "new_successor_trained_checkpoint_or_inference_access_occurred": False,
        "operational_amendment_only": True,
        "outcome": "refused_before_successor_parent_or_external_source_access",
        "predecessor_private_auth_frozen_at_utc": "2026-08-30T00:50:40Z",
        "predecessor_private_auth_implementation_revision": (
            THIRD_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION
        ),
        "predecessor_private_auth_parent_config_file_sha256": (
            THIRD_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256
        ),
        "root_cause": (
            "isolated_runtime_PATH_was_System32_only_while_Git_resolution_used_"
            "shutil_which_git"
        ),
        "scientific_protocol_change": False,
        "successor_external_access_armed_marker_created": False,
        "successor_external_one_shot_claim_created": False,
        "successor_inventory_build_authorization_marker_created": False,
        "successor_inventory_created": False,
        "successor_output_root_created": False,
        "successor_parent_protocol_byte_read": False,
    }
    if inventory_builder_amendment != expected_inventory_builder_amendment:
        raise OODExternalV2ConfigError(
            "successor inventory-builder amendment declaration differs"
        )
    runtime_preflight_amendment = _mapping(
        design.get("x4_runtime_provenance_preflight"),
        "successor runtime-preflight amendment",
    )
    expected_runtime_preflight_amendment = {
        "amendment": (
            "validate_canonical_frozen_modules_dynamic_namespaces_python_alias_"
            "native_images_and_exact_host_security_modules_add_controls_only_"
            "preflight_and_issue_unconsumed_x5_authorization"
        ),
        "amendment_revision_contract": {
            "commit_count_after_predecessor_inventory_builder_revision": 1,
            "exact_modified_paths": list(
                SUCCESSOR_RUNTIME_PREFLIGHT_AMENDMENT_MODIFIED_PATHS
            ),
            "sole_parent": FOURTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
            "status_for_every_path": "modified",
        },
        "controls_only_engineering_triage_observed_follow_on_runtime_refusals": True,
        "new_successor_external_source_archive_header_record_metadata_or_"
        "waveform_byte_read": False,
        "new_successor_trained_checkpoint_or_inference_access_occurred": False,
        "new_x5_inventory_build_authorization_id": "x5_inventory_build_attempt_1",
        "observed_failure": "module_falsely_claims_a_frozen_origin",
        "operational_amendment_only": True,
        "outcome": "refused_before_x4_inventory_build_authorization_consumption",
        "predecessor_inventory_builder_frozen_at_utc": "2026-08-30T02:00:56Z",
        "predecessor_inventory_builder_implementation_revision": (
            FOURTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION
        ),
        "predecessor_inventory_builder_parent_config_file_sha256": (
            FOURTH_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256
        ),
        "root_causes": [
            "legitimate_frozen_module_aliases_were_compared_to_sys_modules_keys",
            "empty_dynamic_six_moves_namespace_was_rejected_before_its_bound_owner",
            "relative_torch_dynamic_namespace_placeholders_were_treated_as_file_origins",
            "verified_cpython_alias_native_paths_were_not_mapped_to_the_bound_target",
            "host_injected_security_modules_were_not_exactly_bound",
        ],
        "scientific_protocol_change": False,
        "successor_external_access_armed_marker_created": False,
        "successor_external_one_shot_claim_created": False,
        "successor_inventory_created": False,
        "successor_output_root_created": False,
        "successor_parent_protocol_byte_read": True,
        "x4_inventory_build_authorization_consumed": False,
        "x4_inventory_build_authorization_id": "x4_inventory_build_attempt_1",
        "x4_inventory_build_authorization_marker_created": False,
        "x4_inventory_build_authorization_path": (
            HISTORICAL_X4_INVENTORY_BUILDER_ATTEMPT_PATH
        ),
        "x4_inventory_build_authorization_state": "RETIRED_UNCONSUMED",
    }
    if runtime_preflight_amendment != expected_runtime_preflight_amendment:
        raise OODExternalV2ConfigError(
            "successor runtime-preflight amendment declaration differs"
        )
    gcm_scratch_amendment = _mapping(
        design.get("x5_gcm_scratch_cleanup_preflight"),
        "successor GCM scratch-cleanup amendment",
    )
    expected_gcm_scratch_amendment = {
        "amendment": (
            "handle_bind_verify_and_delete_exact_empty_gcm_system_commandline_"
            "sentinel_after_each_bound_process_and_in_all_four_launchers_then_"
            "issue_unconsumed_x6_authorization"
        ),
        "amendment_revision_contract": {
            "commit_count_after_predecessor_runtime_preflight_revision": 1,
            "exact_modified_paths": list(
                SUCCESSOR_GCM_SCRATCH_AMENDMENT_MODIFIED_PATHS
            ),
            "sole_parent": FIFTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
            "status_for_every_path": "modified",
        },
        "inner_outcome": "INVENTORY_BUILDER_PREFLIGHT_VERIFIED",
        "inner_report": {
            "authorization_consumed": False,
            "official_source_content_accessed": False,
            "protocol_artifact_written": False,
            "stage": "complete",
            "status": "OOD_V2_INVENTORY_PREFLIGHT_VERIFIED",
        },
        "invocation_mode": "controls_only_preflight",
        "new_successor_external_source_archive_header_record_metadata_or_"
        "waveform_byte_read": False,
        "new_successor_trained_checkpoint_or_inference_access_occurred": False,
        "new_x6_inventory_build_authorization_id": "x6_inventory_build_attempt_1",
        "observed_failure": (
            "exact_empty_gcm_system_commandline_sentinel_directory_remained"
        ),
        "observed_runtime_residue": {
            "entry_count": 0,
            "entry_kind": "direct_directory",
            "gcm_version": EXPECTED_GIT_CREDENTIAL_MANAGER_VERSION,
            "relative_path": (
                f"temp/{GCM_SYSTEM_COMMANDLINE_SENTINEL_DIRECTORY_NAME}"
            ),
        },
        "operational_amendment_only": True,
        "outer_outcome": "refused_during_isolated_runtime_root_cleanup",
        "overall_controls_only_preflight_succeeded": False,
        "predecessor_runtime_preflight_frozen_at_utc": "2026-08-30T03:08:59Z",
        "predecessor_runtime_preflight_implementation_revision": (
            FIFTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION
        ),
        "predecessor_runtime_preflight_parent_config_file_sha256": (
            FIFTH_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256
        ),
        "scientific_protocol_change": False,
        "successor_child_execution_contract_created": False,
        "successor_external_access_armed_marker_created": False,
        "successor_external_one_shot_claim_created": False,
        "successor_inventory_created": False,
        "successor_output_root_created": False,
        "successor_parent_protocol_byte_read": True,
        "successor_public_projection_created": False,
        "x5_inventory_build_authorization_consumed": False,
        "x5_inventory_build_authorization_id": "x5_inventory_build_attempt_1",
        "x5_inventory_build_authorization_marker_created": False,
        "x5_inventory_build_authorization_path": (
            HISTORICAL_X5_INVENTORY_BUILDER_ATTEMPT_PATH
        ),
        "x5_inventory_build_authorization_state": "RETIRED_UNCONSUMED",
    }
    if gcm_scratch_amendment != expected_gcm_scratch_amendment:
        raise OODExternalV2ConfigError(
            "successor GCM scratch-cleanup amendment declaration differs"
        )
    inventory_failure_amendment = _mapping(
        design.get("x6_inventory_build_failure"),
        "successor X6 inventory-build failure amendment",
    )
    expected_inventory_failure_amendment = {
        "amendment": (
            "add_exact_stage_tracking_and_sanitized_immutable_failure_receipt_"
            "require_retained_exact_x6_marker_and_issue_one_new_x7_inventory_"
            "build_authorization"
        ),
        "amendment_revision_contract": {
            "commit_count_after_x6_implementation_revision": 1,
            "exact_modified_paths": list(
                SUCCESSOR_INVENTORY_FAILURE_AMENDMENT_MODIFIED_PATHS
            ),
            "sole_parent": SIXTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
            "status_for_every_path": "modified",
        },
        "controls_only_preflight": {
            "authorization_consumed": False,
            "official_source_content_accessed": False,
            "outcome": "passed",
            "protocol_artifact_written": False,
            "reported_stage": "complete",
            "runtime_cleanup_succeeded": True,
        },
        "frozen_at_utc": "2026-08-30T04:40:56Z",
        "implementation_revision": SIXTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        "new_x7_failure_receipt_path": HISTORICAL_X7_INVENTORY_BUILDER_FAILURE_PATH,
        "new_x7_inventory_build_authorization_id": "x7_inventory_build_attempt_1",
        "new_x7_inventory_build_authorization_path": (
            HISTORICAL_X7_INVENTORY_BUILDER_ATTEMPT_PATH
        ),
        "operational_amendment_only": True,
        "parent_config_file_sha256": SIXTH_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256,
        "post_failure_bounded_forensic_observations": {
            "all_10_frozen_source_size_sha256_and_declared_md5_bindings_matched": True,
            "bounded_failure_reason": "no_output_parent_and_generic_x6_stderr",
            "bounded_failure_stage": (
                "archive_closure_or_later_prewrite_in_memory_stage"
            ),
            "challenge_record_count": 1_000,
            "exact_failure_stage_recovered": False,
            "purpose": "operational_failure_localization_only",
            "scientific_analysis_occurred": False,
            "waveform_sample_decode_occurred": False,
            "zzu_candidate_record_count": 14_190,
            "zzu_excluded_record_count": 1_862,
            "zzu_selected_record_count": 12_328,
        },
        "production_attempt": {
            "authorization_consumed": True,
            "authorization_id": "x6_inventory_build_attempt_1",
            "authorization_marker_artifact_sha256": (
                HISTORICAL_X6_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_SHA256
            ),
            "authorization_marker_created": True,
            "authorization_marker_file_sha256": (
                HISTORICAL_X6_INVENTORY_BUILDER_ATTEMPT_FILE_SHA256
            ),
            "authorization_path": HISTORICAL_X6_INVENTORY_BUILDER_ATTEMPT_PATH,
            "authorization_state": "CONSUMED_FAILED_RETAINED",
            "bounded_failure_stage": (
                "archive_closure_or_later_prewrite_in_memory_stage"
            ),
            "distribution_scoring_occurred": False,
            "embedding_extraction_occurred": False,
            "endpoint_or_subgroup_metrics_observed": False,
            "exact_failure_stage": "unrecoverable_due_to_generic_x6_stderr",
            "model_logits_or_probabilities_observed": False,
            "outcome": (
                "failed_after_authorization_consumption_before_output_parent_creation"
            ),
            "quality_policy_execution_occurred": False,
            "runtime_cleanup_succeeded": True,
            "successor_child_execution_contract_created": False,
            "successor_external_access_armed_marker_created": False,
            "successor_external_one_shot_claim_created": False,
            "successor_inventory_created": False,
            "successor_output_root_created": False,
            "successor_output_parent_created": False,
            "successor_public_projection_created": False,
            "waveform_sample_decode_occurred": False,
        },
        "scientific_protocol_change": False,
        "x6_authorization_retention": (
            "exact_marker_permanently_retained_ignored_untracked_and_never_reused"
        ),
    }
    if inventory_failure_amendment != expected_inventory_failure_amendment:
        raise OODExternalV2ConfigError(
            "successor X6 inventory-build failure amendment declaration differs"
        )
    archive_operand_amendment = _mapping(
        design.get("x7_inventory_build_failure"),
        "successor X7 inventory-build failure amendment",
    )
    expected_archive_operand_amendment = {
        "amendment": (
            "normalize_the_already_bound_ZZU_terminal_zip_operand_to_its_exact_"
            "absolute_direct_path_before_isolated_7zip_execution_require_exact_"
            "x7_marker_and_receipt_and_issue_one_new_x8_authorization"
        ),
        "amendment_revision_contract": {
            "commit_count_after_x7_implementation_revision": 1,
            "exact_modified_paths": list(
                SUCCESSOR_ARCHIVE_OPERAND_AMENDMENT_MODIFIED_PATHS
            ),
            "sole_parent": SEVENTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
            "status_for_every_path": "modified",
        },
        "frozen_at_utc": "2026-08-30T06:34:49Z",
        "implementation_revision": SEVENTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        "new_x8_failure_receipt_path": SUCCESSOR_INVENTORY_BUILDER_FAILURE_PATH,
        "new_x8_inventory_build_authorization_id": "x8_inventory_build_attempt_1",
        "new_x8_inventory_build_authorization_path": (
            SUCCESSOR_INVENTORY_BUILDER_ATTEMPT_PATH
        ),
        "operational_amendment_only": True,
        "parent_config_file_sha256": SEVENTH_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256,
        "post_failure_static_code_path_diagnosis": {
            "absolute_user_path_publication": "forbidden",
            "archive_member_selection_or_role_change": False,
            "correction": (
                "normalize_the_already_bound_terminal_zip_operand_to_its_exact_"
                "absolute_direct_path_before_every_isolated_7zip_listing_test_or_"
                "extraction_call"
            ),
            "failure_boundary": "zzu_archive_listing",
            "new_official_source_access_occurred": False,
            "purpose": "operational_failure_localization_only",
            "root_cause": (
                "project_relative_ZZU_terminal_zip_operand_was_interpreted_from_the_"
                "fresh_isolated_7zip_working_directory"
            ),
            "source_free_synthetic_two_volume_probe": {
                "exact_7zip_version": "26.02",
                "listed_total_entry_count": 42_586,
                "listing_elapsed_seconds": 0.34,
                "listing_output_size_bytes": 12_957_423,
                "listing_stdout_limit_bytes": 67_108_864,
                "listing_stdout_limit_fraction": 0.1931,
                "official_source_or_identifier_accessed": False,
                "parser_elapsed_seconds": 1.04,
                "scientific_result_or_claim": False,
                "small_split_exact_multivolume_marker_observed": True,
                "small_split_exact_two_volume_count_observed": True,
                "standard_error": "exact_empty",
                "standards_compliant_split_archive": True,
                "synthetic_record_count": 14_190,
                "synthetic_regular_file_count": 28_380,
            },
        },
        "production_attempt": {
            "authorization_consumed": True,
            "authorization_id": "x7_inventory_build_attempt_1",
            "authorization_marker_artifact_sha256": (
                HISTORICAL_X7_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_SHA256
            ),
            "authorization_marker_created": True,
            "authorization_marker_file_sha256": (
                HISTORICAL_X7_INVENTORY_BUILDER_ATTEMPT_FILE_SHA256
            ),
            "authorization_path": HISTORICAL_X7_INVENTORY_BUILDER_ATTEMPT_PATH,
            "authorization_state": "CONSUMED_FAILED_RETAINED",
            "distribution_scoring_occurred": False,
            "embedding_extraction_occurred": False,
            "endpoint_or_subgroup_metrics_observed": False,
            "exact_failure_stage": "zzu_archive_listing",
            "failure_receipt_artifact_sha256": (
                HISTORICAL_X7_INVENTORY_BUILDER_FAILURE_ARTIFACT_SHA256
            ),
            "failure_receipt_created": True,
            "failure_receipt_file_sha256": (
                HISTORICAL_X7_INVENTORY_BUILDER_FAILURE_FILE_SHA256
            ),
            "failure_receipt_path": HISTORICAL_X7_INVENTORY_BUILDER_FAILURE_PATH,
            "failure_receipt_state": "PRECLAIM_INVENTORY_BUILD_FAILED",
            "failure_stage_ordinal": 8,
            "model_logits_or_probabilities_observed": False,
            "official_source_content_accessed": True,
            "outcome": (
                "failed_after_authorization_consumption_before_inventory_output"
            ),
            "output_state": "NONE",
            "quality_policy_execution_occurred": False,
            "runtime_cleanup_succeeded": True,
            "successor_child_execution_contract_created": False,
            "successor_external_access_armed_marker_created": False,
            "successor_external_one_shot_claim_created": False,
            "successor_inventory_created": False,
            "successor_public_projection_created": False,
            "waveform_sample_decode_occurred": False,
        },
        "scientific_protocol_change": False,
        "x7_authorization_and_receipt_retention": (
            "exact_artifacts_permanently_retained_ignored_untracked_and_never_reused"
        ),
    }
    if archive_operand_amendment != expected_archive_operand_amendment:
        raise OODExternalV2ConfigError(
            "successor X7 archive-operand amendment declaration differs"
        )
    child_freeze_amendment = _mapping(
        design.get("x8_inventory_build_success_and_pre_x9_child_freeze_failures"),
        "successor X8 inventory success and X9 child-freeze amendment",
    )
    expected_child_freeze_amendment = {
        "amendment": (
            "preserve_exact_successful_x8_inventory_add_single_use_x9_child_freeze_"
            "stage_reason_publication_and_failure_observability"
        ),
        "amendment_revision_contract": {
            "commit_count_after_x8_implementation_revision": 1,
            "exact_modified_paths": list(
                SUCCESSOR_CHILD_FREEZE_AMENDMENT_MODIFIED_PATHS
            ),
            "sole_parent": EIGHTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
            "status_for_every_path": "modified",
        },
        "frozen_at_utc": EIGHTH_FROZEN_SUCCESSOR_PARENT_FROZEN_AT_UTC,
        "implementation_revision": EIGHTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        "new_x9_child_freeze_authorization_id": "x9_child_freeze_attempt_1",
        "new_x9_child_freeze_authorization_path": (
            HISTORICAL_X9_CHILD_FREEZE_ATTEMPT_PATH
        ),
        "new_x9_child_freeze_failure_receipt_path": (
            HISTORICAL_X9_CHILD_FREEZE_FAILURE_PATH
        ),
        "operational_amendment_only": True,
        "parent_config_file_sha256": EIGHTH_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256,
        "pre_x9_child_freeze_attempts": {
            "count": 2,
            "evaluation_or_metric_execution_occurred": False,
            "exact_failure_stage": "UNKNOWN",
            "external_one_shot_claim_created": False,
            "failure_receipt_created": False,
            "generic_failure_status": "OOD_EXTERNAL_V2_CHILD_FREEZE_FAILED",
            "observed_child_state_after_each_attempt": "ABSENT",
            "official_source_content_accessed": "UNKNOWN",
            "retrospective_failure_receipt_fabrication": "forbidden",
            "runtime_cleanup_state_after_each_attempt": "CLEAN",
            "successor_output_root_created": False,
            "trained_checkpoint_or_model_access_occurred": False,
        },
        "production_inventory_build": {
            "authorization_consumed": True,
            "authorization_id": "x8_inventory_build_attempt_1",
            "authorization_marker_artifact_sha256": (
                HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_SHA256
            ),
            "authorization_marker_created": True,
            "authorization_marker_file_sha256": (
                HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_FILE_SHA256
            ),
            "authorization_path": HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_PATH,
            "authorization_state": "CONSUMED_SUCCEEDED_RETAINED",
            "evaluation_or_metric_execution_occurred": False,
            "exact_archive_closure_summaries_revalidated_privately": True,
            "exact_selected_counts": {
                "challenge_records": 1_000,
                "total_records": 13_328,
                "zzu_patients": 10_350,
                "zzu_records": 12_328,
            },
            "external_one_shot_claim_created": False,
            "failure_receipt_created": False,
            "failure_receipt_must_remain_absent": True,
            "failure_receipt_path": HISTORICAL_X8_INVENTORY_BUILDER_FAILURE_PATH,
            "outcome": "INVENTORY_BUILDER_COMPLETED",
            "private_inventory": {
                "artifact_sha256": HISTORICAL_X8_PRIVATE_INVENTORY_ARTIFACT_SHA256,
                "file_sha256": HISTORICAL_X8_PRIVATE_INVENTORY_FILE_SHA256,
                "path": SUCCESSOR_PRIVATE_INVENTORY_PATH,
            },
            "project_source_tree_sha256": (
                HISTORICAL_X8_INVENTORY_BUILDER_PROJECT_SOURCE_TREE_SHA256
            ),
            "public_projection": {
                "artifact_sha256": (
                    HISTORICAL_X8_PUBLIC_PROJECTION_ARTIFACT_SHA256
                ),
                "file_sha256": HISTORICAL_X8_PUBLIC_PROJECTION_FILE_SHA256,
                "path": SUCCESSOR_PUBLIC_PROJECTION_PATH,
            },
            "quality_policy_execution_occurred": False,
            "runtime_cleanup_succeeded": True,
            "successor_output_root_created": False,
            "trained_checkpoint_or_model_access_occurred": False,
            "waveform_sample_decode_occurred": False,
        },
        "scientific_protocol_change": False,
        "x8_failure_receipt_absence": "required",
        "x8_inventory_retention": (
            "exact_private_inventory_public_projection_and_authorization_marker_"
            "preserved_without_rebuild_mutation_or_reuse"
        ),
    }
    if child_freeze_amendment != expected_child_freeze_amendment:
        raise OODExternalV2ConfigError(
            "successor X8/X9 child-freeze amendment declaration differs"
        )
    x10_amendment = _mapping(
        design.get("x9_child_freeze_failure_and_x10_authorization"),
        "successor X9 failure and X10 child-freeze amendment",
    )
    if x10_amendment != {
        "amendment": (
            "bind_the_legacy_demo_policy_by_its_exact_file_hash_only_preserve_the_"
            "success_manifest_logical_identity_in_its_own_role_validate_decisions_"
            "and_runtime_in_preflight_validate_full_nested_child_bytes_before_"
            "publication_and_issue_one_new_x10_authorization"
        ),
        "amendment_revision_contract": {
            "commit_count_after_x9_implementation_revision": 1,
            "exact_modified_paths": list(
                SUCCESSOR_CHILD_FREEZE_DECISION_BINDING_AMENDMENT_MODIFIED_PATHS
            ),
            "sole_parent": NINTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
            "status_for_every_path": "modified",
        },
        "decision_json_or_scientific_setting_change": False,
        "new_x10_child_freeze_authorization_id": "x10_child_freeze_attempt_1",
        "new_x10_child_freeze_authorization_path": (
            SUCCESSOR_CHILD_FREEZE_ATTEMPT_PATH
        ),
        "new_x10_child_freeze_failure_receipt_path": (
            SUCCESSOR_CHILD_FREEZE_FAILURE_PATH
        ),
        "operational_amendment_only": True,
        "root_cause": {
            "category": "legacy_demo_policy_logical_identity_misbinding",
            "failure_was_operational_not_scientific": True,
            "legacy_demo_policy_file_sha256": EXPECTED_DEMO_POLICY_FILE_SHA256,
            "legacy_demo_policy_path": (
                "artifacts/demo/ptbxl_matched_equal_budget_v1/"
                "resnet1d-seed2026.coverage80.demo-policy.json"
            ),
            "legacy_demo_policy_top_level_artifact_sha256": "ABSENT",
            "source_calibration_decision_binding_verified_exact": True,
            "v1_success_manifest_artifact_sha256": (
                "sha256:6f97e0697d661372e62f4aee9245f26014312e6a1d681615314bc9fcb77c5732"
            ),
        },
        "scientific_protocol_change": False,
        "x9_authorization_and_receipt_retention": (
            "exact_artifacts_permanently_retained_ignored_untracked_and_never_reused"
        ),
        "x9_frozen_at_utc": NINTH_FROZEN_SUCCESSOR_PARENT_FROZEN_AT_UTC,
        "x9_implementation_revision": NINTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        "x9_parent_config_file_sha256": NINTH_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256,
        "x9_production_child_freeze": {
            "authorization_consumed": True,
            "authorization_id": "x9_child_freeze_attempt_1",
            "authorization_marker_artifact_sha256": (
                HISTORICAL_X9_CHILD_FREEZE_ATTEMPT_ARTIFACT_SHA256
            ),
            "authorization_marker_created": True,
            "authorization_marker_file_sha256": (
                HISTORICAL_X9_CHILD_FREEZE_ATTEMPT_FILE_SHA256
            ),
            "authorization_path": HISTORICAL_X9_CHILD_FREEZE_ATTEMPT_PATH,
            "authorization_state": "CONSUMED_FAILED_RETAINED",
            "child_contract_created": False,
            "evaluation_or_metric_execution_occurred": False,
            "exact_failure_stage": "decision_and_child_materialization",
            "external_one_shot_claim_created": False,
            "failure_reason": "STAGE_REFUSED",
            "failure_receipt_artifact_sha256": (
                HISTORICAL_X9_CHILD_FREEZE_FAILURE_ARTIFACT_SHA256
            ),
            "failure_receipt_created": True,
            "failure_receipt_file_sha256": (
                HISTORICAL_X9_CHILD_FREEZE_FAILURE_FILE_SHA256
            ),
            "failure_receipt_path": HISTORICAL_X9_CHILD_FREEZE_FAILURE_PATH,
            "failure_stage_ordinal": 9,
            "official_source_content_accessed": True,
            "output_state": "NONE",
            "quality_policy_execution_occurred": False,
            "runtime_cleanup_succeeded": True,
            "successor_output_root_created": False,
            "trained_checkpoint_or_model_access_occurred": False,
            "waveform_sample_decode_occurred": False,
        },
    }:
        raise OODExternalV2ConfigError(
            "successor X9/X10 child-freeze amendment declaration differs"
        )
    revision_boundary = _mapping(
        payload.get("revision_boundary"),
        "successor revision boundary",
    )
    remote = _mapping(revision_boundary.get("remote"), "successor remote boundary")
    if remote != {
        "allowed_static_remote_ref": {
            "local_object_type": "commit",
            "must_be_ancestor_of_every_required_main_revision": True,
            "name": EXPECTED_GIT_REMOTE_BACKUP_TAG_REF,
            "purpose": "encrypted_v1_private_evidence_backup",
            "remote_form": "lightweight_direct_commit_without_peeled_line",
            "revision": EXPECTED_GIT_REMOTE_BACKUP_TAG_REVISION,
        },
        "authentication": {
            "command_local_config_in_exact_order": list(PRIVATE_REMOTE_GIT_CONFIG),
            "credential_helper": {
                "covered_by_bound_mingw64_tree": True,
                "credential_store": "windows_credential_manager",
                "executable_name": EXPECTED_GIT_CREDENTIAL_MANAGER_NAME,
                "file_sha256": EXPECTED_GIT_CREDENTIAL_MANAGER_SHA256,
                "implementation": "git_credential_manager",
                "namespace": "git",
                "size_bytes": EXPECTED_GIT_CREDENTIAL_MANAGER_SIZE_BYTES,
                "version": EXPECTED_GIT_CREDENTIAL_MANAGER_VERSION,
            },
            "failure": "generic_fail_closed_without_helper_output_or_fallback",
            "gcm_environment_exact": dict(PRIVATE_REMOTE_GCM_ENVIRONMENT),
            "process_boundary": {
                "atomic_process_attributes": {
                    "handle_list": f"0x{WINDOWS_PROCESS_ATTRIBUTE_HANDLE_LIST:08X}",
                    "job_list": f"0x{WINDOWS_PROCESS_ATTRIBUTE_JOB_LIST:08X}",
                },
                "creation_flags": {
                    "create_no_window": f"0x{WINDOWS_CREATE_NO_WINDOW:08X}",
                    "create_unicode_environment": (
                        f"0x{WINDOWS_CREATE_UNICODE_ENVIRONMENT:08X}"
                    ),
                    "extended_startupinfo_present": (
                        f"0x{WINDOWS_EXTENDED_STARTUPINFO_PRESENT:08X}"
                    ),
                },
                "inherited_handles_in_exact_order": [
                    "os_devnull_standard_input",
                    "bounded_standard_output_pipe_writer",
                    "bounded_standard_error_pipe_writer",
                ],
                "job_object": {
                    "active_processes_after_cleanup": 0,
                    "descendant_breakaway": "forbidden",
                    "limit_kill_on_job_close": (
                        f"0x{WINDOWS_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE:08X}"
                    ),
                    "name": "unnamed",
                },
                "launcher": "create_process_w_with_startupinfoex",
                "platform": "windows",
                "stream_capture": {
                    "caller_timeouts_seconds": {
                        "gcm_version": GCM_VERSION_TIMEOUT_SECONDS,
                        "git_remote": PRIVATE_REMOTE_TIMEOUT_SECONDS,
                    },
                    "cleanup_timeout_seconds": (
                        WINDOWS_PRIVATE_PROCESS_CLEANUP_TIMEOUT_SECONDS
                    ),
                    "completed_stream_bytes": (
                        "released_after_exact_semantic_validation_without_serialization"
                    ),
                    "execution_failure_mutable_buffers": "overwritten_then_cleared",
                    "failure_disclosure": (
                        "generic_without_cause_context_or_captured_bytes"
                    ),
                    "gcm_version_standard_error_limit_bytes": (
                        GCM_VERSION_STDERR_LIMIT_BYTES
                    ),
                    "gcm_version_standard_output_limit_bytes": (
                        GCM_VERSION_STDOUT_LIMIT_BYTES
                    ),
                    "git_remote_standard_error_limit_bytes": (
                        PRIVATE_REMOTE_STDERR_LIMIT_BYTES
                    ),
                    "git_remote_standard_output_limit_bytes": (
                        PRIVATE_REMOTE_STDOUT_LIMIT_BYTES
                    ),
                    "implementation": "concurrent_win32_read_file",
                    "interrupt_or_other_base_exception": (
                        "cleanup_wipe_and_fresh_generic_integrity_error"
                    ),
                    "temporary_files": "forbidden",
                    "timeout_overflow_or_read_failure": (
                        "terminate_job_and_require_zero_active_processes"
                    ),
                },
            },
            "query_scope": "exact_hardcoded_https_url_only",
            "repository_visibility": "private",
            "secret_boundary": {
                "captured_subprocess_streams": (
                    "streamed_with_fixed_caps_without_"
                    "logging_hashing_evidence_or_error_disclosure"
                ),
                "git_and_gcm_process_memory_and_https_exchange": (
                    "necessarily_contains_secret"
                ),
                "url_argv_environment_python_inputs_logs_hashes_evidence_artifacts": (
                    "secret_forbidden"
                ),
            },
            "standard_error": "exact_empty",
            "standard_input": "exact_os_devnull",
            "standard_output": "exact_raw_utf8_ref_advertisement_only",
            "trusted_os_broker": {
                "credential_authorization_scope_runtime_verifiable": False,
                "credential_contents": "operator_managed_unbound_and_excluded",
                "windows_credential_manager_and_clr": (
                    "trusted_but_not_path_free_hash_bound"
                ),
            },
            "visibility_proof": {
                "anonymous_command_local_config_in_exact_order": list(
                    PRIVATE_REMOTE_ANONYMOUS_GIT_CONFIG
                ),
                "anonymous_environment": (
                    "exact_sanitized_git_environment_without_credential_or_proxy_inputs"
                ),
                "anonymous_return_code": 128,
                "anonymous_standard_error": {
                    "exact_ascii": EXPECTED_PRIVATE_REMOTE_ANONYMOUS_STDERR.decode(
                        "ascii"
                    ),
                    "size_bytes": len(EXPECTED_PRIVATE_REMOTE_ANONYMOUS_STDERR),
                    "treatment": (
                        "byte_compared_then_discarded_without_decoding_or_disclosure"
                    ),
                },
                "anonymous_standard_input": "exact_os_devnull",
                "anonymous_standard_output": "exact_empty",
                "authenticated_query_must_follow_and_succeed": True,
                "method": "anonymous_git_denial_then_authenticated_exact_ref_read",
            },
        },
        "any_other_advertised_remote_ref": "forbidden",
        "fetch_url": EXPECTED_GIT_REMOTE_URL,
        "live_remote_exact_lines": [
            "ref:_refs/heads/main_TAB_HEAD",
            "current_required_main_revision_TAB_HEAD",
            "current_required_main_revision_TAB_refs/heads/main",
            (
                f"{EXPECTED_GIT_REMOTE_BACKUP_TAG_REVISION}_TAB_"
                f"{EXPECTED_GIT_REMOTE_BACKUP_TAG_REF}"
            ),
        ],
        "live_remote_query": "git_ls_remote_symref_exact_https_url_all_advertised_refs",
        "must_equal_execution_revision_preclaim_and_post_evaluation": True,
        "must_equal_implementation_revision_at_child_freeze": True,
        "name": EXPECTED_GIT_REMOTE_NAME,
        "push_url": EXPECTED_GIT_REMOTE_URL,
        "tracking_ref": EXPECTED_GIT_REMOTE_MAIN_REF,
    }:
        raise OODExternalV2ConfigError("successor remote declaration differs")
    git_execution = _mapping(
        revision_boundary.get("git_execution"),
        "successor Git execution",
    )
    if git_execution != {
        "executable": "exact_bound_mingw64_bin_git_exe",
        "executable_resolution": (
            "exact_install_root_cmd_launcher_and_mingw64_binary_without_PATH_lookup"
        ),
        "full_mingw64_tree_path_free_hash_bound": True,
        "global_system_includes_replacements_alternates_grafts_shallow_sparse_"
        "and_worktree_config": "forbidden",
        "install_root_windows": EXPECTED_GIT_INSTALL_ROOT,
        "local_config": "exact_section_key_value_allowlist_only",
        "sanitized_environment_and_exact_git_dir_work_tree": True,
        "status_hardening": {
            "core_checkStat": "default",
            "core_fsmonitor": False,
            "core_ignoreStat": False,
            "core_preloadIndex": False,
            "core_trustctime": True,
            "core_untrackedCache": False,
        },
        "version": "git_version_2.53.0.windows.2",
    }:
        raise OODExternalV2ConfigError("successor Git execution declaration differs")
    forbidden_history = _mapping(
        revision_boundary.get("forbidden_private_history"),
        "successor forbidden private history",
    )
    if forbidden_history != {
        "command": (
            "git_log_full_history_all_reflog_format_H_double_dash_exact_glob_pathspecs"
        ),
        "limitation": "unreachable_object_contents_are_not_claimed_absent",
        "paths": list(FORBIDDEN_GIT_HISTORY_PATHS),
        "proven_scope": "protected_pathnames_reachable_from_local_refs_and_reflogs",
        "required_output": "exact_empty_stdout",
        "timing": ["child_freeze", "immediately_preclaim", "post_evaluation"],
    }:
        raise OODExternalV2ConfigError(
            "successor forbidden private history declaration differs"
        )
    predecessor_parent = root_path.joinpath(*PurePosixPath(PARENT_CONFIG_DEFAULT).parts)
    if load_parent_config(predecessor_parent).file_sha256 != EXPECTED_PARENT_CONFIG_SHA256:
        raise OODExternalV2IntegrityError("predecessor parent bytes differ")

    bindings = _mapping(payload.get("bindings"), "successor bindings")
    termination_binding = _mapping(
        bindings.get("predecessor_termination"),
        "predecessor termination binding",
    )
    note_binding = _mapping(
        bindings.get("predecessor_termination_note"),
        "predecessor termination note binding",
    )
    source_calibration_binding = _mapping(
        bindings.get("source_calibration_result"),
        "source-calibration binding",
    )
    historical_demo_binding = _mapping(
        bindings.get("historical_demo_policy"),
        "historical demo-policy binding",
    )
    if termination_binding != {
        "file_sha256": PREDECESSOR_TERMINATION_FILE_SHA256,
        "path": PREDECESSOR_TERMINATION_PATH,
        "required_status": "PRE_INFERENCE_PROTOCOL_INFEASIBLE",
    } or note_binding != {
        "file_sha256": PREDECESSOR_TERMINATION_NOTE_FILE_SHA256,
        "path": PREDECESSOR_TERMINATION_NOTE_PATH,
    }:
        raise OODExternalV2ConfigError("predecessor termination bindings differ")
    if source_calibration_binding != {
        "artifact_sha256": EXPECTED_SOURCE_CALIBRATION_ARTIFACT_SHA256,
        "file_sha256": EXPECTED_SOURCE_CALIBRATION_FILE_SHA256,
        "frozen_components_sha256": (
            "sha256:a3180e2f5da6cfd7b44499e590202416fe669d51099298353926d828ab2004ca"
        ),
        "path": EXPECTED_SOURCE_CALIBRATION_PATH,
        "required_status": "PREPARED_NOT_RELEASE_READY",
    }:
        raise OODExternalV2ConfigError("source-calibration binding differs")
    if historical_demo_binding != {
        "file_sha256": EXPECTED_DEMO_POLICY_FILE_SHA256,
        "identity_contract": "exact_file_sha256_only",
        "path": EXPECTED_DEMO_POLICY_PATH,
        "top_level_artifact_sha256_field": "absent",
        "v1_success_manifest_artifact_sha256_is_not_demo_policy_identity": True,
    }:
        raise OODExternalV2ConfigError("historical demo-policy binding differs")
    termination_path = root_path.joinpath(
        *PurePosixPath(PREDECESSOR_TERMINATION_PATH).parts
    )
    note_path = root_path.joinpath(
        *PurePosixPath(PREDECESSOR_TERMINATION_NOTE_PATH).parts
    )
    if (
        sha256_file(termination_path) != PREDECESSOR_TERMINATION_FILE_SHA256
        or sha256_file(note_path) != PREDECESSOR_TERMINATION_NOTE_FILE_SHA256
    ):
        raise OODExternalV2IntegrityError("predecessor termination evidence changed")
    termination_raw = _read_bounded(
        termination_path,
        _CONFIG_MAX_BYTES,
        "predecessor termination",
    )
    try:
        termination_text = termination_raw.decode("utf-8")
        _reject_duplicate_yaml_keys(termination_text)
        termination_object: object = yaml.safe_load(termination_text)
    except (UnicodeError, yaml.YAMLError) as error:
        raise OODExternalV2IntegrityError(
            "predecessor termination cannot be parsed"
        ) from error
    termination = _mapping(termination_object, "predecessor termination")
    boundary = _mapping(termination.get("boundary_state"), "termination boundary")
    termination_parent = _mapping(termination.get("parent"), "termination parent")
    preflight = _mapping(
        termination.get("preflight_evidence"),
        "termination preflight evidence",
    )
    predecessor_private = _mapping(
        preflight.get("private_inventory"),
        "termination private inventory",
    )
    predecessor_public = _mapping(
        preflight.get("aggregate_public_projection"),
        "termination public projection",
    )
    selected_counts = _mapping(
        preflight.get("selected_counts"),
        "termination selected counts",
    )
    disposition = _mapping(termination.get("disposition"), "termination disposition")
    if (
        termination.get("schema_version") != 1
        or termination.get("artifact_type")
        != "ecg_trust.ood_external_v2_pre_inference_termination"
        or termination.get("protocol_id") != ORIGINAL_PROTOCOL_ID
        or termination.get("status") != "PRE_INFERENCE_PROTOCOL_INFEASIBLE"
        or termination.get("research_only") is not True
        or not boundary
        or any(value is not False for value in boundary.values())
        or termination_parent.get("path") != PARENT_CONFIG_DEFAULT
        or termination_parent.get("file_sha256") != EXPECTED_PARENT_CONFIG_SHA256
        or predecessor_private.get("path") != PREDECESSOR_PREFLIGHT_PRIVATE_PATH
        or predecessor_private.get("file_sha256")
        != PREDECESSOR_PREFLIGHT_PRIVATE_FILE_SHA256
        or predecessor_private.get("inventory_sha256")
        != "sha256:d170f03a6ed5350c0b7b3e0a90b319751e642bb5ec3d86c1bae3325f51ae0966"
        or predecessor_public.get("path") != PREDECESSOR_PREFLIGHT_PUBLIC_PATH
        or predecessor_public.get("file_sha256")
        != PREDECESSOR_PREFLIGHT_PUBLIC_FILE_SHA256
        or predecessor_public.get("projection_sha256")
        != "sha256:15cc600e2b825b8c9e68502da2e3c579529cb115601dae45233305f58887eed3"
        or selected_counts
        != {
            "challenge_records": 1_000,
            "total_records": 13_328,
            "zzu_patients": 10_350,
            "zzu_records": 12_328,
        }
        or disposition.get("original_claim_and_output_paths_must_never_be_used")
        is not True
        or disposition.get("retry_under_original_protocol") != "forbidden"
    ):
        raise OODExternalV2IntegrityError("predecessor termination semantics differ")
    for relative_path, expected_hash in (
        (PREDECESSOR_PREFLIGHT_PRIVATE_PATH, PREDECESSOR_PREFLIGHT_PRIVATE_FILE_SHA256),
        (PREDECESSOR_PREFLIGHT_PUBLIC_PATH, PREDECESSOR_PREFLIGHT_PUBLIC_FILE_SHA256),
    ):
        evidence_path = root_path.joinpath(*PurePosixPath(relative_path).parts)
        if _is_indirect(evidence_path) or sha256_file(evidence_path) != expected_hash:
            raise OODExternalV2IntegrityError("predecessor preflight evidence changed")
    for forbidden_relative in (PREDECESSOR_OUTPUT_PATH, PREDECESSOR_CLAIM_PATH):
        forbidden = root_path.joinpath(*PurePosixPath(forbidden_relative).parts)
        if forbidden.exists() or _is_indirect(forbidden):
            raise OODExternalV2IntegrityError(
                "original v2 claim or output path was unexpectedly used"
            )

    inventory_counts = _successor_inventory_count_binding(payload)
    inventory_contract = _mapping(
        payload.get("successor_inventory_contract"),
        "successor inventory contract",
    )
    one_shot = _mapping(payload.get("one_shot_external_access"), "successor one shot")
    claim = _mapping(one_shot.get("external_claim"), "successor external claim")
    inventory_authorization = _mapping(
        one_shot.get("inventory_build_authorization"),
        "successor inventory build authorization",
    )
    child_freeze_authorization = _mapping(
        one_shot.get("child_freeze_authorization"),
        "successor child freeze authorization",
    )
    postclaim_no_retry = _mapping(
        one_shot.get("postclaim_no_retry"),
        "successor postclaim no-retry contract",
    )
    if (
        inventory_contract.get("private_path") != SUCCESSOR_PRIVATE_INVENTORY_PATH
        or inventory_contract.get("public_projection_path")
        != SUCCESSOR_PUBLIC_PROJECTION_PATH
        or one_shot.get("output_root")
        != "artifacts/trust_sentinel/ood_external_v2_1"
        or claim.get("path")
        != "artifacts/trust_sentinel/.ood_external_v2_1.one-shot-claim.json"
    ):
        raise OODExternalV2ConfigError("successor namespace paths differ")
    if inventory_authorization != {
        "artifact_type": SUCCESSOR_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_TYPE,
        "authorization_id": "x8_inventory_build_attempt_1",
        "contains_external_source_bytes_or_identifiers": False,
        "contains_model_outputs_embeddings_or_scores": False,
        "creation": "atomic_create_new_no_overwrite",
        "distinct_from_external_waveform_one_shot_claim": True,
        "durability": (
            "exact_file_flush_and_parent_directory_durability_or_fail_closed"
        ),
        "external_one_shot_claim_consumed_at_marker_creation": False,
        "failure_after_consumption_requires_new_frozen_amendment_and_new_"
        "authorization_id": True,
        "first_official_source_byte_requires_durable_marker": True,
        "git_ignored_and_untracked": True,
        "historical_x6_authorization": {
            "authorization_id": "x6_inventory_build_attempt_1",
            "marker_artifact_sha256": (
                HISTORICAL_X6_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_SHA256
            ),
            "marker_file_sha256": (
                HISTORICAL_X6_INVENTORY_BUILDER_ATTEMPT_FILE_SHA256
            ),
            "must_remain_present_and_exact": True,
            "path": HISTORICAL_X6_INVENTORY_BUILDER_ATTEMPT_PATH,
            "state": "CONSUMED_FAILED_RETAINED",
        },
        "maximum_consumptions": 1,
        "path": SUCCESSOR_INVENTORY_BUILDER_ATTEMPT_PATH,
        "predecessor_authorization_marker_artifact_sha256": (
            HISTORICAL_X7_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_SHA256
        ),
        "predecessor_authorization_marker_file_sha256": (
            HISTORICAL_X7_INVENTORY_BUILDER_ATTEMPT_FILE_SHA256
        ),
        "predecessor_authorization_must_remain_present_and_exact": True,
        "predecessor_authorization_path": (
            HISTORICAL_X7_INVENTORY_BUILDER_ATTEMPT_PATH
        ),
        "predecessor_authorization_state": "CONSUMED_FAILED_RETAINED",
        "predecessor_consumed_failed_authorization_id": (
            "x7_inventory_build_attempt_1"
        ),
        "predecessor_failure_output_state": "NONE",
        "predecessor_failure_receipt_artifact_sha256": (
            HISTORICAL_X7_INVENTORY_BUILDER_FAILURE_ARTIFACT_SHA256
        ),
        "predecessor_failure_receipt_file_sha256": (
            HISTORICAL_X7_INVENTORY_BUILDER_FAILURE_FILE_SHA256
        ),
        "predecessor_failure_receipt_must_remain_present_and_exact": True,
        "predecessor_failure_receipt_path": (
            HISTORICAL_X7_INVENTORY_BUILDER_FAILURE_PATH
        ),
        "predecessor_failure_stage": "zzu_archive_listing",
        "predecessor_failure_stage_ordinal": 8,
        "predecessor_official_source_content_accessed": True,
        "postconsumption_first_raw_action": (
            "hash_every_official_source_against_exact_parent_size_sha256_and_md5"
        ),
        "preconsumption_requirements": [
            "repeatable_controls_only_preflight_passed_with_no_durable_state",
            "exact_retained_x6_consumed_failed_authorization_marker_verified",
            "exact_retained_x7_consumed_failed_authorization_marker_and_failure_"
            "receipt_verified",
            "x8_authorization_marker_and_failure_receipt_absent",
            "exact_frozen_parent_schema_and_count_invariants_verified",
            "exact_canonical_dataset_root_and_raw_source_path_keysets_verified",
            "exact_bound_7zip_tool_identity_verified",
            "raw_source_declared_path_size_sha256_and_md5_bindings_loaded_from_parent",
        ],
        "failure_receipt": {
            "absolute_or_relative_paths_and_timestamps": "forbidden",
            "any_retry_requires_future_frozen_amendment_and_new_authorization_id": True,
            "artifact_type": SUCCESSOR_INVENTORY_BUILDER_FAILURE_ARTIFACT_TYPE,
            "canonical_json": "utf8_sorted_keys_compact_separators_single_lf",
            "creation": "immutable_atomic_create_new_no_overwrite",
            "durability": (
                "exact_file_flush_and_parent_directory_durability_or_fail_closed"
            ),
            "exact_ordered_failure_stages": list(INVENTORY_BUILDER_ATTEMPT_STAGES),
            "exact_top_level_fields": [
                "artifact_type",
                "authorization_consumed",
                "authorization_id",
                "authorization_marker_artifact_sha256",
                "authorization_marker_file_sha256",
                "contains_external_source_bytes_or_identifiers",
                "contains_model_outputs_embeddings_or_scores",
                "external_one_shot_claim_consumed",
                "failure_requires_new_frozen_amendment_and_authorization_id",
                "failure_stage",
                "failure_stage_ordinal",
                "implementation_revision",
                "official_source_content_accessed",
                "output_state",
                "parent_config_file_sha256",
                "protocol_id",
                "quality_model_score_logit_probability_or_metric_observed",
                "retry_resume_or_reuse_authorized",
                "schema_version",
                "state",
                "waveform_sample_decode_occurred",
                "artifact_sha256",
            ],
            "exception_class_message_traceback_errno_and_process_output": "forbidden",
            "external_source_bytes_or_identifiers": "forbidden",
            "external_source_record_patient_archive_member_or_file_identifiers": (
                "forbidden"
            ),
            "external_waveform_one_shot_claim_consumed": False,
            "failure_stage_ordinal": (
                "zero_based_index_in_exact_ordered_failure_stages"
            ),
            "git_ignored_and_untracked": True,
            "logical_self_hash_field": "artifact_sha256",
            "model_outputs_embeddings_or_scores": "forbidden",
            "output_state_allowlist": list(INVENTORY_BUILDER_OUTPUT_STATES),
            "path": SUCCESSOR_INVENTORY_BUILDER_FAILURE_PATH,
            "retention": "permanent",
            "retry_resume_or_reuse_authorized": False,
            "schema_version": 1,
            "state": "PRECLAIM_INVENTORY_BUILD_FAILED",
            "timing": (
                "exactly_once_after_x8_authorization_consumption_on_any_failure_"
                "before_success"
            ),
            "waveform_quality_model_score_logit_probability_or_metric_observation": (
                "forbidden"
            ),
            "write_failure_does_not_restore_or_repeat_authorization": True,
        },
        "retention": "permanent",
        "retired_x4_and_x5_authorization_paths_must_remain_absent": True,
        "retry_resume_or_reuse": "forbidden",
        "scope": "sole_x8_preclaim_inventory_build_attempt",
        "timing": (
            "after_exact_preflight_path_schema_and_tool_binding_before_first_"
            "official_source_byte"
        ),
    }:
        raise OODExternalV2ConfigError(
            "successor inventory build authorization differs"
        )
    if child_freeze_authorization != {
        "artifact_type": SUCCESSOR_CHILD_FREEZE_ATTEMPT_ARTIFACT_TYPE,
        "authorization_id": "x10_child_freeze_attempt_1",
        "child_contract_binding": {
            "field": "child_freeze_attempt",
            "inventory_builder_attempt_remains_exact_x8_marker": True,
            "legacy_demo_policy_identity_is_exact_file_hash_only": True,
            "x10_failure_receipt_must_be_absent": True,
            "x9_attempt_and_failure_receipt_bound_as_historical_lineage": True,
        },
        "contains_external_source_bytes_or_identifiers": False,
        "contains_model_outputs_embeddings_or_scores": False,
        "create_new": "immutable_atomic_create_new_no_overwrite",
        "distinct_from_external_waveform_one_shot_claim": True,
        "durability": (
            "exact_file_flush_and_parent_directory_durability_or_fail_closed"
        ),
        "exact_ordered_controls_only_preflight_stages": list(
            CHILD_FREEZE_PREFLIGHT_STAGES
        ),
        "external_one_shot_claim_consumed_at_marker_creation": False,
        "failure_after_consumption_requires_new_frozen_amendment_and_new_"
        "authorization_id": True,
        "failure_receipt": {
            "absolute_or_relative_paths_and_timestamps": "forbidden",
            "any_retry_requires_future_frozen_amendment_and_new_authorization_id": (
                True
            ),
            "artifact_type": SUCCESSOR_CHILD_FREEZE_FAILURE_ARTIFACT_TYPE,
            "canonical_json": "utf8_sorted_keys_compact_separators_single_lf",
            "create_new": "immutable_atomic_create_new_no_overwrite",
            "durability": (
                "exact_file_flush_and_parent_directory_durability_or_fail_closed"
            ),
            "exact_ordered_failure_stages": list(CHILD_FREEZE_ATTEMPT_STAGES),
            "exact_top_level_fields": [
                "artifact_type",
                "authorization_consumed",
                "authorization_id",
                "authorization_marker_artifact_sha256",
                "authorization_marker_file_sha256",
                "contains_external_source_bytes_or_identifiers",
                "contains_model_outputs_embeddings_or_scores",
                "external_one_shot_claim_consumed",
                "failure_reason",
                "failure_requires_new_frozen_amendment_and_authorization_id",
                "failure_stage",
                "failure_stage_ordinal",
                "implementation_revision",
                "official_source_content_accessed",
                "output_state",
                "parent_config_file_sha256",
                "protocol_id",
                "quality_model_score_logit_probability_or_metric_observed",
                "retry_resume_or_reuse_authorized",
                "schema_version",
                "state",
                "waveform_sample_decode_occurred",
                "artifact_sha256",
            ],
            "exception_class_message_traceback_errno_and_process_output": "forbidden",
            "external_source_bytes_or_identifiers": "forbidden",
            "external_source_record_patient_archive_member_or_file_identifiers": (
                "forbidden"
            ),
            "external_waveform_one_shot_claim_consumed": False,
            "failure_reason_allowlist": list(CHILD_FREEZE_FAILURE_REASONS),
            "failure_stage_ordinal": (
                "zero_based_index_in_exact_ordered_failure_stages"
            ),
            "git_ignored_and_untracked": True,
            "logical_self_hash_field": "artifact_sha256",
            "model_outputs_embeddings_or_scores": "forbidden",
            "output_state_allowlist": list(CHILD_FREEZE_OUTPUT_STATES),
            "path": SUCCESSOR_CHILD_FREEZE_FAILURE_PATH,
            "retention": "permanent",
            "retry_resume_or_reuse_authorized": False,
            "schema_version": 1,
            "state": "PRECLAIM_CHILD_FREEZE_FAILED",
            "waveform_quality_model_score_logit_probability_or_metric_observation": (
                "forbidden"
            ),
            "write_failure_does_not_restore_or_repeat_authorization": True,
        },
        "first_official_source_content_reverification_requires_durable_marker": True,
        "git_ignored_and_untracked": True,
        "maximum_consumptions": 1,
        "path": SUCCESSOR_CHILD_FREEZE_ATTEMPT_PATH,
        "preconsumption_requirements": [
            "exact_x10_parent_lineage_clean_head_live_remote_history_runtime_and_"
            "source_tree_verified",
            "exact_historical_x8_inventory_authorization_marker_verified",
            "x8_inventory_build_failure_receipt_absent",
            "exact_historical_x9_child_freeze_authorization_marker_and_failure_"
            "receipt_verified",
            "exact_x8_private_inventory_and_public_projection_hashes_counts_and_"
            "semantics_verified",
            "exact_x8_archive_closure_summaries_revalidated_without_inventory_rebuild",
            "exact_legacy_demo_policy_file_hash_only_source_calibration_and_runtime_"
            "decision_bindings_verified",
            "exact_child_timestamp_and_destination_validated",
            "x10_child_freeze_attempt_marker_failure_receipt_and_child_destination_"
            "absent",
            "external_waveform_one_shot_claim_and_successor_output_root_absent",
        ],
        "prepublication_requirements": [
            "full_nested_child_contract_bytes_decode_and_semantic_validation_completed",
            "exact_x8_inventory_and_x9_failure_lineage_reverified",
            "x10_failure_receipt_and_child_destination_absent",
        ],
        "private_inventory_artifact_sha256": (
            HISTORICAL_X8_PRIVATE_INVENTORY_ARTIFACT_SHA256
        ),
        "private_inventory_file_sha256": HISTORICAL_X8_PRIVATE_INVENTORY_FILE_SHA256,
        "private_inventory_path": SUCCESSOR_PRIVATE_INVENTORY_PATH,
        "public_projection_artifact_sha256": (
            HISTORICAL_X8_PUBLIC_PROJECTION_ARTIFACT_SHA256
        ),
        "public_projection_file_sha256": HISTORICAL_X8_PUBLIC_PROJECTION_FILE_SHA256,
        "public_projection_path": SUCCESSOR_PUBLIC_PROJECTION_PATH,
        "retention": "permanent",
        "retry_resume_or_reuse": "forbidden_after_marker_visibility",
        "scope": (
            "sole_x10_child_freeze_attempt_reusing_exact_successful_x8_inventory_"
            "after_consumed_failed_x9"
        ),
        "timing": (
            "after_repeatable_controls_only_preflight_before_first_official_source_"
            "content_reverification"
        ),
        "visibility_consumes_authorization_before_durability_completion": True,
        "x8_frozen_at_utc": EIGHTH_FROZEN_SUCCESSOR_PARENT_FROZEN_AT_UTC,
        "x8_implementation_revision": EIGHTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        "x8_inventory_build_authorization_marker_artifact_sha256": (
            HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_SHA256
        ),
        "x8_inventory_build_authorization_marker_file_sha256": (
            HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_FILE_SHA256
        ),
        "x8_inventory_build_authorization_path": (
            HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_PATH
        ),
        "x8_inventory_build_failure_receipt_path": (
            HISTORICAL_X8_INVENTORY_BUILDER_FAILURE_PATH
        ),
        "x8_inventory_build_failure_receipt_required_absent": True,
        "x8_parent_config_file_sha256": EIGHTH_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256,
        "x8_project_source_tree_sha256": (
            HISTORICAL_X8_INVENTORY_BUILDER_PROJECT_SOURCE_TREE_SHA256
        ),
        "x9_child_freeze_authorization_consumed_failed_retained": True,
        "x9_child_freeze_authorization_marker_artifact_sha256": (
            HISTORICAL_X9_CHILD_FREEZE_ATTEMPT_ARTIFACT_SHA256
        ),
        "x9_child_freeze_authorization_marker_file_sha256": (
            HISTORICAL_X9_CHILD_FREEZE_ATTEMPT_FILE_SHA256
        ),
        "x9_child_freeze_authorization_path": (
            HISTORICAL_X9_CHILD_FREEZE_ATTEMPT_PATH
        ),
        "x9_child_freeze_failure_reason": "STAGE_REFUSED",
        "x9_child_freeze_failure_receipt_artifact_sha256": (
            HISTORICAL_X9_CHILD_FREEZE_FAILURE_ARTIFACT_SHA256
        ),
        "x9_child_freeze_failure_receipt_file_sha256": (
            HISTORICAL_X9_CHILD_FREEZE_FAILURE_FILE_SHA256
        ),
        "x9_child_freeze_failure_receipt_path": (
            HISTORICAL_X9_CHILD_FREEZE_FAILURE_PATH
        ),
        "x9_child_freeze_failure_stage": "decision_and_child_materialization",
        "x9_child_freeze_failure_stage_ordinal": 9,
        "x9_child_freeze_official_source_content_accessed": True,
        "x9_child_freeze_output_state": "NONE",
        "x9_frozen_at_utc": NINTH_FROZEN_SUCCESSOR_PARENT_FROZEN_AT_UTC,
        "x9_implementation_revision": NINTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        "x9_parent_config_file_sha256": NINTH_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256,
    }:
        raise OODExternalV2ConfigError(
            "successor child freeze authorization differs"
        )
    if (
        postclaim_no_retry
        != {
            "absolute_without_operator_or_failure_exception": True,
            "failure_requires_new_protocol_and_new_output_root": True,
            "integrity_replays_declared_by_this_protocol_are_same_claim_"
            "verification_not_retries": True,
            "retry_resume_reuse_or_second_scientific_inference": "forbidden",
            "scope": "after_external_one_shot_claim_entry_first_becomes_visible",
        }
        or "retry_resume_or_second_inference" in one_shot
        or one_shot.get("prerequisites")
        != [
            "parent_protocol_frozen_and_committed",
            "child_execution_contract_with_exact_input_hashes_frozen_and_committed",
            "exact_v1_bundle_verified",
            "exact_raw_source_hashes_and_semantic_roles_rederived",
            "exact_python_scientific_package_tree_and_7zip_identities_verified",
            "quality_implementation_hash_verified",
            "exact_retained_x6_consumed_failed_inventory_build_authorization_marker_verified",
            "exact_retained_x7_consumed_failed_inventory_build_authorization_marker_"
            "and_failure_receipt_verified",
            "durable_x8_inventory_build_authorization_marker_verified",
            "x8_inventory_build_failure_receipt_absent",
            "exact_retained_x9_consumed_failed_child_freeze_authorization_marker_"
            "and_failure_receipt_verified",
            "durable_x10_child_freeze_authorization_marker_verified",
            "x10_child_freeze_failure_receipt_absent",
            "output_root_absent",
            "clean_committed_revision",
        ]
    ):
        raise OODExternalV2ConfigError(
            "successor postclaim or prerequisite declaration differs"
        )

    raw_sources = _mapping(payload.get("raw_source_bindings"), "successor raw sources")
    raw_files = _mapping(raw_sources.get("files"), "successor raw source files")
    if set(raw_files) != set(REQUIRED_RAW_SOURCE_BINDING_KEYS):
        raise OODExternalV2ConfigError("successor raw source set differs")
    parsed_raw: dict[str, RawSourceBinding] = {}
    for name in REQUIRED_RAW_SOURCE_BINDING_KEYS:
        item = _mapping(raw_files[name], f"successor raw source {name}")
        if set(item) != {"expected_md5", "path", "sha256", "size_bytes"}:
            raise OODExternalV2ConfigError("successor raw source fields differ")
        raw_sha = item["sha256"]
        raw_md5 = item["expected_md5"]
        if not isinstance(raw_sha, str) or re.fullmatch(r"[0-9a-f]{64}", raw_sha) is None:
            raise OODExternalV2ConfigError("successor raw source SHA-256 is invalid")
        if raw_md5 is not None and (
            not isinstance(raw_md5, str)
            or re.fullmatch(r"[0-9a-f]{32}", raw_md5) is None
        ):
            raise OODExternalV2ConfigError("successor raw source MD5 is invalid")
        parsed_raw[name] = RawSourceBinding(
            relative_path=_relative_path(item["path"], f"successor raw source {name}"),
            file_sha256=f"sha256:{raw_sha}",
            size_bytes=_positive_integer(item["size_bytes"], f"successor raw source {name}"),
            official_md5=None if raw_md5 is None else f"md5:{raw_md5}",
        )
    if {
        name: binding.relative_path for name, binding in parsed_raw.items()
    } != dict(EXPECTED_RAW_SOURCE_PATHS):
        raise OODExternalV2ConfigError("successor raw source paths differ")

    runtime = _mapping(payload.get("runtime"), "successor runtime")
    module_audit = _mapping(
        runtime.get("loaded_module_origin_audit"),
        "successor loaded-module audit",
    )
    if module_audit != {
        "all_sys_modules_entries_checked": True,
        "built_in_and_frozen_loader_ownership_verified": True,
        "cpython_alias_native_paths_mapped_to_exact_resolved_target": True,
        "dynamic_originless_namespace_requires_bound_file_backed_owner": True,
        "every_loaded_native_image_enumerated": True,
        "exact_main_must_be_one_of_four_bound_operational_entrypoints": True,
        "frozen_module_canonical_spec_and_exact_alias_map_verified": True,
        "host_security_native_module_set": "exact_required_paths_sizes_and_sha256",
        "host_security_native_modules": [
            {
                "path": path,
                "sha256": digest.removeprefix("sha256:"),
                "size_bytes": size,
            }
            for path, (size, digest) in EXPECTED_HOST_SECURITY_NATIVE_MODULES.items()
        ],
        "namespace_search_locations_must_enter_bound_trees": True,
        "non_OS_native_images_must_enter_complete_cpython_or_site_packages_tree": True,
        "process_main_image": "exact_resolved_cpython_base_tree_python_exe",
        "pyc_pyo_and_unbound_file_origins": "forbidden",
        "uv_redirector": (
            "separately_hash_bound_but_not_required_to_remain_loaded"
        ),
    }:
        raise OODExternalV2ConfigError(
            "successor loaded-module audit declaration differs"
        )
    isolated_launcher = _mapping(
        runtime.get("isolated_launcher"),
        "successor isolated launcher",
    )
    gcm_sentinel_cleanup = _mapping(
        isolated_launcher.get("gcm_system_commandline_sentinel_cleanup"),
        "successor GCM system-commandline sentinel cleanup",
    )
    if gcm_sentinel_cleanup != {
        "accepted_precleanup_state": (
            "absent_or_exact_case_sensitive_non_reparse_empty_directory"
        ),
        "bounded_runner_exception_cleanup": (
            "deferred_to_outer_launcher_after_isolated_child_exit"
        ),
        "handle_binding": {
            "creation_disposition": "OPEN_EXISTING",
            "desired_access": [
                "DELETE",
                "FILE_LIST_DIRECTORY",
                "FILE_READ_ATTRIBUTES",
            ],
            "flags": [
                "FILE_FLAG_BACKUP_SEMANTICS",
                "FILE_FLAG_OPEN_REPARSE_POINT",
            ],
            "open_api": "CreateFileW",
            "share_mode": ["FILE_SHARE_READ"],
        },
        "nonempty_indirect_extra_or_raced_content": (
            "fail_closed_and_retain_runtime_root"
        ),
        "outer_launcher_fallback": (
            "after_isolated_child_exit_in_all_four_operational_entrypoints"
        ),
        "outer_launcher_fallback_scope": (
            "only_when_no_prior_outer_cleanup_attempt"
        ),
        "outer_launcher_cleanup_retry_after_attempt": "forbidden",
        "overall_success_requires_child_exit_zero_and_runtime_root_absent": True,
        "process_boundaries": [
            "after_bound_gcm_version_job_runner_returns_and_reports_zero_"
            "active_processes",
            "after_authenticated_private_remote_job_runner_returns_and_"
            "reports_zero_active_processes",
        ],
        "race_closure": {
            "ancestry_revalidated_while_handle_locked": True,
            "close_api": "CloseHandle",
            "deletion_api": "SetFileInformationByHandle",
            "deletion_information_class": "FileDispositionInfo",
            "deletion_target": "exact_same_locked_handle",
            "disposition_delete_file": True,
            "emptiness_verified_while_handle_locked": True,
            "forbidden_attributes": ["FILE_ATTRIBUTE_REPARSE_POINT"],
            "information_api": "GetFileInformationByHandleEx",
            "information_classes": ["FileAttributeTagInfo", "FileIdInfo"],
            "main_handle_identity_and_attributes_reverified_before_delete": True,
            "pathname_reopen_for_deletion": "forbidden",
            "pathname_witness": {
                "close_before_main_recheck": True,
                "creation_disposition": "OPEN_EXISTING",
                "desired_access": [
                    "FILE_LIST_DIRECTORY",
                    "FILE_READ_ATTRIBUTES",
                ],
                "flags": [
                    "FILE_FLAG_BACKUP_SEMANTICS",
                    "FILE_FLAG_OPEN_REPARSE_POINT",
                ],
                "identity_and_attributes": (
                    "exact_match_to_initial_main_handle"
                ),
                "purpose": "identity_only_never_deletion",
                "share_mode": [
                    "FILE_SHARE_READ",
                    "FILE_SHARE_WRITE",
                    "FILE_SHARE_DELETE",
                ],
            },
            "postclose_path_absent_and_non_indirect": "required",
            "postclose_scratch_empty": "required",
            "required_attributes": ["FILE_ATTRIBUTE_DIRECTORY"],
            "stable_identity_fields": [
                "VolumeSerialNumber_uint64",
                "FileId.Identifier_128_bit",
            ],
        },
        "recursive_or_wildcard_deletion": "forbidden",
        "relative_path": (
            f"temp/{GCM_SYSTEM_COMMANDLINE_SENTINEL_DIRECTORY_NAME}"
        ),
        "runtime_scratch_verifier_allowlist": "forbidden",
        "scratch_must_be_exactly_empty_after_cleanup": True,
    }:
        raise OODExternalV2ConfigError(
            "successor GCM sentinel cleanup declaration differs"
        )
    inventory_builder_boundary = _mapping(
        isolated_launcher.get("inventory_builder_boundary"),
        "successor inventory builder boundary",
    )
    if inventory_builder_boundary != {
        "controls_only_failure_disclosure": (
            "stable_stage_code_without_exception_path_or_external_identifier"
        ),
        "controls_only_mode": (
            "exact_shared_preconsumption_path_repeatable_no_marker_raw_content_"
            "or_output_write"
        ),
        "consumed_failed_x6_authorization_marker_must_be_present_and_exact": True,
        "consumed_failed_x7_authorization_marker_and_failure_receipt_must_be_"
        "present_and_exact": True,
        "current_x8_authorization_and_failure_receipt_paths_must_be_absent_"
        "before_authorization": True,
        "historical_x4_and_x5_authorization_paths_must_remain_absent": True,
        "post_x8_consumption_failure_disclosure": (
            "exact_allowlisted_stage_ordinal_and_output_state_without_exception_"
            "timestamp_path_or_external_source_identifier"
        ),
        "postflight_before_success_report": (
            "exact_same_preflight_plus_strict_private_public_output_hashes"
        ),
        "preflight_before_any_raw_or_inventory_read": (
            "frozen_parent_clean_X_live_remote_history_runtime_git_source_main_"
            "and_absent_claim_output"
        ),
        "private_public_inventory_destinations_must_be_absent_before_"
        "authorization": True,
    }:
        raise OODExternalV2ConfigError(
            "successor inventory builder boundary declaration differs"
        )
    tool_payload = _mapping(runtime.get("split_archive_tool"), "split archive tool")
    tool = SevenZipToolBinding(
        implementation=_exact_string(
            tool_payload.get("implementation"), "7zip", "7-Zip implementation"
        ),
        version=_exact_string(tool_payload.get("version"), "26.02", "7-Zip version"),
        executable_name=_exact_string(
            tool_payload.get("executable_name"), "7z.exe", "7-Zip executable"
        ),
        executable_size_bytes=_exact_integer(
            tool_payload.get("executable_size_bytes"), 576_000, "7-Zip executable size"
        ),
        executable_sha256=_exact_string(
            tool_payload.get("executable_sha256"),
            "83967f1b02b43c4efeda302795722c809e0e81b8307de73558d10484d5676a7d",
            "7-Zip executable SHA-256",
        ),
        library_name=_exact_string(
            tool_payload.get("library_name"), "7z.dll", "7-Zip library"
        ),
        library_size_bytes=_exact_integer(
            tool_payload.get("library_size_bytes"), 1_906_688, "7-Zip library size"
        ),
        library_sha256=_exact_string(
            tool_payload.get("library_sha256"),
            "69fd4df057985c40e510e2fac182881c7f85e90aa13ec703f763a8fdb2ce61f8",
            "7-Zip library SHA-256",
        ),
    )
    execution_isolation = _mapping(
        tool_payload.get("execution_isolation"),
        "successor 7-Zip execution isolation",
    )
    if execution_isolation != {
        "PATH": "replica_tool_directory_then_System32_only",
        "application_directory": (
            "fresh_verified_direct_two_file_replica_of_bound_exe_and_dll"
        ),
        "archive_operand_normalization": {
            "absolute_path_serialization_or_publication": "forbidden",
            "applies_to_commands": [
                "listing",
                "archive_test",
                "isolated_extraction",
            ],
            "archive_bytes_members_roles_selection_or_order_changed": False,
            "direct_ancestry_regular_file_and_stable_identity_reverified": True,
            "input": "already_bound_project_relative_ZZU_terminal_zip_path",
            "output": "exact_absolute_direct_archive_path",
            "timing": (
                "after_exact_direct_archive_binding_before_isolated_process_creation"
            ),
        },
        "caller_cwd_and_environment": "forbidden",
        "codecs_formats_or_other_adjacent_plugins": "absent",
        "environment": "minimal_fixed_SystemRoot_System32_TEMP_TMP_and_PATH",
        "replica_and_original_hashes_rechecked_before_and_after_each_call": True,
        "working_directory": "separate_fresh_verified_empty_directory",
    }:
        raise OODExternalV2ConfigError(
            "successor 7-Zip archive-operand isolation declaration differs"
        )
    slt_normalization = _mapping(
        tool_payload.get("windows_slt_member_path_normalization"),
        "successor 7-Zip Windows SLT path normalization",
    )
    if slt_normalization != {
        "accepted_separator_conventions": [
            "forward_slash_only",
            "backslash_only",
        ],
        "backslash_to_forward_slash_before_shared_canonical_validation": True,
        "mixed_separators": "forbidden",
        "rejected_after_normalization": [
            "absolute_rooted_drive_UNC_or_device_paths",
            "traversal_dot_empty_or_trailing_components",
            "Windows_reserved_names_or_control_characters",
            "exact_or_casefolded_collisions",
        ],
        "scope": "bound_windows_7zip_slt_presentation_only",
        "shared_canonical_path_validation_after_normalization": "required",
        "stored_archive_and_evidence_paths_remain_posix_forward_slash_only": True,
    }:
        raise OODExternalV2ConfigError(
            "successor 7-Zip SLT normalization declaration differs"
        )
    return SuccessorParentPreflight(
        path=source,
        file_sha256=sha256_bytes(raw),
        status=status,
        raw_source_bindings=MappingProxyType(parsed_raw),
        seven_zip_tool_binding=tool,
        inventory_counts=inventory_counts,
    )


def load_successor_parent_config(
    path: str | Path,
    *,
    project_root: str | Path,
) -> OODExternalV2ParentConfig:
    """Load v2.1 only after root freezes and explicitly enables its exact hash."""

    preflight = verify_successor_parent_preflight(path, project_root=project_root)
    if (
        preflight.status != "frozen_parent_preregistration_pre_waveform"
        or EXPECTED_SUCCESSOR_PARENT_CONFIG_SHA256 is None
        or preflight.file_sha256 != EXPECTED_SUCCESSOR_PARENT_CONFIG_SHA256
    ):
        raise OODExternalV2ExecutionError(
            "SUCCESSOR_PARENT_NOT_FROZEN: v2.1 execution remains disabled"
        )
    root = _strict_project_root(project_root)
    predecessor = load_parent_config(
        root.joinpath(*PurePosixPath(PARENT_CONFIG_DEFAULT).parts)
    )
    raw = _read_bounded(preflight.path, _CONFIG_MAX_BYTES, "successor parent")
    decoded = _mapping(yaml.safe_load(raw.decode("utf-8")), "successor parent")
    bindings = _mapping(decoded.get("bindings"), "successor bindings")

    def require_binding(
        value: object,
        expected: BoundFile,
        *,
        context: str,
        inner_hash_key: str | None = None,
    ) -> None:
        item = _mapping(value, context)
        if (
            item.get("path") != expected.relative_path
            or item.get("file_sha256") != expected.file_sha256
            or (
                expected.artifact_sha256 is not None
                and item.get("artifact_sha256") != expected.artifact_sha256
            )
            or (
                inner_hash_key is not None
                and item.get(inner_hash_key) != predecessor.resolved_config_sha256
            )
        ):
            raise OODExternalV2ConfigError(f"{context} differs from sealed v1")

    completion = _mapping(bindings.get("v1_completion_bundle"), "v1 bundle")
    require_binding(completion.get("result"), predecessor.v1_result, context="v1 result")
    require_binding(
        completion.get("success_manifest"),
        predecessor.v1_success_manifest,
        context="v1 manifest",
    )
    require_binding(
        completion.get("distribution_policy"),
        predecessor.v1_distribution_policy,
        context="v1 distribution policy",
    )
    require_binding(bindings.get("v1_checkpoint"), predecessor.checkpoint, context="checkpoint")
    require_binding(
        bindings.get("v1_resolved_config"),
        predecessor.resolved_config,
        context="resolved config",
        inner_hash_key="inner_config_sha256",
    )
    require_binding(
        bindings.get("normalization"),
        predecessor.normalization,
        context="normalization",
    )
    require_binding(
        bindings.get("signal_quality_implementation"),
        predecessor.quality_implementation,
        context="quality implementation",
    )
    require_binding(
        bindings.get("dependency_lock"),
        predecessor.dependency_lock,
        context="dependency lock",
    )
    require_binding(
        bindings.get("project_manifest"),
        predecessor.project_manifest,
        context="project manifest",
    )
    evaluation = _mapping(decoded.get("evaluation"), "successor evaluation")
    primary = _mapping(evaluation.get("primary_endpoints"), "successor endpoints")
    bootstrap = _mapping(evaluation.get("bootstrap"), "successor bootstrap")
    challenge_bootstrap = _mapping(bootstrap.get("challenge"), "Challenge bootstrap")
    zzu_bootstrap = _mapping(bootstrap.get("zzu"), "ZZU bootstrap")
    multiplicity = _mapping(evaluation.get("multiplicity"), "successor multiplicity")
    if (
        _endpoint_minimum(primary, "challenge_group3_technical_block_sensitivity", 0.95)
        != predecessor.challenge_group3_minimum
        or _endpoint_minimum(primary, "challenge_group1_quality_pass_rate", 0.90)
        != predecessor.challenge_group1_minimum
        or _endpoint_minimum(primary, "challenge_external_distribution_recall", 0.90)
        != predecessor.challenge_distribution_minimum
        or _endpoint_minimum(primary, "zzu_external_distribution_recall", 0.90)
        != predecessor.zzu_distribution_minimum
        or bootstrap.get("resamples") != 10_000
        or challenge_bootstrap.get("seed") != 20_260_901
        or zzu_bootstrap.get("seed") != 20_260_902
        or multiplicity.get("required_one_sided_confidence_level") != 0.9875
    ):
        raise OODExternalV2ConfigError("successor endpoint/bootstrap constants differ")
    return replace(
        predecessor,
        path=preflight.path,
        file_sha256=preflight.file_sha256,
        status=preflight.status,
        output_root="artifacts/trust_sentinel/ood_external_v2_1",
        claim_path="artifacts/trust_sentinel/.ood_external_v2_1.one-shot-claim.json",
        raw_source_bindings=preflight.raw_source_bindings,
        seven_zip_tool_binding=preflight.seven_zip_tool_binding,
        inventory_counts=preflight.inventory_counts,
    )


def _parse_successor_parent_copy(
    path: Path,
    *,
    expected_file_sha256: str,
) -> tuple[dict[str, object], Mapping[str, RawSourceBinding], SevenZipToolBinding]:
    """Path-neutral parser for the exact manifest-covered successor copy."""

    raw = _read_bounded(path, _CONFIG_MAX_BYTES, "private successor parent copy")
    if (
        EXPECTED_SUCCESSOR_PARENT_CONFIG_SHA256 is None
        or expected_file_sha256 != EXPECTED_SUCCESSOR_PARENT_CONFIG_SHA256
        or sha256_bytes(raw) != expected_file_sha256
    ):
        raise OODExternalV2IntegrityError("private successor parent hash differs")
    try:
        text = raw.decode("utf-8")
        _reject_duplicate_yaml_keys(text)
        decoded: object = yaml.safe_load(text)
    except (UnicodeError, yaml.YAMLError) as error:
        raise OODExternalV2IntegrityError(
            "private successor parent cannot be parsed"
        ) from error
    payload = _mapping(decoded, "private successor parent")
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_id") != SUCCESSOR_PROTOCOL_ID
        or payload.get("status") != "frozen_parent_preregistration_pre_waveform"
        or not isinstance(payload.get("frozen_at_utc"), str)
        or payload.get("research_only") is not True
    ):
        raise OODExternalV2IntegrityError("private successor parent identity differs")
    try:
        _successor_inventory_count_binding(payload)
    except OODExternalV2ConfigError as error:
        raise OODExternalV2IntegrityError(
            "private successor inventory counts differ"
        ) from error
    raw_sources = _mapping(payload.get("raw_source_bindings"), "private raw sources")
    raw_files = _mapping(raw_sources.get("files"), "private raw source files")
    if set(raw_files) != set(REQUIRED_RAW_SOURCE_BINDING_KEYS):
        raise OODExternalV2IntegrityError("private parent raw source set differs")
    parsed_raw: dict[str, RawSourceBinding] = {}
    for name in REQUIRED_RAW_SOURCE_BINDING_KEYS:
        item = _mapping(raw_files[name], f"private raw source {name}")
        raw_sha = item.get("sha256")
        raw_md5 = item.get("expected_md5")
        if (
            set(item) != {"expected_md5", "path", "sha256", "size_bytes"}
            or not isinstance(raw_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", raw_sha) is None
            or (
                raw_md5 is not None
                and (
                    not isinstance(raw_md5, str)
                    or re.fullmatch(r"[0-9a-f]{32}", raw_md5) is None
                )
            )
        ):
            raise OODExternalV2IntegrityError("private parent raw source differs")
        parsed_raw[name] = RawSourceBinding(
            relative_path=_relative_path(item.get("path"), f"private raw source {name}"),
            file_sha256=f"sha256:{raw_sha}",
            size_bytes=_positive_integer(item.get("size_bytes"), f"private raw source {name}"),
            official_md5=None if raw_md5 is None else f"md5:{raw_md5}",
        )
    runtime = _mapping(payload.get("runtime"), "private successor runtime")
    tool_payload = _mapping(runtime.get("split_archive_tool"), "private 7-Zip")
    try:
        tool = SevenZipToolBinding(
            implementation=cast(str, tool_payload.get("implementation")),
            version=cast(str, tool_payload.get("version")),
            executable_name=cast(str, tool_payload.get("executable_name")),
            executable_size_bytes=cast(int, tool_payload.get("executable_size_bytes")),
            executable_sha256=cast(str, tool_payload.get("executable_sha256")),
            library_name=cast(str, tool_payload.get("library_name")),
            library_size_bytes=cast(int, tool_payload.get("library_size_bytes")),
            library_sha256=cast(str, tool_payload.get("library_sha256")),
        )
    except Exception as error:
        raise OODExternalV2IntegrityError("private parent 7-Zip binding differs") from error
    expected_tool = SevenZipToolBinding(
        implementation="7zip",
        version="26.02",
        executable_name="7z.exe",
        executable_size_bytes=576_000,
        executable_sha256=(
            "83967f1b02b43c4efeda302795722c809e0e81b8307de73558d10484d5676a7d"
        ),
        library_name="7z.dll",
        library_size_bytes=1_906_688,
        library_sha256=(
            "69fd4df057985c40e510e2fac182881c7f85e90aa13ec703f763a8fdb2ce61f8"
        ),
    )
    if tool != expected_tool:
        raise OODExternalV2IntegrityError("private parent 7-Zip identity differs")
    return payload, MappingProxyType(parsed_raw), tool


def _load_parent_for_operation(
    path: str | Path,
    *,
    project_root: str | Path,
) -> OODExternalV2ParentConfig:
    root = _strict_project_root(project_root)
    requested = Path(os.path.abspath(os.fspath(path)))
    original = root.joinpath(*PurePosixPath(PARENT_CONFIG_DEFAULT).parts)
    successor = root.joinpath(*PurePosixPath(SUCCESSOR_PARENT_CONFIG_PATH).parts)
    if requested == original:
        return load_parent_config(requested)
    if requested == successor:
        return load_successor_parent_config(requested, project_root=root)
    raise OODExternalV2ConfigError("parent must use an exact canonical project path")


def _archive_closure_summary(
    closure: ArchiveExtractionClosure,
) -> ArchiveClosureSummaryBinding:
    role_counts = Counter(member.role for member in closure.members)
    return ArchiveClosureSummaryBinding(
        dataset=closure.dataset,
        archive_format=closure.archive_format,
        archive_file_count=len(closure.archive_files),
        archive_bytes_total=closure.archive_bytes_total,
        member_count=closure.member_count,
        member_bytes_total=closure.member_bytes_total,
        member_role_counts=(
            role_counts.get("ignored_release_file", 0),
            role_counts.get("quality_reference", 0),
            role_counts.get("wfdb_data", 0),
            role_counts.get("wfdb_header", 0),
        ),
        closure_sha256=closure.closure_sha256,
        tool_binding=closure.tool_binding,
    )


def _archive_closure_summary_dict(
    summary: ArchiveClosureSummaryBinding,
) -> dict[str, object]:
    return {
        "archive_bytes_total": summary.archive_bytes_total,
        "archive_file_count": summary.archive_file_count,
        "archive_format": summary.archive_format,
        "closure_sha256": summary.closure_sha256,
        "dataset": summary.dataset,
        "member_bytes_total": summary.member_bytes_total,
        "member_count": summary.member_count,
        "member_role_counts": {
            role: count
            for role, count in zip(
                _ARCHIVE_MEMBER_ROLES,
                summary.member_role_counts,
                strict=True,
            )
        },
        "tool_binding": (
            None if summary.tool_binding is None else summary.tool_binding.to_dict()
        ),
    }


def _parse_archive_closure_summary(
    value: object,
    *,
    context: str,
) -> ArchiveClosureSummaryBinding:
    payload = _mapping(value, context)
    expected = {
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
    if set(payload) != expected:
        raise OODExternalV2ConfigError(f"{context} fields differ from protocol")
    dataset = payload["dataset"]
    archive_format = payload["archive_format"]
    if dataset not in {CHALLENGE_2011_DATASET, ZZU_PEDIATRIC_DATASET} or archive_format not in {
        "tar_gzip",
        "split_zip_7zip",
    }:
        raise OODExternalV2ConfigError(f"{context} identity is invalid")
    raw_roles = _mapping(payload["member_role_counts"], f"{context} roles")
    if set(raw_roles) != set(_ARCHIVE_MEMBER_ROLES):
        raise OODExternalV2ConfigError(f"{context} role fields differ")
    role_counts = tuple(
        _nonnegative_integer(raw_roles[role], f"{context} role {role}")
        for role in _ARCHIVE_MEMBER_ROLES
    )
    raw_tool = payload["tool_binding"]
    try:
        tool_binding = (
            None
            if raw_tool is None
            else SevenZipToolBinding.from_dict(_mapping(raw_tool, f"{context} tool"))
        )
    except Exception as error:
        raise OODExternalV2ConfigError(f"{context} tool binding is invalid") from error
    return ArchiveClosureSummaryBinding(
        dataset=dataset,
        archive_format=archive_format,
        archive_file_count=_positive_integer(
            payload["archive_file_count"],
            f"{context} archive file count",
        ),
        archive_bytes_total=_positive_integer(
            payload["archive_bytes_total"],
            f"{context} archive bytes",
        ),
        member_count=_positive_integer(payload["member_count"], f"{context} members"),
        member_bytes_total=_positive_integer(
            payload["member_bytes_total"],
            f"{context} member bytes",
        ),
        member_role_counts=cast(tuple[int, int, int, int], role_counts),
        closure_sha256=_digest(payload["closure_sha256"], f"{context} closure"),
        tool_binding=tool_binding,
    )


def _assert_production_archive_closures(
    closures: tuple[ArchiveExtractionClosure, ...],
    *,
    expected_seven_zip_tool: SevenZipToolBinding | None = None,
) -> tuple[ArchiveClosureSummaryBinding, ...]:
    if tuple(closure.dataset for closure in closures) != (
        CHALLENGE_2011_DATASET,
        ZZU_PEDIATRIC_DATASET,
    ):
        raise OODExternalV2IntegrityError(
            "inventory must contain exactly the two canonical archive closures"
        )
    summaries = tuple(_archive_closure_summary(closure) for closure in closures)
    challenge, zzu = summaries
    expected_challenge_roles = (1_001, 3, 1_000, 1_000)
    expected_zzu_roles = (0, 0, 14_190, 14_190)
    if (
        challenge.archive_format != "tar_gzip"
        or challenge.archive_file_count != 1
        or challenge.member_count != 3_004
        or challenge.member_role_counts != expected_challenge_roles
        or challenge.tool_binding is not None
        or zzu.archive_format != "split_zip_7zip"
        or zzu.archive_file_count != 2
        or zzu.member_count != 28_380
        or zzu.member_role_counts != expected_zzu_roles
        or zzu.tool_binding is None
        or (
            expected_seven_zip_tool is not None
            and zzu.tool_binding != expected_seven_zip_tool
        )
    ):
        raise OODExternalV2IntegrityError(
            "archive closure release-tree counts or roles differ from v2.1"
        )
    return summaries


def _verify_archive_closure_rebuilds(
    inventory: ExternalWaveformInventory,
    *,
    dataset_roots: Mapping[str, Path],
    raw_source_paths: Mapping[str, Path],
    seven_zip_executable: str | Path,
    expected_seven_zip_tool: SevenZipToolBinding | None = None,
    stage_callback: Callable[[str], None] | None = None,
) -> None:
    if stage_callback is not None and not callable(stage_callback):
        raise TypeError("stage_callback must be callable or None")
    summaries = _assert_production_archive_closures(
        inventory.archive_closures,
        expected_seven_zip_tool=expected_seven_zip_tool,
    )
    challenge, zzu = inventory.archive_closures
    requested_tool = Path(os.fspath(seven_zip_executable))
    if zzu.tool_binding is None:
        raise OODExternalV2IntegrityError("ZZU archive closure omits its 7-Zip binding")
    if (
        expected_seven_zip_tool is not None
        and zzu.tool_binding != expected_seven_zip_tool
    ):
        raise OODExternalV2IntegrityError(
            "ZZU closure tool differs from the successor-parent 7-Zip binding"
        )
    current_stage: str | None = None

    def transition(stage: str) -> None:
        nonlocal current_stage
        if stage not in CHILD_FREEZE_ATTEMPT_STAGES:
            raise OODExternalV2IntegrityError(
                "archive closure emitted an invalid child-freeze stage"
            )
        if stage != current_stage:
            current_stage = stage
            if stage_callback is not None:
                stage_callback(stage)

    try:
        transition("challenge_archive_closure")
        challenge_hash = verify_challenge_tar_extraction_closure(
            raw_source_paths["challenge_archive"],
            dataset_roots[CHALLENGE_2011_DATASET],
            challenge,
        )
        transition("zzu_tool_resolution")
        verify_seven_zip_tool_binding(requested_tool, zzu.tool_binding)
        zzu_hash = verify_zzu_split_zip_extraction_closure(
            raw_source_paths["zzu_archive_z01"],
            raw_source_paths["zzu_archive_zip"],
            dataset_roots[ZZU_PEDIATRIC_DATASET],
            requested_tool,
            zzu,
            stage_callback=transition,
        )
    except Exception as error:
        raise OODExternalV2IntegrityError(
            "external archive/extraction/tool closure verification failed"
        ) from error
    if (challenge_hash, zzu_hash) != tuple(
        summary.closure_sha256 for summary in summaries
    ):
        raise OODExternalV2IntegrityError("rebuilt archive closure hashes differ")


def _decode_child_contract_payload(raw: bytes) -> dict[str, object]:
    """Strictly validate canonical child bytes before any publication."""

    if not isinstance(raw, bytes) or not raw or len(raw) > _CHILD_MAX_BYTES:
        raise OODExternalV2ConfigError("child contract byte size is invalid")
    try:
        decoded: object = json.loads(
            raw[:-1].decode("ascii") if raw.endswith(b"\n") else b"".decode(),
            object_pairs_hook=_unique_json_object,
        )
    except OODExternalV2ConfigError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise OODExternalV2ConfigError("child contract is not canonical JSON") from error
    if not isinstance(decoded, dict):
        raise OODExternalV2ConfigError("child contract must contain a JSON object")
    payload = cast(dict[str, object], decoded)
    if canonical_json_bytes(payload) != raw:
        raise OODExternalV2ConfigError("child contract is not in exact canonical form")
    expected = {
        "artifact_sha256",
        "artifact_type",
        "dataset_roots",
        "decision_bindings",
        "frozen_at_utc",
        "implementation_revision",
        "inventory",
        "inventory_builder_attempt",
        "child_freeze_attempt",
        "output_root",
        "parent_config_file_sha256",
        "project_source_tree",
        "protocol_id",
        "public_inventory_projection",
        "raw_source_bindings",
        "runtime_bindings",
        "runtime_environment",
        "schema_version",
    }
    if set(payload) != expected:
        raise OODExternalV2ConfigError("child contract fields differ from protocol")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_type") != CHILD_CONTRACT_ARTIFACT_TYPE
        or payload.get("protocol_id") != PROTOCOL_ID
    ):
        raise OODExternalV2ConfigError("child contract identity is invalid")
    artifact_sha256 = _digest(payload.get("artifact_sha256"), "child artifact")
    if artifact_sha256 != canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    ):
        raise OODExternalV2ConfigError("child contract logical hash differs")
    return payload


def _load_child_contract_bytes(
    raw: bytes,
    *,
    source: str | Path,
) -> OODExternalV2ChildContract:
    """Purely decode and fully validate nested child-contract bytes."""

    source_path = Path(os.path.abspath(os.fspath(source)))
    payload = _decode_child_contract_payload(raw)
    artifact_sha256 = _digest(payload.get("artifact_sha256"), "child artifact")
    frozen = _utc_datetime(payload.get("frozen_at_utc"), "child frozen_at_utc")
    implementation_revision = _revision(
        payload.get("implementation_revision"),
        "child implementation revision",
    )
    raw_builder_attempt = _mapping(
        payload.get("inventory_builder_attempt"),
        "child inventory builder attempt",
    )
    if set(raw_builder_attempt) != {
        "artifact_sha256",
        "file_sha256",
        "relative_path",
    }:
        raise OODExternalV2ConfigError(
            "child inventory builder attempt fields differ from protocol"
        )
    inventory_builder_attempt = _bound_file(
        raw_builder_attempt,
        "child inventory builder attempt",
        require_artifact=True,
    )
    if (
        inventory_builder_attempt.relative_path
        != HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_PATH
    ):
        raise OODExternalV2ConfigError(
            "child inventory builder attempt path differs from protocol"
        )
    raw_child_freeze_attempt = _mapping(
        payload.get("child_freeze_attempt"),
        "child freeze attempt",
    )
    if set(raw_child_freeze_attempt) != {
        "artifact_sha256",
        "file_sha256",
        "relative_path",
    }:
        raise OODExternalV2ConfigError(
            "child freeze attempt fields differ from protocol"
        )
    child_freeze_attempt = _bound_file(
        raw_child_freeze_attempt,
        "child freeze attempt",
        require_artifact=True,
    )
    if child_freeze_attempt.relative_path != SUCCESSOR_CHILD_FREEZE_ATTEMPT_PATH:
        raise OODExternalV2ConfigError(
            "child freeze attempt path differs from protocol"
        )
    raw_inventory = _mapping(payload.get("inventory"), "child inventory")
    inventory_expected = {
        "archive_closures",
        "challenge_records",
        "file_sha256",
        "inventory_sha256",
        "relative_path",
        "selected_records_total",
        "zzu_patients",
        "zzu_records",
    }
    if set(raw_inventory) != inventory_expected:
        raise OODExternalV2ConfigError("child inventory fields differ from protocol")
    raw_archive_closures = raw_inventory["archive_closures"]
    if not isinstance(raw_archive_closures, list) or len(raw_archive_closures) != 2:
        raise OODExternalV2ConfigError(
            "child inventory must bind exactly two archive closures"
        )
    archive_closures = tuple(
        _parse_archive_closure_summary(
            value,
            context=f"child archive closure {index}",
        )
        for index, value in enumerate(raw_archive_closures)
    )
    if tuple(item.dataset for item in archive_closures) != (
        CHALLENGE_2011_DATASET,
        ZZU_PEDIATRIC_DATASET,
    ):
        raise OODExternalV2ConfigError(
            "child archive closures are not in canonical dataset order"
        )
    inventory = InventoryBinding(
        relative_path=_relative_path(raw_inventory.get("relative_path"), "inventory path"),
        file_sha256=_digest(raw_inventory.get("file_sha256"), "inventory file"),
        inventory_sha256=_digest(raw_inventory.get("inventory_sha256"), "inventory"),
        selected_records_total=_positive_integer(
            raw_inventory.get("selected_records_total"),
            "selected record count",
        ),
        challenge_records=_positive_integer(
            raw_inventory.get("challenge_records"),
            "Challenge record count",
        ),
        zzu_records=_positive_integer(raw_inventory.get("zzu_records"), "ZZU record count"),
        zzu_patients=_positive_integer(
            raw_inventory.get("zzu_patients"),
            "ZZU patient count",
        ),
        archive_closures=archive_closures,
    )
    raw_roots = _mapping(payload.get("dataset_roots"), "child dataset roots")
    if set(raw_roots) != {CHALLENGE_2011_DATASET, ZZU_PEDIATRIC_DATASET}:
        raise OODExternalV2ConfigError("child dataset roots must bind exactly both sources")
    roots = MappingProxyType(
        {
            name: _relative_path(raw_roots[name], f"dataset root {name}")
            for name in (CHALLENGE_2011_DATASET, ZZU_PEDIATRIC_DATASET)
        }
    )
    if dict(roots) != dict(EXPECTED_DATASET_ROOTS):
        raise OODExternalV2ConfigError(
            "child dataset roots differ from the exact frozen extraction roots"
        )
    raw_decisions = _mapping(payload.get("decision_bindings"), "decision bindings")
    if set(raw_decisions) != {"demo_policy", "source_calibration_result"}:
        raise OODExternalV2ConfigError("child decision bindings differ from protocol")
    raw_demo_binding = _mapping(raw_decisions["demo_policy"], "demo policy binding")
    raw_source_binding = _mapping(
        raw_decisions["source_calibration_result"],
        "source-calibration binding",
    )
    if set(raw_demo_binding) != {"file_sha256", "relative_path"}:
        raise OODExternalV2ConfigError(
            "demo policy must bind its exact file without a logical artifact hash"
        )
    if set(raw_source_binding) != {
        "artifact_sha256",
        "file_sha256",
        "relative_path",
    }:
        raise OODExternalV2ConfigError(
            "source-calibration binding must include its self hash"
        )
    decision_bindings = MappingProxyType(
        {
            name: _bound_file(raw_decisions[name], f"decision binding {name}")
            for name in ("demo_policy", "source_calibration_result")
        }
    )
    if (
        decision_bindings["demo_policy"].relative_path
        != EXPECTED_DEMO_POLICY_PATH
        or decision_bindings["demo_policy"].file_sha256
        != EXPECTED_DEMO_POLICY_FILE_SHA256
        or decision_bindings["demo_policy"].artifact_sha256 is not None
        or decision_bindings["source_calibration_result"].relative_path
        != EXPECTED_SOURCE_CALIBRATION_PATH
        or decision_bindings["source_calibration_result"].file_sha256
        != EXPECTED_SOURCE_CALIBRATION_FILE_SHA256
        or decision_bindings["source_calibration_result"].artifact_sha256
        != EXPECTED_SOURCE_CALIBRATION_ARTIFACT_SHA256
    ):
        raise OODExternalV2ConfigError(
            "child decision bindings differ from exact frozen components"
        )
    raw_sources = _mapping(payload.get("raw_source_bindings"), "raw source bindings")
    if set(raw_sources) != set(REQUIRED_RAW_SOURCE_BINDING_KEYS):
        raise OODExternalV2ConfigError(
            "child raw-source bindings differ from the exact acquisition set"
        )
    raw_source_bindings = MappingProxyType(
        {
            name: _raw_source_binding(raw_sources[name], f"raw source binding {name}")
            for name in REQUIRED_RAW_SOURCE_BINDING_KEYS
        }
    )
    for name, expected_path in EXPECTED_RAW_SOURCE_PATHS.items():
        if raw_source_bindings[name].relative_path != expected_path:
            raise OODExternalV2ConfigError(
                f"raw source path differs from the exact frozen acquisition: {name}"
            )
    runtime_environment = _runtime_environment_binding(
        payload.get("runtime_environment"),
        context="child runtime environment",
    )
    project_source_tree = _project_source_tree_binding(
        payload.get("project_source_tree"),
        context="child project source tree",
    )
    raw_runtime = _mapping(payload.get("runtime_bindings"), "runtime bindings")
    if set(raw_runtime) != set(REQUIRED_RUNTIME_BINDING_PATHS):
        raise OODExternalV2ConfigError("child runtime bindings differ from exact evaluator set")
    runtime_bindings = MappingProxyType(
        {
            path: _digest(raw_runtime[path], f"runtime binding {path}")
            for path in REQUIRED_RUNTIME_BINDING_PATHS
        }
    )
    raw_projection = payload.get("public_inventory_projection")
    public_projection = (
        None
        if raw_projection is None
        else _bound_file(raw_projection, "public inventory projection")
    )
    if public_projection is None or public_projection.artifact_sha256 is None:
        raise OODExternalV2ConfigError(
            "child must bind the public projection file and logical artifact hashes"
        )
    return OODExternalV2ChildContract(
        path=source_path,
        file_sha256=sha256_bytes(raw),
        artifact_sha256=artifact_sha256,
        frozen_at_utc=frozen,
        parent_config_file_sha256=_digest(
            payload.get("parent_config_file_sha256"),
            "parent config",
        ),
        implementation_revision=implementation_revision,
        inventory=inventory,
        dataset_roots=roots,
        decision_bindings=decision_bindings,
        raw_source_bindings=raw_source_bindings,
        inventory_builder_attempt=inventory_builder_attempt,
        child_freeze_attempt=child_freeze_attempt,
        runtime_environment=runtime_environment,
        runtime_bindings=runtime_bindings,
        project_source_tree=project_source_tree,
        public_inventory_projection=public_projection,
        output_root=_relative_path(payload.get("output_root"), "child output root"),
    )


def load_child_contract(path: str | Path) -> OODExternalV2ChildContract:
    """Load the canonical, self-hashed child execution contract."""

    source = Path(os.path.abspath(os.fspath(path)))
    raw = _read_bounded(source, _CHILD_MAX_BYTES, "child execution contract")
    return _load_child_contract_bytes(raw, source=source)


def child_contract_bytes(body: Mapping[str, object]) -> bytes:
    """Seal a complete child body for metadata-only inventory tooling.

    The helper performs no waveform access.  Callers must provide every field
    except ``artifact_sha256`` and then commit the resulting bytes before the
    one-shot evaluation.
    """

    if "artifact_sha256" in body:
        raise OODExternalV2ConfigError("child body must not self-assert artifact_sha256")
    payload = dict(body)
    payload["artifact_sha256"] = canonical_sha256(payload)
    serialized = canonical_json_bytes(payload)
    # Full nested validation is intentionally performed by the transaction before
    # publication and again by ``load_child_contract`` after publication.
    return serialized


def verify_external_v2_inputs(
    parent: OODExternalV2ParentConfig,
    child: OODExternalV2ChildContract,
    *,
    project_root: str | Path,
    code_revision: str,
    seven_zip_executable: str | Path = "7z",
) -> VerifiedExternalV2Inputs:
    """Verify all metadata, hashes, roles, and v1 public evidence before decode."""

    if not isinstance(parent, OODExternalV2ParentConfig):
        raise TypeError("parent must be OODExternalV2ParentConfig")
    if not isinstance(child, OODExternalV2ChildContract):
        raise TypeError("child must be OODExternalV2ChildContract")
    revision = _revision(code_revision, "execution code revision")
    root = _strict_project_root(project_root)
    expected_parent_path = root.joinpath(
        *PurePosixPath(SUCCESSOR_PARENT_CONFIG_PATH).parts
    )
    expected_child_path = root.joinpath(
        *PurePosixPath(SUCCESSOR_CHILD_CONFIG_PATH).parts
    )
    if parent.file_sha256 != EXPECTED_PARENT_CONFIG_SHA256:
        successor = verify_successor_parent_preflight(
            parent.path,
            project_root=root,
        )
        if (
            parent.path != expected_parent_path
            or child.path != expected_child_path
            or successor.file_sha256 != parent.file_sha256
            or dict(successor.raw_source_bindings)
            != dict(parent.raw_source_bindings or {})
            or successor.seven_zip_tool_binding != parent.seven_zip_tool_binding
            or successor.inventory_counts != parent.inventory_counts
        ):
            raise OODExternalV2IntegrityError(
                "operational successor parent/child lineage differs"
            )
    if child.parent_config_file_sha256 != parent.file_sha256:
        raise OODExternalV2IntegrityError("child does not bind the exact parent bytes")
    if (
        parent.raw_source_bindings is not None
        and dict(child.raw_source_bindings) != dict(parent.raw_source_bindings)
    ):
        raise OODExternalV2IntegrityError(
            "child raw-source provenance differs from the successor parent"
        )
    if child.output_root != parent.output_root:
        raise OODExternalV2IntegrityError("child output root differs from parent")
    if parent.inventory_counts is not None and (
        child.inventory.challenge_records,
        child.inventory.zzu_records,
        child.inventory.zzu_patients,
        child.inventory.selected_records_total,
    ) != (
        parent.inventory_counts.challenge_records,
        parent.inventory_counts.zzu_records,
        parent.inventory_counts.zzu_patients,
        parent.inventory_counts.total_records,
    ):
        raise OODExternalV2IntegrityError(
            "child inventory counts differ from the successor parent"
        )
    if (
        child.inventory.relative_path != SUCCESSOR_PRIVATE_INVENTORY_PATH
        or child.public_inventory_projection is None
        or child.public_inventory_projection.relative_path
        != SUCCESSOR_PUBLIC_PROJECTION_PATH
    ):
        raise OODExternalV2IntegrityError(
            "child inventory/projection paths differ from successor namespace"
        )
    for source, context in (
        (parent.path, "parent protocol"),
        (child.path, "child contract"),
    ):
        _require_project_file(root, source, context=context)
    _verify_revision_boundary(root, child=child, execution_revision=revision)
    _verify_private_history_absent(root)
    _verify_tracked_head_blob(
        root,
        revision=revision,
        relative_path=SUCCESSOR_PARENT_CONFIG_PATH,
        expected_file_sha256=parent.file_sha256,
    )
    _verify_project_source_tree_at_revisions(
        root,
        child.project_source_tree,
        implementation_revision=child.implementation_revision,
        execution_revision=revision,
    )
    _verify_imported_project_module_origins(root, child.project_source_tree)
    if _current_runtime_environment() != child.runtime_environment:
        raise OODExternalV2IntegrityError(
            "active Python/scientific runtime differs from the frozen child contract"
        )
    for relative_path, expected_hash in child.runtime_bindings.items():
        runtime_path = _resolve_project_relative(root, relative_path, require_file=True)
        if sha256_file(runtime_path) != expected_hash:
            raise OODExternalV2IntegrityError(
                f"runtime-critical file differs from child binding: {relative_path}"
            )
    projection_binding = child.public_inventory_projection
    if projection_binding is None:
        raise OODExternalV2IntegrityError("public inventory projection binding is absent")
    projection_path = _resolve_project_relative(
        root,
        projection_binding.relative_path,
        require_file=True,
    )
    if sha256_file(projection_path) != projection_binding.file_sha256:
        raise OODExternalV2IntegrityError("public inventory projection hash differs")

    _verify_child_inventory_builder_attempt(
        parent,
        child,
        project_root=root,
    )
    raw_source_paths: dict[str, Path] = {}
    for name, binding in child.raw_source_bindings.items():
        source_path = _resolve_project_relative(
            root,
            binding.relative_path,
            require_file=True,
        )
        try:
            observed_size = source_path.stat().st_size
        except OSError as error:
            raise OODExternalV2IntegrityError(
                f"raw source binding is unavailable: {name}"
            ) from error
        if (
            observed_size != binding.size_bytes
            or sha256_file(source_path) != binding.file_sha256
            or (
                binding.official_md5 is not None
                and _md5_file(source_path) != binding.official_md5
            )
        ):
            raise OODExternalV2IntegrityError(
                f"raw source provenance differs from child binding: {name}"
            )
        raw_source_paths[name] = source_path

    bound_paths: dict[str, Path] = {}
    for name, bound in (
        ("checkpoint", parent.checkpoint),
        ("resolved_config", parent.resolved_config),
        ("normalization", parent.normalization),
        ("quality_implementation", parent.quality_implementation),
        ("dependency_lock", parent.dependency_lock),
        ("project_manifest", parent.project_manifest),
    ):
        path = _resolve_project_relative(root, bound.relative_path, require_file=True)
        if sha256_file(path) != bound.file_sha256:
            raise OODExternalV2IntegrityError(f"{name} hash differs from parent")
        bound_paths[name] = path
    if DEFAULT_SIGNAL_QUALITY_CONFIG.version != parent.quality_config_version:
        raise OODExternalV2IntegrityError("loaded quality configuration version differs")

    v1 = _verify_v1_public_evidence(parent, project_root=root)
    inventory_path = _resolve_project_relative(
        root,
        child.inventory.relative_path,
        require_file=True,
    )
    _require_git_ignored_and_untracked(
        root,
        inventory_path.relative_to(root).as_posix(),
        context="private external inventory",
    )
    _require_git_ignored_and_untracked(
        root,
        f"{parent.output_root}/private",
        context="private evidence output",
    )
    if sha256_file(inventory_path) != child.inventory.file_sha256:
        raise OODExternalV2IntegrityError("external inventory file hash differs from child")
    try:
        inventory = load_external_inventory(inventory_path)
    except Exception as error:
        raise OODExternalV2IntegrityError("external inventory verification failed") from error
    if inventory.inventory_sha256 != child.inventory.inventory_sha256:
        raise OODExternalV2IntegrityError("external inventory logical identity differs")
    _verify_inventory_counts(parent, child, inventory)
    if parent.inventory_counts is None:
        raise OODExternalV2IntegrityError(
            "successor parent inventory counts are unavailable"
        )
    projection_artifact_sha256 = _verify_public_projection_file(
        projection_path,
        inventory=inventory,
        challenge_records=child.inventory.challenge_records,
        zzu_records=child.inventory.zzu_records,
        expected_counts=parent.inventory_counts,
    )
    if projection_artifact_sha256 != projection_binding.artifact_sha256:
        raise OODExternalV2IntegrityError(
            "public inventory projection logical hash differs from child binding"
        )
    observed_closures = _assert_production_archive_closures(
        inventory.archive_closures,
        expected_seven_zip_tool=parent.seven_zip_tool_binding,
    )
    if observed_closures != child.inventory.archive_closures:
        raise OODExternalV2IntegrityError(
            "inventory archive closures differ from the child contract"
        )

    dataset_roots = MappingProxyType(
        {
            name: _resolve_project_relative(
                root,
                child.dataset_roots[name],
                require_directory=True,
            )
            for name in (CHALLENGE_2011_DATASET, ZZU_PEDIATRIC_DATASET)
        }
    )
    _assert_external_roots_are_not_forbidden(root, dataset_roots)
    _verify_raw_inventory(
        inventory,
        dataset_roots=dataset_roots,
        raw_source_paths=MappingProxyType(raw_source_paths),
        public_projection=(
            None
            if child.public_inventory_projection is None
            else _resolve_project_relative(
                root,
                child.public_inventory_projection.relative_path,
                require_file=True,
            )
        ),
        parent=parent,
    )
    _verify_archive_closure_rebuilds(
        inventory,
        dataset_roots=dataset_roots,
        raw_source_paths=MappingProxyType(raw_source_paths),
        seven_zip_executable=seven_zip_executable,
        expected_seven_zip_tool=parent.seven_zip_tool_binding,
    )
    routing = _load_routing_components(parent, child, project_root=root)
    return VerifiedExternalV2Inputs(
        project_root=root,
        parent=parent,
        child=child,
        inventory=inventory,
        inventory_path=inventory_path,
        dataset_roots=dataset_roots,
        raw_source_paths=MappingProxyType(raw_source_paths),
        v1=v1,
        checkpoint_path=bound_paths["checkpoint"],
        resolved_config_path=bound_paths["resolved_config"],
        normalization_path=bound_paths["normalization"],
        routing=routing,
    )


def _verify_v1_public_evidence(
    parent: OODExternalV2ParentConfig,
    *,
    project_root: Path,
) -> VerifiedV1PublicEvidence:
    """Run the authoritative v1 verifier, then retain only public aggregates.

    The required whole-bundle verifier integrity-checks private v1 embedding
    bytes.  V2 never exposes, subsets, scores, or analyzes those arrays; only
    the verifier's public result, policy, and manifest objects cross this
    function boundary.
    """

    paths = {
        "result": _resolve_project_relative(
            project_root,
            parent.v1_result.relative_path,
            require_file=True,
        ),
        "success": _resolve_project_relative(
            project_root,
            parent.v1_success_manifest.relative_path,
            require_file=True,
        ),
        "policy": _resolve_project_relative(
            project_root,
            parent.v1_distribution_policy.relative_path,
            require_file=True,
        ),
    }
    try:
        verified_whole = verify_ood_completion_bundle(paths["result"].parent)
    except Exception as error:
        raise OODExternalV2IntegrityError(
            "authoritative sealed v1 whole-bundle verification failed"
        ) from error
    snapshots: dict[str, str] = {}
    expected = {
        "result": parent.v1_result.file_sha256,
        "success": parent.v1_success_manifest.file_sha256,
        "policy": parent.v1_distribution_policy.file_sha256,
    }
    for name, path in paths.items():
        observed = sha256_file(path)
        if observed != expected[name]:
            raise OODExternalV2IntegrityError(f"sealed v1 {name} file hash differs")
        snapshots[f"v1_{name}"] = observed
    result = verified_whole.result
    policy = verified_whole.policy
    success = verified_whole.success_manifest
    if (
        result.artifact_sha256 != parent.v1_result.artifact_sha256
        or policy.artifact_sha256 != parent.v1_distribution_policy.artifact_sha256
        or success.artifact_sha256 != parent.v1_success_manifest.artifact_sha256
    ):
        raise OODExternalV2IntegrityError("sealed v1 logical identity differs from parent")
    if result.status != "SOURCE_SUPPORT_GATE_TARGET_MISSED" or result.research_bundle_eligible:
        raise OODExternalV2IntegrityError("historical v1 source result status changed")
    if (
        result.distribution_policy.artifact_sha256 != policy.artifact_sha256
        or result.distribution_policy.file_sha256 != parent.v1_distribution_policy.file_sha256
        or success.result_artifact_sha256 != result.artifact_sha256
        or success.distribution_policy_artifact_sha256 != policy.artifact_sha256
        or policy.detector.threshold != parent.threshold
        or policy.threshold_comparison != "score_strictly_greater_than_threshold"
    ):
        raise OODExternalV2IntegrityError("v1 result, policy, and success metadata disagree")
    public_member_hashes = {
        member.relative_path: member.file_sha256
        for member in success.members
        if member.relative_path in {"distribution-policy.json", "ood-completion-result.json"}
    }
    if public_member_hashes != {
        "distribution-policy.json": parent.v1_distribution_policy.file_sha256,
        "ood-completion-result.json": parent.v1_result.file_sha256,
    }:
        raise OODExternalV2IntegrityError("v1 success metadata public members disagree")

    # The adjacent claim is sanitized metadata.  Its hash is declared by the
    # verified success manifest; no v1 private embedding member is opened here.
    claim_path = paths["result"].parent.parent / success.validation_access_claim_filename
    claim_path = _require_project_file(
        project_root,
        claim_path,
        context="sealed v1 adjacent claim",
    )
    claim_hash = sha256_file(claim_path)
    if claim_hash != success.validation_access_claim_file_sha256:
        raise OODExternalV2IntegrityError("sealed v1 adjacent claim hash differs")
    snapshots["v1_claim"] = claim_hash
    return VerifiedV1PublicEvidence(
        result=result,
        policy=policy,
        success_manifest=success,
        claim_file_sha256=claim_hash,
        snapshots=MappingProxyType(snapshots),
    )


def _verify_inventory_counts(
    parent: OODExternalV2ParentConfig,
    child: OODExternalV2ChildContract,
    inventory: ExternalWaveformInventory,
) -> None:
    challenge = tuple(
        record for record in inventory.records if record.dataset == CHALLENGE_2011_DATASET
    )
    zzu = tuple(record for record in inventory.records if record.dataset == ZZU_PEDIATRIC_DATASET)
    if len(challenge) != parent.challenge_expected_records:
        raise OODExternalV2IntegrityError("Challenge inventory is not complete Set A")
    if len(challenge) != child.inventory.challenge_records:
        raise OODExternalV2IntegrityError("Challenge inventory count differs from child")
    if any(record.patient_key is not None for record in challenge):
        raise OODExternalV2IntegrityError(
            "Challenge inventory patient keys must all be exactly null"
        )
    if len(zzu) != child.inventory.zzu_records or not zzu:
        raise OODExternalV2IntegrityError("ZZU inventory count differs from child")
    patient_keys = {record.patient_key for record in zzu}
    if None in patient_keys or len(patient_keys) != child.inventory.zzu_patients:
        raise OODExternalV2IntegrityError("ZZU patient count differs from child")
    if len(inventory.records) != child.inventory.selected_records_total:
        raise OODExternalV2IntegrityError("total inventory count differs from child")
    if len(challenge) + len(zzu) != len(inventory.records):
        raise OODExternalV2IntegrityError("inventory contains a forbidden dataset")
    try:
        validate_challenge_2011_set_a_inventory(
            build_external_inventory(challenge),
            expected_record_count=parent.challenge_expected_records,
        )
    except Exception as error:
        raise OODExternalV2IntegrityError("Challenge Set A role verification failed") from error


def _verify_raw_inventory(
    inventory: ExternalWaveformInventory,
    *,
    dataset_roots: Mapping[str, Path],
    raw_source_paths: Mapping[str, Path],
    public_projection: Path | None,
    parent: OODExternalV2ParentConfig,
) -> None:
    for dataset in (CHALLENGE_2011_DATASET, ZZU_PEDIATRIC_DATASET):
        records = tuple(record for record in inventory.records if record.dataset == dataset)
        subset = build_external_inventory(records)
        try:
            verified = verify_external_inventory(dataset_roots[dataset], subset)
        except Exception as error:
            raise OODExternalV2IntegrityError(
                f"raw external inventory failed verification for {dataset}"
            ) from error
        if verified != subset.inventory_sha256:
            raise OODExternalV2IntegrityError("external subset inventory identity changed")
        if dataset == CHALLENGE_2011_DATASET:
            validate_challenge_2011_set_a_inventory(
                subset,
                expected_record_count=parent.challenge_expected_records,
            )
    _verify_role_metadata_rejoin(
        inventory,
        dataset_roots=dataset_roots,
        raw_source_paths=raw_source_paths,
        public_projection=public_projection,
        parent=parent,
    )


def _assert_external_roots_are_not_forbidden(
    project_root: Path,
    dataset_roots: Mapping[str, Path],
) -> None:
    expected = {
        dataset: _resolve_project_relative(
            project_root,
            relative_path,
            require_directory=True,
        )
        for dataset, relative_path in EXPECTED_DATASET_ROOTS.items()
    }
    if dict(dataset_roots) != expected:
        raise OODExternalV2IntegrityError(
            "external roots must equal the two exact frozen extraction directories"
        )
    forbidden_roots = (
        project_root / "artifacts" / "trust_sentinel" / "ood_completion_v1",
        project_root / "data" / "raw" / "ptb-xl",
        project_root / "runs",
    )
    for root in dataset_roots.values():
        for forbidden in forbidden_roots:
            try:
                root.relative_to(forbidden)
            except ValueError:
                continue
            raise OODExternalV2IntegrityError("external dataset root enters a forbidden source")
        normalized = root.as_posix().casefold()
        if any(fragment in normalized for fragment in ("/sph/", "fold10", "fold-10")):
            raise OODExternalV2IntegrityError(
                "external dataset root names a previously observed source"
            )


def _verify_revision_boundary(
    project_root: Path,
    *,
    child: OODExternalV2ChildContract,
    execution_revision: str,
) -> None:
    """Require a clean descendant whose only later files are the child freeze."""

    try:
        clean_head = _verify_clean_git_revision(project_root)
    except Exception as error:
        raise OODExternalV2IntegrityError(
            "execution requires a clean committed revision"
        ) from error
    if clean_head != execution_revision:
        raise OODExternalV2IntegrityError("execution revision differs from clean Git HEAD")
    _verify_successor_amendment_revision(
        project_root,
        implementation_revision=child.implementation_revision,
    )
    completed = _run_git(
        project_root,
        "merge-base",
        "--is-ancestor",
        child.implementation_revision,
        execution_revision,
        allow_empty=True,
    )
    if completed.returncode != 0:
        raise OODExternalV2IntegrityError("execution is not a descendant of implementation freeze")
    commit_count = _run_git(
        project_root,
        "rev-list",
        "--count",
        f"{child.implementation_revision}..{execution_revision}",
    ).stdout.strip()
    revision_line = _run_git(
        project_root,
        "rev-list",
        "--parents",
        "-n",
        "1",
        execution_revision,
    ).stdout.strip().casefold()
    if commit_count != "1" or revision_line.split() != [
        execution_revision,
        child.implementation_revision,
    ]:
        raise OODExternalV2IntegrityError(
            "execution must be exactly one child-freeze commit after implementation"
        )
    _verify_git_remote_state(project_root, expected_revision=execution_revision)
    expected_child = project_root.joinpath(
        *PurePosixPath(SUCCESSOR_CHILD_CONFIG_PATH).parts
    )
    if child.path != expected_child:
        raise OODExternalV2IntegrityError(
            "operational child must use the exact canonical successor path"
        )
    child_relative = child.path.relative_to(project_root).as_posix()
    allowed = {child_relative}
    if child.public_inventory_projection is None:
        raise OODExternalV2IntegrityError(
            "operational child must bind the public inventory projection"
        )
    if (
        child.inventory.relative_path != SUCCESSOR_PRIVATE_INVENTORY_PATH
        or child.public_inventory_projection.relative_path
        != SUCCESSOR_PUBLIC_PROJECTION_PATH
    ):
        raise OODExternalV2IntegrityError(
            "operational child inventory paths differ from successor namespace"
        )
    allowed.add(child.public_inventory_projection.relative_path)
    tracked = _run_git(
        project_root,
        "ls-files",
        "--",
        child.public_inventory_projection.relative_path,
        allow_empty=True,
    )
    if tracked.stdout.strip().replace("\\", "/") != (
        child.public_inventory_projection.relative_path
    ):
        raise OODExternalV2IntegrityError(
            "public inventory projection must be tracked in execution Git HEAD"
        )
    _verify_revision_bound_file(
        project_root,
        revision=execution_revision,
        relative_path=child_relative,
        expected_file_sha256=child.file_sha256,
        context="frozen child contract",
    )
    _verify_revision_bound_file(
        project_root,
        revision=execution_revision,
        relative_path=child.public_inventory_projection.relative_path,
        expected_file_sha256=child.public_inventory_projection.file_sha256,
        context="public inventory projection",
    )
    diff = _run_git(
        project_root,
        "diff",
        "--name-status",
        "--no-renames",
        f"{child.implementation_revision}..{execution_revision}",
        "--",
    )
    changed: dict[str, str] = {}
    for line in diff.stdout.splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2 or parts[0] != "A":
            raise OODExternalV2IntegrityError("post-implementation Git diff is not additive-only")
        changed[parts[1].replace("\\", "/")] = parts[0]
    if set(changed) != allowed:
        raise OODExternalV2IntegrityError(
            "post-implementation Git diff must contain exactly the frozen child artifacts"
        )


def _verify_clean_git_revision(project_root: Path) -> str:
    for option in ("-t", "-v"):
        tagged = _run_git(project_root, "ls-files", option).stdout.splitlines()
        if any(len(line) < 3 or not line.startswith("H ") for line in tagged):
            raise OODExternalV2IntegrityError(
                "tracked Git index contains skip-worktree or assume-unchanged state"
            )
    status = _run_git(
        project_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).stdout
    if status != "":
        raise OODExternalV2IntegrityError("Git worktree is not exactly clean")
    revision = _run_git(project_root, "rev-parse", "HEAD").stdout.strip().casefold()
    return _revision(revision, "clean Git revision")


def _verify_git_remote_state(
    project_root: Path,
    *,
    expected_revision: str,
) -> None:
    """Require the exact pushed GitHub origin/main state frozen by v2.1."""

    remotes = tuple(
        item for item in _run_git(project_root, "remote").stdout.splitlines() if item
    )
    fetch_urls = tuple(
        item
        for item in _run_git(
            project_root,
            "remote",
            "get-url",
            "--all",
            EXPECTED_GIT_REMOTE_NAME,
        ).stdout.splitlines()
        if item
    )
    push_urls = tuple(
        item
        for item in _run_git(
            project_root,
            "remote",
            "get-url",
            "--push",
            "--all",
            EXPECTED_GIT_REMOTE_NAME,
        ).stdout.splitlines()
        if item
    )
    remote_revision = _run_git(
        project_root,
        "rev-parse",
        "--verify",
        EXPECTED_GIT_REMOTE_MAIN_REF,
    ).stdout.strip().casefold()
    _verify_private_remote_anonymous_denial(project_root)
    live_remote_stdout = _run_exact_private_live_remote(project_root)
    backup_tag_type = _run_git(
        project_root,
        "cat-file",
        "-t",
        EXPECTED_GIT_REMOTE_BACKUP_TAG_REVISION,
    ).stdout
    backup_tag_ancestor = _run_git(
        project_root,
        "merge-base",
        "--is-ancestor",
        EXPECTED_GIT_REMOTE_BACKUP_TAG_REVISION,
        expected_revision,
        allow_empty=True,
    )
    expected_live_stdout = (
        "ref: refs/heads/main\tHEAD\n"
        f"{expected_revision}\tHEAD\n"
        f"{expected_revision}\trefs/heads/main\n"
        f"{EXPECTED_GIT_REMOTE_BACKUP_TAG_REVISION}\t"
        f"{EXPECTED_GIT_REMOTE_BACKUP_TAG_REF}\n"
    )
    live_remote_matches = live_remote_stdout == expected_live_stdout
    live_remote_stdout = ""
    if (
        remotes != (EXPECTED_GIT_REMOTE_NAME,)
        or fetch_urls != (EXPECTED_GIT_REMOTE_URL,)
        or push_urls != (EXPECTED_GIT_REMOTE_URL,)
        or remote_revision != expected_revision
        or not live_remote_matches
        or backup_tag_type != "commit\n"
        or backup_tag_ancestor.returncode != 0
        or backup_tag_ancestor.stdout != ""
    ):
        raise OODExternalV2IntegrityError(
            "Git origin/main is not the exact pushed frozen revision"
        )


def _verify_exact_modification_child(
    project_root: Path,
    *,
    child_revision: str,
    parent_revision: str,
    modified_paths: tuple[str, ...],
    context: str,
) -> None:
    child = _revision(child_revision, f"{context} child revision")
    parent = _revision(parent_revision, f"{context} parent revision")
    revision_line = _run_git(
        project_root,
        "rev-list",
        "--parents",
        "-n",
        "1",
        child,
    ).stdout.strip().casefold()
    commit_count = _run_git(
        project_root,
        "rev-list",
        "--count",
        f"{parent}..{child}",
    ).stdout.strip()
    if revision_line.split() != [child, parent] or commit_count != "1":
        raise OODExternalV2IntegrityError(
            f"{context} is not the sole direct child of its frozen parent"
        )
    diff = _run_git(
        project_root,
        "diff",
        "--name-status",
        "--no-renames",
        f"{parent}..{child}",
        "--",
    )
    changed: dict[str, str] = {}
    for line in diff.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 2 or parts[0] != "M":
            raise OODExternalV2IntegrityError(
                f"{context} contains a non-modification change"
            )
        path = parts[1].replace("\\", "/")
        if path in changed:
            raise OODExternalV2IntegrityError(f"{context} contains a duplicate path")
        changed[path] = parts[0]
    if changed != {path: "M" for path in modified_paths}:
        raise OODExternalV2IntegrityError(
            f"{context} paths differ from the frozen allowlist"
        )


def _verify_successor_amendment_revision(
    project_root: Path,
    *,
    implementation_revision: str,
) -> None:
    """Bind all pre-inventory amendments to their consecutive frozen parents."""

    revision = _revision(
        implementation_revision,
        "private-remote successor implementation revision",
    )
    _verify_exact_modification_child(
        project_root,
        child_revision=SECOND_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        parent_revision=FIRST_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        modified_paths=SUCCESSOR_AMENDMENT_MODIFIED_PATHS,
        context="first successor amendment",
    )
    _verify_historical_revision_blob(
        project_root,
        revision=FIRST_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        relative_path=SUCCESSOR_PARENT_CONFIG_PATH,
        expected_file_sha256=FIRST_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256,
        context="first frozen successor parent",
    )
    _verify_exact_modification_child(
        project_root,
        child_revision=THIRD_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        parent_revision=SECOND_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        modified_paths=SUCCESSOR_PRIVATE_REMOTE_AMENDMENT_MODIFIED_PATHS,
        context="private-remote successor amendment",
    )
    _verify_historical_revision_blob(
        project_root,
        revision=SECOND_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        relative_path=SUCCESSOR_PARENT_CONFIG_PATH,
        expected_file_sha256=SECOND_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256,
        context="second frozen successor parent",
    )
    _verify_exact_modification_child(
        project_root,
        child_revision=FOURTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        parent_revision=THIRD_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        modified_paths=SUCCESSOR_INVENTORY_BUILDER_AMENDMENT_MODIFIED_PATHS,
        context="inventory-builder successor amendment",
    )
    _verify_historical_revision_blob(
        project_root,
        revision=THIRD_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        relative_path=SUCCESSOR_PARENT_CONFIG_PATH,
        expected_file_sha256=THIRD_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256,
        context="third frozen successor parent",
    )
    _verify_exact_modification_child(
        project_root,
        child_revision=FIFTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        parent_revision=FOURTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        modified_paths=SUCCESSOR_RUNTIME_PREFLIGHT_AMENDMENT_MODIFIED_PATHS,
        context="runtime-preflight successor amendment",
    )
    _verify_historical_revision_blob(
        project_root,
        revision=FOURTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        relative_path=SUCCESSOR_PARENT_CONFIG_PATH,
        expected_file_sha256=FOURTH_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256,
        context="fourth frozen successor parent",
    )
    _verify_exact_modification_child(
        project_root,
        child_revision=SIXTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        parent_revision=FIFTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        modified_paths=SUCCESSOR_GCM_SCRATCH_AMENDMENT_MODIFIED_PATHS,
        context="GCM scratch-cleanup successor amendment",
    )
    _verify_historical_revision_blob(
        project_root,
        revision=FIFTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        relative_path=SUCCESSOR_PARENT_CONFIG_PATH,
        expected_file_sha256=FIFTH_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256,
        context="fifth frozen successor parent",
    )
    _verify_exact_modification_child(
        project_root,
        child_revision=SEVENTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        parent_revision=SIXTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        modified_paths=SUCCESSOR_INVENTORY_FAILURE_AMENDMENT_MODIFIED_PATHS,
        context="inventory-failure observability successor amendment",
    )
    _verify_historical_revision_blob(
        project_root,
        revision=SIXTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        relative_path=SUCCESSOR_PARENT_CONFIG_PATH,
        expected_file_sha256=SIXTH_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256,
        context="sixth frozen successor parent",
    )
    _verify_exact_modification_child(
        project_root,
        child_revision=EIGHTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        parent_revision=SEVENTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        modified_paths=SUCCESSOR_ARCHIVE_OPERAND_AMENDMENT_MODIFIED_PATHS,
        context="archive-operand successor amendment",
    )
    _verify_historical_revision_blob(
        project_root,
        revision=SEVENTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        relative_path=SUCCESSOR_PARENT_CONFIG_PATH,
        expected_file_sha256=SEVENTH_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256,
        context="seventh frozen successor parent",
    )
    _verify_exact_modification_child(
        project_root,
        child_revision=NINTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        parent_revision=EIGHTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        modified_paths=SUCCESSOR_CHILD_FREEZE_AMENDMENT_MODIFIED_PATHS,
        context="child-freeze observability successor amendment",
    )
    _verify_historical_revision_blob(
        project_root,
        revision=EIGHTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        relative_path=SUCCESSOR_PARENT_CONFIG_PATH,
        expected_file_sha256=EIGHTH_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256,
        context="eighth frozen successor parent",
    )
    _verify_exact_modification_child(
        project_root,
        child_revision=revision,
        parent_revision=NINTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        modified_paths=SUCCESSOR_CHILD_FREEZE_DECISION_BINDING_AMENDMENT_MODIFIED_PATHS,
        context="child-freeze decision-binding successor amendment",
    )
    _verify_historical_revision_blob(
        project_root,
        revision=NINTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        relative_path=SUCCESSOR_PARENT_CONFIG_PATH,
        expected_file_sha256=NINTH_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256,
        context="ninth frozen successor parent",
    )


def _verify_private_history_absent(project_root: Path) -> None:
    """Prove protected raw/private/claim/output paths never entered Git history."""

    history = _run_git(
        project_root,
        "log",
        "--full-history",
        "--all",
        "--reflog",
        "--format=%H",
        "--",
        *FORBIDDEN_GIT_HISTORY_PATHS,
    )
    if history.stdout != "":
        raise OODExternalV2IntegrityError(
            "protected raw/private/claim/output paths appear in Git history"
        )


def _build_project_source_tree(project_root: Path) -> ProjectSourceTreeBinding:
    source_root = _assert_direct_ancestry(
        project_root.joinpath(*PurePosixPath(PROJECT_SOURCE_ROOT).parts),
        context="project Python source root",
    )
    if not source_root.is_dir():
        raise OODExternalV2IntegrityError("project Python source root is unavailable")
    paths: list[Path] = []
    for current_text, directory_names, file_names in os.walk(
        source_root,
        followlinks=False,
    ):
        current = Path(current_text)
        _assert_direct_ancestry(current, context="project source directory")
        directory_names.sort()
        file_names.sort()
        for directory_name in directory_names:
            directory = current / directory_name
            _assert_direct_ancestry(directory, context="project source directory")
            if not directory.is_dir():
                raise OODExternalV2IntegrityError(
                    "project source tree contains a non-directory entry"
                )
        for file_name in file_names:
            path = current / file_name
            if path.suffix != ".py":
                continue
            _assert_direct_ancestry(path, context="project Python source file")
            if not path.is_file():
                raise OODExternalV2IntegrityError(
                    "project source tree contains a non-regular Python file"
                )
            paths.append(path)
    for relative_entrypoint in PROJECT_OPERATIONAL_ENTRYPOINTS:
        entrypoint = _assert_direct_ancestry(
            project_root.joinpath(*PurePosixPath(relative_entrypoint).parts),
            context="external protocol entrypoint",
        )
        if not entrypoint.is_file():
            raise OODExternalV2IntegrityError(
                "external protocol entrypoint is unavailable"
            )
        paths.append(entrypoint)
    relative_paths = tuple(
        sorted(path.relative_to(project_root).as_posix() for path in paths)
    )
    if (
        not relative_paths
        or len(relative_paths) != len(set(relative_paths))
        or len({path.casefold() for path in relative_paths}) != len(relative_paths)
    ):
        raise OODExternalV2IntegrityError(
            "project source tree contains duplicate or colliding paths"
        )
    files: list[ProjectSourceFileBinding] = []
    for relative_path in relative_paths:
        path = project_root.joinpath(*PurePosixPath(relative_path).parts)
        entry = _stable_runtime_file_entry(path, context="project source file")
        files.append(
            ProjectSourceFileBinding(
                relative_path=relative_path,
                size_bytes=cast(int, entry["size_bytes"]),
                file_sha256=cast(str, entry["sha256"]),
            )
        )
    frozen_files = tuple(files)
    return ProjectSourceTreeBinding(
        files=frozen_files,
        file_count=len(frozen_files),
        total_bytes=sum(item.size_bytes for item in frozen_files),
        tree_sha256=_project_source_tree_sha256(frozen_files),
    )


def _run_git_bytes(
    project_root: Path,
    *arguments: str,
    maximum_bytes: int = 8_000_000,
) -> bytes:
    completed = _execute_bound_git(project_root, *arguments)
    if completed.returncode != 0 or len(completed.stdout) > maximum_bytes:
        raise OODExternalV2IntegrityError("Git blob preflight failed")
    return completed.stdout


def _verify_project_source_tree_at_revisions(
    project_root: Path,
    binding: ProjectSourceTreeBinding,
    *,
    implementation_revision: str,
    execution_revision: str | None,
) -> None:
    observed = _build_project_source_tree(project_root)
    if observed != binding:
        raise OODExternalV2IntegrityError(
            "project Python source tree differs from the frozen child binding"
        )
    bound_paths = tuple(item.relative_path for item in binding.files)
    tracked_output = _run_git(
        project_root,
        "ls-files",
        "--",
        PROJECT_SOURCE_ROOT,
        *PROJECT_OPERATIONAL_ENTRYPOINTS,
    ).stdout.splitlines()
    tracked_paths = tuple(
        sorted(
            path.replace("\\", "/")
            for path in tracked_output
            if path in PROJECT_OPERATIONAL_ENTRYPOINTS
            or (path.startswith(f"{PROJECT_SOURCE_ROOT}/") and path.endswith(".py"))
        )
    )
    if tracked_paths != bound_paths:
        raise OODExternalV2IntegrityError(
            "tracked project Python source set differs from the frozen binding"
        )
    revisions: tuple[str, ...] = (implementation_revision,)
    if execution_revision is not None and execution_revision != implementation_revision:
        revisions += (execution_revision,)
    by_path = {item.relative_path: item for item in binding.files}
    for relative_path in bound_paths:
        current = _read_bounded(
            project_root.joinpath(*PurePosixPath(relative_path).parts),
            8_000_000,
            "bound project source",
        )
        expected = by_path[relative_path]
        if (
            len(current) != expected.size_bytes
            or sha256_bytes(current) != expected.file_sha256
        ):
            raise OODExternalV2IntegrityError(
                "bound project source file differs from child metadata"
            )
        for revision in revisions:
            blob = _run_git_bytes(
                project_root,
                "show",
                f"{revision}:{relative_path}",
            )
            if blob != current:
                raise OODExternalV2IntegrityError(
                    "project source differs from its exact frozen Git blob"
                )
    if _build_project_source_tree(project_root) != binding:
        raise OODExternalV2IntegrityError(
            "project source tree changed during Git verification"
        )


def _verify_imported_project_module_origins(
    project_root: Path,
    binding: ProjectSourceTreeBinding,
) -> None:
    bound_paths = {item.relative_path for item in binding.files}
    observed_modules = 0
    for name, module in tuple(sys.modules.items()):
        if name != "ecg_trust" and not name.startswith("ecg_trust."):
            continue
        if not isinstance(module, ModuleType):
            raise OODExternalV2IntegrityError("project module registry is invalid")
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None)
        if not isinstance(origin, str) or origin in {"built-in", "frozen"}:
            raise OODExternalV2IntegrityError("project module origin is unavailable")
        source = _assert_direct_ancestry(Path(origin), context="project module origin")
        try:
            relative = source.relative_to(project_root).as_posix()
        except ValueError as error:
            raise OODExternalV2IntegrityError(
                "project module was imported from outside the frozen worktree"
            ) from error
        if relative not in bound_paths or source.suffix != ".py":
            raise OODExternalV2IntegrityError(
                "project module origin is not among the frozen source files"
            )
        observed_modules += 1
    if observed_modules == 0:
        raise OODExternalV2IntegrityError("no frozen project modules are imported")


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _verify_all_file_backed_module_origins(
    *,
    project_root: Path,
    project_sources: ProjectSourceTreeBinding,
    python_base_alias: Path,
    python_base_target: Path,
    site_packages: Path,
) -> None:
    """Reject every loaded file-backed module outside the three bound trees."""

    allowed_project = {item.relative_path for item in project_sources.files}
    checked = 0
    main_observed = False

    def validate_file_origin(name: str, raw_origin: str) -> None:
        nonlocal checked, main_observed
        lexical = Path(os.path.abspath(raw_origin))
        if lexical.suffix.casefold() in {".pyc", ".pyo"}:
            raise OODExternalV2IntegrityError(
                "loaded module originated from forbidden bytecode"
            )
        try:
            alias_relative = lexical.relative_to(python_base_alias)
        except ValueError:
            alias_relative = None
        if alias_relative is not None:
            source = _assert_direct_ancestry(
                python_base_target / alias_relative,
                context="loaded CPython module origin",
            )
            if not source.is_file():
                raise OODExternalV2IntegrityError(
                    "loaded CPython module origin is unavailable"
                )
            checked += 1
            return
        source = _assert_direct_ancestry(lexical, context="loaded module origin")
        for allowed_root in (python_base_target, site_packages):
            try:
                source.relative_to(allowed_root)
                checked += 1
                return
            except ValueError:
                continue
        try:
            relative = source.relative_to(project_root).as_posix()
        except ValueError as error:
            raise OODExternalV2IntegrityError(
                "loaded module originated outside every frozen runtime tree"
            ) from error
        if relative not in allowed_project:
            raise OODExternalV2IntegrityError(
                "loaded project module origin is not frozen in the child tree"
            )
        if name == "__main__":
            if relative not in PROJECT_OPERATIONAL_ENTRYPOINTS:
                raise OODExternalV2IntegrityError(
                    "active __main__ is not a frozen protocol entrypoint"
                )
            main_observed = True
        checked += 1

    def has_bound_dynamic_owner(name: str) -> bool:
        parts = name.split(".")
        for length in range(len(parts) - 1, 0, -1):
            owner = sys.modules.get(".".join(parts[:length]))
            if not isinstance(owner, ModuleType):
                continue
            owner_spec = getattr(owner, "__spec__", None)
            owner_origin = getattr(owner_spec, "origin", None)
            if isinstance(owner_origin, str) and owner_origin not in {
                "built-in",
                "frozen",
            }:
                validate_file_origin(name, owner_origin)
                return True
        return False

    for name, module in tuple(sys.modules.items()):
        if not isinstance(module, ModuleType):
            if name in {"typing.io", "typing.re"} and type(module).__name__ == (
                "_DeprecatedType"
            ):
                continue
            raise OODExternalV2IntegrityError(
                "module registry contains an unapproved non-module entry"
            )
        spec = getattr(module, "__spec__", None)
        raw_origin = getattr(spec, "origin", None)
        if raw_origin is None:
            raw_file = getattr(module, "__file__", None)
            if isinstance(raw_file, str) and Path(raw_file).is_absolute():
                raw_origin = raw_file
        if raw_origin == "built-in":
            if (
                name not in sys.builtin_module_names
                or getattr(spec, "loader", None)
                is not importlib.machinery.BuiltinImporter
            ):
                raise OODExternalV2IntegrityError(
                    "module falsely claims a built-in origin"
                )
            continue
        if raw_origin == "frozen":
            canonical_name = getattr(spec, "name", None)
            try:
                canonical_spec = (
                    None
                    if not isinstance(canonical_name, str) or not canonical_name
                    else importlib.machinery.FrozenImporter.find_spec(canonical_name)
                )
            except (AttributeError, ImportError, ValueError):
                canonical_spec = None
            alias_is_exact = name == canonical_name or (
                ALLOWED_FROZEN_MODULE_ALIASES.get(name) == canonical_name
            )
            if (
                not isinstance(canonical_name, str)
                or not canonical_name
                or canonical_name.partition(".")[0] not in sys.stdlib_module_names
                or not alias_is_exact
                or sys.modules.get(canonical_name) is not module
                or getattr(spec, "loader", None)
                is not importlib.machinery.FrozenImporter
                or canonical_spec is None
                or canonical_spec.origin != "frozen"
                or getattr(canonical_spec, "loader", None)
                is not importlib.machinery.FrozenImporter
            ):
                raise OODExternalV2IntegrityError(
                    "module falsely claims a frozen origin"
                )
            continue
        if raw_origin is None:
            locations = getattr(spec, "submodule_search_locations", None)
            if locations is not None:
                raw_locations = tuple(locations)
                if not raw_locations:
                    if has_bound_dynamic_owner(name):
                        continue
                    raise OODExternalV2IntegrityError(
                        "namespace module has no auditable search location"
                    )
                for location in raw_locations:
                    if not isinstance(location, str):
                        raise OODExternalV2IntegrityError(
                            "namespace search location is invalid"
                        )
                    namespace_path = _assert_direct_ancestry(
                        Path(location),
                        context="namespace module search location",
                    )
                    if not namespace_path.is_dir() or not any(
                        _path_is_within(namespace_path, root)
                        for root in (python_base_target, site_packages, project_root)
                    ):
                        raise OODExternalV2IntegrityError(
                            "namespace module search location is outside frozen trees"
                        )
                checked += 1
                continue
            if name == "__mp_main__" and module is sys.modules.get("__main__"):
                continue
            if has_bound_dynamic_owner(name):
                continue
            if name == "cython_runtime" or re.fullmatch(r"_cython_[0-9_]+", name):
                checked += 1
                continue
            raise OODExternalV2IntegrityError(
                "originless module has no bound file-backed owner"
            )
        if not isinstance(raw_origin, str) or not raw_origin:
            raise OODExternalV2IntegrityError("loaded module origin is invalid")
        validate_file_origin(name, raw_origin)
    if checked == 0 or not main_observed:
        raise OODExternalV2IntegrityError(
            "module audit did not observe the exact frozen __main__ entrypoint"
        )


def _verify_tracked_head_blob(
    project_root: Path,
    *,
    revision: str,
    relative_path: str,
    expected_file_sha256: str,
) -> None:
    tracked = _run_git(
        project_root,
        "ls-files",
        "--",
        relative_path,
        allow_empty=True,
    )
    if tracked.stdout.strip().replace("\\", "/") != relative_path:
        raise OODExternalV2IntegrityError("frozen parent is not tracked in Git")
    blob = _run_git_bytes(project_root, "show", f"{revision}:{relative_path}")
    working_path = project_root.joinpath(*PurePosixPath(relative_path).parts)
    if (
        sha256_bytes(blob) != expected_file_sha256
        or _read_bounded(working_path, _CONFIG_MAX_BYTES, "tracked frozen parent")
        != blob
    ):
        raise OODExternalV2IntegrityError(
            "frozen parent working bytes differ from the exact Git HEAD blob"
        )


def _verify_revision_bound_file(
    project_root: Path,
    *,
    revision: str,
    relative_path: str,
    expected_file_sha256: str,
    context: str,
) -> None:
    source = project_root.joinpath(*PurePosixPath(relative_path).parts)
    current = _read_bounded(source, _CHILD_MAX_BYTES, context)
    blob = _run_git_bytes(project_root, "show", f"{revision}:{relative_path}")
    if (
        sha256_bytes(current) != expected_file_sha256
        or sha256_bytes(blob) != expected_file_sha256
        or blob != current
    ):
        raise OODExternalV2IntegrityError(
            f"{context} differs from its exact execution Git blob"
        )


def _verify_historical_revision_blob(
    project_root: Path,
    *,
    revision: str,
    relative_path: str,
    expected_file_sha256: str,
    context: str,
) -> None:
    """Verify historical bytes without requiring the amended worktree to match."""

    object_type = _run_git(
        project_root,
        "cat-file",
        "-t",
        f"{revision}:{relative_path}",
    ).stdout
    blob = _run_git_bytes(project_root, "show", f"{revision}:{relative_path}")
    if object_type != "blob\n" or sha256_bytes(blob) != expected_file_sha256:
        raise OODExternalV2IntegrityError(
            f"{context} differs from its exact historical Git blob"
        )


def _require_git_ignored_and_untracked(
    project_root: Path,
    relative_path: str,
    *,
    context: str,
) -> None:
    ignored = _run_git(
        project_root,
        "check-ignore",
        "--no-index",
        "-q",
        "--",
        relative_path,
        allow_empty=True,
    )
    tracked = _run_git(
        project_root,
        "ls-files",
        "--",
        relative_path,
        allow_empty=True,
    )
    if ignored.returncode != 0 or tracked.stdout.strip() != "":
        raise OODExternalV2IntegrityError(
            f"{context} must be explicitly Git-ignored and untracked"
        )


def _load_routing_components(
    parent: OODExternalV2ParentConfig,
    child: OODExternalV2ChildContract,
    *,
    project_root: Path,
) -> FrozenRoutingComponents:
    source_binding = child.decision_bindings["source_calibration_result"]
    demo_binding = child.decision_bindings["demo_policy"]
    if (
        source_binding.file_sha256 != EXPECTED_SOURCE_CALIBRATION_FILE_SHA256
        or demo_binding.file_sha256 != EXPECTED_DEMO_POLICY_FILE_SHA256
        or source_binding.artifact_sha256
        != EXPECTED_SOURCE_CALIBRATION_ARTIFACT_SHA256
        or demo_binding.artifact_sha256 is not None
    ):
        raise OODExternalV2IntegrityError("child decision bindings differ from frozen components")
    source_path = _resolve_project_relative(
        project_root,
        source_binding.relative_path,
        require_file=True,
    )
    demo_path = _resolve_project_relative(
        project_root,
        demo_binding.relative_path,
        require_file=True,
    )
    if sha256_file(source_path) != source_binding.file_sha256:
        raise OODExternalV2IntegrityError("source-calibration result hash differs")
    if sha256_file(demo_path) != demo_binding.file_sha256:
        raise OODExternalV2IntegrityError("historical demo policy hash differs")
    try:
        source = load_source_calibration_result_bytes(
            _read_bounded(source_path, _V1_RESULT_MAX_BYTES, "source-calibration result")
        )
        demo = FrozenDecisionPolicy.load(demo_path)
    except Exception as error:
        raise OODExternalV2IntegrityError("frozen decision components cannot be loaded") from error
    if (
        source.status != "PREPARED_NOT_RELEASE_READY"
        or source.open_world.status != "PENDING"
        or source.open_world.release_ready
        or source.provenance.historical_policy_file_sha256 != demo_binding.file_sha256
        or source.provenance.checkpoint_sha256 != parent.checkpoint.file_sha256
        or demo.provenance.checkpoint_sha256 != parent.checkpoint.file_sha256.removeprefix(
            "sha256:"
        )
        or demo.provenance.resolved_config_sha256
        != parent.resolved_config.file_sha256.removeprefix("sha256:")
        or demo.provenance.normalization_sha256
        != parent.normalization.file_sha256.removeprefix("sha256:")
    ):
        raise OODExternalV2IntegrityError("frozen decision lineage differs")
    components = source.frozen_components
    temperature = components.temperature.temperature
    maximum_entropy = components.entropy_gate.maximum_entropy
    conformal_summary = components.conformal
    if (
        temperature != EXPECTED_TEMPERATURE
        or maximum_entropy != EXPECTED_ENTROPY_MAXIMUM
        or conformal_summary.alpha != 0.1
        or conformal_summary.n_samples != 834
        or conformal_summary.quantile_rank != 752
        or conformal_summary.quantile_level != 0.9016786570743405
        or tuple(item.label for item in conformal_summary.per_label)
        != tuple(SUPERCLASSES)
        or tuple(item.threshold for item in conformal_summary.per_label)
        != EXPECTED_CONFORMAL_THRESHOLDS
    ):
        raise OODExternalV2IntegrityError("frozen uncertainty components differ")
    conformal = LabelwiseBinaryConformal(
        label_names=tuple(item.label for item in conformal_summary.per_label),
        alpha=conformal_summary.alpha,
        thresholds=tuple(item.threshold for item in conformal_summary.per_label),
        n_calibration_samples=conformal_summary.n_samples,
        quantile_rank=conformal_summary.quantile_rank,
        quantile_level=conformal_summary.quantile_level,
    )
    if conformal.label_names != tuple(SUPERCLASSES):
        raise OODExternalV2IntegrityError("conformal label order differs")
    return FrozenRoutingComponents(
        source_calibration_result=source,
        historical_demo_policy=demo,
        conformal=conformal,
        temperature=temperature,
        maximum_entropy=maximum_entropy,
        source_calibration_file_sha256=source_binding.file_sha256,
        demo_policy_file_sha256=demo_binding.file_sha256,
    )


def _load_model_and_runtime(
    inputs: VerifiedExternalV2Inputs,
) -> tuple[ResNet1D, NormalizationStats, DeterministicCUDARuntime, str]:
    """Load the exact checkpoint without depending on an unrelated demo gate."""

    try:
        resolved_payload: object = json.loads(
            _read_bounded(
                inputs.resolved_config_path,
                _CONFIG_MAX_BYTES,
                "resolved refit config",
            ).decode("utf-8")
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise OODExternalV2IntegrityError("resolved refit config cannot be decoded") from error
    resolved = _mapping(resolved_payload, "resolved refit config")
    if set(resolved) != {"config", "config_hash"}:
        raise OODExternalV2IntegrityError("resolved refit config fields differ")
    inner = _mapping(resolved.get("config"), "resolved inner config")
    config_hash = _digest(resolved.get("config_hash"), "resolved config hash")
    if config_hash != inputs.parent.resolved_config_sha256:
        raise OODExternalV2IntegrityError("resolved config logical hash differs from parent")
    if canonical_sha256(inner) != config_hash:
        raise OODExternalV2IntegrityError("resolved inner config hash differs from content")
    inner_model = _mapping(inner.get("model"), "resolved model")
    if (
        inner.get("architecture") != "resnet1d"
        or inner.get("confirmation_seed") != 2026
        or inner_model.get("architecture") != "resnet1d"
        or inner_model.get("preset") != "matched_capacity"
        or inner_model.get("class") != "ecg_trust.models.resnet1d.ResNet1D"
    ):
        raise OODExternalV2IntegrityError("resolved model selection differs from v1")
    try:
        checkpoint_payload: object = torch.load(
            inputs.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise OODExternalV2IntegrityError("frozen checkpoint cannot be decoded") from error
    checkpoint = _mapping(checkpoint_payload, "checkpoint")
    expected_checkpoint_fields = {
        "config",
        "config_hash",
        "early_stopping_state_dict",
        "epoch",
        "manifest_hash",
        "model_state_dict",
        "optimizer_state_dict",
        "protocol_hash",
        "scaler_state_dict",
        "schema_version",
    }
    if set(checkpoint) != expected_checkpoint_fields or checkpoint.get("schema_version") != 1:
        raise OODExternalV2IntegrityError("frozen checkpoint fields differ")
    if checkpoint.get("config") != inner or checkpoint.get("config_hash") != config_hash:
        raise OODExternalV2IntegrityError("checkpoint and resolved config differ")
    provenance = inputs.v1.policy.provenance
    if (
        _normalize_unprefixed(checkpoint.get("manifest_hash"), "checkpoint manifest hash")
        != provenance.dataset_manifest_file_sha256
        or checkpoint.get("protocol_hash") != provenance.experiment_protocol_sha256
        or provenance.checkpoint_file_sha256 != inputs.parent.checkpoint.file_sha256
        or provenance.resolved_config_file_sha256
        != inputs.parent.resolved_config.file_sha256
        or provenance.resolved_config_sha256 != inputs.parent.resolved_config_sha256
        or provenance.normalization_file_sha256 != inputs.parent.normalization.file_sha256
    ):
        raise OODExternalV2IntegrityError("checkpoint lineage differs from sealed v1 policy")
    try:
        model = build_experiment_model(
            ModelConfig(architecture="resnet1d", preset="matched_capacity")
        )
        model.load_state_dict(cast(Any, checkpoint["model_state_dict"]), strict=True)
    except Exception as error:
        raise OODExternalV2IntegrityError("frozen checkpoint model is incompatible") from error
    if type(model) is not ResNet1D:
        raise OODExternalV2IntegrityError("frozen model is not exact ResNet1D")
    model.requires_grad_(False)
    model.cpu().eval()
    state_hash = model_state_sha256(model)

    try:
        normalization = NormalizationStats.load(inputs.normalization_path)
    except Exception as error:
        raise OODExternalV2IntegrityError("frozen normalization cannot be loaded") from error
    if (
        normalization.provenance.training_folds != (1, 2, 3, 4, 5, 6, 7)
        or normalization.provenance.samples_per_record != 1000
        or normalization.provenance.sampling_frequency_hz != 100.0
        or normalization.provenance.target_columns != TARGET_COLUMNS
    ):
        raise OODExternalV2IntegrityError("normalization scientific contract differs")

    v1_runtime = inputs.v1.result.reference_and_threshold_execution.runtime
    runtime = configure_deterministic_cuda(
        expected_device_name=v1_runtime.device_name,
        expected_compute_capability=(12, 0),
        expected_python_version=v1_runtime.python_version,
        expected_torch_version=v1_runtime.torch_version,
        expected_cuda_runtime=v1_runtime.cuda_runtime_version,
        expected_cudnn_version=v1_runtime.cudnn_version,
        expected_nvidia_driver_version=v1_runtime.nvidia_driver_version,
        nvidia_smi_executable=_nvidia_driver_tool_paths()[0],
    )
    return prepare_resnet_for_embedding(model, runtime=runtime), normalization, runtime, state_hash


def _quality_report_dict(report: SignalQualityReport) -> dict[str, object]:
    """Serialize every quality metric, issue, and decision boundary privately."""

    def issue_dict(issue: Any) -> dict[str, object]:
        return {
            "boundary_value": issue.boundary_value,
            "code": issue.code.value,
            "lead_name": issue.lead_name,
            "metric_name": issue.metric_name,
            "observed_value": issue.observed_value,
            "status": issue.status.value,
        }

    leads: list[dict[str, object]] = []
    for finding in report.leads:
        metrics = finding.metrics
        leads.append(
            {
                "issues": [issue_dict(issue) for issue in finding.issues],
                "lead_index": finding.lead_index,
                "lead_name": finding.lead_name,
                "metrics": {
                    "baseline_wander_power_ratio": metrics.baseline_wander_power_ratio,
                    "clipping_fraction": metrics.clipping_fraction,
                    "flat_step_fraction": metrics.flat_step_fraction,
                    "high_frequency_power_ratio": metrics.high_frequency_power_ratio,
                    "longest_clipping_run_samples": (
                        metrics.longest_clipping_run_samples
                    ),
                    "maximum_absolute_amplitude_mv": (
                        metrics.maximum_absolute_amplitude_mv
                    ),
                    "maximum_step_mv": metrics.maximum_step_mv,
                    "peak_to_peak_mv": metrics.peak_to_peak_mv,
                    "powerline_50hz_power_ratio": metrics.powerline_50hz_power_ratio,
                    "powerline_60hz_power_ratio": metrics.powerline_60hz_power_ratio,
                    "spike_step_fraction": metrics.spike_step_fraction,
                    "standard_deviation_mv": metrics.standard_deviation_mv,
                },
                "reason_codes": [reason.value for reason in finding.reason_codes],
                "status": finding.status.value,
            }
        )
    reversal: dict[str, object] | None = None
    if report.reversal_evidence is not None:
        value = report.reversal_evidence
        reversal = {
            "correlations": [list(item) for item in value.correlations],
            "dominant_polarities": [list(item) for item in value.dominant_polarities],
            "evidence_codes": list(value.evidence_codes),
            "probable_kind": value.probable_kind.value,
            "score": value.score,
        }
    return {
        "config_version": report.config_version,
        "global_issues": [issue_dict(issue) for issue in report.global_issues],
        "leads": leads,
        "reversal_evidence": reversal,
        "status": report.status.value,
    }


def _quality_report_sha256(report: Mapping[str, object]) -> str:
    payload = canonical_json_bytes(dict(report))[:-1]
    return "sha256:" + hashlib.sha256(_QUALITY_REPORT_DOMAIN + payload).hexdigest()


def _evaluate_external_records(
    inputs: VerifiedExternalV2Inputs,
    *,
    model: ResNet1D,
    normalization: NormalizationStats,
    runtime: DeterministicCUDARuntime,
    model_state_before: str,
) -> _EvaluatedExternalRecords:
    """Decode once after the claim, apply quality, then score the frozen path."""

    evidence: list[_PrivateRecordEvidence] = []
    adapter_success_signals: list[Float32Array] = []
    adapter_success_indices: list[int] = []
    quality_signals: list[Float32Array] = []
    quality_indices: list[int] = []
    metadata = SignalMetadata.canonical(DEFAULT_SIGNAL_QUALITY_CONFIG)
    for index, record in enumerate(inputs.inventory.records):
        root = inputs.dataset_roots[record.dataset]
        base = resolve_inventory_record_base(root, record)
        # Adapter/parser/provenance failures are integrity or implementation
        # failures, not natural technical-quality observations.  Once the
        # one-shot claim exists they must produce a failure receipt; only a
        # successfully adapted signal may contribute an INVALID_INPUT route.
        adapted = _adapter_for_record(record, base)
        _verify_adapter_against_inventory(adapted, record)
        adapter_success_indices.append(index)
        adapter_success_signals.append(adapted.signal_mv)
        report = assess_signal_quality(adapted.signal_mv, metadata)
        quality_report = _quality_report_dict(report)
        reason_codes = tuple(reason.value for reason in report.reason_codes)
        if report.status is QualityStatus.INVALID:
            route = "INVALID_INPUT"
        elif report.status is QualityStatus.PASS:
            route = "PENDING_DISTRIBUTION"
            quality_indices.append(index)
            quality_signals.append(adapted.signal_mv)
        else:
            route = "REACQUIRE"
        evidence.append(
            _PrivateRecordEvidence(
                dataset=record.dataset,
                record_ref=record.record_ref,
                patient_key=record.patient_key,
                challenge_quality_label=record.challenge_quality_label,
                adapter_provenance_sha256=adapted.provenance_sha256,
                adapter_source_sample_count=adapted.provenance.source_sample_count,
                adapter_raw_physical_units=adapted.provenance.raw_physical_units,
                canonical_signal_sha256=_tensor_sha256(adapted.signal_mv),
                quality_report_sha256=_quality_report_sha256(quality_report),
                quality_report=quality_report,
                quality_status=report.status.value,
                quality_reason_codes=reason_codes,
                route=route,
                distribution_score=None,
                entropy=None,
                entropy_accepted=None,
                conformal_decisions=None,
                all_conformal_decisions_singleton=None,
            )
        )
    if len(evidence) != len(inputs.inventory.records):
        raise OODExternalV2ExecutionError("selected-record evaluation skipped a record")
    canonical_signals = (
        np.ascontiguousarray(np.stack(adapter_success_signals, axis=0), dtype=np.float32)
        if adapter_success_signals
        else np.empty((0, len(LEADS), TARGET_SAMPLES), dtype=np.float32)
    )
    canonical_indices = np.asarray(adapter_success_indices, dtype=np.int64)
    if not quality_signals:
        empty_embeddings = np.empty((0, 512), dtype=np.float32)
        empty_scores = np.empty((0,), dtype=np.float64)
        empty_logits = np.empty((0, len(SUPERCLASSES)), dtype=np.float64)
        empty_probabilities = np.empty((0, len(SUPERCLASSES)), dtype=np.float64)
        empty_hash = _tensor_sha256(empty_embeddings)
        empty_logits_hash = _tensor_sha256(empty_logits)
        empty_probabilities_hash = _tensor_sha256(empty_probabilities)
        model_state_after = model_state_sha256(model)
        unchanged = model_state_after == model_state_before
        if not unchanged:
            raise OODExternalV2IntegrityError("frozen model state changed during evaluation")
        return _EvaluatedExternalRecords(
            records=tuple(evidence),
            adapter_success_inventory_indices=canonical_indices,
            canonical_signals=canonical_signals,
            quality_pass_inventory_indices=np.empty((0,), dtype=np.int64),
            embeddings=empty_embeddings,
            repeated_embeddings=empty_embeddings.copy(),
            repeated_embedding_sha256=empty_hash,
            embedding_sha256=empty_hash,
            scores=empty_scores,
            logits=empty_logits,
            repeated_logits=empty_logits.copy(),
            probabilities=empty_probabilities,
            first_logits_sha256=empty_logits_hash,
            repeated_logits_sha256=empty_logits_hash,
            probabilities_sha256=empty_probabilities_hash,
            model_state_before_sha256=model_state_before,
            model_state_after_sha256=model_state_after,
            model_state_unchanged=unchanged,
        )

    signals = np.ascontiguousarray(np.stack(quality_signals, axis=0), dtype=np.float32)
    dataset = _NormalizedSignalDataset(signals, normalization)
    passes = extract_embeddings_twice(model, dataset, runtime=runtime)
    first_hash = _tensor_sha256(passes.first)
    repeated_hash = _tensor_sha256(passes.repeated)
    if first_hash != repeated_hash:
        raise OODExternalV2IntegrityError("repeated external embedding hashes differ")
    detector = inputs.v1.policy.to_detector()
    first_scores = np.ascontiguousarray(detector.score(passes.first), dtype=np.float64)
    repeated_scores = np.ascontiguousarray(detector.score(passes.repeated), dtype=np.float64)
    if not np.array_equal(first_scores, repeated_scores):
        raise OODExternalV2IntegrityError("repeated external distribution scores differ")

    first_logits = _classify_embeddings(model, passes.first, runtime=runtime)
    repeated_logits = _classify_embeddings(model, passes.repeated, runtime=runtime)
    if not np.array_equal(first_logits, repeated_logits):
        raise OODExternalV2IntegrityError("repeated external logits differ")
    probabilities = _sigmoid(first_logits / inputs.routing.temperature)
    entropy = normalized_bernoulli_entropy(probabilities)
    prediction_sets = inputs.routing.conformal.predict(probabilities)
    singleton = np.asarray(
        [
            all(decision is not BinaryDecision.UNCERTAIN for decision in row)
            for row in prediction_sets.decisions
        ],
        dtype=np.bool_,
    )
    entropy_accepted = entropy <= inputs.routing.maximum_entropy
    if not (
        first_scores.shape == entropy.shape == singleton.shape == (len(quality_indices),)
    ):
        raise OODExternalV2ExecutionError("external routing arrays are misaligned")

    threshold = inputs.parent.threshold
    for local_index, inventory_index in enumerate(quality_indices):
        score = float(first_scores[local_index])
        accepted_entropy = bool(entropy_accepted[local_index])
        all_singleton = bool(singleton[local_index])
        decisions = tuple(
            decision.value for decision in prediction_sets.decisions[local_index]
        )
        if score > threshold:
            route = "UNSUPPORTED_INPUT"
        elif not accepted_entropy or not all_singleton:
            route = "ABSTAIN"
        else:
            route = "PREDICTION_ALLOWED"
        prior = evidence[inventory_index]
        evidence[inventory_index] = _PrivateRecordEvidence(
            dataset=prior.dataset,
            record_ref=prior.record_ref,
            patient_key=prior.patient_key,
            challenge_quality_label=prior.challenge_quality_label,
            adapter_provenance_sha256=prior.adapter_provenance_sha256,
            adapter_source_sample_count=prior.adapter_source_sample_count,
            adapter_raw_physical_units=prior.adapter_raw_physical_units,
            canonical_signal_sha256=prior.canonical_signal_sha256,
            quality_report_sha256=prior.quality_report_sha256,
            quality_report=prior.quality_report,
            quality_status=prior.quality_status,
            quality_reason_codes=prior.quality_reason_codes,
            route=route,
            distribution_score=score,
            entropy=float(entropy[local_index]),
            entropy_accepted=accepted_entropy,
            conformal_decisions=decisions,
            all_conformal_decisions_singleton=all_singleton,
        )
    if any(record.route == "PENDING_DISTRIBUTION" for record in evidence):
        raise OODExternalV2ExecutionError("quality-pass record did not receive a final route")
    first_logits_hash = _tensor_sha256(first_logits)
    repeated_logits_hash = _tensor_sha256(repeated_logits)
    probabilities_hash = _tensor_sha256(probabilities)
    model_state_after = model_state_sha256(model)
    unchanged = model_state_after == model_state_before
    if not unchanged:
        raise OODExternalV2IntegrityError("frozen model state changed during evaluation")
    return _EvaluatedExternalRecords(
        records=tuple(evidence),
        adapter_success_inventory_indices=canonical_indices,
        canonical_signals=canonical_signals,
        quality_pass_inventory_indices=np.asarray(quality_indices, dtype=np.int64),
        embeddings=passes.first,
        repeated_embeddings=passes.repeated,
        repeated_embedding_sha256=repeated_hash,
        embedding_sha256=first_hash,
        scores=first_scores,
        logits=first_logits,
        repeated_logits=repeated_logits,
        probabilities=probabilities,
        first_logits_sha256=first_logits_hash,
        repeated_logits_sha256=repeated_logits_hash,
        probabilities_sha256=probabilities_hash,
        model_state_before_sha256=model_state_before,
        model_state_after_sha256=model_state_after,
        model_state_unchanged=unchanged,
    )


def _adapter_for_record(
    record: ExternalInventoryRecord,
    record_base: Path,
) -> CanonicalExternalSignal:
    if record.dataset == CHALLENGE_2011_DATASET:
        return load_challenge_2011_signal(record_base)
    if record.dataset == ZZU_PEDIATRIC_DATASET:
        return load_zzu_pediatric_signal(record_base)
    raise OODExternalV2IntegrityError("inventory selected a forbidden dataset")


def _verify_adapter_against_inventory(
    adapted: CanonicalExternalSignal,
    record: ExternalInventoryRecord,
) -> None:
    provenance = adapted.provenance
    expected_source_sample_count = _expected_source_sample_count(record)
    if (
        provenance.raw_header_sha256 != record.raw_header_sha256
        or provenance.raw_header_size_bytes != record.raw_header_size_bytes
        or provenance.raw_data_sha256 != record.raw_data_sha256
        or provenance.raw_data_size_bytes != record.raw_data_size_bytes
        or provenance.source_frequency_hz != record.sampling_frequency_hz
        or provenance.source_sample_count != expected_source_sample_count
        or provenance.source_duration_seconds != record.duration_seconds
        or provenance.source_lead_names != record.raw_ordered_leads
        or provenance.canonical_leads != record.canonical_ordered_leads
        or provenance.output_leads != LEADS
        or provenance.source_data_file_names != record.raw_data_file_names
        or provenance.raw_physical_units != record.raw_physical_units
        or provenance.physical_units != PHYSICAL_UNITS
    ):
        raise ExternalECGAdapterError("adapter provenance differs from frozen inventory")


def _expected_source_sample_count(record: ExternalInventoryRecord) -> int:
    exact = Fraction(str(record.sampling_frequency_hz)) * Fraction(
        str(record.duration_seconds)
    )
    if (
        exact.denominator != 1
        or exact.numerator <= 0
        or exact.numerator != record.source_sample_count
    ):
        raise OODExternalV2IntegrityError(
            "inventory frequency and duration do not define an exact source sample count"
        )
    return exact.numerator


def _classify_embeddings(
    model: ResNet1D,
    embeddings: Float32Array,
    *,
    runtime: DeterministicCUDARuntime,
) -> Float64Array:
    batches: list[Float64Array] = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        for start in range(0, embeddings.shape[0], 128):
            stop = min(embeddings.shape[0], start + 128)
            batch = torch.from_numpy(embeddings[start:stop]).to(
                runtime.device,
                dtype=torch.float32,
            )
            logits = model.classifier(model.classifier_dropout(batch))
            if (
                logits.shape != (stop - start, len(SUPERCLASSES))
                or logits.dtype is not torch.float32
                or not torch.isfinite(logits).all().item()
            ):
                raise OODExternalV2ExecutionError("frozen classifier logits are invalid")
            batches.append(
                np.ascontiguousarray(
                    logits.detach().cpu().numpy(),
                    dtype=np.float64,
                )
            )
    if not batches:
        raise OODExternalV2ExecutionError("frozen classifier produced no logits")
    return np.ascontiguousarray(np.concatenate(batches, axis=0), dtype=np.float64)


def _sigmoid(values: Float64Array) -> Float64Array:
    result = np.empty_like(values, dtype=np.float64)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponent = np.exp(values[~nonnegative])
    result[~nonnegative] = exponent / (1.0 + exponent)
    if not np.isfinite(result).all() or np.any((result < 0.0) | (result > 1.0)):
        raise OODExternalV2ExecutionError("calibrated probabilities are invalid")
    return result


def _build_endpoint_evidence(
    evaluated: _EvaluatedExternalRecords,
    *,
    inputs: VerifiedExternalV2Inputs,
) -> _EndpointEvidence:
    records = evaluated.records
    challenge_indices = [
        index for index, record in enumerate(records) if record.dataset == CHALLENGE_2011_DATASET
    ]
    group3_indices = [
        index
        for index in challenge_indices
        if records[index].challenge_quality_label == "unacceptable"
    ]
    group1_indices = [
        index
        for index in challenge_indices
        if records[index].challenge_quality_label == "acceptable"
    ]
    challenge_pass = [
        index
        for index in challenge_indices
        if records[index].quality_status == QualityStatus.PASS.value
    ]
    zzu_pass = [
        index
        for index, record in enumerate(records)
        if record.dataset == ZZU_PEDIATRIC_DATASET
        and record.quality_status == QualityStatus.PASS.value
    ]
    group3_blocked = np.asarray(
        [
            records[index].quality_status == QualityStatus.REACQUIRE.value
            for index in group3_indices
        ],
        dtype=np.bool_,
    )
    group1_passed = np.asarray(
        [
            records[index].quality_status == QualityStatus.PASS.value
            for index in group1_indices
        ],
        dtype=np.bool_,
    )
    challenge_detected = np.asarray(
        [
            cast(float, records[index].distribution_score) > inputs.parent.threshold
            for index in challenge_pass
        ],
        dtype=np.bool_,
    )
    zzu_detected = np.asarray(
        [
            cast(float, records[index].distribution_score) > inputs.parent.threshold
            for index in zzu_pass
        ],
        dtype=np.bool_,
    )
    zzu_patient_keys = [cast(str, records[index].patient_key) for index in zzu_pass]
    patient_index = {
        key: value + 1 for value, key in enumerate(sorted(set(zzu_patient_keys)))
    }
    zzu_clusters = np.asarray(
        [patient_index[key] for key in zzu_patient_keys],
        dtype=np.int64,
    )
    challenge_subset = build_external_inventory(
        tuple(
            record
            for record in inputs.inventory.records
            if record.dataset == CHALLENGE_2011_DATASET
        )
    )
    zzu_subset = build_external_inventory(
        tuple(
            record for record in inputs.inventory.records if record.dataset == ZZU_PEDIATRIC_DATASET
        )
    )
    parent = inputs.parent
    technical_values: list[object] = []
    external_values: list[object] = []
    replicate_values: dict[str, Float64Array] = {}
    if group3_blocked.size:
        technical_values.append(
            evaluate_technical_quality_gate(
                group3_blocked,
                endpoint_key="challenge_group3_technical_block_sensitivity",
                cohort_key="physionet-challenge-2011-set-a",
                event_definition="block_unacceptable",
                resampling_unit=ResamplingUnit.RECORD,
                minimum_rate=parent.challenge_group3_minimum,
                seed=parent.challenge_bootstrap_seed,
                replicates=parent.bootstrap_resamples,
                confidence_level=parent.confidence_level,
            )
        )
        replicate_values["challenge_group3_technical_block_sensitivity"] = (
            _bootstrap_rates(
                group3_blocked,
                resampling_unit=ResamplingUnit.RECORD,
                seed=parent.challenge_bootstrap_seed,
                replicates=parent.bootstrap_resamples,
            )
        )
    if group1_passed.size:
        technical_values.append(
            evaluate_technical_quality_gate(
                group1_passed,
                endpoint_key="challenge_group1_quality_pass_rate",
                cohort_key="physionet-challenge-2011-set-a",
                event_definition="pass_acceptable",
                resampling_unit=ResamplingUnit.RECORD,
                minimum_rate=parent.challenge_group1_minimum,
                seed=parent.challenge_bootstrap_seed,
                replicates=parent.bootstrap_resamples,
                confidence_level=parent.confidence_level,
            )
        )
        replicate_values["challenge_group1_quality_pass_rate"] = _bootstrap_rates(
            group1_passed,
            resampling_unit=ResamplingUnit.RECORD,
            seed=parent.challenge_bootstrap_seed,
            replicates=parent.bootstrap_resamples,
        )
    if challenge_detected.size:
        external_values.append(
            evaluate_external_ood_gate(
                challenge_detected,
                endpoint_key="challenge_external_distribution_recall",
                cohort_key="physionet-challenge-2011-set-a",
                dataset_name="PhysioNet Challenge 2011 Set A",
                dataset_version=CHALLENGE_2011_VERSION,
                license_identifier="ODC-By-1.0",
                cohort_manifest_sha256=challenge_subset.inventory_sha256,
                role_assignment_sha256=inputs.child.artifact_sha256,
                evaluation_role=ExternalCohortRole.PHYSIONET_CHALLENGE_2011_SET_A,
                ood_axis=OODAxis.EXTERNAL_ACQUISITION_AND_POPULATION,
                resampling_unit=ResamplingUnit.RECORD,
                minimum_ood_recall=parent.challenge_distribution_minimum,
                seed=parent.challenge_bootstrap_seed,
                replicates=parent.bootstrap_resamples,
                confidence_level=parent.confidence_level,
            )
        )
        replicate_values["challenge_external_distribution_recall"] = _bootstrap_rates(
            challenge_detected,
            resampling_unit=ResamplingUnit.RECORD,
            seed=parent.challenge_bootstrap_seed,
            replicates=parent.bootstrap_resamples,
        )
    if zzu_detected.size:
        external_values.append(
            evaluate_external_ood_gate(
                zzu_detected,
                endpoint_key="zzu_external_distribution_recall",
                cohort_key="zzu-pecg-v1",
                dataset_name="ZZU pediatric ECG",
                dataset_version=ZZU_PEDIATRIC_VERSION,
                license_identifier="CC-BY-4.0",
                cohort_manifest_sha256=zzu_subset.inventory_sha256,
                role_assignment_sha256=inputs.child.artifact_sha256,
                evaluation_role=ExternalCohortRole.ZZU_PECG_V1,
                ood_axis=OODAxis.PEDIATRIC_POPULATION_AND_ACQUISITION,
                resampling_unit=ResamplingUnit.PATIENT_CLUSTER,
                cluster_labels=zzu_clusters,
                subjects=len(patient_index),
                minimum_ood_recall=parent.zzu_distribution_minimum,
                seed=parent.zzu_bootstrap_seed,
                replicates=parent.bootstrap_resamples,
                confidence_level=parent.confidence_level,
            )
        )
        replicate_values["zzu_external_distribution_recall"] = _bootstrap_rates(
            zzu_detected,
            resampling_unit=ResamplingUnit.PATIENT_CLUSTER,
            cluster_labels=zzu_clusters,
            seed=parent.zzu_bootstrap_seed,
            replicates=parent.bootstrap_resamples,
        )
    technical = tuple(technical_values)
    external = tuple(external_values)
    replicate_arrays = MappingProxyType(dict(sorted(replicate_values.items())))
    _verify_replicate_quantiles(
        technical,
        external,
        replicate_arrays,
        parent=parent,
    )
    observed_routes = Counter(record.route for record in records)
    if set(observed_routes) - set(FROZEN_ROUTE_ORDER):
        raise OODExternalV2IntegrityError("record evidence contains an unknown route")
    route_counts = MappingProxyType(
        {route: observed_routes.get(route, 0) for route in FROZEN_ROUTE_ORDER}
    )
    group3_prediction_allowed = sum(
        records[index].route == "PREDICTION_ALLOWED" for index in group3_indices
    )
    return _EndpointEvidence(
        external_cohorts=external,
        technical_quality_endpoints=technical,
        bootstrap_replicates=replicate_arrays,
        challenge_group3_prediction_allowed_count=group3_prediction_allowed,
        route_counts=route_counts,
    )


def _bootstrap_rates(
    events: BoolArray,
    *,
    resampling_unit: ResamplingUnit,
    seed: int,
    replicates: int,
    cluster_labels: Int64Array | None = None,
) -> Float64Array:
    generator = np.random.Generator(np.random.PCG64(seed))
    records = int(events.shape[0])
    if resampling_unit is ResamplingUnit.RECORD:
        chunk_size = max(1, min(replicates, 1_000_000 // records))
        rates = np.empty(replicates, dtype=np.float64)
        for start in range(0, replicates, chunk_size):
            stop = min(replicates, start + chunk_size)
            sampled = generator.integers(
                0,
                records,
                size=(stop - start, records),
                endpoint=False,
            )
            rates[start:stop] = events[sampled].mean(axis=1, dtype=np.float64)
        return rates
    if cluster_labels is None or cluster_labels.shape != events.shape:
        raise OODExternalV2ExecutionError("patient-cluster bootstrap labels are invalid")
    unique, inverse = np.unique(cluster_labels, return_inverse=True)
    clusters = len(unique)
    record_counts = np.bincount(inverse, minlength=clusters).astype(np.int64, copy=False)
    event_counts = np.bincount(
        inverse,
        weights=events.astype(np.int64),
        minlength=clusters,
    ).astype(np.int64, copy=False)
    chunk_size = max(1, min(replicates, 1_000_000 // clusters))
    rates = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, chunk_size):
        stop = min(replicates, start + chunk_size)
        sampled = generator.integers(0, clusters, size=(stop - start, clusters), endpoint=False)
        denominator = record_counts[sampled].sum(axis=1, dtype=np.int64)
        numerator = event_counts[sampled].sum(axis=1, dtype=np.int64)
        if np.any(denominator <= 0):
            raise OODExternalV2ExecutionError("bootstrap produced an empty replicate")
        rates[start:stop] = numerator.astype(np.float64) / denominator.astype(np.float64)
    return rates


def _verify_replicate_quantiles(
    technical: tuple[object, ...],
    external: tuple[object, ...],
    arrays: Mapping[str, Float64Array],
    *,
    parent: OODExternalV2ParentConfig,
) -> None:
    summaries = {
        cast(Any, summary).endpoint_key: cast(Any, summary)
        for summary in (*technical, *external)
    }
    if parent.confidence_level != 0.9875:
        raise OODExternalV2IntegrityError(
            "bootstrap confidence differs from the frozen multiplicity contract"
        )
    for endpoint_key, values in arrays.items():
        quantiles = np.quantile(
            values,
            np.asarray([0.00625, 0.99375, 0.0125, 0.9875], dtype=np.float64),
            method="linear",
        )
        interval = summaries[endpoint_key].interval
        observed = np.asarray(
            [
                interval.two_sided_lower,
                interval.two_sided_upper,
                interval.one_sided_lower,
                interval.one_sided_upper,
            ],
            dtype=np.float64,
        )
        if not np.array_equal(quantiles, observed):
            raise OODExternalV2IntegrityError("stored bootstrap replicates differ from endpoint")


def _raw_adapter_evidence_complete(
    records: tuple[_PrivateRecordEvidence, ...],
    *,
    skipped: int,
) -> bool:
    """Keep adapter integrity separate from natural signal-quality invalidity."""

    return skipped == 0 and all(
        record.adapter_provenance_sha256 is not None
        and record.adapter_source_sample_count is not None
        and record.adapter_raw_physical_units == (PHYSICAL_UNITS,) * len(LEADS)
        and record.canonical_signal_sha256 is not None
        and record.quality_report_sha256 is not None
        for record in records
    )


def _build_result(
    evaluated: _EvaluatedExternalRecords,
    endpoints: _EndpointEvidence,
    *,
    inputs: VerifiedExternalV2Inputs,
    code_revision: str,
    v1_unchanged: bool,
    inventory_unchanged: bool,
    raw_source_to_canonical_replay_verified: bool,
    full_backbone_embedding_replay_verified: bool,
) -> OODV2Result:
    challenge_records = tuple(
        record
        for record in evaluated.records
        if record.dataset == CHALLENGE_2011_DATASET
    )
    challenge_labels_complete = (
        len(challenge_records) == inputs.parent.challenge_expected_records
        and all(
            record.challenge_quality_label
            in {"acceptable", "unacceptable", "indeterminate"}
            for record in challenge_records
        )
    )
    challenge_invalid = sum(record.route == "INVALID_INPUT" for record in challenge_records)
    challenge_quality_pass_count = sum(
        record.quality_status == QualityStatus.PASS.value
        for record in challenge_records
    )
    zzu_records = tuple(
        record
        for record in evaluated.records
        if record.dataset == ZZU_PEDIATRIC_DATASET
    )
    zzu_selected = tuple(
        record
        for record in inputs.inventory.records
        if record.dataset == ZZU_PEDIATRIC_DATASET
    )
    zzu_invalid = sum(record.route == "INVALID_INPUT" for record in zzu_records)
    zzu_quality_pass = tuple(
        record
        for record in zzu_records
        if record.quality_status == QualityStatus.PASS.value
    )
    zzu_selected_patients = {
        record.patient_key
        for record in zzu_selected
        if record.patient_key is not None
    }
    zzu_quality_pass_patients = {
        record.patient_key
        for record in zzu_quality_pass
        if record.patient_key is not None
    }
    zzu_selected_count = len(zzu_selected)
    zzu_selected_patient_count = len(zzu_selected_patients)
    zzu_pass_count = len(zzu_quality_pass)
    zzu_pass_patient_count = len(zzu_quality_pass_patients)
    if zzu_selected_count == 0 or zzu_selected_patient_count == 0:
        raise OODExternalV2IntegrityError("ZZU selected denominator is empty")
    zzu_record_coverage = zzu_pass_count / zzu_selected_count
    zzu_patient_coverage = zzu_pass_patient_count / zzu_selected_patient_count
    skipped = len(inputs.inventory.records) - len(evaluated.records)
    raw_adapter_bindings_verified = _raw_adapter_evidence_complete(
        evaluated.records,
        skipped=skipped,
    )
    deterministic_embeddings_match = (
        evaluated.embedding_sha256 == evaluated.repeated_embedding_sha256
        and evaluated.model_state_unchanged
    )
    all_hard_passed = all(
        (
            challenge_labels_complete,
            challenge_invalid == 0,
            challenge_quality_pass_count >= 1,
            zzu_invalid == 0,
            zzu_record_coverage >= 0.80,
            zzu_patient_coverage >= 0.80,
            endpoints.challenge_group3_prediction_allowed_count == 0,
            skipped == 0,
            v1_unchanged,
            inventory_unchanged,
            raw_adapter_bindings_verified,
            deterministic_embeddings_match,
            raw_source_to_canonical_replay_verified,
            full_backbone_embedding_replay_verified,
        )
    )
    hard_gates = ExternalOODHardGates(
        challenge_reference_label_alignment_complete=challenge_labels_complete,
        challenge_invalid_input_count=challenge_invalid,
        challenge_quality_pass_records=challenge_quality_pass_count,
        zzu_invalid_input_count=zzu_invalid,
        zzu_selected_records=zzu_selected_count,
        zzu_quality_pass_records=zzu_pass_count,
        zzu_quality_pass_record_coverage=zzu_record_coverage,
        zzu_selected_patients=zzu_selected_patient_count,
        zzu_quality_pass_patients=zzu_pass_patient_count,
        zzu_quality_pass_patient_coverage=zzu_patient_coverage,
        challenge_group3_prediction_allowed_count=(
            endpoints.challenge_group3_prediction_allowed_count
        ),
        skipped_selected_records=skipped,
        target_site_fitting_performed=False,
        v1_policy_bytes_unchanged_before_and_after=v1_unchanged,
        exact_v1_whole_bundle_verifier_passes=True,
        external_raw_sources_verified_before_and_after=inventory_unchanged,
        exact_dataset_roots_verified=True,
        exact_selected_input_inventory_verified_before_and_after=inventory_unchanged,
        semantic_roles_rederived_before_and_after=inventory_unchanged,
        raw_canonical_lead_and_data_file_bindings_verified=(
            raw_adapter_bindings_verified
        ),
        active_scientific_package_versions_match_child=True,
        deterministic_repeated_embeddings_match=deterministic_embeddings_match,
        raw_source_to_canonical_signal_replay_matches=(
            raw_source_to_canonical_replay_verified
        ),
        canonical_signal_to_full_backbone_embedding_replay_matches=(
            full_backbone_embedding_replay_verified
        ),
        aggregate_only_publication_verified=True,
        # These two fields describe the terminal transaction produced by
        # prepare_ood_external_v2.  Publication happens only after semantic
        # preverification; a returned result implies terminal verification.
        immutable_success_bundle_verifies=True,
        failure_receipt_exists=False,
        all_passed=all_hard_passed,
    )
    endpoints_complete = (
        len(endpoints.external_cohorts) == 2
        and len(endpoints.technical_quality_endpoints) == 2
    )
    endpoint_pass = all(
        cast(Any, endpoint).gate_passed
        for endpoint in (*endpoints.external_cohorts, *endpoints.technical_quality_endpoints)
    )
    eligible = endpoints_complete and endpoint_pass and hard_gates.all_passed
    if not endpoints_complete:
        status = OODV2Status.EXTERNAL_OOD_INSUFFICIENT_EVIDENCE
    elif eligible:
        status = OODV2Status.EXTERNAL_OOD_EVIDENCE_COMPLETE
    else:
        status = OODV2Status.EXTERNAL_OOD_TARGET_MISSED
    requirements = EvidenceRequirements(
        family_wise_alpha=0.05,
        multiplicity_method="bonferroni",
        co_primary_endpoint_count=4,
        one_sided_alpha_per_endpoint=0.0125,
        co_primary_confidence_level=inputs.parent.confidence_level,
        bootstrap_replicates=inputs.parent.bootstrap_resamples,
        challenge_bootstrap_seed=inputs.parent.challenge_bootstrap_seed,
        zzu_bootstrap_seed=inputs.parent.zzu_bootstrap_seed,
    )
    integrity = OODV2IntegritySummary(
        preregistration_frozen_before_evaluation=True,
        cohort_roles_frozen_before_model_outputs=True,
        dataset_hashes_verified=inventory_unchanged,
        overlap_exclusions_verified=True,
        frozen_detector_verified=v1_unchanged,
        evaluation_alignment_verified=(
            len(evaluated.records) == len(inputs.inventory.records)
            and len(evaluated.quality_pass_inventory_indices) == len(evaluated.scores)
        ),
        aggregate_only_result_verified=True,
        sealed_v1_unchanged_verified=v1_unchanged,
        sealed_v1_source_validation_used_for_tuning=False,
        target_site_fitting_performed=False,
        complete=True,
    )
    return seal_ood_v2_result(
        OODV2ResultBody(
            schema_version=1,
            artifact_type=OOD_V2_ARTIFACT_TYPE,
            protocol_id=PROTOCOL_ID,
            frozen_at_utc=inputs.child.frozen_at_utc,
            status=status,
            preregistration_sha256=inputs.parent.file_sha256,
            cohort_role_manifest_sha256=inputs.child.artifact_sha256,
            detector_policy_sha256=inputs.parent.v1_distribution_policy.file_sha256,
            sealed_v1_result_sha256=inputs.parent.v1_result.file_sha256,
            sealed_v1_claim_sha256=inputs.v1.claim_file_sha256,
            code_revision=code_revision,
            evidence_requirements=requirements,
            source_gate=_historical_source_gate(inputs.v1.result),
            external_cohorts=cast(Any, endpoints.external_cohorts),
            technical_quality_endpoints=cast(
                Any,
                endpoints.technical_quality_endpoints,
            ),
            final_route_counts=AggregateRouteCounts.model_validate(
                {
                    **dict(endpoints.route_counts),
                    "total_records": len(evaluated.records),
                }
            ),
            hard_gates=hard_gates,
            integrity=integrity,
            external_evidence_eligible=eligible,
            integration_permitted=False,
            aggregate_only=True,
            research_only=True,
            clinical_validation=False,
        )
    )


def _historical_source_gate(result: OODCompletionResult) -> SourceGateSummary:
    source = result.source_validation
    interval = source.cluster_bootstrap
    historical = HistoricalSourceBootstrapInterval(
        method="historical_patient_cluster_percentile_bootstrap",
        estimator="record_weighted_event_rate",
        resampling_unit=ResamplingUnit.PATIENT_CLUSTER,
        sampling_with_replacement=True,
        random_generator="numpy.random.Generator_PCG64",
        seed=interval.seed,
        replicates=interval.replicates,
        percentile_function="numpy.quantile",
        quantile_method="linear",
        confidence_level=0.95,
        records=source.records,
        resampling_units=source.patients,
        event_count=source.rejected_records,
        point_estimate=source.record_false_rejection_rate,
        two_sided_lower=interval.two_sided_lower,
        two_sided_upper=interval.two_sided_upper,
        one_sided_upper=interval.one_sided_upper,
        one_sided_lower_published=False,
    )
    return SourceGateSummary(
        cohort_key="sealed-v1-source-validation",
        cohort_manifest_sha256=source.source_assignment_sha256,
        evaluation_role="source_retention",
        records=source.records,
        subjects=source.patients,
        rejected_records=source.rejected_records,
        retained_records=source.accepted_records,
        false_rejection_rate=source.record_false_rejection_rate,
        support_coverage=source.source_record_support_coverage,
        maximum_false_rejection_rate=source.maximum_allowed_record_false_rejection_rate,
        interval=historical,
        gate_passed=False,
        sealed_v1_source_validation_used_for_tuning=False,
        public_contains_record_level_outputs=False,
    )


def _write_private_artifacts(
    staging_root: Path,
    *,
    inputs: VerifiedExternalV2Inputs,
    evaluated: _EvaluatedExternalRecords,
    endpoints: _EndpointEvidence,
) -> None:
    private = staging_root / "private"
    private.mkdir(parents=False, exist_ok=False)
    _atomic_write_new(
        private / "external-inventory.json",
        inputs.inventory.to_canonical_json_bytes(),
    )
    _copy_frozen_lineage_inputs(private, inputs=inputs)
    _write_private_routing_contract(private, inputs=inputs, evaluated=evaluated)
    _write_canonical_signal_bundle(private, inputs=inputs, evaluated=evaluated)
    _write_quality_audit_shards(private, inputs=inputs, evaluated=evaluated)
    evidence_body: dict[str, object] = {
        "artifact_type": PRIVATE_EVIDENCE_ARTIFACT_TYPE,
        "child_contract_file_sha256": inputs.child.file_sha256,
        "decision_bindings": {
            "demo_policy_file_sha256": inputs.routing.demo_policy_file_sha256,
            "source_calibration_file_sha256": (
                inputs.routing.source_calibration_file_sha256
            ),
        },
        "inventory_sha256": inputs.inventory.inventory_sha256,
        "parent_config_file_sha256": inputs.parent.file_sha256,
        "protocol_id": PROTOCOL_ID,
        "record_count": len(evaluated.records),
        "records": [_private_record_index_dict(record) for record in evaluated.records],
        "route_counts": dict(endpoints.route_counts),
        "schema_version": 1,
        "threshold": inputs.parent.threshold,
    }
    evidence_body["artifact_sha256"] = canonical_sha256(evidence_body)
    _atomic_write_new(
        private / "record-evidence.json",
        canonical_json_bytes(evidence_body),
    )
    _write_embedding_bundle(private, inputs=inputs, evaluated=evaluated)
    _write_bootstrap_bundle(private, inputs=inputs, endpoints=endpoints)


def _private_record_index_dict(record: _PrivateRecordEvidence) -> dict[str, object]:
    """Serialize only the bounded row index; full quality bodies live in shards."""

    payload = record.to_dict()
    payload["quality_report"] = None
    return payload


def _copy_frozen_lineage_inputs(
    private: Path,
    *,
    inputs: VerifiedExternalV2Inputs,
) -> None:
    """Copy exact lineage/decision bytes into the manifest-covered private root."""

    bindings = (
        (
            inputs.parent.path,
            inputs.parent.file_sha256,
            _CONFIG_MAX_BYTES,
            "frozen parent protocol",
            "frozen-parent-config.yaml",
        ),
        (
            _resolve_project_relative(
                inputs.project_root,
                inputs.parent.v1_result.relative_path,
                require_file=True,
            ),
            inputs.parent.v1_result.file_sha256,
            _V1_RESULT_MAX_BYTES,
            "sealed v1 aggregate result",
            "frozen-v1-ood-completion-result.json",
        ),
        (
            inputs.child.path,
            inputs.child.file_sha256,
            _CHILD_MAX_BYTES,
            "frozen child contract",
            "frozen-child-contract.json",
        ),
        (
            _resolve_project_relative(
                inputs.project_root,
                inputs.child.decision_bindings[
                    "source_calibration_result"
                ].relative_path,
                require_file=True,
            ),
            inputs.routing.source_calibration_file_sha256,
            _V1_RESULT_MAX_BYTES,
            "frozen source-calibration result",
            "frozen-source-calibration-result.json",
        ),
        (
            _resolve_project_relative(
                inputs.project_root,
                inputs.child.decision_bindings["demo_policy"].relative_path,
                require_file=True,
            ),
            inputs.routing.demo_policy_file_sha256,
            _V1_RESULT_MAX_BYTES,
            "frozen demo policy",
            "frozen-demo-policy.json",
        ),
    )
    for source, expected_hash, maximum, context, destination_name in bindings:
        payload = _read_bounded(source, maximum, context)
        if sha256_bytes(payload) != expected_hash:
            raise OODExternalV2IntegrityError(f"{context} changed before private copy")
        _atomic_write_new(private / destination_name, payload)


def _write_private_routing_contract(
    private: Path,
    *,
    inputs: VerifiedExternalV2Inputs,
    evaluated: _EvaluatedExternalRecords,
) -> None:
    source_policy_path = _resolve_project_relative(
        inputs.project_root,
        inputs.parent.v1_distribution_policy.relative_path,
        require_file=True,
    )
    policy_bytes = _read_bounded(
        source_policy_path,
        _V1_POLICY_MAX_BYTES,
        "frozen distribution policy",
    )
    if sha256_bytes(policy_bytes) != inputs.parent.v1_distribution_policy.file_sha256:
        raise OODExternalV2IntegrityError("frozen distribution policy changed")
    copied_policy_path = private / "frozen-distribution-policy.json"
    _atomic_write_new(copied_policy_path, policy_bytes)
    checkpoint_bytes = _read_bounded(
        inputs.checkpoint_path,
        _BOUND_MAX_BYTES,
        "frozen checkpoint",
    )
    resolved_config_bytes = _read_bounded(
        inputs.resolved_config_path,
        _CONFIG_MAX_BYTES,
        "frozen resolved config",
    )
    normalization_bytes = _read_bounded(
        inputs.normalization_path,
        _CONFIG_MAX_BYTES,
        "frozen normalization",
    )
    copied_checkpoint_path = private / "frozen-model.ckpt"
    copied_resolved_path = private / "frozen-resolved-config.json"
    copied_normalization_path = private / "frozen-normalization.json"
    _atomic_write_new(copied_checkpoint_path, checkpoint_bytes)
    _atomic_write_new(copied_resolved_path, resolved_config_bytes)
    _atomic_write_new(copied_normalization_path, normalization_bytes)
    body: dict[str, object] = {
        "artifact_type": PRIVATE_ROUTING_CONTRACT_ARTIFACT_TYPE,
        "bootstrap": {
            "challenge_seed": inputs.parent.challenge_bootstrap_seed,
            "confidence_level": inputs.parent.confidence_level,
            "replicates": inputs.parent.bootstrap_resamples,
            "zzu_seed": inputs.parent.zzu_bootstrap_seed,
        },
        "conformal": inputs.routing.conformal.to_dict(),
        "checkpoint_file_sha256": sha256_file(copied_checkpoint_path),
        "distribution_policy_artifact_sha256": inputs.v1.policy.artifact_sha256,
        "distribution_policy_file_sha256": sha256_file(copied_policy_path),
        "distribution_threshold": inputs.parent.threshold,
        "demo_policy_file_sha256": inputs.routing.demo_policy_file_sha256,
        "entropy_maximum": inputs.routing.maximum_entropy,
        "inventory_sha256": inputs.inventory.inventory_sha256,
        "label_order": list(SUPERCLASSES),
        "model_state_sha256": evaluated.model_state_before_sha256,
        "normalization_file_sha256": sha256_file(copied_normalization_path),
        "protocol_id": PROTOCOL_ID,
        "quality_config_version": DEFAULT_SIGNAL_QUALITY_CONFIG.version,
        "resolved_config_file_sha256": sha256_file(copied_resolved_path),
        "resolved_config_sha256": inputs.parent.resolved_config_sha256,
        "schema_version": 1,
        "source_calibration_artifact_sha256": (
            inputs.routing.source_calibration_result.artifact_sha256
        ),
        "source_calibration_file_sha256": (
            inputs.routing.source_calibration_file_sha256
        ),
        "temperature": inputs.routing.temperature,
        "threshold_comparison": "score_strictly_greater_than_threshold",
    }
    body["artifact_sha256"] = canonical_sha256(body)
    _atomic_write_new(
        private / "routing-contract.json",
        canonical_json_bytes(body),
    )


def _canonical_signal_array_name(kind: str, shard_index: int) -> str:
    if kind not in {
        "canonical_signal_sha256",
        "dataset",
        "inventory_index",
        "record_ref",
        "signal",
    }:
        raise OODExternalV2IntegrityError("canonical signal array kind is invalid")
    if not 0 <= shard_index < CANONICAL_SIGNAL_SHARD_COUNT:
        raise OODExternalV2IntegrityError("canonical signal shard index is invalid")
    return f"{kind}_{shard_index:05d}"


def _write_canonical_signal_bundle(
    private: Path,
    *,
    inputs: VerifiedExternalV2Inputs,
    evaluated: _EvaluatedExternalRecords,
) -> None:
    """Persist every adapter-success signal in bounded inventory-aligned chunks."""

    indices = np.asarray(evaluated.adapter_success_inventory_indices)
    signals = np.asarray(evaluated.canonical_signals)
    if (
        indices.ndim != 1
        or indices.dtype != np.dtype(np.int64)
        or signals.shape != (len(indices), len(LEADS), TARGET_SAMPLES)
        or signals.dtype != np.dtype(np.float32)
        or not np.isfinite(signals).all()
        or (len(indices) > 1 and np.any(indices[1:] <= indices[:-1]))
        or np.any(indices < 0)
        or np.any(indices >= len(evaluated.records))
    ):
        raise OODExternalV2IntegrityError("canonical signal matrix is invalid")
    expected_success = np.asarray(
        [
            index
            for index, row in enumerate(evaluated.records)
            if row.adapter_provenance_sha256 is not None
        ],
        dtype=np.int64,
    )
    if not np.array_equal(indices, expected_success):
        raise OODExternalV2IntegrityError(
            "canonical signals do not cover exactly adapter-success rows"
        )

    arrays: dict[str, NDArray[np.generic]] = {}
    descriptors: list[dict[str, object]] = []
    for shard_index in range(CANONICAL_SIGNAL_SHARD_COUNT):
        start = shard_index * CANONICAL_SIGNAL_SHARD_RECORDS
        stop = min(start + CANONICAL_SIGNAL_SHARD_RECORDS, len(evaluated.records))
        selected_positions = np.flatnonzero((indices >= start) & (indices < stop))
        shard_indices = np.ascontiguousarray(indices[selected_positions], dtype=np.int64)
        shard_signals = np.ascontiguousarray(signals[selected_positions], dtype=np.float32)
        shard_records = [inputs.inventory.records[int(index)] for index in shard_indices]
        shard_hashes = np.asarray(
            [
                cast(str, evaluated.records[int(index)].canonical_signal_sha256)
                for index in shard_indices
            ],
            dtype=np.str_,
        )
        shard_datasets = np.asarray(
            [record.dataset for record in shard_records],
            dtype=np.str_,
        )
        shard_refs = np.asarray(
            [record.record_ref for record in shard_records],
            dtype=np.str_,
        )
        chunk_arrays: dict[str, NDArray[np.generic]] = {
            _canonical_signal_array_name("canonical_signal_sha256", shard_index): (
                shard_hashes
            ),
            _canonical_signal_array_name("dataset", shard_index): shard_datasets,
            _canonical_signal_array_name("inventory_index", shard_index): shard_indices,
            _canonical_signal_array_name("record_ref", shard_index): shard_refs,
            _canonical_signal_array_name("signal", shard_index): shard_signals,
        }
        arrays.update(chunk_arrays)
        descriptors.append(
            {
                "adapter_success_count": len(shard_indices),
                "canonical_signal_sha256_tensor_sha256": _tensor_sha256(shard_hashes),
                "dataset_tensor_sha256": _tensor_sha256(shard_datasets),
                "inventory_index_tensor_sha256": _tensor_sha256(shard_indices),
                "record_ref_tensor_sha256": _tensor_sha256(shard_refs),
                "shard_index": shard_index,
                "signal_tensor_sha256": _tensor_sha256(shard_signals),
                "start_inventory_index": start,
                "stop_inventory_index_exclusive": stop,
            }
        )

    npz_path = private.parent / PurePosixPath(CANONICAL_SIGNAL_NPZ_PATH)
    _atomic_npz_new(npz_path, arrays)
    _safe_npz_members(npz_path)
    body: dict[str, object] = {
        "artifact_type": PRIVATE_CANONICAL_SIGNAL_ARTIFACT_TYPE,
        "canonical_dtype": "float32",
        "canonical_shape_per_record": [len(LEADS), TARGET_SAMPLES],
        "inventory_record_count": len(evaluated.records),
        "inventory_sha256": inputs.inventory.inventory_sha256,
        "npz_file_sha256": sha256_file(npz_path),
        "protocol_id": PROTOCOL_ID,
        "schema_version": 1,
        "shard_count": CANONICAL_SIGNAL_SHARD_COUNT,
        "shard_inventory_records": CANONICAL_SIGNAL_SHARD_RECORDS,
        "shards": descriptors,
        "successful_adapter_records": len(indices),
    }
    body["artifact_sha256"] = canonical_sha256(body)
    _atomic_write_new(
        private.parent / PurePosixPath(CANONICAL_SIGNAL_SIDECAR_PATH),
        canonical_json_bytes(body),
    )


def _write_quality_audit_shards(
    private: Path,
    *,
    inputs: VerifiedExternalV2Inputs,
    evaluated: _EvaluatedExternalRecords,
) -> None:
    if len(evaluated.records) != QUALITY_AUDIT_EXPECTED_RECORDS:
        raise OODExternalV2IntegrityError(
            "quality audit record count differs from the frozen shard layout"
        )
    audit_root = private / "quality-audit"
    audit_root.mkdir(parents=False, exist_ok=False)
    descriptors: list[dict[str, object]] = []
    for shard_index, relative_path in enumerate(QUALITY_AUDIT_SHARD_PATHS):
        start = shard_index * QUALITY_AUDIT_SHARD_RECORDS
        stop = min(start + QUALITY_AUDIT_SHARD_RECORDS, len(evaluated.records))
        selected = evaluated.records[start:stop]
        rows: list[dict[str, object]] = []
        for offset, evidence in enumerate(selected):
            report = evidence.quality_report
            if (report is None) is not (evidence.quality_report_sha256 is None):
                raise OODExternalV2IntegrityError(
                    "quality report body/hash presence differs"
                )
            if report is not None and (
                _quality_report_sha256(report) != evidence.quality_report_sha256
            ):
                raise OODExternalV2IntegrityError(
                    "quality report hash differs before sharding"
                )
            rows.append(
                {
                    "canonical_signal_sha256": evidence.canonical_signal_sha256,
                    "dataset": evidence.dataset,
                    "inventory_index": start + offset,
                    "quality_report": report,
                    "quality_report_sha256": evidence.quality_report_sha256,
                    "record_ref": evidence.record_ref,
                }
            )
        body: dict[str, object] = {
            "artifact_type": PRIVATE_QUALITY_AUDIT_ARTIFACT_TYPE,
            "inventory_sha256": inputs.inventory.inventory_sha256,
            "protocol_id": PROTOCOL_ID,
            "record_count": len(rows),
            "records": rows,
            "schema_version": 1,
            "shard_index": shard_index,
            "start_inventory_index": start,
            "stop_inventory_index_exclusive": stop,
        }
        body["artifact_sha256"] = canonical_sha256(body)
        payload = canonical_json_bytes(body)
        if len(payload) > QUALITY_AUDIT_SHARD_MAX_BYTES:
            raise OODExternalV2IntegrityError(
                "quality audit shard exceeds its frozen byte limit"
            )
        path = private.parent / PurePosixPath(relative_path)
        _atomic_write_new(path, payload)
        descriptors.append(
            {
                "artifact_sha256": body["artifact_sha256"],
                "file_sha256": sha256_bytes(payload),
                "record_count": len(rows),
                "relative_path": relative_path,
                "size_bytes": len(payload),
                "start_inventory_index": start,
                "stop_inventory_index_exclusive": stop,
            }
        )
    index_body: dict[str, object] = {
        "artifact_type": PRIVATE_QUALITY_AUDIT_INDEX_ARTIFACT_TYPE,
        "inventory_sha256": inputs.inventory.inventory_sha256,
        "protocol_id": PROTOCOL_ID,
        "record_count": len(evaluated.records),
        "schema_version": 1,
        "shard_count": len(descriptors),
        "shard_max_bytes": QUALITY_AUDIT_SHARD_MAX_BYTES,
        "shard_records": QUALITY_AUDIT_SHARD_RECORDS,
        "shards": descriptors,
    }
    index_body["artifact_sha256"] = canonical_sha256(index_body)
    index_path = private.parent / PurePosixPath(QUALITY_AUDIT_INDEX_PATH)
    _atomic_write_new(index_path, canonical_json_bytes(index_body))


def _write_embedding_bundle(
    private: Path,
    *,
    inputs: VerifiedExternalV2Inputs,
    evaluated: _EvaluatedExternalRecords,
) -> None:
    selected = [
        inputs.inventory.records[int(index)]
        for index in evaluated.quality_pass_inventory_indices.tolist()
    ]
    arrays: dict[str, NDArray[np.generic]] = {
        "dataset": np.asarray([record.dataset for record in selected], dtype=np.str_),
        "embedding_first": evaluated.embeddings,
        "embedding_repeated": evaluated.repeated_embeddings,
        "logits_first": evaluated.logits,
        "logits_repeated": evaluated.repeated_logits,
        "patient_key": np.asarray(
            ["" if record.patient_key is None else record.patient_key for record in selected],
            dtype=np.str_,
        ),
        "record_ref": np.asarray([record.record_ref for record in selected], dtype=np.str_),
        "probabilities": evaluated.probabilities,
        "score": evaluated.scores,
    }
    npz_path = private / "quality-pass-embeddings.npz"
    _atomic_npz_new(npz_path, arrays)
    _verify_embedding_npz(npz_path, expected_records=len(selected))
    body: dict[str, object] = {
        "artifact_type": PRIVATE_EMBEDDING_ARTIFACT_TYPE,
        "embedding_dimension": 512,
        "embedding_dtype": "float32",
        "embedding_tensor_sha256": evaluated.embedding_sha256,
        "first_logits_tensor_sha256": evaluated.first_logits_sha256,
        "inventory_sha256": inputs.inventory.inventory_sha256,
        "logits_dtype": "float64",
        "model_state_after_sha256": evaluated.model_state_after_sha256,
        "model_state_before_sha256": evaluated.model_state_before_sha256,
        "model_state_unchanged": evaluated.model_state_unchanged,
        "npz_file_sha256": sha256_file(npz_path),
        "probabilities_dtype": "float64",
        "probabilities_tensor_sha256": evaluated.probabilities_sha256,
        "protocol_id": PROTOCOL_ID,
        "quality_pass_records": len(selected),
        "repeated_embedding_tensor_sha256": evaluated.repeated_embedding_sha256,
        "repeated_logits_tensor_sha256": evaluated.repeated_logits_sha256,
        "repeat_verified": (
            evaluated.embedding_sha256 == evaluated.repeated_embedding_sha256
        ),
        "schema_version": 1,
        "score_dtype": "float64",
        "score_tensor_sha256": _tensor_sha256(evaluated.scores),
    }
    body["artifact_sha256"] = canonical_sha256(body)
    _atomic_write_new(
        private / "quality-pass-embeddings.json",
        canonical_json_bytes(body),
    )


def _write_bootstrap_bundle(
    private: Path,
    *,
    inputs: VerifiedExternalV2Inputs,
    endpoints: _EndpointEvidence,
) -> None:
    arrays = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in sorted(endpoints.bootstrap_replicates.items())
    }
    npz_path = private / "bootstrap-replicates.npz"
    _atomic_npz_new(npz_path, arrays)
    _verify_bootstrap_npz(
        npz_path,
        expected_names=tuple(arrays),
        expected_replicates=inputs.parent.bootstrap_resamples,
    )
    body: dict[str, object] = {
        "artifact_type": PRIVATE_BOOTSTRAP_ARTIFACT_TYPE,
        "endpoint_names": list(arrays),
        "npz_file_sha256": sha256_file(npz_path),
        "protocol_id": PROTOCOL_ID,
        "quantile_method": "linear",
        "replicates_per_endpoint": inputs.parent.bootstrap_resamples,
        "schema_version": 1,
    }
    body["artifact_sha256"] = canonical_sha256(body)
    _atomic_write_new(
        private / "bootstrap-replicates.json",
        canonical_json_bytes(body),
    )


def assert_external_v2_parent_executable(
    parent: OODExternalV2ParentConfig,
) -> None:
    """Refuse the preserved v2 parent before output or claim creation."""

    if parent.file_sha256 == EXPECTED_PARENT_CONFIG_SHA256:
        raise OODExternalV2ExecutionError(
            "PRE_INFERENCE_PROTOCOL_INFEASIBLE: "
            f"{FROZEN_V2_PREINFERENCE_INFEASIBILITY}"
        )


def verify_external_v2_metadata(
    *,
    parent_path: str | Path,
    child_path: str | Path | None,
    project_root: str | Path,
    code_revision: str | None = None,
    seven_zip_executable: str | Path = "7z",
) -> VerifiedExternalV2Inputs | OODExternalV2ParentConfig | SuccessorParentPreflight:
    """Inspect frozen metadata without creating an output, marker, or claim."""

    requested = Path(os.path.abspath(os.fspath(parent_path)))
    root = _strict_project_root(project_root)
    successor_path = root.joinpath(*PurePosixPath(SUCCESSOR_PARENT_CONFIG_PATH).parts)
    if requested == successor_path and child_path is None:
        successor = verify_successor_parent_preflight(
            requested,
            project_root=root,
        )
        return successor
    parent = _load_parent_for_operation(parent_path, project_root=root)
    if child_path is None:
        return parent
    if code_revision is None:
        raise OODExternalV2ConfigError(
            "code_revision is required when verifying a child contract"
        )
    child = load_child_contract(child_path)
    return verify_external_v2_inputs(
        parent,
        child,
        project_root=project_root,
        code_revision=code_revision,
        seven_zip_executable=seven_zip_executable,
    )


def verify_inventory_builder_preflight(
    parent_path: str | Path,
    project_root: str | Path,
    implementation_revision: str,
) -> InventoryBuilderPreflight:
    """Prove the frozen metadata-only builder boundary before raw-byte access."""

    try:
        root = _strict_project_root(project_root)
        revision = _revision(
            implementation_revision,
            "inventory implementation revision",
        )
        _verify_successor_amendment_revision(
            root,
            implementation_revision=revision,
        )
        expected_parent = root.joinpath(
            *PurePosixPath(SUCCESSOR_PARENT_CONFIG_PATH).parts
        )
        parent = _load_parent_for_operation(parent_path, project_root=root)
        assert_external_v2_parent_executable(parent)
        if (
            parent.path != expected_parent
            or parent.status != "frozen_parent_preregistration_pre_waveform"
            or parent.file_sha256 != EXPECTED_SUCCESSOR_PARENT_CONFIG_SHA256
        ):
            raise OODExternalV2IntegrityError(
                "inventory builder requires the exact frozen successor parent"
            )
    except Exception as error:
        raise InventoryBuilderPreflightStageError("parent_lineage") from error

    try:
        # This includes the complete CPython/site/Git/NVIDIA trees, isolated
        # launcher state, __main__, Python modules, and loaded native images.
        runtime = _current_runtime_environment()
        source_tree = _build_project_source_tree(root)
    except Exception as error:
        raise InventoryBuilderPreflightStageError("runtime_environment") from error

    try:
        if _verify_clean_git_revision(root) != revision:
            raise OODExternalV2IntegrityError(
                "inventory builder HEAD differs from the implementation revision"
            )
        _verify_git_remote_state(root, expected_revision=revision)
        _verify_private_history_absent(root)
        _verify_tracked_head_blob(
            root,
            revision=revision,
            relative_path=SUCCESSOR_PARENT_CONFIG_PATH,
            expected_file_sha256=parent.file_sha256,
        )
        _verify_project_source_tree_at_revisions(
            root,
            source_tree,
            implementation_revision=revision,
            execution_revision=None,
        )
        _verify_imported_project_module_origins(root, source_tree)
    except Exception as error:
        raise InventoryBuilderPreflightStageError(
            "git_source_provenance"
        ) from error

    try:
        output_root = _resolve_project_relative(
            root,
            parent.output_root,
            require_file=False,
        )
        claim_path = _resolve_project_relative(
            root,
            parent.claim_path,
            require_file=False,
        )
        for candidate, context in (
            (output_root, "successor output root"),
            (claim_path, "successor one-shot claim"),
        ):
            if candidate.exists() or _is_indirect(candidate):
                raise OODExternalV2IntegrityError(
                    f"{context} must be absent before inventory construction"
                )
            _assert_direct_ancestry(
                candidate.parent,
                context=f"{context} parent",
            )
        _assert_no_marked_staging_retry(output_root)
    except Exception as error:
        raise InventoryBuilderPreflightStageError("namespace_state") from error

    try:
        # Close the long runtime/Git probe against a late source or index change.
        if (
            _verify_clean_git_revision(root) != revision
            or _build_project_source_tree(root) != source_tree
        ):
            raise OODExternalV2IntegrityError(
                "inventory builder controls changed during preflight"
            )
        if (
            parent.raw_source_bindings is None
            or parent.seven_zip_tool_binding is None
            or parent.inventory_counts is None
        ):
            raise OODExternalV2ConfigError(
                "inventory builder parent lacks executable source/count bindings"
            )
    except Exception as error:
        raise InventoryBuilderPreflightStageError(
            "closing_control_state"
        ) from error
    return InventoryBuilderPreflight(
        status="INVENTORY_BUILDER_PREFLIGHT_VERIFIED",
        parent_config_file_sha256=parent.file_sha256,
        implementation_revision=revision,
        project_source_tree_sha256=source_tree.tree_sha256,
        python_environment_sha256=runtime.python_environment_sha256,
        git_runtime_tree_sha256=runtime.git_tool.runtime_tree.tree_sha256,
        raw_source_bindings=parent.raw_source_bindings,
        seven_zip_tool_binding=parent.seven_zip_tool_binding,
        inventory_counts=parent.inventory_counts,
    )


def verify_inventory_builder_input_contract(
    preflight: InventoryBuilderPreflight,
    *,
    project_root: str | Path,
    dataset_roots: Mapping[str, Path],
    raw_source_paths: Mapping[str, Path],
    seven_zip_executable: str | Path,
) -> None:
    """Bind every production path and the tool before official source-byte access."""

    if not isinstance(preflight, InventoryBuilderPreflight):
        raise TypeError("preflight must be InventoryBuilderPreflight")
    root = _strict_project_root(project_root)
    if set(dataset_roots) != set(EXPECTED_DATASET_ROOTS):
        raise OODExternalV2IntegrityError("inventory dataset root set differs")
    for dataset, expected_relative in EXPECTED_DATASET_ROOTS.items():
        requested = Path(os.path.abspath(os.fspath(dataset_roots[dataset])))
        expected = root.joinpath(*PurePosixPath(expected_relative).parts)
        if (
            requested != expected
            or _assert_direct_ancestry(
                requested,
                context=f"inventory dataset root {dataset}",
            )
            != requested
            or not requested.is_dir()
        ):
            raise OODExternalV2IntegrityError(
                "inventory dataset root differs from the frozen path"
            )
    if set(raw_source_paths) != set(INVENTORY_BUILDER_RAW_SOURCE_KEYS):
        raise OODExternalV2IntegrityError("inventory CLI raw-source set differs")
    for name in INVENTORY_BUILDER_RAW_SOURCE_KEYS:
        binding = preflight.raw_source_bindings[name]
        requested = Path(os.path.abspath(os.fspath(raw_source_paths[name])))
        expected = root.joinpath(*PurePosixPath(binding.relative_path).parts)
        if (
            requested != expected
            or _assert_direct_ancestry(
                requested,
                context=f"inventory raw source {name}",
            )
            != requested
            or not requested.is_file()
        ):
            raise OODExternalV2IntegrityError(
                "inventory raw-source path differs from the frozen path"
            )
    for name, binding in preflight.raw_source_bindings.items():
        expected = root.joinpath(*PurePosixPath(binding.relative_path).parts)
        if (
            binding.relative_path != EXPECTED_RAW_SOURCE_PATHS[name]
            or _assert_direct_ancestry(
                expected,
                context=f"inventory frozen raw source {name}",
            )
            != expected
            or not expected.is_file()
        ):
            raise OODExternalV2IntegrityError(
                "inventory frozen raw-source path is unavailable"
            )
    verify_seven_zip_tool_binding(
        Path(os.path.abspath(os.fspath(seven_zip_executable))),
        preflight.seven_zip_tool_binding,
    )


def _inventory_builder_attempt_body(
    preflight: InventoryBuilderPreflight,
) -> dict[str, object]:
    if not isinstance(preflight, InventoryBuilderPreflight):
        raise TypeError("preflight must be InventoryBuilderPreflight")
    body: dict[str, object] = {
        "archive_operand_normalization": {
            "applies_to_commands": [
                "listing",
                "archive_test",
                "isolated_extraction",
            ],
            "input": "already_bound_project_relative_ZZU_terminal_zip_path",
            "output": "exact_absolute_direct_archive_path",
            "scientific_protocol_change": False,
        },
        "artifact_type": SUCCESSOR_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_TYPE,
        "authorization_consumption_ordinal": 1,
        "authorization_id": "x8_inventory_build_attempt_1",
        "contains_external_source_bytes_or_identifiers": False,
        "contains_model_outputs_embeddings_or_scores": False,
        "external_one_shot_claim_consumed_at_marker_creation": False,
        "fresh_frozen_amendment_authorization": True,
        "git_runtime_tree_sha256": preflight.git_runtime_tree_sha256,
        "historical_x6_authorization_artifact_sha256": (
            HISTORICAL_X6_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_SHA256
        ),
        "historical_x6_authorization_file_sha256": (
            HISTORICAL_X6_INVENTORY_BUILDER_ATTEMPT_FILE_SHA256
        ),
        "historical_x6_authorization_path": (
            HISTORICAL_X6_INVENTORY_BUILDER_ATTEMPT_PATH
        ),
        "historical_x6_authorization_state": "CONSUMED_FAILED_RETAINED",
        "implementation_revision": preflight.implementation_revision,
        "maximum_consumptions": 1,
        "parent_config_file_sha256": preflight.parent_config_file_sha256,
        "predecessor_authorization_artifact_sha256": (
            HISTORICAL_X7_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_SHA256
        ),
        "predecessor_authorization_consumed": True,
        "predecessor_authorization_file_sha256": (
            HISTORICAL_X7_INVENTORY_BUILDER_ATTEMPT_FILE_SHA256
        ),
        "predecessor_authorization_id": "x7_inventory_build_attempt_1",
        "predecessor_authorization_must_remain_present": True,
        "predecessor_authorization_path": (
            HISTORICAL_X7_INVENTORY_BUILDER_ATTEMPT_PATH
        ),
        "predecessor_authorization_state": "CONSUMED_FAILED_RETAINED",
        "predecessor_failure_output_state": "NONE",
        "predecessor_failure_receipt_artifact_sha256": (
            HISTORICAL_X7_INVENTORY_BUILDER_FAILURE_ARTIFACT_SHA256
        ),
        "predecessor_failure_receipt_file_sha256": (
            HISTORICAL_X7_INVENTORY_BUILDER_FAILURE_FILE_SHA256
        ),
        "predecessor_failure_receipt_must_remain_present": True,
        "predecessor_failure_receipt_path": (
            HISTORICAL_X7_INVENTORY_BUILDER_FAILURE_PATH
        ),
        "predecessor_failure_stage": "zzu_archive_listing",
        "predecessor_failure_stage_ordinal": 8,
        "predecessor_official_source_content_accessed": True,
        "project_source_tree_sha256": preflight.project_source_tree_sha256,
        "protocol_inventory_build_attempt_ordinal": 3,
        "protocol_id": PROTOCOL_ID,
        "python_environment_sha256": preflight.python_environment_sha256,
        "retry_resume_or_reuse_of_predecessor": False,
        "schema_version": 4,
        "state": "PRECLAIM_INVENTORY_BUILD_AUTHORIZATION_CONSUMED",
    }
    body["artifact_sha256"] = canonical_sha256(body)
    return body


def _inventory_builder_attempt_bytes(preflight: InventoryBuilderPreflight) -> bytes:
    return canonical_json_bytes(_inventory_builder_attempt_body(preflight))


def _historical_x6_inventory_builder_attempt_bytes() -> bytes:
    body: dict[str, object] = {
        "artifact_sha256": (
            HISTORICAL_X6_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_SHA256
        ),
        "artifact_type": SUCCESSOR_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_TYPE,
        "authorization_id": "x6_inventory_build_attempt_1",
        "consumption_ordinal": 1,
        "contains_external_source_bytes_or_identifiers": False,
        "contains_model_outputs_embeddings_or_scores": False,
        "external_one_shot_claim_consumed_at_marker_creation": False,
        "git_runtime_tree_sha256": (
            "sha256:086bd1898a3859d59d4c7184f1039d73cdf75c07de76f70fc375495ed922d9e2"
        ),
        "implementation_revision": SIXTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        "maximum_consumptions": 1,
        "parent_config_file_sha256": SIXTH_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256,
        "project_source_tree_sha256": (
            "sha256:b6fabbcc99028eb7b666c9bb4523706d679f118ca44a688903316c16d36d6881"
        ),
        "protocol_id": PROTOCOL_ID,
        "python_environment_sha256": (
            "sha256:d834e2cf3e6cf1ec7fbf09607cb6fb8b5a05824dfdfba15445e2e5dad74c9188"
        ),
        "schema_version": 2,
        "state": "PRECLAIM_INVENTORY_BUILD_AUTHORIZATION_CONSUMED",
        "superseded_authorization_consumed": False,
        "superseded_authorization_id": "x5_inventory_build_attempt_1",
        "superseded_authorization_path": (
            HISTORICAL_X5_INVENTORY_BUILDER_ATTEMPT_PATH
        ),
        "superseded_authorization_state": "RETIRED_UNCONSUMED",
    }
    payload = canonical_json_bytes(body)
    if sha256_bytes(payload) != HISTORICAL_X6_INVENTORY_BUILDER_ATTEMPT_FILE_SHA256:
        raise OODExternalV2IntegrityError(
            "frozen X6 inventory builder attempt constants are inconsistent"
        )
    return payload


def _verify_historical_x6_inventory_builder_attempt(project_root: Path) -> str:
    marker = _resolve_project_relative(
        project_root,
        HISTORICAL_X6_INVENTORY_BUILDER_ATTEMPT_PATH,
        require_file=True,
    )
    observed = _read_bounded(
        marker,
        _CHILD_MAX_BYTES,
        "historical X6 inventory builder attempt marker",
    )
    if observed != _historical_x6_inventory_builder_attempt_bytes():
        raise OODExternalV2IntegrityError(
            "historical X6 inventory builder attempt marker differs"
        )
    _require_git_ignored_and_untracked(
        project_root,
        HISTORICAL_X6_INVENTORY_BUILDER_ATTEMPT_PATH,
        context="historical X6 inventory builder attempt marker",
    )
    return sha256_bytes(observed)


def _historical_x7_inventory_builder_attempt_bytes() -> bytes:
    body: dict[str, object] = {
        "artifact_type": SUCCESSOR_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_TYPE,
        "authorization_consumption_ordinal": 1,
        "authorization_id": "x7_inventory_build_attempt_1",
        "contains_external_source_bytes_or_identifiers": False,
        "contains_model_outputs_embeddings_or_scores": False,
        "external_one_shot_claim_consumed_at_marker_creation": False,
        "fresh_frozen_amendment_authorization": True,
        "git_runtime_tree_sha256": (
            "sha256:086bd1898a3859d59d4c7184f1039d73cdf75c07de76f70fc375495ed922d9e2"
        ),
        "implementation_revision": SEVENTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        "maximum_consumptions": 1,
        "parent_config_file_sha256": SEVENTH_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256,
        "predecessor_authorization_artifact_sha256": (
            HISTORICAL_X6_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_SHA256
        ),
        "predecessor_authorization_consumed": True,
        "predecessor_authorization_file_sha256": (
            HISTORICAL_X6_INVENTORY_BUILDER_ATTEMPT_FILE_SHA256
        ),
        "predecessor_authorization_id": "x6_inventory_build_attempt_1",
        "predecessor_authorization_must_remain_present": True,
        "predecessor_authorization_path": (
            HISTORICAL_X6_INVENTORY_BUILDER_ATTEMPT_PATH
        ),
        "predecessor_authorization_state": "CONSUMED_FAILED_RETAINED",
        "project_source_tree_sha256": (
            "sha256:5d3127c32134f459a6bac8035fade4200bc2922422ac3aa711178de7d78edea8"
        ),
        "protocol_inventory_build_attempt_ordinal": 2,
        "protocol_id": PROTOCOL_ID,
        "python_environment_sha256": (
            "sha256:d834e2cf3e6cf1ec7fbf09607cb6fb8b5a05824dfdfba15445e2e5dad74c9188"
        ),
        "retry_resume_or_reuse_of_predecessor": False,
        "schema_version": 3,
        "state": "PRECLAIM_INVENTORY_BUILD_AUTHORIZATION_CONSUMED",
    }
    if canonical_sha256(body) != HISTORICAL_X7_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_SHA256:
        raise OODExternalV2IntegrityError(
            "frozen X7 inventory builder attempt artifact constants are inconsistent"
        )
    body["artifact_sha256"] = HISTORICAL_X7_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_SHA256
    payload = canonical_json_bytes(body)
    if sha256_bytes(payload) != HISTORICAL_X7_INVENTORY_BUILDER_ATTEMPT_FILE_SHA256:
        raise OODExternalV2IntegrityError(
            "frozen X7 inventory builder attempt file constants are inconsistent"
        )
    return payload


def _historical_x7_inventory_builder_failure_bytes() -> bytes:
    body: dict[str, object] = {
        "artifact_type": SUCCESSOR_INVENTORY_BUILDER_FAILURE_ARTIFACT_TYPE,
        "authorization_consumed": True,
        "authorization_id": "x7_inventory_build_attempt_1",
        "authorization_marker_artifact_sha256": (
            HISTORICAL_X7_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_SHA256
        ),
        "authorization_marker_file_sha256": (
            HISTORICAL_X7_INVENTORY_BUILDER_ATTEMPT_FILE_SHA256
        ),
        "contains_external_source_bytes_or_identifiers": False,
        "contains_model_outputs_embeddings_or_scores": False,
        "external_one_shot_claim_consumed": False,
        "failure_requires_new_frozen_amendment_and_authorization_id": True,
        "failure_stage": "zzu_archive_listing",
        "failure_stage_ordinal": 8,
        "implementation_revision": SEVENTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        "official_source_content_accessed": True,
        "output_state": "NONE",
        "parent_config_file_sha256": SEVENTH_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256,
        "protocol_id": PROTOCOL_ID,
        "quality_model_score_logit_probability_or_metric_observed": False,
        "retry_resume_or_reuse_authorized": False,
        "schema_version": 1,
        "state": "PRECLAIM_INVENTORY_BUILD_FAILED",
        "waveform_sample_decode_occurred": False,
    }
    if canonical_sha256(body) != HISTORICAL_X7_INVENTORY_BUILDER_FAILURE_ARTIFACT_SHA256:
        raise OODExternalV2IntegrityError(
            "frozen X7 inventory builder failure artifact constants are inconsistent"
        )
    body["artifact_sha256"] = HISTORICAL_X7_INVENTORY_BUILDER_FAILURE_ARTIFACT_SHA256
    payload = canonical_json_bytes(body)
    if sha256_bytes(payload) != HISTORICAL_X7_INVENTORY_BUILDER_FAILURE_FILE_SHA256:
        raise OODExternalV2IntegrityError(
            "frozen X7 inventory builder failure file constants are inconsistent"
        )
    return payload


def _verify_historical_x7_inventory_builder_artifacts(
    project_root: Path,
) -> tuple[str, str]:
    expected = (
        (
            HISTORICAL_X7_INVENTORY_BUILDER_ATTEMPT_PATH,
            _historical_x7_inventory_builder_attempt_bytes(),
            "historical X7 inventory builder attempt marker",
        ),
        (
            HISTORICAL_X7_INVENTORY_BUILDER_FAILURE_PATH,
            _historical_x7_inventory_builder_failure_bytes(),
            "historical X7 inventory builder failure receipt",
        ),
    )
    hashes: list[str] = []
    for relative_path, expected_bytes, context in expected:
        artifact = _resolve_project_relative(
            project_root,
            relative_path,
            require_file=True,
        )
        observed = _read_bounded(artifact, _CHILD_MAX_BYTES, context)
        if observed != expected_bytes:
            raise OODExternalV2IntegrityError(f"{context} differs")
        _require_git_ignored_and_untracked(
            project_root,
            relative_path,
            context=context,
        )
        hashes.append(sha256_bytes(observed))
    return (hashes[0], hashes[1])


def _historical_x8_inventory_builder_attempt_bytes() -> bytes:
    """Reconstruct the successful X8 marker without using X9 runtime identity."""

    body: dict[str, object] = {
        "archive_operand_normalization": {
            "applies_to_commands": [
                "listing",
                "archive_test",
                "isolated_extraction",
            ],
            "input": "already_bound_project_relative_ZZU_terminal_zip_path",
            "output": "exact_absolute_direct_archive_path",
            "scientific_protocol_change": False,
        },
        "artifact_type": SUCCESSOR_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_TYPE,
        "authorization_consumption_ordinal": 1,
        "authorization_id": "x8_inventory_build_attempt_1",
        "contains_external_source_bytes_or_identifiers": False,
        "contains_model_outputs_embeddings_or_scores": False,
        "external_one_shot_claim_consumed_at_marker_creation": False,
        "fresh_frozen_amendment_authorization": True,
        "git_runtime_tree_sha256": EXPECTED_GIT_RUNTIME_TREE_SHA256,
        "historical_x6_authorization_artifact_sha256": (
            HISTORICAL_X6_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_SHA256
        ),
        "historical_x6_authorization_file_sha256": (
            HISTORICAL_X6_INVENTORY_BUILDER_ATTEMPT_FILE_SHA256
        ),
        "historical_x6_authorization_path": (
            HISTORICAL_X6_INVENTORY_BUILDER_ATTEMPT_PATH
        ),
        "historical_x6_authorization_state": "CONSUMED_FAILED_RETAINED",
        "implementation_revision": EIGHTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        "maximum_consumptions": 1,
        "parent_config_file_sha256": EIGHTH_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256,
        "predecessor_authorization_artifact_sha256": (
            HISTORICAL_X7_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_SHA256
        ),
        "predecessor_authorization_consumed": True,
        "predecessor_authorization_file_sha256": (
            HISTORICAL_X7_INVENTORY_BUILDER_ATTEMPT_FILE_SHA256
        ),
        "predecessor_authorization_id": "x7_inventory_build_attempt_1",
        "predecessor_authorization_must_remain_present": True,
        "predecessor_authorization_path": (
            HISTORICAL_X7_INVENTORY_BUILDER_ATTEMPT_PATH
        ),
        "predecessor_authorization_state": "CONSUMED_FAILED_RETAINED",
        "predecessor_failure_output_state": "NONE",
        "predecessor_failure_receipt_artifact_sha256": (
            HISTORICAL_X7_INVENTORY_BUILDER_FAILURE_ARTIFACT_SHA256
        ),
        "predecessor_failure_receipt_file_sha256": (
            HISTORICAL_X7_INVENTORY_BUILDER_FAILURE_FILE_SHA256
        ),
        "predecessor_failure_receipt_must_remain_present": True,
        "predecessor_failure_receipt_path": (
            HISTORICAL_X7_INVENTORY_BUILDER_FAILURE_PATH
        ),
        "predecessor_failure_stage": "zzu_archive_listing",
        "predecessor_failure_stage_ordinal": 8,
        "predecessor_official_source_content_accessed": True,
        "project_source_tree_sha256": (
            HISTORICAL_X8_INVENTORY_BUILDER_PROJECT_SOURCE_TREE_SHA256
        ),
        "protocol_inventory_build_attempt_ordinal": 3,
        "protocol_id": PROTOCOL_ID,
        "python_environment_sha256": (
            "sha256:d834e2cf3e6cf1ec7fbf09607cb6fb8b5a05824dfdfba15445e2e5dad74c9188"
        ),
        "retry_resume_or_reuse_of_predecessor": False,
        "schema_version": 4,
        "state": "PRECLAIM_INVENTORY_BUILD_AUTHORIZATION_CONSUMED",
    }
    if canonical_sha256(body) != HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_SHA256:
        raise OODExternalV2IntegrityError(
            "frozen X8 inventory builder attempt artifact constants are inconsistent"
        )
    body["artifact_sha256"] = HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_SHA256
    payload = canonical_json_bytes(body)
    if sha256_bytes(payload) != HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_FILE_SHA256:
        raise OODExternalV2IntegrityError(
            "frozen X8 inventory builder attempt file constants are inconsistent"
        )
    return payload


def _verify_historical_x8_inventory_builder_evidence(project_root: Path) -> str:
    """Verify the complete retained inventory-attempt lineage through X8."""

    for relative_path, label in (
        (HISTORICAL_X4_INVENTORY_BUILDER_ATTEMPT_PATH, "X4"),
        (HISTORICAL_X5_INVENTORY_BUILDER_ATTEMPT_PATH, "X5"),
    ):
        retired = _resolve_project_relative(
            project_root,
            relative_path,
            require_file=False,
        )
        if retired.exists() or _is_indirect(retired):
            raise OODExternalV2IntegrityError(
                f"retired {label} inventory builder authorization path must remain absent"
            )
        _require_git_ignored_and_untracked(
            project_root,
            relative_path,
            context=f"retired {label} inventory builder authorization",
        )
    _verify_historical_x6_inventory_builder_attempt(project_root)
    _verify_historical_x7_inventory_builder_artifacts(project_root)
    marker = _resolve_project_relative(
        project_root,
        HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_PATH,
        require_file=True,
    )
    observed = _read_bounded(
        marker,
        _CHILD_MAX_BYTES,
        "historical X8 inventory builder attempt marker",
    )
    if observed != _historical_x8_inventory_builder_attempt_bytes():
        raise OODExternalV2IntegrityError(
            "historical X8 inventory builder attempt marker differs"
        )
    _require_git_ignored_and_untracked(
        project_root,
        HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_PATH,
        context="historical X8 inventory builder attempt marker",
    )
    failure = _resolve_project_relative(
        project_root,
        HISTORICAL_X8_INVENTORY_BUILDER_FAILURE_PATH,
        require_file=False,
    )
    if failure.exists() or _is_indirect(failure):
        raise OODExternalV2IntegrityError(
            "X8 inventory builder failure receipt must remain absent"
        )
    _require_git_ignored_and_untracked(
        project_root,
        HISTORICAL_X8_INVENTORY_BUILDER_FAILURE_PATH,
        context="absent X8 inventory builder failure receipt",
    )
    return sha256_bytes(observed)


def _historical_x9_child_freeze_attempt_bytes() -> bytes:
    body: dict[str, object] = {
        "artifact_type": SUCCESSOR_CHILD_FREEZE_ATTEMPT_ARTIFACT_TYPE,
        "authorization_consumption_ordinal": 1,
        "authorization_id": "x9_child_freeze_attempt_1",
        "child_contract_relative_path": SUCCESSOR_CHILD_CONFIG_PATH,
        "child_frozen_at_utc": HISTORICAL_X9_CHILD_FROZEN_AT_UTC,
        "contains_external_source_bytes_or_identifiers": False,
        "contains_model_outputs_embeddings_or_scores": False,
        "declared_counts": {
            "challenge_records": 1_000,
            "selected_records_total": 13_328,
            "zzu_patients": 10_350,
            "zzu_records": 12_328,
        },
        "external_one_shot_claim_consumed_at_marker_creation": False,
        "git_runtime_tree_sha256": EXPECTED_GIT_RUNTIME_TREE_SHA256,
        "historical_x8_failure_receipt_required_absent": True,
        "historical_x8_inventory_builder_attempt_artifact_sha256": (
            HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_SHA256
        ),
        "historical_x8_inventory_builder_attempt_file_sha256": (
            HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_FILE_SHA256
        ),
        "historical_x8_inventory_builder_attempt_path": (
            HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_PATH
        ),
        "historical_x8_private_inventory_artifact_sha256": (
            HISTORICAL_X8_PRIVATE_INVENTORY_ARTIFACT_SHA256
        ),
        "historical_x8_private_inventory_file_sha256": (
            HISTORICAL_X8_PRIVATE_INVENTORY_FILE_SHA256
        ),
        "historical_x8_public_projection_artifact_sha256": (
            HISTORICAL_X8_PUBLIC_PROJECTION_ARTIFACT_SHA256
        ),
        "historical_x8_public_projection_file_sha256": (
            HISTORICAL_X8_PUBLIC_PROJECTION_FILE_SHA256
        ),
        "implementation_revision": NINTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        "inventory_builder_authorization_reused": False,
        "maximum_consumptions": 1,
        "parent_config_file_sha256": NINTH_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256,
        "project_source_tree_sha256": (
            HISTORICAL_X9_CHILD_FREEZE_PROJECT_SOURCE_TREE_SHA256
        ),
        "protocol_child_freeze_attempt_ordinal": 1,
        "protocol_id": PROTOCOL_ID,
        "python_environment_sha256": HISTORICAL_X9_PYTHON_ENVIRONMENT_SHA256,
        "schema_version": 1,
        "state": "PRECLAIM_CHILD_FREEZE_AUTHORIZATION_CONSUMED",
    }
    if canonical_sha256(body) != HISTORICAL_X9_CHILD_FREEZE_ATTEMPT_ARTIFACT_SHA256:
        raise OODExternalV2IntegrityError(
            "frozen X9 child freeze attempt artifact constants are inconsistent"
        )
    body["artifact_sha256"] = HISTORICAL_X9_CHILD_FREEZE_ATTEMPT_ARTIFACT_SHA256
    payload = canonical_json_bytes(body)
    if sha256_bytes(payload) != HISTORICAL_X9_CHILD_FREEZE_ATTEMPT_FILE_SHA256:
        raise OODExternalV2IntegrityError(
            "frozen X9 child freeze attempt file constants are inconsistent"
        )
    return payload


def _historical_x9_child_freeze_failure_bytes() -> bytes:
    body: dict[str, object] = {
        "artifact_type": SUCCESSOR_CHILD_FREEZE_FAILURE_ARTIFACT_TYPE,
        "authorization_consumed": True,
        "authorization_id": "x9_child_freeze_attempt_1",
        "authorization_marker_artifact_sha256": (
            HISTORICAL_X9_CHILD_FREEZE_ATTEMPT_ARTIFACT_SHA256
        ),
        "authorization_marker_file_sha256": (
            HISTORICAL_X9_CHILD_FREEZE_ATTEMPT_FILE_SHA256
        ),
        "contains_external_source_bytes_or_identifiers": False,
        "contains_model_outputs_embeddings_or_scores": False,
        "external_one_shot_claim_consumed": False,
        "failure_reason": "STAGE_REFUSED",
        "failure_requires_new_frozen_amendment_and_authorization_id": True,
        "failure_stage": "decision_and_child_materialization",
        "failure_stage_ordinal": 9,
        "implementation_revision": NINTH_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        "official_source_content_accessed": True,
        "output_state": "NONE",
        "parent_config_file_sha256": NINTH_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256,
        "protocol_id": PROTOCOL_ID,
        "quality_model_score_logit_probability_or_metric_observed": False,
        "retry_resume_or_reuse_authorized": False,
        "schema_version": 1,
        "state": "PRECLAIM_CHILD_FREEZE_FAILED",
        "waveform_sample_decode_occurred": False,
    }
    if canonical_sha256(body) != HISTORICAL_X9_CHILD_FREEZE_FAILURE_ARTIFACT_SHA256:
        raise OODExternalV2IntegrityError(
            "frozen X9 child freeze failure artifact constants are inconsistent"
        )
    body["artifact_sha256"] = HISTORICAL_X9_CHILD_FREEZE_FAILURE_ARTIFACT_SHA256
    payload = canonical_json_bytes(body)
    if sha256_bytes(payload) != HISTORICAL_X9_CHILD_FREEZE_FAILURE_FILE_SHA256:
        raise OODExternalV2IntegrityError(
            "frozen X9 child freeze failure file constants are inconsistent"
        )
    return payload


def _verify_historical_x9_child_freeze_artifacts(project_root: Path) -> None:
    _verify_historical_x8_inventory_builder_evidence(project_root)
    for relative_path, expected, context in (
        (
            HISTORICAL_X9_CHILD_FREEZE_ATTEMPT_PATH,
            _historical_x9_child_freeze_attempt_bytes(),
            "historical X9 child freeze attempt marker",
        ),
        (
            HISTORICAL_X9_CHILD_FREEZE_FAILURE_PATH,
            _historical_x9_child_freeze_failure_bytes(),
            "historical X9 child freeze failure receipt",
        ),
    ):
        path = _resolve_project_relative(
            project_root,
            relative_path,
            require_file=True,
        )
        if _read_bounded(path, _CHILD_MAX_BYTES, context) != expected:
            raise OODExternalV2IntegrityError(f"{context} differs")
        _require_git_ignored_and_untracked(
            project_root,
            relative_path,
            context=context,
        )


def _child_freeze_attempt_body_from_identity(
    *,
    parent_config_file_sha256: str,
    implementation_revision: str,
    project_source_tree_sha256: str,
    python_environment_sha256: str,
    git_runtime_tree_sha256: str,
    frozen_at_utc: datetime,
    counts: tuple[int, int, int, int],
) -> dict[str, object]:
    challenge_records, zzu_records, zzu_patients, selected_records_total = counts
    body: dict[str, object] = {
        "artifact_type": SUCCESSOR_CHILD_FREEZE_ATTEMPT_ARTIFACT_TYPE,
        "authorization_consumption_ordinal": 1,
        "authorization_id": "x10_child_freeze_attempt_1",
        "child_contract_relative_path": SUCCESSOR_CHILD_CONFIG_PATH,
        "child_frozen_at_utc": frozen_at_utc.astimezone(UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        "contains_external_source_bytes_or_identifiers": False,
        "contains_model_outputs_embeddings_or_scores": False,
        "declared_counts": {
            "challenge_records": challenge_records,
            "selected_records_total": selected_records_total,
            "zzu_patients": zzu_patients,
            "zzu_records": zzu_records,
        },
        "external_one_shot_claim_consumed_at_marker_creation": False,
        "git_runtime_tree_sha256": git_runtime_tree_sha256,
        "historical_x8_failure_receipt_required_absent": True,
        "historical_x8_inventory_builder_attempt_artifact_sha256": (
            HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_SHA256
        ),
        "historical_x8_inventory_builder_attempt_file_sha256": (
            HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_FILE_SHA256
        ),
        "historical_x8_inventory_builder_attempt_path": (
            HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_PATH
        ),
        "historical_x8_private_inventory_artifact_sha256": (
            HISTORICAL_X8_PRIVATE_INVENTORY_ARTIFACT_SHA256
        ),
        "historical_x8_private_inventory_file_sha256": (
            HISTORICAL_X8_PRIVATE_INVENTORY_FILE_SHA256
        ),
        "historical_x8_public_projection_artifact_sha256": (
            HISTORICAL_X8_PUBLIC_PROJECTION_ARTIFACT_SHA256
        ),
        "historical_x8_public_projection_file_sha256": (
            HISTORICAL_X8_PUBLIC_PROJECTION_FILE_SHA256
        ),
        "historical_x9_authorization_consumed_failed_retained": True,
        "historical_x9_child_freeze_attempt_artifact_sha256": (
            HISTORICAL_X9_CHILD_FREEZE_ATTEMPT_ARTIFACT_SHA256
        ),
        "historical_x9_child_freeze_attempt_file_sha256": (
            HISTORICAL_X9_CHILD_FREEZE_ATTEMPT_FILE_SHA256
        ),
        "historical_x9_child_freeze_attempt_path": (
            HISTORICAL_X9_CHILD_FREEZE_ATTEMPT_PATH
        ),
        "historical_x9_child_freeze_failure_receipt_artifact_sha256": (
            HISTORICAL_X9_CHILD_FREEZE_FAILURE_ARTIFACT_SHA256
        ),
        "historical_x9_child_freeze_failure_receipt_file_sha256": (
            HISTORICAL_X9_CHILD_FREEZE_FAILURE_FILE_SHA256
        ),
        "historical_x9_child_freeze_failure_receipt_path": (
            HISTORICAL_X9_CHILD_FREEZE_FAILURE_PATH
        ),
        "historical_x9_failure_official_source_content_accessed": True,
        "historical_x9_failure_output_state": "NONE",
        "historical_x9_failure_reason": "STAGE_REFUSED",
        "historical_x9_failure_stage": "decision_and_child_materialization",
        "historical_x9_failure_stage_ordinal": 9,
        "implementation_revision": implementation_revision,
        "inventory_builder_authorization_reused": False,
        "maximum_consumptions": 1,
        "parent_config_file_sha256": parent_config_file_sha256,
        "project_source_tree_sha256": project_source_tree_sha256,
        "protocol_child_freeze_attempt_ordinal": 2,
        "protocol_id": PROTOCOL_ID,
        "python_environment_sha256": python_environment_sha256,
        "schema_version": 1,
        "state": "PRECLAIM_CHILD_FREEZE_AUTHORIZATION_CONSUMED",
    }
    body["artifact_sha256"] = canonical_sha256(body)
    return body


def _child_freeze_attempt_body(preflight: ChildFreezePreflight) -> dict[str, object]:
    if not isinstance(preflight, ChildFreezePreflight):
        raise TypeError("preflight must be ChildFreezePreflight")
    return _child_freeze_attempt_body_from_identity(
        parent_config_file_sha256=preflight.parent.file_sha256,
        implementation_revision=preflight.implementation_revision,
        project_source_tree_sha256=preflight.project_source_tree.tree_sha256,
        python_environment_sha256=(
            preflight.runtime_environment.python_environment_sha256
        ),
        git_runtime_tree_sha256=(
            preflight.runtime_environment.git_tool.runtime_tree.tree_sha256
        ),
        frozen_at_utc=preflight.frozen_at_utc,
        counts=preflight.declared_counts,
    )


def _child_freeze_attempt_bytes(preflight: ChildFreezePreflight) -> bytes:
    return canonical_json_bytes(_child_freeze_attempt_body(preflight))


def _child_freeze_failure_body(
    preflight: ChildFreezePreflight,
    *,
    failure_stage: str,
    reason: str,
    official_source_content_accessed: bool,
    output_state: str,
) -> dict[str, object]:
    if not isinstance(preflight, ChildFreezePreflight):
        raise TypeError("preflight must be ChildFreezePreflight")
    if failure_stage not in CHILD_FREEZE_ATTEMPT_STAGES:
        raise OODExternalV2IntegrityError("child freeze failure stage is invalid")
    if reason not in CHILD_FREEZE_FAILURE_REASONS:
        raise OODExternalV2IntegrityError("child freeze failure reason is invalid")
    if output_state not in CHILD_FREEZE_OUTPUT_STATES:
        raise OODExternalV2IntegrityError("child freeze output state is invalid")
    if type(official_source_content_accessed) is not bool:
        raise TypeError("official_source_content_accessed must be bool")
    attempt = _child_freeze_attempt_body(preflight)
    body: dict[str, object] = {
        "artifact_type": SUCCESSOR_CHILD_FREEZE_FAILURE_ARTIFACT_TYPE,
        "authorization_consumed": True,
        "authorization_id": "x10_child_freeze_attempt_1",
        "authorization_marker_artifact_sha256": attempt["artifact_sha256"],
        "authorization_marker_file_sha256": sha256_bytes(
            _child_freeze_attempt_bytes(preflight)
        ),
        "contains_external_source_bytes_or_identifiers": False,
        "contains_model_outputs_embeddings_or_scores": False,
        "external_one_shot_claim_consumed": False,
        "failure_reason": reason,
        "failure_requires_new_frozen_amendment_and_authorization_id": True,
        "failure_stage": failure_stage,
        "failure_stage_ordinal": CHILD_FREEZE_ATTEMPT_STAGES.index(failure_stage),
        "implementation_revision": preflight.implementation_revision,
        "official_source_content_accessed": official_source_content_accessed,
        "output_state": output_state,
        "parent_config_file_sha256": preflight.parent.file_sha256,
        "protocol_id": PROTOCOL_ID,
        "quality_model_score_logit_probability_or_metric_observed": False,
        "retry_resume_or_reuse_authorized": False,
        "schema_version": 1,
        "state": "PRECLAIM_CHILD_FREEZE_FAILED",
        "waveform_sample_decode_occurred": False,
    }
    body["artifact_sha256"] = canonical_sha256(body)
    return body


def _child_freeze_failure_bytes(
    preflight: ChildFreezePreflight,
    *,
    failure_stage: str,
    reason: str,
    official_source_content_accessed: bool,
    output_state: str,
) -> bytes:
    return canonical_json_bytes(
        _child_freeze_failure_body(
            preflight,
            failure_stage=failure_stage,
            reason=reason,
            official_source_content_accessed=official_source_content_accessed,
            output_state=output_state,
        )
    )


def _verify_child_freeze_marker(
    expected_bytes: bytes,
    *,
    project_root: Path,
) -> str:
    marker = _resolve_project_relative(
        project_root,
        SUCCESSOR_CHILD_FREEZE_ATTEMPT_PATH,
        require_file=True,
    )
    observed = _read_bounded(
        marker,
        _CHILD_MAX_BYTES,
        "X10 child freeze attempt marker",
    )
    if observed != expected_bytes:
        raise OODExternalV2IntegrityError("X10 child freeze attempt marker differs")
    _require_git_ignored_and_untracked(
        project_root,
        SUCCESSOR_CHILD_FREEZE_ATTEMPT_PATH,
        context="X10 child freeze attempt marker",
    )
    return sha256_bytes(observed)


def verify_child_freeze_authorization_available(
    preflight: ChildFreezePreflight,
    *,
    project_root: str | Path,
) -> None:
    if not isinstance(preflight, ChildFreezePreflight):
        raise TypeError("preflight must be ChildFreezePreflight")
    root = _strict_project_root(project_root)
    if root != preflight.project_root:
        raise OODExternalV2IntegrityError("child freeze preflight project root differs")
    _verify_historical_x9_child_freeze_artifacts(root)
    for relative_path, context in (
        (SUCCESSOR_CHILD_FREEZE_ATTEMPT_PATH, "X10 child freeze authorization"),
        (SUCCESSOR_CHILD_FREEZE_FAILURE_PATH, "X10 child freeze failure receipt"),
    ):
        candidate = _resolve_project_relative(root, relative_path, require_file=False)
        if candidate.exists() or _is_indirect(candidate):
            raise OODExternalV2IntegrityError(f"{context} is unavailable")
        _require_git_ignored_and_untracked(root, relative_path, context=context)
    if preflight.output_path.exists() or _is_indirect(preflight.output_path):
        raise OODExternalV2IntegrityError("child contract destination is unavailable")


def consume_child_freeze_authorization(
    preflight: ChildFreezePreflight,
    *,
    project_root: str | Path,
    visibility_witness: Callable[[], None] | None = None,
) -> str:
    if visibility_witness is not None and not callable(visibility_witness):
        raise TypeError("visibility_witness must be callable or None")
    root = _strict_project_root(project_root)
    verify_child_freeze_authorization_available(preflight, project_root=root)
    marker = _resolve_project_relative(
        root,
        SUCCESSOR_CHILD_FREEZE_ATTEMPT_PATH,
        require_file=False,
    )
    _atomic_write_new(
        marker,
        _child_freeze_attempt_bytes(preflight),
        visibility_witness=visibility_witness,
        expected_parent_identity=preflight.protocol_artifact_parent_identity,
    )
    return _verify_child_freeze_marker(
        _child_freeze_attempt_bytes(preflight),
        project_root=root,
    )


def record_child_freeze_failure(
    preflight: ChildFreezePreflight,
    *,
    project_root: str | Path,
    failure_stage: str,
    reason: str,
    official_source_content_accessed: bool,
    output_state: str,
    visibility_witness: Callable[[], None] | None = None,
    publication_witness: Callable[[], None] | None = None,
) -> str:
    for callback, context in (
        (visibility_witness, "visibility_witness"),
        (publication_witness, "publication_witness"),
    ):
        if callback is not None and not callable(callback):
            raise TypeError(f"{context} must be callable or None")
    root = _strict_project_root(project_root)
    _verify_child_freeze_marker(
        _child_freeze_attempt_bytes(preflight),
        project_root=root,
    )
    receipt = _resolve_project_relative(
        root,
        SUCCESSOR_CHILD_FREEZE_FAILURE_PATH,
        require_file=False,
    )
    if receipt.exists() or _is_indirect(receipt):
        raise OODExternalV2IntegrityError(
            "X10 child freeze failure receipt already exists"
        )
    _require_git_ignored_and_untracked(
        root,
        SUCCESSOR_CHILD_FREEZE_FAILURE_PATH,
        context="X10 child freeze failure receipt",
    )
    payload = _child_freeze_failure_bytes(
        preflight,
        failure_stage=failure_stage,
        reason=reason,
        official_source_content_accessed=official_source_content_accessed,
        output_state=output_state,
    )
    visible = False
    published = False

    def mark_visible() -> None:
        nonlocal visible
        visible = True
        if visibility_witness is not None:
            visibility_witness()

    def mark_published() -> None:
        nonlocal published
        published = True
        if publication_witness is not None:
            publication_witness()

    _atomic_write_new(
        receipt,
        payload,
        visibility_witness=mark_visible,
        publication_witness=mark_published,
        expected_parent_identity=preflight.protocol_artifact_parent_identity,
    )
    if not visible or not published:
        raise OODExternalV2IntegrityError(
            "X10 child freeze failure receipt publication was not witnessed"
        )
    observed = _read_bounded(
        receipt,
        _CHILD_MAX_BYTES,
        "X10 child freeze failure receipt",
    )
    if observed != payload:
        raise OODExternalV2IntegrityError("X10 child freeze failure receipt differs")
    _require_git_ignored_and_untracked(
        root,
        SUCCESSOR_CHILD_FREEZE_FAILURE_PATH,
        context="X10 child freeze failure receipt",
    )
    return sha256_bytes(observed)


def _inventory_builder_failure_body(
    preflight: InventoryBuilderPreflight,
    *,
    failure_stage: str,
    official_source_content_accessed: bool,
    output_state: str,
) -> dict[str, object]:
    if failure_stage not in INVENTORY_BUILDER_ATTEMPT_STAGES:
        raise OODExternalV2IntegrityError("inventory builder failure stage is invalid")
    if output_state not in INVENTORY_BUILDER_OUTPUT_STATES:
        raise OODExternalV2IntegrityError("inventory builder output state is invalid")
    if type(official_source_content_accessed) is not bool:
        raise TypeError("official_source_content_accessed must be bool")
    attempt_body = _inventory_builder_attempt_body(preflight)
    body: dict[str, object] = {
        "artifact_type": SUCCESSOR_INVENTORY_BUILDER_FAILURE_ARTIFACT_TYPE,
        "authorization_consumed": True,
        "authorization_id": "x8_inventory_build_attempt_1",
        "authorization_marker_artifact_sha256": attempt_body["artifact_sha256"],
        "authorization_marker_file_sha256": sha256_bytes(
            _inventory_builder_attempt_bytes(preflight)
        ),
        "contains_external_source_bytes_or_identifiers": False,
        "contains_model_outputs_embeddings_or_scores": False,
        "external_one_shot_claim_consumed": False,
        "failure_requires_new_frozen_amendment_and_authorization_id": True,
        "failure_stage": failure_stage,
        "failure_stage_ordinal": INVENTORY_BUILDER_ATTEMPT_STAGES.index(failure_stage),
        "implementation_revision": preflight.implementation_revision,
        "official_source_content_accessed": official_source_content_accessed,
        "output_state": output_state,
        "parent_config_file_sha256": preflight.parent_config_file_sha256,
        "protocol_id": PROTOCOL_ID,
        "quality_model_score_logit_probability_or_metric_observed": False,
        "retry_resume_or_reuse_authorized": False,
        "schema_version": 1,
        "state": "PRECLAIM_INVENTORY_BUILD_FAILED",
        "waveform_sample_decode_occurred": False,
    }
    body["artifact_sha256"] = canonical_sha256(body)
    return body


def _inventory_builder_failure_bytes(
    preflight: InventoryBuilderPreflight,
    *,
    failure_stage: str,
    official_source_content_accessed: bool,
    output_state: str,
) -> bytes:
    return canonical_json_bytes(
        _inventory_builder_failure_body(
            preflight,
            failure_stage=failure_stage,
            official_source_content_accessed=official_source_content_accessed,
            output_state=output_state,
        )
    )


def _inventory_builder_preflight_from_contract(
    parent: OODExternalV2ParentConfig,
    child: OODExternalV2ChildContract,
) -> InventoryBuilderPreflight:
    """Reconstruct the X-revision attempt identity while executing revision Y."""

    if not isinstance(parent, OODExternalV2ParentConfig):
        raise TypeError("parent must be OODExternalV2ParentConfig")
    if not isinstance(child, OODExternalV2ChildContract):
        raise TypeError("child must be OODExternalV2ChildContract")
    if (
        parent.raw_source_bindings is None
        or parent.seven_zip_tool_binding is None
        or parent.inventory_counts is None
        or child.parent_config_file_sha256 != parent.file_sha256
    ):
        raise OODExternalV2IntegrityError(
            "child cannot reconstruct its inventory builder authorization"
        )
    return InventoryBuilderPreflight(
        status="INVENTORY_BUILDER_PREFLIGHT_VERIFIED",
        parent_config_file_sha256=parent.file_sha256,
        implementation_revision=child.implementation_revision,
        project_source_tree_sha256=child.project_source_tree.tree_sha256,
        python_environment_sha256=(
            child.runtime_environment.python_environment_sha256
        ),
        git_runtime_tree_sha256=(
            child.runtime_environment.git_tool.runtime_tree.tree_sha256
        ),
        raw_source_bindings=parent.raw_source_bindings,
        seven_zip_tool_binding=parent.seven_zip_tool_binding,
        inventory_counts=parent.inventory_counts,
    )


def _verify_child_inventory_builder_attempt(
    parent: OODExternalV2ParentConfig,
    child: OODExternalV2ChildContract,
    *,
    project_root: str | Path,
) -> None:
    root = _strict_project_root(project_root)
    _verify_historical_x9_child_freeze_artifacts(root)
    observed_file_sha256 = HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_FILE_SHA256
    if (
        child.inventory_builder_attempt.file_sha256 != observed_file_sha256
        or child.inventory_builder_attempt.artifact_sha256
        != HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_SHA256
        or child.inventory_builder_attempt.relative_path
        != HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_PATH
    ):
        raise OODExternalV2IntegrityError(
            "child inventory builder attempt binding differs"
        )
    if (
        child.inventory.file_sha256 != HISTORICAL_X8_PRIVATE_INVENTORY_FILE_SHA256
        or child.inventory.inventory_sha256
        != HISTORICAL_X8_PRIVATE_INVENTORY_ARTIFACT_SHA256
        or child.public_inventory_projection is None
        or child.public_inventory_projection.file_sha256
        != HISTORICAL_X8_PUBLIC_PROJECTION_FILE_SHA256
        or child.public_inventory_projection.artifact_sha256
        != HISTORICAL_X8_PUBLIC_PROJECTION_ARTIFACT_SHA256
    ):
        raise OODExternalV2IntegrityError(
            "child successful X8 inventory evidence binding differs"
        )
    expected_attempt_body = _child_freeze_attempt_body_from_identity(
        parent_config_file_sha256=parent.file_sha256,
        implementation_revision=child.implementation_revision,
        project_source_tree_sha256=child.project_source_tree.tree_sha256,
        python_environment_sha256=(
            child.runtime_environment.python_environment_sha256
        ),
        git_runtime_tree_sha256=(
            child.runtime_environment.git_tool.runtime_tree.tree_sha256
        ),
        frozen_at_utc=child.frozen_at_utc,
        counts=(
            child.inventory.challenge_records,
            child.inventory.zzu_records,
            child.inventory.zzu_patients,
            child.inventory.selected_records_total,
        ),
    )
    expected_attempt_bytes = canonical_json_bytes(expected_attempt_body)
    observed_child_freeze_file_sha256 = _verify_child_freeze_marker(
        expected_attempt_bytes,
        project_root=root,
    )
    if (
        child.child_freeze_attempt.relative_path
        != SUCCESSOR_CHILD_FREEZE_ATTEMPT_PATH
        or child.child_freeze_attempt.file_sha256
        != observed_child_freeze_file_sha256
        or child.child_freeze_attempt.artifact_sha256
        != expected_attempt_body["artifact_sha256"]
    ):
        raise OODExternalV2IntegrityError("child freeze attempt binding differs")
    child_freeze_failure = _resolve_project_relative(
        root,
        SUCCESSOR_CHILD_FREEZE_FAILURE_PATH,
        require_file=False,
    )
    if child_freeze_failure.exists() or _is_indirect(child_freeze_failure):
        raise OODExternalV2IntegrityError(
            "child verification is forbidden after an X10 child freeze failure"
        )
    _require_git_ignored_and_untracked(
        root,
        SUCCESSOR_CHILD_FREEZE_FAILURE_PATH,
        context="absent X10 child freeze failure receipt",
    )


def verify_inventory_builder_attempt_marker(
    preflight: InventoryBuilderPreflight,
    *,
    project_root: str | Path,
) -> str:
    if not isinstance(preflight, InventoryBuilderPreflight):
        raise TypeError("preflight must be InventoryBuilderPreflight")
    root = _strict_project_root(project_root)
    for relative_path, label in (
        (HISTORICAL_X4_INVENTORY_BUILDER_ATTEMPT_PATH, "X4"),
        (HISTORICAL_X5_INVENTORY_BUILDER_ATTEMPT_PATH, "X5"),
    ):
        historical = _resolve_project_relative(
            root,
            relative_path,
            require_file=False,
        )
        if historical.exists() or _is_indirect(historical):
            raise OODExternalV2IntegrityError(
                f"retired {label} inventory builder authorization path must remain absent"
            )
        _require_git_ignored_and_untracked(
            root,
            relative_path,
            context=f"retired {label} inventory builder authorization",
        )
    _verify_historical_x6_inventory_builder_attempt(root)
    _verify_historical_x7_inventory_builder_artifacts(root)
    marker = _resolve_project_relative(
        root,
        SUCCESSOR_INVENTORY_BUILDER_ATTEMPT_PATH,
        require_file=True,
    )
    expected = _inventory_builder_attempt_bytes(preflight)
    observed = _read_bounded(
        marker,
        _CHILD_MAX_BYTES,
        "inventory builder attempt marker",
    )
    if observed != expected:
        raise OODExternalV2IntegrityError(
            "inventory builder attempt marker differs from its authorization"
        )
    _require_git_ignored_and_untracked(
        root,
        SUCCESSOR_INVENTORY_BUILDER_ATTEMPT_PATH,
        context="inventory builder attempt marker",
    )
    return sha256_bytes(observed)


def verify_inventory_builder_authorization_available(
    preflight: InventoryBuilderPreflight,
    *,
    project_root: str | Path,
) -> None:
    """Prove X4/X5 absent, X6/X7 exact, and the fresh X8 state unused."""

    if not isinstance(preflight, InventoryBuilderPreflight):
        raise TypeError("preflight must be InventoryBuilderPreflight")
    root = _strict_project_root(project_root)
    if _verify_clean_git_revision(root) != preflight.implementation_revision:
        raise OODExternalV2IntegrityError(
            "inventory builder revision changed before authorization availability"
        )
    for relative_path, context in (
        (
            HISTORICAL_X4_INVENTORY_BUILDER_ATTEMPT_PATH,
            "retired X4 inventory builder authorization",
        ),
        (
            HISTORICAL_X5_INVENTORY_BUILDER_ATTEMPT_PATH,
            "retired X5 inventory builder authorization",
        ),
    ):
        marker = _resolve_project_relative(
            root,
            relative_path,
            require_file=False,
        )
        if marker.exists() or _is_indirect(marker):
            raise OODExternalV2IntegrityError(f"{context} is unavailable")
        _require_git_ignored_and_untracked(
            root,
            relative_path,
            context=context,
        )
    _verify_historical_x6_inventory_builder_attempt(root)
    _verify_historical_x7_inventory_builder_artifacts(root)
    for relative_path, context in (
        (
            SUCCESSOR_INVENTORY_BUILDER_ATTEMPT_PATH,
            "X8 inventory builder authorization",
        ),
        (
            SUCCESSOR_INVENTORY_BUILDER_FAILURE_PATH,
            "X8 inventory builder failure receipt",
        ),
    ):
        candidate = _resolve_project_relative(
            root,
            relative_path,
            require_file=False,
        )
        if candidate.exists() or _is_indirect(candidate):
            raise OODExternalV2IntegrityError(f"{context} is unavailable")
        _require_git_ignored_and_untracked(
            root,
            relative_path,
            context=context,
        )


def consume_inventory_builder_authorization(
    preflight: InventoryBuilderPreflight,
    *,
    project_root: str | Path,
    visibility_witness: Callable[[], None] | None = None,
) -> str:
    """Durably consume the sole X8 build authorization before source-byte access."""

    if not isinstance(preflight, InventoryBuilderPreflight):
        raise TypeError("preflight must be InventoryBuilderPreflight")
    if visibility_witness is not None and not callable(visibility_witness):
        raise TypeError("visibility_witness must be callable or None")
    root = _strict_project_root(project_root)
    verify_inventory_builder_authorization_available(
        preflight,
        project_root=root,
    )
    marker = _resolve_project_relative(
        root,
        SUCCESSOR_INVENTORY_BUILDER_ATTEMPT_PATH,
        require_file=False,
    )
    if marker.exists() or _is_indirect(marker):
        raise OODExternalV2IntegrityError(
            "X8 inventory builder authorization is already consumed"
        )
    parent_identity = _owned_directory_identity(marker.parent)
    visible = False
    published = False

    def mark_visible() -> None:
        nonlocal visible
        visible = True
        if visibility_witness is not None:
            visibility_witness()

    def mark_published() -> None:
        nonlocal published
        published = True

    _atomic_write_new(
        marker,
        _inventory_builder_attempt_bytes(preflight),
        visibility_witness=mark_visible,
        publication_witness=mark_published,
        expected_parent_identity=parent_identity,
    )
    if not visible or not published:
        raise OODExternalV2IntegrityError(
            "X8 inventory builder authorization publication was not witnessed"
        )
    return verify_inventory_builder_attempt_marker(preflight, project_root=root)


def verify_inventory_builder_failure_receipt(
    preflight: InventoryBuilderPreflight,
    *,
    project_root: str | Path,
    expected_failure_stage: str | None = None,
    expected_official_source_content_accessed: bool | None = None,
    expected_output_state: str | None = None,
) -> str:
    """Strictly reload the path-free, immutable X8 failure receipt."""

    if not isinstance(preflight, InventoryBuilderPreflight):
        raise TypeError("preflight must be InventoryBuilderPreflight")
    if (
        expected_official_source_content_accessed is not None
        and type(expected_official_source_content_accessed) is not bool
    ):
        raise TypeError("expected_official_source_content_accessed must be bool or None")
    root = _strict_project_root(project_root)
    verify_inventory_builder_attempt_marker(preflight, project_root=root)
    receipt = _resolve_project_relative(
        root,
        SUCCESSOR_INVENTORY_BUILDER_FAILURE_PATH,
        require_file=True,
    )
    observed = _read_bounded(
        receipt,
        _CHILD_MAX_BYTES,
        "inventory builder failure receipt",
    )
    try:
        decoded: object = json.loads(observed)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise OODExternalV2IntegrityError(
            "inventory builder failure receipt cannot be decoded"
        ) from error
    if not isinstance(decoded, dict) or canonical_json_bytes(decoded) != observed:
        raise OODExternalV2IntegrityError(
            "inventory builder failure receipt is not canonical"
        )
    stage = decoded.get("failure_stage")
    official_source_content_accessed = decoded.get(
        "official_source_content_accessed"
    )
    output_state = decoded.get("output_state")
    if (
        not isinstance(stage, str)
        or type(official_source_content_accessed) is not bool
        or not isinstance(output_state, str)
    ):
        raise OODExternalV2IntegrityError(
            "inventory builder failure receipt classification is invalid"
        )
    if expected_failure_stage is not None and stage != expected_failure_stage:
        raise OODExternalV2IntegrityError(
            "inventory builder failure receipt stage differs"
        )
    if (
        expected_official_source_content_accessed is not None
        and official_source_content_accessed
        is not expected_official_source_content_accessed
    ):
        raise OODExternalV2IntegrityError(
            "inventory builder failure receipt source-access state differs"
        )
    if expected_output_state is not None and output_state != expected_output_state:
        raise OODExternalV2IntegrityError(
            "inventory builder failure receipt output state differs"
        )
    expected = _inventory_builder_failure_bytes(
        preflight,
        failure_stage=stage,
        official_source_content_accessed=official_source_content_accessed,
        output_state=output_state,
    )
    if observed != expected:
        raise OODExternalV2IntegrityError(
            "inventory builder failure receipt differs from its authorization"
        )
    _require_git_ignored_and_untracked(
        root,
        SUCCESSOR_INVENTORY_BUILDER_FAILURE_PATH,
        context="inventory builder failure receipt",
    )
    return sha256_bytes(observed)


def record_inventory_builder_failure(
    preflight: InventoryBuilderPreflight,
    *,
    project_root: str | Path,
    failure_stage: str,
    official_source_content_accessed: bool,
    output_state: str,
) -> str:
    """Publish one sanitized receipt after a consumed X8 attempt fails."""

    if not isinstance(preflight, InventoryBuilderPreflight):
        raise TypeError("preflight must be InventoryBuilderPreflight")
    root = _strict_project_root(project_root)
    verify_inventory_builder_attempt_marker(preflight, project_root=root)
    receipt = _resolve_project_relative(
        root,
        SUCCESSOR_INVENTORY_BUILDER_FAILURE_PATH,
        require_file=False,
    )
    if receipt.exists() or _is_indirect(receipt):
        raise OODExternalV2IntegrityError(
            "X8 inventory builder failure receipt already exists"
        )
    parent_identity = _owned_directory_identity(receipt.parent)
    visible = False
    published = False

    def mark_visible() -> None:
        nonlocal visible
        visible = True

    def mark_published() -> None:
        nonlocal published
        published = True

    _atomic_write_new(
        receipt,
        _inventory_builder_failure_bytes(
            preflight,
            failure_stage=failure_stage,
            official_source_content_accessed=official_source_content_accessed,
            output_state=output_state,
        ),
        visibility_witness=mark_visible,
        publication_witness=mark_published,
        expected_parent_identity=parent_identity,
    )
    if not visible or not published:
        raise OODExternalV2IntegrityError(
            "X8 inventory builder failure receipt publication was not witnessed"
        )
    return verify_inventory_builder_failure_receipt(
        preflight,
        project_root=root,
        expected_failure_stage=failure_stage,
        expected_official_source_content_accessed=(
            official_source_content_accessed
        ),
        expected_output_state=output_state,
    )


def verify_inventory_builder_raw_source_bindings(
    preflight: InventoryBuilderPreflight,
    *,
    project_root: str | Path,
    content_access_witness: Callable[[], None] | None = None,
) -> None:
    """Hash every frozen official source after the attempt marker is durable."""

    if not isinstance(preflight, InventoryBuilderPreflight):
        raise TypeError("preflight must be InventoryBuilderPreflight")
    if content_access_witness is not None and not callable(content_access_witness):
        raise TypeError("content_access_witness must be callable or None")
    root = _strict_project_root(project_root)
    verify_inventory_builder_attempt_marker(preflight, project_root=root)
    observed = {
        name: _raw_source_binding_for_path(
            root,
            binding.relative_path,
            context=f"inventory raw source {name}",
            official_md5=binding.official_md5,
            content_access_witness=content_access_witness,
        )
        for name, binding in preflight.raw_source_bindings.items()
    }
    if observed != dict(preflight.raw_source_bindings):
        raise OODExternalV2IntegrityError(
            "inventory raw-source bytes differ from the frozen parent"
        )


def verify_inventory_builder_postflight(
    preflight: InventoryBuilderPreflight,
    *,
    parent_path: str | Path,
    project_root: str | Path,
    implementation_revision: str,
    inventory_path: str | Path,
    public_projection_path: str | Path,
    expected_inventory_file_sha256: str,
    expected_inventory_sha256: str,
    expected_public_projection_file_sha256: str,
    expected_public_projection_artifact_sha256: str,
) -> InventoryBuilderPostflight:
    """Recheck builder controls and exact output bytes before reporting success."""

    if not isinstance(preflight, InventoryBuilderPreflight):
        raise TypeError("preflight must be InventoryBuilderPreflight")
    root = _strict_project_root(project_root)
    failure_receipt = _resolve_project_relative(
        root,
        SUCCESSOR_INVENTORY_BUILDER_FAILURE_PATH,
        require_file=False,
    )
    if failure_receipt.exists() or _is_indirect(failure_receipt):
        raise OODExternalV2IntegrityError(
            "inventory builder postflight is forbidden after a failure receipt"
        )
    repeated = verify_inventory_builder_preflight(
        parent_path,
        root,
        implementation_revision,
    )
    if repeated != preflight:
        raise OODExternalV2IntegrityError(
            "inventory builder control boundary changed after output creation"
        )
    verify_inventory_builder_attempt_marker(preflight, project_root=project_root)
    private = _require_project_file(
        root,
        Path(os.path.abspath(os.fspath(inventory_path))),
        context="private external inventory",
    )
    public = _require_project_file(
        root,
        Path(os.path.abspath(os.fspath(public_projection_path))),
        context="public inventory projection",
    )
    if (
        private.relative_to(root).as_posix() != SUCCESSOR_PRIVATE_INVENTORY_PATH
        or public.relative_to(root).as_posix() != SUCCESSOR_PUBLIC_PROJECTION_PATH
    ):
        raise OODExternalV2IntegrityError(
            "inventory builder outputs differ from the frozen successor paths"
        )
    inventory_file_sha256 = sha256_file(private)
    projection_file_sha256 = sha256_file(public)
    for value, context in (
        (expected_inventory_file_sha256, "expected private inventory file"),
        (expected_inventory_sha256, "expected private inventory artifact"),
        (expected_public_projection_file_sha256, "expected public projection file"),
        (
            expected_public_projection_artifact_sha256,
            "expected public projection artifact",
        ),
    ):
        _digest(value, context)
    try:
        inventory = load_external_inventory(private)
    except Exception as error:
        raise OODExternalV2IntegrityError(
            "private external inventory cannot be reloaded after construction"
        ) from error
    challenge_records = sum(
        record.dataset == CHALLENGE_2011_DATASET for record in inventory.records
    )
    zzu_records = sum(
        record.dataset == ZZU_PEDIATRIC_DATASET for record in inventory.records
    )
    zzu_patients = len(
        {
            record.patient_key
            for record in inventory.records
            if record.dataset == ZZU_PEDIATRIC_DATASET
            and record.patient_key is not None
        }
    )
    counts = preflight.inventory_counts
    if (
        challenge_records != counts.challenge_records
        or zzu_records != counts.zzu_records
        or zzu_patients != counts.zzu_patients
        or len(inventory.records) != counts.total_records
    ):
        raise OODExternalV2IntegrityError(
            "inventory output counts differ from the frozen parent"
        )
    projection_artifact_sha256 = _verify_public_projection_file(
        public,
        inventory=inventory,
        challenge_records=challenge_records,
        zzu_records=zzu_records,
        expected_counts=counts,
    )
    if (
        inventory_file_sha256 != expected_inventory_file_sha256
        or inventory.inventory_sha256 != expected_inventory_sha256
        or projection_file_sha256 != expected_public_projection_file_sha256
        or projection_artifact_sha256
        != expected_public_projection_artifact_sha256
    ):
        raise OODExternalV2IntegrityError(
            "inventory builder output hashes differ from the in-memory build"
        )
    return InventoryBuilderPostflight(
        status="INVENTORY_BUILDER_POSTFLIGHT_VERIFIED",
        preflight=preflight,
        inventory_file_sha256=inventory_file_sha256,
        inventory_sha256=inventory.inventory_sha256,
        public_projection_file_sha256=projection_file_sha256,
        public_projection_artifact_sha256=projection_artifact_sha256,
    )


def _child_freeze_preflight_stage(
    stage: str,
    operation: Callable[[], Any],
) -> Any:
    if stage not in CHILD_FREEZE_PREFLIGHT_STAGES:
        raise ValueError("child freeze preflight stage is not allowlisted")
    try:
        return operation()
    except ChildFreezePreflightStageError:
        raise
    except BaseException:
        raise ChildFreezePreflightStageError(stage) from None


def verify_child_freeze_preflight(
    *,
    parent_path: str | Path,
    project_root: str | Path,
    inventory_path: str | Path,
    public_projection_path: str | Path,
    implementation_revision: str,
    frozen_at_utc: str,
    challenge_root: str | Path,
    zzu_root: str | Path,
    challenge_records: int,
    zzu_records: int,
    zzu_patients: int,
    selected_records_total: int,
    output_path: str | Path,
    seven_zip_executable: str | Path = "7z",
) -> ChildFreezePreflight:
    """Verify X10 controls without reading official/private content or writing state."""

    def parent_lineage() -> tuple[Path, OODExternalV2ParentConfig, str]:
        root_value = _strict_project_root(project_root)
        parent_value = _load_parent_for_operation(parent_path, project_root=root_value)
        assert_external_v2_parent_executable(parent_value)
        revision_value = _revision(
            implementation_revision,
            "implementation revision",
        )
        if _verify_clean_git_revision(root_value) != revision_value:
            raise OODExternalV2IntegrityError(
                "child freeze must run at the implementation revision"
            )
        _verify_successor_amendment_revision(
            root_value,
            implementation_revision=revision_value,
        )
        return root_value, parent_value, revision_value

    root, parent, revision = cast(
        tuple[Path, OODExternalV2ParentConfig, str],
        _child_freeze_preflight_stage("parent_lineage", parent_lineage),
    )
    runtime_environment = cast(
        RuntimeEnvironmentBinding,
        _child_freeze_preflight_stage(
            "runtime_environment",
            _current_runtime_environment,
        ),
    )

    def git_source_provenance() -> ProjectSourceTreeBinding:
        _verify_git_remote_state(root, expected_revision=revision)
        _verify_private_history_absent(root)
        _verify_tracked_head_blob(
            root,
            revision=revision,
            relative_path=SUCCESSOR_PARENT_CONFIG_PATH,
            expected_file_sha256=parent.file_sha256,
        )
        project_tree = _build_project_source_tree(root)
        _verify_project_source_tree_at_revisions(
            root,
            project_tree,
            implementation_revision=revision,
            execution_revision=None,
        )
        _verify_imported_project_module_origins(root, project_tree)
        commit_check = _run_git(
            root,
            "cat-file",
            "-e",
            f"{revision}^{{commit}}",
            allow_empty=True,
        )
        if commit_check.returncode != 0:
            raise OODExternalV2IntegrityError(
                "implementation revision is not a Git commit"
            )
        return project_tree

    project_source_tree = cast(
        ProjectSourceTreeBinding,
        _child_freeze_preflight_stage(
            "git_source_provenance",
            git_source_provenance,
        ),
    )

    def x8_inventory_evidence() -> tuple[Path, Path, Path]:
        _verify_historical_x9_child_freeze_artifacts(root)
        inventory_file = _require_project_file(
            root,
            Path(os.path.abspath(os.fspath(inventory_path))),
            context="private external inventory",
        )
        projection_file = _require_project_file(
            root,
            Path(os.path.abspath(os.fspath(public_projection_path))),
            context="public inventory projection",
        )
        requested_tool = Path(os.path.abspath(os.fspath(seven_zip_executable)))
        if inventory_file.relative_to(root).as_posix() != SUCCESSOR_PRIVATE_INVENTORY_PATH:
            raise OODExternalV2IntegrityError(
                "private inventory path differs from the frozen successor namespace"
            )
        if projection_file.relative_to(root).as_posix() != SUCCESSOR_PUBLIC_PROJECTION_PATH:
            raise OODExternalV2IntegrityError(
                "public projection path differs from the frozen successor namespace"
            )
        _require_git_ignored_and_untracked(
            root,
            SUCCESSOR_PRIVATE_INVENTORY_PATH,
            context="private external inventory",
        )
        projection_tracked = _run_git(
            root,
            "ls-files",
            "--",
            SUCCESSOR_PUBLIC_PROJECTION_PATH,
            allow_empty=True,
        )
        if projection_tracked.stdout.strip() != "":
            raise OODExternalV2IntegrityError(
                "public projection must be added only in the child-freeze commit"
            )
        if parent.seven_zip_tool_binding is None:
            raise OODExternalV2ConfigError(
                "successor parent has no frozen 7-Zip tool binding"
            )
        verify_seven_zip_tool_binding(requested_tool, parent.seven_zip_tool_binding)
        return inventory_file, projection_file, requested_tool

    inventory_file, projection_file, requested_tool = cast(
        tuple[Path, Path, Path],
        _child_freeze_preflight_stage(
            "x8_inventory_evidence",
            x8_inventory_evidence,
        ),
    )

    def decision_and_runtime_bindings() -> tuple[
        Mapping[str, BoundFile], Mapping[str, str]
    ]:
        return (
            _verify_child_freeze_decision_bindings(root),
            _freeze_child_runtime_bindings(root),
        )

    decision_bindings, runtime_bindings = cast(
        tuple[Mapping[str, BoundFile], Mapping[str, str]],
        _child_freeze_preflight_stage(
            "decision_and_runtime_bindings",
            decision_and_runtime_bindings,
        ),
    )

    def namespace_and_timestamp() -> tuple[
        datetime,
        Path,
        Path,
        Path,
        tuple[int, int, int, int],
        _OwnedDirectoryIdentity,
        _OwnedDirectoryIdentity,
    ]:
        frozen = _utc_datetime(frozen_at_utc, "frozen_at_utc")
        challenge_relative = _project_relative_existing_directory(
            root,
            challenge_root,
            context="Challenge extraction root",
        )
        zzu_relative = _project_relative_existing_directory(
            root,
            zzu_root,
            context="ZZU extraction root",
        )
        if {
            CHALLENGE_2011_DATASET: challenge_relative,
            ZZU_PEDIATRIC_DATASET: zzu_relative,
        } != dict(EXPECTED_DATASET_ROOTS):
            raise OODExternalV2IntegrityError(
                "child roots must equal the frozen extraction directories"
            )
        counts = (
            _positive_integer(challenge_records, "Challenge records"),
            _positive_integer(zzu_records, "ZZU records"),
            _positive_integer(zzu_patients, "ZZU patients"),
            _positive_integer(selected_records_total, "selected records"),
        )
        frozen_counts = parent.inventory_counts
        if frozen_counts is None or counts != (
            frozen_counts.challenge_records,
            frozen_counts.zzu_records,
            frozen_counts.zzu_patients,
            frozen_counts.total_records,
        ):
            raise OODExternalV2IntegrityError(
                "declared child counts differ from the frozen X8 inventory counts"
            )
        destination = _resolve_project_relative(
            root,
            _relative_path(
                Path(os.path.abspath(os.fspath(output_path)))
                .relative_to(root)
                .as_posix(),
                "child output path",
            ),
            require_file=False,
        )
        expected_destination = root.joinpath(
            *PurePosixPath(SUCCESSOR_CHILD_CONFIG_PATH).parts
        )
        if destination != expected_destination:
            raise OODExternalV2ConfigError(
                "child contract must use its exact frozen config destination"
            )
        if destination.exists() or _is_indirect(destination):
            raise OODExternalV2IntegrityError(
                "child contract destination must remain absent before X10"
            )
        output_root = _resolve_project_relative(
            root,
            parent.output_root,
            require_file=False,
        )
        claim = _resolve_project_relative(root, parent.claim_path, require_file=False)
        if (
            output_root.exists()
            or _is_indirect(output_root)
            or claim.exists()
            or _is_indirect(claim)
        ):
            raise OODExternalV2IntegrityError(
                "external claim or output root exists before child freeze"
            )
        attempt_path = _resolve_project_relative(
            root,
            SUCCESSOR_CHILD_FREEZE_ATTEMPT_PATH,
            require_file=False,
        )
        receipt_path = _resolve_project_relative(
            root,
            SUCCESSOR_CHILD_FREEZE_FAILURE_PATH,
            require_file=False,
        )
        if attempt_path.parent != receipt_path.parent:
            raise OODExternalV2ConfigError(
                "X10 protocol artifacts must share one parent directory"
            )
        return (
            frozen,
            root.joinpath(*PurePosixPath(challenge_relative).parts),
            root.joinpath(*PurePosixPath(zzu_relative).parts),
            destination,
            counts,
            _owned_directory_identity(destination.parent),
            _owned_directory_identity(attempt_path.parent),
        )

    (
        frozen,
        challenge_directory,
        zzu_directory,
        destination,
        declared_counts,
        output_parent_identity,
        protocol_artifact_parent_identity,
    ) = cast(
        tuple[
            datetime,
            Path,
            Path,
            Path,
            tuple[int, int, int, int],
            _OwnedDirectoryIdentity,
            _OwnedDirectoryIdentity,
        ],
        _child_freeze_preflight_stage(
            "namespace_and_timestamp",
            namespace_and_timestamp,
        ),
    )
    preflight = ChildFreezePreflight(
        status="CHILD_FREEZE_PREFLIGHT_VERIFIED",
        parent=parent,
        project_root=root,
        implementation_revision=revision,
        project_source_tree=project_source_tree,
        runtime_environment=runtime_environment,
        decision_bindings=decision_bindings,
        runtime_bindings=runtime_bindings,
        frozen_at_utc=frozen,
        inventory_path=inventory_file,
        public_projection_path=projection_file,
        challenge_root=challenge_directory,
        zzu_root=zzu_directory,
        declared_counts=declared_counts,
        output_path=destination,
        output_parent_identity=output_parent_identity,
        protocol_artifact_parent_identity=protocol_artifact_parent_identity,
        seven_zip_executable=requested_tool,
    )

    def closing_control_state() -> None:
        if _verify_clean_git_revision(root) != revision:
            raise OODExternalV2IntegrityError(
                "implementation revision changed during child-freeze preflight"
            )
        if _current_runtime_environment() != runtime_environment:
            raise OODExternalV2IntegrityError(
                "runtime changed during child-freeze preflight"
            )
        _verify_git_remote_state(root, expected_revision=revision)
        _verify_private_history_absent(root)
        _verify_project_source_tree_at_revisions(
            root,
            project_source_tree,
            implementation_revision=revision,
            execution_revision=None,
        )
        _verify_imported_project_module_origins(root, project_source_tree)
        _verify_tracked_head_blob(
            root,
            revision=revision,
            relative_path=SUCCESSOR_PARENT_CONFIG_PATH,
            expected_file_sha256=parent.file_sha256,
        )
        if (
            _verify_child_freeze_decision_bindings(root) != decision_bindings
            or _freeze_child_runtime_bindings(root) != runtime_bindings
        ):
            raise OODExternalV2IntegrityError(
                "decision or runtime bindings changed during child-freeze preflight"
            )
        verify_child_freeze_authorization_available(preflight, project_root=root)

    _child_freeze_preflight_stage("closing_control_state", closing_control_state)
    return preflight


def _verify_child_freeze_raw_source_bindings(
    root: Path,
    expected_bindings: Mapping[str, RawSourceBinding],
    *,
    source_access_witness: Callable[[], None] | None,
) -> dict[str, RawSourceBinding]:
    """Verify in frozen order and witness exactly the first successful read."""

    observed: dict[str, RawSourceBinding] = {}
    source_access_confirmed = False

    def witness_source_access_once() -> None:
        nonlocal source_access_confirmed
        if source_access_confirmed:
            return
        source_access_confirmed = True
        if source_access_witness is not None:
            source_access_witness()

    for name, expected_path in EXPECTED_RAW_SOURCE_PATHS.items():
        observed[name] = _raw_source_binding_for_path(
            root,
            expected_path,
            context=f"raw source {name}",
            official_md5=expected_bindings[name].official_md5,
            content_access_witness=witness_source_access_once,
        )
    return observed


def _freeze_external_v2_child_contract_after_x10_authorization(
    *,
    preflight: ChildFreezePreflight,
    stage_callback: Callable[[str], None] | None,
    source_access_witness: Callable[[], None] | None,
    child_visibility_witness: Callable[[], None] | None,
    child_publication_witness: Callable[[], None] | None,
    child_bytes_witness: Callable[[bytes], None],
) -> OODExternalV2ChildContract:
    root = preflight.project_root
    parent = preflight.parent
    revision = preflight.implementation_revision
    project_source_tree = preflight.project_source_tree
    runtime_environment = preflight.runtime_environment
    decisions = preflight.decision_bindings
    runtime_bindings = preflight.runtime_bindings
    inventory_path = preflight.inventory_path
    public_projection_path = preflight.public_projection_path
    challenge_root = preflight.challenge_root
    zzu_root = preflight.zzu_root
    challenge_records, zzu_records, zzu_patients, selected_records_total = (
        preflight.declared_counts
    )
    output_path = preflight.output_path
    seven_zip_executable = preflight.seven_zip_executable
    frozen_at_utc = preflight.frozen_at_utc.isoformat()
    _verify_historical_x9_child_freeze_artifacts(root)
    builder_attempt_file_sha256 = HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_FILE_SHA256
    builder_attempt_artifact_sha256 = (
        HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_ARTIFACT_SHA256
    )
    child_freeze_attempt_body = _child_freeze_attempt_body(preflight)
    child_freeze_attempt_file_sha256 = _verify_child_freeze_marker(
        _child_freeze_attempt_bytes(preflight),
        project_root=root,
    )
    child_freeze_attempt_artifact_sha256 = _digest(
        child_freeze_attempt_body.get("artifact_sha256"),
        "child freeze attempt artifact",
    )
    if stage_callback is not None:
        stage_callback("raw_source_binding_verification")

    inventory_file = _require_project_file(
        root,
        Path(os.path.abspath(os.fspath(inventory_path))),
        context="private external inventory",
    )
    if inventory_file.relative_to(root).as_posix() != SUCCESSOR_PRIVATE_INVENTORY_PATH:
        raise OODExternalV2IntegrityError(
            "private inventory path differs from the frozen successor namespace"
        )
    _require_git_ignored_and_untracked(
        root,
        inventory_file.relative_to(root).as_posix(),
        context="private external inventory",
    )
    _require_git_ignored_and_untracked(
        root,
        f"{parent.output_root}/private",
        context="private evidence output",
    )
    if sha256_file(inventory_file) != HISTORICAL_X8_PRIVATE_INVENTORY_FILE_SHA256:
        raise OODExternalV2IntegrityError(
            "private inventory file differs from the successful X8 output"
        )
    try:
        inventory = load_external_inventory(inventory_file)
    except Exception as error:
        raise OODExternalV2IntegrityError("private inventory is invalid") from error
    if inventory.inventory_sha256 != HISTORICAL_X8_PRIVATE_INVENTORY_ARTIFACT_SHA256:
        raise OODExternalV2IntegrityError(
            "private inventory artifact differs from the successful X8 output"
        )
    archive_summaries = _assert_production_archive_closures(
        inventory.archive_closures,
        expected_seven_zip_tool=parent.seven_zip_tool_binding,
    )
    observed_challenge = sum(
        record.dataset == CHALLENGE_2011_DATASET for record in inventory.records
    )
    observed_zzu = sum(record.dataset == ZZU_PEDIATRIC_DATASET for record in inventory.records)
    observed_patients = len(
        {
            record.patient_key
            for record in inventory.records
            if record.dataset == ZZU_PEDIATRIC_DATASET and record.patient_key is not None
        }
    )
    declared_counts = (
        _positive_integer(challenge_records, "Challenge records"),
        _positive_integer(zzu_records, "ZZU records"),
        _positive_integer(zzu_patients, "ZZU patients"),
        _positive_integer(selected_records_total, "selected records"),
    )
    if declared_counts != (
        observed_challenge,
        observed_zzu,
        observed_patients,
        len(inventory.records),
    ):
        raise OODExternalV2IntegrityError(
            "declared child counts differ from the private inventory"
        )
    frozen_counts = parent.inventory_counts
    if frozen_counts is None or (
        observed_challenge,
        observed_zzu,
        observed_patients,
        len(inventory.records),
    ) != (
        frozen_counts.challenge_records,
        frozen_counts.zzu_records,
        frozen_counts.zzu_patients,
        frozen_counts.total_records,
    ):
        raise OODExternalV2IntegrityError(
            "private inventory counts differ from the frozen successor parent"
        )
    if observed_challenge != parent.challenge_expected_records:
        raise OODExternalV2IntegrityError("private inventory is not complete Challenge Set A")

    requested_roots = {
        CHALLENGE_2011_DATASET: _project_relative_existing_directory(
            root,
            challenge_root,
            context="Challenge extraction root",
        ),
        ZZU_PEDIATRIC_DATASET: _project_relative_existing_directory(
            root,
            zzu_root,
            context="ZZU extraction root",
        ),
    }
    if requested_roots != dict(EXPECTED_DATASET_ROOTS):
        raise OODExternalV2IntegrityError(
            "child roots must equal the two exact frozen extraction directories"
        )

    projection_file = _require_project_file(
        root,
        Path(os.path.abspath(os.fspath(public_projection_path))),
        context="public inventory projection",
    )
    if projection_file.relative_to(root).as_posix() != SUCCESSOR_PUBLIC_PROJECTION_PATH:
        raise OODExternalV2IntegrityError(
            "public projection path differs from the frozen successor namespace"
        )
    projection_tracked = _run_git(
        root,
        "ls-files",
        "--",
        SUCCESSOR_PUBLIC_PROJECTION_PATH,
        allow_empty=True,
    )
    if projection_tracked.stdout.strip() != "":
        raise OODExternalV2IntegrityError(
            "public projection must be added only in the child-freeze commit"
        )
    projection_sha256 = _verify_public_projection_file(
        projection_file,
        inventory=inventory,
        challenge_records=observed_challenge,
        zzu_records=observed_zzu,
        expected_counts=frozen_counts,
    )
    if (
        sha256_file(projection_file) != HISTORICAL_X8_PUBLIC_PROJECTION_FILE_SHA256
        or projection_sha256 != HISTORICAL_X8_PUBLIC_PROJECTION_ARTIFACT_SHA256
    ):
        raise OODExternalV2IntegrityError(
            "public projection differs from the successful X8 output"
        )
    if _current_runtime_environment() != runtime_environment:
        raise OODExternalV2IntegrityError(
            "runtime differs from the child-freeze preflight"
        )
    if parent.raw_source_bindings is None:
        raise OODExternalV2ConfigError(
            "executable successor parent has no frozen raw-source provenance table"
        )
    raw_bindings = _verify_child_freeze_raw_source_bindings(
        root,
        parent.raw_source_bindings,
        source_access_witness=source_access_witness,
    )
    if raw_bindings != dict(parent.raw_source_bindings):
        raise OODExternalV2IntegrityError(
            "installed raw source bytes differ from successor-parent provenance"
        )
    _verify_archive_closure_rebuilds(
        inventory,
        dataset_roots=MappingProxyType(
            {
                dataset: _resolve_project_relative(
                    root,
                    relative_path,
                    require_directory=True,
                )
                for dataset, relative_path in requested_roots.items()
            }
        ),
        raw_source_paths=MappingProxyType(
            {
                name: _resolve_project_relative(
                    root,
                    binding.relative_path,
                    require_file=True,
                )
                for name, binding in raw_bindings.items()
            }
        ),
        seven_zip_executable=seven_zip_executable,
        expected_seven_zip_tool=parent.seven_zip_tool_binding,
        stage_callback=stage_callback,
    )
    if stage_callback is not None:
        stage_callback("decision_and_child_materialization")
    if (
        _verify_child_freeze_decision_bindings(root) != decisions
        or _freeze_child_runtime_bindings(root) != runtime_bindings
    ):
        raise OODExternalV2IntegrityError(
            "decision or runtime bindings changed before child materialization"
        )
    child_body: dict[str, object] = {
        "artifact_type": CHILD_CONTRACT_ARTIFACT_TYPE,
        "dataset_roots": requested_roots,
        "decision_bindings": {
            name: _bound_file_dict(binding) for name, binding in decisions.items()
        },
        "child_freeze_attempt": {
            "artifact_sha256": child_freeze_attempt_artifact_sha256,
            "file_sha256": child_freeze_attempt_file_sha256,
            "relative_path": SUCCESSOR_CHILD_FREEZE_ATTEMPT_PATH,
        },
        "frozen_at_utc": _utc_datetime(frozen_at_utc, "frozen_at_utc")
        .astimezone(UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        "implementation_revision": revision,
        "inventory": {
            "archive_closures": [
                _archive_closure_summary_dict(summary)
                for summary in archive_summaries
            ],
            "challenge_records": observed_challenge,
            "file_sha256": sha256_file(inventory_file),
            "inventory_sha256": inventory.inventory_sha256,
            "relative_path": inventory_file.relative_to(root).as_posix(),
            "selected_records_total": len(inventory.records),
            "zzu_patients": observed_patients,
            "zzu_records": observed_zzu,
        },
        "inventory_builder_attempt": {
            "artifact_sha256": builder_attempt_artifact_sha256,
            "file_sha256": builder_attempt_file_sha256,
            "relative_path": HISTORICAL_X8_INVENTORY_BUILDER_ATTEMPT_PATH,
        },
        "output_root": parent.output_root,
        "parent_config_file_sha256": parent.file_sha256,
        "project_source_tree": _project_source_tree_dict(project_source_tree),
        "protocol_id": PROTOCOL_ID,
        "public_inventory_projection": {
            "artifact_sha256": projection_sha256,
            "file_sha256": sha256_file(projection_file),
            "relative_path": projection_file.relative_to(root).as_posix(),
        },
        "raw_source_bindings": {
            name: {
                "file_sha256": binding.file_sha256,
                "official_md5": binding.official_md5,
                "relative_path": binding.relative_path,
                "size_bytes": binding.size_bytes,
            }
            for name, binding in raw_bindings.items()
        },
        "runtime_bindings": runtime_bindings,
        "runtime_environment": _runtime_environment_dict(runtime_environment),
        "schema_version": 1,
    }
    destination = _resolve_project_relative(
        root,
        _relative_path(
            Path(os.path.abspath(os.fspath(output_path))).relative_to(root).as_posix(),
            "child output path",
        ),
        require_file=False,
    )
    expected_destination = root.joinpath(*PurePosixPath(SUCCESSOR_CHILD_CONFIG_PATH).parts)
    if destination != expected_destination:
        raise OODExternalV2ConfigError(
            "child contract must use its exact frozen config destination"
        )
    child_bytes = child_contract_bytes(child_body)
    _load_child_contract_bytes(child_bytes, source=destination)
    child_bytes_witness(child_bytes)
    if stage_callback is not None:
        stage_callback("prepublication_control_reverification")
    _verify_project_source_tree_at_revisions(
        root,
        project_source_tree,
        implementation_revision=revision,
        execution_revision=None,
    )
    _verify_imported_project_module_origins(root, project_source_tree)
    _verify_tracked_head_blob(
        root,
        revision=revision,
        relative_path=SUCCESSOR_PARENT_CONFIG_PATH,
        expected_file_sha256=parent.file_sha256,
    )
    if (
        _verify_child_freeze_decision_bindings(root) != decisions
        or _freeze_child_runtime_bindings(root) != runtime_bindings
    ):
        raise OODExternalV2IntegrityError(
            "decision or runtime bindings changed before child publication"
        )
    if _current_runtime_environment() != runtime_environment:
        raise OODExternalV2IntegrityError(
            "runtime environment changed during child freeze"
        )
    _verify_git_remote_state(root, expected_revision=revision)
    _verify_private_history_absent(root)
    _verify_historical_x9_child_freeze_artifacts(root)
    if _verify_historical_x8_inventory_builder_evidence(root) != (
        builder_attempt_file_sha256
    ):
        raise OODExternalV2IntegrityError(
            "inventory builder attempt changed during child freeze"
        )
    if _verify_child_freeze_marker(
        _child_freeze_attempt_bytes(preflight),
        project_root=root,
    ) != child_freeze_attempt_file_sha256:
        raise OODExternalV2IntegrityError(
            "child freeze attempt marker changed during child freeze"
        )
    failure_receipt = _resolve_project_relative(
        root,
        SUCCESSOR_CHILD_FREEZE_FAILURE_PATH,
        require_file=False,
    )
    if failure_receipt.exists() or _is_indirect(failure_receipt):
        raise OODExternalV2IntegrityError(
            "child freeze failure receipt exists before child publication"
        )
    if (
        sha256_file(inventory_file) != HISTORICAL_X8_PRIVATE_INVENTORY_FILE_SHA256
        or inventory.inventory_sha256
        != HISTORICAL_X8_PRIVATE_INVENTORY_ARTIFACT_SHA256
        or sha256_file(projection_file)
        != HISTORICAL_X8_PUBLIC_PROJECTION_FILE_SHA256
        or projection_sha256
        != HISTORICAL_X8_PUBLIC_PROJECTION_ARTIFACT_SHA256
    ):
        raise OODExternalV2IntegrityError(
            "successful X8 inventory evidence changed before child publication"
        )
    if destination.exists() or _is_indirect(destination):
        raise OODExternalV2IntegrityError(
            "child contract destination was occupied before publication"
        )
    if stage_callback is not None:
        stage_callback("child_publication")
    _atomic_write_new(
        destination,
        child_bytes,
        visibility_witness=child_visibility_witness,
        publication_witness=child_publication_witness,
        expected_parent_identity=preflight.output_parent_identity,
    )
    if stage_callback is not None:
        stage_callback("child_reload_and_postflight")
    child = load_child_contract(destination)
    _verify_child_inventory_builder_attempt(parent, child, project_root=root)
    return child


def _exact_file_state(path: Path, expected_bytes: bytes) -> str:
    """Return a bounded, path-free state for an immutable protocol artifact."""

    try:
        if _is_indirect(path):
            return "UNVERIFIABLE"
        metadata = path.lstat()
    except FileNotFoundError:
        return "ABSENT"
    except OSError:
        return "UNVERIFIABLE"
    if not path.is_file() or metadata.st_size > _CHILD_MAX_BYTES:
        return "UNVERIFIABLE"
    try:
        observed = _read_bounded(path, _CHILD_MAX_BYTES, "protocol artifact")
    except BaseException:
        return "UNVERIFIABLE"
    return "EXACT" if observed == expected_bytes else "UNVERIFIABLE"


def freeze_external_v2_child_contract(
    *,
    parent_path: str | Path,
    project_root: str | Path,
    inventory_path: str | Path,
    public_projection_path: str | Path,
    implementation_revision: str,
    frozen_at_utc: str,
    challenge_root: str | Path,
    zzu_root: str | Path,
    challenge_records: int,
    zzu_records: int,
    zzu_patients: int,
    selected_records_total: int,
    output_path: str | Path,
    seven_zip_executable: str | Path = "7z",
    stage_callback: Callable[[str], None] | None = None,
    source_access_witness: Callable[[], None] | None = None,
    child_visibility_witness: Callable[[], None] | None = None,
    child_publication_witness: Callable[[], None] | None = None,
) -> OODExternalV2ChildContract:
    """Consume X10 once and freeze the child without scientific evaluation."""

    for callback, context in (
        (stage_callback, "stage_callback"),
        (source_access_witness, "source_access_witness"),
        (child_visibility_witness, "child_visibility_witness"),
        (child_publication_witness, "child_publication_witness"),
    ):
        if callback is not None and not callable(callback):
            raise TypeError(f"{context} must be callable or None")
    # Preserve the original-v2 hard refusal before the successor transaction.
    initial_root = _strict_project_root(project_root)
    initial_parent = _load_parent_for_operation(parent_path, project_root=initial_root)
    assert_external_v2_parent_executable(initial_parent)
    preflight = verify_child_freeze_preflight(
        parent_path=parent_path,
        project_root=project_root,
        inventory_path=inventory_path,
        public_projection_path=public_projection_path,
        implementation_revision=implementation_revision,
        frozen_at_utc=frozen_at_utc,
        challenge_root=challenge_root,
        zzu_root=zzu_root,
        challenge_records=challenge_records,
        zzu_records=zzu_records,
        zzu_patients=zzu_patients,
        selected_records_total=selected_records_total,
        output_path=output_path,
        seven_zip_executable=seven_zip_executable,
    )
    root = preflight.project_root
    current_stage = "authorization_publication"
    authorization_visible = False
    official_source_content_accessed = False
    child_visible = False
    child_durable = False
    expected_child_bytes: bytes | None = None

    def transition(stage: str) -> None:
        nonlocal current_stage
        if stage not in CHILD_FREEZE_ATTEMPT_STAGES:
            raise OODExternalV2IntegrityError("child freeze stage is invalid")
        if CHILD_FREEZE_ATTEMPT_STAGES.index(stage) < (
            CHILD_FREEZE_ATTEMPT_STAGES.index(current_stage)
        ):
            raise OODExternalV2IntegrityError("child freeze stage order regressed")
        if stage != current_stage:
            current_stage = stage
            if stage_callback is not None:
                stage_callback(stage)

    def mark_authorization_visible() -> None:
        nonlocal authorization_visible
        authorization_visible = True

    def mark_source_accessed() -> None:
        nonlocal official_source_content_accessed
        if official_source_content_accessed:
            return
        official_source_content_accessed = True
        if source_access_witness is not None:
            source_access_witness()

    def mark_child_visible() -> None:
        nonlocal child_visible
        child_visible = True
        if child_visibility_witness is not None:
            child_visibility_witness()

    def mark_child_durable() -> None:
        nonlocal child_durable
        child_durable = True
        if child_publication_witness is not None:
            child_publication_witness()

    def remember_child_bytes(payload: bytes) -> None:
        nonlocal expected_child_bytes
        expected_child_bytes = payload

    try:
        if stage_callback is not None:
            stage_callback(current_stage)
        consume_child_freeze_authorization(
            preflight,
            project_root=root,
            visibility_witness=mark_authorization_visible,
        )
    except BaseException:
        marker_state = _exact_file_state(
            _resolve_project_relative(
                root,
                SUCCESSOR_CHILD_FREEZE_ATTEMPT_PATH,
                require_file=False,
            ),
            _child_freeze_attempt_bytes(preflight),
        )
        if marker_state == "ABSENT":
            raise ChildFreezePreflightStageError("closing_control_state") from None
        if marker_state != "EXACT":
            raise OODExternalV2ExecutionError(
                "X10 child freeze authorization state is unverifiable"
            ) from None
        authorization_visible = True
        failure_receipt_visible = False
        failure_receipt_durable = False

        def mark_failure_receipt_visible() -> None:
            nonlocal failure_receipt_visible
            failure_receipt_visible = True

        def mark_failure_receipt_durable() -> None:
            nonlocal failure_receipt_durable
            failure_receipt_durable = True

        try:
            record_child_freeze_failure(
                preflight,
                project_root=root,
                failure_stage=current_stage,
                reason="PUBLICATION_FAILED_AFTER_VISIBILITY",
                official_source_content_accessed=False,
                output_state="NONE",
                visibility_witness=mark_failure_receipt_visible,
                publication_witness=mark_failure_receipt_durable,
            )
            failure_receipt_written = (
                failure_receipt_visible and failure_receipt_durable
            )
        except BaseException:
            failure_receipt_written = failure_receipt_visible
        raise ChildFreezeAttemptError(
            stage=current_stage,
            reason="PUBLICATION_FAILED_AFTER_VISIBILITY",
            output_state="NONE",
            official_source_content_accessed=False,
            failure_receipt_written=failure_receipt_written,
        ) from None

    try:
        return _freeze_external_v2_child_contract_after_x10_authorization(
            preflight=preflight,
            stage_callback=transition,
            source_access_witness=mark_source_accessed,
            child_visibility_witness=mark_child_visible,
            child_publication_witness=mark_child_durable,
            child_bytes_witness=remember_child_bytes,
        )
    except BaseException as error:
        if not authorization_visible:
            raise OODExternalV2ExecutionError(
                "X10 child freeze authorization state is unverifiable"
            ) from None
        output_state: str
        if expected_child_bytes is None:
            try:
                child_state = (
                    "ABSENT"
                    if not preflight.output_path.exists()
                    and not _is_indirect(preflight.output_path)
                    else "UNVERIFIABLE"
                )
            except OSError:
                child_state = "UNVERIFIABLE"
        else:
            child_state = _exact_file_state(
                preflight.output_path,
                expected_child_bytes,
            )
        if child_state == "ABSENT":
            output_state = "NONE"
        elif child_state == "EXACT" and child_durable:
            output_state = "DURABLE_EXACT"
        elif child_state == "EXACT" or child_visible:
            output_state = "VISIBLE_EXACT_DURABILITY_UNCONFIRMED"
        else:
            output_state = "PRESENT_UNVERIFIABLE"
        if current_stage == "child_publication":
            if output_state != "NONE" and not child_visible:
                reason = "DESTINATION_PREEXISTED"
            else:
                reason = (
                    "PUBLICATION_FAILED_BEFORE_VISIBILITY"
                    if output_state == "NONE"
                    else "PUBLICATION_FAILED_AFTER_VISIBILITY"
                )
        elif current_stage == "child_reload_and_postflight":
            reason = "POSTPUBLICATION_RELOAD_REFUSED"
        elif isinstance(
            error,
            (
                OODExternalV2ConfigError,
                OODExternalV2IntegrityError,
                OODExternalV2ExecutionError,
            ),
        ):
            reason = "STAGE_REFUSED"
        else:
            reason = "UNEXPECTED_INTERNAL_FAILURE"
        failure_receipt_visible = False
        failure_receipt_durable = False

        def mark_failure_receipt_visible() -> None:
            nonlocal failure_receipt_visible
            failure_receipt_visible = True

        def mark_failure_receipt_durable() -> None:
            nonlocal failure_receipt_durable
            failure_receipt_durable = True

        try:
            record_child_freeze_failure(
                preflight,
                project_root=root,
                failure_stage=current_stage,
                reason=reason,
                official_source_content_accessed=official_source_content_accessed,
                output_state=output_state,
                visibility_witness=mark_failure_receipt_visible,
                publication_witness=mark_failure_receipt_durable,
            )
            failure_receipt_written = (
                failure_receipt_visible and failure_receipt_durable
            )
        except BaseException:
            failure_receipt_written = failure_receipt_visible
        raise ChildFreezeAttemptError(
            stage=current_stage,
            reason=reason,
            output_state=output_state,
            official_source_content_accessed=official_source_content_accessed,
            failure_receipt_written=failure_receipt_written,
        ) from None


class _OneShotClaimState:
    def __init__(self, claim_bytes: bytes, owner_nonce: str) -> None:
        if _OWNER_NONCE.fullmatch(owner_nonce) is None:
            raise OODExternalV2IntegrityError("external claim owner nonce is invalid")
        self.claim_bytes = claim_bytes
        self.owner_nonce = owner_nonce
        self.published = False

    def mark_published(self) -> None:
        self.published = True


@dataclass(frozen=True, slots=True)
class _OwnedDirectoryIdentity:
    device: int
    inode: int


def _owned_directory_identity(path: Path) -> _OwnedDirectoryIdentity:
    direct = _assert_direct_ancestry(path, context="owned evidence directory")
    if not direct.is_dir():
        raise OODExternalV2ExecutionError("owned evidence directory is unavailable")
    try:
        before = direct.stat()
        tuple(direct.iterdir())
        after = direct.stat()
    except OSError as error:
        raise OODExternalV2ExecutionError(
            "owned evidence directory cannot be identified"
        ) from error
    result = _OwnedDirectoryIdentity(before.st_dev, before.st_ino)
    if (
        _OwnedDirectoryIdentity(after.st_dev, after.st_ino) != result
        or _assert_direct_ancestry(
            direct,
            context="owned evidence directory",
        )
        != direct
    ):
        raise OODExternalV2ExecutionError(
            "owned evidence directory changed during identification"
        )
    return result


def _verify_owned_namespace_parent(
    path: Path,
    *,
    expected_identity: _OwnedDirectoryIdentity,
) -> None:
    direct = _assert_direct_ancestry(path, context="external protocol namespace parent")
    if _owned_directory_identity(direct) != expected_identity:
        raise OODExternalV2ExecutionError(
            "external protocol namespace parent identity changed"
        )


def _verify_owned_evidence_directory(
    path: Path,
    *,
    expected_identity: _OwnedDirectoryIdentity,
    expected_marker_bytes: bytes,
) -> None:
    if _owned_directory_identity(path) != expected_identity:
        raise OODExternalV2ExecutionError(
            "evidence directory identity differs from this execution"
        )
    marker_path = path / ACCESS_MARKER_FILENAME
    if _read_bounded(
        marker_path,
        _ACCESS_RECORD_MAX_BYTES,
        "owned external marker",
    ) != expected_marker_bytes:
        raise OODExternalV2ExecutionError(
            "evidence directory owner marker differs from this execution"
        )


class _OutputRootOwnershipState:
    """Track this process's output rename; never infer ownership from existence."""

    def __init__(self, identity: _OwnedDirectoryIdentity) -> None:
        self.identity = identity
        self.visible = False

    def mark_visible(self) -> None:
        self.visible = True


class _TerminalManifestState:
    """Track manifest link visibility separately from directory durability."""

    def __init__(self) -> None:
        self.visible = False

    def mark_visible(self) -> None:
        self.visible = True


def _verify_immediate_execution_controls(
    inputs: VerifiedExternalV2Inputs,
    *,
    execution_revision: str,
) -> None:
    root = inputs.project_root
    _verify_revision_boundary(
        root,
        child=inputs.child,
        execution_revision=execution_revision,
    )
    _verify_private_history_absent(root)
    _verify_tracked_head_blob(
        root,
        revision=execution_revision,
        relative_path=SUCCESSOR_PARENT_CONFIG_PATH,
        expected_file_sha256=inputs.parent.file_sha256,
    )
    _verify_project_source_tree_at_revisions(
        root,
        inputs.child.project_source_tree,
        implementation_revision=inputs.child.implementation_revision,
        execution_revision=execution_revision,
    )
    _verify_imported_project_module_origins(root, inputs.child.project_source_tree)
    if _current_runtime_environment() != inputs.child.runtime_environment:
        raise OODExternalV2IntegrityError(
            "active runtime changed before the one-shot claim"
        )
    for relative_path, expected_hash in inputs.child.runtime_bindings.items():
        if sha256_file(
            _resolve_project_relative(root, relative_path, require_file=True)
        ) != expected_hash:
            raise OODExternalV2IntegrityError(
                "runtime-critical source changed before the one-shot claim"
            )
    successor = verify_successor_parent_preflight(
        inputs.parent.path,
        project_root=root,
    )
    if successor.file_sha256 != inputs.parent.file_sha256:
        raise OODExternalV2IntegrityError(
            "successor/predecessor boundary changed before the one-shot claim"
        )
    _verify_child_inventory_builder_attempt(
        inputs.parent,
        inputs.child,
        project_root=root,
    )


def prepare_ood_external_v2(
    *,
    parent_path: str | Path,
    child_path: str | Path,
    project_root: str | Path,
    code_revision: str,
    seven_zip_executable: str | Path = "7z",
) -> OODV2Result:
    """Execute a future feasible successor once into an immutable evidence root.

    The preserved original v2 parent is rejected before the child contract,
    output root, staging directory, access marker, or adjacent claim is read or
    created.  The remaining path is retained as the version-neutral execution
    engine to wire only after a successor parent is separately frozen.
    """

    root = _strict_project_root(project_root)
    parent = _load_parent_for_operation(parent_path, project_root=root)
    assert_external_v2_parent_executable(parent)
    child = load_child_contract(child_path)
    revision = _revision(code_revision, "execution code revision")
    inputs = verify_external_v2_inputs(
        parent,
        child,
        project_root=root,
        code_revision=revision,
        seven_zip_executable=seven_zip_executable,
    )
    output_root = _resolve_project_relative(
        inputs.project_root,
        parent.output_root,
        require_file=False,
    )
    claim_path = _resolve_project_relative(
        inputs.project_root,
        parent.claim_path,
        require_file=False,
    )
    if output_root.parent != claim_path.parent:
        raise OODExternalV2IntegrityError(
            "output and claim do not share the frozen protocol namespace parent"
        )
    namespace_parent = _assert_direct_ancestry(
        output_root.parent,
        context="external protocol namespace parent",
    )
    if not namespace_parent.is_dir():
        raise OODExternalV2IntegrityError(
            "external protocol namespace parent is unavailable"
        )
    namespace_parent_identity = _owned_directory_identity(namespace_parent)
    if output_root.exists() or _is_indirect(output_root):
        raise OODExternalV2ExecutionError("immutable external v2 output root exists")
    if claim_path.exists() or _is_indirect(claim_path):
        raise OODExternalV2ExecutionError(
            "external one-shot claim already exists; retry is forbidden"
        )
    _assert_no_marked_staging_retry(output_root)

    model, normalization, runtime, model_state_before = _load_model_and_runtime(inputs)
    staging = _create_durable_staging_directory(
        output_root,
        expected_parent_identity=namespace_parent_identity,
    )
    owned_directory_identity = _owned_directory_identity(staging)
    owner_nonce = secrets.token_hex(32)
    claim_bytes = _external_claim_bytes(inputs, owner_nonce=owner_nonce)
    claim_state = _OneShotClaimState(claim_bytes, owner_nonce)
    output_ownership = _OutputRootOwnershipState(owned_directory_identity)
    terminal_manifest = _TerminalManifestState()
    try:
        claim_hash = sha256_bytes(claim_bytes)
        marker_bytes = _external_marker_bytes(
            inputs,
            owner_nonce=owner_nonce,
            claim_file_sha256=claim_hash,
        )
        _verify_immediate_execution_controls(
            inputs,
            execution_revision=revision,
        )
        # The durable armed marker is intentionally committed before the
        # adjacent claim.  No adapter is called before both bytes reverify.
        _atomic_write_new(
            staging / ACCESS_MARKER_FILENAME,
            marker_bytes,
            expected_parent_identity=owned_directory_identity,
            ownership_verifier=lambda: _verify_owned_namespace_parent(
                namespace_parent,
                expected_identity=namespace_parent_identity,
            ),
        )
        _atomic_write_new(
            claim_path,
            claim_bytes,
            visibility_witness=claim_state.mark_published,
            expected_parent_identity=namespace_parent_identity,
            ownership_verifier=lambda: _verify_owned_namespace_parent(
                namespace_parent,
                expected_identity=namespace_parent_identity,
            ),
        )
        if _read_bounded(claim_path, _ACCESS_RECORD_MAX_BYTES, "external claim") != (
            claim_bytes
        ):
            raise OODExternalV2IntegrityError("published external claim bytes changed")
        if _read_bounded(
            staging / ACCESS_MARKER_FILENAME,
            _ACCESS_RECORD_MAX_BYTES,
            "external marker",
        ) != marker_bytes:
            raise OODExternalV2IntegrityError("external marker bytes changed")

        evaluated = _evaluate_external_records(
            inputs,
            model=model,
            normalization=normalization,
            runtime=runtime,
            model_state_before=model_state_before,
        )
        endpoints = _build_endpoint_evidence(evaluated, inputs=inputs)
        post = verify_external_v2_inputs(
            parent,
            child,
            project_root=inputs.project_root,
            code_revision=revision,
            seven_zip_executable=seven_zip_executable,
        )
        v1_unchanged = post.v1.snapshots == inputs.v1.snapshots
        inventory_unchanged = (
            post.inventory == inputs.inventory
            and sha256_file(post.inventory_path) == child.inventory.file_sha256
            and all(
                sha256_file(post.raw_source_paths[name])
                == child.raw_source_bindings[name].file_sha256
                for name in REQUIRED_RAW_SOURCE_BINDING_KEYS
            )
        )
        if not v1_unchanged or not inventory_unchanged:
            raise OODExternalV2IntegrityError(
                "frozen v1 or external source bytes changed during execution"
            )
        _write_private_artifacts(
            staging,
            inputs=inputs,
            evaluated=evaluated,
            endpoints=endpoints,
        )
        _verify_raw_to_canonical_replay(
            staging / "private",
            inputs=inputs,
            expected_records=evaluated.records,
        )
        replayed_embeddings = _replay_quality_pass_embeddings(
            staging / "private",
            inputs=inputs,
            expected_records=evaluated.records,
            model=model,
            normalization=normalization,
            runtime=runtime,
        )
        _verify_embedding_bundle_semantics(
            staging / "private",
            inputs=inputs,
            quality_pass_rows=tuple(
                row
                for row in evaluated.records
                if row.quality_status == QualityStatus.PASS.value
            ),
            frozen_model=model,
            frozen_runtime=runtime,
            replayed_embeddings=replayed_embeddings,
        )
        result = _build_result(
            evaluated,
            endpoints,
            inputs=inputs,
            code_revision=revision,
            v1_unchanged=v1_unchanged,
            inventory_unchanged=inventory_unchanged,
            raw_source_to_canonical_replay_verified=True,
            full_backbone_embedding_replay_verified=True,
        )
        result_path = staging / OOD_V2_RESULT_FILENAME
        _atomic_write_new(result_path, ood_v2_result_json_bytes(result))
        _verify_staged_members_before_manifest(
            staging,
            inputs=inputs,
            expected_result=result,
            expected_claim_bytes=claim_bytes,
            model=model,
            normalization=normalization,
            runtime=runtime,
        )
        _commit_staged_directory(
            staging,
            output_root,
            visibility_witness=output_ownership.mark_visible,
            expected_directory_identity=output_ownership.identity,
            expected_marker_bytes=marker_bytes,
            expected_parent_identity=namespace_parent_identity,
        )
        manifest = build_success_manifest(
            output_root,
            parent_config_file_sha256=parent.file_sha256,
            child_contract_file_sha256=child.file_sha256,
            child_contract_artifact_sha256=child.artifact_sha256,
            inventory_sha256=inputs.inventory.inventory_sha256,
            result_artifact_sha256=result.artifact_sha256,
            external_claim_file_sha256=claim_hash,
            code_revision=revision,
        )
        # Parse and self-verify the exact bytes before the terminal link.
        type(manifest).from_bytes(manifest.to_bytes())
        preverified = preverify_external_v2_bundle(
            output_root,
            manifest,
            claim_path=claim_path,
            project_root=inputs.project_root,
            seven_zip_executable=seven_zip_executable,
        )
        if (
            preverified.result != result
            or preverified.success_manifest != manifest
            or preverified.claim_file_sha256 != claim_hash
        ):
            raise OODExternalV2IntegrityError(
                "terminal bundle preverification returned inconsistent evidence"
            )

        def verify_terminal_ownership() -> None:
            _verify_owned_namespace_parent(
                namespace_parent,
                expected_identity=namespace_parent_identity,
            )
            _verify_owned_evidence_directory(
                output_root,
                expected_identity=output_ownership.identity,
                expected_marker_bytes=marker_bytes,
            )
            _verify_runtime_scratch_empty(inputs.project_root)

        # Terminal successful write: no output-root member may change after it.
        _atomic_write_terminal_success(
            output_root,
            manifest.to_bytes(),
            visibility_witness=terminal_manifest.mark_visible,
            ownership_verifier=verify_terminal_ownership,
        )
        reloaded = verify_external_v2_bundle(
            output_root,
            claim_path=claim_path,
            project_root=inputs.project_root,
            seven_zip_executable=seven_zip_executable,
        )
        if (
            reloaded.result != result
            or reloaded.success_manifest != manifest
            or reloaded.claim_file_sha256 != claim_hash
        ):
            raise OODExternalV2IntegrityError(
                "independent post-publication bundle reread differs"
            )
        # The terminal verifier itself must not have populated any launcher
        # scratch location. A visible manifest followed by a failure here is
        # retained as an ambiguous terminal commit, never returned as success.
        verify_terminal_ownership()
        return reloaded.result
    except BaseException as error:
        if claim_state.published:
            _retain_postclaim_failure(
                staging=staging,
                output_root=output_root,
                inputs=inputs,
                code_revision=revision,
                error=error,
                output_root_owned=output_ownership.visible,
                terminal_manifest_visible=terminal_manifest.visible,
                external_claim_file_sha256=sha256_bytes(claim_state.claim_bytes),
                owner_nonce=claim_state.owner_nonce,
                expected_directory_identity=output_ownership.identity,
                expected_marker_bytes=marker_bytes,
                expected_parent_identity=namespace_parent_identity,
            )
        elif not (staging / ACCESS_MARKER_FILENAME).exists():
            _remove_staging_root(staging, expected_parent=output_root.parent)
        # An armed marker without an owned claim is retained for forensic
        # review and blocks any retry under this output root.
        raise


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise OODExternalV2ConfigError(f"{context} must be a string-keyed mapping")
    return {str(key): item for key, item in value.items()}


def _digest(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise OODExternalV2ConfigError(f"{context} must be a prefixed SHA-256 digest")
    return value


def _revision(value: object, context: str) -> str:
    if not isinstance(value, str) or _GIT_REVISION.fullmatch(value.casefold()) is None:
        raise OODExternalV2ConfigError(f"{context} must be a full Git revision")
    return value.casefold()


def _positive_integer(value: object, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise OODExternalV2ConfigError(f"{context} must be a positive integer")
    return value


def _nonnegative_integer(value: object, context: str) -> int:
    if type(value) is not int or value < 0:
        raise OODExternalV2ConfigError(f"{context} must be a non-negative integer")
    return value


def _exact_integer(value: object, expected: int, context: str) -> int:
    observed = _positive_integer(value, context)
    if observed != expected:
        raise OODExternalV2ConfigError(f"{context} differs from the frozen value")
    return observed


def _exact_float(value: object, context: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise OODExternalV2ConfigError(f"{context} must be a finite JSON float")
    return value


def _exact_string(value: object, expected: str, context: str) -> str:
    if value != expected:
        raise OODExternalV2ConfigError(f"{context} differs from the frozen value")
    return expected


def _relative_path(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise OODExternalV2ConfigError(f"{context} must be canonical POSIX text")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise OODExternalV2ConfigError(f"{context} must be project-relative")
    if posix.as_posix() != value or any(part in {"", ".", ".."} for part in posix.parts):
        raise OODExternalV2ConfigError(f"{context} contains unsafe path segments")
    return value


def _bound_file(
    value: object,
    context: str,
    *,
    require_artifact: bool = False,
) -> BoundFile:
    mapping = _mapping(value, context)
    path_key = "path" if "path" in mapping else "relative_path"
    if path_key not in mapping or "file_sha256" not in mapping:
        raise OODExternalV2ConfigError(f"{context} is missing its file binding")
    artifact_value = mapping.get("artifact_sha256")
    if require_artifact and artifact_value is None:
        raise OODExternalV2ConfigError(f"{context} must bind an artifact identity")
    return BoundFile(
        relative_path=_relative_path(mapping[path_key], f"{context} path"),
        file_sha256=_digest(mapping["file_sha256"], f"{context} file"),
        artifact_sha256=(
            None
            if artifact_value is None
            else _digest(artifact_value, f"{context} artifact")
        ),
    )


def _raw_source_binding(value: object, context: str) -> RawSourceBinding:
    mapping = _mapping(value, context)
    if set(mapping) != {
        "file_sha256",
        "official_md5",
        "relative_path",
        "size_bytes",
    }:
        raise OODExternalV2ConfigError(f"{context} fields differ from protocol")
    raw_md5 = mapping["official_md5"]
    if raw_md5 is not None and (
        not isinstance(raw_md5, str) or _MD5.fullmatch(raw_md5) is None
    ):
        raise OODExternalV2ConfigError(f"{context} official MD5 is invalid")
    return RawSourceBinding(
        relative_path=_relative_path(mapping["relative_path"], f"{context} path"),
        file_sha256=_digest(mapping["file_sha256"], f"{context} file"),
        size_bytes=_positive_integer(mapping["size_bytes"], f"{context} size"),
        official_md5=raw_md5,
    )


def _runtime_package_tree_binding(
    value: object,
    *,
    context: str,
) -> RuntimePackageTreeBinding:
    payload = _mapping(value, context)
    if set(payload) != {
        "distribution",
        "file_count",
        "import_roots",
        "total_bytes",
        "tree_sha256",
        "version",
    }:
        raise OODExternalV2ConfigError(f"{context} fields differ from protocol")
    distribution = payload["distribution"]
    version = payload["version"]
    if (
        not isinstance(distribution, str)
        or distribution not in EXPECTED_SCIENTIFIC_PACKAGE_VERSIONS
        or not isinstance(version, str)
        or version != EXPECTED_SCIENTIFIC_PACKAGE_VERSIONS[distribution]
    ):
        raise OODExternalV2ConfigError(f"{context} identity differs")
    raw_roots = payload["import_roots"]
    expected_roots = EXPECTED_SCIENTIFIC_PACKAGE_IMPORT_ROOTS[distribution]
    if (
        not isinstance(raw_roots, list)
        or tuple(raw_roots) != expected_roots
        or any(not isinstance(root, str) for root in raw_roots)
    ):
        raise OODExternalV2ConfigError(f"{context} import roots differ")
    return RuntimePackageTreeBinding(
        distribution=distribution,
        version=version,
        import_roots=expected_roots,
        file_count=_positive_integer(payload["file_count"], f"{context} file count"),
        total_bytes=_positive_integer(payload["total_bytes"], f"{context} total bytes"),
        tree_sha256=_digest(payload["tree_sha256"], f"{context} tree"),
    )


def _runtime_filesystem_tree_binding(
    value: object,
    *,
    context: str,
    expected_kind: str,
) -> RuntimeFilesystemTreeBinding:
    payload = _mapping(value, context)
    if set(payload) != {
        "directory_count",
        "file_count",
        "total_bytes",
        "tree_kind",
        "tree_sha256",
    }:
        raise OODExternalV2ConfigError(f"{context} fields differ from protocol")
    if payload["tree_kind"] != expected_kind:
        raise OODExternalV2ConfigError(f"{context} kind differs from protocol")
    return RuntimeFilesystemTreeBinding(
        tree_kind=expected_kind,
        file_count=_positive_integer(payload["file_count"], f"{context} file count"),
        directory_count=_positive_integer(
            payload["directory_count"],
            f"{context} directory count",
        ),
        total_bytes=_positive_integer(payload["total_bytes"], f"{context} bytes"),
        tree_sha256=_digest(payload["tree_sha256"], f"{context} tree"),
    )


def _runtime_filesystem_tree_dict(
    value: RuntimeFilesystemTreeBinding,
) -> dict[str, object]:
    return {
        "directory_count": value.directory_count,
        "file_count": value.file_count,
        "total_bytes": value.total_bytes,
        "tree_kind": value.tree_kind,
        "tree_sha256": value.tree_sha256,
    }


def _git_tool_binding(value: object, *, context: str) -> GitToolBinding:
    payload = _mapping(value, context)
    if set(payload) != {
        "executable_name",
        "executable_sha256",
        "executable_size_bytes",
        "launcher_name",
        "launcher_sha256",
        "launcher_size_bytes",
        "runtime_tree",
        "version",
    }:
        raise OODExternalV2ConfigError(f"{context} fields differ from protocol")
    result = GitToolBinding(
        version=_exact_string(payload["version"], EXPECTED_GIT_VERSION, context),
        launcher_name=_exact_string(
            payload["launcher_name"],
            EXPECTED_GIT_LAUNCHER_NAME,
            context,
        ),
        launcher_size_bytes=_exact_integer(
            payload["launcher_size_bytes"],
            EXPECTED_GIT_LAUNCHER_SIZE_BYTES,
            context,
        ),
        launcher_sha256=_digest(payload["launcher_sha256"], context),
        executable_name=_exact_string(
            payload["executable_name"],
            EXPECTED_GIT_EXECUTABLE_NAME,
            context,
        ),
        executable_size_bytes=_exact_integer(
            payload["executable_size_bytes"],
            EXPECTED_GIT_EXECUTABLE_SIZE_BYTES,
            context,
        ),
        executable_sha256=_digest(payload["executable_sha256"], context),
        runtime_tree=_runtime_filesystem_tree_binding(
            payload["runtime_tree"],
            context=f"{context} runtime tree",
            expected_kind=RUNTIME_FILESYSTEM_TREE_KINDS[2],
        ),
    )
    if (
        result.launcher_sha256 != EXPECTED_GIT_LAUNCHER_SHA256
        or result.executable_sha256 != EXPECTED_GIT_EXECUTABLE_SHA256
        or result.runtime_tree.file_count != EXPECTED_GIT_RUNTIME_FILE_COUNT
        or result.runtime_tree.directory_count
        != EXPECTED_GIT_RUNTIME_DIRECTORY_COUNT
        or result.runtime_tree.total_bytes != EXPECTED_GIT_RUNTIME_TOTAL_BYTES
        or result.runtime_tree.tree_sha256 != EXPECTED_GIT_RUNTIME_TREE_SHA256
    ):
        raise OODExternalV2ConfigError(f"{context} executable identity differs")
    return result


def _git_tool_dict(value: GitToolBinding) -> dict[str, object]:
    return {
        "executable_name": value.executable_name,
        "executable_sha256": value.executable_sha256,
        "executable_size_bytes": value.executable_size_bytes,
        "launcher_name": value.launcher_name,
        "launcher_sha256": value.launcher_sha256,
        "launcher_size_bytes": value.launcher_size_bytes,
        "runtime_tree": _runtime_filesystem_tree_dict(value.runtime_tree),
        "version": value.version,
    }


def _nvidia_driver_tool_binding(
    value: object,
    *,
    context: str,
) -> NvidiaDriverToolBinding:
    payload = _mapping(value, context)
    expected_fields = {
        "driver_version",
        "nvcuda_name",
        "nvcuda_sha256",
        "nvcuda_size_bytes",
        "nvidia_smi_name",
        "nvidia_smi_sha256",
        "nvidia_smi_size_bytes",
        "nvml_name",
        "nvml_sha256",
        "nvml_size_bytes",
    }
    if set(payload) != expected_fields:
        raise OODExternalV2ConfigError(f"{context} fields differ from protocol")
    result = NvidiaDriverToolBinding(
        driver_version=_exact_string(
            payload["driver_version"],
            EXPECTED_NVIDIA_DRIVER_VERSION,
            context,
        ),
        nvidia_smi_name=_exact_string(
            payload["nvidia_smi_name"], "nvidia-smi.exe", context
        ),
        nvidia_smi_size_bytes=_exact_integer(
            payload["nvidia_smi_size_bytes"],
            EXPECTED_NVIDIA_DRIVER_FILES["nvidia-smi.exe"][0],
            context,
        ),
        nvidia_smi_sha256=_digest(payload["nvidia_smi_sha256"], context),
        nvml_name=_exact_string(payload["nvml_name"], "nvml.dll", context),
        nvml_size_bytes=_exact_integer(
            payload["nvml_size_bytes"],
            EXPECTED_NVIDIA_DRIVER_FILES["nvml.dll"][0],
            context,
        ),
        nvml_sha256=_digest(payload["nvml_sha256"], context),
        nvcuda_name=_exact_string(payload["nvcuda_name"], "nvcuda.dll", context),
        nvcuda_size_bytes=_exact_integer(
            payload["nvcuda_size_bytes"],
            EXPECTED_NVIDIA_DRIVER_FILES["nvcuda.dll"][0],
            context,
        ),
        nvcuda_sha256=_digest(payload["nvcuda_sha256"], context),
    )
    if (
        result.nvidia_smi_sha256
        != EXPECTED_NVIDIA_DRIVER_FILES["nvidia-smi.exe"][1]
        or result.nvml_sha256 != EXPECTED_NVIDIA_DRIVER_FILES["nvml.dll"][1]
        or result.nvcuda_sha256 != EXPECTED_NVIDIA_DRIVER_FILES["nvcuda.dll"][1]
    ):
        raise OODExternalV2ConfigError(f"{context} bytes differ from protocol")
    return result


def _nvidia_driver_tool_dict(
    value: NvidiaDriverToolBinding,
) -> dict[str, object]:
    return {
        "driver_version": value.driver_version,
        "nvcuda_name": value.nvcuda_name,
        "nvcuda_sha256": value.nvcuda_sha256,
        "nvcuda_size_bytes": value.nvcuda_size_bytes,
        "nvidia_smi_name": value.nvidia_smi_name,
        "nvidia_smi_sha256": value.nvidia_smi_sha256,
        "nvidia_smi_size_bytes": value.nvidia_smi_size_bytes,
        "nvml_name": value.nvml_name,
        "nvml_sha256": value.nvml_sha256,
        "nvml_size_bytes": value.nvml_size_bytes,
    }


def _project_source_file_dict(value: ProjectSourceFileBinding) -> dict[str, object]:
    return {
        "file_sha256": value.file_sha256,
        "relative_path": value.relative_path,
        "size_bytes": value.size_bytes,
    }


def _project_source_tree_sha256(
    files: tuple[ProjectSourceFileBinding, ...],
) -> str:
    body = {
        "files": [_project_source_file_dict(value) for value in files],
        "schema_version": 1,
    }
    return "sha256:" + hashlib.sha256(
        b"ecg_trust.ood_external_v2_1.project_source_tree.v1\x00"
        + canonical_json_bytes(body)[:-1]
    ).hexdigest()


def _project_source_tree_dict(value: ProjectSourceTreeBinding) -> dict[str, object]:
    return {
        "file_count": value.file_count,
        "files": [_project_source_file_dict(item) for item in value.files],
        "total_bytes": value.total_bytes,
        "tree_sha256": value.tree_sha256,
    }


def _project_source_tree_binding(
    value: object,
    *,
    context: str,
) -> ProjectSourceTreeBinding:
    payload = _mapping(value, context)
    if set(payload) != {"file_count", "files", "total_bytes", "tree_sha256"}:
        raise OODExternalV2ConfigError(f"{context} fields differ from protocol")
    raw_files = payload["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise OODExternalV2ConfigError(f"{context} files must be non-empty")
    files: list[ProjectSourceFileBinding] = []
    for index, raw_file in enumerate(raw_files):
        item = _mapping(raw_file, f"{context} file {index}")
        if set(item) != {"file_sha256", "relative_path", "size_bytes"}:
            raise OODExternalV2ConfigError(
                f"{context} file {index} fields differ from protocol"
            )
        files.append(
            ProjectSourceFileBinding(
                relative_path=_relative_path(
                    item["relative_path"],
                    f"{context} file path",
                ),
                size_bytes=_positive_integer(
                    item["size_bytes"],
                    f"{context} file size",
                ),
                file_sha256=_digest(
                    item["file_sha256"],
                    f"{context} file hash",
                ),
            )
        )
    frozen_files = tuple(files)
    paths = tuple(item.relative_path for item in frozen_files)
    if (
        paths != tuple(sorted(paths))
        or len(paths) != len(set(paths))
        or len({path.casefold() for path in paths}) != len(paths)
        or not set(PROJECT_OPERATIONAL_ENTRYPOINTS).issubset(paths)
        or any(
            path not in PROJECT_OPERATIONAL_ENTRYPOINTS
            and not (
                path.startswith(f"{PROJECT_SOURCE_ROOT}/") and path.endswith(".py")
            )
            for path in paths
        )
    ):
        raise OODExternalV2ConfigError(f"{context} paths differ from protocol")
    result = ProjectSourceTreeBinding(
        files=frozen_files,
        file_count=_positive_integer(payload["file_count"], f"{context} file count"),
        total_bytes=_positive_integer(payload["total_bytes"], f"{context} bytes"),
        tree_sha256=_digest(payload["tree_sha256"], f"{context} tree"),
    )
    if (
        result.file_count != len(result.files)
        or result.total_bytes != sum(item.size_bytes for item in result.files)
        or result.tree_sha256 != _project_source_tree_sha256(result.files)
    ):
        raise OODExternalV2ConfigError(f"{context} aggregate differs")
    return result


def _runtime_environment_binding(
    value: object,
    *,
    context: str,
) -> RuntimeEnvironmentBinding:
    mapping = _mapping(value, context)
    expected = {
        "dont_write_bytecode",
        "git_tool",
        "isolated_mode",
        "no_site",
        "nvidia_driver_tool",
        "numpy_version",
        "package_trees",
        "pycache_prefix_verified_empty",
        "python_base_tree",
        "python_base_alias_name",
        "python_base_target_name",
        "python_environment_sha256",
        "python_executable_file_sha256",
        "python_executable_size_bytes",
        "python_implementation",
        "python_version",
        "pyvenv_config_file_sha256",
        "pyvenv_config_size_bytes",
        "scipy_version",
        "site_packages_tree",
        "sys_path_layout",
        "user_site_disabled",
        "wfdb_version",
    }
    if set(mapping) != expected:
        raise OODExternalV2ConfigError(f"{context} fields differ from protocol")
    raw_package_trees = mapping["package_trees"]
    if not isinstance(raw_package_trees, list):
        raise OODExternalV2ConfigError(f"{context} package trees must be an array")
    package_trees = tuple(
        _runtime_package_tree_binding(
            value,
            context=f"{context} package tree {index}",
        )
        for index, value in enumerate(raw_package_trees)
    )
    if tuple(item.distribution for item in package_trees) != tuple(
        sorted(EXPECTED_SCIENTIFIC_PACKAGE_VERSIONS)
    ):
        raise OODExternalV2ConfigError(
            f"{context} package trees differ from the exact frozen set"
        )
    raw_sys_path_layout = mapping["sys_path_layout"]
    if (
        not isinstance(raw_sys_path_layout, list)
        or tuple(raw_sys_path_layout) != EXPECTED_RUNTIME_SYS_PATH_LAYOUT
        or any(not isinstance(item, str) for item in raw_sys_path_layout)
    ):
        raise OODExternalV2ConfigError(f"{context} sys.path layout differs")
    boolean_values: dict[str, bool] = {}
    for key in (
        "dont_write_bytecode",
        "isolated_mode",
        "no_site",
        "pycache_prefix_verified_empty",
        "user_site_disabled",
    ):
        raw_boolean = mapping[key]
        if type(raw_boolean) is not bool:
            raise OODExternalV2ConfigError(f"{context} {key} must be bool")
        boolean_values[key] = raw_boolean
    text_values: dict[str, str] = {}
    for key in (
        "numpy_version",
        "python_implementation",
        "python_base_alias_name",
        "python_base_target_name",
        "python_version",
        "scipy_version",
        "wfdb_version",
    ):
        raw = mapping[key]
        if not isinstance(raw, str) or not raw or raw != raw.strip():
            raise OODExternalV2ConfigError(f"{context} {key} must be canonical text")
        text_values[key] = raw
    result = RuntimeEnvironmentBinding(
        python_implementation=text_values["python_implementation"],
        python_version=text_values["python_version"],
        python_executable_file_sha256=_digest(
            mapping["python_executable_file_sha256"],
            f"{context} Python executable",
        ),
        python_executable_size_bytes=_positive_integer(
            mapping["python_executable_size_bytes"],
            f"{context} Python executable size",
        ),
        python_base_alias_name=_exact_string(
            mapping["python_base_alias_name"],
            EXPECTED_PYTHON_BASE_ALIAS_NAME,
            context,
        ),
        python_base_target_name=_exact_string(
            mapping["python_base_target_name"],
            EXPECTED_PYTHON_BASE_TARGET_NAME,
            context,
        ),
        python_environment_sha256=_digest(
            mapping["python_environment_sha256"],
            f"{context} Python environment",
        ),
        numpy_version=text_values["numpy_version"],
        scipy_version=text_values["scipy_version"],
        wfdb_version=text_values["wfdb_version"],
        package_trees=package_trees,
        python_base_tree=_runtime_filesystem_tree_binding(
            mapping["python_base_tree"],
            context=f"{context} CPython base tree",
            expected_kind=RUNTIME_FILESYSTEM_TREE_KINDS[0],
        ),
        site_packages_tree=_runtime_filesystem_tree_binding(
            mapping["site_packages_tree"],
            context=f"{context} site-packages tree",
            expected_kind=RUNTIME_FILESYSTEM_TREE_KINDS[1],
        ),
        pyvenv_config_file_sha256=_digest(
            mapping["pyvenv_config_file_sha256"],
            f"{context} pyvenv.cfg",
        ),
        pyvenv_config_size_bytes=_positive_integer(
            mapping["pyvenv_config_size_bytes"],
            f"{context} pyvenv.cfg size",
        ),
        isolated_mode=boolean_values["isolated_mode"],
        no_site=boolean_values["no_site"],
        dont_write_bytecode=boolean_values["dont_write_bytecode"],
        user_site_disabled=boolean_values["user_site_disabled"],
        pycache_prefix_verified_empty=boolean_values[
            "pycache_prefix_verified_empty"
        ],
        sys_path_layout=tuple(cast(list[str], raw_sys_path_layout)),
        git_tool=_git_tool_binding(
            mapping["git_tool"],
            context=f"{context} Git tool",
        ),
        nvidia_driver_tool=_nvidia_driver_tool_binding(
            mapping["nvidia_driver_tool"],
            context=f"{context} NVIDIA driver tool",
        ),
    )
    if (
        result.python_implementation != "CPython"
        or not result.python_version.startswith("3.12.")
        or result.numpy_version != EXPECTED_SCIENTIFIC_PACKAGE_VERSIONS["numpy"]
        or result.scipy_version != EXPECTED_SCIENTIFIC_PACKAGE_VERSIONS["scipy"]
        or result.wfdb_version != EXPECTED_SCIENTIFIC_PACKAGE_VERSIONS["wfdb"]
        or not result.isolated_mode
        or not result.no_site
        or not result.dont_write_bytecode
        or not result.user_site_disabled
        or not result.pycache_prefix_verified_empty
    ):
        raise OODExternalV2ConfigError(
            f"{context} differs from the frozen scientific runtime"
        )
    return result


def _runtime_environment_dict(value: RuntimeEnvironmentBinding) -> dict[str, object]:
    return {
        "dont_write_bytecode": value.dont_write_bytecode,
        "git_tool": _git_tool_dict(value.git_tool),
        "isolated_mode": value.isolated_mode,
        "no_site": value.no_site,
        "nvidia_driver_tool": _nvidia_driver_tool_dict(value.nvidia_driver_tool),
        "numpy_version": value.numpy_version,
        "package_trees": [
            {
                "distribution": item.distribution,
                "file_count": item.file_count,
                "import_roots": list(item.import_roots),
                "total_bytes": item.total_bytes,
                "tree_sha256": item.tree_sha256,
                "version": item.version,
            }
            for item in value.package_trees
        ],
        "pycache_prefix_verified_empty": value.pycache_prefix_verified_empty,
        "python_base_tree": _runtime_filesystem_tree_dict(value.python_base_tree),
        "python_base_alias_name": value.python_base_alias_name,
        "python_base_target_name": value.python_base_target_name,
        "python_environment_sha256": value.python_environment_sha256,
        "python_executable_file_sha256": value.python_executable_file_sha256,
        "python_executable_size_bytes": value.python_executable_size_bytes,
        "python_implementation": value.python_implementation,
        "python_version": value.python_version,
        "pyvenv_config_file_sha256": value.pyvenv_config_file_sha256,
        "pyvenv_config_size_bytes": value.pyvenv_config_size_bytes,
        "scipy_version": value.scipy_version,
        "site_packages_tree": _runtime_filesystem_tree_dict(
            value.site_packages_tree
        ),
        "sys_path_layout": list(value.sys_path_layout),
        "user_site_disabled": value.user_site_disabled,
        "wfdb_version": value.wfdb_version,
    }


def _distribution_version(
    distribution_name: str,
    distribution: importlib.metadata.Distribution,
) -> str:
    try:
        observed = distribution.version
    except Exception:
        observed = None
    if isinstance(observed, str) and observed:
        return observed
    module_names = {
        "PyYAML": "yaml",
        "pydantic": "pydantic",
        "pydantic-core": "pydantic_core",
    }
    module_name = module_names.get(distribution_name)
    if module_name is None:
        raise OODExternalV2IntegrityError(
            "installed distribution version metadata is unavailable"
        )
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        raise OODExternalV2IntegrityError(
            "installed distribution module cannot be imported"
        ) from error
    fallback = getattr(module, "__version__", None)
    if not isinstance(fallback, str) or not fallback:
        raise OODExternalV2IntegrityError(
            "installed distribution fallback version is unavailable"
        )
    return fallback


def _installed_distribution_tree_binding(
    distribution_name: str,
) -> RuntimePackageTreeBinding:
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as error:
        raise OODExternalV2IntegrityError(
            "required installed distribution is unavailable"
        ) from error
    version = _distribution_version(distribution_name, distribution)
    if version != EXPECTED_SCIENTIFIC_PACKAGE_VERSIONS[distribution_name]:
        raise OODExternalV2IntegrityError(
            "installed distribution version differs from the frozen runtime"
        )
    raw_files = distribution.files
    if raw_files is None:
        raise OODExternalV2IntegrityError(
            "installed distribution has no auditable file inventory"
        )
    expected_roots = EXPECTED_SCIENTIFIC_PACKAGE_IMPORT_ROOTS[distribution_name]
    observed_roots: set[str] = set()
    entries: list[dict[str, object]] = []
    seen: set[str] = set()
    seen_casefolded: set[str] = set()
    for package_path in sorted(raw_files, key=lambda value: str(value).replace("\\", "/")):
        relative = str(package_path).replace("\\", "/")
        posix = PurePosixPath(relative)
        matching_root = next(
            (
                root
                for root in expected_roots
                if posix.parts
                and (
                    posix.parts[0] == root
                    or (
                        len(posix.parts) == 1
                        and posix.parts[0].startswith(f"{root}.")
                    )
                )
            ),
            None,
        )
        # Bind only runtime import roots. Distribution metadata, licenses,
        # entry-point scripts, bytecode caches, and pyc files are excluded.
        if (
            matching_root is None
            or
            posix.is_absolute()
            or any(part in {"", ".", ".."} for part in posix.parts)
            or posix.as_posix() != relative
            or "__pycache__" in posix.parts
            or posix.suffix.casefold() in {".pyc", ".pyo"}
        ):
            continue
        observed_roots.add(matching_root)
        if relative in seen or relative.casefold() in seen_casefolded:
            raise OODExternalV2IntegrityError(
                "installed distribution file inventory contains duplicates"
            )
        seen.add(relative)
        seen_casefolded.add(relative.casefold())
        path = Path(os.fspath(cast(Any, distribution.locate_file(package_path))))
        if _is_indirect(path) or not path.is_file():
            raise OODExternalV2IntegrityError(
                "installed distribution tree contains a missing or indirect file"
            )
        try:
            before = path.stat()
            digest = sha256_file(path)
            after = path.stat()
        except OSError as error:
            raise OODExternalV2IntegrityError(
                "installed distribution file cannot be inspected"
            ) from error
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise OODExternalV2IntegrityError(
                "installed distribution file changed while hashing"
            )
        entries.append(
            {
                "relative_path": relative,
                "sha256": digest,
                "size_bytes": before.st_size,
            }
        )
    if not entries or observed_roots != set(expected_roots):
        raise OODExternalV2IntegrityError(
            "installed distribution runtime import roots are incomplete"
        )
    body = {
        "distribution": distribution_name,
        "files": entries,
        "import_roots": list(expected_roots),
        "schema_version": 1,
        "version": version,
    }
    tree_sha256 = "sha256:" + hashlib.sha256(
        b"ecg_trust.runtime_distribution_tree.v1\x00"
        + canonical_json_bytes(body)[:-1]
    ).hexdigest()
    return RuntimePackageTreeBinding(
        distribution=distribution_name,
        version=version,
        import_roots=expected_roots,
        file_count=len(entries),
        total_bytes=sum(cast(int, item["size_bytes"]) for item in entries),
        tree_sha256=tree_sha256,
    )


def _assert_direct_ancestry(path: Path, *, context: str) -> Path:
    """Resolve a path only after rejecting symlink and junction components."""

    lexical = Path(os.path.abspath(os.fspath(path)))
    try:
        for component in (lexical, *lexical.parents):
            if _is_indirect(component):
                raise OODExternalV2IntegrityError(
                    f"{context} contains an indirect filesystem component"
                )
        resolved = lexical.resolve(strict=True)
    except OODExternalV2IntegrityError:
        raise
    except OSError as error:
        raise OODExternalV2IntegrityError(f"{context} is unavailable") from error
    if resolved != lexical:
        raise OODExternalV2IntegrityError(
            f"{context} does not resolve to its exact lexical path"
        )
    return resolved


def _stable_runtime_file_entry(path: Path, *, context: str) -> dict[str, object]:
    direct = _assert_direct_ancestry(path, context=context)
    if not direct.is_file():
        raise OODExternalV2IntegrityError(f"{context} is not a regular file")
    try:
        before = direct.stat()
        digest = hashlib.sha256()
        size = 0
        with direct.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        after = direct.stat()
    except OSError as error:
        raise OODExternalV2IntegrityError(f"{context} cannot be hashed") from error
    if (
        _is_indirect(direct)
        or size != before.st_size
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or getattr(before, "st_ino", None) != getattr(after, "st_ino", None)
    ):
        raise OODExternalV2IntegrityError(f"{context} changed while being hashed")
    return {
        "sha256": "sha256:" + digest.hexdigest(),
        "size_bytes": size,
    }


def _runtime_tree_members(root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    directories: list[str] = ["."]
    files: list[str] = []
    seen_casefolded = {"."}
    for current_text, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_text)
        _assert_direct_ancestry(current, context="runtime tree directory")
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            directory = current / name
            _assert_direct_ancestry(directory, context="runtime tree directory")
            if not directory.is_dir():
                raise OODExternalV2IntegrityError(
                    "runtime tree contains a non-directory entry"
                )
            relative = directory.relative_to(root).as_posix()
            if relative.casefold() in seen_casefolded:
                raise OODExternalV2IntegrityError(
                    "runtime tree contains a case-insensitive path collision"
                )
            seen_casefolded.add(relative.casefold())
            directories.append(relative)
        for name in file_names:
            path = current / name
            _assert_direct_ancestry(path, context="runtime tree file")
            if not path.is_file():
                raise OODExternalV2IntegrityError(
                    "runtime tree contains a non-regular file"
                )
            relative = path.relative_to(root).as_posix()
            if relative.casefold() in seen_casefolded:
                raise OODExternalV2IntegrityError(
                    "runtime tree contains a case-insensitive path collision"
                )
            seen_casefolded.add(relative.casefold())
            files.append(relative)
    return tuple(directories), tuple(files)


def _runtime_filesystem_tree(
    root_path: Path,
    *,
    tree_kind: str,
) -> RuntimeFilesystemTreeBinding:
    if tree_kind not in RUNTIME_FILESYSTEM_TREE_KINDS:
        raise OODExternalV2IntegrityError("runtime tree kind is unsupported")
    root = _assert_direct_ancestry(root_path, context=f"{tree_kind} root")
    if not root.is_dir():
        raise OODExternalV2IntegrityError(f"{tree_kind} root is not a directory")
    directories, file_names = _runtime_tree_members(root)
    if not file_names:
        raise OODExternalV2IntegrityError(f"{tree_kind} contains no files")
    file_entries: list[dict[str, object]] = []
    for relative in file_names:
        entry = _stable_runtime_file_entry(
            root.joinpath(*PurePosixPath(relative).parts),
            context=f"{tree_kind} file",
        )
        file_entries.append({"relative_path": relative, **entry})
    directories_after, files_after = _runtime_tree_members(root)
    if directories_after != directories or files_after != file_names:
        raise OODExternalV2IntegrityError(f"{tree_kind} changed while being bound")
    body = {
        "directories": list(directories),
        "files": file_entries,
        "schema_version": 1,
        "tree_kind": tree_kind,
    }
    return RuntimeFilesystemTreeBinding(
        tree_kind=tree_kind,
        file_count=len(file_entries),
        directory_count=len(directories),
        total_bytes=sum(cast(int, item["size_bytes"]) for item in file_entries),
        tree_sha256="sha256:"
        + hashlib.sha256(
            b"ecg_trust.ood_external_v2_1.runtime_filesystem_tree.v1\x00"
            + canonical_json_bytes(body)[:-1]
        ).hexdigest(),
    )


def _runtime_path_layout(
    *,
    python_base: Path,
    site_packages: Path,
    project_src: Path,
) -> tuple[str, ...]:
    version = f"python{sys.version_info.major}{sys.version_info.minor}.zip"
    roles_by_path = {
        Path(os.path.abspath(os.fspath(python_base / version))): "cpython_zip",
        Path(os.path.abspath(os.fspath(python_base / "DLLs"))): "cpython_dlls",
        Path(os.path.abspath(os.fspath(python_base / "Lib"))): "cpython_stdlib",
        python_base: "cpython_base",
        site_packages: "venv_site_packages",
        project_src: "project_src",
    }
    observed: list[str] = []
    for raw_path in sys.path:
        if not raw_path:
            raise OODExternalV2IntegrityError(
                "isolated runtime sys.path contains the current directory"
            )
        lexical = Path(os.path.abspath(raw_path))
        role = roles_by_path.get(lexical)
        if role is None:
            raise OODExternalV2IntegrityError(
                "isolated runtime sys.path contains an unbound location"
            )
        if not role.startswith("cpython_"):
            _assert_direct_ancestry(lexical, context=f"runtime sys.path {role}")
        observed.append(role)
    result = tuple(observed)
    if result != EXPECTED_RUNTIME_SYS_PATH_LAYOUT:
        raise OODExternalV2IntegrityError(
            "isolated runtime sys.path order differs from the frozen layout"
        )
    return result


def _current_git_tool_binding() -> GitToolBinding:
    launcher, executable, install_root = _git_executable_paths()
    try:
        completed = subprocess.run(
            [os.fspath(executable), "--version"],
            check=False,
            capture_output=True,
            env=_sanitized_git_environment(executable),
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise OODExternalV2IntegrityError("frozen Git version probe failed") from error
    try:
        version = completed.stdout.decode("ascii", errors="strict").strip()
    except UnicodeError as error:
        raise OODExternalV2IntegrityError("Git version output is not ASCII") from error
    if completed.returncode != 0 or completed.stderr or version != EXPECTED_GIT_VERSION:
        raise OODExternalV2IntegrityError("frozen Git version differs")
    launcher_entry = _stable_runtime_file_entry(launcher, context="Git launcher")
    executable_entry = _stable_runtime_file_entry(executable, context="Git executable")
    return GitToolBinding(
        version=version,
        launcher_name=launcher.name,
        launcher_size_bytes=cast(int, launcher_entry["size_bytes"]),
        launcher_sha256=cast(str, launcher_entry["sha256"]),
        executable_name=executable.name,
        executable_size_bytes=cast(int, executable_entry["size_bytes"]),
        executable_sha256=cast(str, executable_entry["sha256"]),
        runtime_tree=_runtime_filesystem_tree(
            install_root / "mingw64",
            tree_kind=RUNTIME_FILESYSTEM_TREE_KINDS[2],
        ),
    )


def _nvidia_driver_tool_paths() -> tuple[Path, Path, Path]:
    system_root = _assert_direct_ancestry(
        Path(os.environ.get("SYSTEMROOT", r"C:\Windows")),
        context="Windows system root",
    )
    system32 = _assert_direct_ancestry(
        system_root / "System32",
        context="Windows System32",
    )
    paths = tuple(
        _assert_direct_ancestry(system32 / name, context=f"NVIDIA driver file {name}")
        for name in ("nvidia-smi.exe", "nvml.dll", "nvcuda.dll")
    )
    for path in paths:
        expected_size, expected_hash = EXPECTED_NVIDIA_DRIVER_FILES[path.name]
        entry = _stable_runtime_file_entry(path, context=f"NVIDIA driver file {path.name}")
        if entry["size_bytes"] != expected_size or entry["sha256"] != expected_hash:
            raise OODExternalV2IntegrityError(
                "NVIDIA driver file differs from the frozen runtime"
            )
    return cast(tuple[Path, Path, Path], paths)


def _current_nvidia_driver_tool_binding() -> NvidiaDriverToolBinding:
    nvidia_smi, nvml, nvcuda = _nvidia_driver_tool_paths()
    for name in ("ProgramFiles", "ProgramW6432"):
        if os.environ.get(name) != r"C:\Program Files":
            raise OODExternalV2IntegrityError(
                "NVIDIA driver environment differs from the frozen Windows layout"
            )
    environment = {
        name: value
        for name in (
            "COMSPEC",
            "ProgramFiles",
            "ProgramW6432",
            "SYSTEMROOT",
            "TEMP",
            "TMP",
            "WINDIR",
        )
        if (value := os.environ.get(name))
    }
    environment["PATH"] = os.fspath(nvidia_smi.parent)
    try:
        completed = subprocess.run(
            [
                os.fspath(nvidia_smi),
                "--id=0",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            cwd=nvidia_smi.parent,
            env=environment,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise OODExternalV2IntegrityError("bound NVIDIA driver query failed") from error
    try:
        values = tuple(
            line.strip()
            for line in completed.stdout.decode("ascii", errors="strict").splitlines()
            if line.strip()
        )
    except UnicodeError as error:
        raise OODExternalV2IntegrityError("NVIDIA driver query is not ASCII") from error
    if (
        completed.returncode != 0
        or completed.stderr
        or values != (EXPECTED_NVIDIA_DRIVER_VERSION,)
    ):
        raise OODExternalV2IntegrityError("bound NVIDIA driver version differs")
    entries = {
        path.name: _stable_runtime_file_entry(path, context="NVIDIA driver file")
        for path in (nvidia_smi, nvml, nvcuda)
    }
    return NvidiaDriverToolBinding(
        driver_version=values[0],
        nvidia_smi_name=nvidia_smi.name,
        nvidia_smi_size_bytes=cast(int, entries[nvidia_smi.name]["size_bytes"]),
        nvidia_smi_sha256=cast(str, entries[nvidia_smi.name]["sha256"]),
        nvml_name=nvml.name,
        nvml_size_bytes=cast(int, entries[nvml.name]["size_bytes"]),
        nvml_sha256=cast(str, entries[nvml.name]["sha256"]),
        nvcuda_name=nvcuda.name,
        nvcuda_size_bytes=cast(int, entries[nvcuda.name]["size_bytes"]),
        nvcuda_sha256=cast(str, entries[nvcuda.name]["sha256"]),
    )


def _resolve_python_base_runtime() -> tuple[Path, Path]:
    alias = Path(os.path.abspath(sys.base_prefix))
    alias_parent = _assert_direct_ancestry(
        alias.parent,
        context="CPython alias parent",
    )
    junction = getattr(alias, "is_junction", None)
    try:
        is_expected_junction = bool(junction is not None and junction())
        target = alias.resolve(strict=True)
    except OSError as error:
        raise OODExternalV2IntegrityError("CPython base alias is unavailable") from error
    if (
        alias.name != EXPECTED_PYTHON_BASE_ALIAS_NAME
        or alias_parent.name != ".python"
        or not is_expected_junction
        or alias.is_symlink()
        or target.parent != alias_parent
        or target.name != EXPECTED_PYTHON_BASE_TARGET_NAME
    ):
        raise OODExternalV2IntegrityError(
            "CPython base alias does not match the frozen in-project junction"
        )
    direct_target = _assert_direct_ancestry(
        target,
        context="resolved CPython base runtime",
    )
    if not direct_target.is_dir():
        raise OODExternalV2IntegrityError("resolved CPython base runtime is unavailable")
    return alias, direct_target


def _frozen_runtime_environment_material(
    runtime_root: Path,
    *,
    project_root: Path,
) -> tuple[tuple[str, str], ...]:
    """Return the exact launcher-sanitized environment without publishing it."""

    direct_runtime_root = _assert_direct_ancestry(
        runtime_root,
        context="isolated launcher runtime root",
    )
    expected_parent = _assert_direct_ancestry(
        project_root / "artifacts" / "trust_sentinel",
        context="isolated launcher runtime parent",
    )
    if (
        direct_runtime_root.parent != expected_parent
        or re.fullmatch(r"\.ood_external_v2_1\.runtime-[0-9a-f]{64}", direct_runtime_root.name)
        is None
    ):
        raise OODExternalV2IntegrityError(
            "isolated launcher runtime root is outside its frozen namespace"
        )
    runtime_roles = {
        "APPDATA": (direct_runtime_root / "home" / "AppData" / "Roaming", "runtime_roaming"),
        "LOCALAPPDATA": (
            direct_runtime_root / "home" / "AppData" / "Local",
            "runtime_local",
        ),
        "TEMP": (direct_runtime_root / "temp", "runtime_temp"),
        "TMP": (direct_runtime_root / "temp", "runtime_temp"),
        "TORCHINDUCTOR_CACHE_DIR": (
            direct_runtime_root / "temp",
            "runtime_temp",
        ),
        "USERPROFILE": (direct_runtime_root / "home", "runtime_home"),
    }
    expected_runtime_entries = {"home", "pycache", "temp"}
    try:
        if {entry.name for entry in direct_runtime_root.iterdir()} != (
            expected_runtime_entries
        ):
            raise OODExternalV2IntegrityError(
                "isolated launcher runtime root layout differs"
            )
        home = direct_runtime_root / "home"
        app_data = home / "AppData"
        if (
            {entry.name for entry in home.iterdir()} != {"AppData"}
            or {entry.name for entry in app_data.iterdir()} != {"Local", "Roaming"}
        ):
            raise OODExternalV2IntegrityError(
                "isolated launcher profile layout differs"
            )
        for path, _ in runtime_roles.values():
            direct = _assert_direct_ancestry(path, context="isolated runtime role")
            if not direct.is_dir() or (
                direct.name not in {"home", "AppData"} and any(direct.iterdir())
            ):
                raise OODExternalV2IntegrityError(
                    "isolated runtime role is unavailable or non-empty"
                )
    except OSError as error:
        raise OODExternalV2IntegrityError(
            "isolated launcher runtime layout cannot be inspected"
        ) from error
    normalized: dict[str, str] = {}
    for name, value in os.environ.items():
        canonical_name = name.upper()
        if canonical_name in normalized:
            raise OODExternalV2IntegrityError(
                "runtime environment contains a case-colliding variable"
            )
        if canonical_name not in ALLOWED_RUNTIME_ENVIRONMENT_VARIABLES:
            raise OODExternalV2IntegrityError(
                "runtime environment contains an unbound variable"
            )
        normalized[canonical_name] = value
    system_root = Path(normalized.get("SYSTEMROOT", r"C:\Windows"))
    expected_path = os.fspath(system_root / "System32")
    if (
        normalized.get("PATH") != expected_path
        or normalized.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or normalized.get("CUDA_CACHE_DISABLE") != "1"
        or normalized.get("TORCHINDUCTOR_CACHE_DIR")
        != os.fspath(direct_runtime_root / "temp")
        or normalized.get("PROGRAMFILES") != r"C:\Program Files"
        or normalized.get("PROGRAMW6432") != r"C:\Program Files"
    ):
        raise OODExternalV2IntegrityError(
            "runtime environment differs from the frozen launcher contract"
        )
    canonicalized: dict[str, str] = dict(normalized)
    for name, (path, role) in runtime_roles.items():
        if normalized.get(name) != os.fspath(path):
            raise OODExternalV2IntegrityError(
                "ephemeral runtime environment path differs from its frozen role"
            )
        canonicalized[name] = f"<{role}>"
    return tuple(sorted(canonicalized.items()))


def _verify_runtime_scratch_empty(project_root: Path) -> None:
    raw_prefix = sys.pycache_prefix
    if not isinstance(raw_prefix, str) or not raw_prefix:
        raise OODExternalV2IntegrityError(
            "isolated runtime scratch prefix is unavailable"
        )
    pycache = _assert_direct_ancestry(
        Path(raw_prefix),
        context="isolated runtime scratch pycache",
    )
    if not pycache.is_dir() or any(pycache.iterdir()):
        raise OODExternalV2IntegrityError(
            "isolated runtime scratch pycache is not empty"
        )
    _frozen_runtime_environment_material(
        pycache.parent,
        project_root=project_root,
    )


@dataclass(frozen=True, slots=True)
class _WindowsDirectoryHandleIdentity:
    attributes: int
    volume_serial_number: int
    file_id: bytes


def _windows_directory_handle_identity(
    handle: int,
    *,
    context: str,
) -> _WindowsDirectoryHandleIdentity:
    """Read a reparse-aware, 128-bit identity from one open directory handle."""

    class _FileAttributeTagInfo(ctypes.Structure):
        _fields_ = [
            ("FileAttributes", ctypes.c_uint32),
            ("ReparseTag", ctypes.c_uint32),
        ]

    class _FileId128(ctypes.Structure):
        _fields_ = [("Identifier", ctypes.c_ubyte * 16)]

    class _FileIdInfo(ctypes.Structure):
        _fields_ = [
            ("VolumeSerialNumber", ctypes.c_uint64),
            ("FileId", _FileId128),
        ]

    if os.name != "nt":
        raise OODExternalV2IntegrityError(f"{context} requires Windows handle identity")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_information = kernel32.GetFileInformationByHandleEx
    get_information.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    get_information.restype = ctypes.c_int
    attribute_info = _FileAttributeTagInfo()
    file_id_info = _FileIdInfo()
    ctypes.set_last_error(0)
    if get_information(
        handle,
        9,  # FileAttributeTagInfo
        ctypes.byref(attribute_info),
        ctypes.sizeof(attribute_info),
    ) == 0:
        raise OODExternalV2IntegrityError(
            f"{context} attributes cannot be read from its bound handle"
        ) from None
    ctypes.set_last_error(0)
    if get_information(
        handle,
        18,  # FileIdInfo
        ctypes.byref(file_id_info),
        ctypes.sizeof(file_id_info),
    ) == 0:
        raise OODExternalV2IntegrityError(
            f"{context} identity cannot be read from its bound handle"
        ) from None
    return _WindowsDirectoryHandleIdentity(
        attributes=int(attribute_info.FileAttributes),
        volume_serial_number=int(file_id_info.VolumeSerialNumber),
        file_id=bytes(file_id_info.FileId.Identifier),
    )


def _remove_exact_empty_windows_directory(path: Path, *, context: str) -> None:
    """Delete one direct empty directory through the handle that was inspected."""

    class _FileDispositionInfo(ctypes.Structure):
        _fields_ = [("DeleteFile", ctypes.c_ubyte)]

    if os.name != "nt":
        raise OODExternalV2IntegrityError(f"{context} cleanup requires Windows")
    direct = _assert_direct_ancestry(path, context=context)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    create_file.restype = ctypes.c_void_p
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [ctypes.c_void_p]
    close_handle.restype = ctypes.c_int
    set_information = kernel32.SetFileInformationByHandle
    set_information.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    set_information.restype = ctypes.c_int

    file_list_directory = 0x00000001
    file_read_attributes = 0x00000080
    delete_access = 0x00010000
    file_share_read = 0x00000001
    file_share_all = 0x00000001 | 0x00000002 | 0x00000004
    open_existing = 3
    file_flag_open_reparse_point = 0x00200000
    file_flag_backup_semantics = 0x02000000
    file_attribute_directory = 0x00000010
    file_attribute_reparse_point = 0x00000400
    invalid_handle = ctypes.c_void_p(-1).value

    def _open(*, delete: bool, restrictive: bool) -> int:
        desired_access = file_list_directory | file_read_attributes
        if delete:
            desired_access |= delete_access
        ctypes.set_last_error(0)
        opened = create_file(
            os.fspath(direct),
            desired_access,
            file_share_read if restrictive else file_share_all,
            None,
            open_existing,
            file_flag_backup_semantics | file_flag_open_reparse_point,
            None,
        )
        if opened in (None, invalid_handle):
            raise OODExternalV2IntegrityError(
                f"{context} cannot be opened with a race-safe directory handle"
            ) from None
        return cast(int, opened)

    def _close(handle: int) -> None:
        ctypes.set_last_error(0)
        if close_handle(handle) == 0:
            raise OODExternalV2IntegrityError(
                f"{context} race-safe directory handle could not be closed"
            ) from None

    handle = _open(delete=True, restrictive=True)
    try:
        initial_identity = _windows_directory_handle_identity(handle, context=context)
        if (
            initial_identity.attributes & file_attribute_directory == 0
            or initial_identity.attributes & file_attribute_reparse_point != 0
        ):
            raise OODExternalV2IntegrityError(
                f"{context} is not a direct directory"
            )
        if _assert_direct_ancestry(direct, context=context) != direct:
            raise OODExternalV2IntegrityError(f"{context} path identity differs")
        try:
            if any(direct.iterdir()):
                raise OODExternalV2IntegrityError(f"{context} is not empty")
        except OODExternalV2IntegrityError:
            raise
        except OSError:
            raise OODExternalV2IntegrityError(
                f"{context} cannot be inspected while locked"
            ) from None

        witness = _open(delete=False, restrictive=False)
        try:
            witness_identity = _windows_directory_handle_identity(
                witness,
                context=context,
            )
        finally:
            _close(witness)
        final_identity = _windows_directory_handle_identity(handle, context=context)
        if witness_identity != initial_identity or final_identity != initial_identity:
            raise OODExternalV2IntegrityError(
                f"{context} identity changed while locked"
            )

        disposition = _FileDispositionInfo(1)
        ctypes.set_last_error(0)
        if set_information(
            handle,
            4,  # FileDispositionInfo
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        ) == 0:
            raise OODExternalV2IntegrityError(
                f"{context} could not be removed through its bound handle"
            ) from None
    finally:
        _close(handle)
    try:
        if direct.exists() or _is_indirect(direct):
            raise OODExternalV2IntegrityError(
                f"{context} remains after bound-handle removal"
            )
    except OODExternalV2IntegrityError:
        raise
    except OSError:
        raise OODExternalV2IntegrityError(
            f"{context} removal cannot be verified"
        ) from None


def _remove_exact_empty_gcm_sentinel_directory(project_root: Path) -> None:
    """Remove only GCM's exact empty System.CommandLine scratch directory."""

    raw_prefix = sys.pycache_prefix
    if not isinstance(raw_prefix, str) or not raw_prefix:
        raise OODExternalV2IntegrityError(
            "GCM scratch cleanup requires the isolated runtime prefix"
        )
    pycache = _assert_direct_ancestry(
        Path(raw_prefix),
        context="GCM scratch cleanup pycache",
    )
    runtime_root = pycache.parent
    expected_parent = _assert_direct_ancestry(
        project_root / "artifacts" / "trust_sentinel",
        context="GCM scratch cleanup runtime parent",
    )
    if (
        pycache.name != "pycache"
        or runtime_root.parent != expected_parent
        or re.fullmatch(
            r"\.ood_external_v2_1\.runtime-[0-9a-f]{64}",
            runtime_root.name,
        )
        is None
    ):
        raise OODExternalV2IntegrityError(
            "GCM scratch cleanup is outside the frozen runtime namespace"
        )
    temporary = _assert_direct_ancestry(
        runtime_root / "temp",
        context="GCM scratch cleanup temporary directory",
    )
    expected_temporary = os.fspath(temporary)
    if (
        not temporary.is_dir()
        or os.environ.get("TEMP") != expected_temporary
        or os.environ.get("TMP") != expected_temporary
    ):
        raise OODExternalV2IntegrityError(
            "GCM scratch cleanup temporary directory differs"
        )
    try:
        entries = tuple(temporary.iterdir())
    except OSError:
        raise OODExternalV2IntegrityError(
            "GCM scratch cleanup temporary directory cannot be inspected"
        ) from None
    if entries:
        if (
            len(entries) != 1
            or entries[0].name
            != GCM_SYSTEM_COMMANDLINE_SENTINEL_DIRECTORY_NAME
        ):
            raise OODExternalV2IntegrityError(
                "GCM scratch cleanup found unexpected temporary content"
            )
        sentinel = _assert_direct_ancestry(
            entries[0],
            context="GCM system-commandline sentinel directory",
        )
        _remove_exact_empty_windows_directory(
            sentinel,
            context="GCM system-commandline sentinel directory",
        )
    _verify_runtime_scratch_empty(project_root)


def _loaded_windows_native_module_paths() -> tuple[Path, ...]:
    """Enumerate every image loaded in the active Windows process."""

    if os.name != "nt":
        raise OODExternalV2IntegrityError(
            "native-module provenance requires the frozen Windows runtime"
        )
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = ctypes.c_void_p
    enum_modules = psapi.EnumProcessModules
    enum_modules.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    enum_modules.restype = ctypes.c_int
    get_module_name = psapi.GetModuleFileNameExW
    get_module_name.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
    ]
    get_module_name.restype = ctypes.c_uint32
    process = get_current_process()
    capacity = 512
    while True:
        handles = (ctypes.c_void_p * capacity)()
        needed = ctypes.c_uint32(0)
        if enum_modules(
            process,
            handles,
            ctypes.sizeof(handles),
            ctypes.byref(needed),
        ) == 0:
            raise OODExternalV2IntegrityError(
                "loaded native modules could not be enumerated"
            )
        count = needed.value // ctypes.sizeof(ctypes.c_void_p)
        if count <= capacity:
            break
        capacity = count + 64
        if capacity > 16_384:
            raise OODExternalV2IntegrityError(
                "loaded native module count exceeds its frozen bound"
            )
    result: set[Path] = set()
    for index in range(count):
        buffer = ctypes.create_unicode_buffer(32_768)
        length = get_module_name(process, handles[index], buffer, len(buffer))
        if length == 0 or length >= len(buffer) - 1:
            raise OODExternalV2IntegrityError(
                "loaded native module path could not be read exactly"
            )
        result.add(Path(os.path.abspath(buffer.value)))
    if not result:
        raise OODExternalV2IntegrityError("loaded native module set is empty")
    return tuple(sorted(result, key=lambda path: os.fspath(path).casefold()))


def _verify_loaded_native_module_origins(
    *,
    python_executable: Path,
    python_base_alias: Path,
    python_base_target: Path,
    site_packages: Path,
) -> None:
    """Require every loaded native image to come from a bound or OS tree."""

    system_root = _assert_direct_ancestry(
        Path(os.environ.get("SYSTEMROOT", r"C:\Windows")),
        context="native-module Windows root",
    )
    system32 = _assert_direct_ancestry(
        system_root / "System32",
        context="native-module System32 root",
    )
    winsxs = _assert_direct_ancestry(
        system_root / "WinSxS",
        context="native-module WinSxS root",
    )
    exact_nvidia = {
        path.name.casefold(): path for path in _nvidia_driver_tool_paths()[1:]
    }
    exact_host_security: dict[Path, tuple[int, str]] = {}
    for raw_host_path, expected in EXPECTED_HOST_SECURITY_NATIVE_MODULES.items():
        path = _assert_direct_ancestry(
            Path(raw_host_path),
            context="bound host-security native module",
        )
        entry = _stable_runtime_file_entry(
            path,
            context="bound host-security native module",
        )
        if (
            not path.is_file()
            or entry["size_bytes"] != expected[0]
            or entry["sha256"] != expected[1]
        ):
            raise OODExternalV2IntegrityError(
                "bound host-security native module differs"
            )
        exact_host_security[path] = expected
    base_python = _assert_direct_ancestry(
        python_base_target / "python.exe",
        context="bound CPython base executable",
    )
    if not base_python.is_file():
        raise OODExternalV2IntegrityError(
            "bound CPython base executable is unavailable"
        )
    observed_base_python = False
    observed_host_security: set[Path] = set()
    for raw_path in _loaded_windows_native_module_paths():
        lexical = Path(os.path.abspath(os.fspath(raw_path)))
        try:
            alias_relative = lexical.relative_to(python_base_alias)
        except ValueError:
            source = _assert_direct_ancestry(
                lexical,
                context="loaded native module",
            )
        else:
            source = _assert_direct_ancestry(
                python_base_target / alias_relative,
                context="loaded CPython native module",
            )
        if not source.is_file() or source.suffix.casefold() not in {
            ".dll",
            ".exe",
            ".pyd",
        }:
            raise OODExternalV2IntegrityError(
                "loaded native module is not a regular executable image"
            )
        if source == base_python:
            observed_base_python = True
            continue
        # ``sys.executable`` is the separately bound uv virtual-environment
        # redirector. Windows replaces that bootstrap image with the exact
        # base CPython image above, so its absence from PSAPI is expected.
        if source == python_executable:
            continue
        if _path_is_within(source, python_base_target) or _path_is_within(
            source, site_packages
        ):
            continue
        if source in exact_host_security:
            observed_host_security.add(source)
            continue
        nvidia_path = exact_nvidia.get(source.name.casefold())
        if nvidia_path is not None:
            if source != nvidia_path:
                raise OODExternalV2IntegrityError(
                    "loaded NVIDIA module differs from the exact bound system file"
                )
            continue
        if source.name.casefold().startswith("nv"):
            raise OODExternalV2IntegrityError(
                "loaded NVIDIA module is not among the exact frozen exceptions"
            )
        if _path_is_within(source, system32) or _path_is_within(source, winsxs):
            continue
        raise OODExternalV2IntegrityError(
            "loaded native module originated outside every frozen runtime tree"
        )
    if not observed_base_python:
        raise OODExternalV2IntegrityError(
            "native-module audit did not observe the exact CPython base executable"
        )
    if observed_host_security != set(exact_host_security):
        raise OODExternalV2IntegrityError(
            "loaded host-security native module set differs from the frozen binding"
        )


def _current_runtime_environment() -> RuntimeEnvironmentBinding:
    if (
        sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or not sys.dont_write_bytecode
        or sys.flags.no_user_site != 1
        or any(os.environ.get(name) for name in FORBIDDEN_CODE_ENVIRONMENT_VARIABLES)
        or any(name in sys.modules for name in FORBIDDEN_BOOTSTRAP_MODULES)
    ):
        raise OODExternalV2IntegrityError(
            "active Python process was not started by the frozen isolated launcher"
        )
    executable = _assert_direct_ancestry(
        Path(sys.executable),
        context="active Python executable",
    )
    if not executable.is_file():
        raise OODExternalV2IntegrityError("active Python executable is missing")
    venv_root = _assert_direct_ancestry(
        executable.parent.parent,
        context="virtual environment root",
    )
    python_base_alias, python_base = _resolve_python_base_runtime()
    site_packages = _assert_direct_ancestry(
        venv_root / "Lib" / "site-packages",
        context="venv site-packages root",
    )
    project_src = _assert_direct_ancestry(
        Path(__file__).parents[2],
        context="project source root",
    )
    pyvenv_config = _assert_direct_ancestry(
        venv_root / "pyvenv.cfg",
        context="pyvenv.cfg",
    )
    raw_pycache_prefix = sys.pycache_prefix
    if not isinstance(raw_pycache_prefix, str) or not raw_pycache_prefix:
        raise OODExternalV2IntegrityError("isolated runtime pycache prefix is absent")
    pycache_prefix = _assert_direct_ancestry(
        Path(raw_pycache_prefix),
        context="isolated runtime pycache prefix",
    )
    try:
        if not pycache_prefix.is_dir() or any(pycache_prefix.iterdir()):
            raise OODExternalV2IntegrityError(
                "isolated runtime pycache prefix is not a verified-empty directory"
            )
    except OSError as error:
        raise OODExternalV2IntegrityError(
            "isolated runtime pycache prefix cannot be inspected"
        ) from error
    project_root = project_src.parent
    sanitized_environment = _frozen_runtime_environment_material(
        pycache_prefix.parent,
        project_root=project_root,
    )
    sys_path_layout = _runtime_path_layout(
        python_base=python_base_alias,
        site_packages=site_packages,
        project_src=project_src,
    )
    try:
        executable_size = executable.stat().st_size
        pyvenv_size = pyvenv_config.stat().st_size
        python_base_tree = _runtime_filesystem_tree(
            python_base,
            tree_kind=RUNTIME_FILESYSTEM_TREE_KINDS[0],
        )
        site_packages_tree = _runtime_filesystem_tree(
            site_packages,
            tree_kind=RUNTIME_FILESYSTEM_TREE_KINDS[1],
        )
        git_tool = _current_git_tool_binding()
        nvidia_driver_tool = _current_nvidia_driver_tool_binding()
        package_trees = tuple(
            _installed_distribution_tree_binding(package)
            for package in sorted(EXPECTED_SCIENTIFIC_PACKAGE_VERSIONS)
        )
        versions = {item.distribution: item.version for item in package_trees}
    except (OSError, importlib.metadata.PackageNotFoundError) as error:
        raise OODExternalV2IntegrityError(
            "active scientific runtime cannot be identified"
        ) from error
    project_sources = _build_project_source_tree(project_root)
    _verify_all_file_backed_module_origins(
        project_root=project_root,
        project_sources=project_sources,
        python_base_alias=python_base_alias,
        python_base_target=python_base,
        site_packages=site_packages,
    )
    _verify_loaded_native_module_origins(
        python_executable=executable,
        python_base_alias=python_base_alias,
        python_base_target=python_base,
        site_packages=site_packages,
    )
    environment_material = canonical_json_bytes(
        {
            "executable_file_sha256": sha256_file(executable),
            "isolated_mode": True,
            "no_site": True,
            "dont_write_bytecode": True,
            "user_site_disabled": True,
            "pycache_prefix_verified_empty": True,
            "python_base_tree": _runtime_filesystem_tree_dict(python_base_tree),
            "python_base_alias_name": python_base_alias.name,
            "python_base_target_name": python_base.name,
            "site_packages_tree": _runtime_filesystem_tree_dict(site_packages_tree),
            "git_tool": _git_tool_dict(git_tool),
            "host_security_native_modules": [
                {
                    "path": path,
                    "sha256": digest,
                    "size_bytes": size,
                }
                for path, (size, digest) in (
                    EXPECTED_HOST_SECURITY_NATIVE_MODULES.items()
                )
            ],
            "nvidia_driver_tool": _nvidia_driver_tool_dict(nvidia_driver_tool),
            "pyvenv_config_file_sha256": sha256_file(pyvenv_config),
            "pyvenv_config_size_bytes": pyvenv_size,
            "sys_path_layout": list(sys_path_layout),
            "versions": versions,
            "package_trees": [
                {
                    "distribution": item.distribution,
                    "file_count": item.file_count,
                    "import_roots": list(item.import_roots),
                    "total_bytes": item.total_bytes,
                    "tree_sha256": item.tree_sha256,
                    "version": item.version,
                }
                for item in package_trees
            ],
            "sanitized_environment": [list(item) for item in sanitized_environment],
        }
    )[:-1]
    return _runtime_environment_binding(
        {
            "numpy_version": versions["numpy"],
            "dont_write_bytecode": True,
            "git_tool": _git_tool_dict(git_tool),
            "nvidia_driver_tool": _nvidia_driver_tool_dict(nvidia_driver_tool),
            "isolated_mode": True,
            "no_site": True,
            "package_trees": [
                {
                    "distribution": item.distribution,
                    "file_count": item.file_count,
                    "import_roots": list(item.import_roots),
                    "total_bytes": item.total_bytes,
                    "tree_sha256": item.tree_sha256,
                    "version": item.version,
                }
                for item in package_trees
            ],
            "pycache_prefix_verified_empty": True,
            "python_base_tree": _runtime_filesystem_tree_dict(python_base_tree),
            "python_base_alias_name": python_base_alias.name,
            "python_base_target_name": python_base.name,
            "python_environment_sha256": sha256_bytes(environment_material),
            "python_executable_file_sha256": sha256_file(executable),
            "python_executable_size_bytes": executable_size,
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "pyvenv_config_file_sha256": sha256_file(pyvenv_config),
            "pyvenv_config_size_bytes": pyvenv_size,
            "scipy_version": versions["scipy"],
            "site_packages_tree": _runtime_filesystem_tree_dict(site_packages_tree),
            "sys_path_layout": list(sys_path_layout),
            "user_site_disabled": True,
            "wfdb_version": versions["wfdb"],
        },
        context="active runtime environment",
    )


def _endpoint_minimum(
    primary: Mapping[str, object],
    endpoint_key: str,
    expected: float,
) -> float:
    endpoint = _mapping(primary.get(endpoint_key), f"primary endpoint {endpoint_key}")
    observed = _exact_float(endpoint.get("minimum"), f"minimum {endpoint_key}")
    if observed != expected:
        raise OODExternalV2ConfigError(f"minimum for {endpoint_key} changed")
    return observed


def _utc_datetime(value: object, context: str) -> datetime:
    if not isinstance(value, str) or not value or value != value.strip():
        raise OODExternalV2ConfigError(f"{context} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise OODExternalV2ConfigError(f"{context} is not a timestamp") from error
    offset = parsed.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise OODExternalV2ConfigError(f"{context} must use UTC")
    return parsed


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise OODExternalV2ConfigError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_duplicate_yaml_keys(text: str) -> None:
    try:
        root = yaml.compose(text)
    except yaml.YAMLError as error:
        raise OODExternalV2ConfigError("parent YAML cannot be composed") from error
    visited: set[int] = set()

    def visit(node: yaml.Node) -> None:
        identity = id(node)
        if identity in visited:
            raise OODExternalV2ConfigError("YAML aliases are forbidden")
        visited.add(identity)
        if isinstance(node, yaml.MappingNode):
            seen: set[tuple[str, str]] = set()
            for key_node, value_node in node.value:
                if not isinstance(key_node, yaml.ScalarNode):
                    raise OODExternalV2ConfigError("YAML keys must be scalar")
                if str(key_node.tag).endswith(":merge") or str(key_node.value) == "<<":
                    raise OODExternalV2ConfigError("YAML merge keys are forbidden")
                key = (str(key_node.tag), str(key_node.value))
                if key in seen:
                    raise OODExternalV2ConfigError("YAML keys must be unique")
                seen.add(key)
                visit(value_node)
        elif isinstance(node, yaml.SequenceNode):
            for child in node.value:
                visit(child)

    if root is not None:
        visit(root)


def _is_indirect(path: Path) -> bool:
    try:
        junction = getattr(path, "is_junction", None)
        return path.is_symlink() or bool(junction is not None and junction())
    except OSError as error:
        raise OODExternalV2IntegrityError("filesystem link state cannot be inspected") from error


def _strict_project_root(value: str | Path) -> Path:
    try:
        root = _assert_direct_ancestry(
            Path(os.path.abspath(os.fspath(value))),
            context="project root",
        )
    except OSError as error:
        raise OODExternalV2IntegrityError("project root is unavailable") from error
    if _is_indirect(root) or not root.is_dir():
        raise OODExternalV2IntegrityError("project root is not a direct directory")
    completed = _run_git(root, "rev-parse", "--show-toplevel")
    try:
        git_root = Path(completed.stdout.strip()).resolve(strict=True)
    except OSError as error:
        raise OODExternalV2IntegrityError("Git worktree root is unavailable") from error
    if git_root != root:
        raise OODExternalV2IntegrityError("project root is not the Git worktree root")
    return root


def _require_project_file(root: Path, path: Path, *, context: str) -> Path:
    try:
        resolved = _assert_direct_ancestry(path, context=context)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise OODExternalV2IntegrityError(
            f"{context} must be a regular file inside the project"
        ) from error
    if not resolved.is_file():
        raise OODExternalV2IntegrityError(f"{context} is missing or indirect")
    return resolved


def _resolve_project_relative(
    root: Path,
    relative_path: str,
    *,
    require_file: bool = False,
    require_directory: bool = False,
) -> Path:
    canonical = _relative_path(relative_path, "project path")
    candidate = root.joinpath(*PurePosixPath(canonical).parts)
    if require_file:
        return _require_project_file(root, candidate, context=canonical)
    if require_directory:
        try:
            resolved = _assert_direct_ancestry(
                candidate,
                context=f"project directory {canonical}",
            )
            resolved.relative_to(root)
        except (OSError, ValueError) as error:
            raise OODExternalV2IntegrityError(
                f"project directory is unavailable: {canonical}"
            ) from error
        if not resolved.is_dir():
            raise OODExternalV2IntegrityError(
                f"project directory is missing or indirect: {canonical}"
            )
        return resolved
    lexical = Path(os.path.abspath(os.fspath(candidate)))
    try:
        lexical.relative_to(root)
        cursor = lexical
        while not cursor.exists():
            if _is_indirect(cursor) or cursor == cursor.parent:
                raise OODExternalV2IntegrityError(
                    "project destination has an indirect or missing ancestry"
                )
            cursor = cursor.parent
        direct_ancestor = _assert_direct_ancestry(
            cursor,
            context=f"project destination ancestor {canonical}",
        )
        direct_ancestor.relative_to(root)
    except (OSError, ValueError) as error:
        raise OODExternalV2IntegrityError("project path escapes the worktree") from error
    return lexical


def _project_relative_existing_directory(
    root: Path,
    value: str | Path,
    *,
    context: str,
) -> str:
    try:
        source = _assert_direct_ancestry(
            Path(os.path.abspath(os.fspath(value))),
            context=context,
        )
        relative = source.relative_to(root).as_posix()
    except (OSError, ValueError) as error:
        raise OODExternalV2IntegrityError(f"{context} is outside the project") from error
    if not source.is_dir():
        raise OODExternalV2IntegrityError(f"{context} is missing or indirect")
    return _relative_path(relative, context)


def _read_bounded(path: Path, maximum_bytes: int, context: str) -> bytes:
    direct = _assert_direct_ancestry(path, context=context)
    if not direct.is_file():
        raise OODExternalV2IntegrityError(f"{context} is missing or indirect")
    try:
        before = direct.stat()
        if not 0 < before.st_size <= maximum_bytes:
            raise OODExternalV2IntegrityError(f"{context} has an invalid size")
        payload = direct.read_bytes()
        after = direct.stat()
    except OODExternalV2IntegrityError:
        raise
    except OSError as error:
        raise OODExternalV2IntegrityError(f"{context} cannot be read") from error
    if (
        len(payload) != before.st_size
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or _assert_direct_ancestry(direct, context=context) != direct
    ):
        raise OODExternalV2IntegrityError(f"{context} changed while being read")
    return payload


def _git_executable_paths() -> tuple[Path, Path, Path]:
    install_root = _assert_direct_ancestry(
        Path(EXPECTED_GIT_INSTALL_ROOT),
        context="Git installation root",
    )
    launcher = _assert_direct_ancestry(
        install_root / "cmd" / EXPECTED_GIT_LAUNCHER_NAME,
        context="Git launcher",
    )
    executable = _assert_direct_ancestry(
        install_root / "mingw64" / "bin" / "git.exe",
        context="Git executable",
    )
    if not launcher.is_file() or not executable.is_file():
        raise OODExternalV2IntegrityError("frozen Git executables are unavailable")
    launcher_entry = _stable_runtime_file_entry(launcher, context="Git launcher")
    executable_entry = _stable_runtime_file_entry(executable, context="Git executable")
    if (
        launcher.name != EXPECTED_GIT_LAUNCHER_NAME
        or launcher_entry["size_bytes"] != EXPECTED_GIT_LAUNCHER_SIZE_BYTES
        or launcher_entry["sha256"] != EXPECTED_GIT_LAUNCHER_SHA256
        or executable.name != EXPECTED_GIT_EXECUTABLE_NAME
        or executable_entry["size_bytes"] != EXPECTED_GIT_EXECUTABLE_SIZE_BYTES
        or executable_entry["sha256"] != EXPECTED_GIT_EXECUTABLE_SHA256
    ):
        raise OODExternalV2IntegrityError(
            "resolved Git launcher or executable differs from frozen bytes"
        )
    return launcher, executable, install_root


def _git_credential_manager_path(executable: Path) -> Path:
    helper = _assert_direct_ancestry(
        executable.parent / EXPECTED_GIT_CREDENTIAL_MANAGER_NAME,
        context="Git credential manager",
    )
    if not helper.is_file():
        raise OODExternalV2IntegrityError("frozen Git credential manager is unavailable")
    helper_entry = _stable_runtime_file_entry(
        helper,
        context="Git credential manager",
    )
    if (
        helper.name != EXPECTED_GIT_CREDENTIAL_MANAGER_NAME
        or helper_entry["size_bytes"]
        != EXPECTED_GIT_CREDENTIAL_MANAGER_SIZE_BYTES
        or helper_entry["sha256"] != EXPECTED_GIT_CREDENTIAL_MANAGER_SHA256
    ):
        raise OODExternalV2IntegrityError("frozen Git credential manager differs")
    return helper


def _sanitized_git_environment(executable: Path) -> dict[str, str]:
    environment: dict[str, str] = {
        "GIT_CONFIG_COUNT": "0",
        "GIT_CONFIG_GLOBAL": "NUL" if os.name == "nt" else "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "PATH": os.pathsep.join(
            (
                os.fspath(executable.parent),
                os.fspath(executable.parent.parent / "libexec" / "git-core"),
                os.fspath(Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32"),
            )
        ),
    }
    for name in ("COMSPEC", "SYSTEMROOT", "TEMP", "TMP", "WINDIR"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    return environment


def _private_live_remote_environment(executable: Path) -> dict[str, str]:
    environment = _sanitized_git_environment(executable)
    environment.update(PRIVATE_REMOTE_GCM_ENVIRONMENT)
    return environment


def _private_remote_command(
    project_root: Path,
    executable: Path,
    *,
    authenticated: bool,
) -> tuple[list[str], dict[str, str]]:
    """Build and immediately validate the sole exact private-remote command."""

    parsed_url = urlsplit(EXPECTED_GIT_REMOTE_URL)
    if (
        parsed_url.scheme != "https"
        or parsed_url.hostname != "github.com"
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.port is not None
        or parsed_url.path != "/Ahmad986Ferdaws/ecg-trust-lab.git"
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise OODExternalV2IntegrityError("private Git remote URL boundary differs")
    config = (
        PRIVATE_REMOTE_GIT_CONFIG
        if authenticated
        else PRIVATE_REMOTE_ANONYMOUS_GIT_CONFIG
    )
    config_arguments = tuple(
        argument
        for config_entry in config
        for argument in ("-c", config_entry)
    )
    command = _bound_git_command(
        project_root,
        executable,
        *config_arguments,
        "ls-remote",
        "--symref",
        EXPECTED_GIT_REMOTE_URL,
    )
    environment = (
        _private_live_remote_environment(executable)
        if authenticated
        else _sanitized_git_environment(executable)
    )
    expected_environment = _sanitized_git_environment(executable)
    if authenticated:
        expected_environment.update(PRIVATE_REMOTE_GCM_ENVIRONMENT)
    if (
        environment != expected_environment
        or any(
            name in environment for name in PRIVATE_REMOTE_FORBIDDEN_ENVIRONMENT_KEYS
        )
        or (not authenticated and any(name.startswith("GCM_") for name in environment))
    ):
        raise OODExternalV2IntegrityError(
            "private Git remote environment boundary differs"
        )
    return command, environment


def _run_bounded_windows_process_inner(
    command: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
    stdout_limit_bytes: int,
    stderr_limit_bytes: int,
    failure_message: str,
    stdout_buffer: bytearray,
    stderr_buffer: bytearray,
) -> subprocess.CompletedProcess[bytes]:
    """Run one private probe in an atomically assigned, kill-on-close job.

    The raw Win32 launch is intentional.  ``subprocess.Popen`` cannot attach a
    process to a Job Object atomically, leaving a pre-assignment window in
    which a credential helper could create an uncontained descendant.  The
    STARTUPINFOEX job-list attribute closes that window, while the handle-list
    attribute ensures that only NUL/stdout/stderr cross into the child.
    """

    class _SecurityAttributes(ctypes.Structure):
        _fields_ = [
            ("nLength", ctypes.c_uint32),
            ("lpSecurityDescriptor", ctypes.c_void_p),
            ("bInheritHandle", ctypes.c_int),
        ]

    class _StartupInfo(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_uint32),
            ("lpReserved", ctypes.c_wchar_p),
            ("lpDesktop", ctypes.c_wchar_p),
            ("lpTitle", ctypes.c_wchar_p),
            ("dwX", ctypes.c_uint32),
            ("dwY", ctypes.c_uint32),
            ("dwXSize", ctypes.c_uint32),
            ("dwYSize", ctypes.c_uint32),
            ("dwXCountChars", ctypes.c_uint32),
            ("dwYCountChars", ctypes.c_uint32),
            ("dwFillAttribute", ctypes.c_uint32),
            ("dwFlags", ctypes.c_uint32),
            ("wShowWindow", ctypes.c_uint16),
            ("cbReserved2", ctypes.c_uint16),
            ("lpReserved2", ctypes.POINTER(ctypes.c_ubyte)),
            ("hStdInput", ctypes.c_void_p),
            ("hStdOutput", ctypes.c_void_p),
            ("hStdError", ctypes.c_void_p),
        ]

    class _StartupInfoEx(ctypes.Structure):
        _fields_ = [
            ("StartupInfo", _StartupInfo),
            ("lpAttributeList", ctypes.c_void_p),
        ]

    class _ProcessInformation(ctypes.Structure):
        _fields_ = [
            ("hProcess", ctypes.c_void_p),
            ("hThread", ctypes.c_void_p),
            ("dwProcessId", ctypes.c_uint32),
            ("dwThreadId", ctypes.c_uint32),
        ]

    class _JobBasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _JobExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JobBasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class _JobBasicAccountingInformation(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", ctypes.c_int64),
            ("TotalKernelTime", ctypes.c_int64),
            ("ThisPeriodTotalUserTime", ctypes.c_int64),
            ("ThisPeriodTotalKernelTime", ctypes.c_int64),
            ("TotalPageFaultCount", ctypes.c_uint32),
            ("TotalProcesses", ctypes.c_uint32),
            ("ActiveProcesses", ctypes.c_uint32),
            ("TotalTerminatedProcesses", ctypes.c_uint32),
        ]

    invalid_handle_value = ctypes.c_void_p(-1).value
    wait_object_0 = 0
    wait_timeout = 258
    startf_use_std_handles = 0x00000100
    handle_flag_inherit = 0x00000001
    generic_read = 0x80000000
    file_share_all = 0x00000001 | 0x00000002 | 0x00000004
    open_existing = 3
    file_attribute_normal = 0x00000080
    error_broken_pipe = 109
    error_insufficient_buffer = 122
    job_object_basic_accounting_information = 1
    job_object_extended_limit_information = 9
    terminate_exit_code = 0xFFFFFFFF

    kernel32: Any | None = None
    close_handle: Any | None = None
    delete_attribute_list: Any | None = None
    terminate_job: Any | None = None
    wait_for_single_object: Any | None = None
    query_job: Any | None = None
    cancel_synchronous_io: Any | None = None
    job_handle: Any | None = None
    process_handle: Any | None = None
    thread_handle: Any | None = None
    signal_handle: Any | None = None
    stdout_read_handle: Any | None = None
    stdout_write_handle: Any | None = None
    stderr_read_handle: Any | None = None
    stderr_write_handle: Any | None = None
    stdin_handle: Any | None = None
    attribute_pointer: Any | None = None
    attribute_buffer: Any | None = None
    attribute_initialized = False
    process_created = False
    operation_failed = False
    cleanup_complete = True
    process_returncode: int | None = None
    reader_threads: list[tuple[int, threading.Thread]] = []
    reader_thread_handles: list[Any | None] = [None, None]
    reader_handle_ready = [threading.Event(), threading.Event()]
    reader_cancel_requested = [threading.Event(), threading.Event()]
    output_overflow = threading.Event()
    reader_failure = threading.Event()

    def _release_owned_handle(handle: Any | None) -> tuple[Any | None, bool]:
        if handle is None or handle == invalid_handle_value:
            return None, True
        if close_handle is None:
            return handle, False
        try:
            if close_handle(handle) != 0:
                return None, True
        except Exception:
            pass
        return handle, False

    try:
        if (
            os.name != "nt"
            or not command
            or not failure_message
            or timeout_seconds <= 0
            or not math.isfinite(timeout_seconds)
            or stdout_limit_bytes < 0
            or stderr_limit_bytes < 0
            or not Path(command[0]).is_absolute()
            or not cwd.is_absolute()
            or any(not isinstance(argument, str) or "\x00" in argument for argument in command)
        ):
            raise RuntimeError
        environment_entries: list[str] = []
        folded_environment_names: set[str] = set()
        for name, value in environment.items():
            folded_name = name.casefold()
            if (
                not name
                or "=" in name
                or "\x00" in name
                or "\x00" in value
                or folded_name in folded_environment_names
            ):
                raise RuntimeError
            folded_environment_names.add(folded_name)
            environment_entries.append(f"{name}={value}")
        environment_entries.sort(key=str.casefold)
        command_line_text = subprocess.list2cmdline(command)
        cwd_text = os.fspath(cwd)
        if (
            len(command_line_text) >= 32_767
            or "\x00" in cwd_text
            or sum(len(entry) + 1 for entry in environment_entries) > 1_000_000
        ):
            raise RuntimeError
        command_line = ctypes.create_unicode_buffer(command_line_text)
        environment_block = ctypes.create_unicode_buffer(
            "\x00".join(environment_entries) + "\x00"
        )

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_job = kernel32.CreateJobObjectW
        create_job.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        create_job.restype = ctypes.c_void_p
        set_job_information = kernel32.SetInformationJobObject
        set_job_information.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        set_job_information.restype = ctypes.c_int
        initialize_attribute_list = kernel32.InitializeProcThreadAttributeList
        initialize_attribute_list.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        initialize_attribute_list.restype = ctypes.c_int
        update_attribute = kernel32.UpdateProcThreadAttribute
        update_attribute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        update_attribute.restype = ctypes.c_int
        delete_attribute_list = kernel32.DeleteProcThreadAttributeList
        delete_attribute_list.argtypes = [ctypes.c_void_p]
        delete_attribute_list.restype = None
        create_pipe = kernel32.CreatePipe
        create_pipe.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(_SecurityAttributes),
            ctypes.c_uint32,
        ]
        create_pipe.restype = ctypes.c_int
        set_handle_information = kernel32.SetHandleInformation
        set_handle_information.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        set_handle_information.restype = ctypes.c_int
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(_SecurityAttributes),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        create_file.restype = ctypes.c_void_p
        create_event = kernel32.CreateEventW
        create_event.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_wchar_p,
        ]
        create_event.restype = ctypes.c_void_p
        set_event = kernel32.SetEvent
        set_event.argtypes = [ctypes.c_void_p]
        set_event.restype = ctypes.c_int
        create_process = kernel32.CreateProcessW
        create_process.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessInformation),
        ]
        create_process.restype = ctypes.c_int
        wait_for_multiple_objects = kernel32.WaitForMultipleObjects
        wait_for_multiple_objects.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_uint32,
        ]
        wait_for_multiple_objects.restype = ctypes.c_uint32
        wait_for_single_object = kernel32.WaitForSingleObject
        wait_for_single_object.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        wait_for_single_object.restype = ctypes.c_uint32
        get_exit_code = kernel32.GetExitCodeProcess
        get_exit_code.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        get_exit_code.restype = ctypes.c_int
        read_file = kernel32.ReadFile
        read_file.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
        ]
        read_file.restype = ctypes.c_int
        terminate_job = kernel32.TerminateJobObject
        terminate_job.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        terminate_job.restype = ctypes.c_int
        query_job = kernel32.QueryInformationJobObject
        query_job.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        query_job.restype = ctypes.c_int
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.argtypes = []
        get_current_process.restype = ctypes.c_void_p
        get_current_thread = kernel32.GetCurrentThread
        get_current_thread.argtypes = []
        get_current_thread.restype = ctypes.c_void_p
        duplicate_handle = kernel32.DuplicateHandle
        duplicate_handle.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_uint32,
        ]
        duplicate_handle.restype = ctypes.c_int
        cancel_synchronous_io = kernel32.CancelSynchronousIo
        cancel_synchronous_io.argtypes = [ctypes.c_void_p]
        cancel_synchronous_io.restype = ctypes.c_int
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [ctypes.c_void_p]
        close_handle.restype = ctypes.c_int

        job_handle = create_job(None, None)
        if not job_handle:
            raise RuntimeError
        job_limits = _JobExtendedLimitInformation()
        job_limits.BasicLimitInformation.LimitFlags = (
            WINDOWS_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        if (
            set_job_information(
                job_handle,
                job_object_extended_limit_information,
                ctypes.byref(job_limits),
                ctypes.sizeof(job_limits),
            )
            == 0
        ):
            raise RuntimeError

        security = _SecurityAttributes()
        security.nLength = ctypes.sizeof(security)
        security.bInheritHandle = 1
        stdout_read = ctypes.c_void_p()
        stdout_write = ctypes.c_void_p()
        stderr_read = ctypes.c_void_p()
        stderr_write = ctypes.c_void_p()
        if (
            create_pipe(
                ctypes.byref(stdout_read),
                ctypes.byref(stdout_write),
                ctypes.byref(security),
                0,
            )
            == 0
        ):
            raise RuntimeError
        stdout_read_handle = stdout_read.value
        stdout_write_handle = stdout_write.value
        if (
            create_pipe(
                ctypes.byref(stderr_read),
                ctypes.byref(stderr_write),
                ctypes.byref(security),
                0,
            )
            == 0
        ):
            raise RuntimeError
        stderr_read_handle = stderr_read.value
        stderr_write_handle = stderr_write.value
        if (
            not stdout_read_handle
            or not stdout_write_handle
            or not stderr_read_handle
            or not stderr_write_handle
            or set_handle_information(stdout_read_handle, handle_flag_inherit, 0) == 0
            or set_handle_information(stderr_read_handle, handle_flag_inherit, 0) == 0
        ):
            raise RuntimeError
        stdin_handle = create_file(
            "NUL",
            generic_read,
            file_share_all,
            ctypes.byref(security),
            open_existing,
            file_attribute_normal,
            None,
        )
        if not stdin_handle or stdin_handle == invalid_handle_value:
            raise RuntimeError
        signal_handle = create_event(None, 1, 0, None)
        if not signal_handle:
            raise RuntimeError

        attribute_size = ctypes.c_size_t(0)
        ctypes.set_last_error(0)
        first_initialize = initialize_attribute_list(
            None,
            2,
            0,
            ctypes.byref(attribute_size),
        )
        if (
            first_initialize != 0
            or ctypes.get_last_error() != error_insufficient_buffer
            or attribute_size.value == 0
        ):
            raise RuntimeError
        attribute_buffer = ctypes.create_string_buffer(attribute_size.value)
        attribute_pointer = ctypes.cast(attribute_buffer, ctypes.c_void_p)
        if (
            initialize_attribute_list(
                attribute_pointer,
                2,
                0,
                ctypes.byref(attribute_size),
            )
            == 0
        ):
            raise RuntimeError
        attribute_initialized = True
        job_list = (ctypes.c_void_p * 1)(job_handle)
        inherited_handles = (ctypes.c_void_p * 3)(
            stdin_handle,
            stdout_write_handle,
            stderr_write_handle,
        )
        if (
            update_attribute(
                attribute_pointer,
                0,
                WINDOWS_PROCESS_ATTRIBUTE_JOB_LIST,
                ctypes.cast(job_list, ctypes.c_void_p),
                ctypes.sizeof(job_list),
                None,
                None,
            )
            == 0
            or update_attribute(
                attribute_pointer,
                0,
                WINDOWS_PROCESS_ATTRIBUTE_HANDLE_LIST,
                ctypes.cast(inherited_handles, ctypes.c_void_p),
                ctypes.sizeof(inherited_handles),
                None,
                None,
            )
            == 0
        ):
            raise RuntimeError

        startup = _StartupInfoEx()
        startup.StartupInfo.cb = ctypes.sizeof(startup)
        startup.StartupInfo.dwFlags = startf_use_std_handles
        startup.StartupInfo.hStdInput = stdin_handle
        startup.StartupInfo.hStdOutput = stdout_write_handle
        startup.StartupInfo.hStdError = stderr_write_handle
        startup.lpAttributeList = attribute_pointer
        process_information = _ProcessInformation()
        creation_flags = (
            WINDOWS_EXTENDED_STARTUPINFO_PRESENT
            | WINDOWS_CREATE_UNICODE_ENVIRONMENT
            | WINDOWS_CREATE_NO_WINDOW
        )
        if (
            create_process(
                command[0],
                command_line,
                None,
                None,
                1,
                creation_flags,
                ctypes.cast(environment_block, ctypes.c_void_p),
                cwd_text,
                ctypes.byref(startup),
                ctypes.byref(process_information),
            )
            == 0
        ):
            raise RuntimeError
        process_created = True
        process_handle = process_information.hProcess
        thread_handle = process_information.hThread
        if not process_handle or not thread_handle:
            raise RuntimeError
        deadline = time.monotonic() + timeout_seconds

        delete_attribute_list(attribute_pointer)
        attribute_initialized = False
        attribute_pointer = None
        attribute_buffer = None
        thread_handle, thread_closed = _release_owned_handle(thread_handle)
        stdout_write_handle, stdout_write_closed = _release_owned_handle(
            stdout_write_handle
        )
        stderr_write_handle, stderr_write_closed = _release_owned_handle(
            stderr_write_handle
        )
        stdin_handle, stdin_closed = _release_owned_handle(stdin_handle)
        if not all(
            (
                thread_closed,
                stdout_write_closed,
                stderr_write_closed,
                stdin_closed,
            )
        ):
            raise RuntimeError

        def _bounded_reader(
            index: int,
            handle: Any,
            sink: bytearray,
            limit: int,
        ) -> None:
            try:
                duplicated_thread = ctypes.c_void_p()
                current_process = get_current_process()
                if (
                    duplicate_handle(
                        current_process,
                        get_current_thread(),
                        current_process,
                        ctypes.byref(duplicated_thread),
                        0x00000001 | 0x00100000,
                        0,
                        0,
                    )
                    == 0
                    or not duplicated_thread.value
                ):
                    reader_failure.set()
                    set_event(signal_handle)
                    return
                reader_thread_handles[index] = duplicated_thread.value
                reader_handle_ready[index].set()
                while True:
                    if reader_cancel_requested[index].is_set():
                        return
                    remaining = limit - len(sink)
                    read_size = min(4_096, remaining + 1)
                    chunk = ctypes.create_string_buffer(read_size)
                    transferred = ctypes.c_uint32(0)
                    if (
                        read_file(
                            handle,
                            chunk,
                            read_size,
                            ctypes.byref(transferred),
                            None,
                        )
                        == 0
                    ):
                        read_error = ctypes.get_last_error()
                        if read_error == error_broken_pipe or (
                            read_error == 995
                            and reader_cancel_requested[index].is_set()
                        ):
                            return
                        reader_failure.set()
                        set_event(signal_handle)
                        return
                    if transferred.value == 0:
                        return
                    if transferred.value > remaining:
                        output_overflow.set()
                        set_event(signal_handle)
                        return
                    sink.extend(ctypes.string_at(chunk, transferred.value))
            except Exception:
                reader_failure.set()
                with suppress(Exception):
                    set_event(signal_handle)
            finally:
                reader_handle_ready[index].set()

        reader_specs = (
            (
                0,
                _bounded_reader,
                (0, stdout_read_handle, stdout_buffer, stdout_limit_bytes),
                "private-probe-stdout-reader",
            ),
            (
                1,
                _bounded_reader,
                (1, stderr_read_handle, stderr_buffer, stderr_limit_bytes),
                "private-probe-stderr-reader",
            ),
        )
        for index, reader_target, reader_arguments, reader_name in reader_specs:
            reader_thread = threading.Thread(
                target=reader_target,
                args=reader_arguments,
                name=reader_name,
                daemon=True,
            )
            reader_thread.start()
            reader_threads.append((index, reader_thread))
        for ready in reader_handle_ready:
            if not ready.wait(max(0.0, deadline - time.monotonic())):
                operation_failed = True
                break

        if not operation_failed:
            wait_handles = (ctypes.c_void_p * 2)(process_handle, signal_handle)
            while True:
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    operation_failed = True
                    break
                remaining_milliseconds = max(
                    1,
                    math.ceil(remaining_seconds * 1_000),
                )
                wait_result = wait_for_multiple_objects(
                    2,
                    wait_handles,
                    0,
                    remaining_milliseconds,
                )
                if wait_result == wait_object_0:
                    exit_code = ctypes.c_uint32(0)
                    if get_exit_code(process_handle, ctypes.byref(exit_code)) == 0:
                        operation_failed = True
                    else:
                        process_returncode = int(exit_code.value)
                    break
                if wait_result == wait_object_0 + 1 or wait_result == wait_timeout:
                    operation_failed = True
                    break
                operation_failed = True
                break
    except Exception:
        operation_failed = True
    finally:
        stdout_write_handle, stdout_write_closed = _release_owned_handle(
            stdout_write_handle
        )
        stderr_write_handle, stderr_write_closed = _release_owned_handle(
            stderr_write_handle
        )
        stdin_handle, stdin_closed = _release_owned_handle(stdin_handle)
        if not all((stdout_write_closed, stderr_write_closed, stdin_closed)):
            cleanup_complete = False
        if attribute_initialized and delete_attribute_list is not None:
            try:
                delete_attribute_list(attribute_pointer)
            except Exception:
                cleanup_complete = False
            attribute_initialized = False
        attribute_pointer = None
        attribute_buffer = None
        thread_handle, thread_closed = _release_owned_handle(thread_handle)
        if not thread_closed:
            cleanup_complete = False

        cleanup_deadline = (
            time.monotonic() + WINDOWS_PRIVATE_PROCESS_CLEANUP_TIMEOUT_SECONDS
        )
        if process_created and (job_handle is None or terminate_job is None):
            cleanup_complete = False
        if process_created and job_handle is not None and terminate_job is not None:
            try:
                termination_requested = (
                    terminate_job(job_handle, terminate_exit_code) != 0
                )
            except Exception:
                termination_requested = False
            if not termination_requested:
                cleanup_complete = False
            if process_handle is None or wait_for_single_object is None:
                cleanup_complete = False
            else:
                remaining_cleanup_ms = max(
                    1,
                    math.ceil((cleanup_deadline - time.monotonic()) * 1_000),
                )
                try:
                    root_stopped = (
                        wait_for_single_object(
                            process_handle,
                            remaining_cleanup_ms,
                        )
                        == wait_object_0
                    )
                except Exception:
                    root_stopped = False
                if not root_stopped:
                    cleanup_complete = False
            active_processes_zero = False
            if query_job is not None:
                while time.monotonic() < cleanup_deadline:
                    accounting = _JobBasicAccountingInformation()
                    try:
                        query_succeeded = (
                            query_job(
                                job_handle,
                                job_object_basic_accounting_information,
                                ctypes.byref(accounting),
                                ctypes.sizeof(accounting),
                                None,
                            )
                            != 0
                        )
                    except Exception:
                        query_succeeded = False
                    if not query_succeeded:
                        break
                    if accounting.ActiveProcesses == 0:
                        active_processes_zero = True
                        break
                    time.sleep(0.001)
            if not active_processes_zero:
                cleanup_complete = False
        job_handle, job_closed = _release_owned_handle(job_handle)
        if not job_closed:
            cleanup_complete = False

        for _, reader_thread in reader_threads:
            reader_thread.join(
                min(0.25, max(0.0, cleanup_deadline - time.monotonic()))
            )
        while (
            any(reader_thread.is_alive() for _, reader_thread in reader_threads)
            and time.monotonic() < cleanup_deadline
        ):
            for index, reader_thread in reader_threads:
                if not reader_thread.is_alive():
                    continue
                duplicated_thread = reader_thread_handles[index]
                if cancel_synchronous_io is None or duplicated_thread is None:
                    cleanup_complete = False
                    continue
                reader_cancel_requested[index].set()
                ctypes.set_last_error(0)
                try:
                    cancellation_result = cancel_synchronous_io(duplicated_thread)
                    cancellation_error = ctypes.get_last_error()
                except Exception:
                    cancellation_result = 0
                    cancellation_error = 0
                if cancellation_result == 0 and cancellation_error != 1168:
                    cleanup_complete = False
            for _, reader_thread in reader_threads:
                reader_thread.join(
                    min(0.05, max(0.0, cleanup_deadline - time.monotonic()))
                )
        readers_stopped = not any(
            reader_thread.is_alive() for _, reader_thread in reader_threads
        )
        if not readers_stopped:
            cleanup_complete = False
        if readers_stopped:
            stdout_read_handle, stdout_read_closed = _release_owned_handle(
                stdout_read_handle
            )
            stderr_read_handle, stderr_read_closed = _release_owned_handle(
                stderr_read_handle
            )
            signal_handle, signal_closed = _release_owned_handle(signal_handle)
            if not all(
                (stdout_read_closed, stderr_read_closed, signal_closed)
            ):
                cleanup_complete = False
            for index, duplicated_thread in enumerate(reader_thread_handles):
                (
                    reader_thread_handles[index],
                    duplicated_thread_closed,
                ) = _release_owned_handle(duplicated_thread)
                if not duplicated_thread_closed:
                    cleanup_complete = False
        process_handle, process_closed = _release_owned_handle(process_handle)
        if not process_closed:
            cleanup_complete = False
        for remaining_handle in (
            stdout_write_handle,
            stderr_write_handle,
            stdin_handle,
            thread_handle,
            job_handle,
            process_handle,
        ):
            _, retry_closed = _release_owned_handle(remaining_handle)
            if not retry_closed:
                cleanup_complete = False

    if (
        not operation_failed
        and cleanup_complete
        and process_returncode is not None
        and not output_overflow.is_set()
        and not reader_failure.is_set()
    ):
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=process_returncode,
            stdout=bytes(stdout_buffer),
            stderr=bytes(stderr_buffer),
        )
    stdout_buffer[:] = b"\x00" * len(stdout_buffer)
    stderr_buffer[:] = b"\x00" * len(stderr_buffer)
    stdout_buffer.clear()
    stderr_buffer.clear()
    raise OODExternalV2IntegrityError(failure_message) from None


def _run_bounded_windows_process(
    command: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
    stdout_limit_bytes: int,
    stderr_limit_bytes: int,
    failure_message: str,
) -> subprocess.CompletedProcess[bytes]:
    """Erase bounded streams and normalize every executor exit, including Ctrl+C."""

    stdout_buffer = bytearray()
    stderr_buffer = bytearray()
    completed: subprocess.CompletedProcess[bytes] | None = None
    try:
        with suppress(BaseException):
            completed = _run_bounded_windows_process_inner(
                command,
                cwd=cwd,
                environment=environment,
                timeout_seconds=timeout_seconds,
                stdout_limit_bytes=stdout_limit_bytes,
                stderr_limit_bytes=stderr_limit_bytes,
                failure_message=failure_message,
                stdout_buffer=stdout_buffer,
                stderr_buffer=stderr_buffer,
            )
    finally:
        stdout_buffer[:] = b"\x00" * len(stdout_buffer)
        stderr_buffer[:] = b"\x00" * len(stderr_buffer)
        stdout_buffer.clear()
        stderr_buffer.clear()
    if completed is None:
        raise OODExternalV2IntegrityError(failure_message) from None
    return completed


def _verify_git_credential_manager(
    executable: Path,
    *,
    project_root: Path,
) -> None:
    helper = _git_credential_manager_path(executable)
    completed = _run_bounded_windows_process(
        [os.fspath(helper), "--version"],
        cwd=helper.parent,
        environment=_private_live_remote_environment(executable),
        timeout_seconds=GCM_VERSION_TIMEOUT_SECONDS,
        stdout_limit_bytes=GCM_VERSION_STDOUT_LIMIT_BYTES,
        stderr_limit_bytes=GCM_VERSION_STDERR_LIMIT_BYTES,
        failure_message="frozen Git credential manager probe failed",
    )
    differs = (
        completed.returncode != 0
        or completed.stderr != b""
        or completed.stdout != EXPECTED_GIT_CREDENTIAL_MANAGER_VERSION_STDOUT
    )
    del completed
    _remove_exact_empty_gcm_sentinel_directory(project_root)
    if differs:
        raise OODExternalV2IntegrityError("frozen Git credential manager differs")


def _verify_git_runtime_tree_before_provenance() -> None:
    global _GIT_RUNTIME_TREE_VERIFIED
    if _GIT_RUNTIME_TREE_VERIFIED:
        return
    _, _, install_root = _git_executable_paths()
    observed = _runtime_filesystem_tree(
        install_root / "mingw64",
        tree_kind=RUNTIME_FILESYSTEM_TREE_KINDS[2],
    )
    if (
        observed.file_count != EXPECTED_GIT_RUNTIME_FILE_COUNT
        or observed.directory_count != EXPECTED_GIT_RUNTIME_DIRECTORY_COUNT
        or observed.total_bytes != EXPECTED_GIT_RUNTIME_TOTAL_BYTES
        or observed.tree_sha256 != EXPECTED_GIT_RUNTIME_TREE_SHA256
    ):
        raise OODExternalV2IntegrityError(
            "Git runtime tree differs before provenance verification"
        )
    _GIT_RUNTIME_TREE_VERIFIED = True


def _verify_git_repository_controls(project_root: Path) -> None:
    git_directory = _assert_direct_ancestry(
        project_root / ".git",
        context="Git metadata directory",
    )
    if not git_directory.is_dir():
        raise OODExternalV2IntegrityError("Git metadata is not a direct directory")
    forbidden = (
        git_directory / "objects" / "info" / "alternates",
        git_directory / "info" / "grafts",
        git_directory / "info" / "sparse-checkout",
        git_directory / "shallow",
        git_directory / "shallow.lock",
        git_directory / "refs" / "replace",
        git_directory / "config.worktree",
        git_directory / "worktrees",
    )
    if any(path.exists() or _is_indirect(path) for path in forbidden):
        raise OODExternalV2IntegrityError(
            "Git object alternates, grafts, shallow state, and replacement refs are forbidden"
        )
    config = _read_bounded(git_directory / "config", 1_000_000, "local Git config")
    try:
        config_text = config.decode("utf-8")
    except UnicodeError as error:
        raise OODExternalV2IntegrityError("local Git config is not UTF-8") from error
    expected_sections: Mapping[str, Mapping[str, str | None]] = {
        "core": {
            "bare": "false",
            "filemode": "false",
            "ignorecase": "true",
            "logallrefupdates": "true",
            "repositoryformatversion": "0",
        },
        'remote "origin"': {
            "fetch": "+refs/heads/*:refs/remotes/origin/*",
            "url": EXPECTED_GIT_REMOTE_URL,
        },
        'branch "main"': {
            "merge": "refs/heads/main",
            "remote": "origin",
        },
        "user": {"email": None, "name": None},
    }
    parsed: dict[str, dict[str, str]] = {}
    current_section: str | None = None
    for raw_line in config_text.splitlines():
        if not raw_line:
            continue
        header = re.fullmatch(r"\[([^\]\r\n]+)\]", raw_line)
        if header is not None:
            current_section = header.group(1)
            if current_section not in expected_sections or current_section in parsed:
                raise OODExternalV2IntegrityError(
                    "local Git config contains an unapproved section"
                )
            parsed[current_section] = {}
            continue
        entry = re.fullmatch(r"\t([a-z]+) = ([^\x00\r\n]+)", raw_line)
        if entry is None or current_section is None:
            raise OODExternalV2IntegrityError(
                "local Git config is not in the exact frozen syntax"
            )
        key, value = entry.groups()
        if key in parsed[current_section]:
            raise OODExternalV2IntegrityError(
                "local Git config contains a duplicate key"
            )
        parsed[current_section][key] = value
    if set(parsed) != set(expected_sections):
        raise OODExternalV2IntegrityError(
            "local Git config section set differs from the frozen repository"
        )
    for section, expected_values in expected_sections.items():
        observed_values = parsed[section]
        if set(observed_values) != set(expected_values):
            raise OODExternalV2IntegrityError(
                "local Git config key set differs from the frozen repository"
            )
        for key, expected_value in expected_values.items():
            observed_value = observed_values[key]
            if expected_value is None:
                if not observed_value or observed_value != observed_value.strip():
                    raise OODExternalV2IntegrityError(
                        "local Git identity value is not canonical"
                    )
            elif observed_value != expected_value:
                raise OODExternalV2IntegrityError(
                    "local Git config value differs from the frozen repository"
                )
    if config_text.encode("utf-8") != config:
        raise OODExternalV2IntegrityError(
            "local Git config bytes changed while parsed"
        )


def _bound_git_command(
    project_root: Path,
    executable: Path,
    *arguments: str,
) -> list[str]:
    git_directory = project_root / ".git"
    return [
        os.fspath(executable),
        "--no-pager",
        "--no-replace-objects",
        f"--git-dir={git_directory}",
        f"--work-tree={project_root}",
        "-c",
        "core.hooksPath=NUL" if os.name == "nt" else "core.hooksPath=/dev/null",
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
        *arguments,
    ]


def _execute_bound_git(
    project_root: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[bytes]:
    _verify_git_runtime_tree_before_provenance()
    _, executable, _ = _git_executable_paths()
    _verify_git_repository_controls(project_root)
    command = _bound_git_command(project_root, executable, *arguments)
    try:
        completed = subprocess.run(
            command,
            cwd=project_root,
            check=False,
            capture_output=True,
            env=_sanitized_git_environment(executable),
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise OODExternalV2IntegrityError("Git preflight could not be executed") from error
    if completed.stderr:
        raise OODExternalV2IntegrityError("Git preflight emitted unexpected stderr")
    if len(completed.stdout) > 8_000_000:
        raise OODExternalV2IntegrityError("Git preflight output exceeds its bound")
    return completed


def _verify_private_remote_anonymous_denial(project_root: Path) -> None:
    """Require the exact HTTPS Git endpoint to reject a credentialless read."""

    _verify_git_runtime_tree_before_provenance()
    _, executable, _ = _git_executable_paths()
    _verify_git_repository_controls(project_root)
    command, environment = _private_remote_command(
        project_root,
        executable,
        authenticated=False,
    )
    completed = _run_bounded_windows_process(
        command,
        cwd=executable.parent,
        environment=environment,
        timeout_seconds=PRIVATE_REMOTE_TIMEOUT_SECONDS,
        stdout_limit_bytes=PRIVATE_REMOTE_STDOUT_LIMIT_BYTES,
        stderr_limit_bytes=PRIVATE_REMOTE_STDERR_LIMIT_BYTES,
        failure_message="private Git remote anonymous-access probe failed",
    )
    denied = (
        completed.returncode == 128
        and completed.stdout == b""
        and completed.stderr == EXPECTED_PRIVATE_REMOTE_ANONYMOUS_STDERR
    )
    del completed
    if not denied:
        raise OODExternalV2IntegrityError(
            "private Git remote anonymous-access denial was not proven"
        )


def _run_exact_private_live_remote(project_root: Path) -> str:
    """Read the one exact private GitHub ref advertisement without exposing auth."""

    _verify_git_runtime_tree_before_provenance()
    _, executable, _ = _git_executable_paths()
    _verify_git_repository_controls(project_root)
    _verify_git_credential_manager(executable, project_root=project_root)
    command, environment = _private_remote_command(
        project_root,
        executable,
        authenticated=True,
    )
    completed = _run_bounded_windows_process(
        command,
        cwd=executable.parent,
        environment=environment,
        timeout_seconds=PRIVATE_REMOTE_TIMEOUT_SECONDS,
        stdout_limit_bytes=PRIVATE_REMOTE_STDOUT_LIMIT_BYTES,
        stderr_limit_bytes=PRIVATE_REMOTE_STDERR_LIMIT_BYTES,
        failure_message="private Git remote preflight could not be executed",
    )
    returncode = completed.returncode
    stderr_empty = completed.stderr == b""
    stdout = completed.stdout
    del completed
    try:
        _remove_exact_empty_gcm_sentinel_directory(project_root)
    except BaseException:
        stdout = b""
        raise
    if not stderr_empty:
        stdout = b""
        raise OODExternalV2IntegrityError(
            "private Git remote preflight emitted unexpected stderr"
        )
    if returncode != 0:
        stdout = b""
        raise OODExternalV2IntegrityError("private Git remote preflight failed")
    if len(stdout) > 4_096:
        stdout = b""
        raise OODExternalV2IntegrityError(
            "private Git remote preflight output exceeds its bound"
        )
    decoded: str | None = None
    with suppress(UnicodeError):
        decoded = stdout.decode("utf-8", errors="strict")
    stdout = b""
    if decoded is None:
        raise OODExternalV2IntegrityError(
            "private Git remote preflight output is not UTF-8"
        ) from None
    return decoded


def _run_git(
    project_root: Path,
    *arguments: str,
    allow_empty: bool = False,
) -> subprocess.CompletedProcess[str]:
    completed = _execute_bound_git(project_root, *arguments)
    if completed.returncode != 0 and not allow_empty:
        raise OODExternalV2IntegrityError("Git preflight command failed")
    try:
        stdout = completed.stdout.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise OODExternalV2IntegrityError("Git preflight output is not UTF-8") from error
    return subprocess.CompletedProcess(
        args=completed.args,
        returncode=completed.returncode,
        stdout=stdout,
        stderr="",
    )


def _normalize_unprefixed(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise OODExternalV2IntegrityError(f"{context} must be text")
    normalized = value.removeprefix("sha256:")
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise OODExternalV2IntegrityError(f"{context} must be a SHA-256 digest")
    return normalized


def _raw_source_binding_for_path(
    root: Path,
    relative_path: str,
    *,
    context: str,
    official_md5: str | None,
    content_access_witness: Callable[[], None] | None = None,
) -> RawSourceBinding:
    if content_access_witness is not None and not callable(content_access_witness):
        raise TypeError("content_access_witness must be callable or None")
    path = _resolve_project_relative(root, relative_path, require_file=True)
    try:
        size = path.stat().st_size
    except OSError as error:
        raise OODExternalV2IntegrityError(f"{context} cannot be inspected") from error
    content_accessed = False

    def witness_content_access() -> None:
        nonlocal content_accessed
        if not content_accessed and content_access_witness is not None:
            content_access_witness()
        content_accessed = True

    if (
        official_md5 is not None
        and _md5_file(
                path,
                content_access_witness=witness_content_access,
            )
        != official_md5
    ):
        raise OODExternalV2IntegrityError(f"{context} official MD5 differs")
    if not content_accessed:
        try:
            with path.open("rb") as handle:
                handle.read(1)
                witness_content_access()
        except OSError as error:
            raise OODExternalV2IntegrityError(
                f"{context} content cannot be opened"
            ) from error
    return RawSourceBinding(
        relative_path=relative_path,
        file_sha256=sha256_file(path),
        size_bytes=_positive_integer(size, f"{context} size"),
        official_md5=official_md5,
    )


def _md5_file(
    path: Path,
    *,
    content_access_witness: Callable[[], None] | None = None,
) -> str:
    if content_access_witness is not None and not callable(content_access_witness):
        raise TypeError("content_access_witness must be callable or None")
    digest = hashlib.md5(usedforsecurity=False)
    try:
        with path.open("rb") as handle:
            first_read = True
            while True:
                chunk = handle.read(1024 * 1024)
                if first_read:
                    if content_access_witness is not None:
                        content_access_witness()
                    first_read = False
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as error:
        raise OODExternalV2IntegrityError("official-MD5 source cannot be read") from error
    return "md5:" + digest.hexdigest()


def _freeze_decision_binding(
    root: Path,
    relative_path: str,
    *,
    expected_file_sha256: str,
) -> BoundFile:
    path = _resolve_project_relative(root, relative_path, require_file=True)
    observed = sha256_file(path)
    if observed != expected_file_sha256:
        raise OODExternalV2IntegrityError("frozen decision file hash differs")
    raw = _read_bounded(path, _V1_RESULT_MAX_BYTES, "frozen decision file")
    try:
        payload: object = json.loads(
            raw[:-1].decode("ascii") if raw.endswith(b"\n") else raw.decode("ascii"),
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise OODExternalV2IntegrityError("frozen decision file is not JSON") from error
    artifact_sha256: str | None = None
    if isinstance(payload, Mapping) and "artifact_sha256" in payload:
        artifact_sha256 = _digest(payload["artifact_sha256"], "decision artifact")
    return BoundFile(
        relative_path=relative_path,
        file_sha256=observed,
        artifact_sha256=artifact_sha256,
    )


def _verify_child_freeze_decision_bindings(
    root: Path,
) -> Mapping[str, BoundFile]:
    """Validate the two real decision files without fabricating demo identity."""

    demo_relative_path = EXPECTED_DEMO_POLICY_PATH
    source_relative_path = EXPECTED_SOURCE_CALIBRATION_PATH
    demo = _freeze_decision_binding(
        root,
        demo_relative_path,
        expected_file_sha256=EXPECTED_DEMO_POLICY_FILE_SHA256,
    )
    source = _freeze_decision_binding(
        root,
        source_relative_path,
        expected_file_sha256=EXPECTED_SOURCE_CALIBRATION_FILE_SHA256,
    )
    if demo.artifact_sha256 is not None:
        raise OODExternalV2IntegrityError(
            "historical demo policy must not assert a logical artifact identity"
        )
    if source.artifact_sha256 != EXPECTED_SOURCE_CALIBRATION_ARTIFACT_SHA256:
        raise OODExternalV2IntegrityError(
            "source-calibration logical artifact differs from its frozen identity"
        )
    demo_path = _resolve_project_relative(root, demo_relative_path, require_file=True)
    source_path = _resolve_project_relative(root, source_relative_path, require_file=True)
    try:
        FrozenDecisionPolicy.load(demo_path)
        source_result = load_source_calibration_result_bytes(
            _read_bounded(
                source_path,
                _V1_RESULT_MAX_BYTES,
                "source-calibration decision file",
            )
        )
    except Exception as error:
        raise OODExternalV2IntegrityError(
            "frozen decision file violates its exact model"
        ) from error
    if (
        source_result.artifact_sha256 != EXPECTED_SOURCE_CALIBRATION_ARTIFACT_SHA256
        or sha256_file(demo_path) != demo.file_sha256
        or sha256_file(source_path) != source.file_sha256
    ):
        raise OODExternalV2IntegrityError(
            "frozen decision file changed during model validation"
        )
    return MappingProxyType(
        {
            "demo_policy": demo,
            "source_calibration_result": source,
        }
    )


def _freeze_child_runtime_bindings(root: Path) -> Mapping[str, str]:
    bindings = {
        relative_path: sha256_file(
            _resolve_project_relative(root, relative_path, require_file=True)
        )
        for relative_path in REQUIRED_RUNTIME_BINDING_PATHS
    }
    if tuple(bindings) != REQUIRED_RUNTIME_BINDING_PATHS:
        raise OODExternalV2IntegrityError(
            "child runtime bindings differ from the exact evaluator set"
        )
    return MappingProxyType(bindings)


def _bound_file_dict(value: BoundFile) -> dict[str, object]:
    payload: dict[str, object] = {
        "file_sha256": value.file_sha256,
        "relative_path": value.relative_path,
    }
    if value.artifact_sha256 is not None:
        payload["artifact_sha256"] = value.artifact_sha256
    return payload


def _verify_public_projection_file(
    path: Path,
    *,
    inventory: ExternalWaveformInventory,
    challenge_records: int,
    zzu_records: int,
    expected_counts: SuccessorInventoryCountBinding,
) -> str:
    raw = _read_bounded(path, _CHILD_MAX_BYTES, "public inventory projection")
    try:
        decoded: object = json.loads(
            raw[:-1].decode("ascii") if raw.endswith(b"\n") else b"".decode(),
            object_pairs_hook=_unique_json_object,
        )
    except OODExternalV2ConfigError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise OODExternalV2IntegrityError(
            "public inventory projection is not canonical JSON"
        ) from error
    if not isinstance(decoded, dict) or canonical_json_bytes(decoded) != raw:
        raise OODExternalV2IntegrityError(
            "public inventory projection is not in exact canonical form"
        )
    payload = cast(dict[str, object], decoded)
    expected = {
        "challenge_record_count",
        "inventory",
        "kind",
        "projection_sha256",
        "schema_version",
        "zzu_candidate_patient_count",
        "zzu_inventory_build_summary",
    }
    if (
        set(payload) != expected
        or payload["kind"] != "ecg_trust.ood_v2.inventory_publication"
        or payload["schema_version"] != 1
        or payload["challenge_record_count"] != challenge_records
        or payload["inventory"] != external_inventory_public_projection(inventory)
    ):
        raise OODExternalV2IntegrityError("public inventory projection differs")
    summary = _mapping(
        payload["zzu_inventory_build_summary"],
        "public ZZU inventory summary",
    )
    if (
        summary.get("dataset") != ZZU_PEDIATRIC_DATASET
        or summary.get("selected_record_count") != zzu_records
        or summary.get("candidate_record_count")
        != expected_counts.zzu_candidate_records
        or summary.get("exclusion_counts")
        != dict(expected_counts.zzu_exclusion_counts)
        or payload.get("zzu_candidate_patient_count")
        != expected_counts.zzu_candidate_patients
        or challenge_records != expected_counts.challenge_records
        or zzu_records != expected_counts.zzu_records
        or len(inventory.records) != expected_counts.total_records
        or len(
            {
                record.patient_key
                for record in inventory.records
                if record.dataset == ZZU_PEDIATRIC_DATASET
                and record.patient_key is not None
            }
        )
        != expected_counts.zzu_patients
    ):
        raise OODExternalV2IntegrityError("public ZZU inventory summary differs")
    body = dict(payload)
    claimed = _digest(body.pop("projection_sha256"), "public projection")
    domain = b"ecg_trust.ood_v2.inventory_publication.v1\x00"
    observed = "sha256:" + hashlib.sha256(
        domain + canonical_json_bytes(body)[:-1]
    ).hexdigest()
    if claimed != observed:
        raise OODExternalV2IntegrityError("public projection logical hash differs")
    return claimed


def _decode_utf8_metadata(path: Path, *, context: str) -> str:
    raw = _read_bounded(path, 64 * 1024 * 1024, context)
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise OODExternalV2IntegrityError(f"{context} must be UTF-8") from error


def _official_record_lines(text: str, *, context: str) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(text.splitlines(), start=1):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        try:
            canonical = _relative_path(value, f"{context} line {line_number}")
        except OODExternalV2ConfigError as error:
            raise OODExternalV2IntegrityError(
                f"{context} contains an unsafe record reference"
            ) from error
        if canonical in seen:
            raise OODExternalV2IntegrityError(f"{context} contains a duplicate record")
        seen.add(canonical)
        values.append(canonical)
    if not values:
        raise OODExternalV2IntegrityError(f"{context} contains no records")
    return tuple(values)


def _verify_role_metadata_rejoin(
    inventory: ExternalWaveformInventory,
    *,
    dataset_roots: Mapping[str, Path],
    raw_source_paths: Mapping[str, Path],
    public_projection: Path | None,
    parent: OODExternalV2ParentConfig,
) -> None:
    challenge = tuple(
        record for record in inventory.records if record.dataset == CHALLENGE_2011_DATASET
    )
    all_text = _decode_utf8_metadata(
        raw_source_paths["challenge_records"],
        context="Challenge RECORDS",
    )
    acceptable_text = _decode_utf8_metadata(
        raw_source_paths["challenge_records_acceptable"],
        context="Challenge acceptable list",
    )
    unacceptable_text = _decode_utf8_metadata(
        raw_source_paths["challenge_records_unacceptable"],
        context="Challenge unacceptable list",
    )
    official_order = _official_record_lines(all_text, context="Challenge RECORDS")
    try:
        official_labels = parse_challenge_2011_quality_lists(
            all_text,
            acceptable_text,
            unacceptable_text,
            expected_record_count=parent.challenge_expected_records,
        )
        validate_challenge_2011_set_a_inventory(
            build_external_inventory(challenge),
            expected_record_count=parent.challenge_expected_records,
            expected_quality_by_record=official_labels,
        )
    except Exception as error:
        raise OODExternalV2IntegrityError(
            "Challenge role/quality metadata does not rejoin to inventory"
        ) from error
    if (
        tuple(record.record_ref for record in challenge) != official_order
        or any(record.patient_key is not None for record in challenge)
    ):
        raise OODExternalV2IntegrityError(
            "Challenge record order or null-patient contract differs"
        )

    attributes_text = _decode_utf8_metadata(
        raw_source_paths["zzu_attributes_dictionary"],
        context="ZZU AttributesDictionary",
    )
    try:
        candidates = parse_zzu_pediatric_attributes_csv(
            attributes_text,
            site="Zhengzhou University pediatric ECG",
            site_alias="zzu-pecg",
            expected_record_count=14_190,
            expected_patient_count=11_643,
        )
        verify_wfdb_candidate_file_set(
            dataset_roots[ZZU_PEDIATRIC_DATASET],
            tuple(candidate.record_ref for candidate in candidates),
        )
        selected, summary = select_zzu_pediatric_inventory_records(
            dataset_roots[ZZU_PEDIATRIC_DATASET],
            candidates,
        )
    except Exception as error:
        raise OODExternalV2IntegrityError(
            "ZZU candidate/header selection cannot be rederived"
        ) from error
    zzu = tuple(
        record for record in inventory.records if record.dataset == ZZU_PEDIATRIC_DATASET
    )
    if selected != zzu:
        raise OODExternalV2IntegrityError(
            "ZZU all-and-only selection or patient mapping differs from inventory"
        )
    if public_projection is None:
        raise OODExternalV2IntegrityError(
            "public projection is required to bind ZZU exclusion accounting"
        )
    projection_raw = _read_bounded(
        public_projection,
        _CHILD_MAX_BYTES,
        "public inventory projection",
    )
    try:
        projection: object = json.loads(projection_raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise OODExternalV2IntegrityError("public projection cannot be decoded") from error
    if not isinstance(projection, dict) or (
        projection.get("zzu_inventory_build_summary") != summary.to_dict()
    ):
        raise OODExternalV2IntegrityError(
            "ZZU exclusion accounting differs from rederived headers"
        )


def _tensor_sha256(value: NDArray[np.generic]) -> str:
    array = np.ascontiguousarray(value)
    header = canonical_json_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )[:-1]
    digest = hashlib.sha256(b"ecg_trust.tensor.v1\x00" + header + array.tobytes())
    return "sha256:" + digest.hexdigest()


def _atomic_write_new(
    path: Path,
    payload: bytes,
    *,
    visibility_witness: Callable[[], None] | None = None,
    publication_witness: Callable[[], None] | None = None,
    expected_parent_identity: _OwnedDirectoryIdentity | None = None,
    ownership_verifier: Callable[[], None] | None = None,
) -> None:
    if not isinstance(payload, bytes) or not payload:
        raise OODExternalV2ExecutionError("immutable artifact payload is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    if expected_parent_identity is not None and (
        _owned_directory_identity(path.parent) != expected_parent_identity
    ):
        raise OODExternalV2ExecutionError(
            "immutable artifact parent is not owned by this execution"
        )
    if ownership_verifier is not None:
        ownership_verifier()
    if path.exists() or _is_indirect(path):
        raise OODExternalV2ExecutionError("immutable artifact already exists")
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(raw_temp)
    temporary_exists = True
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if expected_parent_identity is not None and (
            _owned_directory_identity(path.parent) != expected_parent_identity
        ):
            raise OODExternalV2ExecutionError(
                "immutable artifact parent changed before publication"
            )
        if ownership_verifier is not None:
            ownership_verifier()
        os.link(temporary, path)
        if visibility_witness is not None:
            visibility_witness()
        if expected_parent_identity is not None and (
            _owned_directory_identity(path.parent) != expected_parent_identity
        ):
            raise OODExternalV2ExecutionError(
                "immutable artifact parent changed during publication"
            )
        if ownership_verifier is not None:
            ownership_verifier()
        temporary.unlink()
        temporary_exists = False
        _fsync_directory(path.parent)
        if expected_parent_identity is not None and (
            _owned_directory_identity(path.parent) != expected_parent_identity
        ):
            raise OODExternalV2ExecutionError(
                "immutable artifact parent changed after durability flush"
            )
        if ownership_verifier is not None:
            ownership_verifier()
        if publication_witness is not None:
            publication_witness()
    except FileExistsError as error:
        raise OODExternalV2ExecutionError("immutable artifact already exists") from error
    except OSError as error:
        raise OODExternalV2ExecutionError("atomic artifact commit failed") from error
    finally:
        if temporary_exists:
            with suppress(OSError):
                temporary.unlink()


def _atomic_npz_new(
    path: Path,
    arrays: Mapping[str, NDArray[np.generic]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or _is_indirect(path):
        raise OODExternalV2ExecutionError("immutable NPZ artifact already exists")
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(raw_temp)
    temporary_exists = True
    try:
        with os.fdopen(descriptor, "w+b") as handle:
            savez = cast(Any, np.savez)
            savez(handle, **dict(arrays))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        temporary_exists = False
        _fsync_directory(path.parent)
    except FileExistsError as error:
        raise OODExternalV2ExecutionError("immutable NPZ artifact already exists") from error
    except (OSError, ValueError) as error:
        raise OODExternalV2ExecutionError("atomic NPZ commit failed") from error
    finally:
        if temporary_exists:
            with suppress(OSError):
                temporary.unlink()


def _safe_npz_members(path: Path) -> None:
    try:
        archive_size = path.stat().st_size
        if archive_size <= 0 or archive_size > _PRIVATE_NPZ_MAX_BYTES:
            raise OODExternalV2IntegrityError("private NPZ size is invalid")
        with zipfile.ZipFile(path, "r") as archive:
            items = archive.infolist()
            names = [item.filename for item in items]
            if (
                not items
                or len(items) > _PRIVATE_NPZ_MEMBER_COUNT_MAX
                or len(names) != len(set(names))
                or len({name.casefold() for name in names}) != len(names)
                or archive.comment
            ):
                raise OODExternalV2IntegrityError(
                    "private NPZ member inventory is invalid"
                )
            total_uncompressed = 0
            for item in items:
                member = PurePosixPath(item.filename)
                if (
                    item.flag_bits & 0x1
                    or item.compress_type != zipfile.ZIP_STORED
                    or item.compress_size != item.file_size
                    or item.file_size <= 0
                    or item.file_size > _PRIVATE_NPZ_MEMBER_MAX_BYTES
                    or member.is_absolute()
                    or any(part in {"", ".", ".."} for part in member.parts)
                    or len(member.parts) != 1
                    or member.suffix != ".npy"
                ):
                    raise OODExternalV2IntegrityError(
                        "private NPZ contains an unsafe member"
                    )
                total_uncompressed += item.file_size
                if total_uncompressed > _PRIVATE_NPZ_MAX_BYTES:
                    raise OODExternalV2IntegrityError(
                        "private NPZ uncompressed payload exceeds its bound"
                    )
            # ZIP_STORED permits a tight expansion check: member bytes cannot
            # exceed the archive by more than central-directory overhead.
            if total_uncompressed > archive_size:
                raise OODExternalV2IntegrityError(
                    "private NPZ stored member sizes are inconsistent"
                )
    except OODExternalV2IntegrityError:
        raise
    except (OSError, zipfile.BadZipFile) as error:
        raise OODExternalV2IntegrityError("private NPZ is invalid") from error


def _verify_embedding_npz(path: Path, *, expected_records: int) -> None:
    _safe_npz_members(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "dataset",
                "embedding_first",
                "embedding_repeated",
                "logits_first",
                "logits_repeated",
                "patient_key",
                "probabilities",
                "record_ref",
                "score",
            }:
                raise OODExternalV2IntegrityError(
                    "embedding NPZ fields differ from protocol"
                )
            embedding = archive["embedding_first"]
            repeated_embedding = archive["embedding_repeated"]
            logits = archive["logits_first"]
            repeated_logits = archive["logits_repeated"]
            probabilities = archive["probabilities"]
            score = archive["score"]
            if (
                embedding.shape != (expected_records, 512)
                or embedding.dtype != np.dtype(np.float32)
                or repeated_embedding.shape != (expected_records, 512)
                or repeated_embedding.dtype != np.dtype(np.float32)
                or logits.shape != (expected_records, len(SUPERCLASSES))
                or logits.dtype != np.dtype(np.float64)
                or repeated_logits.shape
                != (expected_records, len(SUPERCLASSES))
                or repeated_logits.dtype != np.dtype(np.float64)
                or probabilities.shape != (expected_records, len(SUPERCLASSES))
                or probabilities.dtype != np.dtype(np.float64)
                or score.shape != (expected_records,)
                or score.dtype != np.dtype(np.float64)
                or not np.isfinite(embedding).all()
                or not np.isfinite(repeated_embedding).all()
                or not np.isfinite(logits).all()
                or not np.isfinite(repeated_logits).all()
                or not np.isfinite(probabilities).all()
                or np.any((probabilities < 0.0) | (probabilities > 1.0))
                or not np.isfinite(score).all()
            ):
                raise OODExternalV2IntegrityError("embedding NPZ arrays are invalid")
            for name in ("dataset", "patient_key", "record_ref"):
                values = archive[name]
                if values.shape != (expected_records,) or values.dtype.kind != "U":
                    raise OODExternalV2IntegrityError(
                        "embedding NPZ identity arrays are invalid"
                    )
    except OODExternalV2IntegrityError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise OODExternalV2IntegrityError("embedding NPZ cannot be verified") from error


def _verify_bootstrap_npz(
    path: Path,
    *,
    expected_names: tuple[str, ...],
    expected_replicates: int,
) -> None:
    _safe_npz_members(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if tuple(sorted(archive.files)) != tuple(sorted(expected_names)):
                raise OODExternalV2IntegrityError(
                    "bootstrap NPZ endpoints differ from result"
                )
            for name in expected_names:
                values = archive[name]
                if (
                    values.shape != (expected_replicates,)
                    or values.dtype != np.dtype(np.float64)
                    or not np.isfinite(values).all()
                    or np.any((values < 0.0) | (values > 1.0))
                ):
                    raise OODExternalV2IntegrityError(
                        "bootstrap NPZ replicate array is invalid"
                    )
    except OODExternalV2IntegrityError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise OODExternalV2IntegrityError("bootstrap NPZ cannot be verified") from error


def _external_claim_bytes(
    inputs: VerifiedExternalV2Inputs,
    *,
    owner_nonce: str,
) -> bytes:
    if _OWNER_NONCE.fullmatch(owner_nonce) is None:
        raise OODExternalV2IntegrityError("external claim nonce is invalid")
    return canonical_json_bytes(
        {
            "artifact_type": ACCESS_CLAIM_ARTIFACT_TYPE,
            "child_contract_file_sha256": inputs.child.file_sha256,
            "contains_embeddings_or_scores": False,
            "contains_record_or_patient_identifiers": False,
            "inventory_sha256": inputs.inventory.inventory_sha256,
            "owner_nonce": owner_nonce,
            "parent_config_file_sha256": inputs.parent.file_sha256,
            "protocol_id": PROTOCOL_ID,
            "schema_version": 1,
            "state": "EXTERNAL_ACCESS_CLAIMED",
        }
    )


def _external_marker_bytes(
    inputs: VerifiedExternalV2Inputs,
    *,
    owner_nonce: str,
    claim_file_sha256: str,
) -> bytes:
    _digest(claim_file_sha256, "external claim file")
    if _OWNER_NONCE.fullmatch(owner_nonce) is None:
        raise OODExternalV2IntegrityError("external marker nonce is invalid")
    return canonical_json_bytes(
        {
            "artifact_type": ACCESS_MARKER_ARTIFACT_TYPE,
            "child_contract_file_sha256": inputs.child.file_sha256,
            "contains_embeddings_or_scores": False,
            "contains_record_or_patient_identifiers": False,
            "external_claim_file_sha256": claim_file_sha256,
            "inventory_sha256": inputs.inventory.inventory_sha256,
            "owner_nonce": owner_nonce,
            "parent_config_file_sha256": inputs.parent.file_sha256,
            "protocol_id": PROTOCOL_ID,
            "schema_version": 1,
            "state": "EXTERNAL_ACCESS_ARMED",
        }
    )


def _verify_staged_members_before_manifest(
    staging: Path,
    *,
    inputs: VerifiedExternalV2Inputs,
    expected_result: OODV2Result,
    expected_claim_bytes: bytes,
    model: ResNet1D,
    normalization: NormalizationStats,
    runtime: DeterministicCUDARuntime,
) -> None:
    result_path = staging / OOD_V2_RESULT_FILENAME
    try:
        loaded_result = load_ood_v2_result_bytes(
            _read_bounded(result_path, _V1_RESULT_MAX_BYTES, "staged aggregate result")
        )
    except Exception as error:
        raise OODExternalV2IntegrityError("staged aggregate result is invalid") from error
    if loaded_result != expected_result:
        raise OODExternalV2IntegrityError("staged aggregate result changed")
    inventory = load_external_inventory(staging / "private" / "external-inventory.json")
    if inventory != inputs.inventory:
        raise OODExternalV2IntegrityError("staged private inventory changed")
    private_rows = _load_private_record_evidence(
        staging / "private" / "record-evidence.json",
        inputs=inputs,
    )
    quality_pass_rows = tuple(
        row for row in private_rows if row.quality_status == QualityStatus.PASS.value
    )
    _verify_raw_to_canonical_replay(
        staging / "private",
        inputs=inputs,
        expected_records=private_rows,
    )
    replayed_embeddings = _replay_quality_pass_embeddings(
        staging / "private",
        inputs=inputs,
        expected_records=private_rows,
        model=model,
        normalization=normalization,
        runtime=runtime,
    )
    _verify_embedding_bundle_semantics(
        staging / "private",
        inputs=inputs,
        quality_pass_rows=quality_pass_rows,
        frozen_model=model,
        frozen_runtime=runtime,
        replayed_embeddings=replayed_embeddings,
    )
    endpoint_names = tuple(
        sorted(
            [endpoint.endpoint_key for endpoint in expected_result.external_cohorts]
            + [
                endpoint.endpoint_key
                for endpoint in expected_result.technical_quality_endpoints
            ]
        )
    )
    _verify_bootstrap_bundle_semantics(
        staging / "private",
        inputs=inputs,
        result=loaded_result,
        private_rows=private_rows,
        endpoint_names=endpoint_names,
    )
    claim_path = _resolve_project_relative(
        inputs.project_root,
        inputs.parent.claim_path,
        require_file=True,
    )
    if _read_bounded(claim_path, _ACCESS_RECORD_MAX_BYTES, "external claim") != (
        expected_claim_bytes
    ):
        raise OODExternalV2IntegrityError("external claim changed before manifest")
    try:
        claim_payload: object = json.loads(expected_claim_bytes)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise OODExternalV2IntegrityError("external claim cannot be decoded") from error
    if not isinstance(claim_payload, dict) or not isinstance(
        claim_payload.get("owner_nonce"),
        str,
    ):
        raise OODExternalV2IntegrityError("external claim nonce is unavailable")
    expected_marker = _external_marker_bytes(
        inputs,
        owner_nonce=cast(str, claim_payload["owner_nonce"]),
        claim_file_sha256=sha256_bytes(expected_claim_bytes),
    )
    if _read_bounded(
        staging / ACCESS_MARKER_FILENAME,
        _ACCESS_RECORD_MAX_BYTES,
        "external marker",
    ) != expected_marker:
        raise OODExternalV2IntegrityError("external marker/claim binding differs")


def _verify_canonical_signal_sidecar(
    private: Path,
    *,
    inputs: VerifiedExternalV2Inputs,
    expected_records: tuple[_PrivateRecordEvidence, ...],
) -> tuple[dict[str, object], ...]:
    sidecar = _load_private_sidecar(
        private.parent / PurePosixPath(CANONICAL_SIGNAL_SIDECAR_PATH),
        artifact_type=PRIVATE_CANONICAL_SIGNAL_ARTIFACT_TYPE,
    )
    fields = {
        "artifact_sha256",
        "artifact_type",
        "canonical_dtype",
        "canonical_shape_per_record",
        "inventory_record_count",
        "inventory_sha256",
        "npz_file_sha256",
        "protocol_id",
        "schema_version",
        "shard_count",
        "shard_inventory_records",
        "shards",
        "successful_adapter_records",
    }
    raw_descriptors = sidecar.get("shards")
    successful_records = sum(
        row.adapter_provenance_sha256 is not None for row in expected_records
    )
    if (
        set(sidecar) != fields
        or sidecar.get("canonical_dtype") != "float32"
        or sidecar.get("canonical_shape_per_record") != [len(LEADS), TARGET_SAMPLES]
        or sidecar.get("inventory_record_count") != len(expected_records)
        or sidecar.get("inventory_sha256") != inputs.inventory.inventory_sha256
        or sidecar.get("shard_count") != CANONICAL_SIGNAL_SHARD_COUNT
        or sidecar.get("shard_inventory_records")
        != CANONICAL_SIGNAL_SHARD_RECORDS
        or sidecar.get("successful_adapter_records") != successful_records
        or not isinstance(raw_descriptors, list)
        or len(raw_descriptors) != CANONICAL_SIGNAL_SHARD_COUNT
    ):
        raise OODExternalV2IntegrityError("canonical signal sidecar differs")
    npz_path = private.parent / PurePosixPath(CANONICAL_SIGNAL_NPZ_PATH)
    if sidecar.get("npz_file_sha256") != sha256_file(npz_path):
        raise OODExternalV2IntegrityError("canonical signal NPZ hash differs")
    _safe_npz_members(npz_path)
    expected_names = {
        _canonical_signal_array_name(kind, shard_index)
        for shard_index in range(CANONICAL_SIGNAL_SHARD_COUNT)
        for kind in (
            "canonical_signal_sha256",
            "dataset",
            "inventory_index",
            "record_ref",
            "signal",
        )
    }
    try:
        with zipfile.ZipFile(npz_path, "r") as zip_archive:
            for shard_index in range(CANONICAL_SIGNAL_SHARD_COUNT):
                info = zip_archive.getinfo(
                    _canonical_signal_array_name("signal", shard_index) + ".npy"
                )
                if info.file_size > CANONICAL_SIGNAL_MEMBER_MAX_BYTES:
                    raise OODExternalV2IntegrityError(
                        "canonical signal shard exceeds its frozen member bound"
                    )
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != expected_names:
                raise OODExternalV2IntegrityError(
                    "canonical signal NPZ members differ from the frozen layout"
                )
    except OODExternalV2IntegrityError:
        raise
    except (KeyError, OSError, ValueError, zipfile.BadZipFile) as error:
        raise OODExternalV2IntegrityError(
            "canonical signal NPZ cannot be inspected"
        ) from error
    return tuple(cast(dict[str, object], item) for item in raw_descriptors)


def _load_canonical_signal_shard(
    private: Path,
    *,
    shard_index: int,
    descriptor: dict[str, object],
    expected_records: tuple[_PrivateRecordEvidence, ...],
) -> dict[int, Float32Array]:
    descriptor_fields = {
        "adapter_success_count",
        "canonical_signal_sha256_tensor_sha256",
        "dataset_tensor_sha256",
        "inventory_index_tensor_sha256",
        "record_ref_tensor_sha256",
        "shard_index",
        "signal_tensor_sha256",
        "start_inventory_index",
        "stop_inventory_index_exclusive",
    }
    start = shard_index * CANONICAL_SIGNAL_SHARD_RECORDS
    stop = min(start + CANONICAL_SIGNAL_SHARD_RECORDS, len(expected_records))
    expected_indices = np.asarray(
        [
            index
            for index in range(start, stop)
            if expected_records[index].adapter_provenance_sha256 is not None
        ],
        dtype=np.int64,
    )
    if (
        set(descriptor) != descriptor_fields
        or descriptor.get("shard_index") != shard_index
        or descriptor.get("start_inventory_index") != start
        or descriptor.get("stop_inventory_index_exclusive") != stop
        or descriptor.get("adapter_success_count") != len(expected_indices)
    ):
        raise OODExternalV2IntegrityError("canonical signal descriptor differs")
    npz_path = private.parent / PurePosixPath(CANONICAL_SIGNAL_NPZ_PATH)
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            indices = np.asarray(
                archive[_canonical_signal_array_name("inventory_index", shard_index)]
            )
            datasets = np.asarray(
                archive[_canonical_signal_array_name("dataset", shard_index)]
            )
            record_refs = np.asarray(
                archive[_canonical_signal_array_name("record_ref", shard_index)]
            )
            signal_hashes = np.asarray(
                archive[
                    _canonical_signal_array_name(
                        "canonical_signal_sha256",
                        shard_index,
                    )
                ]
            )
            signals = np.asarray(
                archive[_canonical_signal_array_name("signal", shard_index)]
            )
    except (KeyError, OSError, ValueError, zipfile.BadZipFile) as error:
        raise OODExternalV2IntegrityError(
            "canonical signal shard cannot be loaded"
        ) from error
    count = len(expected_indices)
    if (
        indices.shape != (count,)
        or indices.dtype != np.dtype(np.int64)
        or not np.array_equal(indices, expected_indices)
        or datasets.shape != (count,)
        or datasets.dtype.kind != "U"
        or record_refs.shape != (count,)
        or record_refs.dtype.kind != "U"
        or signal_hashes.shape != (count,)
        or signal_hashes.dtype.kind != "U"
        or signals.shape != (count, len(LEADS), TARGET_SAMPLES)
        or signals.dtype != np.dtype(np.float32)
        or not np.isfinite(signals).all()
    ):
        raise OODExternalV2IntegrityError("canonical signal shard arrays are invalid")
    observed_datasets = tuple(str(value) for value in datasets.tolist())
    observed_refs = tuple(str(value) for value in record_refs.tolist())
    observed_hashes = tuple(str(value) for value in signal_hashes.tolist())
    expected_datasets = tuple(expected_records[int(index)].dataset for index in indices)
    expected_refs = tuple(expected_records[int(index)].record_ref for index in indices)
    expected_hashes = tuple(
        cast(str, expected_records[int(index)].canonical_signal_sha256)
        for index in indices
    )
    if (
        observed_datasets != expected_datasets
        or observed_refs != expected_refs
        or observed_hashes != expected_hashes
        or descriptor.get("inventory_index_tensor_sha256") != _tensor_sha256(indices)
        or descriptor.get("dataset_tensor_sha256") != _tensor_sha256(datasets)
        or descriptor.get("record_ref_tensor_sha256") != _tensor_sha256(record_refs)
        or descriptor.get("canonical_signal_sha256_tensor_sha256")
        != _tensor_sha256(signal_hashes)
        or descriptor.get("signal_tensor_sha256") != _tensor_sha256(signals)
    ):
        raise OODExternalV2IntegrityError(
            "canonical signal shard identities or tensor hashes differ"
        )
    result: dict[int, Float32Array] = {}
    for local_index, inventory_index in enumerate(indices.tolist()):
        signal = np.ascontiguousarray(signals[local_index], dtype=np.float32)
        expected_hash = expected_records[int(inventory_index)].canonical_signal_sha256
        if _tensor_sha256(signal) != expected_hash:
            raise OODExternalV2IntegrityError("canonical per-record signal hash differs")
        result[int(inventory_index)] = signal
    return result


def _verify_raw_to_canonical_replay(
    private: Path,
    *,
    inputs: VerifiedExternalV2Inputs,
    expected_records: tuple[_PrivateRecordEvidence, ...],
) -> None:
    """Re-run every bound adapter and bind raw bytes to stored canonical signals."""

    descriptors = _verify_canonical_signal_sidecar(
        private,
        inputs=inputs,
        expected_records=expected_records,
    )
    if any(row.adapter_provenance_sha256 is None for row in expected_records):
        raise OODExternalV2IntegrityError(
            "a completed bundle contains an adapter-failure row"
        )
    replayed = 0
    for shard_index, descriptor in enumerate(descriptors):
        stored = _load_canonical_signal_shard(
            private,
            shard_index=shard_index,
            descriptor=descriptor,
            expected_records=expected_records,
        )
        for index, stored_signal in stored.items():
            record = inputs.inventory.records[index]
            evidence = expected_records[index]
            base = resolve_inventory_record_base(
                inputs.dataset_roots[record.dataset],
                record,
            )
            adapted = _adapter_for_record(record, base)
            _verify_adapter_against_inventory(adapted, record)
            if (
                evidence.adapter_provenance_sha256 != adapted.provenance_sha256
                or evidence.canonical_signal_sha256 != _tensor_sha256(adapted.signal_mv)
                or not np.array_equal(stored_signal, adapted.signal_mv)
            ):
                raise OODExternalV2IntegrityError(
                    "raw-source adapter replay differs from stored canonical evidence"
                )
            replayed += 1
    if replayed != len(inputs.inventory.records):
        raise OODExternalV2IntegrityError(
            "raw-source adapter replay did not cover every selected record"
        )
    _verify_raw_source_files_unchanged(inputs)


def _verify_raw_source_files_unchanged(inputs: VerifiedExternalV2Inputs) -> None:
    for name in REQUIRED_RAW_SOURCE_BINDING_KEYS:
        path = inputs.raw_source_paths[name]
        binding = inputs.child.raw_source_bindings[name]
        try:
            size = path.stat().st_size
        except OSError as error:
            raise OODExternalV2IntegrityError(
                "raw-source provenance cannot be reinspected"
            ) from error
        if (
            size != binding.size_bytes
            or sha256_file(path) != binding.file_sha256
            or (
                binding.official_md5 is not None
                and _md5_file(path) != binding.official_md5
            )
        ):
            raise OODExternalV2IntegrityError(
                "raw-source provenance changed during canonical replay"
            )


def _replay_quality_pass_embeddings(
    private: Path,
    *,
    inputs: VerifiedExternalV2Inputs,
    expected_records: tuple[_PrivateRecordEvidence, ...],
    model: ResNet1D,
    normalization: NormalizationStats,
    runtime: DeterministicCUDARuntime,
) -> tuple[Float32Array, Float32Array]:
    """Replay the exact normalized full backbone twice from stored signals."""

    pass_indices = tuple(
        index
        for index, row in enumerate(expected_records)
        if row.quality_status == QualityStatus.PASS.value
    )
    if not pass_indices:
        empty = np.empty((0, 512), dtype=np.float32)
        return empty, empty.copy()
    descriptors = _verify_canonical_signal_sidecar(
        private,
        inputs=inputs,
        expected_records=expected_records,
    )
    signals = np.empty(
        (len(pass_indices), len(LEADS), TARGET_SAMPLES),
        dtype=np.float32,
    )
    output_positions = {
        inventory_index: index
        for index, inventory_index in enumerate(pass_indices)
    }
    seen: set[int] = set()
    for shard_index, descriptor in enumerate(descriptors):
        shard = _load_canonical_signal_shard(
            private,
            shard_index=shard_index,
            descriptor=descriptor,
            expected_records=expected_records,
        )
        for inventory_index, signal in shard.items():
            output_index = output_positions.get(inventory_index)
            if output_index is not None:
                signals[output_index] = signal
                seen.add(inventory_index)
    if seen != set(pass_indices):
        raise OODExternalV2IntegrityError(
            "quality-PASS canonical signals are incomplete for full-model replay"
        )
    state_before = model_state_sha256(model)
    replay = extract_embeddings_twice(
        model,
        _NormalizedSignalDataset(signals, normalization),
        runtime=runtime,
    )
    state_after = model_state_sha256(model)
    if state_before != state_after or not np.array_equal(replay.first, replay.repeated):
        raise OODExternalV2IntegrityError(
            "full-model CUDA embedding replay is nondeterministic or mutated the model"
        )
    return (
        np.ascontiguousarray(replay.first, dtype=np.float32),
        np.ascontiguousarray(replay.repeated, dtype=np.float32),
    )


def _verify_quality_audit_shards(
    private: Path,
    *,
    inputs: VerifiedExternalV2Inputs,
    expected_records: tuple[_PrivateRecordEvidence, ...],
) -> None:
    if len(expected_records) != QUALITY_AUDIT_EXPECTED_RECORDS:
        raise OODExternalV2IntegrityError(
            "quality audit record count differs from the frozen layout"
        )
    signal_descriptors = _verify_canonical_signal_sidecar(
        private,
        inputs=inputs,
        expected_records=expected_records,
    )
    index = _load_private_sidecar(
        private.parent / PurePosixPath(QUALITY_AUDIT_INDEX_PATH),
        artifact_type=PRIVATE_QUALITY_AUDIT_INDEX_ARTIFACT_TYPE,
    )
    expected_index_fields = {
        "artifact_sha256",
        "artifact_type",
        "inventory_sha256",
        "protocol_id",
        "record_count",
        "schema_version",
        "shard_count",
        "shard_max_bytes",
        "shard_records",
        "shards",
    }
    descriptors = index.get("shards")
    if (
        set(index) != expected_index_fields
        or index.get("inventory_sha256") != inputs.inventory.inventory_sha256
        or index.get("record_count") != len(expected_records)
        or index.get("shard_count") != QUALITY_AUDIT_SHARD_COUNT
        or index.get("shard_max_bytes") != QUALITY_AUDIT_SHARD_MAX_BYTES
        or index.get("shard_records") != QUALITY_AUDIT_SHARD_RECORDS
        or not isinstance(descriptors, list)
        or len(descriptors) != QUALITY_AUDIT_SHARD_COUNT
    ):
        raise OODExternalV2IntegrityError("quality audit index differs")
    descriptor_fields = {
        "artifact_sha256",
        "file_sha256",
        "record_count",
        "relative_path",
        "size_bytes",
        "start_inventory_index",
        "stop_inventory_index_exclusive",
    }
    for shard_index, (descriptor, relative_path) in enumerate(
        zip(descriptors, QUALITY_AUDIT_SHARD_PATHS, strict=True)
    ):
        start = shard_index * QUALITY_AUDIT_SHARD_RECORDS
        stop = min(start + QUALITY_AUDIT_SHARD_RECORDS, len(expected_records))
        canonical_signals = _load_canonical_signal_shard(
            private,
            shard_index=shard_index,
            descriptor=signal_descriptors[shard_index],
            expected_records=expected_records,
        )
        if not isinstance(descriptor, dict) or set(descriptor) != descriptor_fields:
            raise OODExternalV2IntegrityError("quality audit descriptor fields differ")
        if (
            descriptor.get("relative_path") != relative_path
            or descriptor.get("record_count") != stop - start
            or descriptor.get("start_inventory_index") != start
            or descriptor.get("stop_inventory_index_exclusive") != stop
        ):
            raise OODExternalV2IntegrityError("quality audit descriptor range differs")
        file_sha256 = _digest(
            descriptor.get("file_sha256"),
            "quality audit shard file",
        )
        artifact_sha256 = _digest(
            descriptor.get("artifact_sha256"),
            "quality audit shard artifact",
        )
        size_bytes = _positive_integer(
            descriptor.get("size_bytes"),
            "quality audit shard size",
        )
        if size_bytes > QUALITY_AUDIT_SHARD_MAX_BYTES:
            raise OODExternalV2IntegrityError("quality audit shard size exceeds limit")
        shard_path = private.parent / PurePosixPath(relative_path)
        raw = _read_bounded(
            shard_path,
            QUALITY_AUDIT_SHARD_MAX_BYTES,
            "quality audit shard",
        )
        if len(raw) != size_bytes or sha256_bytes(raw) != file_sha256:
            raise OODExternalV2IntegrityError("quality audit shard bytes differ")
        try:
            decoded: object = json.loads(
                raw[:-1].decode("ascii") if raw.endswith(b"\n") else "",
                object_pairs_hook=_unique_json_object,
            )
        except (UnicodeError, json.JSONDecodeError) as error:
            raise OODExternalV2IntegrityError(
                "quality audit shard cannot be decoded"
            ) from error
        if not isinstance(decoded, dict) or canonical_json_bytes(decoded) != raw:
            raise OODExternalV2IntegrityError("quality audit shard is not canonical")
        shard = cast(dict[str, object], decoded)
        shard_fields = {
            "artifact_sha256",
            "artifact_type",
            "inventory_sha256",
            "protocol_id",
            "record_count",
            "records",
            "schema_version",
            "shard_index",
            "start_inventory_index",
            "stop_inventory_index_exclusive",
        }
        shard_body = {key: value for key, value in shard.items() if key != "artifact_sha256"}
        rows = shard.get("records")
        if (
            set(shard) != shard_fields
            or shard.get("artifact_type") != PRIVATE_QUALITY_AUDIT_ARTIFACT_TYPE
            or shard.get("protocol_id") != PROTOCOL_ID
            or shard.get("schema_version") != 1
            or shard.get("inventory_sha256") != inputs.inventory.inventory_sha256
            or shard.get("artifact_sha256") != artifact_sha256
            or canonical_sha256(shard_body) != artifact_sha256
            or shard.get("shard_index") != shard_index
            or shard.get("start_inventory_index") != start
            or shard.get("stop_inventory_index_exclusive") != stop
            or shard.get("record_count") != stop - start
            or not isinstance(rows, list)
            or len(rows) != stop - start
        ):
            raise OODExternalV2IntegrityError("quality audit shard lineage differs")
        row_fields = {
            "canonical_signal_sha256",
            "dataset",
            "inventory_index",
            "quality_report",
            "quality_report_sha256",
            "record_ref",
        }
        for offset, raw_row in enumerate(rows):
            evidence = expected_records[start + offset]
            inventory_index = start + offset
            if not isinstance(raw_row, dict) or set(raw_row) != row_fields:
                raise OODExternalV2IntegrityError("quality audit row fields differ")
            report = raw_row.get("quality_report")
            report_sha256 = raw_row.get("quality_report_sha256")
            if (
                raw_row.get("inventory_index") != start + offset
                or raw_row.get("dataset") != evidence.dataset
                or raw_row.get("record_ref") != evidence.record_ref
                or raw_row.get("canonical_signal_sha256")
                != evidence.canonical_signal_sha256
                or report_sha256 != evidence.quality_report_sha256
                or (report is None) is not (report_sha256 is None)
            ):
                raise OODExternalV2IntegrityError("quality audit row alignment differs")
            if report is None:
                if inventory_index in canonical_signals:
                    raise OODExternalV2IntegrityError(
                        "adapter-failure audit row unexpectedly has a canonical signal"
                    )
                _verify_private_quality_report_semantics(evidence)
                continue
            if not isinstance(report, dict) or not all(
                isinstance(key, str) for key in report
            ):
                raise OODExternalV2IntegrityError("quality audit report is invalid")
            if _quality_report_sha256(report) != report_sha256:
                raise OODExternalV2IntegrityError("quality audit report hash differs")
            signal = canonical_signals.get(inventory_index)
            if signal is None:
                raise OODExternalV2IntegrityError(
                    "successful adapter audit row lacks its canonical signal"
                )
            recomputed_report = _quality_report_dict(
                assess_signal_quality(
                    signal,
                    SignalMetadata.canonical(DEFAULT_SIGNAL_QUALITY_CONFIG),
                )
            )
            if recomputed_report != report:
                raise OODExternalV2IntegrityError(
                    "quality report differs from exact canonical-signal reassessment"
                )
            _verify_private_quality_report_semantics(
                replace(
                    evidence,
                    quality_report=cast(dict[str, object], report),
                )
            )
        if set(canonical_signals) != {
            start + offset
            for offset, raw_row in enumerate(rows)
            if isinstance(raw_row, dict) and raw_row.get("quality_report") is not None
        }:
            raise OODExternalV2IntegrityError(
                "canonical signal shard coverage differs from quality reports"
            )


def _load_private_record_evidence(
    path: Path,
    *,
    inputs: VerifiedExternalV2Inputs,
) -> tuple[_PrivateRecordEvidence, ...]:
    raw = _read_bounded(path, _PRIVATE_JSON_MAX_BYTES, "private record evidence")
    try:
        decoded: object = json.loads(
            raw[:-1].decode("ascii") if raw.endswith(b"\n") else b"".decode(),
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise OODExternalV2IntegrityError("private record evidence is invalid") from error
    if not isinstance(decoded, dict) or canonical_json_bytes(decoded) != raw:
        raise OODExternalV2IntegrityError("private record evidence is not canonical")
    payload = cast(dict[str, object], decoded)
    claimed = _digest(payload.get("artifact_sha256"), "private record evidence")
    body = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    if claimed != canonical_sha256(body):
        raise OODExternalV2IntegrityError("private record evidence self-hash differs")
    if (
        payload.get("artifact_type") != PRIVATE_EVIDENCE_ARTIFACT_TYPE
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("record_count") != len(inputs.inventory.records)
        or payload.get("inventory_sha256") != inputs.inventory.inventory_sha256
        or payload.get("parent_config_file_sha256") != inputs.parent.file_sha256
        or payload.get("child_contract_file_sha256") != inputs.child.file_sha256
        or payload.get("threshold") != inputs.parent.threshold
        or payload.get("decision_bindings")
        != {
            "demo_policy_file_sha256": inputs.routing.demo_policy_file_sha256,
            "source_calibration_file_sha256": (
                inputs.routing.source_calibration_file_sha256
            ),
        }
    ):
        raise OODExternalV2IntegrityError("private record evidence lineage differs")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != len(inputs.inventory.records):
        raise OODExternalV2IntegrityError("private record evidence rows are incomplete")
    records: list[_PrivateRecordEvidence] = []
    expected_fields = set(
        _private_record_index_dict(
            _PrivateRecordEvidence(
            dataset="x",
            record_ref="x",
            patient_key=None,
            challenge_quality_label=None,
            adapter_provenance_sha256=None,
            adapter_source_sample_count=None,
            adapter_raw_physical_units=None,
            canonical_signal_sha256=None,
            quality_report_sha256=None,
            quality_report=None,
            quality_status="x",
            quality_reason_codes=(),
            route="x",
            distribution_score=None,
            entropy=None,
            entropy_accepted=None,
            conformal_decisions=None,
            all_conformal_decisions_singleton=None,
            )
        )
    )
    for raw_record in raw_records:
        if not isinstance(raw_record, dict) or set(raw_record) != expected_fields:
            raise OODExternalV2IntegrityError("private record evidence row differs")
        if raw_record["quality_report"] is not None:
            raise OODExternalV2IntegrityError(
                "record evidence must store only the sharded quality-report hash"
            )
        reasons = raw_record["quality_reason_codes"]
        if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
            raise OODExternalV2IntegrityError("private quality reasons are invalid")
        source_sample_count = raw_record["adapter_source_sample_count"]
        if source_sample_count is not None and (
            type(source_sample_count) is not int or source_sample_count <= 0
        ):
            raise OODExternalV2IntegrityError(
                "private adapter source sample count is invalid"
            )
        raw_physical_units = raw_record["adapter_raw_physical_units"]
        if raw_physical_units is not None and (
            not isinstance(raw_physical_units, list)
            or len(raw_physical_units) != len(LEADS)
            or not all(isinstance(item, str) for item in raw_physical_units)
        ):
            raise OODExternalV2IntegrityError(
                "private adapter raw physical units are invalid"
            )
        canonical_signal_sha256 = raw_record["canonical_signal_sha256"]
        if canonical_signal_sha256 is not None:
            canonical_signal_sha256 = _digest(
                canonical_signal_sha256,
                "private canonical signal",
            )
        quality_report_sha256 = raw_record["quality_report_sha256"]
        if quality_report_sha256 is not None:
            quality_report_sha256 = _digest(
                quality_report_sha256,
                "private quality report",
            )
        conformal_decisions = raw_record["conformal_decisions"]
        if conformal_decisions is not None and (
            not isinstance(conformal_decisions, list)
            or len(conformal_decisions) != len(SUPERCLASSES)
            or any(
                item not in {decision.value for decision in BinaryDecision}
                for item in conformal_decisions
            )
        ):
            raise OODExternalV2IntegrityError(
                "private conformal decisions are invalid"
            )
        records.append(
            _PrivateRecordEvidence(
                dataset=cast(str, raw_record["dataset"]),
                record_ref=cast(str, raw_record["record_ref"]),
                patient_key=cast(str | None, raw_record["patient_key"]),
                challenge_quality_label=cast(
                    str | None,
                    raw_record["challenge_quality_label"],
                ),
                adapter_provenance_sha256=cast(
                    str | None,
                    raw_record["adapter_provenance_sha256"],
                ),
                adapter_source_sample_count=source_sample_count,
                adapter_raw_physical_units=(
                    None
                    if raw_physical_units is None
                    else tuple(cast(list[str], raw_physical_units))
                ),
                canonical_signal_sha256=cast(
                    str | None,
                    canonical_signal_sha256,
                ),
                quality_report_sha256=cast(
                    str | None,
                    quality_report_sha256,
                ),
                quality_report=None,
                quality_status=cast(str, raw_record["quality_status"]),
                quality_reason_codes=tuple(cast(list[str], reasons)),
                route=cast(str, raw_record["route"]),
                distribution_score=cast(
                    float | None,
                    raw_record["distribution_score"],
                ),
                entropy=cast(float | None, raw_record["entropy"]),
                entropy_accepted=cast(bool | None, raw_record["entropy_accepted"]),
                conformal_decisions=(
                    None
                    if conformal_decisions is None
                    else tuple(cast(list[str], conformal_decisions))
                ),
                all_conformal_decisions_singleton=cast(
                    bool | None,
                    raw_record["all_conformal_decisions_singleton"],
                ),
            )
        )
    result = tuple(records)
    if any(row.adapter_provenance_sha256 is None for row in result):
        raise OODExternalV2IntegrityError(
            "completed evidence cannot contain an adapter-contract failure row"
        )
    _verify_quality_audit_shards(
        path.parent,
        inputs=inputs,
        expected_records=result,
    )
    for evidence, inventory_record in zip(
        result,
        inputs.inventory.records,
        strict=True,
    ):
        if (
            evidence.dataset != inventory_record.dataset
            or evidence.record_ref != inventory_record.record_ref
            or evidence.patient_key != inventory_record.patient_key
            or evidence.challenge_quality_label
            != inventory_record.challenge_quality_label
        ):
            raise OODExternalV2IntegrityError(
                "private record row order or inventory identity differs"
            )
        _verify_private_route_semantics(
            evidence,
            threshold=inputs.parent.threshold,
        )
        _verify_private_adapter_semantics(evidence, inventory_record)
    observed_route_counts = Counter(item.route for item in result)
    observed_routes = {
        route: observed_route_counts.get(route, 0) for route in FROZEN_ROUTE_ORDER
    }
    if payload.get("route_counts") != observed_routes:
        raise OODExternalV2IntegrityError("private route counts differ from rows")
    return result


def _verify_private_route_semantics(
    evidence: _PrivateRecordEvidence,
    *,
    threshold: float,
) -> None:
    allowed_quality = {status.value for status in QualityStatus}
    if evidence.quality_status not in allowed_quality:
        raise OODExternalV2IntegrityError("private quality status is invalid")
    if not all(isinstance(item, str) and item for item in evidence.quality_reason_codes):
        raise OODExternalV2IntegrityError("private quality reason is invalid")
    if evidence.quality_status == QualityStatus.PASS.value:
        if (
            evidence.distribution_score is None
            or type(evidence.distribution_score) is not float
            or not math.isfinite(evidence.distribution_score)
            or evidence.entropy is None
            or type(evidence.entropy) is not float
            or not math.isfinite(evidence.entropy)
            or type(evidence.entropy_accepted) is not bool
            or evidence.conformal_decisions is None
            or len(evidence.conformal_decisions) != len(SUPERCLASSES)
            or type(evidence.all_conformal_decisions_singleton) is not bool
        ):
            raise OODExternalV2IntegrityError(
                "quality-PASS row lacks complete routing evidence"
            )
        expected_route = (
            "UNSUPPORTED_INPUT"
            if evidence.distribution_score > threshold
            else (
                "PREDICTION_ALLOWED"
                if evidence.entropy_accepted
                and evidence.all_conformal_decisions_singleton
                else "ABSTAIN"
            )
        )
        if evidence.route != expected_route:
            raise OODExternalV2IntegrityError("private strict routing decision differs")
        expected_singleton = all(
            decision != BinaryDecision.UNCERTAIN.value
            for decision in evidence.conformal_decisions
        )
        if evidence.all_conformal_decisions_singleton is not expected_singleton:
            raise OODExternalV2IntegrityError(
                "private conformal singleton summary differs from five decisions"
            )
    else:
        expected_route = (
            "INVALID_INPUT"
            if evidence.quality_status == QualityStatus.INVALID.value
            else "REACQUIRE"
        )
        if evidence.route != expected_route or any(
            value is not None
            for value in (
                evidence.distribution_score,
                evidence.entropy,
                evidence.entropy_accepted,
                evidence.conformal_decisions,
                evidence.all_conformal_decisions_singleton,
            )
        ):
            raise OODExternalV2IntegrityError("non-PASS row has invalid routing evidence")


def _verify_private_adapter_semantics(
    evidence: _PrivateRecordEvidence,
    inventory_record: ExternalInventoryRecord,
) -> None:
    """Reconstruct the exact adapter provenance from inventory-bound metadata."""

    if evidence.adapter_provenance_sha256 is None:
        raise OODExternalV2IntegrityError(
            "completed evidence cannot encode an adapter-contract failure"
        )
    _digest(evidence.adapter_provenance_sha256, "private adapter provenance")
    source_sample_count = evidence.adapter_source_sample_count
    raw_physical_units = evidence.adapter_raw_physical_units
    if source_sample_count is None or raw_physical_units is None:
        raise OODExternalV2IntegrityError(
            "private adapter provenance lacks source sample or unit evidence"
        )
    if (
        evidence.canonical_signal_sha256 is None
        or evidence.quality_report_sha256 is None
    ):
        raise OODExternalV2IntegrityError(
            "successful adapter row lacks signal or quality audit evidence"
        )
    _digest(evidence.canonical_signal_sha256, "private canonical signal")
    _digest(evidence.quality_report_sha256, "private quality report")
    if evidence.quality_report is not None and (
        _quality_report_sha256(evidence.quality_report)
        != evidence.quality_report_sha256
    ):
        raise OODExternalV2IntegrityError("private quality report hash differs")
    if source_sample_count != _expected_source_sample_count(inventory_record):
        raise OODExternalV2IntegrityError(
            "private adapter source sample count differs from inventory"
        )
    if raw_physical_units != inventory_record.raw_physical_units:
        raise OODExternalV2IntegrityError(
            "private adapter raw physical units differ from exact mV"
        )
    source_rate = Fraction(str(inventory_record.sampling_frequency_hz))
    source_window = source_rate * WINDOW_SECONDS
    if source_window.denominator != 1:
        raise OODExternalV2IntegrityError(
            "private adapter source window does not contain an integer sample count"
        )
    ratio = Fraction(TARGET_FREQUENCY_HZ, 1) / source_rate
    try:
        reconstructed = AdapterProvenance(
            adapter_version=ADAPTER_VERSION,
            raw_header_sha256=inventory_record.raw_header_sha256,
            raw_header_size_bytes=inventory_record.raw_header_size_bytes,
            raw_data_sha256=inventory_record.raw_data_sha256,
            raw_data_size_bytes=inventory_record.raw_data_size_bytes,
            source_frequency_hz=inventory_record.sampling_frequency_hz,
            source_sample_count=source_sample_count,
            source_duration_seconds=inventory_record.duration_seconds,
            source_lead_names=inventory_record.raw_ordered_leads,
            canonical_leads=inventory_record.canonical_ordered_leads,
            output_leads=LEADS,
            source_data_file_names=inventory_record.raw_data_file_names,
            raw_physical_units=raw_physical_units,
            physical_units=PHYSICAL_UNITS,
            window_start_sample=0,
            window_source_samples=source_window.numerator,
            window_seconds=WINDOW_SECONDS,
            resample_up=ratio.numerator,
            resample_down=ratio.denominator,
            resample_window=RESAMPLE_WINDOW,
            resample_padtype=RESAMPLE_PADTYPE,
            target_frequency_hz=TARGET_FREQUENCY_HZ,
            target_samples=TARGET_SAMPLES,
        )
    except ExternalECGAdapterError as error:
        raise OODExternalV2IntegrityError(
            "private adapter provenance cannot be reconstructed"
        ) from error
    if reconstructed.sha256 != evidence.adapter_provenance_sha256:
        raise OODExternalV2IntegrityError(
            "private adapter provenance identity differs from frozen inventory"
        )


def _verify_private_quality_report_semantics(
    evidence: _PrivateRecordEvidence,
) -> None:
    report = evidence.quality_report
    if report is None:
        return
    if set(report) != {
        "config_version",
        "global_issues",
        "leads",
        "reversal_evidence",
        "status",
    } or report["config_version"] != DEFAULT_SIGNAL_QUALITY_CONFIG.version:
        raise OODExternalV2IntegrityError("private quality report fields differ")

    allowed_statuses = {status.value for status in QualityStatus}
    allowed_reasons = {reason.value for reason in ReasonCode}
    status_rank = {
        QualityStatus.PASS.value: 0,
        QualityStatus.LIMITED.value: 1,
        QualityStatus.REACQUIRE.value: 2,
        QualityStatus.INVALID.value: 3,
    }

    def issue_values(value: object, *, expected_lead: str | None) -> tuple[str, str]:
        if not isinstance(value, dict) or set(value) != {
            "boundary_value",
            "code",
            "lead_name",
            "metric_name",
            "observed_value",
            "status",
        }:
            raise OODExternalV2IntegrityError("private quality issue fields differ")
        code = value["code"]
        status = value["status"]
        if code not in allowed_reasons or status not in allowed_statuses:
            raise OODExternalV2IntegrityError("private quality issue identity differs")
        if value["lead_name"] != expected_lead:
            raise OODExternalV2IntegrityError("private quality issue lead differs")
        metric_name = value["metric_name"]
        if metric_name is not None and (
            not isinstance(metric_name, str) or not metric_name
        ):
            raise OODExternalV2IntegrityError("private quality metric name is invalid")
        for key in ("observed_value", "boundary_value"):
            numeric = value[key]
            if numeric is not None and (
                type(numeric) is not float or not math.isfinite(numeric)
            ):
                raise OODExternalV2IntegrityError(
                    "private quality issue numeric evidence is invalid"
                )
        return cast(str, code), cast(str, status)

    global_issues = report["global_issues"]
    if not isinstance(global_issues, list):
        raise OODExternalV2IntegrityError("private global quality issues are invalid")
    global_values = tuple(
        issue_values(issue, expected_lead=None) for issue in global_issues
    )

    metric_fields = {
        "baseline_wander_power_ratio",
        "clipping_fraction",
        "flat_step_fraction",
        "high_frequency_power_ratio",
        "longest_clipping_run_samples",
        "maximum_absolute_amplitude_mv",
        "maximum_step_mv",
        "peak_to_peak_mv",
        "powerline_50hz_power_ratio",
        "powerline_60hz_power_ratio",
        "spike_step_fraction",
        "standard_deviation_mv",
    }
    raw_leads = report["leads"]
    if not isinstance(raw_leads, list) or len(raw_leads) != len(LEADS):
        raise OODExternalV2IntegrityError("private quality lead reports are incomplete")
    lead_values: list[tuple[tuple[str, ...], str]] = []
    for index, (raw_lead, expected_name) in enumerate(zip(raw_leads, LEADS, strict=True)):
        if not isinstance(raw_lead, dict) or set(raw_lead) != {
            "issues",
            "lead_index",
            "lead_name",
            "metrics",
            "reason_codes",
            "status",
        }:
            raise OODExternalV2IntegrityError("private quality lead fields differ")
        if raw_lead["lead_index"] != index or raw_lead["lead_name"] != expected_name:
            raise OODExternalV2IntegrityError("private quality lead alignment differs")
        metrics = raw_lead["metrics"]
        if not isinstance(metrics, dict) or set(metrics) != metric_fields:
            raise OODExternalV2IntegrityError("private quality metrics differ")
        for name, numeric in metrics.items():
            if name == "longest_clipping_run_samples":
                valid = type(numeric) is int and numeric >= 0
            else:
                valid = type(numeric) is float and math.isfinite(numeric)
            if not valid:
                raise OODExternalV2IntegrityError(
                    "private quality metric value is invalid"
                )
        issues = raw_lead["issues"]
        if not isinstance(issues, list):
            raise OODExternalV2IntegrityError("private lead quality issues are invalid")
        parsed_issues = tuple(
            issue_values(issue, expected_lead=expected_name) for issue in issues
        )
        reason_codes = raw_lead["reason_codes"]
        expected_reasons = tuple(code for code, _ in parsed_issues)
        if (
            not isinstance(reason_codes, list)
            or tuple(reason_codes) != expected_reasons
            or any(reason not in allowed_reasons for reason in reason_codes)
        ):
            raise OODExternalV2IntegrityError("private lead reason codes differ")
        expected_status = max(
            (status for _, status in parsed_issues),
            key=lambda value: status_rank[value],
            default=QualityStatus.PASS.value,
        )
        if raw_lead["status"] != expected_status:
            raise OODExternalV2IntegrityError("private lead quality status differs")
        lead_values.append((expected_reasons, expected_status))

    reversal = report["reversal_evidence"]
    if reversal is not None:
        if not isinstance(reversal, dict) or set(reversal) != {
            "correlations",
            "dominant_polarities",
            "evidence_codes",
            "probable_kind",
            "score",
        }:
            raise OODExternalV2IntegrityError("private reversal evidence differs")
        score = reversal["score"]
        if type(score) is not float or not math.isfinite(score):
            raise OODExternalV2IntegrityError("private reversal score is invalid")
        for key in ("correlations", "dominant_polarities"):
            pairs = reversal[key]
            if not isinstance(pairs, list) or any(
                not isinstance(pair, list)
                or len(pair) != 2
                or not isinstance(pair[0], str)
                or type(pair[1]) is not float
                or not math.isfinite(pair[1])
                for pair in pairs
            ):
                raise OODExternalV2IntegrityError(
                    "private reversal numeric evidence is invalid"
                )
        if not isinstance(reversal["probable_kind"], str) or not isinstance(
            reversal["evidence_codes"],
            list,
        ):
            raise OODExternalV2IntegrityError("private reversal identity is invalid")

    expected_overall = max(
        [status for _, status in global_values]
        + [status for _, status in lead_values],
        key=lambda value: status_rank[value],
        default=QualityStatus.PASS.value,
    )
    ordered_reasons = [code for code, _ in global_values]
    for reasons, _ in lead_values:
        ordered_reasons.extend(reasons)
    expected_reason_codes = tuple(dict.fromkeys(ordered_reasons))
    if (
        report["status"] != expected_overall
        or evidence.quality_status != expected_overall
        or evidence.quality_reason_codes != expected_reason_codes
    ):
        raise OODExternalV2IntegrityError(
            "private quality status or reason codes differ from full report"
        )


def _load_private_sidecar(
    path: Path,
    *,
    artifact_type: str,
) -> dict[str, object]:
    raw = _read_bounded(path, _PRIVATE_JSON_MAX_BYTES, "private sidecar")
    try:
        decoded: object = json.loads(
            raw[:-1].decode("ascii") if raw.endswith(b"\n") else b"".decode(),
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise OODExternalV2IntegrityError("private sidecar cannot be decoded") from error
    if not isinstance(decoded, dict) or canonical_json_bytes(decoded) != raw:
        raise OODExternalV2IntegrityError("private sidecar is not canonical")
    payload = cast(dict[str, object], decoded)
    claimed = _digest(payload.get("artifact_sha256"), "private sidecar")
    body = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    if (
        claimed != canonical_sha256(body)
        or payload.get("artifact_type") != artifact_type
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("schema_version") != 1
    ):
        raise OODExternalV2IntegrityError("private sidecar identity differs")
    return payload


def _verify_result_endpoint_row_semantics(
    result: OODV2Result,
    rows: tuple[_PrivateRecordEvidence, ...],
) -> None:
    _verify_result_route_counts(result, rows)
    _verify_result_endpoint_row_semantics_after_routes(result, rows)


def _verify_result_route_counts(
    result: OODV2Result,
    rows: tuple[_PrivateRecordEvidence, ...],
) -> None:
    """Cross-check the public five-state aggregate against private row truth."""

    observed_route_counts = Counter(row.route for row in rows)
    expected_public_routes = {
        route: observed_route_counts.get(route, 0) for route in FROZEN_ROUTE_ORDER
    }
    public_routes = result.final_route_counts.model_dump(mode="python")
    if public_routes != {**expected_public_routes, "total_records": len(rows)}:
        raise OODExternalV2IntegrityError(
            "public five-state route counts differ from private rows"
        )


def _verify_result_endpoint_row_semantics_after_routes(
    result: OODV2Result,
    rows: tuple[_PrivateRecordEvidence, ...],
) -> None:
    technical = {
        endpoint.endpoint_key: endpoint
        for endpoint in result.technical_quality_endpoints
    }
    external = {endpoint.endpoint_key: endpoint for endpoint in result.external_cohorts}
    group3 = tuple(
        row
        for row in rows
        if row.dataset == CHALLENGE_2011_DATASET
        and row.challenge_quality_label == "unacceptable"
    )
    group1 = tuple(
        row
        for row in rows
        if row.dataset == CHALLENGE_2011_DATASET
        and row.challenge_quality_label == "acceptable"
    )
    technical_truth = {
        "challenge_group3_technical_block_sensitivity": (
            len(group3),
            sum(row.quality_status == QualityStatus.REACQUIRE.value for row in group3),
        ),
        "challenge_group1_quality_pass_rate": (
            len(group1),
            sum(row.quality_status == QualityStatus.PASS.value for row in group1),
        ),
    }
    if set(technical) != {
        key for key, (records, _) in technical_truth.items() if records > 0
    }:
        raise OODExternalV2IntegrityError(
            "technical endpoint presence differs from row denominators"
        )
    for key, endpoint in technical.items():
        records, events = technical_truth[key]
        if (
            endpoint.records != records
            or endpoint.subjects != records
            or endpoint.events != events
            or endpoint.non_events != records - events
            or endpoint.point_rate != events / records
            or endpoint.interval.records != records
            or endpoint.interval.resampling_units != records
            or endpoint.interval.event_count != events
            or endpoint.interval.point_estimate != events / records
        ):
            raise OODExternalV2IntegrityError(
                "technical endpoint counts or rate differ from private rows"
            )

    challenge_pass = tuple(
        row
        for row in rows
        if row.dataset == CHALLENGE_2011_DATASET
        and row.quality_status == QualityStatus.PASS.value
    )
    zzu_pass = tuple(
        row
        for row in rows
        if row.dataset == ZZU_PEDIATRIC_DATASET
        and row.quality_status == QualityStatus.PASS.value
    )
    zzu_subjects = len({cast(str, row.patient_key) for row in zzu_pass})
    external_truth = {
        "challenge_external_distribution_recall": (
            len(challenge_pass),
            len(challenge_pass),
            sum(row.route == "UNSUPPORTED_INPUT" for row in challenge_pass),
        ),
        "zzu_external_distribution_recall": (
            len(zzu_pass),
            zzu_subjects,
            sum(row.route == "UNSUPPORTED_INPUT" for row in zzu_pass),
        ),
    }
    if set(external) != {
        key for key, (records, _, _) in external_truth.items() if records > 0
    }:
        raise OODExternalV2IntegrityError(
            "external endpoint presence differs from row denominators"
        )
    for key, external_endpoint in external.items():
        records, subjects, detected = external_truth[key]
        rate = detected / records
        if (
            external_endpoint.records != records
            or external_endpoint.subjects != subjects
            or external_endpoint.detected_records != detected
            or external_endpoint.missed_records != records - detected
            or external_endpoint.ood_recall != rate
            or external_endpoint.interval.records != records
            or external_endpoint.interval.resampling_units != subjects
            or external_endpoint.interval.event_count != detected
            or external_endpoint.interval.point_estimate != rate
        ):
            raise OODExternalV2IntegrityError(
                "external endpoint counts or recall differ from private rows"
            )


def _load_private_frozen_model(
    private: Path,
    *,
    routing_payload: Mapping[str, object],
) -> ResNet1D:
    checkpoint_path = private / "frozen-model.ckpt"
    resolved_path = private / "frozen-resolved-config.json"
    if (
        routing_payload.get("checkpoint_file_sha256")
        != sha256_file(checkpoint_path)
        or routing_payload.get("checkpoint_file_sha256")
        != EXPECTED_CHECKPOINT_FILE_SHA256
        or routing_payload.get("resolved_config_file_sha256")
        != sha256_file(resolved_path)
        or routing_payload.get("resolved_config_file_sha256")
        != EXPECTED_RESOLVED_CONFIG_FILE_SHA256
        or routing_payload.get("resolved_config_sha256")
        != EXPECTED_RESOLVED_CONFIG_SHA256
    ):
        raise OODExternalV2IntegrityError("private frozen model file lineage differs")
    try:
        resolved_object: object = json.loads(
            _read_bounded(
                resolved_path,
                _CONFIG_MAX_BYTES,
                "private resolved config",
            ).decode("utf-8")
        )
        resolved = _mapping(resolved_object, "private resolved config")
        inner = _mapping(resolved.get("config"), "private resolved inner config")
        if (
            set(resolved) != {"config", "config_hash"}
            or resolved.get("config_hash") != EXPECTED_RESOLVED_CONFIG_SHA256
            or canonical_sha256(inner) != EXPECTED_RESOLVED_CONFIG_SHA256
        ):
            raise OODExternalV2IntegrityError(
                "private resolved config content differs"
            )
        checkpoint_object: object = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        checkpoint = _mapping(checkpoint_object, "private checkpoint")
        if (
            set(checkpoint)
            != {
                "config",
                "config_hash",
                "early_stopping_state_dict",
                "epoch",
                "manifest_hash",
                "model_state_dict",
                "optimizer_state_dict",
                "protocol_hash",
                "scaler_state_dict",
                "schema_version",
            }
            or checkpoint.get("schema_version") != 1
            or checkpoint.get("config") != inner
            or checkpoint.get("config_hash") != EXPECTED_RESOLVED_CONFIG_SHA256
        ):
            raise OODExternalV2IntegrityError("private checkpoint content differs")
        model = build_experiment_model(
            ModelConfig(architecture="resnet1d", preset="matched_capacity")
        )
        model.load_state_dict(cast(Any, checkpoint["model_state_dict"]), strict=True)
    except OODExternalV2IntegrityError:
        raise
    except Exception as error:
        raise OODExternalV2IntegrityError(
            "private frozen model cannot be reconstructed"
        ) from error
    if type(model) is not ResNet1D:
        raise OODExternalV2IntegrityError("private frozen model type differs")
    model.requires_grad_(False)
    model.cpu().eval()
    if model_state_sha256(model) != routing_payload.get("model_state_sha256"):
        raise OODExternalV2IntegrityError(
            "private frozen model state differs from routing contract"
        )
    return model


def _configure_frozen_external_v2_cuda() -> DeterministicCUDARuntime:
    """Configure the exact preregistered CUDA runtime for head-only replay."""

    return configure_deterministic_cuda(
        expected_device_name="NVIDIA GeForce RTX 5070 Ti Laptop GPU",
        expected_compute_capability=(12, 0),
        expected_python_version="3.12.13",
        expected_torch_version="2.13.0+cu130",
        expected_cuda_runtime="13.0",
        expected_cudnn_version=92_000,
        expected_nvidia_driver_version="596.49",
        nvidia_smi_executable=_nvidia_driver_tool_paths()[0],
    )


def verify_private_external_v2_bundle_semantics(
    output_root: str | Path,
    *,
    result: OODV2Result,
    inventory: ExternalWaveformInventory,
    parent_config_file_sha256: str,
    child_contract_file_sha256: str,
    project_root: str | Path | None,
    seven_zip_executable: str | Path | None,
) -> None:
    """Rederive bundle semantics against the exact live frozen source closure."""

    root = Path(os.path.abspath(os.fspath(output_root)))
    private = root / "private"
    if project_root is None or seven_zip_executable is None:
        raise OODExternalV2IntegrityError(
            "terminal semantic verification requires the live project and 7-Zip tool"
        )
    live_project = _strict_project_root(project_root)
    live_parent = load_successor_parent_config(
        live_project.joinpath(*PurePosixPath(SUCCESSOR_PARENT_CONFIG_PATH).parts),
        project_root=live_project,
    )
    live_child = load_child_contract(
        live_project.joinpath(*PurePosixPath(SUCCESSOR_CHILD_CONFIG_PATH).parts)
    )
    live_inputs = verify_external_v2_inputs(
        live_parent,
        live_child,
        project_root=live_project,
        code_revision=result.code_revision,
        seven_zip_executable=seven_zip_executable,
    )
    expected_output_root = _resolve_project_relative(
        live_project,
        live_child.output_root,
        require_directory=True,
    )
    if (
        root != expected_output_root
        or live_parent.file_sha256 != parent_config_file_sha256
        or live_child.file_sha256 != child_contract_file_sha256
        or live_inputs.inventory != inventory
    ):
        raise OODExternalV2IntegrityError(
            "terminal bundle differs from its exact live project lineage"
        )
    routing_payload = _load_private_sidecar(
        private / "routing-contract.json",
        artifact_type=PRIVATE_ROUTING_CONTRACT_ARTIFACT_TYPE,
    )
    _verify_private_lineage_copies(
        private,
        result=result,
        inventory=inventory,
        routing_payload=routing_payload,
        parent_config_file_sha256=parent_config_file_sha256,
        child_contract_file_sha256=child_contract_file_sha256,
    )
    expected_fields = {
        "artifact_sha256",
        "artifact_type",
        "bootstrap",
        "checkpoint_file_sha256",
        "conformal",
        "distribution_policy_artifact_sha256",
        "distribution_policy_file_sha256",
        "distribution_threshold",
        "demo_policy_file_sha256",
        "entropy_maximum",
        "inventory_sha256",
        "label_order",
        "model_state_sha256",
        "normalization_file_sha256",
        "protocol_id",
        "quality_config_version",
        "resolved_config_file_sha256",
        "resolved_config_sha256",
        "schema_version",
        "source_calibration_artifact_sha256",
        "source_calibration_file_sha256",
        "temperature",
        "threshold_comparison",
    }
    policy_path = private / "frozen-distribution-policy.json"
    policy_bytes = _read_bounded(
        policy_path,
        _V1_POLICY_MAX_BYTES,
        "private frozen distribution policy",
    )
    try:
        policy = load_distribution_policy_bytes(policy_bytes)
        raw_conformal = _mapping(
            routing_payload.get("conformal"),
            "private conformal routing contract",
        )
        conformal = LabelwiseBinaryConformal.from_dict(raw_conformal)
    except Exception as error:
        raise OODExternalV2IntegrityError(
            "private frozen routing components are invalid"
        ) from error
    raw_bootstrap = _mapping(
        routing_payload.get("bootstrap"),
        "private bootstrap routing contract",
    )
    if set(raw_bootstrap) != {
        "challenge_seed",
        "confidence_level",
        "replicates",
        "zzu_seed",
    }:
        raise OODExternalV2IntegrityError("private bootstrap contract fields differ")
    threshold = routing_payload.get("distribution_threshold")
    temperature = routing_payload.get("temperature")
    entropy_maximum = routing_payload.get("entropy_maximum")
    if (
        set(routing_payload) != expected_fields
        or routing_payload.get("inventory_sha256") != inventory.inventory_sha256
        or routing_payload.get("label_order") != list(SUPERCLASSES)
        or routing_payload.get("quality_config_version")
        != DEFAULT_SIGNAL_QUALITY_CONFIG.version
        or routing_payload.get("threshold_comparison")
        != "score_strictly_greater_than_threshold"
        or routing_payload.get("demo_policy_file_sha256")
        != EXPECTED_DEMO_POLICY_FILE_SHA256
        or routing_payload.get("source_calibration_file_sha256")
        != EXPECTED_SOURCE_CALIBRATION_FILE_SHA256
        or routing_payload.get("source_calibration_artifact_sha256")
        != EXPECTED_SOURCE_CALIBRATION_ARTIFACT_SHA256
        or routing_payload.get("distribution_policy_file_sha256")
        != sha256_file(policy_path)
        or sha256_file(policy_path) != result.detector_policy_sha256
        or routing_payload.get("distribution_policy_artifact_sha256")
        != policy.artifact_sha256
        or policy.artifact_sha256
        != EXPECTED_DISTRIBUTION_POLICY_ARTIFACT_SHA256
        or type(threshold) is not float
        or threshold != policy.detector.threshold
        or threshold != EXPECTED_DISTRIBUTION_THRESHOLD
        or type(temperature) is not float
        or temperature != EXPECTED_TEMPERATURE
        or type(entropy_maximum) is not float
        or entropy_maximum != EXPECTED_ENTROPY_MAXIMUM
        or conformal.label_names != SUPERCLASSES
        or conformal.alpha != 0.1
        or conformal.n_calibration_samples != 834
        or conformal.quantile_rank != 752
        or conformal.quantile_level != 0.9016786570743405
        or conformal.thresholds != EXPECTED_CONFORMAL_THRESHOLDS
        or raw_bootstrap.get("replicates")
        != result.evidence_requirements.bootstrap_replicates
        or raw_bootstrap.get("challenge_seed")
        != result.evidence_requirements.challenge_bootstrap_seed
        or raw_bootstrap.get("zzu_seed")
        != result.evidence_requirements.zzu_bootstrap_seed
        or raw_bootstrap.get("confidence_level")
        != result.evidence_requirements.co_primary_confidence_level
    ):
        raise OODExternalV2IntegrityError("private routing contract differs")
    rows = _load_private_record_evidence(
        private / "record-evidence.json",
        inputs=live_inputs,
    )
    _verify_raw_to_canonical_replay(
        private,
        inputs=live_inputs,
        expected_records=rows,
    )
    quality_pass_rows = tuple(
        row for row in rows if row.quality_status == QualityStatus.PASS.value
    )
    frozen_model = _load_private_frozen_model(
        private,
        routing_payload=routing_payload,
    )
    frozen_runtime = _configure_frozen_external_v2_cuda()
    frozen_model = prepare_resnet_for_embedding(
        frozen_model,
        runtime=frozen_runtime,
    )
    normalization = _load_private_normalization(
        private / "frozen-normalization.json",
        expected_file_sha256=_digest(
            routing_payload.get("normalization_file_sha256"),
            "private normalization file",
        ),
    )
    replayed_embeddings = _replay_quality_pass_embeddings(
        private,
        inputs=live_inputs,
        expected_records=rows,
        model=frozen_model,
        normalization=normalization,
        runtime=frozen_runtime,
    )
    _verify_embedding_bundle_semantics(
        private,
        inputs=live_inputs,
        quality_pass_rows=quality_pass_rows,
        frozen_model=frozen_model,
        frozen_runtime=frozen_runtime,
        replayed_embeddings=replayed_embeddings,
    )
    endpoint_names = tuple(
        sorted(
            [endpoint.endpoint_key for endpoint in result.external_cohorts]
            + [endpoint.endpoint_key for endpoint in result.technical_quality_endpoints]
        )
    )
    _verify_bootstrap_bundle_semantics(
        private,
        inputs=live_inputs,
        result=result,
        private_rows=rows,
        endpoint_names=endpoint_names,
    )
    _verify_result_endpoint_row_semantics(result, rows)


def _load_private_normalization(
    path: Path,
    *,
    expected_file_sha256: str,
) -> NormalizationStats:
    _digest(expected_file_sha256, "private normalization file")
    if sha256_file(path) != expected_file_sha256:
        raise OODExternalV2IntegrityError(
            "private frozen normalization hash differs from routing contract"
        )
    try:
        normalization = NormalizationStats.load(path)
    except Exception as error:
        raise OODExternalV2IntegrityError(
            "private frozen normalization cannot be loaded"
        ) from error
    if (
        normalization.provenance.training_folds != (1, 2, 3, 4, 5, 6, 7)
        or normalization.provenance.samples_per_record != TARGET_SAMPLES
        or normalization.provenance.sampling_frequency_hz != TARGET_FREQUENCY_HZ
        or normalization.provenance.target_columns != TARGET_COLUMNS
    ):
        raise OODExternalV2IntegrityError(
            "private frozen normalization scientific contract differs"
        )
    return normalization


def _verify_private_lineage_copies(
    private: Path,
    *,
    result: OODV2Result,
    inventory: ExternalWaveformInventory,
    routing_payload: Mapping[str, object],
    parent_config_file_sha256: str,
    child_contract_file_sha256: str,
) -> None:
    """Parse and cross-link exact manifest-covered parent/child/decision bytes."""

    parent_path = private / "frozen-parent-config.yaml"
    child_path = private / "frozen-child-contract.json"
    source_path = private / "frozen-source-calibration-result.json"
    sealed_v1_result_path = private / "frozen-v1-ood-completion-result.json"
    demo_path = private / "frozen-demo-policy.json"
    if (
        sha256_file(parent_path) != parent_config_file_sha256
        or parent_config_file_sha256 != result.preregistration_sha256
        or sha256_file(child_path) != child_contract_file_sha256
        or sha256_file(source_path) != EXPECTED_SOURCE_CALIBRATION_FILE_SHA256
        or sha256_file(sealed_v1_result_path) != result.sealed_v1_result_sha256
        or sha256_file(demo_path) != EXPECTED_DEMO_POLICY_FILE_SHA256
    ):
        raise OODExternalV2IntegrityError("private frozen lineage file hash differs")
    try:
        if parent_config_file_sha256 == EXPECTED_PARENT_CONFIG_SHA256:
            original_parent = load_parent_config(parent_path)
            parent_file_sha256 = original_parent.file_sha256
            parent_output_root = original_parent.output_root
            parent_checkpoint_sha256 = original_parent.checkpoint.file_sha256
            parent_normalization_sha256 = original_parent.normalization.file_sha256
            parent_raw_sources = original_parent.raw_source_bindings
            parent_seven_zip_tool = original_parent.seven_zip_tool_binding
        else:
            successor_payload, parent_raw_sources, parent_seven_zip_tool = (
                _parse_successor_parent_copy(
                    parent_path,
                    expected_file_sha256=parent_config_file_sha256,
                )
            )
            parent_file_sha256 = parent_config_file_sha256
            successor_one_shot = _mapping(
                successor_payload.get("one_shot_external_access"),
                "private successor one shot",
            )
            parent_output_root = _relative_path(
                successor_one_shot.get("output_root"),
                "private successor output root",
            )
            successor_bindings = _mapping(
                successor_payload.get("bindings"),
                "private successor bindings",
            )
            successor_checkpoint = _mapping(
                successor_bindings.get("v1_checkpoint"),
                "private successor checkpoint",
            )
            parent_checkpoint_sha256 = _digest(
                successor_checkpoint.get("file_sha256"),
                "private successor checkpoint hash",
            )
            successor_normalization = _mapping(
                successor_bindings.get("normalization"),
                "private successor normalization",
            )
            parent_normalization_sha256 = _digest(
                successor_normalization.get("file_sha256"),
                "private successor normalization hash",
            )
        child = load_child_contract(child_path)
        source = load_source_calibration_result_bytes(
            _read_bounded(
                source_path,
                _V1_RESULT_MAX_BYTES,
                "private source-calibration result",
            )
        )
        sealed_v1_result = load_ood_completion_result_bytes(
            _read_bounded(
                sealed_v1_result_path,
                _V1_RESULT_MAX_BYTES,
                "private sealed v1 aggregate result",
            )
        )
        demo = FrozenDecisionPolicy.load(demo_path)
    except Exception as error:
        raise OODExternalV2IntegrityError(
            "private frozen lineage artifact cannot be parsed"
        ) from error
    source_components = source.frozen_components
    if (
        parent_file_sha256 != parent_config_file_sha256
        or child.file_sha256 != child_contract_file_sha256
        or child.parent_config_file_sha256 != parent_file_sha256
        or child.artifact_sha256 != result.cohort_role_manifest_sha256
        or child.inventory.inventory_sha256 != inventory.inventory_sha256
        or sha256_file(private / "external-inventory.json")
        != child.inventory.file_sha256
        or child.output_root != parent_output_root
        or sha256_file(private / "frozen-normalization.json")
        != parent_normalization_sha256
        or routing_payload.get("normalization_file_sha256")
        != parent_normalization_sha256
        or _current_runtime_environment() != child.runtime_environment
        or (
            parent_raw_sources is not None
            and dict(child.raw_source_bindings) != dict(parent_raw_sources)
        )
        or (
            parent_seven_zip_tool is not None
            and child.inventory.archive_closures[1].tool_binding
            != parent_seven_zip_tool
        )
        or child.decision_bindings["source_calibration_result"].file_sha256
        != EXPECTED_SOURCE_CALIBRATION_FILE_SHA256
        or child.decision_bindings["source_calibration_result"].artifact_sha256
        != EXPECTED_SOURCE_CALIBRATION_ARTIFACT_SHA256
        or child.decision_bindings["demo_policy"].file_sha256
        != EXPECTED_DEMO_POLICY_FILE_SHA256
        or child.decision_bindings["demo_policy"].artifact_sha256 is not None
        or source.artifact_sha256 != EXPECTED_SOURCE_CALIBRATION_ARTIFACT_SHA256
        or source.split.assignment_sha256
        != result.source_gate.cohort_manifest_sha256
        or result.source_gate != _historical_source_gate(sealed_v1_result)
        or source.artifact_sha256
        != routing_payload.get("source_calibration_artifact_sha256")
        or source_components.temperature.temperature != EXPECTED_TEMPERATURE
        or source_components.entropy_gate.maximum_entropy
        != EXPECTED_ENTROPY_MAXIMUM
        or demo.provenance.checkpoint_sha256
        != parent_checkpoint_sha256.removeprefix("sha256:")
        or routing_payload.get("source_calibration_file_sha256")
        != sha256_file(source_path)
        or routing_payload.get("demo_policy_file_sha256") != sha256_file(demo_path)
    ):
        raise OODExternalV2IntegrityError("private frozen lineage semantics differ")


def _verify_embedding_bundle_semantics(
    private: Path,
    *,
    inputs: VerifiedExternalV2Inputs,
    quality_pass_rows: tuple[_PrivateRecordEvidence, ...],
    frozen_model: ResNet1D | None = None,
    frozen_runtime: DeterministicCUDARuntime | None = None,
    replayed_embeddings: tuple[Float32Array, Float32Array] | None = None,
) -> None:
    npz_path = private / "quality-pass-embeddings.npz"
    _verify_embedding_npz(npz_path, expected_records=len(quality_pass_rows))
    sidecar = _load_private_sidecar(
        private / "quality-pass-embeddings.json",
        artifact_type=PRIVATE_EMBEDDING_ARTIFACT_TYPE,
    )
    expected_sidecar_fields = {
        "artifact_sha256",
        "artifact_type",
        "embedding_dimension",
        "embedding_dtype",
        "embedding_tensor_sha256",
        "first_logits_tensor_sha256",
        "inventory_sha256",
        "logits_dtype",
        "model_state_after_sha256",
        "model_state_before_sha256",
        "model_state_unchanged",
        "npz_file_sha256",
        "probabilities_dtype",
        "probabilities_tensor_sha256",
        "protocol_id",
        "quality_pass_records",
        "repeated_embedding_tensor_sha256",
        "repeated_logits_tensor_sha256",
        "repeat_verified",
        "schema_version",
        "score_dtype",
        "score_tensor_sha256",
    }
    if (
        set(sidecar) != expected_sidecar_fields
        or sidecar.get("inventory_sha256") != inputs.inventory.inventory_sha256
        or sidecar.get("npz_file_sha256") != sha256_file(npz_path)
        or sidecar.get("quality_pass_records") != len(quality_pass_rows)
        or sidecar.get("embedding_dimension") != 512
        or sidecar.get("embedding_dtype") != "float32"
        or sidecar.get("logits_dtype") != "float64"
        or sidecar.get("probabilities_dtype") != "float64"
        or sidecar.get("score_dtype") != "float64"
        or sidecar.get("repeat_verified") is not True
        or sidecar.get("model_state_unchanged") is not True
    ):
        raise OODExternalV2IntegrityError("embedding sidecar metadata differs")
    model_state_before = _digest(
        sidecar.get("model_state_before_sha256"),
        "private model state before",
    )
    model_state_after = _digest(
        sidecar.get("model_state_after_sha256"),
        "private model state after",
    )
    if model_state_before != model_state_after:
        raise OODExternalV2IntegrityError("private model state changed")
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            embeddings = np.ascontiguousarray(
                archive["embedding_first"],
                dtype=np.float32,
            )
            repeated_embeddings = np.ascontiguousarray(
                archive["embedding_repeated"],
                dtype=np.float32,
            )
            logits = np.ascontiguousarray(archive["logits_first"], dtype=np.float64)
            repeated_logits = np.ascontiguousarray(
                archive["logits_repeated"],
                dtype=np.float64,
            )
            probabilities = np.ascontiguousarray(
                archive["probabilities"],
                dtype=np.float64,
            )
            scores = np.ascontiguousarray(archive["score"], dtype=np.float64)
            observed_dataset = tuple(str(value) for value in archive["dataset"].tolist())
            observed_patient = tuple(str(value) for value in archive["patient_key"].tolist())
            observed_record = tuple(str(value) for value in archive["record_ref"].tolist())
    except (OSError, ValueError) as error:
        raise OODExternalV2IntegrityError("embedding NPZ cannot be loaded") from error
    expected_dataset = tuple(row.dataset for row in quality_pass_rows)
    expected_patient = tuple(
        "" if row.patient_key is None else row.patient_key for row in quality_pass_rows
    )
    expected_record = tuple(row.record_ref for row in quality_pass_rows)
    expected_scores = np.asarray(
        [cast(float, row.distribution_score) for row in quality_pass_rows],
        dtype=np.float64,
    )
    if len(quality_pass_rows) == 0:
        recomputed_scores = np.empty((0,), dtype=np.float64)
        recomputed_probabilities = np.empty(
            (0, len(SUPERCLASSES)), dtype=np.float64
        )
        recomputed_decisions: tuple[tuple[str, ...], ...] = ()
        recomputed_singleton: tuple[bool, ...] = ()
        recomputed_entropy_accepted: tuple[bool, ...] = ()
    else:
        recomputed_scores = np.ascontiguousarray(
            inputs.v1.policy.to_detector().score(embeddings),
            dtype=np.float64,
        )
        recomputed_probabilities = _sigmoid(logits / inputs.routing.temperature)
        recomputed_entropy = normalized_bernoulli_entropy(probabilities)
        recomputed_prediction_sets = inputs.routing.conformal.predict(probabilities)
        recomputed_decisions = tuple(
            tuple(decision.value for decision in row)
            for row in recomputed_prediction_sets.decisions
        )
        recomputed_singleton = tuple(
            all(decision != BinaryDecision.UNCERTAIN.value for decision in row)
            for row in recomputed_decisions
        )
        recomputed_entropy_accepted = tuple(
            bool(value <= inputs.routing.maximum_entropy)
            for value in recomputed_entropy.tolist()
        )
    embedding_hash = _tensor_sha256(embeddings)
    repeated_embedding_hash = _tensor_sha256(repeated_embeddings)
    logits_hash = _tensor_sha256(logits)
    repeated_logits_hash = _tensor_sha256(repeated_logits)
    probabilities_hash = _tensor_sha256(probabilities)
    score_hash = _tensor_sha256(scores)
    if replayed_embeddings is not None:
        replayed_first, replayed_repeated = replayed_embeddings
        if (
            replayed_first.shape != embeddings.shape
            or replayed_repeated.shape != repeated_embeddings.shape
            or replayed_first.dtype != np.dtype(np.float32)
            or replayed_repeated.dtype != np.dtype(np.float32)
            or not np.array_equal(embeddings, replayed_first)
            or not np.array_equal(repeated_embeddings, replayed_repeated)
        ):
            raise OODExternalV2IntegrityError(
                "stored embeddings differ from exact full-model CUDA replay"
            )
    if (
        observed_dataset != expected_dataset
        or observed_patient != expected_patient
        or observed_record != expected_record
        or not np.array_equal(scores, expected_scores)
        or not np.array_equal(scores, recomputed_scores)
        or not np.array_equal(embeddings, repeated_embeddings)
        or not np.array_equal(logits, repeated_logits)
        or not np.array_equal(probabilities, recomputed_probabilities)
        or sidecar.get("embedding_tensor_sha256") != embedding_hash
        or sidecar.get("repeated_embedding_tensor_sha256")
        != repeated_embedding_hash
        or sidecar.get("first_logits_tensor_sha256") != logits_hash
        or sidecar.get("repeated_logits_tensor_sha256") != repeated_logits_hash
        or sidecar.get("probabilities_tensor_sha256") != probabilities_hash
        or sidecar.get("score_tensor_sha256") != score_hash
    ):
        raise OODExternalV2IntegrityError(
            "embedding identities, classifier arrays, scores, or hashes differ"
        )
    if (frozen_model is None) is not (frozen_runtime is None):
        raise OODExternalV2IntegrityError(
            "frozen classifier and CUDA runtime must be supplied together"
        )
    if (
        frozen_model is not None
        and sidecar.get("model_state_before_sha256")
        != model_state_sha256(frozen_model)
    ):
        raise OODExternalV2IntegrityError(
            "stored model state differs from the frozen classifier"
        )
    if (
        frozen_model is not None
        and frozen_runtime is not None
        and len(quality_pass_rows) > 0
    ):
        first_reference = _classify_embeddings(
            frozen_model,
            embeddings,
            runtime=frozen_runtime,
        )
        repeated_reference = _classify_embeddings(
            frozen_model,
            repeated_embeddings,
            runtime=frozen_runtime,
        )
        if (
            not np.array_equal(logits, first_reference)
            or not np.array_equal(repeated_logits, repeated_reference)
        ):
            raise OODExternalV2IntegrityError(
                "stored logits or model state differ from exact CUDA classifier replay"
            )
    for index, row in enumerate(quality_pass_rows):
        if (
            row.entropy != float(recomputed_entropy[index])
            or row.entropy_accepted is not recomputed_entropy_accepted[index]
            or row.conformal_decisions != recomputed_decisions[index]
            or row.all_conformal_decisions_singleton
            is not recomputed_singleton[index]
        ):
            raise OODExternalV2IntegrityError(
                "private entropy or conformal routing audit differs"
            )


def _verify_bootstrap_bundle_semantics(
    private: Path,
    *,
    inputs: VerifiedExternalV2Inputs,
    result: OODV2Result,
    private_rows: tuple[_PrivateRecordEvidence, ...],
    endpoint_names: tuple[str, ...],
) -> None:
    npz_path = private / "bootstrap-replicates.npz"
    _verify_bootstrap_npz(
        npz_path,
        expected_names=endpoint_names,
        expected_replicates=inputs.parent.bootstrap_resamples,
    )
    sidecar = _load_private_sidecar(
        private / "bootstrap-replicates.json",
        artifact_type=PRIVATE_BOOTSTRAP_ARTIFACT_TYPE,
    )
    if (
        sidecar.get("endpoint_names") != list(endpoint_names)
        or sidecar.get("npz_file_sha256") != sha256_file(npz_path)
        or sidecar.get("quantile_method") != "linear"
        or sidecar.get("replicates_per_endpoint")
        != inputs.parent.bootstrap_resamples
    ):
        raise OODExternalV2IntegrityError("bootstrap sidecar metadata differs")

    group3 = np.asarray(
        [
            row.quality_status == QualityStatus.REACQUIRE.value
            for row in private_rows
            if row.dataset == CHALLENGE_2011_DATASET
            and row.challenge_quality_label == "unacceptable"
        ],
        dtype=np.bool_,
    )
    group1 = np.asarray(
        [
            row.quality_status == QualityStatus.PASS.value
            for row in private_rows
            if row.dataset == CHALLENGE_2011_DATASET
            and row.challenge_quality_label == "acceptable"
        ],
        dtype=np.bool_,
    )
    challenge_pass = tuple(
        row
        for row in private_rows
        if row.dataset == CHALLENGE_2011_DATASET
        and row.quality_status == QualityStatus.PASS.value
    )
    zzu_pass = tuple(
        row
        for row in private_rows
        if row.dataset == ZZU_PEDIATRIC_DATASET
        and row.quality_status == QualityStatus.PASS.value
    )
    challenge_detected = np.asarray(
        [row.route == "UNSUPPORTED_INPUT" for row in challenge_pass],
        dtype=np.bool_,
    )
    zzu_detected = np.asarray(
        [row.route == "UNSUPPORTED_INPUT" for row in zzu_pass],
        dtype=np.bool_,
    )
    patient_keys = sorted({cast(str, row.patient_key) for row in zzu_pass})
    patient_index = {key: index + 1 for index, key in enumerate(patient_keys)}
    zzu_clusters = np.asarray(
        [patient_index[cast(str, row.patient_key)] for row in zzu_pass],
        dtype=np.int64,
    )
    parent = inputs.parent
    expected: dict[str, Float64Array] = {}
    for name, events, unit, labels, seed in (
        (
            "challenge_group3_technical_block_sensitivity",
            group3,
            ResamplingUnit.RECORD,
            None,
            parent.challenge_bootstrap_seed,
        ),
        (
            "challenge_group1_quality_pass_rate",
            group1,
            ResamplingUnit.RECORD,
            None,
            parent.challenge_bootstrap_seed,
        ),
        (
            "challenge_external_distribution_recall",
            challenge_detected,
            ResamplingUnit.RECORD,
            None,
            parent.challenge_bootstrap_seed,
        ),
        (
            "zzu_external_distribution_recall",
            zzu_detected,
            ResamplingUnit.PATIENT_CLUSTER,
            zzu_clusters,
            parent.zzu_bootstrap_seed,
        ),
    ):
        if name in endpoint_names:
            expected[name] = _bootstrap_rates(
                events,
                resampling_unit=unit,
                cluster_labels=labels,
                seed=seed,
                replicates=parent.bootstrap_resamples,
            )
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            observed = {
                name: np.ascontiguousarray(archive[name], dtype=np.float64)
                for name in endpoint_names
            }
    except (OSError, ValueError) as error:
        raise OODExternalV2IntegrityError("bootstrap NPZ cannot be loaded") from error
    if set(expected) != set(observed) or any(
        not np.array_equal(expected[name], observed[name]) for name in expected
    ):
        raise OODExternalV2IntegrityError(
            "bootstrap arrays differ from exact record/patient index draws"
        )
    _verify_replicate_quantiles(
        cast(tuple[object, ...], result.technical_quality_endpoints),
        cast(tuple[object, ...], result.external_cohorts),
        MappingProxyType(observed),
        parent=parent,
    )


class _ExternalV2OutputCommitError(OODExternalV2ExecutionError):
    def __init__(self, message: str, *, output_root_committed: bool) -> None:
        super().__init__(message)
        self.output_root_committed = output_root_committed


def _create_durable_staging_directory(
    output_root: Path,
    *,
    expected_parent_identity: _OwnedDirectoryIdentity,
) -> Path:
    """Create staging and persist its parent entry before any armed marker."""

    _verify_owned_namespace_parent(
        output_root.parent,
        expected_identity=expected_parent_identity,
    )
    raw_staging = tempfile.mkdtemp(
        prefix=f".{output_root.name}.staging-",
        dir=output_root.parent,
    )
    staging = Path(raw_staging).resolve(strict=True)
    try:
        _verify_owned_namespace_parent(
            output_root.parent,
            expected_identity=expected_parent_identity,
        )
        _fsync_directory(output_root.parent)
        _verify_owned_namespace_parent(
            output_root.parent,
            expected_identity=expected_parent_identity,
        )
    except OSError as error:
        with suppress(OSError):
            staging.rmdir()
        raise OODExternalV2ExecutionError(
            "staging directory parent entry is not durable"
        ) from error
    return staging


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        # Windows does not expose POSIX directory fsync.  Open the directory
        # itself with backup semantics and write access, then demand a real
        # FlushFileBuffers success.  Unsupported filesystems fail closed.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        create_file.restype = ctypes.c_void_p
        handle = create_file(
            os.fspath(path),
            0x40000000,  # GENERIC_WRITE
            0x00000001 | 0x00000002 | 0x00000004,
            None,
            3,  # OPEN_EXISTING
            0x02000000,  # FILE_FLAG_BACKUP_SEMANTICS
            None,
        )
        invalid_handle = ctypes.c_void_p(-1).value
        if handle == invalid_handle:
            error = ctypes.get_last_error()
            raise OSError(error, "directory durability handle could not be opened")
        try:
            flush = kernel32.FlushFileBuffers
            flush.argtypes = [ctypes.c_void_p]
            flush.restype = ctypes.c_int
            if flush(handle) == 0:
                error = ctypes.get_last_error()
                raise OSError(error, "directory FlushFileBuffers failed")
        finally:
            close = kernel32.CloseHandle
            close.argtypes = [ctypes.c_void_p]
            close.restype = ctypes.c_int
            close(handle)
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        raise
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _commit_staged_directory(
    staging_root: Path,
    output_root: Path,
    *,
    visibility_witness: Callable[[], None] | None = None,
    expected_directory_identity: _OwnedDirectoryIdentity,
    expected_marker_bytes: bytes,
    expected_parent_identity: _OwnedDirectoryIdentity,
) -> None:
    _verify_owned_namespace_parent(
        output_root.parent,
        expected_identity=expected_parent_identity,
    )
    _verify_owned_evidence_directory(
        staging_root,
        expected_identity=expected_directory_identity,
        expected_marker_bytes=expected_marker_bytes,
    )
    if output_root.exists() or _is_indirect(output_root):
        raise _ExternalV2OutputCommitError(
            "immutable output root already exists",
            output_root_committed=False,
        )
    renamed = False
    try:
        _verify_owned_namespace_parent(
            output_root.parent,
            expected_identity=expected_parent_identity,
        )
        os.rename(staging_root, output_root)
        renamed = True
        if visibility_witness is not None:
            visibility_witness()
        _verify_owned_namespace_parent(
            output_root.parent,
            expected_identity=expected_parent_identity,
        )
        _verify_owned_evidence_directory(
            output_root,
            expected_identity=expected_directory_identity,
            expected_marker_bytes=expected_marker_bytes,
        )
        _fsync_directory(output_root.parent)
        _verify_owned_namespace_parent(
            output_root.parent,
            expected_identity=expected_parent_identity,
        )
        _verify_owned_evidence_directory(
            output_root,
            expected_identity=expected_directory_identity,
            expected_marker_bytes=expected_marker_bytes,
        )
    except FileExistsError as error:
        raise _ExternalV2OutputCommitError(
            "immutable output root already exists",
            output_root_committed=False,
        ) from error
    except OSError as error:
        raise _ExternalV2OutputCommitError(
            "atomic output-root commit failed",
            output_root_committed=renamed,
        ) from error


def _atomic_write_terminal_success(
    output_root: Path,
    payload: bytes,
    *,
    visibility_witness: Callable[[], None] | None = None,
    ownership_verifier: Callable[[], None],
) -> None:
    target = output_root / SUCCESS_MANIFEST_FILENAME
    if _is_indirect(output_root) or not output_root.is_dir():
        raise OODExternalV2ExecutionError("committed output root is unavailable")
    if target.exists() or _is_indirect(target):
        raise OODExternalV2ExecutionError("terminal success manifest already exists")
    ownership_verifier()
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=".success-manifest-",
        suffix=".tmp",
        dir=output_root,
    )
    temporary = Path(raw_temp)
    temporary_exists = True
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        ownership_verifier()
        os.link(temporary, target)
        if visibility_witness is not None:
            visibility_witness()
        ownership_verifier()
        temporary.unlink()
        temporary_exists = False
        _fsync_directory(output_root)
        ownership_verifier()
    except FileExistsError as error:
        raise OODExternalV2ExecutionError(
            "terminal success manifest already exists"
        ) from error
    except OSError as error:
        raise OODExternalV2ExecutionError(
            "terminal success-manifest commit failed"
        ) from error
    finally:
        if temporary_exists:
            with suppress(OSError):
                temporary.unlink()


def _failure_code(error: BaseException) -> str:
    if isinstance(error, OODExternalV2ConfigError):
        return "CONFIG_INVALID"
    if isinstance(error, OODExternalV2IntegrityError):
        return "INTEGRITY_CHECK_FAILED"
    if isinstance(error, _ExternalV2OutputCommitError):
        return "OUTPUT_ROOT_COMMIT_FAILED"
    if isinstance(error, ExternalECGAdapterError):
        return "ADAPTER_EXECUTION_FAILED"
    if isinstance(error, (torch.cuda.OutOfMemoryError, MemoryError)):
        return "RESOURCE_EXHAUSTED"
    return "EXECUTION_FAILED"


def _failure_receipt_bytes(
    *,
    inputs: VerifiedExternalV2Inputs,
    code_revision: str,
    error: BaseException,
    ambiguous_terminal_commit: bool,
    external_claim_file_sha256: str,
    owner_nonce: str,
) -> bytes:
    _digest(external_claim_file_sha256, "failure receipt external claim")
    if _OWNER_NONCE.fullmatch(owner_nonce) is None:
        raise OODExternalV2IntegrityError("failure receipt owner nonce is invalid")
    body: dict[str, object] = {
        "artifact_type": FAILURE_ARTIFACT_TYPE,
        "child_contract_file_sha256": inputs.child.file_sha256,
        "code_revision": code_revision,
        "contains_embeddings_or_scores": False,
        "contains_filesystem_paths": False,
        "contains_record_or_patient_identifiers": False,
        "failure_code": _failure_code(error),
        "external_claim_file_sha256": external_claim_file_sha256,
        "inventory_sha256": inputs.inventory.inventory_sha256,
        "parent_config_file_sha256": inputs.parent.file_sha256,
        "owner_nonce": owner_nonce,
        "protocol_id": PROTOCOL_ID,
        "retry_requires_new_protocol_and_output_root": True,
        "schema_version": 1,
        "status": "FAILED",
        "terminal_state": (
            "AMBIGUOUS_TERMINAL_COMMIT"
            if ambiguous_terminal_commit
            else "NO_SUCCESS_MANIFEST_VISIBLE"
        ),
    }
    body["artifact_sha256"] = canonical_sha256(body)
    return canonical_json_bytes(body)


def _retain_postclaim_failure(
    *,
    staging: Path,
    output_root: Path,
    inputs: VerifiedExternalV2Inputs,
    code_revision: str,
    error: BaseException,
    output_root_owned: bool,
    terminal_manifest_visible: bool,
    external_claim_file_sha256: str,
    owner_nonce: str,
    expected_directory_identity: _OwnedDirectoryIdentity,
    expected_marker_bytes: bytes,
    expected_parent_identity: _OwnedDirectoryIdentity,
) -> None:
    if type(output_root_owned) is not bool or type(terminal_manifest_visible) is not bool:
        raise TypeError("output and terminal visibility states must be bool")
    ambiguous_terminal = output_root_owned and terminal_manifest_visible
    receipt = _failure_receipt_bytes(
        inputs=inputs,
        code_revision=code_revision,
        error=error,
        ambiguous_terminal_commit=ambiguous_terminal,
        external_claim_file_sha256=external_claim_file_sha256,
        owner_nonce=owner_nonce,
    )
    target_root = output_root if output_root_owned else staging
    try:
        # A manifest directory entry whose durability or independent reread
        # failed is not success.  Retain an explicit receipt beside it; the
        # whole-root verifier rejects any root containing both artifacts.
        def verifier() -> None:
            _verify_owned_namespace_parent(
                output_root.parent,
                expected_identity=expected_parent_identity,
            )
            _verify_owned_evidence_directory(
                target_root,
                expected_identity=expected_directory_identity,
                expected_marker_bytes=expected_marker_bytes,
            )

        _atomic_write_new(
            target_root / FAILURE_RECEIPT_FILENAME,
            receipt,
            expected_parent_identity=expected_directory_identity,
            ownership_verifier=verifier,
        )
        if not output_root_owned:
            retained_ownership = _OutputRootOwnershipState(
                expected_directory_identity
            )
            try:
                _commit_staged_directory(
                    staging,
                    output_root,
                    visibility_witness=retained_ownership.mark_visible,
                    expected_directory_identity=expected_directory_identity,
                    expected_marker_bytes=expected_marker_bytes,
                    expected_parent_identity=expected_parent_identity,
                )
            except _ExternalV2OutputCommitError as commit_error:
                if commit_error.output_root_committed or retained_ownership.visible:
                    raise
                # A foreign root won the name. Keep this process's marked,
                # receipt-bearing staging directory intact; never write into
                # the foreign root. Future attempts are blocked by the marker.
                return
    except Exception as receipt_error:
        raise OODExternalV2ExecutionError(
            "post-claim failure evidence could not be retained"
        ) from receipt_error


def _assert_no_marked_staging_retry(output_root: Path) -> None:
    prefix = f".{output_root.name}.staging-"
    try:
        candidates = tuple(output_root.parent.iterdir())
    except OSError as error:
        raise OODExternalV2ExecutionError(
            "prior staging roots cannot be inspected"
        ) from error
    for candidate in candidates:
        if not candidate.name.startswith(prefix):
            continue
        marker = candidate / ACCESS_MARKER_FILENAME
        if _is_indirect(candidate) or marker.exists() or _is_indirect(marker):
            raise OODExternalV2ExecutionError(
                "marked external staging evidence exists; retry is forbidden"
            )


def _remove_staging_root(staging: Path, *, expected_parent: Path) -> None:
    resolved = Path(os.path.abspath(os.fspath(staging)))
    parent = Path(os.path.abspath(os.fspath(expected_parent)))
    if (
        resolved.parent != parent
        or not resolved.name.startswith(".")
        or ".staging-" not in resolved.name
        or _is_indirect(resolved)
    ):
        raise OODExternalV2ExecutionError(
            "refusing to remove an unexpected staging root"
        )
    try:
        shutil.rmtree(resolved, ignore_errors=False)
    except FileNotFoundError:
        return
    except OSError as error:
        raise OODExternalV2ExecutionError("failed staging root cannot be removed") from error
