"""Tamper-evident, pre-inference inventory for external OOD v2 waveforms.

Inventory construction is intentionally metadata-only: it parses WFDB headers
and hashes the raw ``.hea``/``.dat`` bytes, but it never opens waveform samples.
That boundary lets dataset roles and record identities be frozen before any
model output or waveform-amplitude analysis is observed.
"""

from __future__ import annotations

import binascii
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Literal, Never, Protocol, cast

import wfdb  # type: ignore[import-untyped]

from ecg_trust.constants import LEADS
from ecg_trust.data.manifest import sha256_file

CHALLENGE_2011_DATASET = "physionet_challenge_2011_set_a"
ZZU_PEDIATRIC_DATASET = "zzu_pecg_v1"
CHALLENGE_2011_VERSION = "1.0.0"
ZZU_PEDIATRIC_VERSION = "1"
CONFIRMATION_LOCKBOX_ROLE = "confirmation_lockbox"
INVENTORY_KIND = "ecg_trust.external_waveform_inventory"
INVENTORY_SCHEMA_VERSION = 2
INVENTORY_DOMAIN = b"ecg_trust.ood_v2.external_waveform_inventory.v2\x00"
PUBLIC_PROJECTION_KIND = "ecg_trust.external_waveform_inventory.public_projection"
PUBLIC_PROJECTION_DOMAIN = b"ecg_trust.ood_v2.inventory_public_projection.v2\x00"
BUILD_SUMMARY_DOMAIN = b"ecg_trust.ood_v2.inventory_build_summary.v1\x00"
ARCHIVE_CLOSURE_KIND = "ecg_trust.archive_extraction_closure"
ARCHIVE_CLOSURE_SCHEMA_VERSION = 1
ARCHIVE_CLOSURE_DOMAIN = b"ecg_trust.ood_v2.archive_extraction_closure.v1\x00"
SEVEN_ZIP_TOOL_KIND = "ecg_trust.seven_zip_tool_binding"
SEVEN_ZIP_TOOL_SCHEMA_VERSION = 1
SEVEN_ZIP_TOOL_DOMAIN = b"ecg_trust.ood_v2.seven_zip_tool_binding.v1\x00"
MAX_INVENTORY_BYTES = 256 * 1024 * 1024
SEVEN_ZIP_TIMEOUT_SECONDS = 30 * 60.0
SEVEN_ZIP_STDOUT_LIMIT_BYTES = 64 * 1024 * 1024
SEVEN_ZIP_STDERR_LIMIT_BYTES = 1024 * 1024
SEVEN_ZIP_POLL_SECONDS = 0.05

ChallengeQualityLabel = Literal["acceptable", "unacceptable", "indeterminate"]
ArchiveFormat = Literal["tar_gzip", "split_zip_7zip"]
ArchiveMemberRole = Literal[
    "wfdb_header",
    "wfdb_data",
    "quality_reference",
    "ignored_release_file",
]
InventoryExclusionReason = Literal[
    "pediatric_12_lead_flag_false",
    "sampling_frequency_not_500_hz",
    "duration_under_10_seconds",
    "lead_count_not_12",
    "noncanonical_lead_set",
]
_ZZUArchiveClosureStage = Literal[
    "zzu_tool_resolution",
    "zzu_archive_listing",
    "zzu_archive_test",
    "zzu_evaluated_tree_snapshot",
    "zzu_isolated_extraction",
    "zzu_archive_comparison",
]
_ZZUArchiveClosureStageCallback = Callable[[_ZZUArchiveClosureStage], None]
_QUALITY_LABELS: tuple[ChallengeQualityLabel, ...] = (
    "acceptable",
    "unacceptable",
    "indeterminate",
)
_EXCLUSION_REASONS: tuple[InventoryExclusionReason, ...] = (
    "pediatric_12_lead_flag_false",
    "sampling_frequency_not_500_hz",
    "duration_under_10_seconds",
    "lead_count_not_12",
    "noncanonical_lead_set",
)
_HEX_DIGITS = frozenset("0123456789abcdef")
SOURCE_LEAD_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "AVR": "aVR",
        "AVL": "aVL",
        "AVF": "aVF",
    }
)


class ExternalInventoryError(ValueError):
    """Raised when an external inventory or its raw-file binding is invalid."""


def _is_indirect(path: Path) -> bool:
    """Return whether ``path`` is a symlink or Windows junction/reparse link."""

    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(callable(is_junction) and is_junction())
    except OSError as error:
        raise ExternalInventoryError("path indirection could not be inspected") from error


def _assert_direct_ancestry(path: Path, *, field: str) -> None:
    """Reject indirection at ``path`` or any lexical ancestor through the root."""

    current = path.absolute()
    while True:
        if _is_indirect(current):
            raise ExternalInventoryError(f"{field} traverses an indirect path")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _resolve_direct_within(path: Path, root: Path, *, field: str) -> Path:
    """Resolve one existing direct path and prove it remains below ``root``."""

    _assert_direct_ancestry(path, field=field)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise ExternalInventoryError(f"{field} escapes its frozen root") from error
    if _is_indirect(path):
        raise ExternalInventoryError(f"{field} became indirect during resolution")
    return resolved


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ExternalInventoryError(f"{field} must be canonical non-empty text")
    if any(character in value for character in ("\x00", "\r", "\n")):
        raise ExternalInventoryError(f"{field} contains a forbidden character")
    return value


def _require_sha256(value: object, field: str) -> str:
    digest = _require_text(value, field)
    if len(digest) != 64 or any(character not in _HEX_DIGITS for character in digest):
        raise ExternalInventoryError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _require_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ExternalInventoryError(f"{field} must be a positive integer")
    return value


def _require_nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExternalInventoryError(f"{field} must be a non-negative integer")
    return value


def _require_positive_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExternalInventoryError(f"{field} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ExternalInventoryError(f"{field} must be finite and positive")
    return result


def _canonical_record_ref(value: object) -> str:
    reference = _require_text(value, "record_ref")
    if "\\" in reference:
        raise ExternalInventoryError("record_ref must use canonical POSIX separators")
    posix = PurePosixPath(reference)
    windows = PureWindowsPath(reference)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ExternalInventoryError("record_ref must be relative")
    if posix.as_posix() != reference or any(part in {"", ".", ".."} for part in posix.parts):
        raise ExternalInventoryError("record_ref contains traversal or non-canonical segments")
    if posix.suffix.casefold() in {".hea", ".dat"}:
        raise ExternalInventoryError("record_ref must be suffix-free")
    return reference


_WINDOWS_RESERVED_LEAVES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)


def _canonical_archive_path(value: object, field: str) -> str:
    """Return one extraction-safe, platform-independent archive member path."""

    member = _require_text(value, field)
    if "\\" in member:
        raise ExternalInventoryError(f"{field} must use POSIX separators")
    posix = PurePosixPath(member)
    windows = PureWindowsPath(member)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ExternalInventoryError(f"{field} must be relative")
    if posix.as_posix() != member or any(part in {"", ".", ".."} for part in posix.parts):
        raise ExternalInventoryError(f"{field} contains traversal or non-canonical segments")
    for part in posix.parts:
        if part != part.strip() or part.endswith((".", " ")):
            raise ExternalInventoryError(f"{field} contains a non-canonical segment")
        if any(ord(character) < 32 or character in '<>:"|?*' for character in part):
            raise ExternalInventoryError(f"{field} is unsafe for extraction")
        if part.split(".", maxsplit=1)[0].upper() in _WINDOWS_RESERVED_LEAVES:
            raise ExternalInventoryError(f"{field} uses a reserved extraction name")
    return member


def _canonical_archive_leaf(value: object, field: str) -> str:
    leaf = _canonical_archive_path(value, field)
    if len(PurePosixPath(leaf).parts) != 1:
        raise ExternalInventoryError(f"{field} must not disclose or bind a directory")
    return leaf


def _canonical_seven_zip_slt_member_path(value: object, field: str) -> str:
    """Canonicalize one 7-Zip Windows ``-slt`` presentation path.

    The bound Windows 7-Zip executable renders nested archive-member paths with
    backslash separators even when the ZIP stores portable path names.  Keep
    the global archive-path contract POSIX-only and normalize only this
    presentation boundary.  A path may use one separator convention, never a
    mixture of both; the shared canonical validator then rejects every unsafe
    or non-canonical normalized path.
    """

    member = _require_text(value, field)
    if "/" in member and "\\" in member:
        raise ExternalInventoryError(f"{field} uses mixed path separators")
    normalized = member.replace("\\", "/")
    return _canonical_archive_path(normalized, field)


def _require_prefixed_sha256(value: object, field: str) -> str:
    digest = _require_text(value, field)
    if not digest.startswith("sha256:"):
        raise ExternalInventoryError(f"{field} must use the sha256: prefix")
    _require_sha256(digest.removeprefix("sha256:"), field)
    return digest


def _text_sequence(value: object, field: str, expected_length: int) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ExternalInventoryError(f"{field} must contain {expected_length} text values")
    result: list[str] = []
    for item in value:
        result.append(_require_text(item, field))
    if len(result) != expected_length:
        raise ExternalInventoryError(f"{field} must contain {expected_length} values")
    return tuple(result)


def _canonicalize_ordered_leads(
    raw_leads: Sequence[str],
    *,
    allowed_aliases: Mapping[str, str],
) -> tuple[str, ...]:
    if any(
        SOURCE_LEAD_ALIASES.get(source) != target
        for source, target in allowed_aliases.items()
    ):
        raise ExternalInventoryError("allowed_aliases exceeds the fixed alias contract")
    if len(raw_leads) != len(LEADS):
        raise ExternalInventoryError("selected record must contain exactly 12 ordered leads")
    canonical: list[str] = []
    for raw_lead in raw_leads:
        canonical_lead: str | None = (
            raw_lead if raw_lead in LEADS else allowed_aliases.get(raw_lead)
        )
        if canonical_lead is None:
            raise ExternalInventoryError(f"selected record has unsupported lead {raw_lead!r}")
        canonical.append(canonical_lead)
    if len(set(canonical)) != len(LEADS) or set(canonical) != set(LEADS):
        raise ExternalInventoryError(
            "selected record does not contain the exact canonical lead set"
        )
    return tuple(canonical)


@dataclass(frozen=True, slots=True)
class ArchiveFileBinding:
    """Public-safe identity of one frozen source-archive part."""

    file_name: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _canonical_archive_leaf(self.file_name, "archive file_name")
        _require_positive_int(self.size_bytes, "archive size_bytes")
        _require_sha256(self.sha256, "archive sha256")

    def to_dict(self) -> dict[str, object]:
        return {
            "file_name": self.file_name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ArchiveFileBinding:
        if set(payload) != {"file_name", "sha256", "size_bytes"}:
            raise ExternalInventoryError("archive file binding fields differ")
        return cls(
            file_name=_canonical_archive_leaf(payload["file_name"], "archive file_name"),
            size_bytes=_require_positive_int(payload["size_bytes"], "archive size_bytes"),
            sha256=_require_sha256(payload["sha256"], "archive sha256"),
        )


@dataclass(frozen=True, slots=True)
class ArchiveMemberBinding:
    """Exact archive-member to evaluated-extraction byte mapping."""

    archive_member_path: str
    extracted_relative_path: str
    role: ArchiveMemberRole
    size_bytes: int
    sha256: str
    archive_crc32: str | None

    def __post_init__(self) -> None:
        _canonical_archive_path(self.archive_member_path, "archive_member_path")
        _canonical_archive_path(self.extracted_relative_path, "extracted_relative_path")
        if self.role not in {
            "wfdb_header",
            "wfdb_data",
            "quality_reference",
            "ignored_release_file",
        }:
            raise ExternalInventoryError("archive member role is unsupported")
        _require_nonnegative_int(self.size_bytes, "archive member size_bytes")
        _require_sha256(self.sha256, "archive member sha256")
        if self.archive_crc32 is not None:
            crc = _require_text(self.archive_crc32, "archive_crc32")
            if len(crc) != 8 or any(character not in "0123456789ABCDEF" for character in crc):
                raise ExternalInventoryError(
                    "archive_crc32 must be an eight-character uppercase hexadecimal value"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_crc32": self.archive_crc32,
            "archive_member_path": self.archive_member_path,
            "extracted_relative_path": self.extracted_relative_path,
            "role": self.role,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ArchiveMemberBinding:
        expected = {
            "archive_crc32",
            "archive_member_path",
            "extracted_relative_path",
            "role",
            "sha256",
            "size_bytes",
        }
        if set(payload) != expected:
            raise ExternalInventoryError("archive member binding fields differ")
        crc = payload["archive_crc32"]
        if crc is not None and not isinstance(crc, str):
            raise ExternalInventoryError("archive_crc32 must be text or null")
        return cls(
            archive_member_path=_canonical_archive_path(
                payload["archive_member_path"], "archive_member_path"
            ),
            extracted_relative_path=_canonical_archive_path(
                payload["extracted_relative_path"], "extracted_relative_path"
            ),
            role=cast(
                ArchiveMemberRole,
                _require_text(payload["role"], "archive member role"),
            ),
            size_bytes=_require_nonnegative_int(
                payload["size_bytes"], "archive member size_bytes"
            ),
            sha256=_require_sha256(payload["sha256"], "archive member sha256"),
            archive_crc32=crc,
        )


def _seven_zip_tool_body(binding: SevenZipToolBinding) -> dict[str, object]:
    return {
        "executable_name": binding.executable_name,
        "executable_sha256": binding.executable_sha256,
        "executable_size_bytes": binding.executable_size_bytes,
        "implementation": binding.implementation,
        "kind": SEVEN_ZIP_TOOL_KIND,
        "library_name": binding.library_name,
        "library_sha256": binding.library_sha256,
        "library_size_bytes": binding.library_size_bytes,
        "schema_version": SEVEN_ZIP_TOOL_SCHEMA_VERSION,
        "version": binding.version,
    }


@dataclass(frozen=True, slots=True)
class SevenZipToolBinding:
    """Path-free identity of the exact real 7-Zip binary and runtime library."""

    implementation: str
    version: str
    executable_name: str
    executable_size_bytes: int
    executable_sha256: str
    library_name: str
    library_size_bytes: int
    library_sha256: str

    def __post_init__(self) -> None:
        if self.implementation != "7zip":
            raise ExternalInventoryError("split archive tool implementation must be 7zip")
        version = _require_text(self.version, "7-Zip version")
        if re.fullmatch(r"[0-9]+\.[0-9]{2}", version) is None:
            raise ExternalInventoryError("7-Zip version must use the canonical major.minor form")
        _canonical_archive_leaf(self.executable_name, "7-Zip executable_name")
        _require_positive_int(
            self.executable_size_bytes, "7-Zip executable_size_bytes"
        )
        _require_sha256(self.executable_sha256, "7-Zip executable_sha256")
        _canonical_archive_leaf(self.library_name, "7-Zip library_name")
        _require_positive_int(self.library_size_bytes, "7-Zip library_size_bytes")
        _require_sha256(self.library_sha256, "7-Zip library_sha256")

    @property
    def tool_sha256(self) -> str:
        digest = hashlib.sha256(
            SEVEN_ZIP_TOOL_DOMAIN + _canonical_json_bytes(_seven_zip_tool_body(self))
        ).hexdigest()
        return f"sha256:{digest}"

    def to_dict(self) -> dict[str, object]:
        payload = _seven_zip_tool_body(self)
        payload["tool_sha256"] = self.tool_sha256
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SevenZipToolBinding:
        expected = {
            "executable_name",
            "executable_sha256",
            "executable_size_bytes",
            "implementation",
            "kind",
            "library_name",
            "library_sha256",
            "library_size_bytes",
            "schema_version",
            "tool_sha256",
            "version",
        }
        if set(payload) != expected:
            raise ExternalInventoryError("7-Zip tool binding fields differ")
        if payload["kind"] != SEVEN_ZIP_TOOL_KIND:
            raise ExternalInventoryError("7-Zip tool binding kind is unsupported")
        if payload["schema_version"] != SEVEN_ZIP_TOOL_SCHEMA_VERSION:
            raise ExternalInventoryError("7-Zip tool binding schema is unsupported")
        binding = cls(
            implementation=_require_text(payload["implementation"], "implementation"),
            version=_require_text(payload["version"], "7-Zip version"),
            executable_name=_canonical_archive_leaf(
                payload["executable_name"], "7-Zip executable_name"
            ),
            executable_size_bytes=_require_positive_int(
                payload["executable_size_bytes"], "7-Zip executable_size_bytes"
            ),
            executable_sha256=_require_sha256(
                payload["executable_sha256"], "7-Zip executable_sha256"
            ),
            library_name=_canonical_archive_leaf(
                payload["library_name"], "7-Zip library_name"
            ),
            library_size_bytes=_require_positive_int(
                payload["library_size_bytes"], "7-Zip library_size_bytes"
            ),
            library_sha256=_require_sha256(
                payload["library_sha256"], "7-Zip library_sha256"
            ),
        )
        if payload["tool_sha256"] != binding.tool_sha256:
            raise ExternalInventoryError("7-Zip tool binding self-hash does not match")
        return binding


def _archive_closure_body_from_parts(
    *,
    dataset: str,
    archive_format: ArchiveFormat,
    archive_root_prefix: str,
    archive_files: Sequence[ArchiveFileBinding],
    members: Sequence[ArchiveMemberBinding],
    tool_binding: SevenZipToolBinding | None,
) -> dict[str, object]:
    return {
        "archive_files": [binding.to_dict() for binding in archive_files],
        "archive_format": archive_format,
        "archive_root_prefix": archive_root_prefix,
        "dataset": dataset,
        "kind": ARCHIVE_CLOSURE_KIND,
        "members": [member.to_dict() for member in members],
        "schema_version": ARCHIVE_CLOSURE_SCHEMA_VERSION,
        "tool_binding": None if tool_binding is None else tool_binding.to_dict(),
    }


def _archive_closure_body(closure: ArchiveExtractionClosure) -> dict[str, object]:
    return _archive_closure_body_from_parts(
        dataset=closure.dataset,
        archive_format=closure.archive_format,
        archive_root_prefix=closure.archive_root_prefix,
        archive_files=closure.archive_files,
        members=closure.members,
        tool_binding=closure.tool_binding,
    )


@dataclass(frozen=True, slots=True)
class ArchiveExtractionClosure:
    """Domain-separated proof that archive bytes equal the evaluated tree."""

    dataset: str
    archive_format: ArchiveFormat
    archive_root_prefix: str
    archive_files: tuple[ArchiveFileBinding, ...]
    members: tuple[ArchiveMemberBinding, ...]
    tool_binding: SevenZipToolBinding | None
    closure_sha256: str

    def __post_init__(self) -> None:
        if self.dataset not in {CHALLENGE_2011_DATASET, ZZU_PEDIATRIC_DATASET}:
            raise ExternalInventoryError("archive closure dataset is unsupported")
        prefix = _canonical_archive_leaf(
            self.archive_root_prefix, "archive_root_prefix"
        )
        if not isinstance(self.archive_files, tuple) or not self.archive_files:
            raise ExternalInventoryError("archive_files must be a non-empty immutable tuple")
        if any(not isinstance(value, ArchiveFileBinding) for value in self.archive_files):
            raise TypeError("archive_files must contain ArchiveFileBinding values")
        expected_archives = tuple(sorted(self.archive_files, key=lambda value: value.file_name))
        if self.archive_files != expected_archives:
            raise ExternalInventoryError("archive_files are not in canonical order")
        archive_names = [value.file_name for value in self.archive_files]
        if len(archive_names) != len(set(archive_names)) or len(
            {value.casefold() for value in archive_names}
        ) != len(archive_names):
            raise ExternalInventoryError("archive_files contain duplicate names")
        if not isinstance(self.members, tuple) or not self.members:
            raise ExternalInventoryError("archive closure members must be non-empty")
        if any(not isinstance(value, ArchiveMemberBinding) for value in self.members):
            raise TypeError("members must contain ArchiveMemberBinding values")
        expected_members = tuple(
            sorted(self.members, key=lambda value: value.archive_member_path)
        )
        if self.members != expected_members:
            raise ExternalInventoryError("archive closure members are not in canonical order")
        archive_paths = [value.archive_member_path for value in self.members]
        extracted_paths = [value.extracted_relative_path for value in self.members]
        for field, paths in (
            ("archive member", archive_paths),
            ("extracted member", extracted_paths),
        ):
            if len(paths) != len(set(paths)) or len({path.casefold() for path in paths}) != len(
                paths
            ):
                raise ExternalInventoryError(f"archive closure contains duplicate {field} paths")
        for member in self.members:
            expected_archive_path = f"{prefix}/{member.extracted_relative_path}"
            if member.archive_member_path != expected_archive_path:
                raise ExternalInventoryError(
                    "archive member path does not map exactly to the extraction root"
                )
            suffix = PurePosixPath(member.extracted_relative_path).suffix
            if self.dataset == CHALLENGE_2011_DATASET:
                if suffix == ".hea":
                    expected_role: ArchiveMemberRole = "wfdb_header"
                elif suffix == ".dat":
                    expected_role = "wfdb_data"
                elif member.extracted_relative_path in {
                    "RECORDS",
                    "RECORDS-acceptable",
                    "RECORDS-unacceptable",
                }:
                    expected_role = "quality_reference"
                elif suffix == ".txt" or member.extracted_relative_path == "HEADER.shtml":
                    expected_role = "ignored_release_file"
                else:
                    raise ExternalInventoryError(
                        "Challenge archive contains a non-allowlisted release file"
                    )
            else:
                if suffix == ".hea":
                    expected_role = "wfdb_header"
                elif suffix == ".dat":
                    expected_role = "wfdb_data"
                else:
                    raise ExternalInventoryError(
                        "ZZU waveform archive may contain only .hea/.dat files"
                    )
            if member.role != expected_role:
                raise ExternalInventoryError(
                    "archive member role does not match its exact dataset path"
                )

        if self.dataset == CHALLENGE_2011_DATASET:
            if self.archive_format != "tar_gzip" or len(self.archive_files) != 1:
                raise ExternalInventoryError("Challenge closure requires one tar_gzip archive")
            if not self.archive_files[0].file_name.endswith(".tar.gz"):
                raise ExternalInventoryError("Challenge archive must use the .tar.gz suffix")
            if self.tool_binding is not None:
                raise ExternalInventoryError("Challenge tar closure cannot bind 7-Zip")
            if any(member.archive_crc32 is not None for member in self.members):
                raise ExternalInventoryError("Challenge tar members cannot carry ZIP CRC values")
        else:
            if self.archive_format != "split_zip_7zip" or len(self.archive_files) != 2:
                raise ExternalInventoryError("ZZU closure requires two split_zip_7zip parts")
            suffixes = {PurePosixPath(value.file_name).suffix for value in self.archive_files}
            stems = {PurePosixPath(value.file_name).stem for value in self.archive_files}
            if suffixes != {".z01", ".zip"} or len(stems) != 1:
                raise ExternalInventoryError("ZZU split archive parts must be matching .z01/.zip")
            if not isinstance(self.tool_binding, SevenZipToolBinding):
                raise ExternalInventoryError("ZZU closure requires an exact 7-Zip tool binding")
            if any(member.archive_crc32 is None for member in self.members):
                raise ExternalInventoryError("ZZU split-ZIP members require central-directory CRC")
        _require_prefixed_sha256(self.closure_sha256, "closure_sha256")
        expected_hash = "sha256:" + hashlib.sha256(
            ARCHIVE_CLOSURE_DOMAIN + _canonical_json_bytes(_archive_closure_body(self))
        ).hexdigest()
        if self.closure_sha256 != expected_hash:
            raise ExternalInventoryError("archive closure self-hash does not match")

    @property
    def member_count(self) -> int:
        return len(self.members)

    @property
    def member_bytes_total(self) -> int:
        return sum(member.size_bytes for member in self.members)

    @property
    def archive_bytes_total(self) -> int:
        return sum(binding.size_bytes for binding in self.archive_files)

    def to_dict(self) -> dict[str, object]:
        payload = _archive_closure_body(self)
        payload["closure_sha256"] = self.closure_sha256
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ArchiveExtractionClosure:
        expected = {
            "archive_files",
            "archive_format",
            "archive_root_prefix",
            "closure_sha256",
            "dataset",
            "kind",
            "members",
            "schema_version",
            "tool_binding",
        }
        if set(payload) != expected:
            raise ExternalInventoryError("archive closure fields differ")
        if payload["kind"] != ARCHIVE_CLOSURE_KIND:
            raise ExternalInventoryError("archive closure kind is unsupported")
        if payload["schema_version"] != ARCHIVE_CLOSURE_SCHEMA_VERSION:
            raise ExternalInventoryError("archive closure schema is unsupported")
        raw_archive_files = payload["archive_files"]
        raw_members = payload["members"]
        if not isinstance(raw_archive_files, list) or not raw_archive_files:
            raise ExternalInventoryError("archive_files must be a non-empty array")
        if not isinstance(raw_members, list) or not raw_members:
            raise ExternalInventoryError("archive closure members must be a non-empty array")
        archive_files = tuple(
            ArchiveFileBinding.from_dict(cast(dict[str, object], value))
            if isinstance(value, dict)
            else (_raise_archive_object("archive file"))
            for value in raw_archive_files
        )
        members = tuple(
            ArchiveMemberBinding.from_dict(cast(dict[str, object], value))
            if isinstance(value, dict)
            else (_raise_archive_object("archive member"))
            for value in raw_members
        )
        raw_tool = payload["tool_binding"]
        if raw_tool is None:
            tool = None
        elif isinstance(raw_tool, dict):
            tool = SevenZipToolBinding.from_dict(cast(dict[str, object], raw_tool))
        else:
            raise ExternalInventoryError("tool_binding must be an object or null")
        archive_format = payload["archive_format"]
        if archive_format not in {"tar_gzip", "split_zip_7zip"}:
            raise ExternalInventoryError("archive_format is unsupported")
        return cls(
            dataset=_require_text(payload["dataset"], "archive closure dataset"),
            archive_format=archive_format,
            archive_root_prefix=_canonical_archive_leaf(
                payload["archive_root_prefix"], "archive_root_prefix"
            ),
            archive_files=archive_files,
            members=members,
            tool_binding=tool,
            closure_sha256=_require_prefixed_sha256(
                payload["closure_sha256"], "closure_sha256"
            ),
        )


def _raise_archive_object(field: str) -> Never:
    raise ExternalInventoryError(f"{field} must be a JSON object")


@dataclass(frozen=True, slots=True)
class ExternalInventoryRecord:
    """Canonical metadata and raw-byte identity for one external WFDB record."""

    dataset: str
    dataset_version: str
    site: str
    site_alias: str
    patient_key: str | None
    record_ref: str
    source_role: str
    raw_header_sha256: str
    raw_header_size_bytes: int
    raw_data_sha256: str
    raw_data_size_bytes: int
    sampling_frequency_hz: float
    source_sample_count: int
    duration_seconds: float
    lead_count: int
    raw_ordered_leads: tuple[str, ...]
    canonical_ordered_leads: tuple[str, ...]
    raw_data_file_names: tuple[str, ...]
    raw_physical_units: tuple[str, ...]
    challenge_quality_label: ChallengeQualityLabel | None
    pediatric_12_lead: bool | None

    def __post_init__(self) -> None:
        if self.dataset not in {CHALLENGE_2011_DATASET, ZZU_PEDIATRIC_DATASET}:
            raise ExternalInventoryError("inventory record has an unsupported dataset")
        _require_text(self.dataset_version, "dataset_version")
        _require_text(self.site, "site")
        _require_text(self.site_alias, "site_alias")
        if self.patient_key is not None:
            _require_text(self.patient_key, "patient_key")
        _canonical_record_ref(self.record_ref)
        if self.source_role != CONFIRMATION_LOCKBOX_ROLE:
            raise ExternalInventoryError("external v2 records must use confirmation_lockbox")
        _require_sha256(self.raw_header_sha256, "raw_header_sha256")
        _require_sha256(self.raw_data_sha256, "raw_data_sha256")
        _require_positive_int(self.raw_header_size_bytes, "raw_header_size_bytes")
        _require_positive_int(self.raw_data_size_bytes, "raw_data_size_bytes")
        frequency = _require_positive_float(
            self.sampling_frequency_hz, "sampling_frequency_hz"
        )
        if not math.isclose(frequency, 500.0, rel_tol=0.0, abs_tol=0.0):
            raise ExternalInventoryError("selected external records must use exactly 500 Hz")
        sample_count = _require_positive_int(self.source_sample_count, "source_sample_count")
        duration = _require_positive_float(self.duration_seconds, "duration_seconds")
        if duration < 10.0:
            raise ExternalInventoryError("external record duration must be at least ten seconds")
        if duration != sample_count / frequency:
            raise ExternalInventoryError(
                "duration_seconds does not match source_sample_count and sampling_frequency_hz"
            )
        lead_count = _require_positive_int(self.lead_count, "lead_count")
        if lead_count != len(LEADS):
            raise ExternalInventoryError("selected external records must contain exactly 12 leads")
        if not isinstance(self.raw_ordered_leads, tuple):
            raise ExternalInventoryError("raw_ordered_leads must be an immutable tuple")
        if not isinstance(self.canonical_ordered_leads, tuple):
            raise ExternalInventoryError(
                "canonical_ordered_leads must be an immutable tuple"
            )
        raw_order = _text_sequence(
            self.raw_ordered_leads, "raw_ordered_leads", len(LEADS)
        )
        allowed_aliases: Mapping[str, str] = (
            {} if self.dataset == CHALLENGE_2011_DATASET else SOURCE_LEAD_ALIASES
        )
        canonical_order = _canonicalize_ordered_leads(
            raw_order,
            allowed_aliases=allowed_aliases,
        )
        if self.canonical_ordered_leads != canonical_order:
            raise ExternalInventoryError(
                "canonical_ordered_leads must be the fixed name-only mapping of "
                "raw_ordered_leads"
            )
        if not isinstance(self.raw_data_file_names, tuple):
            raise ExternalInventoryError("raw_data_file_names must be an immutable tuple")
        data_file_names = _text_sequence(
            self.raw_data_file_names, "raw_data_file_names", len(LEADS)
        )
        expected_data_file_name = f"{PurePosixPath(self.record_ref).name}.dat"
        if any(name != expected_data_file_name for name in data_file_names):
            raise ExternalInventoryError(
                "raw_data_file_names must bind every lead to the record-local .dat file"
            )
        if not isinstance(self.raw_physical_units, tuple):
            raise ExternalInventoryError("raw_physical_units must be an immutable tuple")
        raw_units = _text_sequence(
            self.raw_physical_units, "raw_physical_units", len(LEADS)
        )
        if raw_units != ("mV",) * len(LEADS):
            raise ExternalInventoryError(
                "raw_physical_units must contain exactly 12 case-sensitive 'mV' values"
            )

        if self.dataset == CHALLENGE_2011_DATASET:
            if self.dataset_version != CHALLENGE_2011_VERSION:
                raise ExternalInventoryError("Challenge 2011 dataset_version must be 1.0.0")
            if self.challenge_quality_label not in _QUALITY_LABELS:
                raise ExternalInventoryError(
                    "Challenge 2011 records require an explicit quality label"
                )
            if self.patient_key is not None:
                raise ExternalInventoryError(
                    "Challenge 2011 patient_key must be null because patient identity "
                    "is unavailable"
                )
            if self.pediatric_12_lead is not None:
                raise ExternalInventoryError(
                    "Challenge 2011 records cannot carry pediatric_12_lead metadata"
                )
        else:
            if self.dataset_version != ZZU_PEDIATRIC_VERSION:
                raise ExternalInventoryError("ZZU pediatric dataset_version must be 1")
            if self.patient_key is None:
                raise ExternalInventoryError("ZZU pediatric records require a patient_key")
            if self.challenge_quality_label is not None:
                raise ExternalInventoryError(
                    "ZZU pediatric records cannot carry a Challenge quality label"
                )
            if self.pediatric_12_lead is not True:
                raise ExternalInventoryError(
                    "selected ZZU pediatric records require pediatric_12_lead equal to true"
                )

    @property
    def identity(self) -> tuple[str, str, str]:
        return (self.dataset, self.site, self.record_ref)

    def to_dict(self) -> dict[str, object]:
        """Return the complete private canonical record representation."""

        return {
            "challenge_quality_label": self.challenge_quality_label,
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "duration_seconds": self.duration_seconds,
            "lead_count": self.lead_count,
            "canonical_ordered_leads": list(self.canonical_ordered_leads),
            "patient_key": self.patient_key,
            "pediatric_12_lead": self.pediatric_12_lead,
            "raw_data_sha256": self.raw_data_sha256,
            "raw_data_file_names": list(self.raw_data_file_names),
            "raw_data_size_bytes": self.raw_data_size_bytes,
            "raw_header_sha256": self.raw_header_sha256,
            "raw_header_size_bytes": self.raw_header_size_bytes,
            "raw_ordered_leads": list(self.raw_ordered_leads),
            "raw_physical_units": list(self.raw_physical_units),
            "record_ref": self.record_ref,
            "sampling_frequency_hz": self.sampling_frequency_hz,
            "source_sample_count": self.source_sample_count,
            "site": self.site,
            "site_alias": self.site_alias,
            "source_role": self.source_role,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ExternalInventoryRecord:
        """Parse one record while rejecting missing and unknown fields."""

        expected = {
            "challenge_quality_label",
            "dataset",
            "dataset_version",
            "duration_seconds",
            "lead_count",
            "canonical_ordered_leads",
            "patient_key",
            "pediatric_12_lead",
            "raw_data_sha256",
            "raw_data_file_names",
            "raw_data_size_bytes",
            "raw_header_sha256",
            "raw_header_size_bytes",
            "raw_ordered_leads",
            "raw_physical_units",
            "record_ref",
            "sampling_frequency_hz",
            "source_sample_count",
            "site",
            "site_alias",
            "source_role",
        }
        if set(payload) != expected:
            missing = sorted(expected.difference(payload))
            unknown = sorted(set(payload).difference(expected))
            raise ExternalInventoryError(
                f"inventory record fields differ; missing={missing!r}, unknown={unknown!r}"
            )
        quality_value = payload["challenge_quality_label"]
        if quality_value is not None and quality_value not in _QUALITY_LABELS:
            raise ExternalInventoryError("invalid Challenge quality label")
        pediatric_value = payload["pediatric_12_lead"]
        if pediatric_value is not None and not isinstance(pediatric_value, bool):
            raise ExternalInventoryError("pediatric_12_lead must be boolean or null")
        patient_value = payload["patient_key"]
        if patient_value is not None and not isinstance(patient_value, str):
            raise ExternalInventoryError("patient_key must be text or null")
        return cls(
            dataset=_require_text(payload["dataset"], "dataset"),
            dataset_version=_require_text(payload["dataset_version"], "dataset_version"),
            site=_require_text(payload["site"], "site"),
            site_alias=_require_text(payload["site_alias"], "site_alias"),
            patient_key=patient_value,
            record_ref=_canonical_record_ref(payload["record_ref"]),
            source_role=_require_text(payload["source_role"], "source_role"),
            raw_header_sha256=_require_sha256(
                payload["raw_header_sha256"], "raw_header_sha256"
            ),
            raw_header_size_bytes=_require_positive_int(
                payload["raw_header_size_bytes"], "raw_header_size_bytes"
            ),
            raw_data_sha256=_require_sha256(payload["raw_data_sha256"], "raw_data_sha256"),
            raw_data_file_names=_text_sequence(
                payload["raw_data_file_names"], "raw_data_file_names", len(LEADS)
            ),
            raw_data_size_bytes=_require_positive_int(
                payload["raw_data_size_bytes"], "raw_data_size_bytes"
            ),
            sampling_frequency_hz=_require_positive_float(
                payload["sampling_frequency_hz"], "sampling_frequency_hz"
            ),
            source_sample_count=_require_positive_int(
                payload["source_sample_count"], "source_sample_count"
            ),
            duration_seconds=_require_positive_float(
                payload["duration_seconds"], "duration_seconds"
            ),
            lead_count=_require_positive_int(payload["lead_count"], "lead_count"),
            raw_ordered_leads=_text_sequence(
                payload["raw_ordered_leads"], "raw_ordered_leads", len(LEADS)
            ),
            canonical_ordered_leads=_text_sequence(
                payload["canonical_ordered_leads"],
                "canonical_ordered_leads",
                len(LEADS),
            ),
            raw_physical_units=_text_sequence(
                payload["raw_physical_units"], "raw_physical_units", len(LEADS)
            ),
            challenge_quality_label=quality_value,
            pediatric_12_lead=pediatric_value,
        )


def _inventory_body(
    records: Sequence[ExternalInventoryRecord],
    archive_closures: Sequence[ArchiveExtractionClosure],
) -> dict[str, object]:
    return {
        "archive_closures": [closure.to_dict() for closure in archive_closures],
        "kind": INVENTORY_KIND,
        "records": [record.to_dict() for record in records],
        "schema_version": INVENTORY_SCHEMA_VERSION,
    }


def _inventory_sha256(
    records: Sequence[ExternalInventoryRecord],
    archive_closures: Sequence[ArchiveExtractionClosure],
) -> str:
    digest = hashlib.sha256(
        INVENTORY_DOMAIN
        + _canonical_json_bytes(_inventory_body(records, archive_closures))
    )
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True, slots=True)
class ExternalWaveformInventory:
    """Canonical, self-hashed collection of private record metadata."""

    records: tuple[ExternalInventoryRecord, ...]
    archive_closures: tuple[ArchiveExtractionClosure, ...]
    inventory_sha256: str

    def __post_init__(self) -> None:
        if not self.records:
            raise ExternalInventoryError("external inventory must not be empty")
        if any(not isinstance(record, ExternalInventoryRecord) for record in self.records):
            raise TypeError("records must contain ExternalInventoryRecord values")
        expected_order = tuple(sorted(self.records, key=lambda record: record.identity))
        if self.records != expected_order:
            raise ExternalInventoryError("inventory records are not in canonical order")
        identities = [record.identity for record in self.records]
        if len(identities) != len(set(identities)):
            raise ExternalInventoryError("inventory contains duplicate record identities")
        byte_pairs = [
            (record.raw_header_sha256, record.raw_data_sha256) for record in self.records
        ]
        if len(byte_pairs) != len(set(byte_pairs)):
            raise ExternalInventoryError("inventory binds the same raw WFDB bytes more than once")
        if not isinstance(self.archive_closures, tuple):
            raise ExternalInventoryError("archive_closures must be an immutable tuple")
        if any(
            not isinstance(closure, ArchiveExtractionClosure)
            for closure in self.archive_closures
        ):
            raise TypeError("archive_closures must contain ArchiveExtractionClosure values")
        expected_closure_order = tuple(
            sorted(self.archive_closures, key=lambda closure: closure.dataset)
        )
        if self.archive_closures != expected_closure_order:
            raise ExternalInventoryError("archive_closures are not in canonical order")
        closure_datasets = [closure.dataset for closure in self.archive_closures]
        if len(closure_datasets) != len(set(closure_datasets)):
            raise ExternalInventoryError("inventory contains duplicate archive closures")
        record_datasets = {record.dataset for record in self.records}
        if self.archive_closures and set(closure_datasets) != record_datasets:
            raise ExternalInventoryError(
                "archive closures must cover exactly the inventory record datasets"
            )
        expected_hash = _inventory_sha256(self.records, self.archive_closures)
        if self.inventory_sha256 != expected_hash:
            raise ExternalInventoryError("inventory self-hash does not match its canonical records")

    @property
    def record_count(self) -> int:
        return len(self.records)

    def to_dict(self) -> dict[str, object]:
        payload = _inventory_body(self.records, self.archive_closures)
        payload["inventory_sha256"] = self.inventory_sha256
        return payload

    def to_canonical_json_bytes(self) -> bytes:
        """Return the one accepted on-disk representation, including final newline."""

        return _canonical_json_bytes(self.to_dict()) + b"\n"


def build_external_inventory(
    records: Sequence[ExternalInventoryRecord],
    *,
    archive_closures: Sequence[ArchiveExtractionClosure] = (),
) -> ExternalWaveformInventory:
    """Sort, deduplicate, and self-hash canonical external inventory records."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence) or not records:
        raise ExternalInventoryError("records must be a non-empty sequence")
    normalized: list[ExternalInventoryRecord] = []
    for record in records:
        if not isinstance(record, ExternalInventoryRecord):
            raise TypeError("records must contain ExternalInventoryRecord values")
        normalized.append(record)
    ordered = tuple(sorted(normalized, key=lambda record: record.identity))
    if isinstance(archive_closures, (str, bytes)) or not isinstance(
        archive_closures, Sequence
    ):
        raise TypeError("archive_closures must be a sequence")
    normalized_closures: list[ArchiveExtractionClosure] = []
    for closure in archive_closures:
        if not isinstance(closure, ArchiveExtractionClosure):
            raise TypeError("archive_closures must contain ArchiveExtractionClosure values")
        normalized_closures.append(closure)
    ordered_closures = tuple(
        sorted(normalized_closures, key=lambda closure: closure.dataset)
    )
    return ExternalWaveformInventory(
        ordered,
        ordered_closures,
        _inventory_sha256(ordered, ordered_closures),
    )


def save_external_inventory(inventory: ExternalWaveformInventory, path: str | Path) -> None:
    """Atomically save the exact canonical JSON bytes for an inventory."""

    if not isinstance(inventory, ExternalWaveformInventory):
        raise TypeError("inventory must be ExternalWaveformInventory")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(inventory.to_canonical_json_bytes())
    os.replace(temporary, destination)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ExternalInventoryError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def load_external_inventory(path: str | Path) -> ExternalWaveformInventory:
    """Load exact canonical JSON and verify its domain-separated self-hash."""

    source = Path(path)
    _assert_direct_ancestry(source, field="external inventory")
    try:
        with source.open("rb") as source_file:
            declared_size = os.fstat(source_file.fileno()).st_size
            if declared_size > MAX_INVENTORY_BYTES:
                raise ExternalInventoryError("external inventory exceeds the bounded size limit")
            raw = source_file.read(MAX_INVENTORY_BYTES + 1)
        if _is_indirect(source):
            raise ExternalInventoryError("external inventory became indirect while read")
        if len(raw) > MAX_INVENTORY_BYTES or len(raw) != declared_size:
            raise ExternalInventoryError(
                "external inventory exceeded its bound or changed while being read"
            )
        decoded: object = json.loads(raw, object_pairs_hook=_unique_object)
    except ExternalInventoryError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExternalInventoryError(f"could not load external inventory: {error}") from error
    if not isinstance(decoded, dict):
        raise ExternalInventoryError("external inventory must contain a JSON object")
    payload = cast(dict[str, object], decoded)
    expected = {
        "archive_closures",
        "inventory_sha256",
        "kind",
        "records",
        "schema_version",
    }
    if set(payload) != expected:
        raise ExternalInventoryError("external inventory has missing or unknown top-level fields")
    if payload["kind"] != INVENTORY_KIND:
        raise ExternalInventoryError("external inventory kind is unsupported")
    if payload["schema_version"] != INVENTORY_SCHEMA_VERSION:
        raise ExternalInventoryError("external inventory schema version is unsupported")
    raw_records = payload["records"]
    if not isinstance(raw_records, list) or not raw_records:
        raise ExternalInventoryError("external inventory records must be a non-empty array")
    records: list[ExternalInventoryRecord] = []
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            raise ExternalInventoryError("every inventory record must be a JSON object")
        records.append(ExternalInventoryRecord.from_dict(cast(dict[str, object], raw_record)))
    raw_closures = payload["archive_closures"]
    if not isinstance(raw_closures, list):
        raise ExternalInventoryError("archive_closures must be an array")
    closures: list[ArchiveExtractionClosure] = []
    for raw_closure in raw_closures:
        if not isinstance(raw_closure, dict):
            raise ExternalInventoryError("every archive closure must be a JSON object")
        closures.append(
            ArchiveExtractionClosure.from_dict(cast(dict[str, object], raw_closure))
        )
    digest = payload["inventory_sha256"]
    if not isinstance(digest, str):
        raise ExternalInventoryError("inventory_sha256 must be text")
    inventory = ExternalWaveformInventory(tuple(records), tuple(closures), digest)
    if raw != inventory.to_canonical_json_bytes():
        raise ExternalInventoryError("external inventory JSON is not in exact canonical form")
    return inventory


@dataclass(frozen=True, slots=True)
class _HeaderMetadata:
    sampling_frequency_hz: float
    sample_count: int
    duration_seconds: float
    lead_count: int
    raw_ordered_leads: tuple[str, ...]
    raw_data_file_names: tuple[str, ...]
    raw_physical_units: tuple[str, ...]


def _read_header_metadata(record_base: Path) -> _HeaderMetadata:
    """Read only WFDB header metadata; this function never opens signal amplitudes."""

    try:
        header = wfdb.rdheader(str(record_base))
    except Exception as error:
        raise ExternalInventoryError(
            f"could not read WFDB header {record_base!s}: {error}"
        ) from error
    sampling_frequency_hz = _require_positive_float(
        getattr(header, "fs", None), "header sampling_frequency_hz"
    )
    sample_count = _require_positive_int(getattr(header, "sig_len", None), "header sample_count")
    lead_count = _require_positive_int(getattr(header, "n_sig", None), "header lead_count")
    raw_ordered_leads = _text_sequence(
        getattr(header, "sig_name", None), "header raw_ordered_leads", lead_count
    )
    raw_data_file_names = _text_sequence(
        getattr(header, "file_name", None), "header raw_data_file_names", lead_count
    )
    expected_data_file_name = f"{record_base.name}.dat"
    if any(name != expected_data_file_name for name in raw_data_file_names):
        raise ExternalInventoryError(
            f"WFDB header must bind every lead to {expected_data_file_name!r}"
        )
    raw_physical_units = _text_sequence(
        getattr(header, "units", None), "header raw_physical_units", lead_count
    )
    duration_seconds = sample_count / sampling_frequency_hz
    if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ExternalInventoryError("WFDB header declares a non-positive duration")
    return _HeaderMetadata(
        sampling_frequency_hz,
        sample_count,
        duration_seconds,
        lead_count,
        raw_ordered_leads,
        raw_data_file_names,
        raw_physical_units,
    )


def _selected_lead_orders(
    metadata: _HeaderMetadata,
    *,
    allowed_aliases: Mapping[str, str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not math.isclose(
        metadata.sampling_frequency_hz,
        500.0,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ExternalInventoryError("selected external record must use exactly 500 Hz")
    if metadata.duration_seconds < 10.0:
        raise ExternalInventoryError("selected external record is shorter than ten seconds")
    if metadata.lead_count != len(LEADS):
        raise ExternalInventoryError("selected external record must contain exactly 12 leads")
    if metadata.raw_physical_units != ("mV",) * len(LEADS):
        raise ExternalInventoryError(
            "selected external record must use exactly 12 case-sensitive 'mV' units"
        )
    canonical_ordered_leads = _canonicalize_ordered_leads(
        metadata.raw_ordered_leads,
        allowed_aliases=allowed_aliases,
    )
    return metadata.raw_ordered_leads, canonical_ordered_leads


def _safe_record_files(dataset_root: Path, record_ref: str) -> tuple[Path, Path, Path]:
    if not isinstance(dataset_root, Path):
        raise TypeError("dataset_root must be a pathlib.Path")
    canonical_ref = _canonical_record_ref(record_ref)
    _assert_direct_ancestry(dataset_root, field="dataset_root")
    try:
        root = dataset_root.resolve(strict=True)
    except OSError as error:
        raise ExternalInventoryError(f"dataset_root is unavailable: {error}") from error
    if not root.is_dir():
        raise ExternalInventoryError("dataset_root must be a directory")
    _assert_direct_ancestry(root, field="resolved dataset_root")
    base = root.joinpath(*PurePosixPath(canonical_ref).parts)
    try:
        base.relative_to(root)
    except ValueError as error:
        raise ExternalInventoryError("record_ref escapes dataset_root") from error

    current = root
    for part in PurePosixPath(canonical_ref).parts[:-1]:
        current = current / part
        if _is_indirect(current):
            raise ExternalInventoryError("record_ref traverses an indirect directory")
        if not current.is_dir():
            raise ExternalInventoryError(f"record directory is missing: {current!s}")
        _resolve_direct_within(current, root, field="record directory")
    header_path = Path(f"{base}.hea")
    data_path = Path(f"{base}.dat")
    for kind, path in (("header", header_path), ("data", data_path)):
        if _is_indirect(path):
            raise ExternalInventoryError(
                f"raw {kind} file must not be a symlink or junction"
            )
        if not path.is_file():
            raise ExternalInventoryError(f"raw {kind} file is missing: {path!s}")
        _resolve_direct_within(path, root, field=f"raw {kind} file")
    return base, header_path, data_path


def enumerate_wfdb_record_refs(dataset_root: Path) -> tuple[str, ...]:
    """Enumerate the exact paired ``.hea``/``.dat`` identities without decoding data."""

    if not isinstance(dataset_root, Path):
        raise TypeError("dataset_root must be a pathlib.Path")
    _assert_direct_ancestry(dataset_root, field="dataset_root")
    try:
        root = dataset_root.resolve(strict=True)
    except OSError as error:
        raise ExternalInventoryError("dataset_root is unavailable") from error
    if not root.is_dir():
        raise ExternalInventoryError("dataset_root must be a directory")
    _assert_direct_ancestry(root, field="resolved dataset_root")

    suffixes_by_ref: dict[str, set[str]] = {}
    for current_text, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_text)
        _resolve_direct_within(current, root, field="dataset directory")
        directory_names.sort()
        file_names.sort()
        for directory_name in directory_names:
            directory = current / directory_name
            if _is_indirect(directory):
                raise ExternalInventoryError("dataset tree contains an indirect directory")
            _resolve_direct_within(directory, root, field="dataset directory")
        for file_name in file_names:
            path = current / file_name
            if _is_indirect(path):
                raise ExternalInventoryError("dataset tree contains an indirect file")
            _resolve_direct_within(path, root, field="dataset file")
            suffix = path.suffix
            if suffix.casefold() not in {".hea", ".dat"}:
                continue
            if suffix not in {".hea", ".dat"}:
                raise ExternalInventoryError("WFDB file suffixes must be lowercase")
            relative = path.relative_to(root).as_posix()
            record_ref = _canonical_record_ref(relative[: -len(suffix)])
            seen_suffixes = suffixes_by_ref.setdefault(record_ref, set())
            if suffix in seen_suffixes:
                raise ExternalInventoryError("dataset tree contains a duplicate WFDB member")
            seen_suffixes.add(suffix)
    if not suffixes_by_ref:
        raise ExternalInventoryError("dataset tree contains no WFDB records")
    incomplete = sorted(
        record_ref
        for record_ref, suffixes in suffixes_by_ref.items()
        if suffixes != {".hea", ".dat"}
    )
    if incomplete:
        raise ExternalInventoryError("dataset tree contains an unpaired WFDB record")
    return tuple(sorted(suffixes_by_ref))


def verify_wfdb_candidate_file_set(
    dataset_root: Path,
    expected_record_refs: Sequence[str],
) -> tuple[str, ...]:
    """Fail when the extracted WFDB candidate set has any extra or missing pair."""

    if isinstance(expected_record_refs, (str, bytes)) or not isinstance(
        expected_record_refs, Sequence
    ):
        raise TypeError("expected_record_refs must be a sequence")
    expected = tuple(_canonical_record_ref(value) for value in expected_record_refs)
    if not expected or len(expected) != len(set(expected)):
        raise ExternalInventoryError("expected WFDB record identities must be non-empty and unique")
    observed = enumerate_wfdb_record_refs(dataset_root)
    if set(observed) != set(expected):
        raise ExternalInventoryError("extracted WFDB candidate set has extra or missing records")
    return observed


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    size_bytes: int
    sha256: str
    crc32: str


@dataclass(frozen=True, slots=True)
class _ExtractionTreeSnapshot:
    files: Mapping[str, _FileSnapshot]
    directories: frozenset[str]


def _snapshot_binary_stream(handle: object) -> _FileSnapshot:
    reader = getattr(handle, "read", None)
    if not callable(reader):
        raise ExternalInventoryError("archive member stream is unreadable")
    digest = hashlib.sha256()
    crc = 0
    size = 0
    while True:
        chunk = reader(1024 * 1024)
        if not isinstance(chunk, bytes):
            raise ExternalInventoryError("archive member stream returned non-byte data")
        if not chunk:
            break
        size += len(chunk)
        digest.update(chunk)
        crc = binascii.crc32(chunk, crc)
    return _FileSnapshot(size, digest.hexdigest(), f"{crc & 0xFFFFFFFF:08X}")


def _stable_file_snapshot(path: Path, *, field: str) -> _FileSnapshot:
    if not isinstance(path, Path):
        raise TypeError(f"{field} must be a pathlib.Path")
    if _is_indirect(path) or not path.is_file():
        raise ExternalInventoryError(f"{field} must be a regular direct file")
    before = path.stat()
    try:
        with path.open("rb") as handle:
            snapshot = _snapshot_binary_stream(handle)
    except OSError as error:
        raise ExternalInventoryError(f"could not hash {field}") from error
    after = path.stat()
    if _is_indirect(path) or (
        before.st_size,
        before.st_mtime_ns,
        getattr(before, "st_ino", None),
    ) != (
        after.st_size,
        after.st_mtime_ns,
        getattr(after, "st_ino", None),
    ):
        raise ExternalInventoryError(f"{field} changed while it was being hashed")
    if snapshot.size_bytes != after.st_size:
        raise ExternalInventoryError(f"{field} size changed while it was being hashed")
    return snapshot


def _snapshot_extraction_tree(root_path: Path) -> _ExtractionTreeSnapshot:
    if not isinstance(root_path, Path):
        raise TypeError("extraction root must be a pathlib.Path")
    _assert_direct_ancestry(root_path, field="extraction root")
    try:
        root = root_path.resolve(strict=True)
    except OSError as error:
        raise ExternalInventoryError("extraction root is unavailable") from error
    if not root.is_dir():
        raise ExternalInventoryError("extraction root must be a directory")
    _assert_direct_ancestry(root, field="resolved extraction root")

    files: dict[str, _FileSnapshot] = {}
    directories: set[str] = set()
    casefolded_paths: set[str] = set()
    for current_text, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_text)
        _resolve_direct_within(current, root, field="extraction directory")
        directory_names.sort()
        file_names.sort()
        for directory_name in directory_names:
            directory = current / directory_name
            if _is_indirect(directory):
                raise ExternalInventoryError("extraction tree contains an indirect directory")
            _resolve_direct_within(directory, root, field="extraction directory")
            relative = _canonical_archive_path(
                directory.relative_to(root).as_posix(), "extracted directory path"
            )
            if relative.casefold() in casefolded_paths:
                raise ExternalInventoryError(
                    "extraction tree contains a case-insensitive path collision"
                )
            casefolded_paths.add(relative.casefold())
            directories.add(relative)
        for file_name in file_names:
            path = current / file_name
            if _is_indirect(path) or not path.is_file():
                raise ExternalInventoryError("extraction tree contains a non-regular file")
            _resolve_direct_within(path, root, field="extracted file")
            relative = _canonical_archive_path(
                path.relative_to(root).as_posix(), "extracted file path"
            )
            if relative.casefold() in casefolded_paths:
                raise ExternalInventoryError(
                    "extraction tree contains a case-insensitive path collision"
                )
            casefolded_paths.add(relative.casefold())
            files[relative] = _stable_file_snapshot(path, field="extracted file")
    if not files:
        raise ExternalInventoryError("extraction tree contains no regular files")
    return _ExtractionTreeSnapshot(MappingProxyType(files), frozenset(directories))


def _archive_file_binding(path: Path, snapshot: _FileSnapshot) -> ArchiveFileBinding:
    return ArchiveFileBinding(
        file_name=_canonical_archive_leaf(path.name, "archive file_name"),
        size_bytes=snapshot.size_bytes,
        sha256=snapshot.sha256,
    )


def _mapped_archive_path(path: str, *, root_prefix: str) -> str | None:
    canonical_path = _canonical_archive_path(path, "archive member path")
    prefix = _canonical_archive_leaf(root_prefix, "archive_root_prefix")
    parts = PurePosixPath(canonical_path).parts
    if not parts or parts[0] != prefix:
        raise ExternalInventoryError("archive member is outside the exact frozen root prefix")
    if len(parts) == 1:
        return None
    return PurePosixPath(*parts[1:]).as_posix()


def _parent_directories(paths: Sequence[str]) -> set[str]:
    result: set[str] = set()
    for value in paths:
        parent = PurePosixPath(value).parent
        while parent.as_posix() != ".":
            result.add(parent.as_posix())
            parent = parent.parent
    return result


def _normalize_required_archive_paths(paths: Sequence[str]) -> frozenset[str]:
    if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence):
        raise TypeError("expected_required_relative_paths must be a sequence")
    normalized = tuple(
        _canonical_archive_path(value, "required extracted path") for value in paths
    )
    if not normalized or len(normalized) != len(set(normalized)):
        raise ExternalInventoryError("required extracted paths must be non-empty and unique")
    return frozenset(normalized)


def _archive_member_role(dataset: str, relative_path: str) -> ArchiveMemberRole:
    suffix = PurePosixPath(relative_path).suffix
    if suffix == ".hea":
        return "wfdb_header"
    if suffix == ".dat":
        return "wfdb_data"
    if dataset == CHALLENGE_2011_DATASET and relative_path in {
        "RECORDS",
        "RECORDS-acceptable",
        "RECORDS-unacceptable",
    }:
        return "quality_reference"
    if dataset == CHALLENGE_2011_DATASET and (
        suffix == ".txt" or relative_path == "HEADER.shtml"
    ):
        return "ignored_release_file"
    raise ExternalInventoryError("archive contains a non-allowlisted release file")


def _build_archive_closure(
    *,
    dataset: str,
    archive_format: ArchiveFormat,
    archive_root_prefix: str,
    archive_files: Sequence[ArchiveFileBinding],
    members: Sequence[ArchiveMemberBinding],
    tool_binding: SevenZipToolBinding | None,
) -> ArchiveExtractionClosure:
    ordered_archives = tuple(sorted(archive_files, key=lambda value: value.file_name))
    ordered_members = tuple(sorted(members, key=lambda value: value.archive_member_path))
    body = _archive_closure_body_from_parts(
        dataset=dataset,
        archive_format=archive_format,
        archive_root_prefix=archive_root_prefix,
        archive_files=ordered_archives,
        members=ordered_members,
        tool_binding=tool_binding,
    )
    closure_sha256 = "sha256:" + hashlib.sha256(
        ARCHIVE_CLOSURE_DOMAIN + _canonical_json_bytes(body)
    ).hexdigest()
    return ArchiveExtractionClosure(
        dataset=dataset,
        archive_format=archive_format,
        archive_root_prefix=archive_root_prefix,
        archive_files=ordered_archives,
        members=ordered_members,
        tool_binding=tool_binding,
        closure_sha256=closure_sha256,
    )


def _validate_archive_and_extraction_sets(
    *,
    archive_file_paths: set[str],
    archive_directory_paths: set[str],
    extraction: _ExtractionTreeSnapshot,
    expected_required_relative_paths: Sequence[str],
) -> None:
    required = _normalize_required_archive_paths(expected_required_relative_paths)
    if not required.issubset(archive_file_paths):
        raise ExternalInventoryError("archive is missing a required evaluated member")
    if set(extraction.files) != archive_file_paths:
        raise ExternalInventoryError(
            "archive and evaluated extraction have extra or missing file members"
        )
    expected_directories = archive_directory_paths.union(
        _parent_directories(tuple(archive_file_paths))
    )
    if extraction.directories != expected_directories:
        raise ExternalInventoryError(
            "archive and evaluated extraction have extra or missing directories"
        )


def build_challenge_tar_extraction_closure(
    archive_path: Path,
    extraction_root: Path,
    *,
    expected_required_relative_paths: Sequence[str],
    archive_root_prefix: str = "set-a",
) -> ArchiveExtractionClosure:
    """Close every safe tar file byte to the evaluated Challenge extraction tree."""

    _assert_direct_ancestry(archive_path, field="Challenge archive")
    _assert_direct_ancestry(extraction_root, field="Challenge extraction root")
    prefix = _canonical_archive_leaf(archive_root_prefix, "archive_root_prefix")
    before_archive = _stable_file_snapshot(archive_path, field="Challenge archive")
    extracted = _snapshot_extraction_tree(extraction_root)
    member_snapshots: dict[str, tuple[str, _FileSnapshot]] = {}
    archive_directories: set[str] = set()
    seen_archive_paths: set[str] = set()
    seen_casefolded_paths: set[str] = set()
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                raw_member_name = (
                    member.name[:-1]
                    if member.isdir() and member.name.endswith("/")
                    else member.name
                )
                archive_member_path = _canonical_archive_path(
                    raw_member_name, "Challenge tar member path"
                )
                if (
                    archive_member_path in seen_archive_paths
                    or archive_member_path.casefold() in seen_casefolded_paths
                ):
                    raise ExternalInventoryError(
                        "Challenge tar contains duplicate or colliding member paths"
                    )
                seen_archive_paths.add(archive_member_path)
                seen_casefolded_paths.add(archive_member_path.casefold())
                relative_path = _mapped_archive_path(
                    archive_member_path, root_prefix=prefix
                )
                if member.isdir():
                    if relative_path is not None:
                        archive_directories.add(relative_path)
                    continue
                if not member.isfile():
                    raise ExternalInventoryError(
                        "Challenge tar contains a link or non-regular member"
                    )
                if relative_path is None:
                    raise ExternalInventoryError("Challenge tar root member must be a directory")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ExternalInventoryError("Challenge tar member could not be read")
                with stream:
                    member_snapshot = _snapshot_binary_stream(stream)
                if member_snapshot.size_bytes != member.size:
                    raise ExternalInventoryError("Challenge tar member size is inconsistent")
                member_snapshots[relative_path] = (
                    archive_member_path,
                    member_snapshot,
                )
    except ExternalInventoryError:
        raise
    except (OSError, tarfile.TarError) as error:
        raise ExternalInventoryError("Challenge archive could not be safely read") from error

    after_archive = _stable_file_snapshot(archive_path, field="Challenge archive")
    if after_archive != before_archive:
        raise ExternalInventoryError("Challenge archive changed during closure construction")
    extracted_after = _snapshot_extraction_tree(extraction_root)
    if extracted_after != extracted:
        raise ExternalInventoryError(
            "Challenge extraction changed during closure construction"
        )
    _validate_archive_and_extraction_sets(
        archive_file_paths=set(member_snapshots),
        archive_directory_paths=archive_directories,
        extraction=extracted,
        expected_required_relative_paths=expected_required_relative_paths,
    )
    bindings: list[ArchiveMemberBinding] = []
    for relative_path, (archive_member_path, member_snapshot) in member_snapshots.items():
        extracted_snapshot = extracted.files[relative_path]
        if (
            extracted_snapshot.size_bytes != member_snapshot.size_bytes
            or extracted_snapshot.sha256 != member_snapshot.sha256
        ):
            raise ExternalInventoryError(
                "Challenge archive member bytes differ from the evaluated extraction"
            )
        bindings.append(
            ArchiveMemberBinding(
                archive_member_path=archive_member_path,
                extracted_relative_path=relative_path,
                role=_archive_member_role(CHALLENGE_2011_DATASET, relative_path),
                size_bytes=member_snapshot.size_bytes,
                sha256=member_snapshot.sha256,
                archive_crc32=None,
            )
        )
    return _build_archive_closure(
        dataset=CHALLENGE_2011_DATASET,
        archive_format="tar_gzip",
        archive_root_prefix=prefix,
        archive_files=(_archive_file_binding(archive_path, before_archive),),
        members=bindings,
        tool_binding=None,
    )


def verify_challenge_tar_extraction_closure(
    archive_path: Path,
    extraction_root: Path,
    closure: ArchiveExtractionClosure,
) -> str:
    """Rebuild and exactly compare a frozen Challenge archive closure."""

    if closure.dataset != CHALLENGE_2011_DATASET:
        raise ExternalInventoryError("expected a Challenge archive closure")
    rebuilt = build_challenge_tar_extraction_closure(
        archive_path,
        extraction_root,
        expected_required_relative_paths=tuple(
            member.extracted_relative_path for member in closure.members
        ),
        archive_root_prefix=closure.archive_root_prefix,
    )
    if rebuilt != closure:
        raise ExternalInventoryError("Challenge archive closure no longer matches")
    return closure.closure_sha256


class _SevenZipRunner(Protocol):
    def __call__(self, executable: Path, arguments: tuple[str, ...]) -> str: ...


@dataclass(frozen=True, slots=True)
class _ResolvedSevenZipTool:
    executable: Path
    library: Path
    binding: SevenZipToolBinding


@dataclass(frozen=True, slots=True)
class _SevenZipListedMember:
    path: str
    size_bytes: int
    crc32: str | None
    is_directory: bool


def _run_seven_zip(executable: Path, arguments: tuple[str, ...]) -> str:
    """Run the bound CLI in a fresh, two-file application directory.

    A normal 7-Zip installation can discover adjacent ``Codecs``/``Formats``
    plug-ins and Windows DLL search can consult the caller's working directory
    and ``PATH``.  Production execution therefore copies only the already
    byte-bound ``7z.exe`` and ``7z.dll`` into a fresh direct directory, uses a
    distinct fresh working directory, and supplies a minimal environment.
    """

    _assert_direct_ancestry(executable, field="bound executable")
    if not executable.is_file() or _is_indirect(executable):
        raise ExternalInventoryError("bound executable must be a regular direct file")
    source_executable_snapshot = _stable_file_snapshot(
        executable, field="bound executable"
    )
    source_library = executable.with_name("7z.dll")
    use_two_file_replica = executable.name.casefold() == "7z.exe"
    source_library_snapshot: _FileSnapshot | None = None
    if use_two_file_replica:
        _assert_direct_ancestry(source_library, field="bound 7-Zip library")
        source_library_snapshot = _stable_file_snapshot(
            source_library, field="bound 7-Zip library"
        )

    process: subprocess.Popen[bytes] | None = None
    try:
        with (
            tempfile.TemporaryDirectory(prefix="ecg-trust-7zip-run-") as raw_run_root,
            tempfile.TemporaryFile(mode="w+b") as stdout_file,
            tempfile.TemporaryFile(mode="w+b") as stderr_file,
        ):
            run_root = Path(raw_run_root)
            _assert_direct_ancestry(run_root, field="7-Zip run root")
            if tuple(run_root.iterdir()):
                raise ExternalInventoryError("fresh 7-Zip run root is not empty")
            tool_root = run_root / "tool"
            working_root = run_root / "work"
            tool_root.mkdir()
            working_root.mkdir()
            _assert_direct_ancestry(tool_root, field="7-Zip tool replica")
            _assert_direct_ancestry(working_root, field="7-Zip working directory")
            command_executable = executable
            if use_two_file_replica:
                replica_executable = tool_root / executable.name
                replica_library = tool_root / source_library.name
                shutil.copyfile(executable, replica_executable)
                shutil.copyfile(source_library, replica_library)
                if (
                    _stable_file_snapshot(
                        replica_executable, field="replicated 7-Zip executable"
                    )
                    != source_executable_snapshot
                    or _stable_file_snapshot(
                        replica_library, field="replicated 7-Zip library"
                    )
                    != source_library_snapshot
                    or tuple(sorted(path.name for path in tool_root.iterdir()))
                    != (source_library.name, executable.name)
                ):
                    raise ExternalInventoryError("7-Zip two-file replica differs")
                command_executable = replica_executable
            system_root = Path(os.environ.get("SYSTEMROOT", r"C:\Windows"))
            system32 = system_root / "System32"
            path_separator = os.pathsep
            safe_path = path_separator.join((os.fspath(tool_root), os.fspath(system32)))
            safe_environment = {
                "COMSPEC": os.fspath(system32 / "cmd.exe"),
                "PATHEXT": ".COM;.EXE;.BAT;.CMD",
                "PATH": safe_path,
                "SystemDrive": system_root.drive or "C:",
                "SystemRoot": os.fspath(system_root),
                "TEMP": os.fspath(working_root),
                "TMP": os.fspath(working_root),
                "WINDIR": os.fspath(system_root),
            }
            process = subprocess.Popen(
                [os.fspath(command_executable), *arguments],
                stdout=stdout_file,
                stderr=stderr_file,
                cwd=working_root,
                env=safe_environment,
                close_fds=True,
                shell=False,
            )
            deadline = time.monotonic() + SEVEN_ZIP_TIMEOUT_SECONDS
            while process.poll() is None:
                stdout_size = os.fstat(stdout_file.fileno()).st_size
                stderr_size = os.fstat(stderr_file.fileno()).st_size
                if (
                    stdout_size > SEVEN_ZIP_STDOUT_LIMIT_BYTES
                    or stderr_size > SEVEN_ZIP_STDERR_LIMIT_BYTES
                ):
                    process.kill()
                    process.wait()
                    raise ExternalInventoryError("7-Zip output exceeded its bounded size limit")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    process.kill()
                    process.wait()
                    raise ExternalInventoryError("the bound 7-Zip command timed out")
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=min(SEVEN_ZIP_POLL_SECONDS, remaining))

            stdout_size = os.fstat(stdout_file.fileno()).st_size
            stderr_size = os.fstat(stderr_file.fileno()).st_size
            if (
                stdout_size > SEVEN_ZIP_STDOUT_LIMIT_BYTES
                or stderr_size > SEVEN_ZIP_STDERR_LIMIT_BYTES
            ):
                raise ExternalInventoryError("7-Zip output exceeded its bounded size limit")
            return_code = process.returncode
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout_bytes = stdout_file.read(SEVEN_ZIP_STDOUT_LIMIT_BYTES + 1)
            stderr_bytes = stderr_file.read(SEVEN_ZIP_STDERR_LIMIT_BYTES + 1)
    except (OSError, subprocess.SubprocessError) as error:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        raise ExternalInventoryError("the bound 7-Zip command could not be executed") from error
    if (
        _stable_file_snapshot(executable, field="bound executable")
        != source_executable_snapshot
        or (
            use_two_file_replica
            and _stable_file_snapshot(source_library, field="bound 7-Zip library")
            != source_library_snapshot
        )
    ):
        raise ExternalInventoryError("bound 7-Zip code changed during execution")
    if return_code != 0:
        raise ExternalInventoryError("the bound 7-Zip command failed closed")
    try:
        stdout = stdout_bytes.decode("utf-8", errors="strict")
        stderr = stderr_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ExternalInventoryError("7-Zip output was not exact UTF-8") from error
    if stderr.strip() or re.search(r"(?im)^\s*Warnings?\s*:", stdout):
        raise ExternalInventoryError("7-Zip emitted a warning or unexpected error stream")
    return stdout


def _resolve_seven_zip_executable(requested_path: Path) -> tuple[Path, Path]:
    if not isinstance(requested_path, Path):
        raise TypeError("seven_zip_executable must be a pathlib.Path")
    candidate = requested_path
    if not candidate.is_file():
        located = shutil.which(os.fspath(requested_path))
        if located is None:
            raise ExternalInventoryError("7-Zip executable is unavailable")
        candidate = Path(located)
    if _is_indirect(candidate):
        raise ExternalInventoryError("7-Zip executable path must not be indirect")

    shim_file = candidate.with_suffix(".shim")
    if shim_file.is_file():
        if _is_indirect(shim_file) or shim_file.stat().st_size > 8_192:
            raise ExternalInventoryError("7-Zip Scoop shim metadata is unsafe")
        try:
            shim_text = shim_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise ExternalInventoryError("7-Zip Scoop shim metadata is unreadable") from error
        path_values: list[str] = []
        for raw_line in shim_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            matched = re.fullmatch(r'path\s*=\s*"([^"]+)"', line)
            if matched is None:
                raise ExternalInventoryError("7-Zip Scoop shim metadata is unsupported")
            path_values.append(matched.group(1))
        if len(path_values) != 1:
            raise ExternalInventoryError("7-Zip Scoop shim must bind exactly one real binary")
        candidate = Path(path_values[0])
    try:
        executable = candidate.resolve(strict=True)
    except OSError as error:
        raise ExternalInventoryError("resolved 7-Zip executable is unavailable") from error
    _assert_direct_ancestry(executable, field="resolved 7-Zip executable")
    if _is_indirect(executable) or not executable.is_file():
        raise ExternalInventoryError("resolved 7-Zip executable must be a regular file")
    library = executable.with_name("7z.dll")
    try:
        library = library.resolve(strict=True)
    except OSError as error:
        raise ExternalInventoryError("resolved 7-Zip runtime library is unavailable") from error
    _assert_direct_ancestry(library, field="resolved 7-Zip library")
    if _is_indirect(library) or not library.is_file():
        raise ExternalInventoryError("resolved 7-Zip runtime library must be a regular file")
    return executable, library


def _resolve_seven_zip_tool(
    requested_path: Path,
    *,
    runner: _SevenZipRunner = _run_seven_zip,
) -> _ResolvedSevenZipTool:
    executable, library = _resolve_seven_zip_executable(requested_path)
    executable_before = _stable_file_snapshot(
        executable, field="resolved 7-Zip executable"
    )
    library_before = _stable_file_snapshot(library, field="resolved 7-Zip library")
    version_output = runner(executable, ("i", "-sccUTF-8"))
    match = re.search(r"(?m)^7-Zip(?: \(z\))? ([0-9]+\.[0-9]{2})(?:\s|$)", version_output)
    if match is None:
        raise ExternalInventoryError("could not parse the exact 7-Zip version")
    executable_after = _stable_file_snapshot(
        executable, field="resolved 7-Zip executable"
    )
    library_after = _stable_file_snapshot(library, field="resolved 7-Zip library")
    if executable_before != executable_after or library_before != library_after:
        raise ExternalInventoryError("7-Zip binary or library changed during resolution")
    binding = SevenZipToolBinding(
        implementation="7zip",
        version=match.group(1),
        executable_name=executable.name,
        executable_size_bytes=executable_before.size_bytes,
        executable_sha256=executable_before.sha256,
        library_name=library.name,
        library_size_bytes=library_before.size_bytes,
        library_sha256=library_before.sha256,
    )
    return _ResolvedSevenZipTool(executable, library, binding)


def resolve_seven_zip_tool_binding(
    requested_path: Path,
    *,
    runner: _SevenZipRunner = _run_seven_zip,
) -> SevenZipToolBinding:
    """Resolve a shim and return only the path-free real 7-Zip identity."""

    return _resolve_seven_zip_tool(requested_path, runner=runner).binding


def verify_seven_zip_tool_binding(
    requested_path: Path,
    binding: SevenZipToolBinding,
    *,
    runner: _SevenZipRunner = _run_seven_zip,
) -> str:
    """Fail if the real executable, library, or reported version has changed."""

    if not isinstance(binding, SevenZipToolBinding):
        raise TypeError("binding must be SevenZipToolBinding")
    observed = resolve_seven_zip_tool_binding(requested_path, runner=runner)
    if observed != binding:
        raise ExternalInventoryError("the resolved 7-Zip tool binding has changed")
    return binding.tool_sha256


def parse_seven_zip_slt_listing(text: str) -> tuple[_SevenZipListedMember, ...]:
    """Parse the exact safe file/directory subset of ``7z l -slt`` output."""

    if not isinstance(text, str):
        raise TypeError("text must be str")
    blocks: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        if not line:
            if current:
                blocks.append(current)
                current = {}
            continue
        if re.fullmatch(r"-{2,}", line):
            if current:
                blocks.append(current)
                current = {}
            continue
        if " = " not in line:
            continue
        key, value = line.split(" = ", maxsplit=1)
        if not key or key in current:
            raise ExternalInventoryError("7-Zip listing contains a duplicate metadata key")
        current[key] = value
    if current:
        blocks.append(current)

    members: list[_SevenZipListedMember] = []
    seen: set[str] = set()
    seen_casefolded: set[str] = set()
    for block in blocks:
        if "Path" not in block:
            continue
        if "Size" not in block:
            if "Type" in block or "Physical Size" in block:
                continue
            raise ExternalInventoryError("7-Zip member listing omits Size")
        path = _canonical_seven_zip_slt_member_path(
            block["Path"], "7-Zip member path"
        )
        if path in seen or path.casefold() in seen_casefolded:
            raise ExternalInventoryError("7-Zip listing contains duplicate member paths")
        seen.add(path)
        seen_casefolded.add(path.casefold())
        raw_size = block["Size"]
        try:
            size = int(raw_size)
        except ValueError as error:
            raise ExternalInventoryError("7-Zip member size is not an integer") from error
        if str(size) != raw_size or size < 0:
            raise ExternalInventoryError("7-Zip member size is noncanonical")
        if any(
            key in block and block[key] not in {"", "-"}
            for key in ("Symbolic Link", "Hard Link", "Reparse Point")
        ):
            raise ExternalInventoryError("7-Zip listing contains a link member")
        attributes = block.get("Attributes", "")
        if attributes.lower().startswith("l"):
            raise ExternalInventoryError("7-Zip listing contains a link member")
        folder = block.get("Folder")
        if folder is not None and folder not in {"+", "-"}:
            raise ExternalInventoryError("7-Zip Folder metadata is unsupported")
        if folder is not None:
            is_directory = folder == "+"
        elif attributes:
            is_directory = attributes[0] in {"D", "d"}
        else:
            raise ExternalInventoryError("7-Zip member omits directory metadata")
        if block.get("Encrypted", "-") not in {"", "-"}:
            raise ExternalInventoryError("encrypted 7-Zip members are forbidden")
        raw_crc = block.get("CRC", "")
        if is_directory:
            if size != 0 or raw_crc not in {"", "00000000"}:
                raise ExternalInventoryError("7-Zip directory metadata is inconsistent")
            crc: str | None = None
        else:
            if len(raw_crc) != 8 or any(
                character not in "0123456789ABCDEF" for character in raw_crc
            ):
                raise ExternalInventoryError(
                    "7-Zip file CRC must be exact uppercase hexadecimal"
                )
            crc = raw_crc
        members.append(_SevenZipListedMember(path, size, crc, is_directory))
    if not members or not any(not member.is_directory for member in members):
        raise ExternalInventoryError("7-Zip listing contains no regular file members")
    return tuple(sorted(members, key=lambda member: member.path))


def _require_two_volume_listing(text: str) -> None:
    if re.search(r"(?m)^Multivolume = \+\r?$", text) is None or re.search(
        r"(?m)^Volumes = 2\r?$", text
    ) is None:
        raise ExternalInventoryError("7-Zip listing is not the exact two-part split archive")


def _verify_tool_unchanged(
    requested_path: Path,
    expected: _ResolvedSevenZipTool,
    *,
    runner: _SevenZipRunner,
) -> None:
    observed = _resolve_seven_zip_tool(requested_path, runner=runner)
    if observed.binding != expected.binding or observed.executable != expected.executable:
        raise ExternalInventoryError("7-Zip tool changed during archive closure")


def build_zzu_split_zip_extraction_closure(
    archive_z01_path: Path,
    archive_zip_path: Path,
    extraction_root: Path,
    seven_zip_executable: Path,
    *,
    expected_required_relative_paths: Sequence[str],
    archive_root_prefix: str = "Child_ecg",
    runner: _SevenZipRunner = _run_seven_zip,
    stage_callback: _ZZUArchiveClosureStageCallback | None = None,
) -> ArchiveExtractionClosure:
    """Test, isolate-extract, and SHA-close every ZZU split-ZIP member."""

    if stage_callback is not None and not callable(stage_callback):
        raise TypeError("stage_callback must be callable or None")
    if stage_callback is not None:
        stage_callback("zzu_tool_resolution")
    _assert_direct_ancestry(archive_z01_path, field="ZZU .z01 archive part")
    _assert_direct_ancestry(archive_zip_path, field="ZZU .zip archive part")
    _assert_direct_ancestry(extraction_root, field="ZZU extraction root")
    # The bound 7-Zip process uses a fresh isolated cwd, so archive operands
    # must be caller-independent before they cross that process boundary.
    try:
        archive_z01_path = archive_z01_path.resolve(strict=True)
        archive_zip_path = archive_zip_path.resolve(strict=True)
    except OSError as error:
        raise ExternalInventoryError("ZZU split archive part is unavailable") from error
    _assert_direct_ancestry(archive_z01_path, field="resolved ZZU .z01 archive part")
    _assert_direct_ancestry(archive_zip_path, field="resolved ZZU .zip archive part")
    prefix = _canonical_archive_leaf(archive_root_prefix, "archive_root_prefix")
    try:
        z01_parent = archive_z01_path.parent.resolve(strict=True)
        zip_parent = archive_zip_path.parent.resolve(strict=True)
    except OSError as error:
        raise ExternalInventoryError("ZZU split archive parent is unavailable") from error
    if z01_parent != zip_parent:
        raise ExternalInventoryError("ZZU split archive parts must share one directory")
    if (
        archive_z01_path.suffix != ".z01"
        or archive_zip_path.suffix != ".zip"
        or archive_z01_path.stem != archive_zip_path.stem
    ):
        raise ExternalInventoryError("ZZU split archive parts must be matching .z01/.zip")
    z01_before = _stable_file_snapshot(archive_z01_path, field="ZZU .z01 archive part")
    zip_before = _stable_file_snapshot(archive_zip_path, field="ZZU .zip archive part")
    tool = _resolve_seven_zip_tool(seven_zip_executable, runner=runner)

    if stage_callback is not None:
        stage_callback("zzu_archive_listing")
    listing_text = runner(
        tool.executable,
        ("l", "-slt", "-sccUTF-8", os.fspath(archive_zip_path)),
    )
    _require_two_volume_listing(listing_text)
    listed = parse_seven_zip_slt_listing(listing_text)
    listed_files: dict[str, _SevenZipListedMember] = {}
    listed_directories: set[str] = set()
    mapped_files: dict[str, _SevenZipListedMember] = {}
    mapped_directories: set[str] = set()
    for member in listed:
        relative = _mapped_archive_path(member.path, root_prefix=prefix)
        if member.is_directory:
            listed_directories.add(member.path)
            if relative is not None:
                mapped_directories.add(relative)
        else:
            if relative is None:
                raise ExternalInventoryError("ZZU split archive root must be a directory")
            _archive_member_role(ZZU_PEDIATRIC_DATASET, relative)
            listed_files[member.path] = member
            mapped_files[relative] = member

    if stage_callback is not None:
        stage_callback("zzu_archive_test")
    test_output = runner(
        tool.executable,
        ("t", "-bd", "-bb0", "-sccUTF-8", os.fspath(archive_zip_path)),
    )
    if "Everything is Ok" not in test_output:
        raise ExternalInventoryError("7-Zip archive test omitted its exact success marker")
    _verify_tool_unchanged(seven_zip_executable, tool, runner=runner)
    if (
        _stable_file_snapshot(archive_z01_path, field="ZZU .z01 archive part")
        != z01_before
        or _stable_file_snapshot(archive_zip_path, field="ZZU .zip archive part")
        != zip_before
    ):
        raise ExternalInventoryError("ZZU split archive changed before isolated extraction")

    if stage_callback is not None:
        stage_callback("zzu_evaluated_tree_snapshot")
    evaluated = _snapshot_extraction_tree(extraction_root)
    isolated_snapshots: Mapping[str, _FileSnapshot]
    if stage_callback is not None:
        stage_callback("zzu_isolated_extraction")
    with tempfile.TemporaryDirectory(prefix="ecg-trust-zzu-closure-") as raw_temp:
        temporary_root = Path(raw_temp).resolve(strict=True)
        _assert_direct_ancestry(temporary_root, field="isolated extraction root")
        extract_output = runner(
            tool.executable,
            (
                "x",
                "-y",
                "-bd",
                "-bb0",
                "-sccUTF-8",
                f"-o{temporary_root!s}",
                os.fspath(archive_zip_path),
            ),
        )
        if "Everything is Ok" not in extract_output:
            raise ExternalInventoryError(
                "7-Zip isolated extraction omitted its exact success marker"
            )
        isolated = _snapshot_extraction_tree(temporary_root)
        if set(isolated.files) != set(listed_files):
            raise ExternalInventoryError(
                "7-Zip listing and isolated extraction have extra or missing files"
            )
        expected_isolated_directories = listed_directories.union(
            _parent_directories(tuple(listed_files))
        )
        if isolated.directories != expected_isolated_directories:
            raise ExternalInventoryError(
                "7-Zip listing and isolated extraction have extra or missing directories"
            )
        isolated_snapshots = MappingProxyType(dict(isolated.files))

    if stage_callback is not None:
        stage_callback("zzu_archive_comparison")
    _validate_archive_and_extraction_sets(
        archive_file_paths=set(mapped_files),
        archive_directory_paths=mapped_directories,
        extraction=evaluated,
        expected_required_relative_paths=expected_required_relative_paths,
    )
    evaluated_after = _snapshot_extraction_tree(extraction_root)
    if evaluated_after != evaluated:
        raise ExternalInventoryError("ZZU evaluated extraction changed during closure")
    bindings: list[ArchiveMemberBinding] = []
    for relative_path, listed_member in mapped_files.items():
        isolated_snapshot = isolated_snapshots[listed_member.path]
        evaluated_snapshot = evaluated.files[relative_path]
        if listed_member.crc32 != isolated_snapshot.crc32:
            raise ExternalInventoryError("7-Zip central-directory CRC differs after extraction")
        if listed_member.size_bytes != isolated_snapshot.size_bytes:
            raise ExternalInventoryError("7-Zip central-directory size differs after extraction")
        if (
            isolated_snapshot.size_bytes != evaluated_snapshot.size_bytes
            or isolated_snapshot.sha256 != evaluated_snapshot.sha256
        ):
            raise ExternalInventoryError(
                "ZZU archive member bytes differ from the evaluated extraction"
            )
        bindings.append(
            ArchiveMemberBinding(
                archive_member_path=listed_member.path,
                extracted_relative_path=relative_path,
                role=_archive_member_role(ZZU_PEDIATRIC_DATASET, relative_path),
                size_bytes=isolated_snapshot.size_bytes,
                sha256=isolated_snapshot.sha256,
                archive_crc32=listed_member.crc32,
            )
        )

    _verify_tool_unchanged(seven_zip_executable, tool, runner=runner)
    if (
        _stable_file_snapshot(archive_z01_path, field="ZZU .z01 archive part")
        != z01_before
        or _stable_file_snapshot(archive_zip_path, field="ZZU .zip archive part")
        != zip_before
    ):
        raise ExternalInventoryError("ZZU split archive changed during closure construction")
    archive_files = (
        _archive_file_binding(archive_z01_path, z01_before),
        _archive_file_binding(archive_zip_path, zip_before),
    )
    return _build_archive_closure(
        dataset=ZZU_PEDIATRIC_DATASET,
        archive_format="split_zip_7zip",
        archive_root_prefix=prefix,
        archive_files=archive_files,
        members=bindings,
        tool_binding=tool.binding,
    )


def verify_zzu_split_zip_extraction_closure(
    archive_z01_path: Path,
    archive_zip_path: Path,
    extraction_root: Path,
    seven_zip_executable: Path,
    closure: ArchiveExtractionClosure,
    *,
    runner: _SevenZipRunner = _run_seven_zip,
    stage_callback: _ZZUArchiveClosureStageCallback | None = None,
) -> str:
    """Rebuild and exactly compare a frozen ZZU archive/tool closure."""

    if closure.dataset != ZZU_PEDIATRIC_DATASET:
        raise ExternalInventoryError("expected a ZZU archive closure")
    rebuilt = build_zzu_split_zip_extraction_closure(
        archive_z01_path,
        archive_zip_path,
        extraction_root,
        seven_zip_executable,
        expected_required_relative_paths=tuple(
            member.extracted_relative_path for member in closure.members
        ),
        archive_root_prefix=closure.archive_root_prefix,
        runner=runner,
        stage_callback=stage_callback,
    )
    if rebuilt != closure:
        raise ExternalInventoryError("ZZU archive closure no longer matches")
    return closure.closure_sha256


def resolve_inventory_record_base(
    dataset_root: Path,
    record: ExternalInventoryRecord,
) -> Path:
    """Resolve a record for a later adapter call without permitting path escape."""

    if not isinstance(record, ExternalInventoryRecord):
        raise TypeError("record must be ExternalInventoryRecord")
    base, _, _ = _safe_record_files(dataset_root, record.record_ref)
    return base


def _inventory_file_metadata(
    dataset_root: Path,
    record_ref: str,
) -> tuple[str, int, str, int, _HeaderMetadata]:
    base, header_path, data_path = _safe_record_files(dataset_root, record_ref)
    before_hashes = (sha256_file(header_path), sha256_file(data_path))
    before_sizes = (header_path.stat().st_size, data_path.stat().st_size)
    header = _read_header_metadata(base)
    after_hashes = (sha256_file(header_path), sha256_file(data_path))
    after_sizes = (header_path.stat().st_size, data_path.stat().st_size)
    rebound_base, rebound_header, rebound_data = _safe_record_files(
        dataset_root, record_ref
    )
    if (
        rebound_base != base
        or rebound_header != header_path
        or rebound_data != data_path
        or before_hashes != after_hashes
        or before_sizes != after_sizes
    ):
        raise ExternalInventoryError("raw WFDB files changed during inventory construction")
    return (
        before_hashes[0],
        before_sizes[0],
        before_hashes[1],
        before_sizes[1],
        header,
    )


@dataclass(frozen=True, slots=True)
class ZZUPediatricCandidate:
    """Header-only candidate metadata supplied before lockbox selection."""

    dataset_version: str
    site: str
    site_alias: str
    patient_key: str
    record_ref: str
    ecg_id: str
    declared_lead_count: int
    pediatric_12_lead: bool
    declared_sample_count: int | None = None

    def __post_init__(self) -> None:
        _require_text(self.dataset_version, "dataset_version")
        if self.dataset_version != ZZU_PEDIATRIC_VERSION:
            raise ExternalInventoryError("ZZU pediatric candidate dataset_version must be 1")
        _require_text(self.site, "site")
        _require_text(self.site_alias, "site_alias")
        _require_text(self.patient_key, "patient_key")
        record_ref = _canonical_record_ref(self.record_ref)
        ecg_id = _require_text(self.ecg_id, "ecg_id")
        if ecg_id != PurePosixPath(record_ref).name:
            raise ExternalInventoryError(
                "ZZU ecg_id must exactly equal the Filename record leaf"
            )
        patient_prefix = f"{ZZU_PEDIATRIC_DATASET}:"
        if not self.patient_key.startswith(patient_prefix):
            raise ExternalInventoryError("ZZU patient_key must use the dataset scope")
        raw_patient = self.patient_key.removeprefix(patient_prefix)
        _require_text(raw_patient, "ZZU raw patient identity")
        parts = PurePosixPath(record_ref).parts
        if (
            len(parts) != 3
            or parts[0] != raw_patient[:3]
            or parts[1] != raw_patient
            or not ecg_id.startswith(f"{raw_patient}_E")
        ):
            raise ExternalInventoryError(
                "ZZU Filename shard, patient directory, ECG_ID, and Patient_ID disagree"
            )
        lead_count = _require_positive_int(
            self.declared_lead_count, "declared_lead_count"
        )
        if lead_count not in {9, 12}:
            raise ExternalInventoryError("declared_lead_count must be exactly 9 or 12")
        if not isinstance(self.pediatric_12_lead, bool):
            raise ExternalInventoryError("pediatric_12_lead must be boolean")
        if self.pediatric_12_lead is not (lead_count == 12):
            raise ExternalInventoryError(
                "pediatric_12_lead must exactly match declared_lead_count"
            )
        if self.declared_sample_count is not None:
            _require_positive_int(self.declared_sample_count, "declared_sample_count")


def parse_zzu_pediatric_attributes_csv(
    text: str,
    *,
    site: str,
    site_alias: str,
    dataset_version: str = ZZU_PEDIATRIC_VERSION,
    record_column: str = "Filename",
    ecg_id_column: str = "ECG_ID",
    patient_column: str = "Patient_ID",
    lead_count_column: str = "Lead",
    sampling_point_column: str = "Sampling_point",
    delimiter: str = ",",
    expected_record_count: int | None = None,
    expected_patient_count: int | None = None,
) -> tuple[ZZUPediatricCandidate, ...]:
    """Parse the official ZZU identity/eligibility mapping without waveform access."""

    if not isinstance(text, str):
        raise TypeError("text must be str")
    _require_text(site, "site")
    _require_text(site_alias, "site_alias")
    if dataset_version != ZZU_PEDIATRIC_VERSION:
        raise ExternalInventoryError("ZZU metadata dataset_version must be 1")
    columns = (
        record_column,
        ecg_id_column,
        patient_column,
        lead_count_column,
        sampling_point_column,
    )
    if len(set(columns)) != len(columns):
        raise ExternalInventoryError("ZZU metadata column names must be distinct")
    for column in columns:
        _require_text(column, "ZZU metadata column")
    if delimiter not in {",", "\t", ";"}:
        raise ExternalInventoryError("ZZU metadata delimiter is unsupported")
    if expected_record_count is not None:
        _require_positive_int(expected_record_count, "expected_record_count")
    if expected_patient_count is not None:
        _require_positive_int(expected_patient_count, "expected_patient_count")

    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
    try:
        header = next(reader)
    except StopIteration as error:
        raise ExternalInventoryError("ZZU metadata has no header") from error
    except csv.Error as error:
        raise ExternalInventoryError("ZZU metadata header is malformed") from error
    if (
        not header
        or len(header) != len(set(header))
        or any(not value or value != value.strip() for value in header)
    ):
        raise ExternalInventoryError("ZZU metadata header is noncanonical or duplicated")
    missing = sorted(set(columns).difference(header))
    if missing:
        raise ExternalInventoryError(f"ZZU metadata is missing columns {missing!r}")
    positions = {column: header.index(column) for column in columns}

    candidates: list[ZZUPediatricCandidate] = []
    try:
        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(header) or not any(row):
                raise ExternalInventoryError(
                    f"ZZU metadata row {row_number} is malformed or blank"
                )
            values = {
                column: _require_text(row[positions[column]], f"{column} row {row_number}")
                for column in columns
            }
            raw_ref = values[record_column]
            record_ref = _canonical_record_ref(raw_ref)
            ecg_id = values[ecg_id_column]
            if ecg_id != PurePosixPath(record_ref).name:
                raise ExternalInventoryError(
                    f"ZZU metadata row {row_number} ECG_ID does not match Filename"
                )
            raw_lead_count = values[lead_count_column]
            raw_sample_count = values[sampling_point_column]
            try:
                lead_count = int(raw_lead_count)
                sample_count = int(raw_sample_count)
            except ValueError as error:
                raise ExternalInventoryError(
                    f"ZZU metadata row {row_number} has a non-integer count"
                ) from error
            if str(lead_count) != raw_lead_count or lead_count not in {9, 12}:
                raise ExternalInventoryError(
                    f"ZZU metadata row {row_number} lead count must be exactly 9 or 12"
                )
            if str(sample_count) != raw_sample_count or sample_count < 1:
                raise ExternalInventoryError(
                    f"ZZU metadata row {row_number} sample count must be canonical and positive"
                )
            candidates.append(
                ZZUPediatricCandidate(
                    dataset_version=dataset_version,
                    site=site,
                    site_alias=site_alias,
                    patient_key=(
                        f"{ZZU_PEDIATRIC_DATASET}:{values[patient_column]}"
                    ),
                    record_ref=record_ref,
                    ecg_id=ecg_id,
                    declared_lead_count=lead_count,
                    pediatric_12_lead=lead_count == 12,
                    declared_sample_count=sample_count,
                )
            )
    except csv.Error as error:
        raise ExternalInventoryError("ZZU metadata CSV is malformed") from error
    if not candidates:
        raise ExternalInventoryError("ZZU metadata contains no records")
    record_refs = [candidate.record_ref for candidate in candidates]
    ecg_ids = [candidate.ecg_id for candidate in candidates]
    if len(record_refs) != len(set(record_refs)):
        raise ExternalInventoryError("ZZU metadata contains duplicate Filename identities")
    if len(ecg_ids) != len(set(ecg_ids)):
        raise ExternalInventoryError("ZZU metadata contains duplicate ECG_ID identities")
    if expected_record_count is not None and len(candidates) != expected_record_count:
        raise ExternalInventoryError("ZZU metadata record count differs from expectation")
    patient_count = len({candidate.patient_key for candidate in candidates})
    if expected_patient_count is not None and patient_count != expected_patient_count:
        raise ExternalInventoryError("ZZU metadata patient count differs from expectation")
    return tuple(candidates)


def _build_summary_body(
    *,
    candidate_record_count: int,
    selected_record_count: int,
    exclusion_counts: Mapping[str, int],
) -> dict[str, object]:
    return {
        "candidate_record_count": candidate_record_count,
        "dataset": ZZU_PEDIATRIC_DATASET,
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "selected_record_count": selected_record_count,
    }


def _build_summary_sha256(
    *,
    candidate_record_count: int,
    selected_record_count: int,
    exclusion_counts: Mapping[str, int],
) -> str:
    body = _build_summary_body(
        candidate_record_count=candidate_record_count,
        selected_record_count=selected_record_count,
        exclusion_counts=exclusion_counts,
    )
    return "sha256:" + hashlib.sha256(
        BUILD_SUMMARY_DOMAIN + _canonical_json_bytes(body)
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ExternalInventoryBuildSummary:
    """Aggregate accounting proving ineligible metadata never entered selection."""

    candidate_record_count: int
    selected_record_count: int
    exclusion_counts: Mapping[str, int]
    summary_sha256: str

    def __post_init__(self) -> None:
        candidates = _require_positive_int(
            self.candidate_record_count, "candidate_record_count"
        )
        if (
            isinstance(self.selected_record_count, bool)
            or not isinstance(self.selected_record_count, int)
            or self.selected_record_count < 0
        ):
            raise ExternalInventoryError("selected_record_count must be a non-negative integer")
        normalized = dict(self.exclusion_counts)
        if set(normalized) != set(_EXCLUSION_REASONS):
            raise ExternalInventoryError("exclusion_counts must contain every frozen reason")
        for reason, count in normalized.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ExternalInventoryError(
                    f"exclusion count for {reason!r} must be a non-negative integer"
                )
        if self.selected_record_count + sum(normalized.values()) != candidates:
            raise ExternalInventoryError(
                "selected and excluded record counts must equal candidate_record_count"
            )
        expected_hash = _build_summary_sha256(
            candidate_record_count=candidates,
            selected_record_count=self.selected_record_count,
            exclusion_counts=normalized,
        )
        if self.summary_sha256 != expected_hash:
            raise ExternalInventoryError("inventory build summary self-hash does not match")
        object.__setattr__(self, "exclusion_counts", MappingProxyType(normalized))

    @property
    def excluded_record_count(self) -> int:
        return sum(self.exclusion_counts.values())

    def to_dict(self) -> dict[str, object]:
        body = _build_summary_body(
            candidate_record_count=self.candidate_record_count,
            selected_record_count=self.selected_record_count,
            exclusion_counts=self.exclusion_counts,
        )
        body["summary_sha256"] = self.summary_sha256
        return body


def _zzu_exclusion_reason(
    candidate: ZZUPediatricCandidate,
    metadata: _HeaderMetadata,
) -> InventoryExclusionReason | None:
    if candidate.pediatric_12_lead is not True:
        return "pediatric_12_lead_flag_false"
    if not math.isclose(
        metadata.sampling_frequency_hz,
        500.0,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        return "sampling_frequency_not_500_hz"
    if metadata.duration_seconds < 10.0:
        return "duration_under_10_seconds"
    if metadata.lead_count != len(LEADS):
        return "lead_count_not_12"
    try:
        _canonicalize_ordered_leads(
            metadata.raw_ordered_leads,
            allowed_aliases=SOURCE_LEAD_ALIASES,
        )
    except ExternalInventoryError:
        return "noncanonical_lead_set"
    return None


def select_zzu_pediatric_inventory_records(
    dataset_root: Path,
    candidates: Sequence[ZZUPediatricCandidate],
) -> tuple[tuple[ExternalInventoryRecord, ...], ExternalInventoryBuildSummary]:
    """Select eligible pediatric records using headers only and count exclusions.

    The returned selected rows are byte-hashed private inventory records.  Short,
    non-500-Hz, non-12-lead, or metadata-flagged candidates are represented only
    in the aggregate build summary and cannot enter the selected inventory.
    """

    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise TypeError("candidates must be a sequence of ZZUPediatricCandidate values")
    if not candidates:
        raise ExternalInventoryError("candidates must not be empty")
    normalized: list[ZZUPediatricCandidate] = []
    seen_refs: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, ZZUPediatricCandidate):
            raise TypeError("candidates must contain ZZUPediatricCandidate values")
        if candidate.record_ref in seen_refs:
            raise ExternalInventoryError("ZZU candidate record_ref values must be unique")
        seen_refs.add(candidate.record_ref)
        normalized.append(candidate)

    verify_wfdb_candidate_file_set(
        dataset_root,
        tuple(candidate.record_ref for candidate in normalized),
    )

    selected: list[ExternalInventoryRecord] = []
    counts: Counter[str] = Counter()
    for candidate in sorted(normalized, key=lambda value: value.record_ref):
        base, _, _ = _safe_record_files(dataset_root, candidate.record_ref)
        metadata = _read_header_metadata(base)
        _safe_record_files(dataset_root, candidate.record_ref)
        if (
            candidate.declared_sample_count is not None
            and candidate.declared_sample_count != metadata.sample_count
        ):
            raise ExternalInventoryError(
                "ZZU declared sample count differs from the raw WFDB header"
            )
        if candidate.declared_lead_count != metadata.lead_count:
            raise ExternalInventoryError(
                "ZZU declared lead count differs from the raw WFDB header"
            )
        reason = _zzu_exclusion_reason(candidate, metadata)
        if reason is not None:
            counts[reason] += 1
            continue
        selected.append(
            inventory_zzu_pediatric_record(
                dataset_root,
                dataset_version=candidate.dataset_version,
                site=candidate.site,
                site_alias=candidate.site_alias,
                patient_key=candidate.patient_key,
                record_ref=candidate.record_ref,
                pediatric_12_lead=candidate.pediatric_12_lead,
            )
        )
    exclusion_counts: dict[str, int] = {
        reason: counts.get(reason, 0) for reason in _EXCLUSION_REASONS
    }
    summary_hash = _build_summary_sha256(
        candidate_record_count=len(normalized),
        selected_record_count=len(selected),
        exclusion_counts=exclusion_counts,
    )
    summary = ExternalInventoryBuildSummary(
        candidate_record_count=len(normalized),
        selected_record_count=len(selected),
        exclusion_counts=exclusion_counts,
        summary_sha256=summary_hash,
    )
    return tuple(selected), summary


def inventory_challenge_2011_record(
    dataset_root: Path,
    *,
    dataset_version: str,
    site: str,
    site_alias: str,
    record_ref: str,
    quality_label: ChallengeQualityLabel,
    patient_key: str | None = None,
    source_role: str = CONFIRMATION_LOCKBOX_ROLE,
) -> ExternalInventoryRecord:
    """Create one Challenge 2011 Set A inventory row without reading amplitudes."""

    canonical_ref = _canonical_record_ref(record_ref)
    header_hash, header_size, data_hash, data_size, metadata = _inventory_file_metadata(
        dataset_root, canonical_ref
    )
    if patient_key is not None:
        raise ExternalInventoryError(
            "Challenge 2011 patient_key must be null because patient identity is unavailable"
        )
    raw_ordered_leads, canonical_ordered_leads = _selected_lead_orders(
        metadata,
        allowed_aliases={},
    )
    if not math.isclose(metadata.duration_seconds, 10.0, rel_tol=0.0, abs_tol=0.0):
        raise ExternalInventoryError("Challenge 2011 Set A records must be exactly ten seconds")
    return ExternalInventoryRecord(
        dataset=CHALLENGE_2011_DATASET,
        dataset_version=dataset_version,
        site=site,
        site_alias=site_alias,
        patient_key=patient_key,
        record_ref=canonical_ref,
        source_role=source_role,
        raw_header_sha256=header_hash,
        raw_header_size_bytes=header_size,
        raw_data_sha256=data_hash,
        raw_data_size_bytes=data_size,
        sampling_frequency_hz=metadata.sampling_frequency_hz,
        source_sample_count=metadata.sample_count,
        duration_seconds=metadata.duration_seconds,
        lead_count=metadata.lead_count,
        raw_ordered_leads=raw_ordered_leads,
        canonical_ordered_leads=canonical_ordered_leads,
        raw_data_file_names=metadata.raw_data_file_names,
        raw_physical_units=metadata.raw_physical_units,
        challenge_quality_label=quality_label,
        pediatric_12_lead=None,
    )


def inventory_zzu_pediatric_record(
    dataset_root: Path,
    *,
    dataset_version: str,
    site: str,
    site_alias: str,
    patient_key: str,
    record_ref: str,
    pediatric_12_lead: bool,
    source_role: str = CONFIRMATION_LOCKBOX_ROLE,
) -> ExternalInventoryRecord:
    """Create one ZZU pediatric inventory row without reading amplitudes."""

    canonical_ref = _canonical_record_ref(record_ref)
    header_hash, header_size, data_hash, data_size, metadata = _inventory_file_metadata(
        dataset_root, canonical_ref
    )
    if pediatric_12_lead is not True:
        raise ExternalInventoryError(
            "selected ZZU records require pediatric_12_lead metadata equal to true"
        )
    raw_ordered_leads, canonical_ordered_leads = _selected_lead_orders(
        metadata,
        allowed_aliases=SOURCE_LEAD_ALIASES,
    )
    return ExternalInventoryRecord(
        dataset=ZZU_PEDIATRIC_DATASET,
        dataset_version=dataset_version,
        site=site,
        site_alias=site_alias,
        patient_key=patient_key,
        record_ref=canonical_ref,
        source_role=source_role,
        raw_header_sha256=header_hash,
        raw_header_size_bytes=header_size,
        raw_data_sha256=data_hash,
        raw_data_size_bytes=data_size,
        sampling_frequency_hz=metadata.sampling_frequency_hz,
        source_sample_count=metadata.sample_count,
        duration_seconds=metadata.duration_seconds,
        lead_count=metadata.lead_count,
        raw_ordered_leads=raw_ordered_leads,
        canonical_ordered_leads=canonical_ordered_leads,
        raw_data_file_names=metadata.raw_data_file_names,
        raw_physical_units=metadata.raw_physical_units,
        challenge_quality_label=None,
        pediatric_12_lead=pediatric_12_lead,
    )


def verify_external_inventory(
    dataset_root: Path,
    inventory: ExternalWaveformInventory,
) -> str:
    """Re-hash raw files and headers and return the verified inventory identity."""

    if not isinstance(inventory, ExternalWaveformInventory):
        raise TypeError("inventory must be ExternalWaveformInventory")
    for record in inventory.records:
        header_hash, header_size, data_hash, data_size, metadata = _inventory_file_metadata(
            dataset_root, record.record_ref
        )
        allowed_aliases: Mapping[str, str] = (
            {} if record.dataset == CHALLENGE_2011_DATASET else SOURCE_LEAD_ALIASES
        )
        observed_raw_leads, observed_canonical_leads = _selected_lead_orders(
            metadata,
            allowed_aliases=allowed_aliases,
        )
        observed = (
            header_hash,
            header_size,
            data_hash,
            data_size,
            metadata.sampling_frequency_hz,
            metadata.sample_count,
            metadata.duration_seconds,
            metadata.lead_count,
            observed_raw_leads,
            observed_canonical_leads,
            metadata.raw_data_file_names,
            metadata.raw_physical_units,
        )
        expected = (
            record.raw_header_sha256,
            record.raw_header_size_bytes,
            record.raw_data_sha256,
            record.raw_data_size_bytes,
            record.sampling_frequency_hz,
            record.source_sample_count,
            record.duration_seconds,
            record.lead_count,
            record.raw_ordered_leads,
            record.canonical_ordered_leads,
            record.raw_data_file_names,
            record.raw_physical_units,
        )
        if observed != expected:
            raise ExternalInventoryError(
                f"raw WFDB binding no longer matches inventory record {record.record_ref!r}"
            )
    return inventory.inventory_sha256


def _record_lines(text: str, field: str) -> tuple[str, ...]:
    if not isinstance(text, str):
        raise TypeError(f"{field} must be text")
    records: list[str] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            record_ref = _canonical_record_ref(line)
        except ExternalInventoryError as error:
            raise ExternalInventoryError(
                f"{field} line {line_number} is not a safe record reference"
            ) from error
        if record_ref in seen:
            raise ExternalInventoryError(f"{field} contains duplicate record {record_ref!r}")
        seen.add(record_ref)
        records.append(record_ref)
    if not records:
        raise ExternalInventoryError(f"{field} must contain at least one record")
    return tuple(records)


def parse_challenge_2011_quality_lists(
    all_records_text: str,
    acceptable_records_text: str,
    unacceptable_records_text: str,
    *,
    expected_record_count: int | None = 1_000,
) -> Mapping[str, ChallengeQualityLabel]:
    """Bind official record lists to acceptable/unacceptable/indeterminate labels."""

    all_records = _record_lines(all_records_text, "all_records_text")
    if expected_record_count is not None:
        expected = _require_positive_int(expected_record_count, "expected_record_count")
        if len(all_records) != expected:
            raise ExternalInventoryError(
                f"Challenge 2011 Set A must contain exactly {expected} records"
            )
    acceptable = set(_record_lines(acceptable_records_text, "acceptable_records_text"))
    unacceptable = set(_record_lines(unacceptable_records_text, "unacceptable_records_text"))
    overlap = acceptable.intersection(unacceptable)
    if overlap:
        raise ExternalInventoryError(
            f"Challenge quality lists overlap at {sorted(overlap)[0]!r}"
        )
    universe = set(all_records)
    unknown = acceptable.union(unacceptable).difference(universe)
    if unknown:
        raise ExternalInventoryError(
            f"Challenge quality list references unknown record {sorted(unknown)[0]!r}"
        )
    result: dict[str, ChallengeQualityLabel] = {}
    for record_ref in sorted(all_records):
        if record_ref in acceptable:
            result[record_ref] = "acceptable"
        elif record_ref in unacceptable:
            result[record_ref] = "unacceptable"
        else:
            result[record_ref] = "indeterminate"
    return MappingProxyType(result)


def validate_challenge_2011_set_a_inventory(
    inventory: ExternalWaveformInventory,
    *,
    expected_record_count: int = 1_000,
    expected_quality_by_record: Mapping[str, ChallengeQualityLabel] | None = None,
) -> str:
    """Prove Set A is complete and every selected record has a frozen label."""

    if not isinstance(inventory, ExternalWaveformInventory):
        raise TypeError("inventory must be ExternalWaveformInventory")
    expected = _require_positive_int(expected_record_count, "expected_record_count")
    if any(record.dataset != CHALLENGE_2011_DATASET for record in inventory.records):
        raise ExternalInventoryError("Set A inventory must contain only Challenge 2011 records")
    if inventory.record_count != expected:
        raise ExternalInventoryError(
            f"Challenge 2011 Set A inventory must contain exactly {expected} records"
        )
    if any(record.challenge_quality_label not in _QUALITY_LABELS for record in inventory.records):
        raise ExternalInventoryError("every Challenge 2011 record must have a quality label")
    if expected_quality_by_record is not None:
        observed = {
            record.record_ref: record.challenge_quality_label for record in inventory.records
        }
        normalized_expected = dict(expected_quality_by_record)
        if observed != normalized_expected:
            raise ExternalInventoryError(
                "Challenge inventory record identities or quality labels differ from the "
                "frozen list"
            )
    return inventory.inventory_sha256


def external_inventory_public_projection(
    inventory: ExternalWaveformInventory,
) -> dict[str, object]:
    """Return aggregate-only publication data with no record or patient identifiers."""

    if not isinstance(inventory, ExternalWaveformInventory):
        raise TypeError("inventory must be ExternalWaveformInventory")
    grouped: dict[tuple[str, str, str, str], list[ExternalInventoryRecord]] = {}
    for record in inventory.records:
        key = (
            record.dataset,
            record.dataset_version,
            record.site_alias,
            record.source_role,
        )
        grouped.setdefault(key, []).append(record)

    groups: list[dict[str, object]] = []
    for key in sorted(grouped):
        dataset, dataset_version, site_alias, source_role = key
        records = grouped[key]
        known_patients = {
            record.patient_key for record in records if record.patient_key is not None
        }
        quality_counts = Counter(
            record.challenge_quality_label
            for record in records
            if record.challenge_quality_label is not None
        )
        pediatric_counts = Counter(
            str(record.pediatric_12_lead).lower()
            for record in records
            if record.pediatric_12_lead is not None
        )
        lead_counts = Counter(record.lead_count for record in records)
        group: dict[str, object] = {
            "challenge_quality_counts": {
                label: quality_counts.get(label, 0) for label in _QUALITY_LABELS
            },
            "dataset": dataset,
            "dataset_version": dataset_version,
            "duration_seconds_max": max(record.duration_seconds for record in records),
            "duration_seconds_min": min(record.duration_seconds for record in records),
            "known_patient_count": len(known_patients),
            "lead_count_distribution": {
                str(count): occurrences for count, occurrences in sorted(lead_counts.items())
            },
            "missing_patient_key_records": sum(
                record.patient_key is None for record in records
            ),
            "pediatric_12_lead_counts": {
                "false": pediatric_counts.get("false", 0),
                "true": pediatric_counts.get("true", 0),
            },
            "raw_bytes_total": sum(
                record.raw_header_size_bytes + record.raw_data_size_bytes for record in records
            ),
            "record_count": len(records),
            "sampling_frequencies_hz": sorted(
                {record.sampling_frequency_hz for record in records}
            ),
            "site_alias": site_alias,
            "source_role": source_role,
        }
        groups.append(group)

    archive_summaries: list[dict[str, object]] = []
    for closure in inventory.archive_closures:
        role_counts = Counter(member.role for member in closure.members)
        archive_summaries.append(
            {
                "archive_bytes_total": closure.archive_bytes_total,
                "archive_file_count": len(closure.archive_files),
                "archive_format": closure.archive_format,
                "closure_sha256": closure.closure_sha256,
                "dataset": closure.dataset,
                "member_bytes_total": closure.member_bytes_total,
                "member_count": closure.member_count,
                "member_role_counts": {
                    role: role_counts.get(role, 0)
                    for role in (
                        "ignored_release_file",
                        "quality_reference",
                        "wfdb_data",
                        "wfdb_header",
                    )
                },
                "tool_binding": (
                    None
                    if closure.tool_binding is None
                    else closure.tool_binding.to_dict()
                ),
            }
        )

    body: dict[str, object] = {
        "archive_closures": archive_summaries,
        "group_count": len(groups),
        "groups": groups,
        "kind": PUBLIC_PROJECTION_KIND,
        "record_count": inventory.record_count,
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "source_inventory_sha256": inventory.inventory_sha256,
    }
    digest = hashlib.sha256(PUBLIC_PROJECTION_DOMAIN + _canonical_json_bytes(body)).hexdigest()
    body["projection_sha256"] = f"sha256:{digest}"
    return body


__all__ = [
    "ARCHIVE_CLOSURE_DOMAIN",
    "ARCHIVE_CLOSURE_KIND",
    "ARCHIVE_CLOSURE_SCHEMA_VERSION",
    "BUILD_SUMMARY_DOMAIN",
    "CHALLENGE_2011_DATASET",
    "CHALLENGE_2011_VERSION",
    "CONFIRMATION_LOCKBOX_ROLE",
    "INVENTORY_DOMAIN",
    "INVENTORY_KIND",
    "INVENTORY_SCHEMA_VERSION",
    "PUBLIC_PROJECTION_DOMAIN",
    "PUBLIC_PROJECTION_KIND",
    "SEVEN_ZIP_TOOL_DOMAIN",
    "SEVEN_ZIP_TOOL_KIND",
    "SEVEN_ZIP_TOOL_SCHEMA_VERSION",
    "SOURCE_LEAD_ALIASES",
    "ZZU_PEDIATRIC_DATASET",
    "ZZU_PEDIATRIC_VERSION",
    "ArchiveExtractionClosure",
    "ArchiveFileBinding",
    "ArchiveFormat",
    "ArchiveMemberBinding",
    "ArchiveMemberRole",
    "ChallengeQualityLabel",
    "ExternalInventoryBuildSummary",
    "ExternalInventoryError",
    "ExternalInventoryRecord",
    "ExternalWaveformInventory",
    "InventoryExclusionReason",
    "SevenZipToolBinding",
    "ZZUPediatricCandidate",
    "build_external_inventory",
    "build_challenge_tar_extraction_closure",
    "build_zzu_split_zip_extraction_closure",
    "enumerate_wfdb_record_refs",
    "external_inventory_public_projection",
    "inventory_challenge_2011_record",
    "inventory_zzu_pediatric_record",
    "load_external_inventory",
    "parse_challenge_2011_quality_lists",
    "parse_seven_zip_slt_listing",
    "parse_zzu_pediatric_attributes_csv",
    "resolve_inventory_record_base",
    "save_external_inventory",
    "select_zzu_pediatric_inventory_records",
    "resolve_seven_zip_tool_binding",
    "validate_challenge_2011_set_a_inventory",
    "verify_external_inventory",
    "verify_challenge_tar_extraction_closure",
    "verify_seven_zip_tool_binding",
    "verify_wfdb_candidate_file_set",
    "verify_zzu_split_zip_extraction_closure",
]
