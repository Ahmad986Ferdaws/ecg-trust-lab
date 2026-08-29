"""Official waveform checksum-subset binding for OOD completion v1."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from ecg_trust.data.manifest import ManifestError, verify_sha256sums
from ecg_trust.ood_completion.cohorts import CohortRecord

CHECKSUM_SUBSET_ALGORITHM = "official_checksum_subset_v1"
CHECKSUM_SUBSET_DOMAIN = b"ecg_trust.official_checksum_subset.v1\x00"


class OODWaveformIntegrityError(ValueError):
    """Raised when the selected raw waveform inventory is incomplete or altered."""


@dataclass(frozen=True, slots=True)
class OfficialWaveformSubset:
    """Selected official path/digest pairs without waveform contents."""

    record_count: int
    relative_paths: tuple[str, ...]
    official_sha256_by_path: Mapping[str, str]
    subset_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "official_sha256_by_path",
            MappingProxyType(dict(self.official_sha256_by_path)),
        )

    @property
    def file_count(self) -> int:
        return len(self.relative_paths)


def build_official_waveform_subset(
    records: Sequence[CohortRecord],
    *,
    official_checksums: Mapping[str, str],
) -> OfficialWaveformSubset:
    """Bind each selected record to exactly one official DAT/HEA pair."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence) or not records:
        raise OODWaveformIntegrityError("waveform subset requires cohort records")
    if not isinstance(official_checksums, Mapping) or not official_checksums:
        raise OODWaveformIntegrityError("official checksum inventory must not be empty")
    ecg_ids: set[int] = set()
    selected: dict[str, str] = {}
    for record in records:
        if not isinstance(record, CohortRecord):
            raise TypeError("records must contain CohortRecord values")
        if record.ecg_id in ecg_ids:
            raise OODWaveformIntegrityError("selected waveform ECG identities are duplicated")
        ecg_ids.add(record.ecg_id)
        base = PurePosixPath(record.record_path)
        if base.suffix.casefold() in {".dat", ".hea"}:
            raise OODWaveformIntegrityError("manifest record_path must be suffix-free")
        for extension in (".dat", ".hea"):
            relative_path = base.with_suffix(extension).as_posix()
            digest = official_checksums.get(relative_path)
            if digest is None:
                raise OODWaveformIntegrityError(
                    "selected waveform file is absent from the official inventory"
                )
            normalized_digest = _unprefixed_sha256(digest)
            if relative_path in selected:
                raise OODWaveformIntegrityError("selected waveform file occurs more than once")
            selected[relative_path] = normalized_digest
    expected_files = len(records) * 2
    if len(selected) != expected_files:
        raise OODWaveformIntegrityError("selected waveform subset is not one DAT/HEA pair per ECG")
    ordered = dict(sorted(selected.items(), key=lambda item: item[0].encode("utf-8")))
    return OfficialWaveformSubset(
        record_count=len(records),
        relative_paths=tuple(ordered),
        official_sha256_by_path=ordered,
        subset_sha256=official_checksum_subset_sha256(tuple(ordered.items())),
    )


def official_checksum_subset_sha256(pairs: Sequence[tuple[str, str]]) -> str:
    """Apply the exact domain-separated canonical encoding from the protocol."""

    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence) or not pairs:
        raise OODWaveformIntegrityError("checksum subset requires path/digest pairs")
    normalized: dict[str, str] = {}
    for pair in pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise OODWaveformIntegrityError("checksum subset entries must be two-item tuples")
        relative_path, raw_digest = pair
        if not isinstance(relative_path, str) or not relative_path:
            raise OODWaveformIntegrityError("checksum subset path must be non-empty text")
        path = PurePosixPath(relative_path)
        if path.is_absolute() or path.as_posix() != relative_path or ".." in path.parts:
            raise OODWaveformIntegrityError("checksum subset path must be normalized and relative")
        if relative_path in normalized:
            raise OODWaveformIntegrityError("checksum subset paths must be unique")
        normalized[relative_path] = _unprefixed_sha256(raw_digest)
    files = [
        {"relative_path": path, "sha256": digest}
        for path, digest in sorted(normalized.items(), key=lambda item: item[0].encode("utf-8"))
    ]
    payload = {
        "algorithm": CHECKSUM_SUBSET_ALGORITHM,
        "files": files,
        "schema_version": 1,
    }
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(CHECKSUM_SUBSET_DOMAIN + canonical).hexdigest()


def verify_official_waveform_subset(
    dataset_root: str | Path,
    subset: OfficialWaveformSubset,
) -> str:
    """Hash every selected raw file and return the verified subset identity."""

    if not isinstance(subset, OfficialWaveformSubset):
        raise TypeError("subset must be an OfficialWaveformSubset")
    root = Path(dataset_root).resolve(strict=True)
    try:
        observed = verify_sha256sums(
            root,
            subset.official_sha256_by_path,
            subset.relative_paths,
        )
    except (OSError, ManifestError) as error:
        raise OODWaveformIntegrityError(
            "selected waveform bytes do not match the official inventory"
        ) from error
    observed_hash = official_checksum_subset_sha256(tuple(observed.items()))
    if observed_hash != subset.subset_sha256:
        raise OODWaveformIntegrityError("verified waveform subset identity changed")
    return observed_hash


def _unprefixed_sha256(value: object) -> str:
    if not isinstance(value, str):
        raise OODWaveformIntegrityError("official digest must be lowercase SHA-256 text")
    digest = value[7:] if value.startswith("sha256:") else value
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise OODWaveformIntegrityError("official digest must be lowercase SHA-256 text")
    return digest


__all__ = [
    "CHECKSUM_SUBSET_ALGORITHM",
    "CHECKSUM_SUBSET_DOMAIN",
    "OODWaveformIntegrityError",
    "OfficialWaveformSubset",
    "build_official_waveform_subset",
    "official_checksum_subset_sha256",
    "verify_official_waveform_subset",
]
