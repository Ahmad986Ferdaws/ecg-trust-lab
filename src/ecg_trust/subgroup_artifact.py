"""Deterministic, label-free PTB-XL subgroup metadata for final evaluation.

The final-test subgroup definitions are frozen from demographic metadata before
any fold-10 prediction is opened.  The artifact is self-hashed, binds the exact
source manifest, and deliberately reads no diagnostic target columns.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, cast

import pandas as pd  # type: ignore[import-untyped]

from ecg_trust.protocol import FINAL_TEST_FOLDS, ExperimentProtocol

SUBGROUP_SCHEMA_VERSION = 1
SUBGROUP_ARTIFACT_TYPE = "ecg_trust.ptbxl_subgroup_metadata"
SUBGROUP_ATTRIBUTES: tuple[str, ...] = ("sex", "age_band")
SEX_GROUPS: tuple[str, ...] = ("male", "female", "unknown")
AGE_GROUPS: tuple[str, ...] = ("<40", "40-59", "60-79", "80+", "unknown")
_METADATA_COLUMNS: tuple[str, ...] = (
    "ecg_id",
    "patient_id",
    "strat_fold",
    "age",
    "sex",
)
_DEFINITIONS: dict[str, object] = {
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


class SubgroupArtifactError(ValueError):
    """Raised when subgroup metadata violates the frozen final-test contract."""


class SubgroupIntegrityError(SubgroupArtifactError):
    """Raised when a stored subgroup artifact or its source has changed."""


@dataclass(frozen=True, slots=True)
class SubgroupArtifact:
    """Validated subgroup rows aligned to the canonical final-test fold."""

    dataset_name: str
    dataset_version: str
    protocol_hash: str
    manifest_path: Path
    manifest_sha256: str
    ecg_id: tuple[int, ...]
    patient_id: tuple[int, ...]
    sex: tuple[str, ...]
    age_band: tuple[str, ...]
    group_counts: tuple[Mapping[str, object], ...]
    artifact_sha256: str | None = None

    @property
    def record_count(self) -> int:
        return len(self.ecg_id)

    @property
    def patient_count(self) -> int:
        return len(set(self.patient_id))

    def to_payload(self, *, include_integrity: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": SUBGROUP_SCHEMA_VERSION,
            "artifact_type": SUBGROUP_ARTIFACT_TYPE,
            "dataset": {
                "name": self.dataset_name,
                "version": self.dataset_version,
            },
            "protocol_hash": self.protocol_hash,
            "source_manifest": {
                "path": str(self.manifest_path),
                "sha256": self.manifest_sha256,
                "columns_read": list(_METADATA_COLUMNS),
                "diagnostic_target_columns_read": False,
            },
            "folds": list(FINAL_TEST_FOLDS),
            "definitions": _definitions_payload(),
            "ecg_id": list(self.ecg_id),
            "patient_id": list(self.patient_id),
            "attributes": {
                "sex": list(self.sex),
                "age_band": list(self.age_band),
            },
            "summary": {
                "record_count": self.record_count,
                "patient_count": self.patient_count,
                "groups": [dict(item) for item in self.group_counts],
            },
        }
        if include_integrity and self.artifact_sha256 is not None:
            payload["artifact_sha256"] = self.artifact_sha256
        return payload

    def with_integrity(self) -> Self:
        digest = canonical_sha256(self.to_payload(include_integrity=False))
        return type(self)(
            dataset_name=self.dataset_name,
            dataset_version=self.dataset_version,
            protocol_hash=self.protocol_hash,
            manifest_path=self.manifest_path,
            manifest_sha256=self.manifest_sha256,
            ecg_id=self.ecg_id,
            patient_id=self.patient_id,
            sex=self.sex,
            age_band=self.age_band,
            group_counts=self.group_counts,
            artifact_sha256=digest,
        )


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
        raise SubgroupArtifactError("subgroup payload must be finite JSON") from error
    return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash one required regular file."""

    source = Path(path)
    if not source.is_file():
        raise SubgroupIntegrityError(f"required subgroup source is missing: {source}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as error:
        raise SubgroupIntegrityError(f"could not hash subgroup source {source}: {error}") from error
    return "sha256:" + digest.hexdigest()


def build_subgroup_artifact(
    manifest_path: str | Path,
    *,
    protocol: ExperimentProtocol,
    expected_manifest_sha256: str | None = None,
) -> SubgroupArtifact:
    """Build the canonical sex/age-band artifact without reading label columns."""

    source = Path(manifest_path).resolve()
    manifest_sha256 = sha256_file(source)
    if expected_manifest_sha256 is not None and (
        _normalize_sha256(expected_manifest_sha256, "expected_manifest_sha256")
        != manifest_sha256
    ):
        raise SubgroupIntegrityError("source manifest hash differs from the release bundle")
    frame = _read_metadata_only(source)
    _validate_identity_frame(frame)
    selected = frame.loc[frame["strat_fold"].eq(FINAL_TEST_FOLDS[0])].copy()
    if selected.empty:
        raise SubgroupArtifactError("manifest contains no final-test subgroup rows")
    selected = selected.sort_values("ecg_id", kind="stable").reset_index(drop=True)

    ecg_id = _integer_tuple(selected["ecg_id"].tolist(), "ecg_id", minimum=1)
    patient_id = _integer_tuple(
        selected["patient_id"].tolist(), "patient_id", minimum=1
    )
    sex = tuple(_sex_group(value) for value in selected["sex"].tolist())
    age_band = tuple(_age_group(value) for value in selected["age"].tolist())
    return _validated_artifact(
        dataset_name=protocol.dataset_name,
        dataset_version=protocol.dataset_version,
        protocol_hash=protocol.protocol_hash,
        manifest_path=source,
        manifest_sha256=manifest_sha256,
        ecg_id=ecg_id,
        patient_id=patient_id,
        sex=sex,
        age_band=age_band,
    ).with_integrity()


def save_subgroup_artifact(
    artifact: SubgroupArtifact, path: str | Path
) -> tuple[Path, str]:
    """Atomically save a new self-hashed subgroup artifact without overwrite."""

    validated = _validated_artifact(
        dataset_name=artifact.dataset_name,
        dataset_version=artifact.dataset_version,
        protocol_hash=artifact.protocol_hash,
        manifest_path=artifact.manifest_path,
        manifest_sha256=artifact.manifest_sha256,
        ecg_id=artifact.ecg_id,
        patient_id=artifact.patient_id,
        sex=artifact.sex,
        age_band=artifact.age_band,
    ).with_integrity()
    if artifact.artifact_sha256 not in {None, validated.artifact_sha256}:
        raise SubgroupIntegrityError("in-memory subgroup artifact hash is invalid")
    destination = Path(path).resolve()
    if destination.suffix.casefold() != ".json":
        raise SubgroupArtifactError("subgroup artifact path must end in .json")
    _write_new_json(destination, validated.to_payload())
    return destination, cast(str, validated.artifact_sha256)


def load_subgroup_artifact(
    path: str | Path,
    *,
    protocol: ExperimentProtocol,
    expected_manifest_sha256: str | None = None,
    verify_source: bool = True,
) -> SubgroupArtifact:
    """Load and reverify a stored subgroup artifact and, by default, its source."""

    source = Path(path).resolve()
    try:
        decoded: object = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SubgroupIntegrityError(f"could not decode subgroup artifact: {error}") from error
    root = _mapping(decoded, "subgroup artifact")
    required = {
        "schema_version",
        "artifact_type",
        "dataset",
        "protocol_hash",
        "source_manifest",
        "folds",
        "definitions",
        "ecg_id",
        "patient_id",
        "attributes",
        "summary",
        "artifact_sha256",
    }
    _exact_keys(root, required, "subgroup artifact")
    if root["schema_version"] != SUBGROUP_SCHEMA_VERSION:
        raise SubgroupIntegrityError("unsupported subgroup schema_version")
    if root["artifact_type"] != SUBGROUP_ARTIFACT_TYPE:
        raise SubgroupIntegrityError("unexpected subgroup artifact_type")
    stored_hash = _normalize_sha256(root["artifact_sha256"], "artifact_sha256")
    unhashed = dict(root)
    del unhashed["artifact_sha256"]
    if canonical_sha256(unhashed) != stored_hash:
        raise SubgroupIntegrityError("subgroup artifact self-hash mismatch")

    dataset = _mapping(root["dataset"], "dataset")
    _exact_keys(dataset, {"name", "version"}, "dataset")
    if dataset != {
        "name": protocol.dataset_name,
        "version": protocol.dataset_version,
    }:
        raise SubgroupIntegrityError("subgroup dataset differs from the protocol")
    if root["protocol_hash"] != protocol.protocol_hash:
        raise SubgroupIntegrityError("subgroup protocol hash differs from the protocol")
    if _integer_tuple(_sequence(root["folds"], "folds"), "folds", minimum=1) != (
        FINAL_TEST_FOLDS
    ):
        raise SubgroupIntegrityError("subgroup artifact must contain fold 10 only")
    if root["definitions"] != _definitions_payload():
        raise SubgroupIntegrityError("subgroup definitions are not canonical")

    manifest = _mapping(root["source_manifest"], "source_manifest")
    _exact_keys(
        manifest,
        {"path", "sha256", "columns_read", "diagnostic_target_columns_read"},
        "source_manifest",
    )
    if manifest["columns_read"] != list(_METADATA_COLUMNS) or (
        manifest["diagnostic_target_columns_read"] is not False
    ):
        raise SubgroupIntegrityError("subgroup source column declaration is unsafe")
    manifest_path = Path(_string(manifest["path"], "source_manifest.path")).resolve()
    manifest_sha256 = _normalize_sha256(
        manifest["sha256"], "source_manifest.sha256"
    )
    if expected_manifest_sha256 is not None and manifest_sha256 != _normalize_sha256(
        expected_manifest_sha256, "expected_manifest_sha256"
    ):
        raise SubgroupIntegrityError("subgroup manifest differs from the release bundle")

    attributes = _mapping(root["attributes"], "attributes")
    _exact_keys(attributes, set(SUBGROUP_ATTRIBUTES), "attributes")
    artifact = _validated_artifact(
        dataset_name=protocol.dataset_name,
        dataset_version=protocol.dataset_version,
        protocol_hash=protocol.protocol_hash,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        ecg_id=_integer_tuple(_sequence(root["ecg_id"], "ecg_id"), "ecg_id", minimum=1),
        patient_id=_integer_tuple(
            _sequence(root["patient_id"], "patient_id"), "patient_id", minimum=1
        ),
        sex=_string_tuple(_sequence(attributes["sex"], "attributes.sex"), "sex"),
        age_band=_string_tuple(
            _sequence(attributes["age_band"], "attributes.age_band"), "age_band"
        ),
    ).with_integrity()
    if artifact.artifact_sha256 != stored_hash:
        raise SubgroupIntegrityError("subgroup values do not reproduce the artifact hash")
    if artifact.to_payload()["summary"] != root["summary"]:
        raise SubgroupIntegrityError("subgroup summary does not match its rows")
    if verify_source:
        if sha256_file(manifest_path) != manifest_sha256:
            raise SubgroupIntegrityError("subgroup source manifest changed")
        rebuilt = build_subgroup_artifact(
            manifest_path,
            protocol=protocol,
            expected_manifest_sha256=manifest_sha256,
        )
        if rebuilt.to_payload() != artifact.to_payload():
            raise SubgroupIntegrityError("subgroup rows differ from the bound manifest")
    return artifact


def _read_metadata_only(path: Path) -> pd.DataFrame:
    try:
        if path.suffix.casefold() == ".parquet":
            frame = pd.read_parquet(path, columns=list(_METADATA_COLUMNS))
        elif path.suffix.casefold() == ".csv":
            frame = pd.read_csv(path, usecols=list(_METADATA_COLUMNS))
        else:
            raise SubgroupArtifactError("manifest must be a .parquet or .csv file")
    except (OSError, TypeError, ValueError) as error:
        raise SubgroupArtifactError(f"could not read subgroup metadata: {error}") from error
    if frame.empty:
        raise SubgroupArtifactError("manifest must not be empty")
    if tuple(frame.columns) != _METADATA_COLUMNS:
        missing = sorted(set(_METADATA_COLUMNS).difference(frame.columns))
        raise SubgroupArtifactError(f"manifest is missing subgroup columns: {missing}")
    return cast(pd.DataFrame, frame)


def _validate_identity_frame(frame: pd.DataFrame) -> None:
    if frame.loc[:, ["ecg_id", "patient_id", "strat_fold"]].isna().any().any():
        raise SubgroupArtifactError("subgroup identities and folds must not be missing")
    try:
        ecg_ids = pd.to_numeric(frame["ecg_id"], errors="raise")
        patient_ids = pd.to_numeric(frame["patient_id"], errors="raise")
        folds = pd.to_numeric(frame["strat_fold"], errors="raise")
    except (TypeError, ValueError) as error:
        raise SubgroupArtifactError("subgroup identities and folds must be numeric") from error
    for values, name in (
        (ecg_ids, "ecg_id"),
        (patient_ids, "patient_id"),
        (folds, "strat_fold"),
    ):
        numeric = values.to_numpy(dtype=float)
        if not all(math.isfinite(value) and value.is_integer() for value in numeric):
            raise SubgroupArtifactError(f"{name} must contain finite integers")
    if ecg_ids.duplicated().any():
        raise SubgroupArtifactError("manifest ecg_id values must be unique")
    if not folds.isin(range(1, 11)).all():
        raise SubgroupArtifactError("manifest folds must lie in 1-10")
    fold_frame = pd.DataFrame({"patient_id": patient_ids, "fold": folds})
    if (fold_frame.groupby("patient_id")["fold"].nunique() != 1).any():
        raise SubgroupArtifactError("a patient occurs in multiple manifest folds")
    frame["ecg_id"] = ecg_ids.astype("int64")
    frame["patient_id"] = patient_ids.astype("int64")
    frame["strat_fold"] = folds.astype("int8")


def _sex_group(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SubgroupArtifactError("sex must use PTB-XL numeric coding")
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer() or int(numeric) not in {0, 1}:
        raise SubgroupArtifactError("sex must be 0, 1, or missing")
    return "male" if int(numeric) == 0 else "female"


def _age_group(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SubgroupArtifactError("age must be numeric")
    numeric = float(value)
    if numeric == 300.0:
        return "80+"
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 120.0:
        raise SubgroupArtifactError(
            "age must be in 0-120 or the PTB-XL censored sentinel 300"
        )
    if numeric < 40.0:
        return "<40"
    if numeric < 60.0:
        return "40-59"
    if numeric < 80.0:
        return "60-79"
    return "80+"


def _validated_artifact(
    *,
    dataset_name: str,
    dataset_version: str,
    protocol_hash: str,
    manifest_path: Path,
    manifest_sha256: str,
    ecg_id: Sequence[int],
    patient_id: Sequence[int],
    sex: Sequence[str],
    age_band: Sequence[str],
) -> SubgroupArtifact:
    ids = _integer_tuple(tuple(ecg_id), "ecg_id", minimum=1)
    patients = _integer_tuple(tuple(patient_id), "patient_id", minimum=1)
    sexes = tuple(sex)
    ages = tuple(age_band)
    lengths = {len(ids), len(patients), len(sexes), len(ages)}
    if lengths != {len(ids)} or not ids:
        raise SubgroupArtifactError("subgroup arrays must be non-empty and aligned")
    if len(set(ids)) != len(ids) or tuple(sorted(ids)) != ids:
        raise SubgroupArtifactError("subgroup ecg_id values must be unique and sorted")
    if any(value not in SEX_GROUPS for value in sexes):
        raise SubgroupArtifactError("subgroup sex values are not canonical")
    if any(value not in AGE_GROUPS for value in ages):
        raise SubgroupArtifactError("subgroup age-band values are not canonical")
    counts = _group_counts(patients, sexes, ages)
    return SubgroupArtifact(
        dataset_name=_string(dataset_name, "dataset_name"),
        dataset_version=_string(dataset_version, "dataset_version"),
        protocol_hash=_normalize_sha256(protocol_hash, "protocol_hash"),
        manifest_path=manifest_path.resolve(),
        manifest_sha256=_normalize_sha256(manifest_sha256, "manifest_sha256"),
        ecg_id=ids,
        patient_id=patients,
        sex=sexes,
        age_band=ages,
        group_counts=counts,
    )


def _group_counts(
    patient_id: Sequence[int], sex: Sequence[str], age_band: Sequence[str]
) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    for attribute, values, groups in (
        ("sex", sex, SEX_GROUPS),
        ("age_band", age_band, AGE_GROUPS),
    ):
        for group in groups:
            positions = [index for index, value in enumerate(values) if value == group]
            rows.append(
                {
                    "attribute": attribute,
                    "group": group,
                    "records": len(positions),
                    "patients": len({patient_id[index] for index in positions}),
                }
            )
    return tuple(rows)


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable subgroup artifact already exists: {path}")
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(raw_temp)
    try:
        serialized = json.dumps(
            payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True
        )
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"immutable subgroup artifact already exists: {path}")
        os.replace(temporary, path)
    finally:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)


def _definitions_payload() -> dict[str, object]:
    return cast(
        dict[str, object],
        json.loads(json.dumps(_DEFINITIONS, sort_keys=True, allow_nan=False)),
    )


def _normalize_sha256(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise SubgroupIntegrityError(f"{context} must be a SHA-256 string")
    normalized = value.removeprefix("sha256:").casefold()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise SubgroupIntegrityError(f"{context} must be a SHA-256 string")
    return "sha256:" + normalized


def _integer_tuple(value: Sequence[object], context: str, *, minimum: int) -> tuple[int, ...]:
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise SubgroupArtifactError(f"{context} must contain integers")
        numeric = float(item)
        if not math.isfinite(numeric) or not numeric.is_integer() or numeric < minimum:
            raise SubgroupArtifactError(f"{context} must contain integers >= {minimum}")
        result.append(int(numeric))
    return tuple(result)


def _string_tuple(value: Sequence[object], context: str) -> tuple[str, ...]:
    return tuple(_string(item, context) for item in value)


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SubgroupArtifactError(f"{context} must be a non-empty string")
    return value


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise SubgroupIntegrityError(f"{context} must be a string-keyed mapping")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise SubgroupIntegrityError(f"{context} must be a sequence")
    return cast(Sequence[object], value)


def _exact_keys(
    value: Mapping[str, object], expected: set[str], context: str
) -> None:
    if set(value) != expected:
        raise SubgroupIntegrityError(f"{context} keys are not canonical")


__all__ = [
    "AGE_GROUPS",
    "SEX_GROUPS",
    "SUBGROUP_ARTIFACT_TYPE",
    "SUBGROUP_ATTRIBUTES",
    "SUBGROUP_SCHEMA_VERSION",
    "SubgroupArtifact",
    "SubgroupArtifactError",
    "SubgroupIntegrityError",
    "build_subgroup_artifact",
    "load_subgroup_artifact",
    "save_subgroup_artifact",
]
