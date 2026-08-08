"""Build and verify the canonical PTB-XL five-superclass manifest.

The functions in this module deliberately fail closed.  The official source
counts, patient-safe folds, relative waveform paths, and diagnostic label
mapping are validated before an artifact is written.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path, PurePosixPath, PureWindowsPath

import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from ecg_trust.constants import (
    EXPECTED_PATIENTS,
    EXPECTED_RECORDS,
    PTBXL_VERSION,
    SUPERCLASSES,
    TARGET_COLUMNS,
)

MANIFEST_SCHEMA_VERSION = "1"
EXPECTED_FOLDS: tuple[int, ...] = tuple(range(1, 11))
EXPECTED_SUPERCLASS_COUNTS: dict[str, int] = {
    "NORM": 9_514,
    "MI": 5_469,
    "STTC": 5_235,
    "CD": 4_898,
    "HYP": 2_649,
}
MAX_SCP_CODES_TEXT_LENGTH = 100_000
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

IDENTITY_COLUMNS: tuple[str, ...] = (
    "ecg_id",
    "patient_id",
    "strat_fold",
    "split",
    "record_path",
    "scp_codes",
    "labels",
)
OPTIONAL_AUDIT_COLUMNS: tuple[str, ...] = (
    "age",
    "sex",
    "height",
    "weight",
    "recording_date",
    "validated_by_human",
)
LABEL_COLUMNS: tuple[str, ...] = TARGET_COLUMNS


class ManifestError(ValueError):
    """Raised when source data violates the PTB-XL manifest contract."""


@dataclass(frozen=True)
class ManifestArtifacts:
    """Paths and stable hashes produced by :func:`write_manifest_artifacts`."""

    csv_path: Path
    parquet_path: Path
    summary_path: Path
    checksums_path: Path
    csv_sha256: str
    parquet_sha256: str
    summary_sha256: str


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the lower-case SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_scp_codes(value: object) -> dict[str, float]:
    """Safely parse one ``scp_codes`` cell into a validated dictionary.

    PTB-XL stores Python dictionary literals in CSV cells.  ``literal_eval``
    is used instead of ``eval`` and the parsed shape and scalar values are
    checked so malformed metadata cannot become executable input.
    """

    if not isinstance(value, str):
        raise ManifestError(f"scp_codes must be text, got {type(value).__name__}")
    if len(value) > MAX_SCP_CODES_TEXT_LENGTH:
        raise ManifestError("scp_codes cell is unreasonably large")
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
        raise ManifestError(f"invalid scp_codes literal: {value[:120]!r}") from exc
    if not isinstance(parsed, dict):
        raise ManifestError("scp_codes must contain a dictionary literal")

    result: dict[str, float] = {}
    for raw_code, raw_likelihood in parsed.items():
        if not isinstance(raw_code, str) or not raw_code.strip():
            raise ManifestError("every scp_codes key must be a non-empty string")
        if isinstance(raw_likelihood, bool) or not isinstance(raw_likelihood, Real):
            raise ManifestError(f"likelihood for {raw_code!r} must be numeric")
        likelihood = float(raw_likelihood)
        if not math.isfinite(likelihood):
            raise ManifestError(f"likelihood for {raw_code!r} must be finite")
        result[raw_code.strip()] = likelihood
    return dict(sorted(result.items()))


def parse_sha256sums(text: str) -> dict[str, str]:
    """Parse the GNU-style checksum inventory published by PhysioNet."""

    checksums: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ManifestError(f"invalid SHA256SUMS line {line_number}")
        digest, raw_name = parts
        digest = digest.lower()
        if not SHA256_PATTERN.fullmatch(digest):
            raise ManifestError(f"invalid SHA-256 digest on line {line_number}")
        name = raw_name.lstrip("* ").replace("\\", "/")
        while name.startswith("./"):
            name = name[2:]
        normalized = validate_relative_path(name)
        if normalized in checksums and checksums[normalized] != digest:
            raise ManifestError(f"conflicting checksums for {normalized}")
        checksums[normalized] = digest
    if not checksums:
        raise ManifestError("SHA256SUMS inventory is empty")
    return dict(sorted(checksums.items()))


def validate_relative_path(value: object) -> str:
    """Return a canonical POSIX path after rejecting absolute/traversal paths."""

    if not isinstance(value, str) or not value.strip():
        raise ManifestError("record path must be non-empty text")
    normalized = value.strip().replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(normalized)
    if posix_path.is_absolute() or windows_path.is_absolute():
        raise ManifestError(f"absolute path is forbidden: {value!r}")
    if any(part in {"", ".", ".."} for part in posix_path.parts):
        raise ManifestError(f"unsafe relative path: {value!r}")
    return posix_path.as_posix()


def resolve_relative_path(dataset_root: Path, relative_path: str) -> Path:
    """Resolve a validated dataset path and prove it remains under its root."""

    canonical = validate_relative_path(relative_path)
    root = dataset_root.resolve()
    resolved = root.joinpath(*PurePosixPath(canonical).parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ManifestError(f"path escapes dataset root: {relative_path!r}") from exc
    return resolved


def verify_sha256sums(
    dataset_root: Path,
    checksums: Mapping[str, str],
    relative_paths: Iterable[str] | None = None,
) -> dict[str, str]:
    """Verify selected files against an official checksum mapping."""

    selected = sorted(checksums) if relative_paths is None else sorted(set(relative_paths))
    verified: dict[str, str] = {}
    for relative_path in selected:
        canonical = validate_relative_path(relative_path)
        expected = checksums.get(canonical)
        if expected is None:
            raise ManifestError(f"no official checksum for {canonical}")
        path = resolve_relative_path(dataset_root, canonical)
        if not path.is_file():
            raise ManifestError(f"checksummed file is missing: {canonical}")
        observed = sha256_file(path)
        if observed != expected:
            raise ManifestError(
                f"SHA-256 mismatch for {canonical}: expected {expected}, observed {observed}"
            )
        verified[canonical] = observed
    return verified


def load_diagnostic_mapping(statements_path: Path) -> dict[str, str]:
    """Load SCP statement to fixed diagnostic-superclass mapping."""

    statements = pd.read_csv(statements_path, index_col=0)
    required = {"diagnostic", "diagnostic_class"}
    missing = required.difference(statements.columns)
    if missing:
        raise ManifestError(f"scp_statements.csv is missing columns: {sorted(missing)}")

    diagnostic = pd.to_numeric(statements["diagnostic"], errors="coerce").fillna(0).eq(1)
    mapping: dict[str, str] = {}
    for raw_code, row in statements.loc[diagnostic].iterrows():
        code = str(raw_code).strip()
        raw_class = row["diagnostic_class"]
        if pd.isna(raw_class):
            continue
        superclass = str(raw_class).strip()
        if superclass in SUPERCLASSES:
            mapping[code] = superclass
    if not mapping:
        raise ManifestError("no diagnostic superclass mappings were found")
    return dict(sorted(mapping.items()))


def map_superclasses(
    scp_codes: Mapping[str, float], statement_mapping: Mapping[str, str]
) -> tuple[str, ...]:
    """Aggregate present diagnostic statements in fixed superclass order."""

    present = {statement_mapping[code] for code in scp_codes if code in statement_mapping}
    return tuple(superclass for superclass in SUPERCLASSES if superclass in present)


def split_for_fold(fold: int) -> str:
    """Map official folds onto the leakage-resistant research protocol."""

    if fold in range(1, 8):
        return "development_train"
    if fold == 8:
        return "model_selection"
    if fold == 9:
        return "calibration"
    if fold == 10:
        return "test"
    raise ManifestError(f"strat_fold must be between 1 and 10, got {fold}")


def assert_patient_fold_disjoint(metadata: pd.DataFrame) -> None:
    """Assert that every patient belongs to exactly one official fold."""

    fold_counts = metadata.groupby("patient_id", dropna=False)["strat_fold"].nunique()
    offenders = fold_counts[fold_counts != 1]
    if not offenders.empty:
        preview = ", ".join(str(value) for value in offenders.index[:10])
        raise ManifestError(f"patients assigned to multiple folds: {preview}")


def _integer_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not values.map(lambda value: float(value).is_integer()).all():
        raise ManifestError(f"{column} must contain non-null integers")
    return values.astype("int64")


def _canonical_record_stem(value: object) -> str:
    canonical = validate_relative_path(value)
    path = PurePosixPath(canonical)
    if not path.parts or path.parts[0] != "records100":
        raise ManifestError(f"filename_lr must point into records100: {canonical!r}")
    if path.suffix in {".hea", ".dat"}:
        path = path.with_suffix("")
    elif path.suffix:
        raise ManifestError(f"unexpected waveform path suffix: {canonical!r}")
    return path.as_posix()


def verify_record_files(dataset_root: Path, record_stems: Iterable[str]) -> None:
    """Assert both WFDB header and sample files exist for every record stem."""

    missing: list[str] = []
    for stem in sorted(set(record_stems)):
        for suffix in (".hea", ".dat"):
            relative_path = f"{stem}{suffix}"
            if not resolve_relative_path(dataset_root, relative_path).is_file():
                missing.append(relative_path)
                if len(missing) == 20:
                    break
        if len(missing) == 20:
            break
    if missing:
        suffix = " (first 20 shown)" if len(missing) == 20 else ""
        raise ManifestError(f"missing waveform files{suffix}: {', '.join(missing)}")


def _load_metadata(metadata_path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    if "ecg_id" not in metadata.columns and "Unnamed: 0" in metadata.columns:
        metadata = metadata.rename(columns={"Unnamed: 0": "ecg_id"})
    required = {"ecg_id", "patient_id", "strat_fold", "filename_lr", "scp_codes"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ManifestError(f"ptbxl_database.csv is missing columns: {sorted(missing)}")
    metadata = metadata.copy()
    for column in ("ecg_id", "patient_id", "strat_fold"):
        metadata[column] = _integer_series(metadata, column)
    if metadata["ecg_id"].duplicated().any():
        duplicates = metadata.loc[metadata["ecg_id"].duplicated(), "ecg_id"].head().tolist()
        raise ManifestError(f"duplicate ecg_id values: {duplicates}")
    folds = tuple(sorted(metadata["strat_fold"].unique().tolist()))
    if any(fold not in EXPECTED_FOLDS for fold in folds):
        raise ManifestError(f"unexpected strat_fold values: {folds}")
    assert_patient_fold_disjoint(metadata)
    return metadata.sort_values("ecg_id", kind="mergesort").reset_index(drop=True)


def _validate_official_source_counts(metadata: pd.DataFrame) -> None:
    if len(metadata) != EXPECTED_RECORDS:
        raise ManifestError(f"expected {EXPECTED_RECORDS} source records, found {len(metadata)}")
    patient_count = int(metadata["patient_id"].nunique())
    if patient_count != EXPECTED_PATIENTS:
        raise ManifestError(f"expected {EXPECTED_PATIENTS} patients, found {patient_count}")
    observed_folds = tuple(sorted(int(value) for value in metadata["strat_fold"].unique()))
    if observed_folds != EXPECTED_FOLDS:
        raise ManifestError(f"expected folds {EXPECTED_FOLDS}, found {observed_folds}")


def build_manifest(
    dataset_root: Path,
    *,
    strict_official_counts: bool = True,
    verify_files: bool = True,
    include_unlabeled: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build the canonical manifest and its deterministic summary payload."""

    root = dataset_root.resolve()
    metadata_path = root / "ptbxl_database.csv"
    statements_path = root / "scp_statements.csv"
    for source in (metadata_path, statements_path):
        if not source.is_file():
            raise ManifestError(f"required source file is missing: {source.name}")

    metadata = _load_metadata(metadata_path)
    if strict_official_counts:
        _validate_official_source_counts(metadata)
    statement_mapping = load_diagnostic_mapping(statements_path)

    parsed_codes = metadata["scp_codes"].map(parse_scp_codes)
    label_sets = parsed_codes.map(lambda codes: map_superclasses(codes, statement_mapping))
    record_stems = metadata["filename_lr"].map(_canonical_record_stem)
    if verify_files:
        verify_record_files(root, record_stems)

    manifest = pd.DataFrame(
        {
            "ecg_id": metadata["ecg_id"],
            "patient_id": metadata["patient_id"],
            "strat_fold": metadata["strat_fold"].astype("int8"),
            "split": metadata["strat_fold"].map(split_for_fold),
            "record_path": record_stems,
            "scp_codes": parsed_codes.map(
                lambda codes: json.dumps(codes, sort_keys=True, separators=(",", ":"))
            ),
            "labels": label_sets.map(lambda labels: "|".join(labels)),
        }
    )
    for column in OPTIONAL_AUDIT_COLUMNS:
        if column in metadata.columns:
            manifest[column] = metadata[column]
    for superclass, column in zip(SUPERCLASSES, LABEL_COLUMNS, strict=True):
        manifest[column] = label_sets.map(
            lambda labels, name=superclass: int(name in labels)
        ).astype("int8")

    source_records = len(manifest)
    source_patients = int(manifest["patient_id"].nunique())
    observed_label_counts = {
        superclass: int(manifest[column].sum())
        for superclass, column in zip(SUPERCLASSES, LABEL_COLUMNS, strict=True)
    }
    if strict_official_counts and observed_label_counts != EXPECTED_SUPERCLASS_COUNTS:
        raise ManifestError(
            "official superclass counts do not match: "
            f"expected {EXPECTED_SUPERCLASS_COUNTS}, found {observed_label_counts}"
        )

    if not include_unlabeled:
        manifest = manifest.loc[manifest[list(LABEL_COLUMNS)].sum(axis=1) > 0].copy()
    manifest = manifest.sort_values("ecg_id", kind="mergesort").reset_index(drop=True)
    assert_patient_fold_disjoint(manifest)

    ordered_columns = [*IDENTITY_COLUMNS]
    ordered_columns.extend(
        column for column in OPTIONAL_AUDIT_COLUMNS if column in manifest.columns
    )
    ordered_columns.extend(LABEL_COLUMNS)
    manifest = manifest.loc[:, ordered_columns]

    summary: dict[str, object] = {
        "dataset": "PTB-XL",
        "dataset_version": PTBXL_VERSION,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "sampling_frequency_hz": 100,
        "signal_duration_seconds": 10,
        "signal_shape": [12, 1000],
        "superclasses": list(SUPERCLASSES),
        "superclass_source_counts": observed_label_counts,
        "source_records": source_records,
        "source_patients": source_patients,
        "manifest_records": len(manifest),
        "manifest_patients": int(manifest["patient_id"].nunique()),
        "unlabeled_records_excluded": source_records - len(manifest),
        "fold_record_counts": {
            str(fold): int((manifest["strat_fold"] == fold).sum()) for fold in EXPECTED_FOLDS
        },
        "fold_patient_counts": {
            str(fold): int(manifest.loc[manifest["strat_fold"] == fold, "patient_id"].nunique())
            for fold in EXPECTED_FOLDS
        },
        "statement_mapping": statement_mapping,
        "source_sha256": {
            "ptbxl_database.csv": sha256_file(metadata_path),
            "scp_statements.csv": sha256_file(statements_path),
        },
    }
    return manifest, summary


def _atomic_replace(temp_path: Path, destination: Path) -> None:
    os.replace(temp_path, destination)


def write_manifest_artifacts(
    manifest: pd.DataFrame,
    summary: Mapping[str, object],
    output_dir: Path,
    *,
    stem: str = "ptbxl_superclasses_v1.0.3",
) -> ManifestArtifacts:
    """Write deterministic CSV, Parquet, summary JSON, and hash inventory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    parquet_path = output_dir / f"{stem}.parquet"
    summary_path = output_dir / f"{stem}.summary.json"
    checksums_path = output_dir / f"{stem}.sha256"
    csv_temp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    parquet_temp = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    summary_temp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    checksums_temp = checksums_path.with_suffix(checksums_path.suffix + ".tmp")

    manifest.to_csv(
        csv_temp,
        index=False,
        lineterminator="\n",
        encoding="utf-8",
        float_format="%.10g",
        na_rep="",
    )
    table = pa.Table.from_pandas(manifest, preserve_index=False)
    pq.write_table(
        table,
        parquet_temp,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        version="2.6",
    )
    csv_sha256 = sha256_file(csv_temp)
    parquet_sha256 = sha256_file(parquet_temp)

    complete_summary = dict(summary)
    complete_summary["manifest"] = {
        "csv": csv_path.name,
        "csv_sha256": csv_sha256,
        "parquet": parquet_path.name,
        "parquet_sha256": parquet_sha256,
        "rows": len(manifest),
        "columns": list(manifest.columns),
    }
    summary_temp.write_text(
        json.dumps(complete_summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    summary_sha256 = sha256_file(summary_temp)
    checksums_temp.write_text(
        "".join(
            (
                f"{csv_sha256}  {csv_path.name}\n",
                f"{parquet_sha256}  {parquet_path.name}\n",
                f"{summary_sha256}  {summary_path.name}\n",
            )
        ),
        encoding="utf-8",
        newline="\n",
    )

    for temporary, destination in (
        (csv_temp, csv_path),
        (parquet_temp, parquet_path),
        (summary_temp, summary_path),
        (checksums_temp, checksums_path),
    ):
        _atomic_replace(temporary, destination)

    return ManifestArtifacts(
        csv_path=csv_path,
        parquet_path=parquet_path,
        summary_path=summary_path,
        checksums_path=checksums_path,
        csv_sha256=csv_sha256,
        parquet_sha256=parquet_sha256,
        summary_sha256=summary_sha256,
    )
