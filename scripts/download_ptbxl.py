#!/usr/bin/env python3
"""Download and verify the official PTB-XL 1.0.3 100 Hz subset.

Downloads are written to ``.part`` files and resumed with HTTP Range requests.
Every selected file represented in PhysioNet's SHA256SUMS inventory is checked
before the command reports success.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from ecg_trust.constants import EXPECTED_RECORDS, PTBXL_VERSION
from ecg_trust.data.manifest import (
    ManifestError,
    parse_sha256sums,
    resolve_relative_path,
    sha256_file,
    validate_relative_path,
    verify_sha256sums,
)

# PhysioNet documents this public, unsigned S3 mirror on the dataset page.  The
# object bytes match the canonical release and the canonical SHA256SUMS file is
# still the authority used below.
BASE_URL = f"https://physionet-open.s3.amazonaws.com/ptb-xl/{PTBXL_VERSION}/"
ROOT_FILES: tuple[str, ...] = (
    "LICENSE.txt",
    "RECORDS",
    "SHA256SUMS.txt",
    "example_physionet.py",
    "ptbxl_database.csv",
    "ptbxl_v102_changelog.txt",
    "ptbxl_v103_changelog.txt",
    "scp_statements.csv",
)
USER_AGENT = "ecg-trust-ptbxl-downloader/0.1 (+https://physionet.org/)"


@dataclass(frozen=True)
class DownloadResult:
    relative_path: str
    status: str
    bytes_written: int


def _url_for(relative_path: str) -> str:
    canonical = validate_relative_path(relative_path)
    return BASE_URL + "/".join(quote(part) for part in canonical.split("/"))


def _download_once(
    relative_path: str,
    destination_root: Path,
    *,
    expected_sha256: str | None,
    timeout: float,
    force: bool,
) -> DownloadResult:
    canonical = validate_relative_path(relative_path)
    destination = resolve_relative_path(destination_root, canonical)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if (
        destination.is_file()
        and not force
        and (expected_sha256 is None or sha256_file(destination) == expected_sha256)
    ):
        return DownloadResult(canonical, "verified-existing", 0)

    partial = destination.with_name(destination.name + ".part")
    if force:
        partial.unlink(missing_ok=True)
    start = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": USER_AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"

    request = Request(_url_for(canonical), headers=headers)
    try:
        response_context = urlopen(request, timeout=timeout)  # noqa: S310 - fixed HTTPS origin
    except HTTPError as exc:
        if exc.code == 416 and start:
            if expected_sha256 is not None and sha256_file(partial) == expected_sha256:
                os.replace(partial, destination)
                return DownloadResult(canonical, "resumed-complete", 0)
            partial.unlink(missing_ok=True)
        raise

    with response_context as response:
        status = getattr(response, "status", response.getcode())
        append = bool(start and status == 206)
        mode = "ab" if append else "wb"
        with partial.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)

    if expected_sha256 is not None:
        observed = sha256_file(partial)
        if observed != expected_sha256:
            raise ManifestError(
                f"SHA-256 mismatch after download for {canonical}: "
                f"expected {expected_sha256}, observed {observed}"
            )
    bytes_written = partial.stat().st_size
    os.replace(partial, destination)
    return DownloadResult(canonical, "downloaded", bytes_written)


def download_file(
    relative_path: str,
    destination_root: Path,
    *,
    expected_sha256: str | None,
    timeout: float,
    retries: int,
    force: bool = False,
) -> DownloadResult:
    """Download one file atomically, resuming an existing partial transfer."""

    last_error: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            return _download_once(
                relative_path,
                destination_root,
                expected_sha256=expected_sha256,
                timeout=timeout,
                force=force,
            )
        except (OSError, ManifestError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(2 ** (attempt - 1), 16))
    assert last_error is not None
    raise last_error


def read_record_stems(metadata_path: Path) -> list[str]:
    """Read and validate the unique 100 Hz waveform stems from metadata."""

    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "filename_lr" not in reader.fieldnames:
            raise ManifestError("ptbxl_database.csv has no filename_lr column")
        stems = [validate_relative_path(row["filename_lr"]) for row in reader]
    if len(stems) != EXPECTED_RECORDS:
        raise ManifestError(f"expected {EXPECTED_RECORDS} metadata rows, found {len(stems)}")
    if len(set(stems)) != EXPECTED_RECORDS:
        raise ManifestError("filename_lr values are not unique")
    if any(not stem.startswith("records100/") for stem in stems):
        raise ManifestError("metadata contains a filename_lr outside records100")
    return sorted(stems)


def selected_paths(metadata_path: Path) -> list[str]:
    stems = read_record_stems(metadata_path)
    waveforms = [f"{stem}{suffix}" for stem in stems for suffix in (".hea", ".dat")]
    return sorted(set((*ROOT_FILES, *waveforms)))


def _bootstrap_file(
    name: str,
    destination: Path,
    *,
    timeout: float,
    retries: int,
    verify_only: bool,
) -> None:
    path = resolve_relative_path(destination, name)
    if verify_only:
        if not path.is_file():
            raise ManifestError(f"required file is missing in verify-only mode: {name}")
        return
    result = download_file(
        name,
        destination,
        expected_sha256=None,
        timeout=timeout,
        retries=retries,
        # These two small bootstrap files establish the checksum inventory and
        # full record list, so never trust a prior unverified copy.
        force=True,
    )
    print(f"{result.status}: {name}")


def acquire(
    destination: Path,
    *,
    workers: int,
    timeout: float,
    retries: int,
    verify_only: bool,
    allow_missing_checksums: bool,
    force: bool,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    # Bootstrap the authoritative inventory and record metadata first.
    for name in ("SHA256SUMS.txt", "ptbxl_database.csv"):
        _bootstrap_file(
            name,
            destination,
            timeout=timeout,
            retries=retries,
            verify_only=verify_only,
        )

    checksums_path = resolve_relative_path(destination, "SHA256SUMS.txt")
    checksums = parse_sha256sums(checksums_path.read_text(encoding="utf-8"))
    paths = selected_paths(resolve_relative_path(destination, "ptbxl_database.csv"))
    paths_to_verify = [path for path in paths if path != "SHA256SUMS.txt"]
    missing_checksum_entries = sorted(path for path in paths_to_verify if path not in checksums)
    if missing_checksum_entries and not allow_missing_checksums:
        preview = ", ".join(missing_checksum_entries[:10])
        raise ManifestError(
            f"official inventory lacks {len(missing_checksum_entries)} selected files: {preview}"
        )

    if not verify_only:
        pending = [path for path in paths if path not in {"SHA256SUMS.txt", "ptbxl_database.csv"}]
        totals: Counter[str] = Counter()
        failures: list[tuple[str, BaseException]] = []
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ptbxl") as executor:
            futures: dict[Future[DownloadResult], str] = {
                executor.submit(
                    download_file,
                    path,
                    destination,
                    expected_sha256=checksums.get(path),
                    timeout=timeout,
                    retries=retries,
                    force=force,
                ): path
                for path in pending
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                relative_path = futures[future]
                try:
                    result = future.result()
                    totals[result.status] += 1
                except BaseException as exc:  # collect all transfer failures before exiting
                    failures.append((relative_path, exc))
                if completed % 500 == 0 or completed == len(pending):
                    print(f"progress: {completed}/{len(pending)} files")
        if failures:
            details = "\n".join(f"  {path}: {error}" for path, error in failures[:20])
            raise ManifestError(f"{len(failures)} downloads failed:\n{details}")
        print("download summary: " + ", ".join(f"{key}={value}" for key, value in totals.items()))

    verifiable = [path for path in paths_to_verify if path in checksums]
    print(f"verifying {len(verifiable)} files against official SHA-256 digests ...")
    verify_sha256sums(destination, checksums, verifiable)
    print(
        f"PTB-XL {PTBXL_VERSION} 100 Hz acquisition verified: "
        f"{EXPECTED_RECORDS} records, {len(verifiable)} checksummed files"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=project_root / "data" / "raw" / "ptb-xl" / PTBXL_VERSION,
        help="dataset root (default: %(default)s)",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--allow-missing-checksums", action="store_true")
    parser.add_argument("--force", action="store_true", help="redownload all selected files")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.workers > 64:
        parser.error("--workers must be between 1 and 64")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.retries < 1:
        parser.error("--retries must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        acquire(
            args.destination,
            workers=args.workers,
            timeout=args.timeout,
            retries=args.retries,
            verify_only=args.verify_only,
            allow_missing_checksums=args.allow_missing_checksums,
            force=args.force,
        )
    except (ManifestError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
