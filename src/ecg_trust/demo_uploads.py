"""Bounded WFDB upload reads and matched-pair validation for the local demo.

This module owns upload filenames, byte limits and header containment. Model
loading, temporary-file lifetime, inference and HTTP errors stay in demo_app.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

from fastapi import UploadFile

from ecg_trust.constants import LEADS

_SAFE_UPLOAD_STEM = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}\Z")


class DemoWebError(ValueError):
    """Raised when web-layer configuration or an upload is unsafe."""


def _validated_upload_names(header: UploadFile, signal: UploadFile) -> str:
    names = (header.filename, signal.filename)
    if any(name is None or Path(name).name != name for name in names):
        raise DemoWebError("uploaded filenames must not contain paths")
    header_name = cast(str, header.filename)
    signal_name = cast(str, signal.filename)
    if Path(header_name).suffix.casefold() != ".hea":
        raise DemoWebError("header upload must use the .hea extension")
    if Path(signal_name).suffix.casefold() != ".dat":
        raise DemoWebError("signal upload must use the .dat extension")
    header_stem = Path(header_name).stem
    signal_stem = Path(signal_name).stem
    if header_stem != signal_stem:
        raise DemoWebError("WFDB header and signal filenames must share one stem")
    if _SAFE_UPLOAD_STEM.fullmatch(header_stem) is None:
        raise DemoWebError("uploaded record name contains unsupported characters")
    return header_stem


async def _read_limited(upload: UploadFile, limit: int, label: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(min(64 * 1024, limit + 1 - total))
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise DemoWebError(f"{label} exceeds the {limit}-byte upload limit")
        chunks.append(chunk)
    if total == 0:
        raise DemoWebError(f"{label} is empty")
    return b"".join(chunks)


def _validate_uploaded_header(payload: bytes, stem: str) -> None:
    """Reject WFDB headers that could reference any file outside the upload pair."""

    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise DemoWebError("WFDB header must contain ASCII text") from error
    if "\x00" in text:
        raise DemoWebError("WFDB header contains a NUL byte")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise DemoWebError("WFDB header is empty")
    record_fields = lines[0].split()
    if len(record_fields) < 4 or record_fields[0] != stem:
        raise DemoWebError("WFDB header record name must match the uploaded filenames")
    try:
        signal_count = int(record_fields[1])
        frequency = float(record_fields[2].split("/", maxsplit=1)[0])
        sample_count = int(record_fields[3])
    except ValueError as error:
        raise DemoWebError("WFDB header has invalid record dimensions") from error
    if signal_count != len(LEADS) or frequency != 100.0 or sample_count != 1000:
        raise DemoWebError("WFDB header must declare 12 signals, 100 Hz, and 1000 samples")
    signal_lines = [line for line in lines[1:] if not line.startswith("#")]
    if len(signal_lines) != len(LEADS):
        raise DemoWebError("WFDB header must contain exactly 12 signal specification lines")
    expected_data_file = f"{stem}.dat"
    if any(line.split()[0] != expected_data_file for line in signal_lines):
        raise DemoWebError("WFDB header may reference only its matched uploaded .dat file")
