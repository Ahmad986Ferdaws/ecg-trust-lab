"""Verify and export an existing aggregate model passport without running inference."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from ecg_trust.passport import (
    MAX_CANONICAL_PASSPORT_BYTES,
    ModelPassportError,
    model_passport_from_json_bytes,
    model_passport_to_json_bytes,
    render_model_passport_markdown,
)
from ecg_trust.passport.models import Sha256Digest


def _digest(value: str) -> str:
    try:
        return TypeAdapter(Sha256Digest).validate_python(value)
    except ValidationError:
        raise argparse.ArgumentTypeError(
            "expected sha256: followed by 64 lowercase hex digits"
        ) from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command, description in (
        ("verify", "Verify canonical content, privacy, and independently supplied identities."),
        ("export", "Verify, then export canonical JSON or escaped Markdown to a new file."),
    ):
        subparser = commands.add_parser(command, help=description)
        subparser.add_argument("passport", type=Path, help="existing canonical passport JSON")
        subparser.add_argument("--expected-release-sha256", type=_digest, required=True)
        subparser.add_argument("--expected-bundle-sha256", type=_digest, required=True)
        subparser.add_argument("--expected-protocol-sha256", type=_digest, required=True)
        subparser.add_argument("--expected-release-id")
        if command == "export":
            subparser.add_argument("--format", choices=("json", "markdown"), default="json")
            subparser.add_argument("--output", type=Path, required=True)
    return parser


def _read_payload(path: Path) -> bytes:
    # O_NONBLOCK prevents a FIFO from hanging before its descriptor is checked.
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_CANONICAL_PASSPORT_BYTES:
            raise ValueError("passport must be a bounded regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read(MAX_CANONICAL_PASSPORT_BYTES + 1)
    finally:
        os.close(descriptor)


def _write_new_output(path: Path, payload: bytes) -> None:
    # Stage beside the destination, then publish through an atomic no-replace
    # link. A racing file, directory, or dangling symlink is never overwritten.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".passport-", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        passport = model_passport_from_json_bytes(
            _read_payload(args.passport),
            expected_release_sha256=args.expected_release_sha256,
            expected_bundle_sha256=args.expected_bundle_sha256,
            expected_protocol_sha256=args.expected_protocol_sha256,
            expected_release_id=args.expected_release_id,
        )
    except (ModelPassportError, OSError, ValueError):
        # Pydantic validation errors may echo unsafe input. Never print their
        # messages or the input path at this boundary.
        print("error: passport verification failed; no output was written", file=sys.stderr)
        return 1

    if args.command == "export":
        try:
            payload = (
                model_passport_to_json_bytes(passport)
                if args.format == "json"
                else render_model_passport_markdown(passport).encode("utf-8")
            )
            _write_new_output(args.output, payload)
        except FileExistsError:
            print("error: export destination already exists; no file was replaced", file=sys.stderr)
            return 1
        except (ModelPassportError, OSError, ValueError):
            print("error: passport export failed", file=sys.stderr)
            return 1
    print(
        json.dumps({"verified": True, "passport_sha256": passport.passport_sha256}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
