#!/usr/bin/env python3
"""Compare aggregate subgroup metrics under explicitly selected research limits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

from ecg_trust.slice_gates import (
    InsufficientPolicy,
    SliceGateError,
    SliceObservation,
    evaluate_slice_gate,
)

MAX_INPUT_BYTES = 1024 * 1024


class InputError(ValueError):
    """The aggregate input does not satisfy the command's closed schema."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise InputError("input contains duplicate JSON keys")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise InputError("input must contain finite JSON numbers")


def _read_input(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_INPUT_BYTES:
            raise InputError("input must be a regular JSON file no larger than 1 MiB")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(MAX_INPUT_BYTES + 1)
        if len(payload) > MAX_INPUT_BYTES:
            raise InputError("input must be no larger than 1 MiB")
        return payload
    finally:
        os.close(descriptor)


def _rows(value: object) -> tuple[SliceObservation, ...]:
    if not isinstance(value, list) or not value:
        raise InputError("reference and candidate must be non-empty arrays")
    rows = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"group", "count", "value"}:
            raise InputError("each group must contain exactly group, count and value")
        rows.append(SliceObservation(**item))
    return tuple(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="aggregate JSON with reference and candidate groups"
    )
    parser.add_argument("--metric", required=True, help="name of the compared metric")
    parser.add_argument("--direction", choices=("higher", "lower"), required=True)
    parser.add_argument("--max-overall-drop", type=float, required=True)
    parser.add_argument("--max-slice-drop", type=float, required=True)
    parser.add_argument("--min-count", type=int, required=True)
    parser.add_argument(
        "--insufficient-policy",
        choices=[item.value for item in InsufficientPolicy],
        default="block",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        metric = args.metric.strip()
        if not metric or len(metric) > 120 or any(ord(char) < 32 for char in metric):
            raise InputError("metric must be a non-empty name of at most 120 characters")
        payload = _read_input(args.input)
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        if not isinstance(document, dict) or set(document) != {
            "schema_version",
            "reference",
            "candidate",
        }:
            raise InputError("input must contain exactly schema_version, reference and candidate")
        if type(document["schema_version"]) is not int or document["schema_version"] != 1:
            raise InputError("input schema_version must be integer 1")
        reference = _rows(document["reference"])
        candidate = _rows(document["candidate"])
        parameters = {
            "max_overall_drop": args.max_overall_drop,
            "max_slice_drop": args.max_slice_drop,
            "min_count": args.min_count,
            "higher_is_better": args.direction == "higher",
            "insufficient_policy": InsufficientPolicy(args.insufficient_policy),
        }
        result = evaluate_slice_gate(reference, candidate, **parameters)
        candidate_counts = {item.group: item.count for item in candidate}
        report = {
            "schema_version": 1,
            "report_type": "ecg_trust.aggregate_slice_comparison",
            "metric": metric,
            "input_sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            "parameters": parameters,
            "overall_weighting": "reference_counts_among_sufficient_groups",
            "interpretation": "point_estimates_only_not_statistical_or_clinical_validation",
            "group_counts": [
                {
                    "group": item.group,
                    "reference": item.count,
                    "candidate": candidate_counts[item.group],
                }
                for item in sorted(reference, key=lambda item: item.group)
            ],
            "result": result.to_dict(),
        }
        serialized = json.dumps(report, allow_nan=False, indent=2, sort_keys=True)
    except InputError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except (SliceGateError, ValueError, TypeError, OverflowError, OSError, RecursionError):
        # Upstream messages may include input group names or local paths.
        print("error: invalid or unreadable aggregate input or gate controls", file=sys.stderr)
        return 2
    print(serialized)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
