#!/usr/bin/env python3
"""Report the pass probability of an upper-bound proportion gate before freezing it."""

from __future__ import annotations

import argparse
import json
import sys

from ecg_trust.gate_power import (
    BoundMethod,
    GatePowerError,
    plan_gate,
    required_sample_size,
    split_conformal_rejection_rate,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--maximum-rate", type=float, required=True)
    rate = parser.add_mutually_exclusive_group(required=True)
    rate.add_argument("--true-rate", type=float)
    rate.add_argument(
        "--conformal",
        nargs=2,
        type=int,
        metavar=("CALIBRATION_SIZE", "RANK"),
        help="derive the design rate from a split-conformal order-statistic threshold",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--method",
        choices=[method.value for method in BoundMethod],
        default=BoundMethod.CLOPPER_PEARSON.value,
    )
    parser.add_argument("--target-probability", type=float, default=0.8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        true_rate = (
            split_conformal_rejection_rate(*args.conformal)
            if args.conformal is not None
            else args.true_rate
        )
        method = BoundMethod(args.method)
        plan = plan_gate(
            sample_size=args.sample_size,
            maximum_rate=args.maximum_rate,
            true_rate=true_rate,
            confidence=args.confidence,
            method=method,
        )
        needed = required_sample_size(
            maximum_rate=args.maximum_rate,
            true_rate=true_rate,
            target_probability=args.target_probability,
            confidence=args.confidence,
            method=method,
        )
    except GatePowerError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    report = plan.to_dict()
    report["target_probability"] = args.target_probability
    report["sample_size_for_target"] = needed
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
