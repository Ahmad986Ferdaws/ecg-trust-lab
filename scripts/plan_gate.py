#!/usr/bin/env python3
"""Report the pass probability of an upper-bound proportion gate before freezing it."""

from __future__ import annotations

import argparse
import json
import sys

from ecg_trust.gate_power import (
    BoundMethod,
    ConformalThreshold,
    GatePowerError,
    plan_gate,
    required_sample_size,
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
        help=(
            "model a split-conformal order-statistic threshold; the shared random "
            "threshold makes the count beta-binomial"
        ),
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--method",
        choices=[method.value for method in BoundMethod],
        default=BoundMethod.CLOPPER_PEARSON.value,
    )
    parser.add_argument("--target-probability", type=float, default=0.8)
    parser.add_argument(
        "--search-limit",
        type=int,
        default=20_000,
        help="largest sample size examined when searching for the target probability",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        conformal = ConformalThreshold(*args.conformal) if args.conformal is not None else None
        true_rate = None if conformal is not None else args.true_rate
        method = BoundMethod(args.method)
        plan = plan_gate(
            sample_size=args.sample_size,
            maximum_rate=args.maximum_rate,
            true_rate=true_rate,
            conformal=conformal,
            confidence=args.confidence,
            method=method,
        )
        needed = required_sample_size(
            maximum_rate=args.maximum_rate,
            true_rate=true_rate,
            conformal=conformal,
            target_probability=args.target_probability,
            confidence=args.confidence,
            method=method,
            limit=args.search_limit,
        )
    except GatePowerError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    report = plan.to_dict()
    report["target_probability"] = args.target_probability
    report["sample_size_for_target"] = needed
    report["sample_size_search_limit"] = args.search_limit
    report["sample_size_search_exhausted"] = needed is None
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
