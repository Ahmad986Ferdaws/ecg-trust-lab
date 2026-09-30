#!/usr/bin/env python3
"""Compare quality-gate presets on a bank of synthetic acquisition faults.

Every case starts from one deterministic synthetic 12x1000 mV ECG, so the report
contains no patient data. Each case records whether a fault *should* be refused
and what each preset decides, which makes fail-open gaps visible at a glance:

    uv run --no-sync python scripts/quality_gate_stress_report.py
    uv run --no-sync python scripts/quality_gate_stress_report.py --format json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ecg_trust.quality.signal_quality import (
    DEFAULT_SIGNAL_QUALITY_CONFIG,
    SENTINEL_V2_SIGNAL_QUALITY_CONFIG,
    SignalMetadata,
    SignalQualityConfig,
    assess_signal_quality,
)

FloatArray = NDArray[np.float64]
PRESETS: dict[str, SignalQualityConfig] = {
    "v1": DEFAULT_SIGNAL_QUALITY_CONFIG,
    "v2": SENTINEL_V2_SIGNAL_QUALITY_CONFIG,
}
V2 = 7


def _beat(time: FloatArray) -> FloatArray:
    phase = np.mod(time, 1.0)

    def pulse(center: float, width: float, amplitude: float) -> FloatArray:
        return amplitude * np.exp(-0.5 * np.square((phase - center) / width))

    return (
        pulse(0.18, 0.035, 0.10)
        + pulse(0.375, 0.014, -0.12)
        + pulse(0.400, 0.016, 1.10)
        + pulse(0.430, 0.018, -0.24)
        + pulse(0.660, 0.075, 0.28)
    )


def synthetic_ecg() -> FloatArray:
    """Deterministic, Einthoven-consistent 12-lead ECG with no duplicated leads."""

    time = np.arange(1_000, dtype=np.float64) / 100.0
    beat = _beat(time)
    lead_i = 0.75 * _beat(time + 0.02)
    lead_ii = beat
    return np.stack(
        (
            lead_i,
            lead_ii,
            lead_ii - lead_i,
            -(lead_i + lead_ii) / 2.0,
            lead_i - lead_ii / 2.0,
            lead_ii - lead_i / 2.0,
            -0.40 * beat,
            -0.15 * beat,
            0.30 * beat,
            0.65 * beat,
            0.90 * beat,
            0.80 * beat,
        )
    )


def _flat_tail(seconds: float, leads: tuple[int, ...]) -> Callable[[FloatArray], FloatArray]:
    def apply(signal: FloatArray) -> FloatArray:
        start = 1_000 - round(seconds * 100)
        signal[list(leads), start:] = 0.0
        return signal

    return apply


def _copy_lead(source: int, target: int) -> Callable[[FloatArray], FloatArray]:
    def apply(signal: FloatArray) -> FloatArray:
        signal[target] = signal[source]
        return signal

    return apply


def _spike(signal: FloatArray) -> FloatArray:
    signal[V2, 500] += 6.0
    return signal


def _clip(signal: FloatArray) -> FloatArray:
    signal[V2] = np.clip(signal[V2], -0.05, 0.05)
    return signal


@dataclass(frozen=True, slots=True)
class StressCase:
    case_id: str
    description: str
    should_refuse: bool
    apply: Callable[[FloatArray], FloatArray]


CASES: tuple[StressCase, ...] = (
    StressCase("clean", "unaltered synthetic ECG", False, lambda signal: signal),
    StressCase("v2_flat_0_6s", "V2 flat for the last 0.6 s", True, _flat_tail(0.6, (V2,))),
    StressCase("v2_flat_3s", "V2 flat for the last 3 s", True, _flat_tail(3.0, (V2,))),
    StressCase("v2_flat_7s", "V2 flat for the last 7 s", True, _flat_tail(7.0, (V2,))),
    StressCase(
        "all_flat_5s", "all 12 leads flat for the last 5 s", True, _flat_tail(5.0, tuple(range(12)))
    ),
    StressCase("v2_unplugged", "V2 flat for the full 10 s", True, _flat_tail(10.0, (V2,))),
    StressCase("v5_copies_v3", "V3 waveform exported as V5", True, _copy_lead(8, 10)),
    StressCase("v2_spike", "single 6 mV spike in V2", True, _spike),
    StressCase("v2_clipped", "V2 clipped at +/-0.05 mV", True, _clip),
)


def evaluate() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in CASES:
        signal = case.apply(synthetic_ecg())
        row: dict[str, object] = {
            "case_id": case.case_id,
            "description": case.description,
            "should_refuse": case.should_refuse,
        }
        for name, config in PRESETS.items():
            report = assess_signal_quality(signal, SignalMetadata.canonical(config), config=config)
            row[name] = {
                "status": report.status.value,
                "classification_allowed": report.classification_allowed,
                "reason_codes": [code.value for code in report.reason_codes],
                "fail_open": case.should_refuse and report.classification_allowed,
            }
        rows.append(row)
    return rows


def _markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Case | Should refuse | v1 | v2 |",
        "|---|---|---|---|",
    ]
    for row in rows:
        cells = []
        for name in PRESETS:
            result = row[name]
            assert isinstance(result, dict)
            marker = " (fail-open)" if result["fail_open"] else ""
            cells.append(f"{result['status']}{marker}")
        lines.append(
            f"| {row['description']} | {'yes' if row['should_refuse'] else 'no'} | "
            f"{cells[0]} | {cells[1]} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args(argv)
    rows = evaluate()
    if args.format == "json":
        print(json.dumps({"presets": list(PRESETS), "cases": rows}, indent=2))
    else:
        print(_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
