#!/usr/bin/env python3
"""Benchmark ECG models with synthetic inputs; no PTB-XL access is performed."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from ecg_trust.benchmark import (
    MODEL_SPECS,
    Precision,
    benchmark_train_steps,
    environment_metadata,
    probe_safe_batch_size,
    selected_model_specs,
)
from ecg_trust.models import MATCHED_CAPACITY_PRESET

DEFAULT_MODELS: tuple[str, ...] = tuple(MODEL_SPECS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=DEFAULT_MODELS,
        default=list(DEFAULT_MODELS),
        help="model variants to benchmark (default: all)",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp32"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--probe-max-batch",
        type=int,
        default=None,
        help="optionally probe powers of two up to this inclusive CUDA batch",
    )
    parser.add_argument(
        "--cpu-smoke",
        action="store_true",
        help="force CPU/FP32 with batch=1, no warmup, and one measured step",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "artifacts" / "benchmarks" / "models.json",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps cannot be negative")
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.probe_max_batch is not None and args.probe_max_batch < args.batch_size:
        parser.error("--probe-max-batch must be at least --batch-size")
    return args


def _resolve_device(requested: str, *, cpu_smoke: bool) -> torch.device:
    if cpu_smoke or requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_precision(requested: str, device: torch.device, *, cpu_smoke: bool) -> Precision:
    if cpu_smoke:
        return "fp32"
    if requested == "auto":
        return "bf16" if device.type == "cuda" else "fp32"
    if requested == "bf16" and device.type != "cuda":
        raise ValueError("BF16 requires CUDA; use --precision fp32 for CPU")
    return "bf16" if requested == "bf16" else "fp32"


def _write_json(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, output_path)


def run(args: argparse.Namespace) -> dict[str, object]:
    device = _resolve_device(args.device, cpu_smoke=args.cpu_smoke)
    precision = _resolve_precision(args.precision, device, cpu_smoke=args.cpu_smoke)
    batch_size = 1 if args.cpu_smoke else args.batch_size
    warmup_steps = 0 if args.cpu_smoke else args.warmup_steps
    measured_steps = 1 if args.cpu_smoke else args.steps
    if args.probe_max_batch is not None and device.type != "cuda":
        raise ValueError("--probe-max-batch requires a CUDA benchmark")

    specs = selected_model_specs(args.models)
    results: list[dict[str, object]] = []
    probes: dict[str, object] = {}
    for spec in specs:
        effective_batch_size = batch_size
        if args.probe_max_batch is not None:
            probe = probe_safe_batch_size(
                spec.build,
                model_name=spec.name,
                device=device,
                maximum_batch_size=args.probe_max_batch,
                precision=precision,
                seed=args.seed,
            )
            probes[spec.name] = probe.to_dict()
            if probe.maximum_successful_batch is None:
                raise RuntimeError(f"{spec.name} could not train even at batch size 1")
            effective_batch_size = min(batch_size, probe.maximum_successful_batch)

        model = spec.build()
        try:
            result = benchmark_train_steps(
                model,
                model_name=spec.name,
                device=device,
                batch_size=effective_batch_size,
                warmup_steps=warmup_steps,
                measured_steps=measured_steps,
                precision=precision,
                seed=args.seed,
            )
        finally:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        result_payload = result.to_dict()
        result_payload["specification"] = spec.metadata()
        results.append(result_payload)
        print(
            f"{spec.name}: batch={effective_batch_size}, "
            f"median={result.median_step_latency_ms:.2f} ms, "
            f"throughput={result.throughput_samples_per_second:.1f} samples/s, "
            f"peak_allocated={result.peak_allocated_vram_mib} MiB"
        )

    return {
        "schema_version": "1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "workload": {
            "kind": "synthetic_train_step",
            "input_shape": [batch_size, 12, 1000],
            "target_shape": [batch_size, 5],
            "optimizer": "AdamW",
            "loss": "BCEWithLogitsLoss",
            "requested_batch_size": batch_size,
            "warmup_steps": warmup_steps,
            "measured_steps": measured_steps,
            "precision": precision,
            "seed": args.seed,
            "dataset_accessed": False,
        },
        "environment": environment_metadata(device),
        "matched_capacity_preset": MATCHED_CAPACITY_PRESET.metadata(),
        "batch_probes": probes,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = run(args)
        _write_json(payload, args.output)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"benchmark JSON: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
