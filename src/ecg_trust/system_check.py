"""Verify that the local Python, PyTorch, and NVIDIA stack can train ECG models."""

from __future__ import annotations

import json
import platform
import sys
import time
from typing import Any, cast

import torch
from torch import Tensor, nn


class CudaSmokeModel(nn.Module):
    """Small Conv1D/Transformer model exercising both planned architecture families."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=15, stride=5, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(128, 5)

    def forward(self, signals: Tensor) -> Tensor:
        features = self.stem(signals).transpose(1, 2)
        features = self.encoder(features)
        return cast(Tensor, self.head(features.mean(dim=1)))


def _device_report() -> dict[str, Any]:
    available = torch.cuda.is_available()
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": available,
        "cudnn": torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
    }
    if not available:
        return report

    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    report.update(
        {
            "device_index": index,
            "device_name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "vram_gib": round(properties.total_memory / 1024**3, 2),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    )
    return report


def _training_smoke_test() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to silently pass on CPU.")

    device = torch.device("cuda")
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    model = CudaSmokeModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    signals = torch.randn(16, 12, 1000, device=device)
    targets = torch.randint(0, 2, (16, 5), device=device, dtype=torch.float32)

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    losses: list[float] = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(signals)
            loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    if not gradients_finite or not all(torch.isfinite(torch.tensor(losses))):
        raise RuntimeError("Non-finite value detected during CUDA training smoke test.")

    return {
        "autocast_dtype": str(amp_dtype).removeprefix("torch."),
        "batch_shape": list(signals.shape),
        "output_shape": list(logits.shape),
        "steps": len(losses),
        "final_loss": round(losses[-1], 6),
        "elapsed_seconds": round(elapsed, 3),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "gradients_finite": gradients_finite,
    }


def main() -> None:
    """Print a machine-readable report and fail if real CUDA training does not work."""
    report: dict[str, Any] = {"environment": _device_report()}
    try:
        report["training_smoke_test"] = _training_smoke_test()
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        print(json.dumps(report, indent=2))
        raise SystemExit(1) from error

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
