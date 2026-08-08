"""FastAPI adapter for the provenance-checked ECG research demo."""

from __future__ import annotations

import asyncio
import json
import math
import re
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Protocol, cast

import anyio
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from plotly.offline import get_plotlyjs  # type: ignore[import-untyped]
from torch import Tensor

from ecg_trust.constants import LEADS, SUPERCLASSES
from ecg_trust.demo_backend import (
    LIMITATIONS,
    RESEARCH_ONLY_NOTICE,
    AttributionMethod,
    DemoArtifactError,
    DemoInputError,
    DemoPrediction,
    FrozenDecisionPolicy,
    load_wfdb_physical_signal,
)

_SAFE_UPLOAD_STEM = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}\Z")
_SAFE_EXAMPLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


class DemoWebError(ValueError):
    """Raised when web-layer configuration or an upload is unsafe."""


class InferenceBackend(Protocol):
    """Narrow interface used by the web adapter and its endpoint tests."""

    policy: FrozenDecisionPolicy
    artifact_provenance: Mapping[str, object]

    def predict_record(
        self,
        record_path: str | Path,
        *,
        attribution_method: AttributionMethod | None = None,
        attribution_target: str | int | None = None,
        integrated_gradients_steps: int = 32,
    ) -> DemoPrediction: ...


@dataclass(frozen=True, slots=True)
class DemoExample:
    example_id: str
    label: str
    record_path: Path

    def __post_init__(self) -> None:
        if _SAFE_EXAMPLE_ID.fullmatch(self.example_id) is None:
            raise DemoWebError("example IDs must be short, path-free identifiers")
        if not self.label.strip():
            raise DemoWebError("example labels must be non-empty")

    def public_dict(self) -> dict[str, str]:
        return {"id": self.example_id, "label": self.label}


@dataclass(frozen=True, slots=True)
class DemoAppConfig:
    """Operational limits; none of these values changes model decisions."""

    max_header_bytes: int = 128 * 1024
    max_signal_bytes: int = 8 * 1024 * 1024
    integrated_gradients_steps: int = 32
    inference_concurrency: int = 1
    examples: tuple[DemoExample, ...] = ()

    def __post_init__(self) -> None:
        if self.max_header_bytes < 1024:
            raise DemoWebError("max_header_bytes must be at least 1024")
        if self.max_signal_bytes < 1024:
            raise DemoWebError("max_signal_bytes must be at least 1024")
        if not 2 <= self.integrated_gradients_steps <= 256:
            raise DemoWebError("integrated_gradients_steps must be in [2, 256]")
        if not 1 <= self.inference_concurrency <= 8:
            raise DemoWebError("inference_concurrency must be in [1, 8]")
        identifiers = [example.example_id for example in self.examples]
        if len(identifiers) != len(set(identifiers)):
            raise DemoWebError("example IDs must be unique")

    @classmethod
    def with_example_manifest(cls, path: str | Path, **kwargs: int) -> DemoAppConfig:
        """Load a local-only example registry without exposing record paths."""

        source = Path(path).resolve()
        try:
            decoded: object = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DemoWebError(f"could not load example manifest: {error}") from error
        if not isinstance(decoded, dict) or set(decoded) != {"examples"}:
            raise DemoWebError("example manifest must contain only an examples array")
        raw_examples = decoded["examples"]
        if not isinstance(raw_examples, list):
            raise DemoWebError("example manifest examples must be an array")
        examples: list[DemoExample] = []
        for index, value in enumerate(raw_examples):
            if not isinstance(value, dict) or set(value) != {"id", "label", "record_path"}:
                raise DemoWebError(f"example {index} has invalid keys")
            if not all(isinstance(value[key], str) for key in value):
                raise DemoWebError(f"example {index} values must be strings")
            raw_path = Path(cast(str, value["record_path"]))
            record_path = (
                raw_path.resolve()
                if raw_path.is_absolute()
                else (source.parent / raw_path).resolve()
            )
            header_path = record_path.with_suffix(".hea")
            if not header_path.is_file():
                raise DemoWebError(f"example {index} header does not exist")
            examples.append(
                DemoExample(
                    example_id=cast(str, value["id"]),
                    label=cast(str, value["label"]),
                    record_path=record_path.with_suffix(""),
                )
            )
        return cls(examples=tuple(examples), **kwargs)


def _validated_attribution(value: str | None) -> AttributionMethod | None:
    if value in {None, "", "none"}:
        return None
    if value not in {"grad_cam", "integrated_gradients"}:
        raise DemoWebError("unsupported attribution method")
    return cast(AttributionMethod, value)


def _validated_target(value: str | None) -> str | None:
    if value in {None, ""}:
        return None
    if value not in SUPERCLASSES:
        raise DemoWebError("attribution target is not a canonical label")
    return value


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


def _waveform_payload(signal: Tensor) -> dict[str, object]:
    array = signal.tolist()
    return {
        "lead_order": list(LEADS),
        "sampling_frequency_hz": 100.0,
        "units": "mV",
        "samples": 1000,
        "time_seconds": [index / 100.0 for index in range(1000)],
        "values": array,
    }


def _sanitize_number(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        raise DemoArtifactError("prediction response contains a non-finite value")
    if isinstance(value, dict):
        return {str(key): _sanitize_number(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_number(item) for item in value]
    return value


def _run_prediction(
    backend: InferenceBackend,
    record_path: Path,
    *,
    public_source: str,
    attribution_method: AttributionMethod | None,
    attribution_target: str | None,
    integrated_gradients_steps: int,
) -> dict[str, object]:
    signal = load_wfdb_physical_signal(record_path)
    prediction = backend.predict_record(
        record_path,
        attribution_method=attribution_method,
        attribution_target=attribution_target,
        integrated_gradients_steps=integrated_gradients_steps,
    )
    payload = prediction.to_dict()
    payload["source"] = public_source
    payload["waveform"] = _waveform_payload(signal)
    return cast(dict[str, object], _sanitize_number(payload))


def _backend_or_503(app: FastAPI) -> InferenceBackend:
    backend = cast(InferenceBackend | None, getattr(app.state, "backend", None))
    if backend is None:
        raise HTTPException(status_code=503, detail="model artifacts are not loaded")
    return backend


def create_app(
    *,
    backend: InferenceBackend | None = None,
    config: DemoAppConfig | None = None,
) -> FastAPI:
    """Create an app; passing a backend keeps startup explicit and testable."""

    settings = DemoAppConfig() if config is None else config
    app = FastAPI(
        title="PTB-XL Trustworthy ECG Classifier",
        version="0.1.0",
        description="Research-only five-superclass multi-label ECG demonstration.",
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.state.backend = backend
    app.state.config = settings
    app.state.inference_semaphore = asyncio.Semaphore(settings.inference_concurrency)
    examples = {example.example_id: example for example in settings.examples}
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

    @app.middleware("http")
    async def security_headers(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        )
        if request.url.path != "/assets/plotly.min.js":
            response.headers["Cache-Control"] = "no-store"
            response.headers["Pragma"] = "no-cache"
        return response

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> Response:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "labels": SUPERCLASSES,
                "leads": LEADS,
                "notice": RESEARCH_ONLY_NOTICE,
            },
        )

    @app.get("/assets/plotly.min.js", include_in_schema=False)
    async def plotly_javascript() -> Response:
        return Response(
            content=cast(str, get_plotlyjs()),
            media_type="application/javascript",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    @app.get("/health")
    async def health() -> JSONResponse:
        ready = app.state.backend is not None
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"status": "ready" if ready else "not_ready", "model_loaded": ready},
        )

    @app.get("/metadata")
    async def metadata() -> dict[str, object]:
        loaded = _backend_or_503(app)
        return {
            "task": "five-superclass multi-label ECG classification",
            "label_order": list(SUPERCLASSES),
            "input": {
                "format": "matched WFDB .hea/.dat pair",
                "lead_order": list(LEADS),
                "sampling_frequency_hz": 100,
                "duration_seconds": 10,
                "samples_per_lead": 1000,
                "units": "mV",
            },
            "decision_policy": {
                "temperature": loaded.policy.temperature,
                "classification_thresholds": list(
                    loaded.policy.classification_thresholds
                ),
                "uncertainty_threshold": loaded.policy.uncertainty_threshold,
                "calibration_folds": list(loaded.policy.provenance.calibration_folds),
            },
            "artifact_provenance": dict(loaded.artifact_provenance),
            "safety": {"notice": RESEARCH_ONLY_NOTICE, "limitations": list(LIMITATIONS)},
        }

    @app.get("/examples")
    async def list_examples() -> dict[str, object]:
        return {"examples": [example.public_dict() for example in settings.examples]}

    @app.post("/predict")
    async def predict_upload(
        header: Annotated[UploadFile, File(description="WFDB header (.hea)")],
        signal: Annotated[UploadFile, File(description="WFDB signal (.dat)")],
        attribution_method: Annotated[str | None, Form()] = None,
        attribution_target: Annotated[str | None, Form()] = None,
    ) -> dict[str, object]:
        loaded = _backend_or_503(app)
        try:
            stem = _validated_upload_names(header, signal)
            method = _validated_attribution(attribution_method)
            target = _validated_target(attribution_target)
            header_bytes, signal_bytes = await asyncio.gather(
                _read_limited(header, settings.max_header_bytes, "WFDB header"),
                _read_limited(signal, settings.max_signal_bytes, "WFDB signal"),
            )
            _validate_uploaded_header(header_bytes, stem)
            with tempfile.TemporaryDirectory(prefix="ecg-demo-") as temporary_name:
                directory = Path(temporary_name)
                record = directory / stem
                record.with_suffix(".hea").write_bytes(header_bytes)
                record.with_suffix(".dat").write_bytes(signal_bytes)
                async with app.state.inference_semaphore:
                    return await anyio.to_thread.run_sync(
                        lambda: _run_prediction(
                            loaded,
                            record,
                            public_source=f"upload:{stem}",
                            attribution_method=method,
                            attribution_target=target,
                            integrated_gradients_steps=settings.integrated_gradients_steps,
                        )
                    )
        except DemoWebError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except DemoInputError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except DemoArtifactError as error:
            raise HTTPException(
                status_code=503, detail="model artifact validation failed"
            ) from error
        finally:
            await header.close()
            await signal.close()

    @app.post("/predict/example/{example_id}")
    async def predict_example(
        example_id: str,
        attribution_method: Annotated[str | None, Form()] = None,
        attribution_target: Annotated[str | None, Form()] = None,
    ) -> dict[str, object]:
        loaded = _backend_or_503(app)
        example = examples.get(example_id)
        if example is None:
            raise HTTPException(status_code=404, detail="unknown example ID")
        try:
            method = _validated_attribution(attribution_method)
            target = _validated_target(attribution_target)
            async with app.state.inference_semaphore:
                return await anyio.to_thread.run_sync(
                    lambda: _run_prediction(
                        loaded,
                        example.record_path,
                        public_source=f"example:{example.example_id}",
                        attribution_method=method,
                        attribution_target=target,
                        integrated_gradients_steps=settings.integrated_gradients_steps,
                    )
                )
        except DemoWebError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except DemoInputError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except DemoArtifactError as error:
            raise HTTPException(
                status_code=503, detail="model artifact validation failed"
            ) from error

    return app


app = create_app()


__all__ = [
    "DemoAppConfig",
    "DemoExample",
    "DemoWebError",
    "InferenceBackend",
    "app",
    "create_app",
]
