"""Fail-closed, research-only HTTP boundary for ECG trust decisions.

The service intentionally accepts opaque identifiers rather than waveforms or file
locations.  Backend implementations are injected behind small protocols so this
module can enforce one public contract regardless of how authorized cases and
verified releases are stored.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import Annotated, Final, Literal, Protocol, TypedDict, cast

from fastapi import FastAPI, Header, Path, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from ecg_trust.contracts import TrustDecision

API_PREFIX: Final = "/api/v1"
RESEARCH_ONLY_BOUNDARY: Final = (
    "Research use only. Not for diagnosis, treatment, or clinical decision-making."
)
OPAQUE_ID_PATTERN: Final = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"
IDEMPOTENCY_KEY_PATTERN: Final = r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$"
LABEL_PATTERN: Final = re.compile(r"^[A-Z][A-Z0-9_]{0,31}$")
SHA256_PATTERN: Final = re.compile(r"^[a-f0-9]{64}$")
SERVICE_VERSION_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")
OPAQUE_ID_RE: Final = re.compile(OPAQUE_ID_PATTERN)

OpaqueId = Annotated[
    str,
    StringConstraints(
        strict=True,
        strip_whitespace=False,
        min_length=1,
        max_length=64,
        pattern=OPAQUE_ID_PATTERN,
    ),
]
ReleaseIdPath = Annotated[
    str,
    Path(min_length=1, max_length=64, pattern=OPAQUE_ID_PATTERN),
]
IdempotencyKeyHeader = Annotated[
    str,
    Header(
        alias="Idempotency-Key",
        min_length=8,
        max_length=128,
        pattern=IDEMPOTENCY_KEY_PATTERN,
    ),
]


class ReasonCode(StrEnum):
    """Stable, non-sensitive reason codes safe to expose to API consumers."""

    RELEASE_INTEGRITY_UNVERIFIED = "RELEASE_INTEGRITY_UNVERIFIED"
    INPUT_CONTRACT_INVALID = "INPUT_CONTRACT_INVALID"
    QUALITY_COMPONENT_UNAVAILABLE = "QUALITY_COMPONENT_UNAVAILABLE"
    SIGNAL_QUALITY_INVALID = "SIGNAL_QUALITY_INVALID"
    SIGNAL_REACQUISITION_REQUIRED = "SIGNAL_REACQUISITION_REQUIRED"
    DISTRIBUTION_COMPONENT_UNAVAILABLE = "DISTRIBUTION_COMPONENT_UNAVAILABLE"
    OUTSIDE_VALIDATED_DISTRIBUTION = "OUTSIDE_VALIDATED_DISTRIBUTION"
    UNCERTAINTY_COMPONENT_UNAVAILABLE = "UNCERTAINTY_COMPONENT_UNAVAILABLE"
    LEGACY_ENTROPY_GATE_REJECTED = "LEGACY_ENTROPY_GATE_REJECTED"
    CONFORMAL_SET_UNCERTAIN = "CONFORMAL_SET_UNCERTAIN"
    CONFIDENCE_GATE_ABSTAINED = "CONFIDENCE_GATE_ABSTAINED"
    ALL_TRUST_GATES_PASSED = "ALL_TRUST_GATES_PASSED"
    RELEASE_NOT_READY = "RELEASE_NOT_READY"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"


class CaseKind(StrEnum):
    """Permitted provenance classes; arbitrary public uploads are out of scope."""

    AUTHORIZED = "AUTHORIZED"
    SYNTHETIC = "SYNTHETIC"


def _is_opaque_id(value: str) -> bool:
    return OPAQUE_ID_RE.fullmatch(value) is not None


@dataclass(frozen=True, slots=True)
class ResolvedCase:
    """Internal authorized case handle; ``analysis_handle`` is never serialized."""

    case_id: str
    kind: CaseKind
    analysis_handle: object

    def __post_init__(self) -> None:
        if not _is_opaque_id(self.case_id):
            raise ValueError("case_id must be an opaque identifier")
        if not isinstance(self.kind, CaseKind):
            raise ValueError("kind must identify an authorized or synthetic case")


@dataclass(frozen=True, slots=True)
class VerifiedRelease:
    """Minimal safe release metadata returned by a verification boundary."""

    release_id: str
    artifact_sha256: str
    verified: bool = True
    locked: bool = True

    def __post_init__(self) -> None:
        if not _is_opaque_id(self.release_id):
            raise ValueError("release_id must be an opaque identifier")
        if SHA256_PATTERN.fullmatch(self.artifact_sha256) is None:
            raise ValueError("artifact_sha256 must be 64 lowercase hexadecimal characters")


@dataclass(frozen=True, slots=True)
class AnalysisOutcome:
    """Validated engine result with a structural no-leakage invariant."""

    decision: TrustDecision
    reason_codes: tuple[ReasonCode, ...]
    labels: tuple[str, ...] | None = None
    probabilities: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, TrustDecision):
            raise ValueError("decision must use the closed trust-decision vocabulary")
        if not self.reason_codes or len(self.reason_codes) > 16:
            raise ValueError("one to sixteen reason codes are required")
        if any(not isinstance(reason, ReasonCode) for reason in self.reason_codes):
            raise ValueError("reason codes must use the closed reason vocabulary")
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise ValueError("reason codes must be unique")

        if self.decision is not TrustDecision.PREDICTION_ALLOWED:
            if self.labels is not None or self.probabilities is not None:
                raise ValueError("non-allowed outcomes cannot contain prediction results")
            return

        if not self.labels or not self.probabilities:
            raise ValueError("allowed outcomes require labels and probabilities")
        if len(self.labels) != len(self.probabilities) or len(self.labels) > 71:
            raise ValueError("labels and probabilities must have matching bounded lengths")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be unique")
        if any(LABEL_PATTERN.fullmatch(label) is None for label in self.labels):
            raise ValueError("labels must use the bounded research-label vocabulary format")
        if any(
            isinstance(probability, bool)
            or not isinstance(probability, (float, int))
            or not math.isfinite(float(probability))
            or not 0.0 <= float(probability) <= 1.0
            for probability in self.probabilities
        ):
            raise ValueError("probabilities must be finite values between zero and one")


class CaseResolver(Protocol):
    """Resolves only pre-authorized or explicitly synthetic opaque cases."""

    def is_ready(self) -> bool: ...

    def resolve_case(self, case_id: str) -> ResolvedCase: ...


class VerifiedReleaseProvider(Protocol):
    """Provides immutable release records that passed an external verification gate."""

    def is_ready(self) -> bool: ...

    def get_active_release(self) -> VerifiedRelease: ...

    def get_release(self, release_id: str) -> VerifiedRelease: ...


class AnalysisEngine(Protocol):
    """Runs contract validation and gated research inference."""

    def is_ready(self) -> bool: ...

    def is_ready_for_release(self, release: VerifiedRelease) -> bool: ...

    def validate_case(self, case: ResolvedCase, release: VerifiedRelease) -> AnalysisOutcome: ...

    def infer(self, case: ResolvedCase, release: VerifiedRelease) -> AnalysisOutcome: ...


class ResourceNotFoundError(LookupError):
    """Expected private-backend miss, translated to a generic public 404."""


class DependencyUnavailableError(RuntimeError):
    """Expected dependency outage, translated to a fail-closed response."""


class _ReleaseNotReadyError(RuntimeError):
    pass


class _IdempotencyConflictError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SentinelServiceConfig:
    """Validated immutable operational limits for the public boundary."""

    service_version: str = "sentinel-v1"
    max_idempotency_entries: int = 1024

    def __post_init__(self) -> None:
        if SERVICE_VERSION_PATTERN.fullmatch(self.service_version) is None:
            raise ValueError("service_version must be a bounded machine identifier")
        if not 1 <= self.max_idempotency_entries <= 100_000:
            raise ValueError("max_idempotency_entries must be between 1 and 100000")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class _BoundaryModel(_StrictModel):
    api_version: Literal["v1"] = "v1"
    service_version: str = Field(min_length=1, max_length=32)
    research_only: Literal[True] = True
    boundary: str = Field(default=RESEARCH_ONLY_BOUNDARY, max_length=160)


class CaseRequest(_StrictModel):
    case_id: OpaqueId
    release_id: OpaqueId


class HealthResponse(_BoundaryModel):
    status: Literal["alive"] = "alive"


class ReadinessResponse(_BoundaryModel):
    status: Literal["ready", "not_ready"]
    reason_codes: tuple[ReasonCode, ...] = ()


class ReleaseResponse(_BoundaryModel):
    release_id: OpaqueId
    artifact_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")
    verified: Literal[True] = True
    locked: Literal[True] = True


class CaseValidationResponse(_BoundaryModel):
    case_id: OpaqueId
    release_id: OpaqueId
    decision: TrustDecision
    reason_codes: tuple[ReasonCode, ...] = Field(min_length=1, max_length=16)


class InferenceResponse(CaseValidationResponse):
    labels: tuple[str, ...] | None = None
    probabilities: tuple[float, ...] | None = None

    @model_validator(mode="after")
    def enforce_result_disclosure_gate(self) -> InferenceResponse:
        if self.decision is TrustDecision.PREDICTION_ALLOWED:
            if not self.labels or not self.probabilities:
                raise ValueError("prediction results are required when prediction is allowed")
            if len(self.labels) != len(self.probabilities):
                raise ValueError("prediction result lengths must match")
        elif self.labels is not None or self.probabilities is not None:
            raise ValueError("prediction results are forbidden unless prediction is allowed")
        return self


class ErrorDetail(_StrictModel):
    code: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    message: str = Field(min_length=1, max_length=120)


class ErrorResponse(_BoundaryModel):
    error: ErrorDetail


@dataclass(frozen=True, slots=True)
class _ServiceResult:
    content: dict[str, object]
    status_code: int
    cacheable: bool = False


@dataclass(frozen=True, slots=True)
class _StoredInference:
    fingerprint: str
    result: _ServiceResult


class _InMemoryIdempotencyStore:
    """Process-local, bounded replay store containing only already-safe JSON."""

    def __init__(self, max_entries: int) -> None:
        self._max_entries = max_entries
        self._entries: OrderedDict[str, _StoredInference] = OrderedDict()
        self._lock = RLock()

    def execute(
        self,
        *,
        key: str,
        fingerprint: str,
        operation: Callable[[], _ServiceResult],
    ) -> tuple[_ServiceResult, bool]:
        # The lock deliberately spans the operation: concurrent duplicate keys execute once.
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                if existing.fingerprint != fingerprint:
                    raise _IdempotencyConflictError
                self._entries.move_to_end(key)
                return existing.result, True

            result = operation()
            if result.cacheable:
                self._entries[key] = _StoredInference(fingerprint=fingerprint, result=result)
                while len(self._entries) > self._max_entries:
                    self._entries.popitem(last=False)
            return result, False


class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        return response


def _dump(model: BaseModel) -> dict[str, object]:
    return cast(dict[str, object], model.model_dump(mode="json", exclude_none=True))


class _BoundaryFields(TypedDict):
    api_version: Literal["v1"]
    service_version: str
    research_only: Literal[True]
    boundary: str


def _boundary_fields(config: SentinelServiceConfig) -> _BoundaryFields:
    return {
        "api_version": "v1",
        "service_version": config.service_version,
        "research_only": True,
        "boundary": RESEARCH_ONLY_BOUNDARY,
    }


def _error_result(
    config: SentinelServiceConfig,
    *,
    status_code: int,
    code: str,
    message: str,
) -> _ServiceResult:
    response = ErrorResponse(
        **_boundary_fields(config),
        error=ErrorDetail(code=code, message=message),
    )
    return _ServiceResult(content=_dump(response), status_code=status_code)


def _decision_result(
    config: SentinelServiceConfig,
    *,
    case_id: str,
    release_id: str,
    outcome: AnalysisOutcome,
    include_results: bool,
    status_code: int = 200,
) -> _ServiceResult:
    if include_results:
        response: BaseModel = InferenceResponse(
            **_boundary_fields(config),
            case_id=case_id,
            release_id=release_id,
            decision=outcome.decision,
            reason_codes=outcome.reason_codes,
            labels=outcome.labels,
            probabilities=outcome.probabilities,
        )
    else:
        response = CaseValidationResponse(
            **_boundary_fields(config),
            case_id=case_id,
            release_id=release_id,
            decision=outcome.decision,
            reason_codes=outcome.reason_codes,
        )
    return _ServiceResult(
        content=_dump(response),
        status_code=status_code,
        cacheable=status_code == 200,
    )


def _fail_closed_result(
    config: SentinelServiceConfig,
    *,
    case_id: str,
    release_id: str,
    reason: ReasonCode,
) -> _ServiceResult:
    outcome = AnalysisOutcome(
        decision=TrustDecision.ABSTAIN,
        reason_codes=(reason,),
    )
    return _decision_result(
        config,
        case_id=case_id,
        release_id=release_id,
        outcome=outcome,
        include_results=False,
        status_code=503,
    )


def _json_response(result: _ServiceResult, *, replayed: bool | None = None) -> JSONResponse:
    headers: dict[str, str] = {}
    if replayed is not None:
        headers["Idempotency-Replayed"] = "true" if replayed else "false"
    return JSONResponse(
        content=result.content,
        status_code=result.status_code,
        headers=headers,
    )


def _dependency_is_ready(dependency: object | None) -> bool:
    if dependency is None:
        return False
    try:
        ready_method = cast(CaseResolver, dependency).is_ready
        return ready_method() is True
    except Exception:
        return False


def _release_and_engine_are_coherently_ready(
    provider: VerifiedReleaseProvider | None,
    engine: AnalysisEngine | None,
) -> bool:
    if provider is None or engine is None:
        return False
    try:
        release = provider.get_active_release()
        return (
            isinstance(release, VerifiedRelease)
            and release.verified
            and release.locked
            and engine.is_ready_for_release(release) is True
        )
    except Exception:
        return False


def _resolve_case(resolver: CaseResolver | None, case_id: str) -> ResolvedCase:
    if resolver is None:
        raise DependencyUnavailableError
    try:
        case = resolver.resolve_case(case_id)
    except ResourceNotFoundError:
        raise
    except Exception:
        raise DependencyUnavailableError from None
    if not isinstance(case, ResolvedCase) or case.case_id != case_id:
        raise DependencyUnavailableError
    return case


def _resolve_release(
    provider: VerifiedReleaseProvider | None,
    release_id: str,
) -> VerifiedRelease:
    if provider is None:
        raise DependencyUnavailableError
    try:
        release = provider.get_release(release_id)
    except ResourceNotFoundError:
        raise
    except Exception:
        raise DependencyUnavailableError from None
    if not isinstance(release, VerifiedRelease) or release.release_id != release_id:
        raise DependencyUnavailableError
    if not release.verified or not release.locked:
        raise _ReleaseNotReadyError
    return release


def _analyze(
    engine: AnalysisEngine | None,
    *,
    case: ResolvedCase,
    release: VerifiedRelease,
    inference: bool,
) -> AnalysisOutcome:
    if engine is None:
        raise DependencyUnavailableError
    try:
        outcome = engine.infer(case, release) if inference else engine.validate_case(case, release)
    except Exception:
        raise DependencyUnavailableError from None
    if not isinstance(outcome, AnalysisOutcome):
        raise DependencyUnavailableError
    return outcome


def _run_analysis(
    config: SentinelServiceConfig,
    *,
    resolver: CaseResolver | None,
    provider: VerifiedReleaseProvider | None,
    engine: AnalysisEngine | None,
    request_body: CaseRequest,
    inference: bool,
) -> _ServiceResult:
    try:
        case = _resolve_case(resolver, request_body.case_id)
        release = _resolve_release(provider, request_body.release_id)
        outcome = _analyze(engine, case=case, release=release, inference=inference)
    except ResourceNotFoundError:
        return _error_result(
            config,
            status_code=404,
            code="resource_not_found",
            message="The requested resource was not found.",
        )
    except _ReleaseNotReadyError:
        return _fail_closed_result(
            config,
            case_id=request_body.case_id,
            release_id=request_body.release_id,
            reason=ReasonCode.RELEASE_NOT_READY,
        )
    except DependencyUnavailableError:
        return _fail_closed_result(
            config,
            case_id=request_body.case_id,
            release_id=request_body.release_id,
            reason=ReasonCode.BACKEND_UNAVAILABLE,
        )

    try:
        return _decision_result(
            config,
            case_id=request_body.case_id,
            release_id=request_body.release_id,
            outcome=outcome,
            include_results=inference,
        )
    except Exception:
        return _fail_closed_result(
            config,
            case_id=request_body.case_id,
            release_id=request_body.release_id,
            reason=ReasonCode.BACKEND_UNAVAILABLE,
        )


def create_sentinel_app(
    *,
    case_resolver: CaseResolver | None = None,
    release_provider: VerifiedReleaseProvider | None = None,
    analysis_engine: AnalysisEngine | None = None,
    config: SentinelServiceConfig | None = None,
) -> FastAPI:
    """Create the versioned Sentinel app with explicitly injected dependencies."""

    effective_config = config or SentinelServiceConfig()
    idempotency_store = _InMemoryIdempotencyStore(effective_config.max_idempotency_entries)
    app = FastAPI(
        title="ECG Sentinel Research Service",
        version=effective_config.service_version,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(_SecurityHeadersMiddleware)

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        del request, exc
        return _json_response(
            _error_result(
                effective_config,
                status_code=422,
                code="request_validation_failed",
                message="The request does not match the API contract.",
            )
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        del request
        if exc.status_code == 404:
            code, message = "resource_not_found", "The requested resource was not found."
        elif exc.status_code == 405:
            code, message = "method_not_allowed", "The request method is not allowed."
        else:
            code, message = "request_rejected", "The request was rejected."
        return _json_response(
            _error_result(
                effective_config,
                status_code=exc.status_code,
                code=code,
                message=message,
            )
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        del request, exc
        return _json_response(
            _error_result(
                effective_config,
                status_code=503,
                code="service_unavailable",
                message="The service is temporarily unavailable.",
            )
        )

    @app.get(f"{API_PREFIX}/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse(**_boundary_fields(effective_config))

    @app.get(
        f"{API_PREFIX}/readyz",
        response_model=ReadinessResponse,
        responses={503: {"model": ReadinessResponse}},
    )
    def readyz() -> JSONResponse:
        dependencies = (case_resolver, release_provider, analysis_engine)
        ready = all(_dependency_is_ready(dependency) for dependency in dependencies)
        if ready:
            ready = _release_and_engine_are_coherently_ready(
                release_provider,
                analysis_engine,
            )
        response = ReadinessResponse(
            **_boundary_fields(effective_config),
            status="ready" if ready else "not_ready",
            reason_codes=() if ready else (ReasonCode.BACKEND_UNAVAILABLE,),
        )
        return JSONResponse(
            content=_dump(response),
            status_code=200 if ready else 503,
        )

    @app.get(
        f"{API_PREFIX}/releases/{{release_id}}",
        response_model=ReleaseResponse,
        responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    def get_release(release_id: ReleaseIdPath) -> JSONResponse:
        try:
            release = _resolve_release(release_provider, release_id)
        except ResourceNotFoundError:
            return _json_response(
                _error_result(
                    effective_config,
                    status_code=404,
                    code="resource_not_found",
                    message="The requested resource was not found.",
                )
            )
        except (_ReleaseNotReadyError, DependencyUnavailableError):
            return _json_response(
                _error_result(
                    effective_config,
                    status_code=503,
                    code="release_unavailable",
                    message="The verified release is unavailable.",
                )
            )
        response = ReleaseResponse(
            **_boundary_fields(effective_config),
            release_id=release.release_id,
            artifact_sha256=release.artifact_sha256,
        )
        return JSONResponse(content=_dump(response), status_code=200)

    @app.post(
        f"{API_PREFIX}/cases:validate",
        response_model=CaseValidationResponse,
        response_model_exclude_none=True,
        responses={404: {"model": ErrorResponse}, 503: {"model": CaseValidationResponse}},
    )
    def validate_case(request_body: CaseRequest) -> JSONResponse:
        result = _run_analysis(
            effective_config,
            resolver=case_resolver,
            provider=release_provider,
            engine=analysis_engine,
            request_body=request_body,
            inference=False,
        )
        return _json_response(result)

    @app.post(
        f"{API_PREFIX}/inferences",
        response_model=InferenceResponse,
        response_model_exclude_none=True,
        responses={
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            503: {"model": InferenceResponse},
        },
    )
    def create_inference(
        request_body: CaseRequest,
        idempotency_key: IdempotencyKeyHeader,
    ) -> JSONResponse:
        fingerprint = hashlib.sha256(
            f"{request_body.case_id}\0{request_body.release_id}".encode("ascii")
        ).hexdigest()
        try:
            result, replayed = idempotency_store.execute(
                key=idempotency_key,
                fingerprint=fingerprint,
                operation=lambda: _run_analysis(
                    effective_config,
                    resolver=case_resolver,
                    provider=release_provider,
                    engine=analysis_engine,
                    request_body=request_body,
                    inference=True,
                ),
            )
        except _IdempotencyConflictError:
            result = _error_result(
                effective_config,
                status_code=409,
                code="idempotency_key_conflict",
                message="The idempotency key was already used for another request.",
            )
            replayed = False
        return _json_response(result, replayed=replayed)

    return app
