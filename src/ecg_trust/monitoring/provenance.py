"""Content-bound monitoring replay without changing the v1 telemetry schema."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Self

from ecg_trust.monitoring.trust_monitoring import (
    AggregateTelemetryWindow,
    MonitoringComparison,
    TelemetryValidationError,
    TrustMonitoringConfig,
    _require_float,
    _require_int,
    _require_mapping,
    _require_string,
    compare_telemetry_windows,
    replay_against_frozen_reference,
)

REFERENCE_SCHEMA = "ecg_trust.frozen_monitoring_reference.v1"
REPLAY_SCHEMA = "ecg_trust.bound_monitoring_replay.v1"
MAX_REFERENCE_BYTES = 1024 * 1024


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def monitoring_config_to_dict(config: TrustMonitoringConfig) -> dict[str, object]:
    """Export every threshold, evidence requirement, bin edge, and version."""

    if not isinstance(config, TrustMonitoringConfig):
        raise TelemetryValidationError("monitoring configuration is required")
    result = asdict(config)
    result["score_bin_edges"] = list(config.score_bin_edges)
    return result


def monitoring_config_from_dict(payload: Mapping[str, object]) -> TrustMonitoringConfig:
    """Read an exact, complete configuration; omitted thresholds are never defaulted."""

    if set(payload) != {field.name for field in fields(TrustMonitoringConfig)}:
        raise TelemetryValidationError("monitoring configuration fields do not match the contract")
    edges = payload["score_bin_edges"]
    if not isinstance(edges, (list, tuple)):
        raise TelemetryValidationError("score_bin_edges must be an array")
    return TrustMonitoringConfig(
        version=_require_string(payload["version"], "version"),
        score_bin_edges=tuple(_require_float(value, "score_bin_edges") for value in edges),
        minimum_reference_samples=_require_int(
            payload["minimum_reference_samples"], "minimum_reference_samples", minimum=1
        ),
        minimum_window_samples=_require_int(
            payload["minimum_window_samples"], "minimum_window_samples", minimum=1
        ),
        minimum_score_samples=_require_int(
            payload["minimum_score_samples"], "minimum_score_samples", minimum=1
        ),
        rate_investigate_delta=_require_float(payload["rate_investigate_delta"], "rate delta"),
        rate_restrict_delta=_require_float(payload["rate_restrict_delta"], "rate delta"),
        rate_pause_delta=_require_float(payload["rate_pause_delta"], "rate delta"),
        rate_rollback_delta=_require_float(payload["rate_rollback_delta"], "rate delta"),
        psi_investigate=_require_float(payload["psi_investigate"], "PSI threshold"),
        psi_restrict=_require_float(payload["psi_restrict"], "PSI threshold"),
        psi_pause=_require_float(payload["psi_pause"], "PSI threshold"),
        psi_rollback=_require_float(payload["psi_rollback"], "PSI threshold"),
        psi_smoothing_count=_require_float(payload["psi_smoothing_count"], "PSI smoothing"),
    )


def monitoring_config_sha256(config: TrustMonitoringConfig) -> str:
    """Fingerprint normalized configuration values rather than a version label alone."""

    normalized = monitoring_config_from_dict(monitoring_config_to_dict(config))
    return _digest(monitoring_config_to_dict(normalized))


@dataclass(frozen=True, slots=True)
class FrozenMonitoringReference:
    """A validated configuration and aggregate reference sealed together for replay.

    Digests detect drift against a retained reference; they are not signatures or
    proof that a reference came from a trusted producer.
    """

    reference: AggregateTelemetryWindow
    config: TrustMonitoringConfig

    def __post_init__(self) -> None:
        if not isinstance(self.reference, AggregateTelemetryWindow):
            raise TelemetryValidationError("an aggregate monitoring reference is required")
        config = monitoring_config_from_dict(monitoring_config_to_dict(self.config))
        reference = AggregateTelemetryWindow.from_dict(self.reference.to_dict())
        if reference.config_version != config.version:
            raise TelemetryValidationError("reference config_version does not match configuration")
        if reference.score_histogram.bin_edges != config.score_bin_edges:
            raise TelemetryValidationError("reference histogram does not use configured bins")
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "reference", reference)

    @property
    def config_sha256(self) -> str:
        return monitoring_config_sha256(self.config)

    @property
    def reference_sha256(self) -> str:
        return _digest(self.reference.to_dict())

    def _body(self) -> dict[str, object]:
        return {
            "schema_version": REFERENCE_SCHEMA,
            "config": monitoring_config_to_dict(self.config),
            "config_sha256": self.config_sha256,
            "reference": self.reference.to_dict(),
            "reference_sha256": self.reference_sha256,
        }

    @property
    def envelope_sha256(self) -> str:
        return _digest(self._body())

    def to_json(self) -> str:
        return _canonical_json({**self._body(), "envelope_sha256": self.envelope_sha256})

    @classmethod
    def from_json(cls, payload: str, *, expected_sha256: str | None = None) -> Self:
        """Load canonical JSON and optionally bind it to a separately retained digest."""

        if not isinstance(payload, str) or len(payload.encode("utf-8")) > MAX_REFERENCE_BYTES:
            raise TelemetryValidationError("frozen monitoring reference size is invalid")
        try:
            decoded: object = json.loads(payload)
        except (ValueError, RecursionError) as error:
            raise TelemetryValidationError("invalid frozen monitoring reference JSON") from error
        body = _require_mapping(decoded, "frozen monitoring reference")
        if body.get("schema_version") != REFERENCE_SCHEMA:
            raise TelemetryValidationError("unsupported frozen monitoring reference schema")
        result = cls(
            reference=AggregateTelemetryWindow.from_dict(
                _require_mapping(body.get("reference"), "reference")
            ),
            config=monitoring_config_from_dict(_require_mapping(body.get("config"), "config")),
        )
        if payload != result.to_json():
            raise TelemetryValidationError(
                "reference is noncanonical or its content digests differ"
            )
        if expected_sha256 is not None and expected_sha256 != result.envelope_sha256:
            raise TelemetryValidationError("frozen monitoring reference identity mismatch")
        return result

    def _checked_config(self, config: TrustMonitoringConfig | None) -> TrustMonitoringConfig:
        effective = self.config if config is None else config
        if monitoring_config_sha256(effective) != self.config_sha256:
            raise TelemetryValidationError("monitoring configuration drift from frozen reference")
        return self.config

    def compare(
        self,
        current: AggregateTelemetryWindow,
        *,
        config: TrustMonitoringConfig | None = None,
    ) -> MonitoringComparison:
        return compare_telemetry_windows(
            self.reference, current, config=self._checked_config(config)
        )

    def replay(
        self,
        windows: Sequence[AggregateTelemetryWindow],
        *,
        config: TrustMonitoringConfig | None = None,
    ) -> tuple[MonitoringComparison, ...]:
        return replay_against_frozen_reference(
            self.reference, windows, config=self._checked_config(config)
        )

    def replay_to_json(
        self,
        windows: Sequence[AggregateTelemetryWindow],
        *,
        config: TrustMonitoringConfig | None = None,
    ) -> str:
        """Bind each unchanged v1 comparison to its input and frozen reference."""

        snapshots = tuple(
            AggregateTelemetryWindow.from_dict(window.to_dict()) for window in windows
        )
        comparisons = self.replay(snapshots, config=config)
        return _canonical_json(
            {
                "schema_version": REPLAY_SCHEMA,
                "frozen_reference_sha256": self.envelope_sha256,
                "config_sha256": self.config_sha256,
                "reference_sha256": self.reference_sha256,
                "comparisons": [
                    {"window_sha256": _digest(window.to_dict()), "comparison": comparison.to_dict()}
                    for window, comparison in zip(snapshots, comparisons, strict=True)
                ],
            }
        )
