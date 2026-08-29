"""Strict Failure Lab scenarios for one canonical physical-mV ECG.

The dispatcher is intentionally an adapter around :mod:`ecg_trust.robustness`.
Those batch transforms remain the single implementation of every corruption
except bounded clipping, which that module does not provide.

All outputs are controlled model-sensitivity probes.  They are not evidence of
clinical distribution shift, prevalence, safety, or diagnostic performance.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import numpy as np
import torch
from torch import Tensor

from ecg_trust import robustness
from ecg_trust.constants import LEADS

SCENARIO_SCHEMA_VERSION = "1.0.0"
SCENARIO_ARTIFACT_TYPE = "ecg_trust.stress_scenario"
PROVENANCE_SCHEMA_VERSION = "1.0.0"
PROVENANCE_ARTIFACT_TYPE = "ecg_trust.stress_provenance"
CONTROLLED_SENSITIVITY_LABEL = "controlled_model_sensitivity_not_clinical_shift"
PHYSICAL_MV_DOMAIN = "physical_millivolts"
CANONICAL_SAMPLING_FREQUENCY_HZ = 100.0

_SCENARIO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class StressScenarioError(ValueError):
    """Raised when a scenario or single-case ECG violates the contract."""


class StressKind(StrEnum):
    """Supported deterministic Failure Lab interventions."""

    BASELINE_WANDER = "baseline_wander"
    POWERLINE = "powerline"
    GAUSSIAN_NOISE = "gaussian_noise"
    GAIN = "gain"
    DC_OFFSET = "dc_offset"
    TIME_SHIFT = "time_shift"
    CONTIGUOUS_MASK = "contiguous_mask"
    LEAD_DROPOUT = "lead_dropout"
    LEAD_PERMUTATION = "lead_permutation"
    BOUNDED_CLIPPING = "bounded_clipping"


def _canonical_json(value: Mapping[str, object]) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise StressScenarioError("scenario values must be canonical JSON values") from exc


def _canonical_sha256(value: Mapping[str, object]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StressScenarioError(f"{name} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise StressScenarioError(f"{name} must be a finite number")
    return parsed


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StressScenarioError(f"{name} must be an integer")
    return value


def _exact_keys(parameters: Mapping[str, object], expected: set[str]) -> None:
    actual = set(parameters)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise StressScenarioError(f"invalid parameters; missing={missing}, extra={extra}")


def _fraction(value: object, name: str) -> float:
    parsed = _finite_float(value, name)
    if not 0.0 < parsed <= 2.0:
        raise StressScenarioError(f"{name} must be in (0, 2]")
    return parsed


def _phase(value: object) -> float:
    parsed = _finite_float(value, "phase_radians")
    if not 0.0 <= parsed < 2.0 * math.pi:
        raise StressScenarioError("phase_radians must use the canonical interval [0, 2*pi)")
    return parsed


def _lead_names(value: object, name: str = "leads") -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise StressScenarioError(f"{name} must be a non-empty list of canonical lead names")
    parsed = list(value)
    if not parsed or any(not isinstance(lead, str) for lead in parsed):
        raise StressScenarioError(f"{name} must be a non-empty list of canonical lead names")
    names = cast(list[str], parsed)
    if len(set(names)) != len(names):
        raise StressScenarioError(f"{name} must not contain duplicate leads")
    if any(lead not in LEADS for lead in names):
        raise StressScenarioError(f"{name} contains a noncanonical lead")
    canonical_subset = [lead for lead in LEADS if lead in names]
    if names != canonical_subset:
        raise StressScenarioError(f"{name} must be ordered according to the canonical lead order")
    return names


def _permutation(value: object) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise StressScenarioError("ordered_leads must be a full explicit lead permutation")
    parsed = list(value)
    if any(not isinstance(lead, str) for lead in parsed):
        raise StressScenarioError("ordered_leads must contain canonical lead names")
    names = cast(list[str], parsed)
    if len(names) != len(LEADS) or set(names) != set(LEADS):
        raise StressScenarioError("ordered_leads must contain each canonical lead exactly once")
    if tuple(names) == LEADS:
        raise StressScenarioError("an identity lead permutation is an exact no-op")
    return names


def _validated_parameters(kind: StressKind, parameters: Mapping[str, object]) -> dict[str, object]:
    if kind is StressKind.BASELINE_WANDER:
        _exact_keys(parameters, {"amplitude_fraction", "frequency_hz", "phase_radians"})
        frequency = _finite_float(parameters["frequency_hz"], "frequency_hz")
        if not 0.0 < frequency <= 1.0:
            raise StressScenarioError("baseline frequency_hz must be in (0, 1]")
        return {
            "amplitude_fraction": _fraction(parameters["amplitude_fraction"], "amplitude_fraction"),
            "frequency_hz": frequency,
            "phase_radians": _phase(parameters["phase_radians"]),
        }
    if kind is StressKind.POWERLINE:
        _exact_keys(
            parameters,
            {"amplitude_fraction", "mains_frequency_hz", "phase_radians"},
        )
        mains = _finite_float(parameters["mains_frequency_hz"], "mains_frequency_hz")
        if mains not in {50.0, 60.0}:
            raise StressScenarioError("mains_frequency_hz must be exactly 50 or 60")
        return {
            "amplitude_fraction": _fraction(parameters["amplitude_fraction"], "amplitude_fraction"),
            "mains_frequency_hz": mains,
            "phase_radians": _phase(parameters["phase_radians"]),
        }
    if kind is StressKind.GAUSSIAN_NOISE:
        _exact_keys(parameters, {"snr_db"})
        snr = _finite_float(parameters["snr_db"], "snr_db")
        if not -40.0 <= snr <= 100.0:
            raise StressScenarioError("snr_db must be in [-40, 100]")
        return {"snr_db": snr}
    if kind is StressKind.GAIN:
        _exact_keys(parameters, {"factor"})
        factor = _finite_float(parameters["factor"], "factor")
        if not 0.1 <= factor <= 10.0:
            raise StressScenarioError("factor must be in [0.1, 10]")
        if factor == 1.0:
            raise StressScenarioError("factor=1 is an exact no-op")
        return {"factor": factor}
    if kind is StressKind.DC_OFFSET:
        _exact_keys(parameters, {"offset_fraction"})
        offset = _finite_float(parameters["offset_fraction"], "offset_fraction")
        if not -5.0 <= offset <= 5.0:
            raise StressScenarioError("offset_fraction must be in [-5, 5]")
        if offset == 0.0:
            raise StressScenarioError("offset_fraction=0 is an exact no-op")
        return {"offset_fraction": offset}
    if kind is StressKind.TIME_SHIFT:
        _exact_keys(parameters, {"samples"})
        samples = _integer(parameters["samples"], "samples")
        if samples == 0:
            raise StressScenarioError("samples=0 is an exact no-op")
        if abs(samples) > 10_000_000:
            raise StressScenarioError("absolute samples must not exceed 10,000,000")
        return {"samples": samples}
    if kind is StressKind.CONTIGUOUS_MASK:
        _exact_keys(parameters, {"start_sample", "width_samples", "leads"})
        start = _integer(parameters["start_sample"], "start_sample")
        width = _integer(parameters["width_samples"], "width_samples")
        if start < 0 or start > 10_000_000:
            raise StressScenarioError("start_sample must be in [0, 10,000,000]")
        if width < 1 or width > 10_000_000:
            raise StressScenarioError("width_samples must be in [1, 10,000,000]")
        return {
            "start_sample": start,
            "width_samples": width,
            "leads": _lead_names(parameters["leads"]),
        }
    if kind is StressKind.LEAD_DROPOUT:
        _exact_keys(parameters, {"leads"})
        return {"leads": _lead_names(parameters["leads"])}
    if kind is StressKind.LEAD_PERMUTATION:
        _exact_keys(parameters, {"ordered_leads"})
        return {"ordered_leads": _permutation(parameters["ordered_leads"])}
    if kind is StressKind.BOUNDED_CLIPPING:
        _exact_keys(parameters, {"minimum_mv", "maximum_mv"})
        minimum = _finite_float(parameters["minimum_mv"], "minimum_mv")
        maximum = _finite_float(parameters["maximum_mv"], "maximum_mv")
        if not -100.0 <= minimum < maximum <= 100.0:
            raise StressScenarioError(
                "clipping bounds must satisfy -100 <= minimum_mv < maximum_mv <= 100"
            )
        return {"minimum_mv": minimum, "maximum_mv": maximum}
    raise StressScenarioError(f"unsupported stress kind: {kind}")


@dataclass(frozen=True, slots=True, init=False)
class StressScenario:
    """Immutable, strict, integrity-bound request for one stress intervention."""

    scenario_id: str
    kind: StressKind
    _parameters_json: str
    seed: int | None

    def __init__(
        self,
        *,
        scenario_id: str,
        kind: StressKind | str,
        parameters: Mapping[str, object],
        seed: int | None = None,
    ) -> None:
        if not isinstance(scenario_id, str) or _SCENARIO_ID.fullmatch(scenario_id) is None:
            raise StressScenarioError("scenario_id must be a safe 1-128 character identifier")
        try:
            parsed_kind = StressKind(kind)
        except (TypeError, ValueError) as exc:
            raise StressScenarioError("kind is not a supported stress intervention") from exc
        if not isinstance(parameters, Mapping):
            raise StressScenarioError("parameters must be a mapping")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise StressScenarioError("seed must be an integer or null")
        if parsed_kind is StressKind.GAUSSIAN_NOISE:
            if seed is None:
                raise StressScenarioError("Gaussian noise requires an explicit seed")
            if not 0 <= seed <= 2**63 - 1:
                raise StressScenarioError("seed must be in [0, 2^63-1]")
        elif seed is not None:
            raise StressScenarioError("seed is only valid for Gaussian noise")
        normalized = _validated_parameters(parsed_kind, parameters)
        object.__setattr__(self, "scenario_id", scenario_id)
        object.__setattr__(self, "kind", parsed_kind)
        object.__setattr__(self, "_parameters_json", _canonical_json(normalized))
        object.__setattr__(self, "seed", seed)

    @property
    def parameters(self) -> dict[str, object]:
        """Return a detached JSON-compatible parameter dictionary."""

        return cast(dict[str, object], json.loads(self._parameters_json))

    def _body(self) -> dict[str, object]:
        return {
            "schema_version": SCENARIO_SCHEMA_VERSION,
            "artifact_type": SCENARIO_ARTIFACT_TYPE,
            "scenario_id": self.scenario_id,
            "kind": self.kind.value,
            "parameters": self.parameters,
            "seed": self.seed,
            "interpretation": CONTROLLED_SENSITIVITY_LABEL,
        }

    @property
    def scenario_sha256(self) -> str:
        """Canonical content hash of the scenario body."""

        return _canonical_sha256(self._body())

    def to_dict(self) -> dict[str, object]:
        """Serialize with schema/type sentinels and a verified content hash."""

        return {**self._body(), "scenario_sha256": self.scenario_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> StressScenario:
        """Strictly deserialize and verify a scenario artifact."""

        if not isinstance(value, Mapping):
            raise StressScenarioError("scenario artifact must be a mapping")
        expected = {
            "schema_version",
            "artifact_type",
            "scenario_id",
            "kind",
            "parameters",
            "seed",
            "interpretation",
            "scenario_sha256",
        }
        if set(value) != expected:
            raise StressScenarioError("scenario artifact has missing or unknown fields")
        if value["schema_version"] != SCENARIO_SCHEMA_VERSION:
            raise StressScenarioError("unsupported scenario schema_version")
        if value["artifact_type"] != SCENARIO_ARTIFACT_TYPE:
            raise StressScenarioError("invalid scenario artifact_type")
        if value["interpretation"] != CONTROLLED_SENSITIVITY_LABEL:
            raise StressScenarioError("invalid scenario interpretation label")
        parameters = value["parameters"]
        if not isinstance(parameters, Mapping):
            raise StressScenarioError("parameters must be a mapping")
        scenario_id = value["scenario_id"]
        kind = value["kind"]
        seed = value["seed"]
        if not isinstance(scenario_id, str) or not isinstance(kind, str):
            raise StressScenarioError("scenario_id and kind must be strings")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise StressScenarioError("seed must be an integer or null")
        restored = cls(
            scenario_id=scenario_id,
            kind=kind,
            parameters=cast(Mapping[str, object], parameters),
            seed=seed,
        )
        stored_hash = value["scenario_sha256"]
        if not isinstance(stored_hash, str) or _SHA256.fullmatch(stored_hash) is None:
            raise StressScenarioError("scenario_sha256 is malformed")
        if stored_hash != restored.scenario_sha256:
            raise StressScenarioError("scenario artifact integrity check failed")
        return restored


@dataclass(frozen=True, slots=True)
class StressProvenance:
    """Serializable lineage binding a scenario to exact input/output bytes."""

    scenario_id: str
    scenario_sha256: str
    kind: StressKind
    parent_waveform_sha256: str
    output_waveform_sha256: str
    sampling_frequency_hz: float
    ordered_leads: tuple[str, ...]
    affected_leads: tuple[str, ...]
    _resolved_parameters_json: str
    seed: int | None

    @property
    def resolved_parameters(self) -> dict[str, object]:
        return cast(dict[str, object], json.loads(self._resolved_parameters_json))

    def _body(self) -> dict[str, object]:
        return {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "artifact_type": PROVENANCE_ARTIFACT_TYPE,
            "scenario_id": self.scenario_id,
            "scenario_sha256": self.scenario_sha256,
            "kind": self.kind.value,
            "parent_waveform_sha256": self.parent_waveform_sha256,
            "output_waveform_sha256": self.output_waveform_sha256,
            "sampling_frequency_hz": self.sampling_frequency_hz,
            "ordered_leads": list(self.ordered_leads),
            "affected_leads": list(self.affected_leads),
            "resolved_parameters": self.resolved_parameters,
            "seed": self.seed,
            "input_domain": PHYSICAL_MV_DOMAIN,
            "output_dtype": "float32",
            "interpretation": CONTROLLED_SENSITIVITY_LABEL,
        }

    @property
    def provenance_sha256(self) -> str:
        return _canonical_sha256(self._body())

    def to_dict(self) -> dict[str, object]:
        return {**self._body(), "provenance_sha256": self.provenance_sha256}


@dataclass(frozen=True, slots=True)
class AppliedStress:
    """A changed waveform and the deterministic provenance that binds it."""

    waveform: Tensor
    provenance: StressProvenance


def _validate_single_case(
    waveform: Tensor,
    *,
    sampling_frequency_hz: float,
    ordered_leads: Sequence[str],
) -> None:
    if not isinstance(waveform, Tensor):
        raise TypeError("waveform must be a torch.Tensor")
    if waveform.device.type != "cpu":
        raise StressScenarioError("waveform must be on CPU for canonical hashing and replay")
    if waveform.dtype is not torch.float32:
        raise StressScenarioError("waveform must use canonical float32 physical-mV values")
    if waveform.ndim != 2 or waveform.shape[0] != len(LEADS) or waveform.shape[1] < 2:
        raise StressScenarioError(f"waveform must have shape [{len(LEADS)}, time>=2]")
    if not waveform.is_contiguous():
        raise StressScenarioError("waveform must be contiguous")
    if waveform.requires_grad:
        raise StressScenarioError("waveform must be a detached inference input")
    if not torch.isfinite(waveform).all().item():
        raise StressScenarioError("waveform must contain only finite physical-mV values")
    if isinstance(sampling_frequency_hz, bool) or not isinstance(
        sampling_frequency_hz, (int, float)
    ):
        raise StressScenarioError("sampling_frequency_hz must be numeric")
    if float(sampling_frequency_hz) != CANONICAL_SAMPLING_FREQUENCY_HZ:
        raise StressScenarioError("Failure Lab input must use the canonical 100 Hz sample rate")
    if tuple(ordered_leads) != LEADS:
        raise StressScenarioError("ordered_leads must exactly match the canonical 12-lead order")


def waveform_sha256(waveform: Tensor) -> str:
    """Hash a canonical float32 single-case waveform, including its shape contract."""

    _validate_single_case(
        waveform,
        sampling_frequency_hz=CANONICAL_SAMPLING_FREQUENCY_HZ,
        ordered_leads=LEADS,
    )
    array = waveform.detach().numpy().astype(np.dtype("<f4"), copy=False)
    header = f"ecg_trust.float32_waveform.v1|shape=12,{waveform.shape[1]}|little-endian|".encode()
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(array.tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


def _resolved_powerline(
    mains_frequency_hz: float, phase_radians: float
) -> tuple[float, float, int]:
    """Resolve sampled alias magnitude, phase, and signed alias direction."""

    sampling = CANONICAL_SAMPLING_FREQUENCY_HZ
    signed_alias = (mains_frequency_hz + sampling / 2.0) % sampling - sampling / 2.0
    if math.isclose(abs(signed_alias), sampling / 2.0):
        signed_alias = sampling / 2.0
    direction = 1 if signed_alias >= 0.0 else -1
    effective_frequency = abs(signed_alias)
    applied_phase = phase_radians if direction > 0 else math.pi - phase_radians
    applied_phase %= 2.0 * math.pi
    return effective_frequency, applied_phase, direction


def _indices(leads: Sequence[str]) -> tuple[int, ...]:
    return tuple(LEADS.index(lead) for lead in leads)


def _affected_permutation(ordered_leads: Sequence[str]) -> tuple[str, ...]:
    involved: set[str] = set()
    for index, source_lead in enumerate(ordered_leads):
        if source_lead != LEADS[index]:
            involved.add(LEADS[index])
            involved.add(source_lead)
    return tuple(lead for lead in LEADS if lead in involved)


def apply_stress_scenario(
    waveform: Tensor,
    scenario: StressScenario,
    *,
    sampling_frequency_hz: float = CANONICAL_SAMPLING_FREQUENCY_HZ,
    ordered_leads: Sequence[str] = LEADS,
) -> AppliedStress:
    """Apply one deterministic stress scenario without mutating ``waveform``.

    ``waveform`` is one contiguous CPU float32 ECG in physical millivolts with
    shape ``[12, time]`` and canonical lead order.  The 100 Hz contract is
    explicit so a requested 60 Hz mains signal is resolved to its sampled 40 Hz
    alias rather than passed above Nyquist to the underlying transform.
    """

    if not isinstance(scenario, StressScenario):
        raise TypeError("scenario must be a StressScenario")
    _validate_single_case(
        waveform,
        sampling_frequency_hz=sampling_frequency_hz,
        ordered_leads=ordered_leads,
    )
    parent_hash = waveform_sha256(waveform)
    batch = waveform.clone().unsqueeze(0)
    parameters = scenario.parameters
    resolved = dict(parameters)
    affected: tuple[str, ...] = LEADS

    try:
        if scenario.kind is StressKind.BASELINE_WANDER:
            output = robustness.baseline_wander(
                batch,
                amplitude_fraction=cast(float, parameters["amplitude_fraction"]),
                frequency_hz=cast(float, parameters["frequency_hz"]),
                sampling_frequency_hz=CANONICAL_SAMPLING_FREQUENCY_HZ,
                phase_radians=cast(float, parameters["phase_radians"]),
            )
        elif scenario.kind is StressKind.POWERLINE:
            effective_frequency, applied_phase, alias_direction = _resolved_powerline(
                cast(float, parameters["mains_frequency_hz"]),
                cast(float, parameters["phase_radians"]),
            )
            resolved.update(
                {
                    "effective_frequency_hz": effective_frequency,
                    "applied_phase_radians": applied_phase,
                    "alias_direction": alias_direction,
                }
            )
            output = robustness.powerline_interference(
                batch,
                amplitude_fraction=cast(float, parameters["amplitude_fraction"]),
                frequency_hz=effective_frequency,
                sampling_frequency_hz=CANONICAL_SAMPLING_FREQUENCY_HZ,
                phase_radians=applied_phase,
            )
        elif scenario.kind is StressKind.GAUSSIAN_NOISE:
            if scenario.seed is None:  # Defensive; constructor already enforces this.
                raise StressScenarioError("Gaussian noise requires an explicit seed")
            generator = torch.Generator(device="cpu")
            generator.manual_seed(scenario.seed)
            output = robustness.gaussian_noise_at_snr(
                batch,
                snr_db=cast(float, parameters["snr_db"]),
                generator=generator,
            )
        elif scenario.kind is StressKind.GAIN:
            output = robustness.amplitude_scale(
                batch,
                factor=cast(float, parameters["factor"]),
            )
        elif scenario.kind is StressKind.DC_OFFSET:
            output = robustness.dc_offset(
                batch,
                offset_fraction=cast(float, parameters["offset_fraction"]),
            )
        elif scenario.kind is StressKind.TIME_SHIFT:
            output = robustness.zero_padded_time_shift(
                batch,
                samples=cast(int, parameters["samples"]),
            )
        elif scenario.kind is StressKind.CONTIGUOUS_MASK:
            leads = cast(list[str], parameters["leads"])
            affected = tuple(leads)
            output = robustness.mask_contiguous_time(
                batch,
                start_sample=cast(int, parameters["start_sample"]),
                width_samples=cast(int, parameters["width_samples"]),
                lead_indices=_indices(leads),
            )
        elif scenario.kind is StressKind.LEAD_DROPOUT:
            leads = cast(list[str], parameters["leads"])
            affected = tuple(leads)
            output = robustness.drop_leads(batch, lead_indices=_indices(leads))
        elif scenario.kind is StressKind.LEAD_PERMUTATION:
            permutation = cast(list[str], parameters["ordered_leads"])
            affected = _affected_permutation(permutation)
            output = robustness.permute_leads(batch, permutation=_indices(permutation))
        elif scenario.kind is StressKind.BOUNDED_CLIPPING:
            minimum = cast(float, parameters["minimum_mv"])
            maximum = cast(float, parameters["maximum_mv"])
            output = torch.clamp(batch, min=minimum, max=maximum)
            changed_by_lead = torch.ne(output, batch).any(dim=2).squeeze(0)
            affected = tuple(lead for index, lead in enumerate(LEADS) if changed_by_lead[index])
        else:  # pragma: no cover - exhaustive enum defense
            raise StressScenarioError(f"unsupported stress kind: {scenario.kind}")
    except robustness.CorruptionValidationError as exc:
        raise StressScenarioError(f"scenario is incompatible with this ECG: {exc}") from exc

    changed = output.squeeze(0).to(dtype=torch.float32).contiguous()
    if changed.shape != waveform.shape or not torch.isfinite(changed).all().item():
        raise StressScenarioError("stress transform produced a noncanonical waveform")
    output_hash = waveform_sha256(changed)
    if output_hash == parent_hash:
        raise StressScenarioError("stress scenario produced an exact no-op on this waveform")

    provenance = StressProvenance(
        scenario_id=scenario.scenario_id,
        scenario_sha256=scenario.scenario_sha256,
        kind=scenario.kind,
        parent_waveform_sha256=parent_hash,
        output_waveform_sha256=output_hash,
        sampling_frequency_hz=CANONICAL_SAMPLING_FREQUENCY_HZ,
        ordered_leads=LEADS,
        affected_leads=affected,
        _resolved_parameters_json=_canonical_json(resolved),
        seed=scenario.seed,
    )
    return AppliedStress(waveform=changed, provenance=provenance)
