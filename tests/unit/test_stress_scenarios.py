from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from ecg_trust import robustness
from ecg_trust.constants import LEADS
from ecg_trust.stress import (
    CONTROLLED_SENSITIVITY_LABEL,
    StressKind,
    StressScenario,
    StressScenarioError,
    apply_stress_scenario,
    waveform_sha256,
)
from ecg_trust.stress import scenarios as scenario_module


@pytest.fixture
def waveform() -> Tensor:
    time = torch.arange(256, dtype=torch.float32) / 100.0
    leads = [
        (0.45 + index * 0.025) * torch.sin(2.0 * math.pi * (1.0 + index / 20.0) * time)
        + 0.015 * index
        for index in range(len(LEADS))
    ]
    return torch.stack(leads).contiguous()


def _scenarios() -> list[StressScenario]:
    swapped = [LEADS[1], LEADS[0], *LEADS[2:]]
    return [
        StressScenario(
            scenario_id="baseline-v1",
            kind=StressKind.BASELINE_WANDER,
            parameters={
                "amplitude_fraction": 0.15,
                "frequency_hz": 0.33,
                "phase_radians": 0.0,
            },
        ),
        StressScenario(
            scenario_id="powerline-60-v1",
            kind=StressKind.POWERLINE,
            parameters={
                "amplitude_fraction": 0.1,
                "mains_frequency_hz": 60,
                "phase_radians": math.pi / 2.0,
            },
        ),
        StressScenario(
            scenario_id="noise-v1",
            kind=StressKind.GAUSSIAN_NOISE,
            parameters={"snr_db": 12.0},
            seed=741,
        ),
        StressScenario(
            scenario_id="gain-v1",
            kind=StressKind.GAIN,
            parameters={"factor": 1.2},
        ),
        StressScenario(
            scenario_id="offset-v1",
            kind=StressKind.DC_OFFSET,
            parameters={"offset_fraction": -0.2},
        ),
        StressScenario(
            scenario_id="shift-v1",
            kind=StressKind.TIME_SHIFT,
            parameters={"samples": 9},
        ),
        StressScenario(
            scenario_id="mask-v1",
            kind=StressKind.CONTIGUOUS_MASK,
            parameters={"start_sample": 20, "width_samples": 31, "leads": ["I", "II"]},
        ),
        StressScenario(
            scenario_id="drop-v1",
            kind=StressKind.LEAD_DROPOUT,
            parameters={"leads": ["III", "aVR"]},
        ),
        StressScenario(
            scenario_id="permutation-v1",
            kind=StressKind.LEAD_PERMUTATION,
            parameters={"ordered_leads": swapped},
        ),
        StressScenario(
            scenario_id="clip-v1",
            kind=StressKind.BOUNDED_CLIPPING,
            parameters={"minimum_mv": -0.2, "maximum_mv": 0.2},
        ),
    ]


@pytest.mark.parametrize("scenario", _scenarios(), ids=lambda item: item.kind.value)
def test_every_kind_is_deterministic_finite_hash_bound_and_non_mutating(
    waveform: Tensor, scenario: StressScenario
) -> None:
    before = waveform.clone()

    first = apply_stress_scenario(waveform, scenario)
    second = apply_stress_scenario(waveform, scenario)

    assert torch.equal(waveform, before)
    assert first.waveform.data_ptr() != waveform.data_ptr()
    assert first.waveform.shape == waveform.shape
    assert first.waveform.dtype is torch.float32
    assert first.waveform.is_contiguous()
    assert torch.isfinite(first.waveform).all().item()
    assert not torch.equal(first.waveform, waveform)
    assert torch.equal(first.waveform, second.waveform)
    assert first.provenance.to_dict() == second.provenance.to_dict()
    assert first.provenance.parent_waveform_sha256 == waveform_sha256(waveform)
    assert first.provenance.output_waveform_sha256 == waveform_sha256(first.waveform)
    assert first.provenance.parent_waveform_sha256 != first.provenance.output_waveform_sha256
    assert first.provenance.scenario_sha256 == scenario.scenario_sha256
    assert first.provenance.affected_leads
    artifact = first.provenance.to_dict()
    assert artifact["interpretation"] == CONTROLLED_SENSITIVITY_LABEL
    assert artifact["input_domain"] == "physical_millivolts"
    assert artifact["output_dtype"] == "float32"
    assert artifact["provenance_sha256"].startswith("sha256:")


def test_scenario_serialization_round_trip_is_strict_and_integrity_bound() -> None:
    scenario = _scenarios()[0]
    artifact = scenario.to_dict()

    restored = StressScenario.from_dict(artifact)

    assert restored == scenario
    assert restored.to_dict() == artifact
    assert artifact["schema_version"] == "1.0.0"
    assert artifact["artifact_type"] == "ecg_trust.stress_scenario"
    assert artifact["interpretation"] == CONTROLLED_SENSITIVITY_LABEL
    assert isinstance(artifact["scenario_sha256"], str)
    assert len(str(artifact["scenario_sha256"])) == 71

    tampered = dict(artifact)
    tampered_parameters = dict(tampered["parameters"])  # type: ignore[arg-type]
    tampered_parameters["frequency_hz"] = 0.5
    tampered["parameters"] = tampered_parameters
    with pytest.raises(StressScenarioError, match="integrity"):
        StressScenario.from_dict(tampered)

    unknown = {**artifact, "unknown": True}
    with pytest.raises(StressScenarioError, match="missing or unknown"):
        StressScenario.from_dict(unknown)


def test_scenario_parameters_are_detached_from_caller_and_property_mutation() -> None:
    supplied: dict[str, object] = {"leads": ["I", "II"]}
    scenario = StressScenario(
        scenario_id="detached",
        kind=StressKind.LEAD_DROPOUT,
        parameters=supplied,
    )
    supplied_leads = supplied["leads"]
    assert isinstance(supplied_leads, list)
    supplied_leads.append("III")
    detached = scenario.parameters
    detached["leads"] = ["V6"]

    assert scenario.parameters == {"leads": ["I", "II"]}


@pytest.mark.parametrize(
    ("kind", "parameters", "seed", "message"),
    [
        (
            StressKind.BASELINE_WANDER,
            {"amplitude_fraction": 0, "frequency_hz": 0.3, "phase_radians": 0},
            None,
            r"\(0, 2\]",
        ),
        (
            StressKind.POWERLINE,
            {"amplitude_fraction": 0.1, "mains_frequency_hz": 55, "phase_radians": 0},
            None,
            "50 or 60",
        ),
        (StressKind.GAUSSIAN_NOISE, {"snr_db": 10}, None, "explicit seed"),
        (StressKind.GAIN, {"factor": 1}, None, "no-op"),
        (StressKind.DC_OFFSET, {"offset_fraction": 0}, None, "no-op"),
        (StressKind.TIME_SHIFT, {"samples": 0}, None, "no-op"),
        (
            StressKind.CONTIGUOUS_MASK,
            {"start_sample": 1, "width_samples": 2, "leads": []},
            None,
            "non-empty",
        ),
        (StressKind.LEAD_DROPOUT, {"leads": ["II", "I"]}, None, "canonical lead order"),
        (StressKind.LEAD_PERMUTATION, {"ordered_leads": list(LEADS)}, None, "identity"),
        (StressKind.BOUNDED_CLIPPING, {"minimum_mv": 1, "maximum_mv": -1}, None, "bounds"),
    ],
)
def test_unsafe_or_exact_noop_specs_are_rejected(
    kind: StressKind,
    parameters: dict[str, object],
    seed: int | None,
    message: str,
) -> None:
    with pytest.raises(StressScenarioError, match=message):
        StressScenario(
            scenario_id="invalid",
            kind=kind,
            parameters=parameters,
            seed=seed,
        )


def test_unknown_ambiguous_parameters_and_irrelevant_seed_are_rejected() -> None:
    with pytest.raises(StressScenarioError, match="missing=.*extra"):
        StressScenario(
            scenario_id="ambiguous",
            kind=StressKind.LEAD_DROPOUT,
            parameters={"lead_indices": [0]},
        )
    with pytest.raises(StressScenarioError, match="only valid"):
        StressScenario(
            scenario_id="irrelevant-seed",
            kind=StressKind.GAIN,
            parameters={"factor": 1.1},
            seed=3,
        )
    with pytest.raises(StressScenarioError, match="safe"):
        StressScenario(
            scenario_id="unsafe id with spaces",
            kind=StressKind.GAIN,
            parameters={"factor": 1.1},
        )


def test_runtime_exact_noops_are_rejected(waveform: Tensor) -> None:
    clipping = StressScenario(
        scenario_id="wide-clip",
        kind=StressKind.BOUNDED_CLIPPING,
        parameters={"minimum_mv": -99.0, "maximum_mv": 99.0},
    )
    with pytest.raises(StressScenarioError, match="exact no-op"):
        apply_stress_scenario(waveform, clipping)

    zero = torch.zeros_like(waveform)
    gain = StressScenario(
        scenario_id="zero-gain",
        kind=StressKind.GAIN,
        parameters={"factor": 2.0},
    )
    with pytest.raises(StressScenarioError, match="exact no-op"):
        apply_stress_scenario(zero, gain)


def test_noise_seed_controls_replay(waveform: Tensor) -> None:
    first = StressScenario(
        scenario_id="noise-a",
        kind=StressKind.GAUSSIAN_NOISE,
        parameters={"snr_db": 5.0},
        seed=11,
    )
    second = StressScenario(
        scenario_id="noise-b",
        kind=StressKind.GAUSSIAN_NOISE,
        parameters={"snr_db": 5.0},
        seed=12,
    )

    first_output = apply_stress_scenario(waveform, first)
    second_output = apply_stress_scenario(waveform, second)

    assert not torch.equal(first_output.waveform, second_output.waveform)
    assert first_output.provenance.output_waveform_sha256 != (
        second_output.provenance.output_waveform_sha256
    )
    assert first_output.provenance.seed == 11


@pytest.mark.parametrize(
    ("mains", "effective", "direction"),
    [(50.0, 50.0, 1), (60.0, 40.0, -1)],
)
def test_powerline_resolves_100hz_alias_with_phase_and_records_it(
    waveform: Tensor, mains: float, effective: float, direction: int
) -> None:
    requested_phase = 0.4
    scenario = StressScenario(
        scenario_id=f"powerline-{int(mains)}",
        kind=StressKind.POWERLINE,
        parameters={
            "amplitude_fraction": 0.12,
            "mains_frequency_hz": mains,
            "phase_radians": requested_phase,
        },
    )

    applied = apply_stress_scenario(waveform, scenario)
    applied_phase = (
        requested_phase if direction > 0 else (math.pi - requested_phase) % (2 * math.pi)
    )
    expected = robustness.powerline_interference(
        waveform.unsqueeze(0),
        amplitude_fraction=0.12,
        frequency_hz=effective,
        sampling_frequency_hz=100.0,
        phase_radians=applied_phase,
    ).squeeze(0)

    assert torch.equal(applied.waveform, expected)
    assert applied.provenance.resolved_parameters["effective_frequency_hz"] == effective
    assert applied.provenance.resolved_parameters["alias_direction"] == direction
    assert applied.provenance.resolved_parameters["applied_phase_radians"] == pytest.approx(
        applied_phase
    )


def test_dispatcher_uses_existing_robustness_primitive_as_single_case_adapter(
    waveform: Tensor, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}
    original = robustness.baseline_wander

    def spy(waveforms: Tensor, **kwargs: float) -> Tensor:
        observed["shape"] = tuple(waveforms.shape)
        observed["kwargs"] = kwargs
        return original(waveforms, **kwargs)

    monkeypatch.setattr(scenario_module.robustness, "baseline_wander", spy)
    scenario = _scenarios()[0]

    apply_stress_scenario(waveform, scenario)

    assert observed["shape"] == (1, len(LEADS), waveform.shape[1])
    assert observed["kwargs"] == {
        "amplitude_fraction": 0.15,
        "frequency_hz": 0.33,
        "sampling_frequency_hz": 100.0,
        "phase_radians": 0.0,
    }


def test_targeted_leads_and_resolved_parameters_are_recorded(waveform: Tensor) -> None:
    scenario = StressScenario(
        scenario_id="targeted-mask",
        kind=StressKind.CONTIGUOUS_MASK,
        parameters={
            "start_sample": 10,
            "width_samples": 8,
            "leads": ["I", "III", "V2"],
        },
    )

    applied = apply_stress_scenario(waveform, scenario)

    assert applied.provenance.affected_leads == ("I", "III", "V2")
    assert applied.provenance.resolved_parameters == scenario.parameters
    assert torch.count_nonzero(applied.waveform[[0, 2, 7], 10:18]).item() == 0
    assert torch.equal(applied.waveform[1], waveform[1])


def test_zero_padded_shift_does_not_wrap(waveform: Tensor) -> None:
    scenario = StressScenario(
        scenario_id="right-shift",
        kind=StressKind.TIME_SHIFT,
        parameters={"samples": 5},
    )

    shifted = apply_stress_scenario(waveform, scenario).waveform

    assert torch.count_nonzero(shifted[:, :5]).item() == 0
    assert torch.equal(shifted[:, 5:], waveform[:, :-5])


def test_clipping_is_bounded_and_records_only_changed_leads(waveform: Tensor) -> None:
    scenario = StressScenario(
        scenario_id="clip",
        kind=StressKind.BOUNDED_CLIPPING,
        parameters={"minimum_mv": -0.3, "maximum_mv": 0.3},
    )

    applied = apply_stress_scenario(waveform, scenario)

    assert applied.waveform.min().item() >= -0.3 - 1e-6
    assert applied.waveform.max().item() <= 0.3 + 1e-6
    expected = tuple(
        lead
        for index, lead in enumerate(LEADS)
        if ((waveform[index] < -0.3) | (waveform[index] > 0.3)).any().item()
    )
    assert applied.provenance.affected_leads == expected


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.to(dtype=torch.float64),
        lambda value: value[:11],
        lambda value: value[:, ::2],
        lambda value: value.clone().requires_grad_(True),
    ],
)
def test_noncanonical_tensors_are_rejected(
    waveform: Tensor, mutate: Callable[[Tensor], Tensor]
) -> None:
    scenario = _scenarios()[3]
    with pytest.raises(StressScenarioError):
        apply_stress_scenario(mutate(waveform), scenario)


def test_nonfinite_sampling_rate_and_lead_order_contracts_are_rejected(waveform: Tensor) -> None:
    nonfinite = waveform.clone()
    nonfinite[0, 0] = float("nan")
    scenario = _scenarios()[3]
    with pytest.raises(StressScenarioError, match="finite"):
        apply_stress_scenario(nonfinite, scenario)
    with pytest.raises(StressScenarioError, match="100 Hz"):
        apply_stress_scenario(waveform, scenario, sampling_frequency_hz=500.0)
    with pytest.raises(StressScenarioError, match="canonical 12-lead"):
        apply_stress_scenario(waveform, scenario, ordered_leads=LEADS[::-1])


def test_waveform_hash_is_stable_and_sensitive(waveform: Tensor) -> None:
    first = waveform_sha256(waveform)
    second = waveform_sha256(waveform.clone())
    changed = waveform.clone()
    changed[0, 0] += torch.finfo(torch.float32).eps

    assert first == second
    assert first.startswith("sha256:")
    assert len(first) == 71
    assert waveform_sha256(changed) != first
