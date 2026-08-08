from __future__ import annotations

import math

import pytest
import torch

from ecg_trust.robustness import (
    CorruptionValidationError,
    amplitude_scale,
    baseline_wander,
    drop_leads,
    gaussian_noise_at_snr,
    permute_leads,
    powerline_interference,
    zero_padded_time_shift,
)


def _waveforms() -> torch.Tensor:
    values = torch.linspace(-1.0, 1.0, 1_000)
    return torch.stack([values + lead for lead in range(12)]).unsqueeze(0)


def test_corruptions_preserve_shape_dtype_finiteness_and_input() -> None:
    original = _waveforms()
    before = original.clone()
    generator = torch.Generator().manual_seed(7)
    variants = (
        baseline_wander(original, amplitude_fraction=0.2),
        powerline_interference(original, amplitude_fraction=0.05),
        gaussian_noise_at_snr(original, snr_db=10.0, generator=generator),
        amplitude_scale(original, factor=1.2),
        zero_padded_time_shift(original, samples=25),
        drop_leads(original, lead_indices=(0, 5)),
        permute_leads(original, permutation=tuple(reversed(range(12)))),
    )
    torch.testing.assert_close(original, before)
    for variant in variants:
        assert variant.shape == original.shape
        assert variant.dtype == original.dtype
        assert torch.isfinite(variant).all()


def test_noise_is_deterministic_and_matches_requested_snr() -> None:
    waveforms = _waveforms()
    first = gaussian_noise_at_snr(
        waveforms, snr_db=20.0, generator=torch.Generator().manual_seed(123)
    )
    second = gaussian_noise_at_snr(
        waveforms, snr_db=20.0, generator=torch.Generator().manual_seed(123)
    )
    torch.testing.assert_close(first, second)
    signal_rms = waveforms.square().mean().sqrt()
    noise_rms = (first - waveforms).square().mean().sqrt()
    observed_snr = 20.0 * math.log10(float(signal_rms / noise_rms))
    assert observed_snr == pytest.approx(20.0, abs=1e-5)


def test_time_shift_is_zero_padded_not_circular() -> None:
    waveforms = _waveforms()
    right = zero_padded_time_shift(waveforms, samples=3)
    left = zero_padded_time_shift(waveforms, samples=-4)
    assert torch.count_nonzero(right[..., :3]) == 0
    assert torch.count_nonzero(left[..., -4:]) == 0
    torch.testing.assert_close(right[..., 3:], waveforms[..., :-3])
    torch.testing.assert_close(left[..., :-4], waveforms[..., 4:])


def test_lead_operations_are_explicit_and_validated() -> None:
    waveforms = _waveforms()
    dropped = drop_leads(waveforms, lead_indices=(1, 11))
    assert torch.count_nonzero(dropped[:, [1, 11]]) == 0
    torch.testing.assert_close(dropped[:, 0], waveforms[:, 0])

    permutation = (1, 0, *range(2, 12))
    permuted = permute_leads(waveforms, permutation=permutation)
    torch.testing.assert_close(permuted[:, 0], waveforms[:, 1])
    torch.testing.assert_close(permuted[:, 1], waveforms[:, 0])

    with pytest.raises(CorruptionValidationError, match="each index"):
        permute_leads(waveforms, permutation=tuple(range(11)))
    with pytest.raises(CorruptionValidationError, match="unique"):
        drop_leads(waveforms, lead_indices=(2, 2))


@pytest.mark.parametrize(
    "invalid",
    [torch.zeros(12, 1_000), torch.zeros(1, 11, 1_000), torch.full((1, 12, 10), math.nan)],
)
def test_rejects_invalid_waveform_contract(invalid: torch.Tensor) -> None:
    with pytest.raises(CorruptionValidationError):
        amplitude_scale(invalid, factor=1.0)
