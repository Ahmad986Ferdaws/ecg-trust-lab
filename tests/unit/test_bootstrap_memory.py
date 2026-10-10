"""Bound bootstrap working memory without changing the frozen PCG64 stream."""

from __future__ import annotations

import numpy as np
import pytest

from ecg_trust.ood_completion import statistics


def test_source_bootstrap_bounds_draws_and_matches_monolithic_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patients = np.concatenate((np.arange(1, 410), np.arange(1, 57)))
    rejected = np.arange(len(patients)) % 19 == 0
    _, inverse = np.unique(patients, return_inverse=True)
    counts = np.bincount(inverse)
    events = np.bincount(inverse, weights=rejected).astype(np.int64)
    generator_type = np.random.Generator
    reference = generator_type(np.random.PCG64(20_260_829))
    samples = reference.integers(0, len(counts), size=(10_000, len(counts)))
    rates = events[samples].sum(axis=1) / counts[samples].sum(axis=1)
    expected = np.quantile(rates, [0.025, 0.975, 0.95], method="linear")
    requested_sizes: list[tuple[int, int]] = []

    class RecordingGenerator:
        def __init__(self, bit_generator: np.random.PCG64) -> None:
            self.generator = generator_type(bit_generator)

        def integers(self, *args: object, **kwargs: object) -> np.ndarray:
            size = kwargs["size"]
            assert isinstance(size, tuple)
            requested_sizes.append(size)
            return self.generator.integers(*args, **kwargs)

    monkeypatch.setattr(statistics.np.random, "Generator", RecordingGenerator)
    interval = statistics.patient_cluster_bootstrap_interval(patients, rejected)

    assert (
        interval.two_sided_lower,
        interval.two_sided_upper,
        interval.one_sided_upper,
    ) == tuple(expected)
    assert sum(rows for rows, _ in requested_sizes) == 10_000
    assert all(rows * columns <= 1_000_000 for rows, columns in requested_sizes)
    assert len(requested_sizes) > 1
