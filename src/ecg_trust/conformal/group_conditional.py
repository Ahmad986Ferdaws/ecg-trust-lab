"""Group-conditional (Mondrian) label-wise conformal prediction sets.

``LabelwiseBinaryConformal`` guarantees coverage on average over the whole
calibration population. A group whose scores differ from the rest (the r3
audit found age 80+ under-covered by the global gates) can therefore be
under-covered while the marginal guarantee still holds. Mondrian conformal
prediction (Vovk et al., *Algorithmic Learning in a Random World*) calibrates
separately inside each declared group, which gives the finite-sample
guarantee within every group, at the cost of wider or even trivial ``{0, 1}``
sets for small groups.

Groups must be known before calibration and assigned without looking at
outcomes (for example age band or sex from metadata). A group unseen during
calibration is refused at prediction time rather than silently pooled.

Nothing in the Sentinel engine, case contracts, or frozen configurations
selects this artifact. ``predict`` returns plain ``BinaryPredictionSets``, which
carry no provenance, and ``conformal_prediction_sets_to_contracts`` stamps
whatever sets it receives with the pooled artifact type and coverage scope. It
cannot detect these sets, so it must not be used to convert them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike

from ecg_trust.conformal.multilabel import (
    BinaryPredictionSets,
    ConformalValidationError,
    LabelwiseBinaryConformal,
)

_ARTIFACT_TYPE = "ecg_trust.group_conditional_conformal"
_SCHEMA_VERSION = 1
_COVERAGE_SCOPE = "labelwise_marginal_within_each_group_under_exchangeability"


def _groups(values: Sequence[str], count: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ConformalValidationError("groups must be a sequence of group names")
    groups = tuple(values)
    if len(groups) != count:
        raise ConformalValidationError("groups must provide one group name per row")
    for group in groups:
        if not isinstance(group, str) or not group.strip():
            raise ConformalValidationError("group names must be non-empty strings")
    return groups


def _array(values: ArrayLike, name: str) -> np.ndarray:
    try:
        return np.asarray(values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ConformalValidationError(f"{name} must be a rectangular numeric array") from error


def _row_count(matrix: np.ndarray) -> int:
    if matrix.ndim != 2:
        raise ConformalValidationError("probabilities must be a two-dimensional matrix")
    return int(matrix.shape[0])


@dataclass(frozen=True, slots=True)
class GroupConditionalConformal:
    """One frozen label-wise conformal model per declared group."""

    models: Mapping[str, LabelwiseBinaryConformal]

    def __post_init__(self) -> None:
        if not isinstance(self.models, Mapping) or not self.models:
            raise ConformalValidationError("at least one group model is required")
        for group, model in self.models.items():
            if not isinstance(group, str) or not group.strip():
                raise ConformalValidationError("group names must be non-empty strings")
            if not isinstance(model, LabelwiseBinaryConformal):
                raise ConformalValidationError("group models must be LabelwiseBinaryConformal")
        names = {model.label_names for model in self.models.values()}
        alphas = {model.alpha for model in self.models.values()}
        if len(names) != 1 or len(alphas) != 1:
            raise ConformalValidationError("group models must share label names and alpha")
        # Snapshot into a read-only view so callers cannot swap group models later.
        object.__setattr__(self, "models", MappingProxyType(dict(sorted(self.models.items()))))

    @property
    def label_names(self) -> tuple[str, ...]:
        return next(iter(self.models.values())).label_names

    @property
    def trivial_groups(self) -> tuple[str, ...]:
        """Groups too small for any threshold below one (every set is ``{0, 1}``)."""

        return tuple(
            group
            for group, model in self.models.items()
            if model.quantile_rank > model.n_calibration_samples
        )

    @classmethod
    def fit(
        cls,
        probabilities: ArrayLike,
        targets: ArrayLike,
        groups: Sequence[str],
        *,
        label_names: Sequence[str],
        alpha: float = 0.1,
    ) -> GroupConditionalConformal:
        """Fit a separate label-wise conformal model inside each group."""

        matrix = _array(probabilities, "probabilities")
        target_matrix = _array(targets, "targets")
        names = _groups(groups, _row_count(matrix))
        if target_matrix.ndim < 1 or target_matrix.shape[0] != len(names):
            raise ConformalValidationError("targets must provide one row per group name")
        models: dict[str, LabelwiseBinaryConformal] = {}
        for group in sorted(set(names)):
            rows = np.fromiter((name == group for name in names), dtype=bool, count=len(names))
            models[group] = LabelwiseBinaryConformal.fit(
                matrix[rows],
                target_matrix[rows],
                label_names=label_names,
                alpha=alpha,
            )
        return cls(models=models)

    def predict(self, probabilities: ArrayLike, groups: Sequence[str]) -> BinaryPredictionSets:
        """Apply each row's own group thresholds; unknown groups are refused."""

        matrix = _array(probabilities, "probabilities")
        names = _groups(groups, _row_count(matrix))
        unknown = sorted(set(names) - set(self.models))
        if unknown:
            raise ConformalValidationError(f"groups were not calibrated: {unknown}")
        include_not_supported = np.zeros((len(names), len(self.label_names)), dtype=bool)
        include_supported = np.zeros_like(include_not_supported)
        for group, model in self.models.items():
            rows = np.fromiter((name == group for name in names), dtype=bool, count=len(names))
            if not rows.any():
                continue
            sets = model.predict(matrix[rows])
            include_not_supported[rows] = np.asarray(sets.include_not_supported, dtype=bool)
            include_supported[rows] = np.asarray(sets.include_supported, dtype=bool)
        return BinaryPredictionSets.from_masks(
            label_names=self.label_names,
            include_not_supported=include_not_supported,
            include_supported=include_supported,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "artifact_type": _ARTIFACT_TYPE,
            "coverage_scope": _COVERAGE_SCOPE,
            "groups": {group: model.to_dict() for group, model in self.models.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> GroupConditionalConformal:
        """Validate and restore a serialized group-conditional artifact."""

        if not isinstance(payload, Mapping) or set(payload) != {
            "schema_version",
            "artifact_type",
            "coverage_scope",
            "groups",
        }:
            raise ConformalValidationError("group-conditional conformal artifact keys are invalid")
        version = payload["schema_version"]
        if type(version) is not int or version != _SCHEMA_VERSION:
            raise ConformalValidationError("unsupported schema_version")
        if payload["artifact_type"] != _ARTIFACT_TYPE:
            raise ConformalValidationError("unexpected artifact_type")
        if payload["coverage_scope"] != _COVERAGE_SCOPE:
            raise ConformalValidationError("unsupported conformal coverage_scope")
        groups = payload["groups"]
        if not isinstance(groups, Mapping):
            raise ConformalValidationError("groups must map group names to artifacts")
        models: dict[str, LabelwiseBinaryConformal] = {}
        for group, artifact in groups.items():
            if not isinstance(artifact, Mapping):
                raise ConformalValidationError("each group artifact must be a mapping")
            models[group] = LabelwiseBinaryConformal.from_dict(artifact)
        return cls(models=models)


__all__ = ["GroupConditionalConformal"]
