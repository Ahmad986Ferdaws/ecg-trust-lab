"""Immutable experiment protocol and final-test access guardrails.

The PTB-XL benchmark folds are part of the scientific protocol, not a tuning
parameter.  This module keeps their roles and the output-label order fixed,
produces a deterministic protocol fingerprint, and makes access to fold 10 an
explicit action.  The gate is intended to prevent accidental test-set use; it
is not a security boundary against deliberately bypassing Python code.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self, cast

import yaml  # type: ignore[import-untyped]

from ecg_trust.constants import PTBXL_VERSION, SUPERCLASSES

PROTOCOL_SCHEMA_VERSION = 1
PROTOCOL_ID = "ptbxl-superclass-trust-v1"
DATASET_NAME = "PTB-XL"
TASK_KIND = "multilabel_classification"

LABEL_ORDER: tuple[str, ...] = SUPERCLASSES
ALL_FOLDS: tuple[int, ...] = tuple(range(1, 11))
TRAIN_FOLDS: tuple[int, ...] = tuple(range(1, 8))
MODEL_SELECTION_FOLDS: tuple[int, ...] = (8,)
CALIBRATION_FOLDS: tuple[int, ...] = (9,)
FINAL_TEST_FOLDS: tuple[int, ...] = (10,)

FINAL_TEST_CONFIRMATION = "I understand fold 10 is the one-time final test set."

DEFAULT_PROTOCOL_PATH = Path(__file__).resolve().parents[2] / "configs" / "protocol.yaml"


class ProtocolValidationError(ValueError):
    """Raised when a protocol violates the fixed experimental contract."""


class FinalTestAccessError(PermissionError):
    """Raised when code attempts to access fold 10 without an explicit token."""


class FoldRole(StrEnum):
    """The only roles a PTB-XL benchmark fold may have in this project."""

    TRAIN = "train"
    MODEL_SELECTION = "model_selection"
    CALIBRATION = "calibration"
    FINAL_TEST = "final_test"


_TOKEN_SEAL = object()


class FinalTestAccessToken:
    """Opaque, immutable, process-local authorization for final-test access."""

    __slots__ = ("_protocol_hash", "_purpose", "_seal")
    _protocol_hash: str
    _purpose: str
    _seal: object

    def __init__(self, protocol_hash: str, purpose: str, *, _seal: object) -> None:
        if _seal is not _TOKEN_SEAL:
            raise TypeError(
                "FinalTestAccessToken cannot be constructed directly; "
                "use authorize_final_test_access()."
            )
        object.__setattr__(self, "_protocol_hash", protocol_hash)
        object.__setattr__(self, "_purpose", purpose)
        object.__setattr__(self, "_seal", _seal)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("FinalTestAccessToken is immutable")

    @property
    def protocol_hash(self) -> str:
        """Fingerprint of the protocol for which this token was issued."""

        return self._protocol_hash

    @property
    def purpose(self) -> str:
        """Human-readable audit reason supplied when the token was issued."""

        return self._purpose

    def _is_valid_for(self, protocol_hash: str) -> bool:
        return self._seal is _TOKEN_SEAL and self._protocol_hash == protocol_hash

    def __repr__(self) -> str:
        return (
            "FinalTestAccessToken("
            f"protocol_hash={self._protocol_hash!r}, purpose={self._purpose!r})"
        )


@dataclass(frozen=True, slots=True, init=False)
class ExperimentProtocol:
    """Resolved, validated, and immutable PTB-XL experiment protocol."""

    schema_version: int
    protocol_id: str
    dataset_name: str
    dataset_version: str
    task_kind: str
    label_order: tuple[str, ...]
    _all_folds: tuple[int, ...]
    _train_folds: tuple[int, ...]
    _model_selection_folds: tuple[int, ...]
    _calibration_folds: tuple[int, ...]
    _final_test_folds: tuple[int, ...]

    @classmethod
    def _create(
        cls,
        *,
        schema_version: int,
        protocol_id: str,
        dataset_name: str,
        dataset_version: str,
        task_kind: str,
        label_order: tuple[str, ...],
        all_folds: tuple[int, ...],
        train_folds: tuple[int, ...],
        model_selection_folds: tuple[int, ...],
        calibration_folds: tuple[int, ...],
        final_test_folds: tuple[int, ...],
    ) -> Self:
        instance = object.__new__(cls)
        object.__setattr__(instance, "schema_version", schema_version)
        object.__setattr__(instance, "protocol_id", protocol_id)
        object.__setattr__(instance, "dataset_name", dataset_name)
        object.__setattr__(instance, "dataset_version", dataset_version)
        object.__setattr__(instance, "task_kind", task_kind)
        object.__setattr__(instance, "label_order", label_order)
        object.__setattr__(instance, "_all_folds", all_folds)
        object.__setattr__(instance, "_train_folds", train_folds)
        object.__setattr__(instance, "_model_selection_folds", model_selection_folds)
        object.__setattr__(instance, "_calibration_folds", calibration_folds)
        object.__setattr__(instance, "_final_test_folds", final_test_folds)
        instance._validate()
        return instance

    @classmethod
    def canonical(cls) -> Self:
        """Construct the code-defined canonical protocol."""

        return cls._create(
            schema_version=PROTOCOL_SCHEMA_VERSION,
            protocol_id=PROTOCOL_ID,
            dataset_name=DATASET_NAME,
            dataset_version=PTBXL_VERSION,
            task_kind=TASK_KIND,
            label_order=LABEL_ORDER,
            all_folds=ALL_FOLDS,
            train_folds=TRAIN_FOLDS,
            model_selection_folds=MODEL_SELECTION_FOLDS,
            calibration_folds=CALIBRATION_FOLDS,
            final_test_folds=FINAL_TEST_FOLDS,
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        """Resolve and validate a protocol mapping.

        A mapping produced by :meth:`to_resolved_dict` can be loaded again.  If
        it contains ``protocol_hash``, that hash is verified before returning.
        """

        root = _expect_mapping(raw, "protocol")
        _expect_keys(
            root,
            required={"schema_version", "protocol_id", "dataset", "task", "folds"},
            optional={"protocol_hash"},
            context="protocol",
        )

        dataset = _expect_mapping(root["dataset"], "protocol.dataset")
        _expect_keys(dataset, required={"name", "version"}, context="protocol.dataset")

        task = _expect_mapping(root["task"], "protocol.task")
        _expect_keys(task, required={"kind", "label_order"}, context="protocol.task")

        folds = _expect_mapping(root["folds"], "protocol.folds")
        _expect_keys(folds, required={"universe", "roles"}, context="protocol.folds")
        roles = _expect_mapping(folds["roles"], "protocol.folds.roles")
        _expect_keys(
            roles,
            required={role.value for role in FoldRole},
            context="protocol.folds.roles",
        )

        instance = cls._create(
            schema_version=_expect_int(root["schema_version"], "protocol.schema_version"),
            protocol_id=_expect_string(root["protocol_id"], "protocol.protocol_id"),
            dataset_name=_expect_string(dataset["name"], "protocol.dataset.name"),
            dataset_version=_expect_string(dataset["version"], "protocol.dataset.version"),
            task_kind=_expect_string(task["kind"], "protocol.task.kind"),
            label_order=_expect_string_tuple(task["label_order"], "protocol.task.label_order"),
            all_folds=_expect_fold_tuple(folds["universe"], "protocol.folds.universe"),
            train_folds=_expect_fold_tuple(
                roles[FoldRole.TRAIN.value], "protocol.folds.roles.train"
            ),
            model_selection_folds=_expect_fold_tuple(
                roles[FoldRole.MODEL_SELECTION.value],
                "protocol.folds.roles.model_selection",
            ),
            calibration_folds=_expect_fold_tuple(
                roles[FoldRole.CALIBRATION.value],
                "protocol.folds.roles.calibration",
            ),
            final_test_folds=_expect_fold_tuple(
                roles[FoldRole.FINAL_TEST.value], "protocol.folds.roles.final_test"
            ),
        )

        supplied_hash = root.get("protocol_hash")
        if supplied_hash is not None:
            expected_hash = _expect_string(supplied_hash, "protocol.protocol_hash")
            if expected_hash != instance.protocol_hash:
                raise ProtocolValidationError(
                    "protocol.protocol_hash does not match the resolved protocol: "
                    f"expected {instance.protocol_hash!r}, received {expected_hash!r}"
                )
        return instance

    @classmethod
    def from_json(cls, serialized: str) -> Self:
        """Deserialize a protocol from JSON and verify an embedded hash."""

        try:
            raw: object = json.loads(serialized)
        except json.JSONDecodeError as exc:
            raise ProtocolValidationError(f"invalid protocol JSON: {exc}") from exc
        return cls.from_mapping(_expect_mapping(raw, "protocol"))

    @property
    def all_folds(self) -> tuple[int, ...]:
        """All benchmark folds; use :meth:`guard_fold_access` before data access."""

        return self._all_folds

    @property
    def development_folds(self) -> tuple[int, ...]:
        """Folds permitted during model development (1 through 9)."""

        return self._train_folds + self._model_selection_folds + self._calibration_folds

    def folds_for(
        self,
        role: FoldRole | str,
        *,
        test_access: FinalTestAccessToken | None = None,
    ) -> tuple[int, ...]:
        """Return folds assigned to ``role``, enforcing the fold-10 gate."""

        try:
            resolved_role = role if isinstance(role, FoldRole) else FoldRole(role)
        except ValueError as exc:
            valid = ", ".join(item.value for item in FoldRole)
            raise ProtocolValidationError(
                f"unknown fold role {role!r}; expected one of: {valid}"
            ) from exc

        role_folds = {
            FoldRole.TRAIN: self._train_folds,
            FoldRole.MODEL_SELECTION: self._model_selection_folds,
            FoldRole.CALIBRATION: self._calibration_folds,
            FoldRole.FINAL_TEST: self._final_test_folds,
        }[resolved_role]
        return self.guard_fold_access(role_folds, test_access=test_access)

    def guard_fold_access(
        self,
        requested_folds: Iterable[int],
        *,
        test_access: FinalTestAccessToken | None = None,
    ) -> tuple[int, ...]:
        """Validate fold requests and reject fold 10 unless explicitly unlocked.

        Data-loading and evaluation entry points should call this method before
        resolving record paths.  The returned tuple is normalized and safe for
        downstream use under the supplied authorization.
        """

        folds = _coerce_requested_folds(requested_folds)
        unknown = set(folds).difference(self._all_folds)
        if unknown:
            raise ProtocolValidationError(
                f"requested folds are outside the protocol: {sorted(unknown)}"
            )
        if set(folds).intersection(self._final_test_folds) and (
            not isinstance(test_access, FinalTestAccessToken)
            or not test_access._is_valid_for(self.protocol_hash)
        ):
            raise FinalTestAccessError(
                "fold 10 is sealed; obtain a protocol-bound token with "
                "authorize_final_test_access() and pass it as test_access"
            )
        return folds

    @property
    def protocol_hash(self) -> str:
        """Self-describing SHA-256 fingerprint of the canonical payload."""

        digest = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    def canonical_json(self) -> str:
        """Serialize the hash payload deterministically, without whitespace."""

        return json.dumps(
            self._payload(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def to_resolved_dict(self) -> dict[str, object]:
        """Return a JSON-compatible, fully resolved mapping with its hash."""

        resolved = self._payload()
        resolved["protocol_hash"] = self.protocol_hash
        return resolved

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the resolved protocol to stable JSON."""

        serialized = json.dumps(
            self.to_resolved_dict(), ensure_ascii=True, indent=indent, sort_keys=True
        )
        return f"{serialized}\n" if indent is not None else serialized

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "dataset": {
                "name": self.dataset_name,
                "version": self.dataset_version,
            },
            "task": {
                "kind": self.task_kind,
                "label_order": list(self.label_order),
            },
            "folds": {
                "universe": list(self._all_folds),
                "roles": {
                    FoldRole.TRAIN.value: list(self._train_folds),
                    FoldRole.MODEL_SELECTION.value: list(self._model_selection_folds),
                    FoldRole.CALIBRATION.value: list(self._calibration_folds),
                    FoldRole.FINAL_TEST.value: list(self._final_test_folds),
                },
            },
        }

    def _validate(self) -> None:
        if self.schema_version != PROTOCOL_SCHEMA_VERSION:
            raise ProtocolValidationError(
                f"schema_version must be {PROTOCOL_SCHEMA_VERSION}, received {self.schema_version}"
            )
        if not self.protocol_id.strip():
            raise ProtocolValidationError("protocol_id must not be blank")
        if self.dataset_name != DATASET_NAME:
            raise ProtocolValidationError(
                f"dataset.name must be {DATASET_NAME!r}, received {self.dataset_name!r}"
            )
        if self.dataset_version != PTBXL_VERSION:
            raise ProtocolValidationError(
                f"dataset.version must be {PTBXL_VERSION!r}, received {self.dataset_version!r}"
            )
        if self.task_kind != TASK_KIND:
            raise ProtocolValidationError(
                f"task.kind must be {TASK_KIND!r}, received {self.task_kind!r}"
            )
        if self.label_order != LABEL_ORDER:
            raise ProtocolValidationError(
                "task.label_order is immutable and must be exactly "
                f"{list(LABEL_ORDER)!r}; received {list(self.label_order)!r}"
            )

        role_folds = {
            FoldRole.TRAIN: self._train_folds,
            FoldRole.MODEL_SELECTION: self._model_selection_folds,
            FoldRole.CALIBRATION: self._calibration_folds,
            FoldRole.FINAL_TEST: self._final_test_folds,
        }
        for role, folds in role_folds.items():
            if not folds:
                raise ProtocolValidationError(f"fold role {role.value!r} must not be empty")
            duplicates = _duplicates(folds)
            if duplicates:
                raise ProtocolValidationError(
                    f"fold role {role.value!r} contains duplicates: {duplicates}"
                )

        seen: dict[int, FoldRole] = {}
        overlaps: list[str] = []
        for role, folds in role_folds.items():
            for fold in folds:
                previous = seen.get(fold)
                if previous is not None:
                    overlaps.append(f"fold {fold}: {previous.value}/{role.value}")
                else:
                    seen[fold] = role
        if overlaps:
            raise ProtocolValidationError(
                "fold roles must be disjoint; overlaps: " + ", ".join(overlaps)
            )

        role_union = set(seen)
        universe = set(self._all_folds)
        missing = sorted(universe.difference(role_union))
        unexpected = sorted(role_union.difference(universe))
        if missing or unexpected:
            raise ProtocolValidationError(
                "fold roles must exhaust the fold universe; "
                f"missing={missing}, unexpected={unexpected}"
            )

        if self._all_folds != ALL_FOLDS:
            raise ProtocolValidationError(f"fold universe is immutable and must be {ALL_FOLDS!r}")
        expected_roles = {
            FoldRole.TRAIN: TRAIN_FOLDS,
            FoldRole.MODEL_SELECTION: MODEL_SELECTION_FOLDS,
            FoldRole.CALIBRATION: CALIBRATION_FOLDS,
            FoldRole.FINAL_TEST: FINAL_TEST_FOLDS,
        }
        for role, expected in expected_roles.items():
            if role_folds[role] != expected:
                raise ProtocolValidationError(
                    f"fold role {role.value!r} is immutable and must be {expected!r}; "
                    f"received {role_folds[role]!r}"
                )


def authorize_final_test_access(
    protocol: ExperimentProtocol,
    *,
    purpose: str,
    confirmation: str,
) -> FinalTestAccessToken:
    """Issue an explicit, protocol-bound token for the one-time final test.

    Callers must provide a meaningful purpose and repeat
    :data:`FINAL_TEST_CONFIRMATION`.  This deliberate friction prevents fold 10
    from becoming part of ordinary model-development code paths.
    """

    normalized_purpose = purpose.strip()
    if not normalized_purpose:
        raise FinalTestAccessError("a non-empty final-test access purpose is required")
    if confirmation != FINAL_TEST_CONFIRMATION:
        raise FinalTestAccessError("final-test confirmation did not match FINAL_TEST_CONFIRMATION")
    return FinalTestAccessToken(
        protocol.protocol_hash,
        normalized_purpose,
        _seal=_TOKEN_SEAL,
    )


def load_protocol(path: str | Path = DEFAULT_PROTOCOL_PATH) -> ExperimentProtocol:
    """Load and validate a YAML or JSON experiment protocol."""

    protocol_path = Path(path)
    try:
        serialized = protocol_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProtocolValidationError(
            f"could not read protocol file {protocol_path}: {exc}"
        ) from exc
    try:
        raw: object = yaml.safe_load(serialized)
    except yaml.YAMLError as exc:
        raise ProtocolValidationError(f"invalid protocol YAML in {protocol_path}: {exc}") from exc
    return ExperimentProtocol.from_mapping(_expect_mapping(raw, str(protocol_path)))


def _expect_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolValidationError(f"{context} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ProtocolValidationError(f"{context} keys must all be strings")
    return cast(Mapping[str, object], value)


def _expect_keys(
    mapping: Mapping[str, object],
    *,
    required: set[str],
    context: str,
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    present = set(mapping)
    missing = sorted(required.difference(present))
    unknown = sorted(present.difference(required | optional))
    if missing or unknown:
        raise ProtocolValidationError(
            f"{context} has invalid keys; missing={missing}, unknown={unknown}"
        )


def _expect_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolValidationError(f"{context} must be a non-empty string")
    return value


def _expect_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolValidationError(f"{context} must be an integer")
    return value


def _expect_string_tuple(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ProtocolValidationError(f"{context} must be a list of strings")
    if not all(isinstance(item, str) for item in value):
        raise ProtocolValidationError(f"{context} must contain only strings")
    return tuple(cast(list[str], value))


def _expect_fold_tuple(value: object, context: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ProtocolValidationError(f"{context} must be a list of integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ProtocolValidationError(f"{context} must contain only integers")
    folds = tuple(cast(list[int], value))
    if any(fold <= 0 for fold in folds):
        raise ProtocolValidationError(f"{context} must contain positive fold numbers")
    return folds


def _coerce_requested_folds(requested_folds: Iterable[int]) -> tuple[int, ...]:
    folds = tuple(requested_folds)
    if not folds:
        raise ProtocolValidationError("at least one fold must be requested")
    if any(isinstance(fold, bool) or not isinstance(fold, int) for fold in folds):
        raise ProtocolValidationError("requested folds must contain only integers")
    duplicates = _duplicates(folds)
    if duplicates:
        raise ProtocolValidationError(f"requested folds contain duplicates: {duplicates}")
    return folds


def _duplicates(values: tuple[int, ...]) -> list[int]:
    seen: set[int] = set()
    duplicates: set[int] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)
