"""Integrity-bound publication outputs for the frozen PTB-XL evaluation.

This module has two deliberately separate operations.  ``probability`` derives
descriptive probability, reliability, selective-risk, and subgroup artifacts
from already sealed fold-10 outputs.  ``finalize`` will not write a report until
the probability, robustness, explanations, and demo artifacts all verify.
Neither operation fits or changes a model, calibrator, threshold, or gate.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from ecg_trust.decisioning import CalibrationDecisionArtifact, load_calibration_decisions
from ecg_trust.evaluation import compute_multilabel_metrics, stable_sigmoid
from ecg_trust.post_analysis import derive_probability_audit
from ecg_trust.predictions import PredictionArtifact, load_prediction_artifact
from ecg_trust.protocol import (
    FINAL_TEST_CONFIRMATION,
    LABEL_ORDER,
    ExperimentProtocol,
    authorize_final_test_access,
)
from ecg_trust.subgroup_artifact import SubgroupArtifact, load_subgroup_artifact

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ecg_trust.post_evaluation import PostEvaluationSpec

PROBABILITY_AUDIT_SCHEMA_VERSION = 1
PROBABILITY_AUDIT_TYPE = "ecg_trust.post_evaluation_probability_audit"
DERIVED_MANIFEST_SCHEMA_VERSION = 1
DERIVED_MANIFEST_TYPE = "ecg_trust.post_evaluation_derived_manifest"
RELIABILITY_BINS = 15

TABLE_FILENAMES: tuple[str, ...] = (
    "member_metrics.csv",
    "architecture_metrics.csv",
    "paired_deltas.csv",
    "reliability_seed2026.csv",
    "mean_seed_risk_coverage.csv",
    "subgroup_metrics.csv",
    "subgroup_coverage.csv",
)
FIGURE_FILENAMES: tuple[str, ...] = (
    "architecture_member_metrics.png",
    "paired_deltas_ci.png",
    "raw_vs_calibrated_reliability_seed2026.png",
    "mean_seed_risk_coverage.png",
    "subgroup_performance_coverage.png",
)

FloatArray = NDArray[np.float64]
_PYPLOT: Any | None = None


class FinalResultsError(ValueError):
    """Raised when publication output violates the frozen contract."""


class FinalResultsIntegrityError(FinalResultsError):
    """Raised when a source or derived artifact fails verification."""


@dataclass(frozen=True, slots=True)
class ProbabilityRenderResult:
    """Committed probability-audit publication files."""

    audit_path: Path
    audit_artifact_sha256: str
    files: tuple[Mapping[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": "probability",
            "probability_audit": {
                "path": str(self.audit_path),
                "artifact_sha256": self.audit_artifact_sha256,
                "file_sha256": sha256_file(self.audit_path),
            },
            "derived_files": [dict(item) for item in self.files],
        }


@dataclass(frozen=True, slots=True)
class FinalizationResult:
    """Committed final narrative and all-derived-files manifest."""

    report_path: Path
    manifest_path: Path
    manifest_artifact_sha256: str
    file_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": "finalize",
            "final_results": {
                "path": str(self.report_path),
                "file_sha256": sha256_file(self.report_path),
            },
            "derived_manifest": {
                "path": str(self.manifest_path),
                "artifact_sha256": self.manifest_artifact_sha256,
                "file_sha256": sha256_file(self.manifest_path),
                "file_count": self.file_count,
            },
        }


@dataclass(frozen=True, slots=True)
class _VerifiedContext:
    spec: PostEvaluationSpec
    payload: Mapping[str, object]
    root: Path
    artifacts: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _MemberAnalysis:
    member_id: str
    architecture: str
    seed: int
    prediction: PredictionArtifact
    decision: CalibrationDecisionArtifact
    report: Mapping[str, object]
    audit: Mapping[str, object]


def canonical_sha256(payload: Mapping[str, object]) -> str:
    """Return a prefixed SHA-256 over finite canonical JSON."""

    try:
        serialized = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise FinalResultsError("publication payload must contain finite JSON") from error
    return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash one required regular file."""

    source = Path(path)
    if not source.is_file():
        raise FinalResultsIntegrityError(f"required file is missing: {source}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as error:
        raise FinalResultsIntegrityError(f"could not hash {source}: {error}") from error
    return "sha256:" + digest.hexdigest()


def render_probability_results(
    spec_path: str | Path,
    *,
    protocol: ExperimentProtocol,
    verify_git: bool = True,
) -> ProbabilityRenderResult:
    """Derive and immutably publish probability tables and figures."""

    context = _load_context(spec_path, protocol=protocol, verify_git=verify_git)
    destinations = _publication_destinations(context)
    audit_path = _artifact_path(context, "probability_audit")
    if audit_path.exists():
        committed = load_and_verify_probability_audit(audit_path, context=context)
        return ProbabilityRenderResult(
            audit_path=audit_path,
            audit_artifact_sha256=_hash(committed["artifact_sha256"], "probability audit hash"),
            files=_probability_file_bindings(committed),
        )

    subgroup = _load_bound_subgroups(context, protocol=protocol)
    analyses = _derive_member_analyses(context, protocol=protocol, subgroup=subgroup)
    architecture = _load_architecture_summaries(context)
    paired = _load_paired_reports(context)

    table_bytes = _render_tables(analyses, architecture, paired)
    figure_bytes = _render_figures(analyses, architecture, paired, output_root=context.root)
    rendered = {**table_bytes, **figure_bytes}
    if set(rendered) != set(destinations):
        raise FinalResultsIntegrityError("renderer did not produce the canonical file set")

    bindings: list[dict[str, str]] = []
    for filename in (*TABLE_FILENAMES, *FIGURE_FILENAMES):
        destination = destinations[filename]
        content = rendered[filename]
        bindings.append(
            {
                "path": str(destination.relative_to(context.root)),
                "file_sha256": _sha256_bytes(content),
                "media_type": "text/csv" if filename.endswith(".csv") else "image/png",
            }
        )

    body: dict[str, object] = {
        "schema_version": PROBABILITY_AUDIT_SCHEMA_VERSION,
        "artifact_type": PROBABILITY_AUDIT_TYPE,
        "post_evaluation_spec_sha256": context.spec.artifact_sha256,
        "analysis_class": "post_evaluation_descriptive_not_confirmatory",
        "reliability_bins": RELIABILITY_BINS,
        "label_order": list(LABEL_ORDER),
        "subgroup_source_sha256": subgroup.artifact_sha256,
        "members": {
            item.member_id: {
                "architecture": item.architecture,
                "seed": item.seed,
                "sources": {
                    "prediction_artifact_sha256": item.prediction.integrity_sha256,
                    "prediction_alignment_sha256": item.prediction.alignment_sha256,
                    "calibration_decision_sha256": item.decision.integrity_sha256,
                },
                "analysis": dict(item.audit),
            }
            for item in analyses
        },
        "publication_files": bindings,
    }
    artifact_sha256 = canonical_sha256(body)
    audit_payload = {**body, "artifact_sha256": artifact_sha256}
    audit_content = _json_bytes(audit_payload)

    # All bytes are complete and hashed before the first immutable publish.  Exact
    # pre-audit files from an interrupted prior attempt are safely adopted.
    _preflight_exact_content(
        {destinations[name]: rendered[name] for name in rendered}, root=context.root
    )
    for filename in (*TABLE_FILENAMES, *FIGURE_FILENAMES):
        _ensure_exact_bytes(destinations[filename], rendered[filename], root=context.root)
    _ensure_exact_bytes(audit_path, audit_content, root=context.root)
    return ProbabilityRenderResult(
        audit_path=audit_path,
        audit_artifact_sha256=artifact_sha256,
        files=tuple(bindings),
    )


def finalize_results(
    spec_path: str | Path,
    *,
    protocol: ExperimentProtocol,
    verify_git: bool = True,
) -> FinalizationResult:
    """Publish final results only after every post-evaluation branch verifies."""

    context = _load_context(spec_path, protocol=protocol, verify_git=verify_git)
    report_path = _artifact_path(context, "final_results_markdown")
    manifest_path = _artifact_path(context, "derived_manifest")
    if manifest_path.exists() and not report_path.is_file():
        raise FinalResultsIntegrityError(
            "derived manifest exists without the required final report"
        )

    probability = load_and_verify_probability_audit(
        _artifact_path(context, "probability_audit"), context=context
    )
    robustness_path = _artifact_path(context, "robustness_manifest")
    robustness = _load_completed_branch_manifest(
        robustness_path,
        context=context,
        branch="robustness",
    )
    robustness = _verify_canonical_branch_manifest(
        robustness_path, context=context, branch="robustness", generic=robustness
    )
    explanations_path = _artifact_path(context, "explanations_manifest")
    explanations = _load_completed_branch_manifest(
        explanations_path,
        context=context,
        branch="explanation",
    )
    explanations = _verify_canonical_branch_manifest(
        explanations_path,
        context=context,
        branch="explanation",
        generic=explanations,
    )
    robustness_summary = _summarize_robustness(robustness)
    explanations_summary = _summarize_explanations(explanations)
    demo = _load_demo_binding(context, protocol=protocol)
    run_log_path, run_log_sha256 = _load_operational_run_log(context)
    deviations = _mapping(
        _mapping(context.payload["sealed_evaluation"], "sealed_evaluation")["protocol_deviations"],
        "protocol_deviations",
    )
    deviations_path = Path(_string(deviations["path"], "protocol deviation path")).resolve()
    if sha256_file(deviations_path) != _hash(
        deviations["file_sha256"], "protocol deviation file hash"
    ):
        raise FinalResultsIntegrityError("bound protocol-deviation log changed")
    if "DEV-001" not in _read_text(deviations_path):
        raise FinalResultsIntegrityError("bound protocol-deviation log lacks DEV-001")

    architecture = _load_architecture_summaries(context)
    paired = _load_paired_reports(context)
    markdown = _build_final_results_markdown(
        context=context,
        probability=probability,
        architecture=architecture,
        paired=paired,
        robustness_sha256=_hash(robustness["artifact_sha256"], "robustness hash"),
        explanations_sha256=_hash(explanations["artifact_sha256"], "explanations hash"),
        demo_sha256=demo.binding_artifact_sha256,
        robustness_summary=robustness_summary,
        explanations_summary=explanations_summary,
        run_log_path=run_log_path,
        run_log_sha256=run_log_sha256,
        deviations_path=deviations_path,
    )
    _ensure_exact_bytes(report_path, markdown.encode("utf-8"), root=context.root)

    file_bindings = _all_root_file_bindings(context.root, excluding={manifest_path})
    body: dict[str, object] = {
        "schema_version": DERIVED_MANIFEST_SCHEMA_VERSION,
        "artifact_type": DERIVED_MANIFEST_TYPE,
        "post_evaluation_spec_sha256": context.spec.artifact_sha256,
        "self_hash_scope": "canonical_json_body_excluding_artifact_sha256",
        "source_anchors": {
            "audit_spec": {
                "path": str(cast(Path, context.spec.path)),
                "artifact_sha256": context.spec.artifact_sha256,
                "file_sha256": sha256_file(cast(Path, context.spec.path)),
            },
            "sealed_final_batch": _mapping(
                _mapping(context.payload["sealed_evaluation"], "sealed_evaluation")[
                    "final_batch_summary"
                ],
                "final_batch_summary",
            ),
            "protocol_deviations": {
                "path": str(deviations_path),
                "file_sha256": sha256_file(deviations_path),
                "disclosure": "DEV-001",
            },
            "operational_run_log": {
                "path": str(run_log_path),
                "file_sha256": run_log_sha256,
                "classification": "post_completion_operational_disclosure",
            },
        },
        "verified_prerequisites": {
            "probability_audit_sha256": probability["artifact_sha256"],
            "robustness_manifest_sha256": robustness["artifact_sha256"],
            "explanations_manifest_sha256": explanations["artifact_sha256"],
            "demo_binding_sha256": demo.binding_artifact_sha256,
        },
        "files": file_bindings,
    }
    artifact_sha256 = canonical_sha256(body)
    manifest_payload = {**body, "artifact_sha256": artifact_sha256}
    manifest_content = _json_bytes(manifest_payload)
    if manifest_path.exists():
        committed = _load_self_hashed_json(manifest_path, expected_type=DERIVED_MANIFEST_TYPE)
        if dict(committed) != manifest_payload:
            raise FinalResultsIntegrityError(
                "existing derived manifest differs from fully reverified state"
            )
    else:
        _ensure_exact_bytes(manifest_path, manifest_content, root=context.root)
    return FinalizationResult(
        report_path=report_path,
        manifest_path=manifest_path,
        manifest_artifact_sha256=artifact_sha256,
        file_count=len(file_bindings),
    )


def load_and_verify_probability_audit(
    path: str | Path,
    *,
    context: _VerifiedContext | None = None,
    protocol: ExperimentProtocol | None = None,
    spec_path: str | Path | None = None,
    verify_git: bool = True,
) -> Mapping[str, object]:
    """Load the probability audit and verify every declared publication file."""

    source = Path(path).resolve()
    if context is None:
        if protocol is None:
            raise TypeError("protocol is required when no verified context is supplied")
        context = _load_context(
            source.parent / "audit_spec.json" if spec_path is None else spec_path,
            protocol=protocol,
            verify_git=verify_git,
        )
    if source != _artifact_path(context, "probability_audit"):
        raise FinalResultsIntegrityError("probability audit is not at its canonical path")
    root = _load_self_hashed_json(source, expected_type=PROBABILITY_AUDIT_TYPE)
    if root.get("schema_version") != PROBABILITY_AUDIT_SCHEMA_VERSION:
        raise FinalResultsIntegrityError("unsupported probability audit schema version")
    if root.get("post_evaluation_spec_sha256") != context.spec.artifact_sha256:
        raise FinalResultsIntegrityError("probability audit binds a different specification")
    if (
        root.get("analysis_class") != "post_evaluation_descriptive_not_confirmatory"
        or root.get("reliability_bins") != RELIABILITY_BINS
        or list(_sequence(root.get("label_order"), "probability label_order")) != list(LABEL_ORDER)
    ):
        raise FinalResultsIntegrityError("probability analysis contract differs")
    members = _mapping(root.get("members"), "probability members")
    if set(members) != set(context.spec.member_ids):
        raise FinalResultsIntegrityError("probability audit member set differs")
    files = _sequence(root.get("publication_files"), "probability publication_files")
    expected = {
        str(path.relative_to(context.root)): path
        for path in _publication_destinations(context).values()
    }
    observed: set[str] = set()
    for raw in files:
        binding = _mapping(raw, "probability publication binding")
        relative = _string(binding.get("path"), "publication path")
        if relative in observed or relative not in expected:
            raise FinalResultsIntegrityError("probability publication file set differs")
        observed.add(relative)
        expected_hash = _hash(binding.get("file_sha256"), "publication file hash")
        if sha256_file(expected[relative]) != expected_hash:
            raise FinalResultsIntegrityError(f"publication file changed: {relative}")
    if observed != set(expected):
        raise FinalResultsIntegrityError("probability publication file set is incomplete")
    return root


def _load_context(
    spec_path: str | Path,
    *,
    protocol: ExperimentProtocol,
    verify_git: bool,
) -> _VerifiedContext:
    if not isinstance(protocol, ExperimentProtocol):
        raise TypeError("protocol must be an ExperimentProtocol")
    source = Path(spec_path).resolve()
    bootstrap = _read_json(source, "post-evaluation specification bootstrap")
    bootstrap_output = _mapping(bootstrap.get("output_contract"), "output_contract")
    declared_root = Path(_string(bootstrap_output.get("root"), "output root")).resolve()
    if source != declared_root / "audit_spec.json":
        raise FinalResultsIntegrityError(
            "specification must be located at its declared output root"
        )
    with _matplotlib_config_scope(declared_root):
        from ecg_trust.post_evaluation import load_post_evaluation_spec

        spec = load_post_evaluation_spec(
            source,
            protocol=protocol,
            verify_sources=True,
            verify_git=verify_git,
        )
    payload = spec.payload
    output = _mapping(payload["output_contract"], "output_contract")
    root = Path(_string(output["root"], "output root")).resolve()
    if root != spec.output_root.resolve():
        raise FinalResultsIntegrityError("spec output-root representations differ")
    artifacts = _mapping(output["artifacts"], "output artifacts")
    for name, value in artifacts.items():
        if name == "demo_directory" or name.endswith("_directory"):
            _ensure_within(Path(_string(value, name)).resolve(), root, allow_root=False)
        else:
            _ensure_within(Path(_string(value, name)).resolve(), root, allow_root=False)
    return _VerifiedContext(spec=spec, payload=payload, root=root, artifacts=artifacts)


@contextmanager
def _matplotlib_config_scope(root: Path) -> Iterator[None]:
    """Contain transitive Matplotlib font-cache writes inside the output root."""

    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".matplotlib-", dir=root) as raw_cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = raw_cache
        try:
            yield
        finally:
            if previous is None:
                del os.environ["MPLCONFIGDIR"]
            else:
                os.environ["MPLCONFIGDIR"] = previous


def _artifact_path(context: _VerifiedContext, name: str) -> Path:
    path = Path(_string(context.artifacts[name], f"output artifact {name}")).resolve()
    _ensure_within(path, context.root, allow_root=False)
    return path


def _publication_destinations(context: _VerifiedContext) -> dict[str, Path]:
    table_root = _artifact_path(context, "publication_tables_directory")
    figure_root = _artifact_path(context, "publication_figures_directory")
    destinations = {name: (table_root / name).resolve() for name in TABLE_FILENAMES}
    destinations.update({name: (figure_root / name).resolve() for name in FIGURE_FILENAMES})
    for path in destinations.values():
        _ensure_within(path, context.root, allow_root=False)
    return destinations


def _load_bound_subgroups(
    context: _VerifiedContext, *, protocol: ExperimentProtocol
) -> SubgroupArtifact:
    sealed = _mapping(context.payload["sealed_evaluation"], "sealed_evaluation")
    final_spec_binding = _mapping(sealed["final_evaluation_spec"], "final_evaluation_spec binding")
    final_spec = _read_bound_json(final_spec_binding, context="final evaluation spec")
    subgroup_binding = _mapping(final_spec.get("subgroup_artifact"), "subgroup artifact")
    subgroup_path = Path(_string(subgroup_binding.get("path"), "subgroup path")).resolve()
    if sha256_file(subgroup_path) != _hash(
        subgroup_binding.get("file_sha256"), "subgroup file hash"
    ):
        raise FinalResultsIntegrityError("frozen subgroup artifact changed")
    protocol_binding = _mapping(context.payload["protocol"], "protocol binding")
    loaded = load_subgroup_artifact(
        subgroup_path,
        protocol=protocol,
        expected_manifest_sha256=_hash(protocol_binding["manifest_sha256"], "manifest hash"),
        verify_source=True,
    )
    if loaded.artifact_sha256 != _hash(
        subgroup_binding.get("artifact_sha256"), "subgroup artifact hash"
    ):
        raise FinalResultsIntegrityError("subgroup artifact binding differs")
    return loaded


def _derive_member_analyses(
    context: _VerifiedContext,
    *,
    protocol: ExperimentProtocol,
    subgroup: SubgroupArtifact,
) -> tuple[_MemberAnalysis, ...]:
    access = authorize_final_test_access(
        protocol,
        purpose="Read-only frozen post-evaluation probability audit",
        confirmation=FINAL_TEST_CONFIRMATION,
    )
    protocol_binding = _mapping(context.payload["protocol"], "protocol binding")
    manifest_hash = _hash(protocol_binding["manifest_sha256"], "manifest hash")
    members: list[_MemberAnalysis] = []
    reference_alignment: str | None = None
    for raw_member in _sequence(context.payload["members"], "members"):
        member = _mapping(raw_member, "member")
        member_id = _string(member["member_id"], "member_id")
        architecture = _string(member["architecture"], "architecture")
        seed = _integer(member["seed"], "seed")
        prediction_binding = _mapping(member["prediction"], "prediction binding")
        prediction_path = Path(_string(prediction_binding["npz_path"], "prediction path")).resolve()
        if sha256_file(prediction_path) != _hash(
            prediction_binding["npz_file_sha256"], "prediction file hash"
        ):
            raise FinalResultsIntegrityError(f"sealed prediction changed for {member_id}")
        config = _mapping(member["resolved_config"], "resolved_config binding")
        prediction = load_prediction_artifact(
            prediction_path,
            protocol=protocol,
            test_access=access,
            expected_config_hash=_hash(config["config_hash"], "config hash"),
            expected_manifest_hash=manifest_hash,
        )
        if (
            prediction.integrity_sha256
            != _hash(prediction_binding["artifact_sha256"], "prediction artifact hash")
            or prediction.alignment_sha256
            != _hash(prediction_binding["alignment_sha256"], "alignment hash")
            or prediction.model_seed != seed
            or prediction.label_order != LABEL_ORDER
        ):
            raise FinalResultsIntegrityError(f"prediction binding differs for {member_id}")
        if reference_alignment is None:
            reference_alignment = prediction.alignment_sha256
        elif prediction.alignment_sha256 != reference_alignment:
            raise FinalResultsIntegrityError("member predictions are not aligned")

        decision_binding = _mapping(member["calibration_decision"], "calibration decision binding")
        decision_path = Path(
            _string(decision_binding["path"], "calibration decision path")
        ).resolve()
        if sha256_file(decision_path) != _hash(
            decision_binding["file_sha256"], "calibration decision file hash"
        ):
            raise FinalResultsIntegrityError(f"calibration decision changed for {member_id}")
        decision = load_calibration_decisions(decision_path, protocol=protocol)
        if (
            decision.integrity_sha256
            != _hash(decision_binding["artifact_sha256"], "decision artifact hash")
            or decision.model_seed != seed
            or decision.config_hash != prediction.config_hash
            or decision.label_order != LABEL_ORDER
        ):
            raise FinalResultsIntegrityError(f"calibration decision differs for {member_id}")

        _assert_subgroup_alignment(prediction, subgroup)
        subgroups: dict[str, NDArray[np.object_]] = {
            "sex": np.asarray(subgroup.sex, dtype=object),
            "age_band": np.asarray(subgroup.age_band, dtype=object),
        }
        raw_probability = stable_sigmoid(prediction.raw_logits)
        calibrated = decision.temperature_scaling.predict_proba(
            prediction.raw_logits, label_order=LABEL_ORDER
        )
        if prediction.calibrated_probabilities is not None and not np.array_equal(
            calibrated, prediction.calibrated_probabilities
        ):
            raise FinalResultsIntegrityError(
                f"stored calibrated probabilities differ from frozen decision for {member_id}"
            )
        gates = tuple(gate.to_dict() for gate in decision.coverage_gates)
        derived = derive_probability_audit(
            prediction.targets,
            raw_probability,
            calibrated,
            thresholds=decision.threshold_optimization.thresholds,
            gates=gates,
            subgroups=subgroups,
            reliability_bins=RELIABILITY_BINS,
        )
        derived["subgroup_performance"] = _subgroup_performance(prediction, calibrated, subgroups)

        report_binding = _mapping(member["final_report"], "final report binding")
        report = _read_bound_json(report_binding, context=f"{member_id} final report")
        _verify_subgroup_performance_against_report(derived, report, member_id=member_id)
        members.append(
            _MemberAnalysis(
                member_id=member_id,
                architecture=architecture,
                seed=seed,
                prediction=prediction,
                decision=decision,
                report=report,
                audit=derived,
            )
        )
    if len(members) != 6:
        raise FinalResultsIntegrityError("probability audit requires exactly six members")
    return tuple(members)


def _assert_subgroup_alignment(prediction: PredictionArtifact, subgroup: SubgroupArtifact) -> None:
    if not np.array_equal(prediction.ecg_id, np.asarray(subgroup.ecg_id, dtype=np.int64)):
        raise FinalResultsIntegrityError("subgroup ecg_id rows do not align with predictions")
    if not np.array_equal(prediction.patient_id, np.asarray(subgroup.patient_id, dtype=np.int64)):
        raise FinalResultsIntegrityError("subgroup patient_id rows do not align with predictions")


def _subgroup_performance(
    prediction: PredictionArtifact,
    probabilities: FloatArray,
    subgroups: Mapping[str, NDArray[np.object_]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for attribute in sorted(subgroups):
        values = subgroups[attribute]
        for group in sorted({str(item) for item in values.tolist()}):
            selected = values == group
            if not np.any(selected):
                continue
            metrics = compute_multilabel_metrics(
                prediction.targets[selected], probabilities[selected], ece_bins=RELIABILITY_BINS
            ).to_dict()
            result.append(
                {
                    "attribute": attribute,
                    "group": group,
                    "n_samples": int(selected.sum()),
                    "n_patients": len(
                        set(np.asarray(prediction.patient_id)[selected].astype(int).tolist())
                    ),
                    "metrics": metrics,
                }
            )
    return result


def _verify_subgroup_performance_against_report(
    audit: Mapping[str, object], report: Mapping[str, object], *, member_id: str
) -> None:
    subgroup_audit = _mapping(report.get("subgroup_audit"), "report subgroup_audit")
    report_groups = {
        (
            _string(group["attribute"], "report subgroup attribute"),
            _string(group["group_value"], "report subgroup group"),
        ): group
        for raw in _sequence(subgroup_audit.get("groups"), "report subgroup groups")
        for group in (_mapping(raw, "report subgroup group"),)
    }
    derived_groups = _sequence(audit.get("subgroup_performance"), "derived subgroups")
    if len(report_groups) != len(derived_groups):
        raise FinalResultsIntegrityError(f"subgroup group set differs for {member_id}")
    for raw in derived_groups:
        item = _mapping(raw, "derived subgroup")
        key = (
            _string(item["attribute"], "derived subgroup attribute"),
            _string(item["group"], "derived subgroup group"),
        )
        if key not in report_groups:
            raise FinalResultsIntegrityError(f"subgroup group differs for {member_id}")
        derived_macro = _mapping(
            _mapping(item["metrics"], "derived subgroup metrics")["macro"],
            "derived subgroup macro",
        )
        report_macro = _mapping(
            _mapping(report_groups[key]["metrics"], "report subgroup metrics")["macro"],
            "report subgroup macro",
        )
        for metric in ("roc_auc", "average_precision", "brier_score", "ece"):
            left = _optional_number(derived_macro.get(metric), f"derived {metric}")
            right = _optional_number(report_macro.get(metric), f"report {metric}")
            if left is None or right is None:
                if left != right:
                    raise FinalResultsIntegrityError(f"subgroup {metric} differs for {member_id}")
            elif not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
                raise FinalResultsIntegrityError(f"subgroup {metric} differs for {member_id}")


def _load_architecture_summaries(
    context: _VerifiedContext,
) -> tuple[Mapping[str, object], ...]:
    aggregates = _mapping(context.payload["aggregate_outputs"], "aggregate_outputs")
    results: list[Mapping[str, object]] = []
    for raw in _sequence(aggregates["architecture_summaries"], "architecture summaries"):
        binding = _mapping(raw, "architecture summary binding")
        payload = _read_bound_json(binding, context="architecture summary")
        if payload.get("architecture") != binding.get("architecture"):
            raise FinalResultsIntegrityError("architecture summary identity differs")
        results.append(payload)
    if len(results) != 2:
        raise FinalResultsIntegrityError("exactly two architecture summaries are required")
    return tuple(results)


def _load_paired_reports(
    context: _VerifiedContext,
) -> tuple[Mapping[str, object], ...]:
    aggregates = _mapping(context.payload["aggregate_outputs"], "aggregate_outputs")
    results: list[Mapping[str, object]] = []
    for raw in _sequence(aggregates["paired_reports"], "paired reports"):
        binding = _mapping(raw, "paired report binding")
        payload = _read_bound_json(binding, context="paired report")
        if payload.get("seed") != binding.get("seed"):
            raise FinalResultsIntegrityError("paired report seed differs")
        results.append(payload)
    if len(results) != 3:
        raise FinalResultsIntegrityError("exactly three paired reports are required")
    return tuple(results)


def _render_tables(
    analyses: Sequence[_MemberAnalysis],
    architecture: Sequence[Mapping[str, object]],
    paired: Sequence[Mapping[str, object]],
) -> dict[str, bytes]:
    return {
        "member_metrics.csv": _member_metrics_csv(analyses),
        "architecture_metrics.csv": _architecture_metrics_csv(architecture),
        "paired_deltas.csv": _paired_deltas_csv(paired),
        "reliability_seed2026.csv": _reliability_csv(analyses),
        "mean_seed_risk_coverage.csv": _risk_coverage_csv(analyses),
        "subgroup_metrics.csv": _subgroup_metrics_csv(analyses),
        "subgroup_coverage.csv": _subgroup_coverage_csv(analyses),
    }


def _member_metrics_csv(analyses: Sequence[_MemberAnalysis]) -> bytes:
    rows: list[list[object]] = []
    for item in analyses:
        metrics = _mapping(item.report["metrics"], "report metrics")
        macro = _mapping(metrics["macro"], "report macro")
        rows.append(
            [
                item.member_id,
                item.architecture,
                item.seed,
                metrics["n_samples"],
                macro["roc_auc"],
                macro["average_precision"],
                macro["brier_score"],
                macro["ece"],
            ]
        )
    return _csv_bytes(
        [
            "member_id",
            "architecture",
            "seed",
            "n_samples",
            "macro_roc_auc",
            "macro_average_precision",
            "macro_brier_score",
            "macro_ece",
        ],
        rows,
    )


def _architecture_metrics_csv(summaries: Sequence[Mapping[str, object]]) -> bytes:
    rows: list[list[object]] = []
    for summary in summaries:
        cross = _mapping(summary["cross_seed_summary"], "cross_seed_summary")
        for metric in ("roc_auc", "average_precision", "brier_score", "ece"):
            values = _mapping(cross[metric], f"architecture {metric}")
            rows.append(
                [
                    summary["architecture"],
                    metric,
                    values["mean"],
                    values["sample_standard_deviation"],
                    values["minimum"],
                    values["median"],
                    values["maximum"],
                    values["valid_seeds"],
                ]
            )
    return _csv_bytes(
        ["architecture", "metric", "mean", "sample_sd", "minimum", "median", "maximum", "n_seeds"],
        rows,
    )


def _paired_deltas_csv(reports: Sequence[Mapping[str, object]]) -> bytes:
    rows: list[list[object]] = []
    for report in reports:
        comparison = _mapping(report["comparison"], "paired comparison")
        macro = _mapping(comparison["macro"], "paired macro")
        for metric in ("roc_auc", "average_precision", "brier_score", "ece"):
            result = _mapping(macro[metric], f"paired {metric}")
            rows.append(
                [
                    report["seed"],
                    comparison["model_a"],
                    comparison["model_b"],
                    "ecg_transformer_minus_resnet1d",
                    metric,
                    result["estimate"],
                    result["lower"],
                    result["upper"],
                    result["confidence_level"],
                    result["higher_is_better"],
                    result["status"],
                ]
            )
    return _csv_bytes(
        [
            "seed",
            "model_a",
            "model_b",
            "direction",
            "metric",
            "estimate",
            "ci_lower",
            "ci_upper",
            "confidence_level",
            "higher_is_better",
            "status",
        ],
        rows,
    )


def _reliability_csv(analyses: Sequence[_MemberAnalysis]) -> bytes:
    rows: list[list[object]] = []
    for item in analyses:
        if item.seed != 2026:
            continue
        for state in ("raw", "calibrated"):
            state_audit = _mapping(item.audit[state], f"{state} audit")
            for raw_curve in _sequence(state_audit["reliability"], "reliability"):
                curve = _mapping(raw_curve, "reliability curve")
                for index, raw_bin in enumerate(_sequence(curve["bins"], "reliability bins")):
                    bin_value = _mapping(raw_bin, "reliability bin")
                    rows.append(
                        [
                            item.member_id,
                            item.architecture,
                            state,
                            curve["label"],
                            index,
                            bin_value["count"],
                            bin_value["probability_minimum"],
                            bin_value["probability_maximum"],
                            bin_value["mean_probability"],
                            bin_value["event_rate"],
                        ]
                    )
    return _csv_bytes(
        [
            "member_id",
            "architecture",
            "state",
            "label",
            "bin_index",
            "count",
            "probability_minimum",
            "probability_maximum",
            "mean_probability",
            "event_rate",
        ],
        rows,
    )


def _mean_risk_curves(
    analyses: Sequence[_MemberAnalysis],
) -> dict[str, dict[str, FloatArray]]:
    result: dict[str, dict[str, FloatArray]] = {}
    for architecture in sorted({item.architecture for item in analyses}):
        selected = [item for item in analyses if item.architecture == architecture]
        curves: dict[str, list[FloatArray]] = {
            "coverage": [],
            "hamming_risk": [],
            "exact_match_error": [],
            "brier_score": [],
            "log_loss": [],
        }
        for item in selected:
            dense = _mapping(item.audit["dense_risk_coverage"], "dense risk coverage")
            for name in curves:
                curves[name].append(np.asarray(dense[name], dtype=np.float64))
        lengths = {array.size for values in curves.values() for array in values}
        if len(lengths) != 1:
            raise FinalResultsIntegrityError("risk-coverage curve lengths differ")
        result[architecture] = {
            name: np.mean(np.stack(values, axis=0), axis=0) for name, values in curves.items()
        }
    return result


def _risk_coverage_csv(analyses: Sequence[_MemberAnalysis]) -> bytes:
    rows: list[list[object]] = []
    for architecture, curves in _mean_risk_curves(analyses).items():
        for index in range(curves["coverage"].size):
            rows.append(
                [
                    architecture,
                    index + 1,
                    curves["coverage"][index],
                    curves["hamming_risk"][index],
                    curves["exact_match_error"][index],
                    curves["brier_score"][index],
                    curves["log_loss"][index],
                ]
            )
    return _csv_bytes(
        [
            "architecture",
            "selected_prefix_count",
            "mean_coverage",
            "mean_hamming_risk",
            "mean_exact_match_error",
            "mean_brier_score",
            "mean_log_loss",
        ],
        rows,
    )


def _subgroup_metrics_csv(analyses: Sequence[_MemberAnalysis]) -> bytes:
    rows: list[list[object]] = []
    for item in analyses:
        for raw in _sequence(item.audit["subgroup_performance"], "subgroup performance"):
            subgroup = _mapping(raw, "subgroup performance")
            macro = _mapping(
                _mapping(subgroup["metrics"], "subgroup metrics")["macro"],
                "subgroup macro",
            )
            rows.append(
                [
                    item.member_id,
                    item.architecture,
                    item.seed,
                    subgroup["attribute"],
                    subgroup["group"],
                    subgroup["n_samples"],
                    subgroup["n_patients"],
                    macro["roc_auc"],
                    macro["average_precision"],
                    macro["brier_score"],
                    macro["ece"],
                ]
            )
    return _csv_bytes(
        [
            "member_id",
            "architecture",
            "seed",
            "attribute",
            "group",
            "n_samples",
            "n_patients",
            "macro_roc_auc",
            "macro_average_precision",
            "macro_brier_score",
            "macro_ece",
        ],
        rows,
    )


def _subgroup_coverage_csv(analyses: Sequence[_MemberAnalysis]) -> bytes:
    rows: list[list[object]] = []
    for item in analyses:
        for raw_gate in _sequence(item.audit["frozen_gates"], "frozen gates"):
            gate = _mapping(raw_gate, "frozen gate")
            coverage = _mapping(gate["subgroup_coverage"], "subgroup coverage")
            for attribute in sorted(coverage):
                groups = _mapping(coverage[attribute], "subgroup coverage groups")
                for group in sorted(groups):
                    values = _mapping(groups[group], "subgroup coverage value")
                    rows.append(
                        [
                            item.member_id,
                            item.architecture,
                            item.seed,
                            gate["target_coverage"],
                            gate["achieved_coverage"],
                            attribute,
                            group,
                            values["count"],
                            values["selected_count"],
                            values["coverage"],
                        ]
                    )
    return _csv_bytes(
        [
            "member_id",
            "architecture",
            "seed",
            "target_global_coverage",
            "achieved_global_coverage",
            "attribute",
            "group",
            "count",
            "selected_count",
            "subgroup_coverage",
        ],
        rows,
    )


def _render_figures(
    analyses: Sequence[_MemberAnalysis],
    architecture: Sequence[Mapping[str, object]],
    paired: Sequence[Mapping[str, object]],
    *,
    output_root: Path,
) -> dict[str, bytes]:
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".matplotlib-", dir=output_root) as raw_cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = raw_cache
        try:
            _configure_matplotlib()
            return {
                "architecture_member_metrics.png": _architecture_member_figure(analyses),
                "paired_deltas_ci.png": _paired_figure(paired),
                "raw_vs_calibrated_reliability_seed2026.png": _reliability_figure(analyses),
                "mean_seed_risk_coverage.png": _risk_coverage_figure(analyses),
                "subgroup_performance_coverage.png": _subgroup_figure(analyses, architecture),
            }
        finally:
            if previous is None:
                del os.environ["MPLCONFIGDIR"]
            else:
                os.environ["MPLCONFIGDIR"] = previous


def _configure_matplotlib() -> None:
    global _PYPLOT
    if _PYPLOT is not None:
        return
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot

    _PYPLOT = pyplot


def _pyplot() -> Any:
    if _PYPLOT is None:
        raise FinalResultsIntegrityError("Matplotlib was not configured inside output root")
    return _PYPLOT


def _style_axes(axis: Any) -> None:
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _figure_bytes(figure: Any) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": "ecg_trust"},
    )
    _pyplot().close(figure)
    return buffer.getvalue()


def _architecture_member_figure(analyses: Sequence[_MemberAnalysis]) -> bytes:
    figure, axes = _pyplot().subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    colors = {"resnet1d": "#1f77b4", "ecg_transformer": "#d62728"}
    for axis, metric, title in zip(
        axes,
        ("roc_auc", "average_precision"),
        ("Macro AUROC", "Macro average precision"),
        strict=True,
    ):
        labels: list[str] = []
        values: list[float] = []
        bar_colors: list[str] = []
        for item in analyses:
            macro = _mapping(_mapping(item.report["metrics"], "metrics")["macro"], "macro")
            labels.append(f"{item.architecture}\n{item.seed}")
            values.append(_number(macro[metric], metric))
            bar_colors.append(colors[item.architecture])
        axis.bar(np.arange(len(values)), values, color=bar_colors, width=0.72)
        axis.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right", fontsize=8)
        axis.set_ylim(max(0.0, min(values) - 0.08), min(1.0, max(values) + 0.03))
        axis.set_title(title)
        _style_axes(axis)
    figure.suptitle("Sealed fold-10 member metrics")
    return _figure_bytes(figure)


def _paired_figure(reports: Sequence[Mapping[str, object]]) -> bytes:
    metrics = ("roc_auc", "average_precision", "brier_score", "ece")
    figure, axes = _pyplot().subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, metric in zip(axes.flat, metrics, strict=True):
        estimates: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []
        seeds: list[str] = []
        for report in reports:
            result = _mapping(
                _mapping(_mapping(report["comparison"], "comparison")["macro"], "macro")[metric],
                metric,
            )
            estimates.append(_number(result["estimate"], "estimate"))
            lowers.append(_number(result["lower"], "lower"))
            uppers.append(_number(result["upper"], "upper"))
            seeds.append(str(report["seed"]))
        y = np.arange(len(estimates))
        error = np.vstack(
            [np.asarray(estimates) - np.asarray(lowers), np.asarray(uppers) - estimates]
        )
        axis.errorbar(estimates, y, xerr=error, fmt="o", color="#4c4c4c", capsize=4)
        axis.axvline(0.0, color="#b22222", linestyle="--", linewidth=1)
        axis.set_yticks(y, seeds)
        axis.set_title(metric.replace("_", " ").title())
        axis.set_xlabel("Transformer - ResNet1D")
        _style_axes(axis)
    figure.suptitle("Paired patient-bootstrap differences (95% CI)")
    return _figure_bytes(figure)


def _reliability_figure(analyses: Sequence[_MemberAnalysis]) -> bytes:
    selected = [item for item in analyses if item.seed == 2026]
    if len(selected) != 2:
        raise FinalResultsIntegrityError("reliability figure requires two seed-2026 members")
    figure, axes = _pyplot().subplots(2, len(LABEL_ORDER), figsize=(15, 6), constrained_layout=True)
    for row, item in enumerate(selected):
        curves_by_state: dict[str, dict[str, Mapping[str, object]]] = {}
        for state in ("raw", "calibrated"):
            state_audit = _mapping(item.audit[state], f"{state} audit")
            curves_by_state[state] = {
                _string(curve["label"], "reliability label"): curve
                for raw in _sequence(state_audit["reliability"], "reliability")
                for curve in (_mapping(raw, "reliability curve"),)
            }
        for column, label in enumerate(LABEL_ORDER):
            axis = axes[row, column]
            axis.plot([0, 1], [0, 1], color="#888888", linestyle=":", linewidth=1)
            for state, color, marker in (
                ("raw", "#7f7f7f", "o"),
                ("calibrated", "#1f77b4", "s"),
            ):
                bins = _sequence(curves_by_state[state][label]["bins"], "reliability bins")
                x = [
                    _number(_mapping(raw, "bin")["mean_probability"], "mean probability")
                    for raw in bins
                ]
                y = [_number(_mapping(raw, "bin")["event_rate"], "event rate") for raw in bins]
                axis.plot(x, y, color=color, marker=marker, markersize=3, linewidth=1, label=state)
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.set_title(label)
            if column == 0:
                axis.set_ylabel(f"{item.architecture}\nObserved rate")
            if row == 1:
                axis.set_xlabel("Predicted probability")
            if row == 0 and column == len(LABEL_ORDER) - 1:
                axis.legend(fontsize=7, loc="lower right")
            _style_axes(axis)
    figure.suptitle("Raw vs frozen-calibrated reliability, matched seed 2026")
    return _figure_bytes(figure)


def _risk_coverage_figure(analyses: Sequence[_MemberAnalysis]) -> bytes:
    figure, axis = _pyplot().subplots(figsize=(7.5, 5), constrained_layout=True)
    colors = {"resnet1d": "#1f77b4", "ecg_transformer": "#d62728"}
    for architecture, curves in _mean_risk_curves(analyses).items():
        axis.plot(
            curves["coverage"],
            curves["hamming_risk"],
            color=colors[architecture],
            linewidth=2,
            label=f"{architecture} (3-seed mean)",
        )
    axis.set_xlabel("Coverage")
    axis.set_ylabel("Hamming risk")
    axis.set_xlim(0, 1)
    axis.set_title("Entropy-ranked post-evaluation risk-coverage")
    axis.legend()
    _style_axes(axis)
    return _figure_bytes(figure)


def _subgroup_figure(
    analyses: Sequence[_MemberAnalysis], architecture: Sequence[Mapping[str, object]]
) -> bytes:
    del architecture  # kept in the signature to make the figure input contract explicit
    performance: dict[tuple[str, str], list[float]] = {}
    coverage: dict[tuple[str, str], list[float]] = {}
    for item in analyses:
        for raw in _sequence(item.audit["subgroup_performance"], "subgroup performance"):
            subgroup = _mapping(raw, "subgroup")
            key = (_string(subgroup["attribute"], "attribute"), _string(subgroup["group"], "group"))
            macro = _mapping(_mapping(subgroup["metrics"], "metrics")["macro"], "macro")
            value = _optional_number(macro["roc_auc"], "subgroup roc_auc")
            if value is not None:
                performance.setdefault(key, []).append(value)
        gates = _sequence(item.audit["frozen_gates"], "frozen gates")
        gate = next(
            (
                _mapping(raw, "gate")
                for raw in gates
                if math.isclose(
                    _number(_mapping(raw, "gate")["target_coverage"], "target coverage"),
                    0.8,
                    abs_tol=1e-12,
                    rel_tol=0.0,
                )
            ),
            None,
        )
        if gate is None:
            raise FinalResultsIntegrityError("frozen coverage-0.8 gate is missing")
        groups = _mapping(gate["subgroup_coverage"], "subgroup coverage")
        for attribute in groups:
            for group, raw_value in _mapping(groups[attribute], "groups").items():
                key = (attribute, group)
                coverage.setdefault(key, []).append(
                    _number(_mapping(raw_value, "coverage")["coverage"], "coverage")
                )
    keys = sorted(set(performance).intersection(coverage))
    figure, axis = _pyplot().subplots(figsize=(8, 5.5), constrained_layout=True)
    colors = {"sex": "#2ca02c", "age_band": "#9467bd"}
    for key in keys:
        x = float(np.mean(coverage[key]))
        y = float(np.mean(performance[key]))
        axis.scatter(x, y, color=colors.get(key[0], "#333333"), s=55)
        axis.annotate(
            f"{key[0]}={key[1]}", (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8
        )
    axis.axvline(0.8, color="#777777", linestyle=":", linewidth=1)
    axis.set_xlabel("Mean subgroup coverage at frozen 0.8 global gate")
    axis.set_ylabel("Mean macro AUROC across six members")
    axis.set_title("Subgroup performance and abstention coverage")
    _style_axes(axis)
    return _figure_bytes(figure)


def _load_completed_branch_manifest(
    path: Path, *, context: _VerifiedContext, branch: str
) -> Mapping[str, object]:
    root = _load_self_hashed_json(path)
    artifact_type = _string(root.get("artifact_type"), f"{branch} artifact_type")
    expected_type = {
        "robustness": "ecg_trust.robustness_audit_manifest",
        "explanation": "ecg_trust.explanation_audit_manifest",
    }[branch]
    if artifact_type != expected_type:
        raise FinalResultsIntegrityError(f"{branch} manifest artifact type differs")
    if not _contains_spec_binding(root, context.spec.artifact_sha256):
        raise FinalResultsIntegrityError(f"{branch} manifest does not bind the audit spec")
    verified_bindings = 0
    derived_bindings = 0
    branch_root = path.parent.resolve()
    for binding in _nested_file_bindings(root):
        bound_path = Path(_string(binding["path"], f"{branch} bound path"))
        if not bound_path.is_absolute():
            bound_path = context.root / bound_path
        bound_path = bound_path.resolve()
        expected_hash = _hash(binding["file_sha256"], f"{branch} bound file hash")
        if sha256_file(bound_path) != expected_hash:
            raise FinalResultsIntegrityError(f"{branch} bound file changed: {bound_path}")
        verified_bindings += 1
        if bound_path != path and bound_path.is_relative_to(branch_root):
            derived_bindings += 1
    if verified_bindings < 1:
        raise FinalResultsIntegrityError(f"{branch} manifest contains no file bindings")
    if derived_bindings < 1:
        raise FinalResultsIntegrityError(f"{branch} manifest contains no branch-local derived file")
    return root


def _verify_canonical_branch_manifest(
    path: Path,
    *,
    context: _VerifiedContext,
    branch: str,
    generic: Mapping[str, object],
) -> Mapping[str, object]:
    """Invoke each runner's strict semantic verifier after generic hash checks."""

    verified: Mapping[str, object]
    with _matplotlib_config_scope(context.root):
        if branch == "robustness":
            from ecg_trust.robustness_audit import load_robustness_manifest

            verified = load_robustness_manifest(
                path, spec=context.spec, verify_sources=True
            ).payload
        elif branch == "explanation":
            from ecg_trust.explanation_audit import load_explanation_manifest

            verified = load_explanation_manifest(
                path,
                expected_spec_sha256=context.spec.artifact_sha256,
                spec=context.spec,
                verify_sources=True,
            ).payload
        else:  # pragma: no cover - internal invariant
            raise FinalResultsIntegrityError(f"unsupported audit branch: {branch}")
    if dict(verified) != dict(generic):
        raise FinalResultsIntegrityError(
            f"{branch} canonical verifier and generic manifest view differ"
        )
    return verified


def _summarize_robustness(manifest: Mapping[str, object]) -> dict[str, object]:
    """Extract conservative worst-case summaries from verified case sidecars."""

    observations: dict[str, tuple[float, str, str] | None] = {
        "minimum_macro_auroc_delta": None,
        "maximum_macro_brier_delta": None,
        "maximum_aurc_hamming_delta": None,
        "maximum_absolute_gate_coverage_delta": None,
    }
    nonclean = 0
    for raw in _sequence(manifest.get("artifacts"), "robustness artifacts"):
        entry = _mapping(raw, "robustness artifact")
        case_id = _string(entry.get("case_id"), "robustness case_id")
        if case_id == "clean":
            continue
        nonclean += 1
        member_id = _string(entry.get("member_id"), "robustness member_id")
        sidecar = _mapping(entry.get("sidecar"), "robustness sidecar binding")
        payload = _read_json(
            Path(_string(sidecar.get("path"), "robustness sidecar path")).resolve(),
            "robustness sidecar",
        )
        metadata = _mapping(payload.get("metadata"), "robustness metadata")
        delta = _mapping(metadata.get("delta_summary"), "robustness delta_summary")
        calibrated = _mapping(delta.get("calibrated"), "calibrated delta")
        macro = _mapping(calibrated.get("macro"), "calibrated macro delta")
        dense = _mapping(delta.get("dense_risk_coverage"), "dense delta")
        _update_extreme(
            observations,
            "minimum_macro_auroc_delta",
            _optional_number(macro.get("roc_auc"), "macro AUROC delta"),
            member_id,
            case_id,
            minimum=True,
        )
        _update_extreme(
            observations,
            "maximum_macro_brier_delta",
            _optional_number(macro.get("brier_score"), "macro Brier delta"),
            member_id,
            case_id,
            minimum=False,
        )
        _update_extreme(
            observations,
            "maximum_aurc_hamming_delta",
            _optional_number(dense.get("aurc_hamming"), "AURC hamming delta"),
            member_id,
            case_id,
            minimum=False,
        )
        for raw_gate in _sequence(delta.get("frozen_gates"), "frozen gate deltas"):
            gate = _mapping(raw_gate, "frozen gate delta")
            coverage = _optional_number(gate.get("coverage"), "gate coverage delta")
            if coverage is not None:
                _update_extreme(
                    observations,
                    "maximum_absolute_gate_coverage_delta",
                    abs(coverage),
                    member_id,
                    case_id,
                    minimum=False,
                )
    expected_nonclean = _integer(manifest.get("member_count"), "robustness member_count") * (
        _integer(manifest.get("case_count"), "robustness case_count") - 1
    )
    if nonclean != expected_nonclean or any(value is None for value in observations.values()):
        raise FinalResultsIntegrityError("robustness summary grid is incomplete")
    return {
        "members": manifest["member_count"],
        "cases_per_member": manifest["case_count"],
        "member_cases": manifest["member_case_count"],
        "nonclean_member_cases": nonclean,
        "all_clean_logits_exact": True,
        **{
            key: {
                "value": cast(tuple[float, str, str], value)[0],
                "member_id": cast(tuple[float, str, str], value)[1],
                "case_id": cast(tuple[float, str, str], value)[2],
            }
            for key, value in observations.items()
        },
    }


def _update_extreme(
    observations: dict[str, tuple[float, str, str] | None],
    key: str,
    value: float | None,
    member_id: str,
    case_id: str,
    *,
    minimum: bool,
) -> None:
    if value is None:
        return
    current = observations[key]
    if current is None or (value < current[0] if minimum else value > current[0]):
        observations[key] = (value, member_id, case_id)


def _summarize_explanations(manifest: Mapping[str, object]) -> dict[str, object]:
    """Aggregate verified explanation controls without claiming localization truth."""

    members = _sequence(manifest.get("members"), "explanation members")
    cohort = _mapping(manifest.get("cohort"), "explanation cohort")
    cohort_records = _integer(cohort.get("records"), "cohort records")
    if cohort_records < 1:
        raise FinalResultsIntegrityError("explanation cohort must not be empty")

    settings = _mapping(manifest.get("settings"), "explanation settings")
    target_score = _mapping(settings.get("target_score"), "explanation target score")
    execution = _mapping(settings.get("execution"), "explanation execution")
    expected_target_score = {
        "positive_cell": "+1_times_target_label_logit",
        "negative_cell": "-1_times_target_label_logit",
        "probability": "sigmoid(signed_correct_status_logit_over_frozen_temperature)",
        "attribution_orientation": "multiply_target_label_map_by_cell_sign",
    }
    if any(target_score.get(key) != value for key, value in expected_target_score.items()):
        raise FinalResultsIntegrityError("explanation target-score semantics differ")
    if (
        execution.get("numeric_precision") != "float32"
        or execution.get("sealed_clean_equivalence_precision") != "bf16_as_frozen_in_final_batch"
        or execution.get("fp32_vs_sealed_cohort_logit_drift_required") is not True
    ):
        raise FinalResultsIntegrityError("explanation precision bridge differs")

    runtime = _mapping(manifest.get("attribution_runtime"), "attribution runtime")
    method_summaries: list[Mapping[str, object]] = []
    maximum_drift: tuple[float, str, str] | None = None
    cross_pairs = 0
    cross_cosine_pairs_with_values = 0
    cross_spearman_pairs_with_values = 0
    cross_cosine_valid = 0
    cross_spearman_valid = 0
    cross_cosine_weighted_sum = 0.0
    cross_spearman_weighted_sum = 0.0
    member_ids: list[str] = []
    for raw_member in members:
        member = _mapping(raw_member, "explanation member")
        member_id = _string(member.get("member_id"), "explanation member_id")
        member_ids.append(member_id)
        runtime_block = _mapping(runtime.get(member_id), "member attribution runtime")
        if (
            runtime_block.get("model_dtype") != "float32"
            or runtime_block.get("sealed_clean_equivalence_precision") != "bf16"
        ):
            raise FinalResultsIntegrityError("explanation member precision bridge differs")
        for raw_method in _sequence(member.get("methods"), "explanation methods"):
            method = _mapping(raw_method, "explanation method")
            method_name = _string(method.get("method"), "explanation method name")
            method_summary = _mapping(method.get("summary"), "explanation method summary")
            method_summaries.append(method_summary)
            drift = _mapping(
                method_summary.get("fp32_vs_sealed_bf16_logit_drift"),
                "FP32 versus sealed BF16 logit drift",
            )
            for name in ("mean_absolute", "maximum_absolute", "root_mean_square"):
                if _number(drift.get(name), f"{name} logit drift") < 0.0:
                    raise FinalResultsIntegrityError("absolute logit drift cannot be negative")
            observed_maximum = _number(drift.get("maximum_absolute"), "maximum logit drift")
            if maximum_drift is None or observed_maximum > maximum_drift[0]:
                maximum_drift = (observed_maximum, member_id, method_name)
        cross = _mapping(member.get("cross_method"), "cross-method binding")
        summary = _mapping(cross.get("summary"), "cross-method summary")
        pairs = _sequence(summary.get("pairs"), "cross-method pairs")
        cosine = _sequence(summary.get("mean_cosine"), "mean cosine")
        spearman = _sequence(summary.get("mean_spearman"), "mean spearman")
        cosine_counts = _sequence(summary.get("valid_cosine_examples"), "valid cosine examples")
        spearman_counts = _sequence(
            summary.get("valid_spearman_examples"), "valid spearman examples"
        )
        lengths = {len(pairs), len(cosine), len(spearman), len(cosine_counts), len(spearman_counts)}
        if len(lengths) != 1:
            raise FinalResultsIntegrityError("cross-method summary arrays differ in length")
        for index, raw_pair in enumerate(pairs):
            _string(raw_pair, "cross-method pair")
            cross_pairs += 1
            for metric, means, counts in (
                ("cosine", cosine, cosine_counts),
                ("spearman", spearman, spearman_counts),
            ):
                valid_count = _integer(counts[index], f"valid {metric} examples")
                if valid_count < 0 or valid_count > cohort_records:
                    raise FinalResultsIntegrityError(
                        f"valid {metric} examples exceed the explanation cohort"
                    )
                mean = _optional_number(means[index], f"cross-method {metric}")
                if (valid_count == 0) != (mean is None):
                    raise FinalResultsIntegrityError(
                        f"cross-method {metric} validity count and mean disagree"
                    )
                if metric == "cosine":
                    cross_cosine_valid += valid_count
                    if mean is not None:
                        cross_cosine_pairs_with_values += 1
                        cross_cosine_weighted_sum += mean * valid_count
                else:
                    cross_spearman_valid += valid_count
                    if mean is not None:
                        cross_spearman_pairs_with_values += 1
                        cross_spearman_weighted_sum += mean * valid_count

    if (
        len(members) != 6
        or len(set(member_ids)) != len(member_ids)
        or set(runtime) != set(member_ids)
        or len(method_summaries) != 15
        or cross_pairs != 12
        or maximum_drift is None
    ):
        raise FinalResultsIntegrityError("explanation summary grid is incomplete")

    def values(name: str) -> list[float]:
        return [_number(summary.get(name), name) for summary in method_summaries]

    repeat = values("mean_repeat_cosine")
    stability = values("mean_stability_cosine_40db")
    randomization = values("mean_parameter_randomization_cosine")
    deletion = values("mean_guided_vs_random_deletion_advantage")
    exact = all(summary.get("deterministic_repeat_exact") is True for summary in method_summaries)
    cross_method_examples = cross_pairs * cohort_records
    return {
        "members": len(members),
        "method_artifacts": len(method_summaries),
        "ecgs_per_method": cohort_records,
        "method_ecg_evaluations": len(method_summaries) * cohort_records,
        "deterministic_repeat_exact_all": exact,
        "minimum_mean_repeat_cosine": min(repeat),
        "mean_stability_cosine_40db": float(np.mean(stability)),
        "mean_parameter_randomization_cosine": float(np.mean(randomization)),
        "mean_guided_vs_random_deletion_advantage": float(np.mean(deletion)),
        "maximum_fp32_vs_sealed_bf16_logit_drift": {
            "value": maximum_drift[0],
            "member_id": maximum_drift[1],
            "method": maximum_drift[2],
        },
        "target_score_positive_cell": target_score["positive_cell"],
        "target_score_negative_cell": target_score["negative_cell"],
        "faithfulness_curve_probability": target_score["probability"],
        "attribution_orientation": target_score["attribution_orientation"],
        "attribution_precision": "float32",
        "sealed_confirmation_precision": "bf16",
        "precision_bridge": "fp32_attribution_vs_sealed_bf16_confirmation",
        "cross_method_pairs": cross_pairs,
        "cross_method_examples": cross_method_examples,
        "cross_method_cosine_pairs_with_valid_values": cross_cosine_pairs_with_values,
        "cross_method_spearman_pairs_with_valid_values": cross_spearman_pairs_with_values,
        "cross_method_cosine_valid_examples": cross_cosine_valid,
        "cross_method_cosine_invalid_examples": cross_method_examples - cross_cosine_valid,
        "cross_method_spearman_valid_examples": cross_spearman_valid,
        "cross_method_spearman_invalid_examples": cross_method_examples - cross_spearman_valid,
        "mean_cross_method_cosine": (
            cross_cosine_weighted_sum / cross_cosine_valid if cross_cosine_valid else None
        ),
        "mean_cross_method_spearman": (
            cross_spearman_weighted_sum / cross_spearman_valid if cross_spearman_valid else None
        ),
        "localization_ground_truth_available": False,
    }


def _contains_spec_binding(value: object, expected: str, *, key: str = "") -> bool:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                continue
            lowered = raw_key.casefold()
            if child == expected and "spec" in lowered and "sha256" in lowered:
                return True
            if _contains_spec_binding(child, expected, key=raw_key):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_spec_binding(child, expected, key=key) for child in value)
    return False


def _nested_file_bindings(value: object) -> list[Mapping[str, object]]:
    result: list[Mapping[str, object]] = []
    if isinstance(value, Mapping):
        if "path" in value and "file_sha256" in value:
            result.append(cast(Mapping[str, object], value))
        for child in value.values():
            result.extend(_nested_file_bindings(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            result.extend(_nested_file_bindings(child))
    return result


def _verify_demo_contract(context: _VerifiedContext, binding_path: Path) -> None:
    expected = _artifact_path(context, "demo_binding")
    if binding_path.resolve() != expected:
        raise FinalResultsIntegrityError("demo binding is not at its canonical path")
    binding = _load_self_hashed_json(
        expected, expected_type="ecg_trust.demo_materialization_binding"
    )
    demo = _mapping(
        _mapping(context.payload["audit_protocols"], "audit_protocols")["demo"],
        "demo protocol",
    )
    selection = _mapping(binding["selection"], "demo selection")
    for key in ("member_id", "architecture", "seed", "target_coverage"):
        if selection.get(key) != demo.get(key):
            raise FinalResultsIntegrityError(f"demo selection differs for {key}")
    if (
        selection.get("fold10_predictions_read") is not False
        or selection.get("fold10_performance_used") is not False
        or demo.get("retuning_allowed") is not False
    ):
        raise FinalResultsIntegrityError("demo selection/retuning boundary differs")
    sources = _mapping(binding["sources"], "demo sources")
    expected_sources = {
        "checkpoint": _mapping(demo["checkpoint"], "demo checkpoint"),
        "resolved_config": _mapping(demo["resolved_config"], "demo resolved_config"),
        "fold9_decision": _mapping(demo["calibration_decision"], "demo calibration_decision"),
    }
    for name, frozen in expected_sources.items():
        observed = _mapping(sources.get(name), f"demo source {name}")
        for key, expected_value in frozen.items():
            if observed.get(key) != expected_value:
                raise FinalResultsIntegrityError(f"demo {name} source differs for {key}")
    policy_binding = _mapping(binding["policy"], "demo policy")
    policy_path = Path(_string(policy_binding["path"], "demo policy path")).resolve()
    if policy_path != _artifact_path(context, "demo_policy"):
        raise FinalResultsIntegrityError("demo policy is not at its canonical path")
    policy = _read_json(policy_path, "demo policy")
    gate = _mapping(policy["gate"], "demo policy gate")
    frozen_gate = _mapping(demo["gate"], "frozen demo gate")
    if gate.get("method") != demo.get("entropy_method") or gate.get(
        "uncertainty_threshold"
    ) != frozen_gate.get("maximum_entropy"):
        raise FinalResultsIntegrityError("demo policy gate differs from the frozen spec")


def _load_demo_binding(context: _VerifiedContext, *, protocol: ExperimentProtocol) -> Any:
    from ecg_trust.demo_materialization import load_and_verify_demo_binding

    demo = load_and_verify_demo_binding(_artifact_path(context, "demo_binding"), protocol=protocol)
    _verify_demo_contract(context, demo.binding_path)
    return demo


def _load_operational_run_log(context: _VerifiedContext) -> tuple[Path, str]:
    runtime = _mapping(context.payload["analysis_runtime"], "analysis_runtime")
    project_root = Path(_string(runtime["project_root"], "project_root")).resolve()
    path = (project_root / "reports" / "FINAL_EVALUATION_RUN_LOG.md").resolve()
    text = _read_text(path)
    required = (
        "DEV-001",
        "batch_interrupted",
        "representation-only",
        "two-minute command limit",
        "c75a12b",
        "did not produce or alter any sealed evaluation artifact",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise FinalResultsIntegrityError(
            "operational run log lacks required disclosure markers: " + ", ".join(missing)
        )
    return path, sha256_file(path)


def _build_final_results_markdown(
    *,
    context: _VerifiedContext,
    probability: Mapping[str, object],
    architecture: Sequence[Mapping[str, object]],
    paired: Sequence[Mapping[str, object]],
    robustness_sha256: str,
    explanations_sha256: str,
    demo_sha256: str,
    robustness_summary: Mapping[str, object],
    explanations_summary: Mapping[str, object],
    run_log_path: Path,
    run_log_sha256: str,
    deviations_path: Path,
) -> str:
    architecture_rows: list[str] = []
    for summary in architecture:
        cross = _mapping(summary["cross_seed_summary"], "cross_seed_summary")
        architecture_rows.append(
            "| {architecture} | {auroc:.4f} | {ap:.4f} | {brier:.4f} | {ece:.4f} |".format(
                architecture=summary["architecture"],
                auroc=_number(_mapping(cross["roc_auc"], "roc_auc")["mean"], "mean AUROC"),
                ap=_number(
                    _mapping(cross["average_precision"], "average_precision")["mean"], "mean AP"
                ),
                brier=_number(_mapping(cross["brier_score"], "brier_score")["mean"], "mean Brier"),
                ece=_number(_mapping(cross["ece"], "ece")["mean"], "mean ECE"),
            )
        )
    paired_rows: list[str] = []
    for report in paired:
        macro = _mapping(
            _mapping(report["comparison"], "paired comparison")["macro"], "paired macro"
        )
        auroc = _mapping(macro["roc_auc"], "paired AUROC")
        paired_rows.append(
            "| {seed} | {estimate:.4f} | [{lower:.4f}, {upper:.4f}] |".format(
                seed=report["seed"],
                estimate=_number(auroc["estimate"], "paired AUROC estimate"),
                lower=_number(auroc["lower"], "paired AUROC lower"),
                upper=_number(auroc["upper"], "paired AUROC upper"),
            )
        )
    probability_hash = _hash(probability["artifact_sha256"], "probability hash")
    spec_path = cast(Path, context.spec.path)
    robustness_auroc = _mapping(
        robustness_summary["minimum_macro_auroc_delta"], "robustness AUROC summary"
    )
    robustness_brier = _mapping(
        robustness_summary["maximum_macro_brier_delta"], "robustness Brier summary"
    )
    robustness_aurc = _mapping(
        robustness_summary["maximum_aurc_hamming_delta"], "robustness AURC summary"
    )
    robustness_gate = _mapping(
        robustness_summary["maximum_absolute_gate_coverage_delta"],
        "robustness gate summary",
    )
    repeat_cosine = _number(explanations_summary["minimum_mean_repeat_cosine"], "repeat cosine")
    stability_cosine = _number(
        explanations_summary["mean_stability_cosine_40db"], "stability cosine"
    )
    randomization_cosine = _number(
        explanations_summary["mean_parameter_randomization_cosine"],
        "randomization cosine",
    )
    deletion_advantage = _number(
        explanations_summary["mean_guided_vs_random_deletion_advantage"],
        "deletion advantage",
    )
    cross_method_cosine = _optional_number(
        explanations_summary["mean_cross_method_cosine"], "cross-method cosine"
    )
    cross_method_spearman = _optional_number(
        explanations_summary["mean_cross_method_spearman"], "cross-method spearman"
    )
    cross_method_cosine_text = (
        f"{cross_method_cosine:.4f}" if cross_method_cosine is not None else "not estimable"
    )
    cross_method_spearman_text = (
        f"{cross_method_spearman:.4f}" if cross_method_spearman is not None else "not estimable"
    )
    cross_method_pairs = _integer(explanations_summary["cross_method_pairs"], "method pairs")
    cross_method_examples = _integer(
        explanations_summary["cross_method_examples"], "cross-method examples"
    )
    cosine_pairs_with_values = _integer(
        explanations_summary["cross_method_cosine_pairs_with_valid_values"],
        "cosine pairs with valid values",
    )
    spearman_pairs_with_values = _integer(
        explanations_summary["cross_method_spearman_pairs_with_valid_values"],
        "Spearman pairs with valid values",
    )
    cosine_valid = _integer(
        explanations_summary["cross_method_cosine_valid_examples"],
        "valid cosine examples",
    )
    cosine_invalid = _integer(
        explanations_summary["cross_method_cosine_invalid_examples"],
        "invalid cosine examples",
    )
    spearman_valid = _integer(
        explanations_summary["cross_method_spearman_valid_examples"],
        "valid Spearman examples",
    )
    spearman_invalid = _integer(
        explanations_summary["cross_method_spearman_invalid_examples"],
        "invalid Spearman examples",
    )
    maximum_drift = _mapping(
        explanations_summary["maximum_fp32_vs_sealed_bf16_logit_drift"],
        "maximum FP32 versus sealed BF16 logit drift",
    )
    maximum_drift_value = _number(maximum_drift["value"], "maximum cohort logit drift")
    maximum_drift_member = _string(maximum_drift["member_id"], "maximum drift member")
    maximum_drift_method = _string(maximum_drift["method"], "maximum drift method")
    if (
        explanations_summary.get("target_score_positive_cell") != "+1_times_target_label_logit"
        or explanations_summary.get("target_score_negative_cell") != "-1_times_target_label_logit"
        or explanations_summary.get("faithfulness_curve_probability")
        != "sigmoid(signed_correct_status_logit_over_frozen_temperature)"
        or explanations_summary.get("attribution_orientation")
        != "multiply_target_label_map_by_cell_sign"
        or explanations_summary.get("attribution_precision") != "float32"
        or explanations_summary.get("sealed_confirmation_precision") != "bf16"
        or explanations_summary.get("precision_bridge")
        != "fp32_attribution_vs_sealed_bf16_confirmation"
        or cosine_valid + cosine_invalid != cross_method_examples
        or spearman_valid + spearman_invalid != cross_method_examples
    ):
        raise FinalResultsIntegrityError("explanation reporting semantics differ")
    repeat_exact = explanations_summary.get("deterministic_repeat_exact_all") is True
    lines = [
        "# Final PTB-XL results",
        "",
        "## Interpretation boundary",
        "",
        (
            "The architecture/member metrics and preregistered within-seed paired "
            "patient-bootstrap comparisons below are the **confirmatory sealed fold-10 "
            "results**. Reliability curves, dense risk-coverage analysis, subgroup "
            "coverage views, robustness corruptions, explanations, and the local demo "
            "are **post-evaluation descriptive analyses** frozen only after the "
            "confirmatory batch was complete. They must not be presented as additional "
            "confirmatory tests or as model-selection evidence."
        ),
        "",
        (
            "This is research-only benchmark evidence. It does not establish clinical "
            "validity, external validity, diagnostic safety, medical-device performance, "
            "or fitness for patient care, and no clinical or diagnostic claim is made."
        ),
        "",
        "## Confirmatory sealed fold-10 results",
        "",
        (
            "Each value below is the descriptive mean across the three preregistered "
            "seeds; individual-member values remain in "
            "`publication/tables/member_metrics.csv`."
        ),
        "",
        "| Architecture | Macro AUROC | Macro AP | Brier | ECE |",
        "|---|---:|---:|---:|---:|",
        *architecture_rows,
        "",
        (
            "The paired direction is ECG Transformer minus ResNet1D on aligned patients. "
            "Negative AUROC differences favor ResNet1D."
        ),
        "",
        "| Seed | Macro AUROC difference | Paired 95% CI |",
        "|---:|---:|---:|",
        *paired_rows,
        "",
        "### ECE interval caveat",
        "",
        (
            "ECE patient-bootstrap intervals require special caution: patient resampling "
            "duplicates patient rows while the fixed 15-bin ECE calculation repartitions "
            "those rows, which can bias the bootstrap ECE distribution. In particular, "
            "an ECE point estimate may fall outside its percentile interval. ECE "
            "intervals are therefore descriptive sensitivity summaries, not conventional "
            "inferential guarantees."
        ),
        "",
        "## Post-evaluation descriptive analyses",
        "",
        (
            f"The probability audit (`{probability_hash}`) applies only the frozen fold-9 "
            "temperatures, thresholds, and entropy gates. It contains "
            "raw-versus-calibrated reliability, full entropy-ranked risk-coverage curves, "
            "error-detection summaries, and sex/age-band performance and coverage. The "
            f"robustness manifest (`{robustness_sha256}`), explanation manifest "
            f"(`{explanations_sha256}`), and demo binding (`{demo_sha256}`) were "
            "reverified before this report was written."
        ),
        "",
        "### Controlled-corruption robustness audit",
        "",
        (
            "The verified grid contains "
            f"{robustness_summary['member_cases']} member-cases "
            f"({robustness_summary['members']} members × "
            f"{robustness_summary['cases_per_member']} cases), with exact clean-logit "
            "equivalence for all six members before corruptions. The extrema below are "
            "corrupted-minus-clean descriptive deltas across the 240 non-clean "
            "member-cases; they are sensitivity bounds for this controlled matrix, not "
            "deployment-shift estimates."
        ),
        "",
        "| Robustness statistic | Worst observed delta | Member / case |",
        "|---|---:|---|",
        (
            "| Minimum macro AUROC delta | "
            f"{_number(robustness_auroc['value'], 'robustness AUROC value'):.4f} | "
            f"{robustness_auroc['member_id']} / {robustness_auroc['case_id']} |"
        ),
        (
            "| Maximum macro Brier delta | "
            f"{_number(robustness_brier['value'], 'robustness Brier value'):.4f} | "
            f"{robustness_brier['member_id']} / {robustness_brier['case_id']} |"
        ),
        (
            "| Maximum hamming-AURC delta | "
            f"{_number(robustness_aurc['value'], 'robustness AURC value'):.4f} | "
            f"{robustness_aurc['member_id']} / {robustness_aurc['case_id']} |"
        ),
        (
            "| Maximum absolute frozen-gate coverage delta | "
            f"{_number(robustness_gate['value'], 'robustness gate value'):.4f} | "
            f"{robustness_gate['member_id']} / {robustness_gate['case_id']} |"
        ),
        "",
        "### Explanation-control audit",
        "",
        (
            f"The fixed 60-ECG cohort produced {explanations_summary['method_artifacts']} "
            "method artifacts across six members "
            f"({explanations_summary['method_ecg_evaluations']} method–ECG evaluations). "
            f"Deterministic repeats were exact for every method: {str(repeat_exact).lower()}. "
            "These are model-behavior controls; "
            "the cohort has no clinical localization ground truth."
        ),
        "",
        (
            "Every attribution and faithfulness curve uses a signed correct-status score: "
            "+1 × the target-label logit for a positive cell and -1 × that logit for a "
            "negative cell. Curve probabilities are sigmoid(signed correct-status logit / "
            "frozen temperature), so deletion and insertion summarize confidence in the "
            "cell's correct status, not positive-class confidence alone."
        ),
        "",
        (
            "The confirmatory path and its exact clean-logit replay use sealed BF16 precision "
            "as frozen for the final batch, while gradient attribution reloads the checkpoints "
            "deterministically in FP32. This FP32-attribution-to-sealed-BF16 precision bridge "
            "does not imply bitwise logit identity. The maximum absolute raw-logit drift "
            f"on the attribution cohort was {maximum_drift_value:.6g} for "
            f"{maximum_drift_member} / {maximum_drift_method}."
        ),
        "",
        (
            "Cross-method means below are weighted over valid ECG-level correlations. "
            "Undefined comparisons, such as constant-map correlations, are excluded rather "
            "than coerced to zero and are disclosed by their valid and invalid counts."
        ),
        "",
        "| Explanation control | Verified aggregate |",
        "|---|---:|",
        (f"| Minimum method-level mean repeat cosine | {repeat_cosine:.4f} |"),
        f"| Mean stability cosine at 40 dB | {stability_cosine:.4f} |",
        (f"| Mean parameter-randomization cosine | {randomization_cosine:.4f} |"),
        (f"| Mean guided-vs-random deletion advantage | {deletion_advantage:.4f} |"),
        (
            "| Mean cross-method cosine "
            f"(valid n={cosine_valid}/{cross_method_examples}; invalid n={cosine_invalid}; "
            f"{cosine_pairs_with_values}/{cross_method_pairs} pairs estimable) | "
            f"{cross_method_cosine_text} |"
        ),
        (
            "| Mean cross-method Spearman "
            f"(valid n={spearman_valid}/{cross_method_examples}; invalid n={spearman_invalid}; "
            f"{spearman_pairs_with_values}/{cross_method_pairs} pairs estimable) | "
            f"{cross_method_spearman_text} |"
        ),
        (
            "| Maximum absolute FP32–sealed-BF16 cohort raw-logit drift "
            f"({maximum_drift_member} / {maximum_drift_method}) | {maximum_drift_value:.6g} |"
        ),
        "",
        (
            "These analyses characterize this fixed internal benchmark cohort. Controlled "
            "corruptions are not evidence of real-world distribution-shift robustness; "
            "saliency is not causal explanation; subgroup estimates do not establish "
            "fairness; and abstention does not create a clinically safe decision rule."
        ),
        "",
        "## Required protocol-deviation disclosure",
        "",
        (
            f"DEV-001 in `{deviations_path}` records accidental pre-evaluation exposure "
            "of raw fold-10 label-bearing rows during a metadata search. No exposed "
            "row-level value was used for modeling, calibration, thresholds, coverage "
            "gates, subgroup choices, or reporting choices, and no fold-10 model metric "
            "or prediction was observed at that time. Nevertheless, strict operator-level "
            "fold-10 outcome-label blindness was breached, so this work must not claim a "
            "completely blind test. The limitation weakens the strongest "
            "confirmatory-blinding claim and remains part of every scientific "
            "interpretation."
        ),
        "",
        "## Operational run-log disclosure",
        "",
        (
            f"The post-completion operational record is `{run_log_path}` "
            f"(`{run_log_sha256}`). The sealed batch logged six `batch_interrupted` "
            "events caused by a redundant comparison of equivalent prefixed and bare "
            "SHA-256 representations immediately after immutable prediction publication. "
            "Exact resumes fully revalidated and adopted each existing pair; they did not "
            "overwrite predictions or repeat a scientific query. One supervised shell "
            "invocation also reached an external two-minute limit before any new ledger "
            "event or artifact commit. Post-evaluation commit `c75a12b` corrected only "
            "that representation check and did not produce or alter a sealed evaluation "
            "artifact. These events mean the scientific opening was ledgered and "
            "recoverable, but the process was not an uninterrupted single invocation."
        ),
        "",
        "## Reproducibility anchors",
        "",
        f"- Post-evaluation audit specification: `{spec_path}` (`{context.spec.artifact_sha256}`)",
        f"- Probability audit: `{probability_hash}`",
        f"- Operational run log: `{run_log_sha256}`",
        (
            "- All generated and prerequisite post-evaluation files are hashed in "
            "`derived_artifacts.manifest.json`."
        ),
        "",
    ]
    return "\n".join(lines)


def _all_root_file_bindings(root: Path, *, excluding: set[Path]) -> list[dict[str, object]]:
    excluded = {path.resolve() for path in excluding}
    files = sorted(
        (path.resolve() for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    result: list[dict[str, object]] = []
    for path in files:
        _ensure_within(path, root, allow_root=False)
        if path in excluded:
            continue
        result.append(
            {
                "path": path.relative_to(root).as_posix(),
                "file_sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return result


def _read_bound_json(binding: Mapping[str, object], *, context: str) -> Mapping[str, object]:
    path = Path(_string(binding.get("path"), f"{context} path")).resolve()
    expected_file = _hash(binding.get("file_sha256"), f"{context} file hash")
    if sha256_file(path) != expected_file:
        raise FinalResultsIntegrityError(f"{context} file hash differs")
    root = _read_json(path, context)
    expected_artifact = binding.get("artifact_sha256")
    if expected_artifact is not None:
        stored = root.get("artifact_sha256", root.get("report_sha256"))
        if _hash(stored, f"{context} artifact hash") != _hash(
            expected_artifact, f"{context} bound artifact hash"
        ):
            raise FinalResultsIntegrityError(f"{context} artifact binding differs")
        body = dict(root)
        if "artifact_sha256" in body:
            del body["artifact_sha256"]
        elif "report_sha256" in body:
            del body["report_sha256"]
        if canonical_sha256(body) != _hash(stored, f"{context} self hash"):
            raise FinalResultsIntegrityError(f"{context} self-hash mismatch")
    return root


def _load_self_hashed_json(path: Path, *, expected_type: str | None = None) -> Mapping[str, object]:
    root = _read_json(path, str(path))
    stored = _hash(root.get("artifact_sha256"), "artifact_sha256")
    body = dict(root)
    del body["artifact_sha256"]
    if canonical_sha256(body) != stored:
        raise FinalResultsIntegrityError(f"self-hash mismatch: {path}")
    if expected_type is not None and root.get("artifact_type") != expected_type:
        raise FinalResultsIntegrityError(f"artifact type differs: {path}")
    return root


def _read_json(path: Path, context: str) -> Mapping[str, object]:
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FinalResultsIntegrityError(f"could not decode {context}: {error}") from error
    return _mapping(decoded, context)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise FinalResultsIntegrityError(f"could not read {path}: {error}") from error


def _csv_bytes(header: Sequence[str], rows: Sequence[Sequence[object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow([_csv_cell(value) for value in row])
    return buffer.getvalue().encode("utf-8")


def _csv_cell(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FinalResultsError("CSV values must be finite")
        return format(value, ".17g")
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return value


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    try:
        return (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError) as error:
        raise FinalResultsError("publication payload must contain finite JSON") from error


def _sha256_bytes(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _probability_file_bindings(
    payload: Mapping[str, object],
) -> tuple[Mapping[str, str], ...]:
    result: list[Mapping[str, str]] = []
    for raw in _sequence(payload.get("publication_files"), "publication_files"):
        binding = _mapping(raw, "publication file binding")
        result.append(
            {
                "path": _string(binding.get("path"), "publication path"),
                "file_sha256": _hash(binding.get("file_sha256"), "publication file hash"),
                "media_type": _string(binding.get("media_type"), "publication media_type"),
            }
        )
    return tuple(result)


def _preflight_exact_content(values: Mapping[Path, bytes], *, root: Path) -> None:
    for raw_path, expected in values.items():
        path = raw_path.resolve()
        _ensure_within(path, root, allow_root=False)
        if not path.exists():
            continue
        if not path.is_file():
            raise FinalResultsIntegrityError(f"publication path is not a file: {path}")
        try:
            observed = path.read_bytes()
        except OSError as error:
            raise FinalResultsIntegrityError(
                f"could not read existing publication file {path}: {error}"
            ) from error
        if observed != expected:
            raise FinalResultsIntegrityError(f"immutable crash-recovery file differs: {path}")


def _ensure_exact_bytes(path: Path, content: bytes, *, root: Path) -> None:
    destination = path.resolve()
    _ensure_within(destination, root, allow_root=False)
    if destination.exists():
        _preflight_exact_content({destination: content}, root=root)
        return
    try:
        _write_new_bytes(destination, content, root=root)
    except FileExistsError:
        # Another exact publisher may have won the exclusive-create race.
        _preflight_exact_content({destination: content}, root=root)


def _write_new_bytes(path: Path, content: bytes, *, root: Path) -> None:
    destination = path.resolve()
    _ensure_within(destination, root, allow_root=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise FileExistsError(
            f"publication artifacts are immutable; refusing to overwrite {destination}"
        ) from None


def _preflight_new_paths(paths: Sequence[Path], *, root: Path) -> None:
    normalized: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        _ensure_within(resolved, root, allow_root=False)
        normalized.append(resolved)
    if len(set(normalized)) != len(normalized):
        raise FinalResultsError("publication destination paths must be unique")
    existing = [path for path in normalized if path.exists()]
    if existing:
        raise FileExistsError(
            "publication artifacts are immutable; refusing to overwrite "
            + ", ".join(str(path) for path in existing)
        )


def _ensure_within(path: Path, root: Path, *, allow_root: bool) -> None:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as error:
        raise FinalResultsIntegrityError(
            f"publication path escapes frozen output root: {resolved}"
        ) from error
    if not allow_root and relative == Path("."):
        raise FinalResultsIntegrityError("a publication file cannot be the output root")


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise FinalResultsIntegrityError(f"{context} must be a string-keyed object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise FinalResultsIntegrityError(f"{context} must be an array")
    return cast(Sequence[object], value)


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FinalResultsIntegrityError(f"{context} must be a non-empty string")
    return value


def _hash(value: object, context: str) -> str:
    text = _string(value, context)
    digest = text.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise FinalResultsIntegrityError(f"{context} must be a prefixed SHA-256")
    return "sha256:" + digest


def _integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FinalResultsIntegrityError(f"{context} must be an integer")
    return value


def _number(value: object, context: str) -> float:
    parsed = _optional_number(value, context)
    if parsed is None:
        raise FinalResultsIntegrityError(f"{context} must be numeric")
    return parsed


def _optional_number(value: object, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalResultsIntegrityError(f"{context} must be numeric or null")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise FinalResultsIntegrityError(f"{context} must be finite")
    return parsed


__all__ = [
    "DERIVED_MANIFEST_TYPE",
    "FIGURE_FILENAMES",
    "FinalResultsError",
    "FinalResultsIntegrityError",
    "FinalizationResult",
    "PROBABILITY_AUDIT_TYPE",
    "ProbabilityRenderResult",
    "TABLE_FILENAMES",
    "canonical_sha256",
    "finalize_results",
    "load_and_verify_probability_audit",
    "render_probability_results",
    "sha256_file",
]
