"""Deterministic Markdown rendering for aggregate model passports."""

from __future__ import annotations

from ecg_trust.passport.models import (
    AggregateMetric,
    EvidenceStatus,
    EvidenceSummaryBase,
    ModelPassport,
)


def render_model_passport_markdown(passport: ModelPassport) -> str:
    """Render a stable, escaped research passport without row-level material."""

    lines = [
        f"# Model Passport: {_escape(passport.release_id)}",
        "",
        f"> {_escape(passport.safety_notice)}",
        "",
        "## Release identity",
        "",
        f"- Passport ID: `{_escape(passport.passport_id)}`",
        f"- Passport SHA-256: `{passport.passport_sha256}`",
        f"- Release SHA-256: `{passport.release_sha256}`",
        f"- TrustBundle SHA-256: `{passport.bundle_sha256}`",
        f"- Protocol SHA-256: `{passport.protocol_sha256}`",
        f"- Generated: `{passport.generated_at.isoformat()}`",
        "",
        "## Supported input",
        "",
        "- Resting clinical 12-lead ECG metadata contract",
        f"- Lead order: `{', '.join(passport.supported_input.lead_order)}`",
        f"- Sampling: `{passport.supported_input.sampling_frequency_hz:g} Hz`",
        (
            f"- Duration: `{passport.supported_input.duration_seconds:g} s` "
            f"(`{passport.supported_input.samples_per_lead}` samples per lead)"
        ),
        (
            f"- Representation: `{passport.supported_input.dtype}`, "
            f"`{passport.supported_input.physical_units}`"
        ),
        "",
        "## Dataset and site evidence",
        "",
        "| Cohort | Dataset | Site | Role | Samples | Patients | Manifest |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for cohort in passport.cohorts:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape(cohort.cohort_id),
                    _escape(f"{cohort.dataset_name} {cohort.dataset_version}"),
                    _escape(cohort.site_name),
                    cohort.role.value,
                    str(cohort.sample_count),
                    str(cohort.patient_count),
                    f"`{cohort.manifest_sha256}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Discrimination and calibration",
            "",
            "| Scope | Discrimination | Calibration |",
            "|---|---|---|",
        ]
    )
    for item in passport.label_performance:
        lines.append(
            f"| {_escape(item.label)} | {_metric_text(item.discrimination)} | "
            f"{_metric_text(item.calibration)} |"
        )
    lines.append(
        f"| Macro | {_metric_text(passport.macro_performance.discrimination)} | "
        f"{_metric_text(passport.macro_performance.calibration)} |"
    )

    lines.extend(
        [
            "",
            "## Subgroup evidence",
            "",
            "| Group | Cohort | Samples | Patients | Minimum evidence | Metrics |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for subgroup in passport.subgroup_evidence:
        metric_text = ", ".join(_metric_text(metric) for metric in subgroup.metrics)
        if not metric_text:
            metric_text = "Suppressed: insufficient evidence"
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape(f"{subgroup.attribute}: {subgroup.group_name}"),
                    _escape(subgroup.cohort_id),
                    str(subgroup.sample_count),
                    str(subgroup.patient_count),
                    subgroup.status.value,
                    metric_text,
                ]
            )
            + " |"
        )

    for transport in passport.external_transport:
        lines.extend(_summary_lines("External transport", transport))
        lines.append("- Target adaptation: `NONE`; frozen source model: `true`")
    lines.extend(_summary_lines("Open-world / OOD evidence", passport.ood_evidence))
    lines.append(
        "- Score direction: `HIGHER_IS_MORE_OUT_OF_DISTRIBUTION`; "
        "threshold scope: `SOURCE_CALIBRATION_ONLY`"
    )
    lines.extend(_summary_lines("Signal-quality evidence", passport.quality_evidence))
    lines.extend(_summary_lines("Selective-prediction evidence", passport.selective_evidence))
    lines.extend(_summary_lines("Conformal evidence", passport.conformal_evidence))
    lines.append(f"- Coverage scope: {_escape(passport.conformal_evidence.coverage_scope_text)}")

    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {_escape(limitation)}" for limitation in passport.limitations)
    lines.extend(
        [
            "",
            "## Safety boundary",
            "",
            "- Research only: `true`",
            "- Clinically validated: `false`",
            "- Clinical use permitted: `false`",
            f"- {_escape(passport.safety_notice)}",
            "",
        ]
    )
    return "\n".join(lines)


def _summary_lines(title: str, summary: EvidenceSummaryBase) -> list[str]:
    artifact = "not available" if summary.artifact_sha256 is None else summary.artifact_sha256
    lines = [
        "",
        f"## {_escape(title)}",
        "",
        f"- Status: `{summary.status.value}`",
        f"- Method: `{_escape(summary.method)}`",
        f"- Artifact SHA-256: `{artifact}`",
        f"- Cohorts: `{', '.join(_escape(value) for value in summary.cohort_ids)}`",
        f"- Aggregate counts: `{summary.sample_count}` samples, `{summary.patient_count}` patients",
        f"- Summary: {_escape(summary.summary)}",
    ]
    if summary.status is EvidenceStatus.AVAILABLE:
        lines.extend(f"- {_metric_text(metric)}" for metric in summary.metrics)
    lines.extend(f"- Limitation: {_escape(value)}" for value in summary.limitations)
    return lines


def _metric_text(metric: AggregateMetric) -> str:
    name = _escape(metric.display_name)
    if metric.status is not EvidenceStatus.AVAILABLE or metric.value is None:
        return f"{name}: {metric.status.value}"
    estimate = f"{metric.value:.6g} {_escape(metric.unit)}"
    interval = metric.confidence_interval
    if interval is not None:
        estimate += (
            f" ({interval.confidence_level:.3g} CI "
            f"{interval.lower:.6g}–{interval.upper:.6g}, {_escape(interval.method)})"
        )
    return f"{name}: {estimate} (n={metric.sample_count})"


def _escape(value: str) -> str:
    escaped = value.replace("\\", "\\\\")
    for character in ("`", "*", "_", "{", "}", "[", "]", "<", ">", "#", "|"):
        escaped = escaped.replace(character, "\\" + character)
    return escaped


__all__ = ["render_model_passport_markdown"]
