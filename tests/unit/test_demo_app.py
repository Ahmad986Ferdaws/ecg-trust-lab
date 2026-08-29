from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
import wfdb  # type: ignore[import-untyped]
from fastapi.testclient import TestClient

from ecg_trust.constants import LEADS, PTBXL_VERSION, SUPERCLASSES
from ecg_trust.demo_app import DemoAppConfig, DemoExample, DemoWebError, create_app
from ecg_trust.demo_backend import (
    AttributionMethod,
    AttributionPayload,
    DecisionProvenance,
    DemoPrediction,
    FrozenDecisionPolicy,
    load_wfdb_physical_signal,
)


class FakeBackend:
    def __init__(self, *, abstain: bool = False) -> None:
        self.policy = FrozenDecisionPolicy(
            temperature=1.25,
            classification_thresholds=(0.4, 0.5, 0.6, 0.5, 0.7),
            uncertainty_threshold=0.8,
            provenance=DecisionProvenance(
                dataset_version=PTBXL_VERSION,
                protocol_hash="sha256:" + "a" * 64,
                manifest_hash="b" * 64,
                checkpoint_config_hash="sha256:" + "c" * 64,
                checkpoint_sha256="d" * 64,
                resolved_config_sha256="e" * 64,
                normalization_sha256="f" * 64,
                calibration_folds=(9,),
            ),
        )
        self.artifact_provenance = {"checkpoint_sha256": "d" * 64, "seed": 2026}
        self.seen_paths: list[Path] = []
        self.abstain = abstain

    def predict_record(
        self,
        record_path: str | Path,
        *,
        attribution_method: AttributionMethod | None = None,
        attribution_target: str | int | None = None,
        integrated_gradients_steps: int = 32,
    ) -> DemoPrediction:
        path = Path(record_path)
        self.seen_paths.append(path)
        load_wfdb_physical_signal(path)
        logits = torch.tensor([2.0, -1.0, 0.5, -2.0, 1.0], dtype=torch.float64)
        raw = logits.sigmoid()
        calibrated = (logits / self.policy.temperature).sigmoid()
        attribution = None
        if attribution_method is not None:
            target = "MI" if attribution_target is None else str(attribution_target)
            attribution = AttributionPayload(
                method=attribution_method,
                target_label=target,
                values=torch.linspace(-1.0, 1.0, 1000).unsqueeze(0),
                coordinate_space="test",
            )
        return DemoPrediction(
            label_order=SUPERCLASSES,
            raw_logits=logits,
            raw_probabilities=raw,
            calibrated_probabilities=calibrated,
            threshold_predictions=(True, False, False, False, False),
            uncertainty=0.95 if self.abstain else 0.25,
            gate_threshold=self.policy.uncertainty_threshold,
            decision="abstain" if self.abstain else "accept",
            decision_reason=(
                "uncertainty_exceeds_frozen_fold9_gate"
                if self.abstain
                else "uncertainty_within_frozen_fold9_gate"
            ),
            source=str(path),
            attribution=attribution,
            artifact_provenance=self.artifact_provenance,
        )


def _wfdb_pair(root: Path, name: str = "record") -> tuple[Path, Path]:
    time = np.linspace(0.0, 10.0, 1000, endpoint=False)
    physical = np.stack(
        [0.1 * np.sin(2 * np.pi * (index + 1) * time) for index in range(12)], axis=1
    )
    wfdb.wrsamp(
        name,
        fs=100,
        units=["mV"] * 12,
        sig_name=list(LEADS),
        p_signal=physical,
        fmt=["16"] * 12,
        write_dir=str(root),
    )
    return root / f"{name}.hea", root / f"{name}.dat"


def test_unconfigured_app_stays_available_but_reports_not_ready() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")
        metadata = client.get("/metadata")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready", "model_loaded": False}
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert metadata.status_code == 503


def test_index_metadata_and_local_plotly_are_self_contained() -> None:
    backend = FakeBackend()
    with TestClient(create_app(backend=backend)) as client:
        page = client.get("/")
        metadata = client.get("/metadata")
        javascript = client.get("/assets/plotly.min.js")

    assert page.status_code == 200
    assert "Research use only" in page.text
    assert "ECG Trust Lab" in page.text
    assert "historical entropy-gated research baseline" in page.text
    assert "not ECG Trust Sentinel" in page.text
    assert '<link rel="icon" href="data:,">' in page.text
    assert "type: 'scatter'" in page.text
    assert "scattergl" not in page.text.casefold()
    assert "webgl" not in page.text.casefold()
    assert metadata.json()["label_order"] == list(SUPERCLASSES)
    assert metadata.json()["input"]["samples_per_lead"] == 1000
    assert metadata.json()["decision_policy"]["calibration_folds"] == [9]
    assert javascript.status_code == 200
    assert "plotly" in javascript.text[:1000].casefold()


def test_index_csp_nonce_authorizes_inline_controller_per_request() -> None:
    with TestClient(create_app(backend=FakeBackend())) as client:
        pages = (client.get("/"), client.get("/"))

    nonces: list[str] = []
    for page in pages:
        script_source = next(
            directive.strip()
            for directive in page.headers["content-security-policy"].split(";")
            if directive.strip().startswith("script-src ")
        )
        match = re.search(r"'nonce-([A-Za-z0-9_-]+)'", script_source)
        assert match is not None
        nonce = match.group(1)
        nonces.append(nonce)
        assert "'unsafe-inline'" not in script_source
        assert page.text.count(f'<script nonce="{nonce}">') == 1

    assert nonces[0] != nonces[1]


def test_upload_prediction_returns_waveform_and_hides_temporary_path(tmp_path: Path) -> None:
    header, signal = _wfdb_pair(tmp_path)
    backend = FakeBackend()
    with (
        TestClient(create_app(backend=backend)) as client,
        header.open("rb") as header_handle,
        signal.open("rb") as signal_handle,
    ):
        response = client.post(
            "/predict",
            files={
                "header": (header.name, header_handle, "text/plain"),
                "signal": (signal.name, signal_handle, "application/octet-stream"),
            },
            data={"attribution_method": "grad_cam", "attribution_target": "MI"},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["source"] == "upload:record"
    assert payload["waveform"]["lead_order"] == list(LEADS)
    assert len(payload["waveform"]["values"]) == 12
    assert len(payload["waveform"]["values"][0]) == 1000
    assert payload["attribution"]["target_label"] == "MI"
    assert list(payload["calibrated_probabilities"]) == list(SUPERCLASSES)
    assert payload["predictions_exposed"] is True
    assert payload["system_scope"] == "legacy_entropy_baseline_not_trust_sentinel"
    assert backend.seen_paths and not backend.seen_paths[0].parent.exists()
    assert str(backend.seen_paths[0]) not in response.text


def test_abstaining_legacy_demo_withholds_every_label_level_result(tmp_path: Path) -> None:
    header, signal = _wfdb_pair(tmp_path)
    backend = FakeBackend(abstain=True)
    with (
        TestClient(create_app(backend=backend)) as client,
        header.open("rb") as header_handle,
        signal.open("rb") as signal_handle,
    ):
        response = client.post(
            "/predict",
            files={
                "header": (header.name, header_handle, "text/plain"),
                "signal": (signal.name, signal_handle, "application/octet-stream"),
            },
            data={"attribution_method": "grad_cam", "attribution_target": "MI"},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["decision"]["status"] == "ABSTAIN"
    assert payload["predictions_exposed"] is False
    assert payload["system_scope"] == "legacy_entropy_baseline_not_trust_sentinel"
    assert "waveform" in payload
    assert not {
        "raw_logits",
        "raw_probabilities",
        "calibrated_probabilities",
        "threshold_predictions",
        "positive_labels",
        "uncertainty",
        "gate_threshold",
        "attribution",
    }.intersection(payload)


def test_upload_rejects_mismatched_and_path_bearing_names(tmp_path: Path) -> None:
    header, signal = _wfdb_pair(tmp_path)
    backend = FakeBackend()
    with TestClient(create_app(backend=backend)) as client:
        response = client.post(
            "/predict",
            files={
                "header": ("record.hea", header.read_bytes(), "text/plain"),
                "signal": ("other.dat", signal.read_bytes(), "application/octet-stream"),
            },
        )
        traversal = client.post(
            "/predict",
            files={
                "header": ("../record.hea", header.read_bytes(), "text/plain"),
                "signal": ("record.dat", signal.read_bytes(), "application/octet-stream"),
            },
        )

    assert response.status_code == 400
    assert "share one stem" in response.json()["detail"]
    assert traversal.status_code == 400
    assert "paths" in traversal.json()["detail"]
    assert backend.seen_paths == []


def test_upload_header_cannot_reference_another_data_file(tmp_path: Path) -> None:
    header, signal = _wfdb_pair(tmp_path)
    malicious = header.read_text(encoding="ascii").replace("record.dat", "../secret.dat")
    backend = FakeBackend()
    with TestClient(create_app(backend=backend)) as client:
        response = client.post(
            "/predict",
            files={
                "header": ("record.hea", malicious.encode("ascii"), "text/plain"),
                "signal": ("record.dat", signal.read_bytes(), "application/octet-stream"),
            },
        )

    assert response.status_code == 400
    assert "only its matched" in response.json()["detail"]
    assert backend.seen_paths == []


def test_example_registry_exposes_only_id_and_label(tmp_path: Path) -> None:
    header, _ = _wfdb_pair(tmp_path, "known")
    backend = FakeBackend()
    config = DemoAppConfig(
        examples=(DemoExample("example-1", "Representative local ECG", header.with_suffix("")),)
    )
    with TestClient(create_app(backend=backend, config=config)) as client:
        listing = client.get("/examples")
        prediction = client.post("/predict/example/example-1")

    assert listing.json() == {
        "examples": [{"id": "example-1", "label": "Representative local ECG"}]
    }
    assert str(tmp_path) not in listing.text
    assert prediction.status_code == 200
    assert prediction.json()["source"] == "example:example-1"
    assert str(tmp_path) not in prediction.text


def test_example_manifest_is_strict_and_resolves_relative_records(tmp_path: Path) -> None:
    _wfdb_pair(tmp_path, "sample")
    manifest = tmp_path / "examples.json"
    manifest.write_text(
        json.dumps({"examples": [{"id": "sample", "label": "Sample", "record_path": "sample"}]}),
        encoding="utf-8",
    )

    config = DemoAppConfig.with_example_manifest(manifest)

    assert config.examples[0].record_path == (tmp_path / "sample").resolve()
    manifest.write_text(json.dumps({"examples": [], "extra": True}), encoding="utf-8")
    try:
        DemoAppConfig.with_example_manifest(manifest)
    except DemoWebError as error:
        assert "only an examples array" in str(error)
    else:
        raise AssertionError("invalid example manifest was accepted")
