from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from ecg_trust.data.multisite import (
    DatasetRole,
    ExternalWFDBRecord,
    LockboxMutationError,
    MappingAction,
    MultisiteIntegrityError,
    MultisiteManifest,
    MultisiteManifestError,
    OntologyMappingDecision,
    OntologyReviewError,
    ReviewedSharedOntology,
    ReviewStatus,
    RoleIsolationError,
    assert_lockbox_unchanged,
    build_multisite_manifest,
    build_reviewed_shared_ontology,
    parse_native_snomed_codes,
)

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
NORMAL = "426783006"
AF = "164889003"
RBBB = "59118001"


def _record(
    *,
    dataset: str = "CPSC",
    version: str = "2018",
    site: str = "CPSC-2018",
    record_ref: str = "records/A0001",
    patient_key: str = "patient-1",
    role: DatasetRole = DatasetRole.DEVELOPMENT,
    codes: tuple[str, ...] = (NORMAL, AF),
    duration: float = 10.0,
) -> ExternalWFDBRecord:
    return ExternalWFDBRecord.create(
        source_dataset=dataset,
        source_version=version,
        source_site=site,
        record_ref=record_ref,
        patient_key=patient_key,
        sampling_rate_hz=500.0,
        duration_seconds=duration,
        ordered_leads=LEADS,
        native_snomed_codes=codes,
        role=role,
    )


def _reviewed_decision(
    *,
    dataset: str,
    version: str,
    code: str,
    label: str | None,
    action: MappingAction = MappingAction.MAP,
) -> OntologyMappingDecision:
    return OntologyMappingDecision.create(
        source_dataset=dataset,
        source_version=version,
        native_snomed_code=code,
        action=action,
        shared_label=label,
        status=ReviewStatus.REVIEWED,
        reviewed_by="ontology-panel-v1",
        reviewed_at_utc="2026-08-24T02:00:00Z",
        review_reference="decision-log:SNOMED-v1",
        rationale="Reviewed against the declared shared-label protocol.",
    )


def _three_source_manifest() -> MultisiteManifest:
    return build_multisite_manifest(
        [
            _record(),
            _record(
                dataset="Georgia",
                version="1.0",
                site="G12EC",
                record_ref="g/0002",
                patient_key="g-2",
                role=DatasetRole.PREVIOUSLY_OBSERVED,
                codes=(RBBB,),
            ),
            _record(
                dataset="Chapman-Shaoxing",
                version="1.0",
                site="Ningbo",
                record_ref="ningbo/0003",
                patient_key="n-3",
                role=DatasetRole.UNTOUCHED_LOCKBOX,
                codes=(AF,),
            ),
        ]
    )


def test_native_snomed_parser_preserves_exact_source_order() -> None:
    assert parse_native_snomed_codes(f"#Dx: {NORMAL},{AF},{RBBB}") == (
        NORMAL,
        AF,
        RBBB,
    )
    assert parse_native_snomed_codes(f"{RBBB}, {NORMAL}") == (RBBB, NORMAL)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "#Dx:",
        f"{NORMAL},",
        f"{NORMAL},{NORMAL}",
        "0426783006",
        "+426783006",
        "426783006.0",
        "ABC783006",
        "٤٢٦٧٨٣٠٠٦",
        "12345",
    ],
)
def test_native_snomed_parser_rejects_malformed_or_lossy_values(value: object) -> None:
    with pytest.raises(MultisiteManifestError):
        parse_native_snomed_codes(value)


def test_record_is_immutable_native_ordered_and_json_round_trippable() -> None:
    record = _record(codes=(AF, NORMAL))

    assert record.native_snomed_codes == (AF, NORMAL)
    assert record.ordered_leads == LEADS
    assert record.role is DatasetRole.DEVELOPMENT
    payload = json.loads(json.dumps(record.to_dict(), allow_nan=False))
    assert ExternalWFDBRecord.from_dict(payload) == record
    with pytest.raises(FrozenInstanceError):
        record.duration_seconds = 20.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"record_ref": "../A0001"}, "traversal"),
        ({"record_ref": "records/A0001.hea"}, "without suffix"),
        ({"record_ref": "C:\\records\\A0001"}, "forbidden"),
        ({"patient_key": "patient one"}, "forbidden"),
        ({"sampling_rate_hz": 0.0}, "positive"),
        ({"duration_seconds": float("nan")}, "finite"),
        ({"ordered_leads": ("I", "I")}, "duplicates"),
        ({"ordered_leads": ("I", "unknown")}, "non-canonical"),
        ({"native_snomed_codes": f"{NORMAL},{AF}"}, "parsed into a sequence"),
        ({"role": "test"}, "invalid dataset role"),
    ],
)
def test_record_rejects_malformed_wfdb_metadata(overrides: dict[str, object], match: str) -> None:
    arguments: dict[str, object] = {
        "source_dataset": "CPSC",
        "source_version": "2018",
        "source_site": "CPSC-2018",
        "record_ref": "records/A0001",
        "patient_key": "patient-1",
        "sampling_rate_hz": 500.0,
        "duration_seconds": 10.0,
        "ordered_leads": LEADS,
        "native_snomed_codes": (NORMAL,),
        "role": DatasetRole.DEVELOPMENT,
    }
    arguments.update(overrides)
    with pytest.raises(MultisiteManifestError, match=match):
        ExternalWFDBRecord.create(**arguments)  # type: ignore[arg-type]


def test_manifest_hash_and_order_are_deterministic_and_json_round_trip() -> None:
    records = list(_three_source_manifest().records)
    first = build_multisite_manifest(records)
    second = build_multisite_manifest(reversed(records))

    assert first.manifest_sha256 == second.manifest_sha256
    assert first.records == second.records
    payload = json.loads(json.dumps(first.to_dict(), allow_nan=False))
    assert MultisiteManifest.from_dict(payload) == first


def test_manifest_rejects_duplicate_record_identity() -> None:
    first = _record()
    duplicate = _record(patient_key="different-patient")
    with pytest.raises(MultisiteManifestError, match="duplicate external record"):
        build_multisite_manifest([first, duplicate])


def test_patient_keys_cannot_cross_roles_within_a_dataset_release() -> None:
    development = _record(site="site-a", role=DatasetRole.DEVELOPMENT)
    calibration = _record(
        site="site-b",
        record_ref="records/B0001",
        role=DatasetRole.CALIBRATION,
    )
    with pytest.raises(RoleIsolationError, match="patient.*roles"):
        build_multisite_manifest([development, calibration])


def test_an_entire_source_site_cannot_cross_roles() -> None:
    development = _record(role=DatasetRole.DEVELOPMENT)
    calibration = _record(
        record_ref="records/A0002",
        patient_key="patient-2",
        role=DatasetRole.CALIBRATION,
    )
    with pytest.raises(RoleIsolationError, match="source.*roles"):
        build_multisite_manifest([development, calibration])


@pytest.mark.parametrize(
    ("dataset", "site"),
    [
        ("PhysioNet Challenge 2021", "PTB and PTB-XL"),
        ("PhysioNet/Computing in Cardiology Challenge 2020", "PTB-XL"),
        ("Challenge-2021", "Physikalisch-Technische Bundesanstalt"),
        ("PhysioNet Challenge 2021 PTB-XL", "source-3"),
    ],
)
def test_physionet_challenge_ptb_overlap_is_explicitly_excluded(dataset: str, site: str) -> None:
    overlap = _record(
        dataset=dataset,
        version="1.0.3",
        site=site,
        role=DatasetRole.PREVIOUSLY_OBSERVED,
    )
    with pytest.raises(MultisiteManifestError, match="PTB/PTB-XL source is excluded"):
        build_multisite_manifest([overlap])


def test_standalone_ptbxl_development_source_is_not_mistaken_for_overlap() -> None:
    record = _record(
        dataset="PTB-XL",
        version="1.0.3",
        site="PTB",
        role=DatasetRole.DEVELOPMENT,
    )
    assert build_multisite_manifest([record]).records == (record,)


def test_untouched_lockbox_set_and_metadata_are_immutable() -> None:
    frozen = _three_source_manifest()
    assert_lockbox_unchanged(frozen, build_multisite_manifest(reversed(frozen.records)))

    lockbox = next(
        record for record in frozen.records if record.role is DatasetRole.UNTOUCHED_LOCKBOX
    )
    non_lockbox = tuple(
        record for record in frozen.records if record.role is not DatasetRole.UNTOUCHED_LOCKBOX
    )
    changed = _record(
        dataset=lockbox.source_dataset,
        version=lockbox.source_version,
        site=lockbox.source_site,
        record_ref=lockbox.record_ref,
        patient_key=lockbox.patient_key,
        role=DatasetRole.UNTOUCHED_LOCKBOX,
        codes=lockbox.native_snomed_codes,
        duration=11.0,
    )
    with pytest.raises(LockboxMutationError, match="changed"):
        assert_lockbox_unchanged(
            frozen,
            build_multisite_manifest([*non_lockbox, changed]),
        )

    relabeled = _record(
        dataset=lockbox.source_dataset,
        version=lockbox.source_version,
        site=lockbox.source_site,
        record_ref=lockbox.record_ref,
        patient_key=lockbox.patient_key,
        role=DatasetRole.PREVIOUSLY_OBSERVED,
        codes=lockbox.native_snomed_codes,
    )
    with pytest.raises(LockboxMutationError, match="removed"):
        assert_lockbox_unchanged(
            frozen,
            build_multisite_manifest([*non_lockbox, relabeled]),
        )


def test_non_lockbox_metadata_can_change_without_mutating_frozen_lockbox() -> None:
    frozen = _three_source_manifest()
    records: list[ExternalWFDBRecord] = []
    for record in frozen.records:
        if record.role is DatasetRole.DEVELOPMENT:
            records.append(
                _record(
                    dataset=record.source_dataset,
                    version=record.source_version,
                    site=record.source_site,
                    record_ref=record.record_ref,
                    patient_key=record.patient_key,
                    role=record.role,
                    codes=record.native_snomed_codes,
                    duration=12.0,
                )
            )
        else:
            records.append(record)
    assert_lockbox_unchanged(frozen, build_multisite_manifest(records))


def test_manifest_loader_rejects_hash_tampering_and_noncanonical_order() -> None:
    manifest = _three_source_manifest()
    payload = json.loads(json.dumps(manifest.to_dict(), allow_nan=False))
    payload["records"][0]["duration_seconds"] = 99.0
    with pytest.raises(MultisiteIntegrityError, match="SHA-256 mismatch"):
        MultisiteManifest.from_dict(payload)

    reordered = json.loads(json.dumps(manifest.to_dict(), allow_nan=False))
    reordered["records"] = list(reversed(reordered["records"]))
    with pytest.raises(MultisiteIntegrityError, match="canonically sorted"):
        MultisiteManifest.from_dict(reordered)


def test_reviewed_ontology_maps_or_explicitly_excludes_every_native_code() -> None:
    record = _record()
    manifest = build_multisite_manifest([record])
    decisions = [
        _reviewed_decision(
            dataset=record.source_dataset,
            version=record.source_version,
            code=NORMAL,
            label="NORM",
        ),
        _reviewed_decision(
            dataset=record.source_dataset,
            version=record.source_version,
            code=AF,
            label=None,
            action=MappingAction.EXCLUDE,
        ),
    ]
    ontology = build_reviewed_shared_ontology(
        manifest,
        ontology_name="PTB-XL five-superclass bridge",
        ontology_version="1.0",
        shared_labels=("NORM", "MI", "STTC", "CD", "HYP"),
        decisions=reversed(decisions),
    )

    assert ontology.labels_for(record) == ("NORM",)
    assert ontology.manifest_sha256 == manifest.manifest_sha256
    repeated = build_reviewed_shared_ontology(
        manifest,
        ontology_name="PTB-XL five-superclass bridge",
        ontology_version="1.0",
        shared_labels=("NORM", "MI", "STTC", "CD", "HYP"),
        decisions=decisions,
    )
    assert repeated.artifact_sha256 == ontology.artifact_sha256
    payload = json.loads(json.dumps(ontology.to_dict(), allow_nan=False))
    assert ReviewedSharedOntology.from_dict(payload, manifest=manifest) == ontology


def test_shared_ontology_refuses_unreviewed_decisions() -> None:
    record = _record(codes=(NORMAL,))
    manifest = build_multisite_manifest([record])
    unreviewed = OntologyMappingDecision.create(
        source_dataset=record.source_dataset,
        source_version=record.source_version,
        native_snomed_code=NORMAL,
        action=MappingAction.MAP,
        shared_label="NORM",
        status=ReviewStatus.UNREVIEWED,
        rationale="Awaiting ontology panel review.",
    )

    with pytest.raises(OntologyReviewError, match="refuses unreviewed"):
        build_reviewed_shared_ontology(
            manifest,
            ontology_name="shared",
            ontology_version="1",
            shared_labels=("NORM",),
            decisions=(unreviewed,),
        )


def test_shared_ontology_requires_exact_manifest_code_coverage() -> None:
    record = _record()
    manifest = build_multisite_manifest([record])
    normal = _reviewed_decision(
        dataset=record.source_dataset,
        version=record.source_version,
        code=NORMAL,
        label="NORM",
    )
    with pytest.raises(OntologyReviewError, match="missing"):
        build_reviewed_shared_ontology(
            manifest,
            ontology_name="shared",
            ontology_version="1",
            shared_labels=("NORM",),
            decisions=(normal,),
        )

    extra = _reviewed_decision(
        dataset=record.source_dataset,
        version=record.source_version,
        code=RBBB,
        label="CD",
    )
    af = _reviewed_decision(
        dataset=record.source_dataset,
        version=record.source_version,
        code=AF,
        label=None,
        action=MappingAction.EXCLUDE,
    )
    with pytest.raises(OntologyReviewError, match="extra"):
        build_reviewed_shared_ontology(
            manifest,
            ontology_name="shared",
            ontology_version="1",
            shared_labels=("NORM", "CD"),
            decisions=(normal, af, extra),
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: OntologyMappingDecision.create(
            source_dataset="CPSC",
            source_version="2018",
            native_snomed_code=NORMAL,
            action=MappingAction.MAP,
            shared_label=None,
            status=ReviewStatus.REVIEWED,
            reviewed_by="panel",
            reviewed_at_utc="2026-08-24T02:00:00Z",
            review_reference="log",
            rationale="reviewed",
        ),
        lambda: OntologyMappingDecision.create(
            source_dataset="CPSC",
            source_version="2018",
            native_snomed_code=NORMAL,
            action=MappingAction.EXCLUDE,
            shared_label="NORM",
            status=ReviewStatus.REVIEWED,
            reviewed_by="panel",
            reviewed_at_utc="2026-08-24T02:00:00Z",
            review_reference="log",
            rationale="reviewed",
        ),
        lambda: OntologyMappingDecision.create(
            source_dataset="CPSC",
            source_version="2018",
            native_snomed_code="not-a-code",
            action=MappingAction.MAP,
            shared_label="NORM",
            status=ReviewStatus.REVIEWED,
            reviewed_by="panel",
            reviewed_at_utc="2026-08-24T02:00:00Z",
            review_reference="log",
            rationale="reviewed",
        ),
        lambda: OntologyMappingDecision.create(
            source_dataset="CPSC",
            source_version="2018",
            native_snomed_code=NORMAL,
            action=MappingAction.MAP,
            shared_label="NORM",
            status=ReviewStatus.REVIEWED,
            reviewed_by=None,
            reviewed_at_utc="2026-08-24T02:00:00Z",
            review_reference="log",
            rationale="reviewed",
        ),
        lambda: OntologyMappingDecision.create(
            source_dataset="CPSC",
            source_version="2018",
            native_snomed_code=NORMAL,
            action=MappingAction.MAP,
            shared_label="NORM",
            status=ReviewStatus.REVIEWED,
            reviewed_by="panel",
            reviewed_at_utc="2026-99-99T02:00:00Z",
            review_reference="log",
            rationale="reviewed",
        ),
    ],
)
def test_malformed_ontology_decisions_are_rejected(factory: object) -> None:
    with pytest.raises((OntologyReviewError, MultisiteManifestError)):
        factory()  # type: ignore[operator]


def test_ontology_rejects_undeclared_target_and_hash_tampering() -> None:
    record = _record(codes=(NORMAL,))
    manifest = build_multisite_manifest([record])
    decision = _reviewed_decision(
        dataset=record.source_dataset,
        version=record.source_version,
        code=NORMAL,
        label="NORM",
    )
    with pytest.raises(OntologyReviewError, match="not a declared"):
        build_reviewed_shared_ontology(
            manifest,
            ontology_name="shared",
            ontology_version="1",
            shared_labels=("MI",),
            decisions=(decision,),
        )

    ontology = build_reviewed_shared_ontology(
        manifest,
        ontology_name="shared",
        ontology_version="1",
        shared_labels=("NORM",),
        decisions=(decision,),
    )
    payload = json.loads(json.dumps(ontology.to_dict(), allow_nan=False))
    payload["decisions"][0]["rationale"] = "tampered"
    with pytest.raises(MultisiteIntegrityError, match="SHA-256 mismatch"):
        ReviewedSharedOntology.from_dict(payload, manifest=manifest)


def test_ontology_is_bound_to_exact_manifest_hash() -> None:
    first_record = _record(codes=(NORMAL,))
    first_manifest = build_multisite_manifest([first_record])
    decision = _reviewed_decision(
        dataset=first_record.source_dataset,
        version=first_record.source_version,
        code=NORMAL,
        label="NORM",
    )
    ontology = build_reviewed_shared_ontology(
        first_manifest,
        ontology_name="shared",
        ontology_version="1",
        shared_labels=("NORM",),
        decisions=(decision,),
    )
    second_manifest = build_multisite_manifest(
        [_record(record_ref="records/A0002", codes=(NORMAL,))]
    )

    with pytest.raises(MultisiteIntegrityError, match="different manifest"):
        ReviewedSharedOntology.from_dict(ontology.to_dict(), manifest=second_manifest)
