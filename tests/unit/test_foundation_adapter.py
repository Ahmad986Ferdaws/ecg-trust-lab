from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import Tensor, nn

from ecg_trust.foundation import (
    EXTERNAL_ONLY_LIMIT,
    RESEARCH_USE_LIMIT,
    ComparisonCohortMember,
    ComparisonRole,
    EvaluationMode,
    FoundationAdapterError,
    FoundationEncoderAdapter,
    FoundationEncoderProtocol,
    FoundationError,
    FoundationIntegrityError,
    FoundationModelSpec,
    IndependenceAssessment,
    KnownOverlapDisclosure,
    OverlapStatus,
    PretrainingDatasetDisclosure,
    TrainabilityError,
    TrainabilityPolicy,
    assert_comparison_role_isolation,
    model_state_sha256,
    validate_foundation_comparison_plan,
    verify_trainable_parameters,
)


class TinyEncoder(nn.Module):
    def __init__(self, embedding_dimension: int = 4) -> None:
        super().__init__()
        self.projection = nn.Linear(12, embedding_dimension)
        with torch.no_grad():
            self.projection.weight.fill_(0.05)
            self.projection.bias.copy_(
                torch.linspace(0.0, 0.03, embedding_dimension, dtype=torch.float32)
            )
        self.saw_eval = False
        self.saw_inference = False

    def forward(self, waveforms: Tensor) -> Tensor:
        self.saw_eval = not self.training
        self.saw_inference = torch.is_inference_mode_enabled()
        return self.projection(waveforms.mean(dim=2))


class WrongShapeEncoder(TinyEncoder):
    def forward(self, waveforms: Tensor) -> Tensor:
        return torch.zeros((waveforms.shape[0], 5), dtype=torch.float32)


class WrongDtypeEncoder(TinyEncoder):
    def forward(self, waveforms: Tensor) -> Tensor:
        return super().forward(waveforms).to(dtype=torch.float64)


class NonfiniteEncoder(TinyEncoder):
    def forward(self, waveforms: Tensor) -> Tensor:
        output = super().forward(waveforms)
        output[0, 0] = float("nan")
        return output


class NonTensorEncoder(TinyEncoder):
    def forward(self, waveforms: Tensor) -> list[Tensor]:  # type: ignore[override]
        return [super().forward(waveforms)]


class StateMutatingEncoder(TinyEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("invocation_count", torch.zeros((), dtype=torch.int64))

    def forward(self, waveforms: Tensor) -> Tensor:
        self.invocation_count.add_(1)
        return super().forward(waveforms)


class InputMutatingEncoder(TinyEncoder):
    def forward(self, waveforms: Tensor) -> Tensor:
        waveforms.add_(0.1)
        return super().forward(waveforms)


@pytest.fixture
def batch() -> Tensor:
    return torch.linspace(-1.0, 1.0, 2 * 12 * 1_000, dtype=torch.float32).reshape(2, 12, 1_000)


def _pretraining(
    name: str = "PTB-XL",
    version: str = "1.0.3",
) -> PretrainingDatasetDisclosure:
    return PretrainingDatasetDisclosure.create(
        dataset_name=name,
        dataset_version=version,
        source_url="https://physionet.org/content/ptb-xl/1.0.3/",
        patient_deduplication_disclosed=False,
    )


def _overlap(
    *,
    evaluation_name: str = "PTB-XL",
    evaluation_version: str = "1.0.3",
    pretraining_name: str = "PTB-XL",
    pretraining_version: str = "1.0.3",
    status: OverlapStatus = OverlapStatus.CONFIRMED,
) -> KnownOverlapDisclosure:
    return KnownOverlapDisclosure.create(
        evaluation_dataset_name=evaluation_name,
        evaluation_dataset_version=evaluation_version,
        pretraining_dataset_name=pretraining_name,
        pretraining_dataset_version=pretraining_version,
        status=status,
        rationale="Provider disclosure reviewed for this retrospective comparison.",
    )


def _spec(
    *,
    pretraining: tuple[PretrainingDatasetDisclosure, ...] | None = None,
    overlaps: tuple[KnownOverlapDisclosure, ...] | None = None,
) -> FoundationModelSpec:
    datasets = (_pretraining(),) if pretraining is None else pretraining
    disclosures = (_overlap(),) if overlaps is None else overlaps
    return FoundationModelSpec.create(
        model_id="external-ecg-encoder",
        architecture="tiny-test-encoder",
        checkpoint_sha256="sha256:" + "a" * 64,
        checkpoint_source_url="https://models.example.org/ecg/checkpoint.bin",
        checkpoint_revision="release-1",
        license_identifier="Apache-2.0",
        license_url="https://www.apache.org/licenses/LICENSE-2.0",
        pretraining_datasets=datasets,
        known_overlaps=disclosures,
        embedding_dimension=4,
    )


def _frozen_policy() -> TrainabilityPolicy:
    return TrainabilityPolicy.create(
        mode=EvaluationMode.FROZEN_ENCODER,
        allowed_trainable_parameter_names=(),
    )


def _freeze(model: nn.Module) -> nn.Module:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _adapter(encoder: nn.Module) -> FoundationEncoderAdapter:
    return FoundationEncoderAdapter(
        spec=_spec(),
        encoder=_freeze(encoder),
        trainability_policy=_frozen_policy(),
    )


def test_foundation_spec_is_immutable_canonical_hash_bound_and_round_trips() -> None:
    second_dataset = _pretraining("MIMIC-IV-ECG", "1.0")
    first_dataset = _pretraining()
    second_overlap = _overlap(
        evaluation_name="SPH",
        evaluation_version="1.0",
        pretraining_name="MIMIC-IV-ECG",
        pretraining_version="1.0",
        status=OverlapStatus.NONE_KNOWN,
    )
    first_overlap = _overlap()
    spec = _spec(
        pretraining=(first_dataset, second_dataset),
        overlaps=(second_overlap, first_overlap),
    )

    artifact = spec.to_dict()
    restored = FoundationModelSpec.from_dict(artifact)

    assert restored == spec
    assert restored.to_dict() == artifact
    assert [item.dataset_name for item in spec.pretraining_datasets] == [
        "MIMIC-IV-ECG",
        "PTB-XL",
    ]
    assert artifact["local_pretraining_performed"] is False
    assert artifact["scope_limit"] == EXTERNAL_ONLY_LIMIT
    assert artifact["research_use_limit"] == RESEARCH_USE_LIMIT
    assert artifact["input_contract"] == {
        "ordered_leads": [
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ],
        "samples": 1_000,
        "sampling_frequency_hz": 100,
        "unit": "millivolts",
        "dtype": "float32",
    }
    assert artifact["embedding_contract"] == {"dimension": 4, "dtype": "float32"}
    assert str(artifact["spec_sha256"]).startswith("sha256:")
    with pytest.raises(FrozenInstanceError):
        spec.embedding_dimension = 8  # type: ignore[misc]


def test_foundation_spec_integrity_and_external_lineage_fail_closed() -> None:
    artifact = _spec().to_dict()
    tampered = dict(artifact)
    tampered["architecture"] = "different"
    with pytest.raises(FoundationIntegrityError, match="SHA-256"):
        FoundationModelSpec.from_dict(tampered)

    local_claim = dict(artifact)
    local_claim["local_pretraining_performed"] = True
    with pytest.raises(FoundationIntegrityError, match="prohibited"):
        FoundationModelSpec.from_dict(local_claim)

    with pytest.raises(FoundationError, match="HTTPS"):
        FoundationModelSpec.create(
            model_id="model",
            architecture="encoder",
            checkpoint_sha256="sha256:" + "b" * 64,
            checkpoint_source_url="C:/local/checkpoint.pt",
            checkpoint_revision="v1",
            license_identifier="MIT",
            license_url="https://opensource.org/license/mit",
            pretraining_datasets=(_pretraining(),),
            known_overlaps=(),
            embedding_dimension=4,
        )


def test_spec_rejects_bad_hash_dimension_and_undeclared_overlap() -> None:
    with pytest.raises(FoundationIntegrityError, match="SHA-256"):
        FoundationModelSpec.create(
            model_id="model",
            architecture="encoder",
            checkpoint_sha256="not-a-hash",
            checkpoint_source_url="https://example.org/model.pt",
            checkpoint_revision="v1",
            license_identifier="MIT",
            license_url="https://opensource.org/license/mit",
            pretraining_datasets=(_pretraining(),),
            known_overlaps=(),
            embedding_dimension=4,
        )
    with pytest.raises(FoundationError, match="embedding_dimension"):
        FoundationModelSpec.create(
            model_id="model",
            architecture="encoder",
            checkpoint_sha256="sha256:" + "b" * 64,
            checkpoint_source_url="https://example.org/model.pt",
            checkpoint_revision="v1",
            license_identifier="MIT",
            license_url="https://opensource.org/license/mit",
            pretraining_datasets=(_pretraining(),),
            known_overlaps=(),
            embedding_dimension=0,
        )
    undeclared = _overlap(
        pretraining_name="MIMIC-IV-ECG",
        pretraining_version="1.0",
    )
    with pytest.raises(FoundationError, match="undeclared"):
        _spec(overlaps=(undeclared,))


def test_trainability_policies_are_explicit_sorted_and_mode_specific() -> None:
    assert _frozen_policy().allowed_trainable_parameter_names == ()
    linear = TrainabilityPolicy.create(
        mode=EvaluationMode.LINEAR_PROBE,
        allowed_trainable_parameter_names=("projection.bias", "projection.weight"),
    )
    peft = TrainabilityPolicy.create(
        mode=EvaluationMode.PARAMETER_EFFICIENT_TUNING,
        allowed_trainable_parameter_names=("projection.bias",),
    )
    assert linear.to_dict()["verification"] == "exact_set_fail_closed"
    assert peft.mode is EvaluationMode.PARAMETER_EFFICIENT_TUNING

    with pytest.raises(FoundationError, match="permits no"):
        TrainabilityPolicy.create(
            mode=EvaluationMode.FROZEN_ENCODER,
            allowed_trainable_parameter_names=("projection.weight",),
        )
    with pytest.raises(FoundationError, match="explicit set"):
        TrainabilityPolicy.create(
            mode=EvaluationMode.LINEAR_PROBE,
            allowed_trainable_parameter_names=(),
        )
    with pytest.raises(FoundationError, match="sorted"):
        TrainabilityPolicy.create(
            mode=EvaluationMode.LINEAR_PROBE,
            allowed_trainable_parameter_names=("projection.weight", "projection.bias"),
        )


def test_exact_trainable_parameter_verification_for_all_modes() -> None:
    frozen = _freeze(TinyEncoder())
    verify_trainable_parameters(frozen, _frozen_policy())

    linear_model = TinyEncoder()
    linear_policy = TrainabilityPolicy.create(
        mode=EvaluationMode.LINEAR_PROBE,
        allowed_trainable_parameter_names=("projection.bias", "projection.weight"),
    )
    verify_trainable_parameters(linear_model, linear_policy)

    linear_model.projection.bias.requires_grad_(False)
    with pytest.raises(TrainabilityError, match="declared_but_frozen"):
        verify_trainable_parameters(linear_model, linear_policy)

    missing_policy = TrainabilityPolicy.create(
        mode=EvaluationMode.PARAMETER_EFFICIENT_TUNING,
        allowed_trainable_parameter_names=("adapter.weight",),
    )
    with pytest.raises(TrainabilityError, match="missing"):
        verify_trainable_parameters(linear_model, missing_policy)


@pytest.mark.parametrize(
    "mode",
    [EvaluationMode.LINEAR_PROBE, EvaluationMode.PARAMETER_EFFICIENT_TUNING],
)
def test_adapter_records_nonfrozen_mode_after_exact_trainability_verification(
    batch: Tensor,
    mode: EvaluationMode,
) -> None:
    encoder = TinyEncoder()
    policy = TrainabilityPolicy.create(
        mode=mode,
        allowed_trainable_parameter_names=("projection.bias", "projection.weight"),
    )
    adapter = FoundationEncoderAdapter(
        spec=_spec(),
        encoder=encoder,
        trainability_policy=policy,
    )

    artifact = adapter.extract(batch)

    assert artifact.evaluation_mode is mode
    assert artifact.to_private_metadata()["evaluation_mode"] == mode.value
    assert not artifact.embeddings.requires_grad


def test_frozen_adapter_runs_eval_inference_restores_mode_and_is_deterministic(
    batch: Tensor,
) -> None:
    encoder = _freeze(TinyEncoder())
    encoder.train()
    adapter = FoundationEncoderAdapter(
        spec=_spec(),
        encoder=encoder,
        trainability_policy=_frozen_policy(),
    )
    before = batch.clone()

    first = adapter.extract(batch)
    second = adapter.extract(batch)

    assert isinstance(encoder, FoundationEncoderProtocol)
    assert encoder.saw_eval is True
    assert encoder.saw_inference is True
    assert encoder.training is True
    assert torch.equal(batch, before)
    assert first.embeddings.shape == (2, 4)
    assert first.embeddings.dtype is torch.float32
    assert torch.isfinite(first.embeddings).all().item()
    assert not first.embeddings.requires_grad
    assert first.artifact_sha256 == second.artifact_sha256
    assert first.embedding_tensor_sha256 == second.embedding_tensor_sha256
    assert first.encoder_state_sha256 == model_state_sha256(encoder)


def test_embedding_artifact_is_defensive_hash_bound_and_identifier_free(batch: Tensor) -> None:
    artifact = _adapter(TinyEncoder()).extract(batch)
    detached = artifact.embeddings
    detached[0, 0] += 100.0

    assert not torch.equal(detached, artifact.embeddings)
    metadata = artifact.to_private_metadata()
    serialized = json.dumps(metadata, sort_keys=True)
    assert metadata["input_shape"] == [2, 12, 1_000]
    assert metadata["embedding_shape"] == [2, 4]
    assert metadata["scope_limit"] == EXTERNAL_ONLY_LIMIT
    assert metadata["research_use_limit"] == RESEARCH_USE_LIMIT
    assert str(metadata["artifact_sha256"]).startswith("sha256:")
    assert "patient_key" not in serialized
    assert "record" not in serialized
    assert "path" not in serialized
    assert "checkpoint.bin" not in serialized

    changed = batch.clone()
    changed[0, 0, 0] += torch.finfo(torch.float32).eps
    changed_artifact = _adapter(TinyEncoder()).extract(changed)
    assert changed_artifact.input_batch_sha256 != artifact.input_batch_sha256
    assert changed_artifact.artifact_sha256 != artifact.artifact_sha256


@pytest.mark.parametrize(
    ("transform", "message"),
    [
        (lambda value: value.to(dtype=torch.float64), "float32"),
        (lambda value: value[:, :11], "shape"),
        (
            lambda value: value.transpose(1, 2).contiguous().transpose(1, 2),
            "contiguous",
        ),
        (lambda value: value.clone().requires_grad_(True), "detached"),
    ],
)
def test_adapter_rejects_noncanonical_batches(
    batch: Tensor,
    transform: object,
    message: str,
) -> None:
    transformed = transform(batch)  # type: ignore[operator]
    with pytest.raises(FoundationAdapterError, match=message):
        _adapter(TinyEncoder()).extract(transformed)


def test_adapter_rejects_nonfinite_input(batch: Tensor) -> None:
    invalid = batch.clone()
    invalid[0, 0, 0] = float("nan")
    with pytest.raises(FoundationAdapterError, match="finite"):
        _adapter(TinyEncoder()).extract(invalid)


@pytest.mark.parametrize(
    ("encoder", "message"),
    [
        (WrongShapeEncoder(), "shape"),
        (WrongDtypeEncoder(), "dtype"),
        (NonfiniteEncoder(), "non-finite"),
        (NonTensorEncoder(), "torch.Tensor"),
    ],
)
def test_adapter_rejects_invalid_embedding_outputs(
    batch: Tensor,
    encoder: nn.Module,
    message: str,
) -> None:
    with pytest.raises(FoundationAdapterError, match=message):
        _adapter(encoder).extract(batch)


def test_frozen_adapter_rejects_and_restores_state_mutation(batch: Tensor) -> None:
    encoder = _freeze(StateMutatingEncoder())
    before = encoder.invocation_count.clone()

    with pytest.raises(FoundationAdapterError, match="mutated parameters or buffers"):
        _adapter(encoder).extract(batch)

    assert torch.equal(encoder.invocation_count, before)


def test_adapter_rejects_input_mutation_and_preserves_caller_batch(batch: Tensor) -> None:
    before = batch.clone()
    with pytest.raises(FoundationAdapterError, match="mutated its canonical input"):
        _adapter(InputMutatingEncoder()).extract(batch)
    assert torch.equal(batch, before)


def test_frozen_mode_rejects_any_trainable_encoder_parameter(batch: Tensor) -> None:
    encoder = TinyEncoder()
    adapter = FoundationEncoderAdapter(
        spec=_spec(),
        encoder=encoder,
        trainability_policy=_frozen_policy(),
    )
    with pytest.raises(TrainabilityError, match="unexpected"):
        adapter.extract(batch)


def _member(
    patient: str,
    *,
    dataset: str = "PTB-XL",
    version: str = "1.0.3",
    site: str = "site-a",
    role: ComparisonRole = ComparisonRole.DEVELOPMENT,
) -> ComparisonCohortMember:
    return ComparisonCohortMember.create(
        dataset_name=dataset,
        dataset_version=version,
        site_key=site,
        patient_key=patient,
        role=role,
    )


def test_comparison_plan_detects_exact_overlap_and_downgrades_independence() -> None:
    plan = validate_foundation_comparison_plan(
        _spec(),
        trainability_policy=_frozen_policy(),
        cohort_members=[_member("secret-1"), _member("secret-2")],
    )

    assert plan.independence_assessment is IndependenceAssessment.NON_INDEPENDENT_KNOWN_OVERLAP
    assert plan.independence_claim == "non_independent_external_representation_comparison"
    assert "exact_dataset_version_in_pretraining" in plan.overlap_reason_codes
    public = plan.to_public_dict()
    serialized = json.dumps(public, sort_keys=True)
    assert public["local_pretraining_performed"] is False
    assert public["patient_count"] == 2
    assert "secret-1" not in serialized
    assert "site-a" not in serialized


def test_none_known_possible_and_undisclosed_overlap_are_distinct() -> None:
    no_known_spec = _spec(
        overlaps=(
            _overlap(
                evaluation_name="SPH",
                evaluation_version="1.0",
                status=OverlapStatus.NONE_KNOWN,
            ),
        )
    )
    no_known = validate_foundation_comparison_plan(
        no_known_spec,
        trainability_policy=_frozen_policy(),
        cohort_members=[_member("one", dataset="SPH", version="1.0")],
    )
    assert no_known.independence_assessment is IndependenceAssessment.NO_DISCLOSED_OVERLAP
    assert no_known.independence_claim == "no_disclosed_overlap_not_proof_of_independence"

    possible = validate_foundation_comparison_plan(
        _spec(overlaps=()),
        trainability_policy=_frozen_policy(),
        cohort_members=[_member("one", version="2.0")],
    )
    assert possible.independence_assessment is IndependenceAssessment.LIMITED_POSSIBLE_OVERLAP

    undisclosed = validate_foundation_comparison_plan(
        _spec(overlaps=()),
        trainability_policy=_frozen_policy(),
        cohort_members=[_member("one", dataset="SPH", version="1.0")],
    )
    assert undisclosed.independence_assessment is (
        IndependenceAssessment.LIMITED_UNDISCLOSED_OVERLAP
    )


def test_exact_dataset_match_overrides_incorrect_none_known_disclosure() -> None:
    contradictory = _spec(
        overlaps=(_overlap(status=OverlapStatus.NONE_KNOWN),),
    )
    plan = validate_foundation_comparison_plan(
        contradictory,
        trainability_policy=_frozen_policy(),
        cohort_members=[_member("one")],
    )
    assert plan.independence_assessment is IndependenceAssessment.NON_INDEPENDENT_KNOWN_OVERLAP
    assert "none_known_disclosure_overridden_by_exact_dataset_match" in plan.overlap_reason_codes


def test_provider_confirmed_cross_dataset_overlap_is_nonindependent() -> None:
    confirmed = _spec(
        overlaps=(
            _overlap(
                evaluation_name="SPH",
                evaluation_version="1.0",
                status=OverlapStatus.CONFIRMED,
            ),
        )
    )
    plan = validate_foundation_comparison_plan(
        confirmed,
        trainability_policy=_frozen_policy(),
        cohort_members=[_member("one", dataset="SPH", version="1.0")],
    )
    assert plan.independence_assessment is IndependenceAssessment.NON_INDEPENDENT_KNOWN_OVERLAP
    assert "provider_disclosed_confirmed_overlap" in plan.overlap_reason_codes


def test_comparison_patient_and_site_role_isolation_fail_closed() -> None:
    patient_collision = [
        _member("same", site="site-a", role=ComparisonRole.DEVELOPMENT),
        _member("same", site="site-b", role=ComparisonRole.CALIBRATION),
    ]
    with pytest.raises(FoundationError, match="patient"):
        assert_comparison_role_isolation(patient_collision)

    site_collision = [
        _member("one", role=ComparisonRole.DEVELOPMENT),
        _member("two", role=ComparisonRole.CALIBRATION),
    ]
    with pytest.raises(FoundationError, match="site"):
        validate_foundation_comparison_plan(
            _spec(),
            trainability_policy=_frozen_policy(),
            cohort_members=site_collision,
        )


def test_comparison_rejects_local_pretraining_claims_and_duplicates() -> None:
    with pytest.raises(FoundationError, match="pretraining claims are prohibited"):
        validate_foundation_comparison_plan(
            _spec(),
            trainability_policy=_frozen_policy(),
            cohort_members=[_member("one")],
            local_pretraining_claimed=True,
        )
    duplicate = _member("one")
    with pytest.raises(FoundationError, match="duplicate"):
        validate_foundation_comparison_plan(
            _spec(),
            trainability_policy=_frozen_policy(),
            cohort_members=[duplicate, duplicate],
        )
