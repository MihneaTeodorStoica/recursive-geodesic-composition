import pytest
import torch
import torch.nn.functional as F

from recursive_geodesic_composition.architecture import GeodesicMerge, recursive_reduce, weighted_average
from recursive_geodesic_composition.prepare_kary import _segmentations_for_role, kary_target_spans, make_kary_row
from recursive_geodesic_composition.evaluate_kary_validation import _arm_label, reduction_depth, roots_for_rows
from recursive_geodesic_composition.train import _assert_finite_batch, _merge_level, _validate_cache_arity


def make(arity=4, decoder=False):
    return GeodesicMerge(
        d=8, hidden=16, layers=1, heads=4, dropout=0.0,
        max_angle=0.5, geodesic_parameterization="tangent_expmap",
        importance_mode="semantic", arity=arity, decoder=decoder,
        decoder_mode="strict" if decoder else "legacy", decoder_layers=1,
    )


def kary_inputs(batch=2):
    torch.manual_seed(81)
    children = F.normalize(torch.randn(batch, 4, 8), dim=-1)
    lengths = torch.tensor([[2.0, 3.0, 5.0, 7.0], [11.0, 13.0, 0.0, 0.0]])[:batch]
    return children, lengths


def test_explicit_arity2_is_exact_default_legacy_compatibility():
    kwargs = dict(d=8, hidden=16, layers=1, heads=4, dropout=0.0,
                  max_angle=0.5, decoder=True, decoder_layers=1)
    torch.manual_seed(901)
    default = GeodesicMerge(**kwargs)
    default_rng = torch.get_rng_state().clone()
    torch.manual_seed(901)
    explicit = GeodesicMerge(**kwargs, arity=2)
    assert torch.equal(default_rng, torch.get_rng_state())
    assert list(default.state_dict()) == list(explicit.state_dict())
    assert all(torch.equal(value, explicit.state_dict()[key])
               for key, value in default.state_dict().items())
    children = F.normalize(torch.randn(3, 2, 8), dim=-1)
    lengths = torch.tensor([[1.0, 2.0], [4.0, 3.0], [1.0, 0.0]])
    assert torch.equal(default.merge(children, lengths), explicit.merge(children, lengths))


def test_k4_shapes_masks_and_order_sensitivity():
    model = make()
    children, lengths = kary_inputs()
    root, aux = model.merge(children, lengths, return_aux=True)
    assert root.shape == (2, 8)
    assert aux["effective_weights"].shape == (2, 4)
    assert torch.equal(aux["active_mask"], lengths > 0)
    assert torch.equal(aux["effective_weights"][1, 2:], torch.zeros(2))
    assert torch.allclose(aux["effective_weights"].sum(1), torch.ones(2))

    three = lengths[:1].clone(); three[:, 3] = 0
    assert model.merge(children[:1], three).shape == (1, 8)
    with pytest.raises(ValueError, match="contiguous prefix"):
        model.merge(children[:1], torch.tensor([[1.0, 0.0, 1.0, 0.0]]))
    with pytest.raises(ValueError, match="at least one"):
        model.merge(children[:1], torch.zeros(1, 4))
    with pytest.raises(ValueError, match="carry singletons"):
        model.merge(children[:1], torch.tensor([[1.0, 0.0, 0.0, 0.0]]))

    # Nonzero contextual correction makes learned ordered slots observable.
    with torch.no_grad():
        model.direction.weight.normal_(0.0, 0.1)
        model.direction.bias.normal_(0.0, 0.1)
    original = model.merge(children[:1], lengths[:1])
    permutation = torch.tensor([1, 0, 2, 3])
    reordered = model.merge(children[:1, permutation], lengths[:1, permutation])
    assert not torch.allclose(original, reordered, atol=1e-7, rtol=0)


def test_evaluator_casts_fp16_cached_leaves_to_model_dtype():
    model = make().float().eval()
    children, lengths = kary_inputs(batch=1)
    rows = [{"leaves": children[0].half(), "lengths": lengths[0]}]
    roots, saturation = roots_for_rows(model, rows, torch.device("cpu"), batch_size=1)
    assert roots.dtype == torch.float32
    assert roots.shape == (1, 8)
    assert torch.isfinite(roots).all()
    assert saturation == 0.0


def test_count_conservation_and_genuine_four_way_call_depth():
    model = make()
    torch.manual_seed(8)
    leaves = F.normalize(torch.randn(2, 17, 8), dim=-1)
    lengths = torch.arange(1, 18, dtype=torch.float32).repeat(2, 1)
    calls = []
    original = model.merge

    def recording_merge(children, masses, *args, **kwargs):
        calls.append((children.shape, (masses > 0).sum(1).tolist()))
        return original(children, masses, *args, **kwargs)

    model.merge = recording_merge
    roots, levels = recursive_reduce(model, leaves, lengths)
    assert roots.shape == (2, 8)
    assert len(calls) == len(levels) == 3
    assert all(shape[1] == 4 for shape, _ in calls)
    assert set(calls[0][1]) == {4}
    assert set(calls[1][1]) == {4}
    assert set(calls[2][1]) == {2}

    nodes, sizes = leaves, lengths
    while nodes.shape[1] > 1:
        nodes, sizes, _, _, _, _ = _merge_level(model, nodes, sizes)
    assert torch.equal(sizes[:, 0], lengths.sum(1))


def test_validation_segmentations_cover_the_identical_safe_span():
    class Document:
        id = "same-source"
    cfg = {"teacher_max_tokens": 6144, "chunk_tokens": 510,
           "validation_segmentations": 2, "random_chunk_min": 256,
           "random_chunk_max": 510, "seed": 4}
    token_ids = list(range(6144))
    segmentations = _segmentations_for_role(cfg, Document(), token_ids, "validation")
    assert len(segmentations) == 2
    assert [token for chunk in segmentations[0][1] for token in chunk] == [
        token for chunk in segmentations[1][1] for token in chunk]


def test_target_provenance_is_direct_safe_and_binary_cache_is_rejected():
    chunks = [[1, 2], [3, 4, 5], [6], [7, 8], [9, 10]]
    spans, provenance = kary_target_spans(chunks, 4, teacher_max_tokens=6)
    assert [item["group_arity"] for item in provenance] == [4, 2]
    assert spans[0] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert provenance[0]["leaf_start"] == 0 and provenance[0]["leaf_end"] == 4
    assert provenance[-1]["leaf_start"] == 0 and provenance[-1]["leaf_end"] == 5
    assert [item["teacher_eligible"] for item in provenance] == [False, False]

    binary = {"leaves": torch.ones(4, 8), "lengths": torch.ones(4), "targets": []}
    with pytest.raises(ValueError, match="binary caches"):
        _validate_cache_arity([binary], 4, "train")


class _FakeTeacher:
    def embed(self, spans, batch_size):
        values = []
        for span in spans:
            vector = torch.arange(1, 9, dtype=torch.float32) + sum(span)
            values.append(F.normalize(vector, dim=0))
        return torch.stack(values) if values else torch.empty(0, 8)


def test_target_provenance_guard_rejects_fabrication_above_safe_limit():
    cfg = {"arity": 4, "teacher_max_tokens": 6, "teacher_target_batch_size": 8,
           "teacher_target_max_padded_tokens": 128}
    row = make_kary_row("validation-only", [[1, 2], [3, 4], [5, 6], [7, 8]],
                        _FakeTeacher(), cfg, segmentation="fixed")
    assert row["targets"] == [None]
    _assert_finite_batch([(0, row)])
    row["targets"][0] = torch.ones(8)
    with pytest.raises(ValueError, match="target/provenance mismatch"):
        _assert_finite_batch([(0, row)])


def test_k4_zero_init_tangent_has_finite_nonzero_teacher_gradients():
    model = make()
    children, lengths = kary_inputs()
    root, aux = model.merge(children, lengths, return_aux=True)
    assert torch.equal(root, weighted_average(children, lengths))
    axis = torch.arange(1, 9, dtype=root.dtype).expand_as(root)
    axis = F.normalize(axis - (axis * root).sum(-1, keepdim=True) * root, dim=-1)
    target = F.normalize(root.detach() + 0.25 * axis, dim=-1)
    (1.0 - F.cosine_similarity(root, target, dim=-1).mean()).backward()
    assert model.direction.weight.grad is not None
    assert torch.isfinite(model.direction.weight.grad).all()
    assert model.direction.weight.grad.norm() > 0


def test_k8_masks_contiguous_groups_counts_and_true_depth():
    model = make(arity=8)
    torch.manual_seed(808)
    children = F.normalize(torch.randn(2, 8, 8), dim=-1)
    lengths = torch.tensor([
        [1., 2., 3., 4., 5., 6., 7., 8.],
        [11., 13., 17., 19., 23., 0., 0., 0.],
    ])
    root, aux = model.merge(children, lengths, return_aux=True)
    assert root.shape == (2, 8)
    assert torch.equal(aux["active_mask"], lengths > 0)
    assert torch.equal(aux["effective_weights"][1, 5:], torch.zeros(3))
    assert torch.allclose(aux["effective_weights"].sum(1), torch.ones(2))
    with pytest.raises(ValueError, match="contiguous prefix"):
        model.merge(children[:1], torch.tensor([[1., 1., 0., 1., 0., 0., 0., 0.]]))

    leaves = F.normalize(torch.randn(2, 65, 8), dim=-1)
    counts = torch.arange(1, 66, dtype=torch.float32).repeat(2, 1)
    calls = []
    original = model.merge
    def recording_merge(kids, masses, *args, **kwargs):
        calls.append((kids.shape[1], (masses > 0).sum(1).tolist()))
        return original(kids, masses, *args, **kwargs)
    model.merge = recording_merge
    roots, levels = recursive_reduce(model, leaves, counts)
    assert roots.shape == (2, 8)
    assert len(levels) == reduction_depth(65, 8) == 3
    assert len(calls) == 3 and all(width == 8 for width, _ in calls)
    assert set(calls[0][1]) == {8}
    assert set(calls[1][1]) == {8}
    assert set(calls[2][1]) == {2}
    nodes, masses = leaves, counts
    while nodes.shape[1] > 1:
        nodes, masses, _, _, _, _ = _merge_level(model, nodes, masses)
    assert torch.equal(masses[:, 0], counts.sum(1))


def test_k8_teacher_exactly_6144_is_eligible_and_above_is_omitted():
    cfg = {"arity": 8, "teacher_max_tokens": 6144,
           "teacher_target_batch_size": 8, "teacher_target_max_padded_tokens": 65536}
    exact = make_kary_row("exact-6144", [list(range(i * 768, (i + 1) * 768)) for i in range(8)],
                          _FakeTeacher(), cfg, segmentation="fixed")
    assert exact["target_provenance"][-1]["token_count"] == 6144
    assert exact["target_provenance"][-1]["teacher_eligible"] is True
    assert exact["targets"][-1] is not None
    above = make_kary_row("above-6144", [list(range(i * 769, (i + 1) * 769)) for i in range(8)],
                          _FakeTeacher(), cfg, segmentation="fixed")
    assert above["target_provenance"][-1]["token_count"] == 6152
    assert above["target_provenance"][-1]["teacher_eligible"] is False
    assert above["targets"][-1] is None
    _assert_finite_batch([(0, exact), (1, above)])


def test_k8_fp16_evaluation_and_arity_aware_labels():
    model = make(arity=8).float().eval()
    torch.manual_seed(18)
    rows = [{"leaves": F.normalize(torch.randn(8, 8), dim=-1).half(),
             "lengths": torch.arange(1, 9, dtype=torch.float32)}]
    roots, saturation = roots_for_rows(model, rows, torch.device("cpu"), batch_size=32)
    assert roots.dtype == torch.float32 and roots.shape == (1, 8)
    assert torch.isfinite(roots).all() and saturation == 0.0
    assert _arm_label("reference", 4) == "kary4_reference"
    assert _arm_label("candidate", 8) == "kary8"
    assert _arm_label("reference", 2) == "binary_reference"


def test_k4_strict_parent_only_decoder_and_checkpoint_metadata():
    model = make(decoder=True)
    parent = F.normalize(torch.randn(3, 8), dim=-1)
    children, aux = model.decode(parent, return_aux=True)
    assert children.shape == (3, 4, 8)
    assert aux["importance_logits"].shape == (3, 4)
    assert aux["importance_weights"].shape == (3, 4)
    assert torch.allclose(aux["importance_weights"].sum(1), torch.ones(3))
    metadata = model.architecture_metadata()
    assert metadata["arity"] == 4 and metadata["version"] >= 4
