import torch
import torch.nn.functional as F

from recursive_geodesic_composition.architecture import GeodesicMerge, weighted_average
from recursive_geodesic_composition.train import step_loss_batch


def make(mode="tangent_expmap", max_angle=0.5):
    return GeodesicMerge(
        d=8, hidden=16, layers=1, heads=4, dropout=0.0,
        max_angle=max_angle, geodesic_parameterization=mode,
    )


def inputs(batch=4):
    torch.manual_seed(202)
    children = F.normalize(torch.randn(batch, 2, 8), dim=-1)
    lengths = torch.tensor([[1.0, 3.0], [2.0, 1.0], [5.0, 4.0], [1.0, 1.0]])[:batch]
    return children, lengths


def off_anchor_target(anchor):
    axis = torch.arange(1, anchor.shape[-1] + 1, dtype=anchor.dtype, device=anchor.device)
    axis = axis.expand_as(anchor)
    axis = axis - (axis * anchor).sum(-1, keepdim=True) * anchor
    axis = F.normalize(axis, dim=-1)
    return F.normalize(anchor + 0.35 * axis, dim=-1)


def test_legacy_default_preserves_exact_state_rng_and_outputs():
    kwargs = dict(d=8, hidden=16, layers=1, heads=4, dropout=0.0,
                  max_angle=0.5, decoder=True, decoder_layers=1)
    torch.manual_seed(101)
    default = GeodesicMerge(**kwargs)
    default_rng = torch.get_rng_state().clone()
    torch.manual_seed(101)
    explicit = GeodesicMerge(**kwargs, geodesic_parameterization="legacy")
    assert torch.equal(default_rng, torch.get_rng_state())
    assert list(default.state_dict()) == list(explicit.state_dict())
    assert all(torch.equal(value, explicit.state_dict()[key])
               for key, value in default.state_dict().items())

    children, lengths = inputs(3)
    default_root = default.merge(children, lengths)
    explicit_root = explicit.merge(children, lengths)
    assert torch.equal(default_root, explicit_root)
    assert torch.equal(default.decode(default_root, lengths),
                       explicit.decode(explicit_root, lengths))
    assert default.angle is not None
    assert torch.count_nonzero(default.direction.weight) > 0


def test_tangent_expmap_is_finite_and_exact_anchor_at_zero():
    model = make()
    children, lengths = inputs()
    root, aux = model.merge(children, lengths, return_aux=True)
    anchor = weighted_average(children, lengths)
    assert torch.equal(root, anchor)
    assert torch.equal(aux["anchor"], anchor)
    assert torch.count_nonzero(aux["geodesic_correction"]) == 0
    assert torch.count_nonzero(aux["angle"]) == 0
    assert torch.isfinite(root).all()
    assert torch.isfinite(aux["raw_tangent_norm"]).all()
    assert model.angle is None
    assert torch.count_nonzero(model.direction.weight) == 0
    assert torch.count_nonzero(model.direction.bias) == 0


def test_off_anchor_teacher_has_nonzero_gradient_at_zero_correction():
    model = make()
    children, lengths = inputs(2)
    root, aux = model.merge(children, lengths, return_aux=True)
    target = off_anchor_target(aux["anchor"].detach())
    loss = 1.0 - F.cosine_similarity(root, target, dim=-1).mean()
    loss.backward()
    assert model.direction.weight.grad is not None
    assert torch.isfinite(model.direction.weight.grad).all()
    assert model.direction.weight.grad.norm() > 0
    assert model.direction.bias.grad.norm() > 0


def test_tangent_norm_bound_expmap_norm_and_tangency():
    max_angle = 0.8
    model = make(max_angle=max_angle)
    with torch.no_grad():
        model.direction.weight.normal_(0.0, 20.0)
        model.direction.bias.normal_(0.0, 20.0)
    children, lengths = inputs()
    root, aux = model.merge(children, lengths, return_aux=True)
    correction = aux["geodesic_correction"]
    assert torch.isfinite(root).all() and torch.isfinite(correction).all()
    assert torch.all(aux["angle"] <= max_angle + 1e-6)
    assert torch.allclose(aux["angle"], correction.norm(dim=-1), atol=1e-7, rtol=1e-6)
    assert torch.allclose((correction * aux["anchor"]).sum(-1),
                          torch.zeros(root.shape[0]), atol=2e-5, rtol=0)
    assert torch.allclose(root.norm(dim=-1), torch.ones(root.shape[0]), atol=2e-5, rtol=0)


def test_parameterization_can_leave_pooling_after_one_teacher_step():
    model = make()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("direction."))
    children, lengths = inputs(2)
    pooled = weighted_average(children, lengths)
    target = off_anchor_target(pooled)
    optimizer = torch.optim.SGD(model.direction.parameters(), lr=0.2)
    optimizer.zero_grad()
    before = model.merge(children, lengths)
    (1.0 - F.cosine_similarity(before, target, dim=-1).mean()).backward()
    optimizer.step()
    after, aux = model.merge(children, lengths, return_aux=True)
    assert torch.equal(before, pooled)
    assert aux["angle"].max() > 0
    assert not torch.equal(after, pooled)
    assert F.cosine_similarity(after, target, dim=-1).mean() > F.cosine_similarity(before, target, dim=-1).mean()


def test_geodesic_diagnostics_are_returned_by_training_collection():
    model = make(max_angle=0.5)
    children, lengths = inputs(1)
    row = {"id": "validation-only-fixture", "leaves": children[0],
           "lengths": lengths[0], "targets": [off_anchor_target(weighted_average(children, lengths))[0]]}
    loss, parts = step_loss_batch(model, [row], False, torch.device("cpu"), 0.0)
    assert torch.isfinite(loss)
    assert {"geodesic_angle_mean", "geodesic_angle_p95", "geodesic_angle_max",
            "geodesic_saturation_fraction"}.issubset(parts)
    assert parts["geodesic_angle_mean"] == 0
    assert parts["geodesic_saturation_fraction"] == 0
