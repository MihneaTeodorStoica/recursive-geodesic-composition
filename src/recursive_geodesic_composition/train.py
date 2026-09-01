"""Train one clean encoder variant from prepared HF tensors."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, shutil, tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
import yaml
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from .architecture import GeodesicMerge, contiguous_reduction_groups

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # tensorboard is optional at runtime
    SummaryWriter = None


def _require_free_storage(path, minimum_bytes, context):
    minimum = int(minimum_bytes or 0)
    if minimum < 0:
        raise ValueError("minimum free storage must be non-negative")
    free = shutil.disk_usage(Path(path).resolve()).free
    if free < minimum:
        raise RuntimeError(f"storage guard failed before {context}: free={free} required={minimum}")
    return free


def _require_checkpoint_storage(path, cfg, context):
    """Guard both current free space and projected free space after one checkpoint."""
    minimum = int(cfg.get('minimum_free_bytes_before_checkpoint', 0) or 0)
    estimate = int(cfg.get('estimated_checkpoint_bytes', 0) or 0)
    if estimate < 0:
        raise ValueError("estimated checkpoint bytes must be non-negative")
    return _require_free_storage(path, minimum + estimate, context)


def atomic_torch_save(value, destination):
    """Durably replace a torch checkpoint without exposing a partial file."""
    destination = Path(destination)
    fd, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix=f'.{destination.name}.', suffix='.tmp')
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, 'wb') as handle:
            fd = -1
            # Keep torch's standard zip checkpoint format: existing torch.load
            # callers remain compatible and storages are not duplicated.
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        # Persist the directory entry as well as the checkpoint contents.
        directory_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


_DECODER_PARAMETER_PREFIXES = ("decoder.", "strict_decoder.", "ratio_head.")
_KNOWN_CHECKPOINT_PREFIXES = ("module.", "_orig_mod.")


def _is_decoder_parameter(name):
    return name.startswith(_DECODER_PARAMETER_PREFIXES)


def _set_encoder_trainable(model, trainable):
    """Toggle only merger/semantic-encoder parameters, leaving decoders alone."""
    changed = []
    for name, parameter in model.named_parameters():
        if not _is_decoder_parameter(name):
            parameter.requires_grad_(bool(trainable))
            changed.append(name)
    if not changed:
        raise RuntimeError("model has no encoder parameters to freeze/unfreeze")
    return changed


def _optimizer_parameter_groups(model, encoder_learning_rate_scale=1.0):
    """Return opt-in encoder/decoder LR groups without perturbing old recipes."""
    scale = float(encoder_learning_rate_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("encoder_learning_rate_scale must be finite and positive")
    if scale == 1.0:
        # Preserve the historical single-group AdamW construction exactly when
        # the new option is omitted (or explicitly set to one).
        return model.parameters()
    encoder, decoder = [], []
    for name, parameter in model.named_parameters():
        (decoder if _is_decoder_parameter(name) else encoder).append(parameter)
    if not encoder:
        raise RuntimeError("model has no encoder parameters for LR scaling")
    groups = [{"params": encoder, "lr_scale": scale, "group_name": "encoder"}]
    if decoder:
        groups.append({"params": decoder, "lr_scale": 1.0, "group_name": "decoder"})
    return groups


def _set_optimizer_base_learning_rate(optimizer, base_learning_rate):
    """Apply one base schedule to every group, including its cosine floor."""
    base = float(base_learning_rate)
    if not math.isfinite(base) or base < 0:
        raise ValueError("base learning rate must be finite and non-negative")
    values = {}
    for index, group in enumerate(optimizer.param_groups):
        scale = float(group.get("lr_scale", 1.0))
        group["lr"] = base * scale
        values[str(group.get("group_name", f"group_{index}"))] = group["lr"]
    return values


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _migrate_checkpoint_key(key):
    """Apply only explicit, reversible wrapper-prefix migrations."""
    for prefix in _KNOWN_CHECKPOINT_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):], f"{prefix}*->*"
    return key, None


def _load_encoder_initialization(model, checkpoint_path, expected_sha256=None, *,
                                 exact=False, require_encoder_variant=False,
                                 allow_autoencoder_source=False):
    """Safely load every shared encoder key and no decoder key.

    The checkpoint is read with ``weights_only=True``.  We reject architecture
    mismatches, shape mismatches, partial base encoders, duplicate migrations,
    and decoder-bearing sources rather than silently accepting a weak load.
    Target-only semantic salience and decoder modules retain their deliberate
    initialization when absent from an encoder checkpoint.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"encoder initialization checkpoint not found: {path}")
    actual_hash = _sha256_file(path)
    if expected_sha256 is not None and actual_hash.lower() != str(expected_sha256).lower():
        raise ValueError(
            f"encoder initialization SHA-256 mismatch: expected {expected_sha256}, got {actual_hash}")
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
        raise ValueError("encoder initialization must be a checkpoint containing state_dict")
    source_variant = checkpoint.get('variant')
    if require_encoder_variant and allow_autoencoder_source:
        raise ValueError("exact encoder continuation cannot allow an autoencoder source")
    allowed_variants = (None, 'encoder', 'autoencoder') if allow_autoencoder_source else (
        ('encoder',) if require_encoder_variant else (None, 'encoder'))
    if source_variant not in allowed_variants:
        if require_encoder_variant:
            raise ValueError(f"encoder initialization requires variant='encoder', got {source_variant!r}")
        raise ValueError(f"encoder initialization has disallowed source variant {source_variant!r}")
    architecture = checkpoint.get('architecture') or {}
    source_geometry = architecture.get('geodesic_parameterization')
    if source_geometry is not None and source_geometry != model.geodesic_parameterization:
        raise ValueError(
            f"encoder geometry mismatch: checkpoint={source_geometry}, model={model.geodesic_parameterization}")
    source_angle = architecture.get('max_geodesic_angle')
    if source_angle is not None and not math.isclose(float(source_angle), model.max_angle,
                                                      rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"encoder max-angle mismatch: checkpoint={source_angle}, model={model.max_angle}")
    if exact:
        target_architecture = model.architecture_metadata()
        for key in ('arity', 'importance_mode', 'decoder_mode', 'geodesic_parameterization'):
            if key in architecture and architecture[key] != target_architecture.get(key):
                raise ValueError(
                    f"encoder architecture mismatch for {key}: checkpoint={architecture[key]!r}, "
                    f"model={target_architecture.get(key)!r}")
        source_cfg = checkpoint.get('config') or {}
        exact_controls = {
            'arity': model.arity,
            'encoder_hidden': model.hidden,
            'encoder_layers': len(model.context.layers),
            'heads': model.context.layers[0].self_attn.num_heads,
            'importance_mode': model.importance_mode,
            'geodesic_parameterization': model.geodesic_parameterization,
        }
        for key, target_value in exact_controls.items():
            if key in source_cfg and source_cfg[key] != target_value:
                raise ValueError(
                    f"encoder architecture mismatch for {key}: checkpoint={source_cfg[key]!r}, "
                    f"model={target_value!r}")

    own = model.state_dict()
    migrated, migrations = {}, []
    source_state = checkpoint['state_dict']
    if not isinstance(source_state, dict) or not source_state:
        raise ValueError("encoder initialization state_dict is empty or invalid")
    ignored_decoder_keys = []
    for original_key, value in source_state.items():
        key, migration = _migrate_checkpoint_key(str(original_key))
        if _is_decoder_parameter(key) and allow_autoencoder_source:
            ignored_decoder_keys.append(key)
            continue
        if key in migrated:
            raise ValueError(f"duplicate encoder key after migration: {key}")
        if _is_decoder_parameter(key):
            raise ValueError(f"encoder initialization unexpectedly contains decoder key: {original_key}")
        if key not in own:
            raise ValueError(f"encoder initialization has unknown key: {original_key}")
        if own[key].shape != value.shape:
            raise ValueError(
                f"encoder initialization shape mismatch for {key}: checkpoint={tuple(value.shape)} "
                f"model={tuple(own[key].shape)}")
        migrated[key] = value
        if migration is not None:
            migrations.append(migration)

    # Salience is a target-only semantic extension for token-importance encoder
    # checkpoints. Every other target encoder key is mandatory. Continuations,
    # unlike autoencoder initialization, require the exact encoder state.
    required = (set(own) if exact else
                {key for key in own if not _is_decoder_parameter(key) and
                 not key.startswith('salience.')})
    missing_shared = sorted(required - set(migrated))
    if missing_shared:
        raise ValueError(f"encoder initialization is missing shared keys: {missing_shared[:8]}")
    result = model.load_state_dict(migrated, strict=exact)
    if result.unexpected_keys:
        raise RuntimeError(f"unexpected keys after checked encoder load: {result.unexpected_keys}")
    return {
        'initialization_checkpoint': str(path),
        'initialization_checkpoint_sha256': actual_hash,
        'initialization_shared_key_count': len(migrated),
        'initialization_key_migrations': sorted(set(migrations)),
        'initialization_source_selected_step': checkpoint.get('selected_step'),
        'initialization_source_variant': source_variant,
        'initialization_ignored_decoder_key_count': len(ignored_decoder_keys),
        'continued_autoencoder_retraining': bool(source_variant == 'autoencoder'),
    }


def _load_encoder_continuation(model, checkpoint_path, expected_sha256):
    """Load an exact encoder checkpoint while deliberately restarting AdamW."""
    if expected_sha256 is None:
        raise ValueError("continue_encoder_from_checkpoint_sha256 is required")
    expected = str(expected_sha256)
    if len(expected) != 64 or any(character not in '0123456789abcdefABCDEF' for character in expected):
        raise ValueError("continue_encoder_from_checkpoint_sha256 must be a full SHA-256")
    loaded = _load_encoder_initialization(
        model, checkpoint_path, expected, exact=True, require_encoder_variant=True)
    return {
        'continuation_checkpoint': loaded['initialization_checkpoint'],
        'continuation_checkpoint_sha256': loaded['initialization_checkpoint_sha256'],
        'continuation_loaded_key_count': loaded['initialization_shared_key_count'],
        'continuation_key_migrations': loaded['initialization_key_migrations'],
        'continuation_source_selected_step': loaded['initialization_source_selected_step'],
        'continuation_optimizer_restarted': True,
        'continuation_optimizer_state_loaded': False,
    }


def _autoencoder_objective_multiplier(step, start_step=0, ramp_steps=0):
    """Linear post-start multiplier with exact zero/full endpoints.

    ``step`` is one-based, while ``start_step`` is the number of completed
    steps before the ramp. Omitted/zero controls therefore preserve weight 1.
    """
    step, start, ramp = int(step), int(start_step), int(ramp_steps)
    if step < 1 or start < 0 or ramp < 0:
        raise ValueError("objective step must be positive and start/ramp must be non-negative")
    if ramp == 0:
        return 1.0 if step > start else 0.0
    return max(0.0, min(1.0, (step - start) / ramp))


def _effective_training_weights(cfg, autoencoder, encoder_frozen, multiplier=1.0):
    """Return the actual coefficients used on this optimizer step."""
    multiplier = float(multiplier)
    weights = {
        'teacher': 1.0,
        'reconstruction': float(cfg.get('reconstruction_weight', 0.0)),
        'cycle': float(cfg.get('cycle_weight', 0.0)),
        'distribution_mean': float(cfg.get('distribution_mean_weight', 0.0)),
        'distribution_variance': float(cfg.get('distribution_variance_weight', 0.0)),
        'prior': float(cfg.get('importance_prior_weight', 0.0)),
        'delta': float(cfg.get('delta_loss_weight', 0.0)),
        'sharpness': float(cfg.get('sharpness_loss_weight', 0.0)),
        'slot': float(cfg.get('slot_loss_weight', 0.0)),
        'importance': float(cfg.get('importance_loss_weight', 0.0)),
        'adversarial_merger': -float(cfg.get('adversarial_alignment_weight', 0.0)),
    }
    if not autoencoder:
        return weights
    if encoder_frozen:
        # The decoder warm-start intentionally fits only direct children and
        # predicted importance. Cycle/semantic/teacher/distribution objectives
        # are encoder-facing and remain disabled at this stage.
        for name in ('teacher', 'cycle', 'distribution_mean', 'distribution_variance',
                     'prior', 'delta', 'sharpness', 'slot', 'adversarial_merger'):
            weights[name] = 0.0
    else:
        for name in ('reconstruction', 'cycle', 'prior', 'delta', 'sharpness',
                     'slot', 'importance'):
            weights[name] *= multiplier
    return weights


def step_loss(model, row, autoencoder, device, recon_weight, cycle_weight=0.0,
              distribution_mean_weight=0.0, distribution_variance_weight=0.0,
              distribution_min_samples=2):
    return step_loss_batch(
        model, [row], autoencoder, device, recon_weight, cycle_weight,
        distribution_mean_weight, distribution_variance_weight, distribution_min_samples)


def _target_span_lengths(row):
    """Return target token lengths in serialized target order."""
    if 'target_provenance' in row:
        return [int(item['token_count']) for item in row['target_provenance']]
    nodes = [int(x) for x in row['lengths'].tolist()]
    if 'tree_template' in row:
        plan = _tree_plan_for_row(row)
    elif 'merge_plan' in row:
        plan = [[int(pos)] for pos in row['merge_plan']]
    else:
        count, plan = len(nodes), []
        while count > 1:
            positions = list(range(0, count - 1, 2))
            plan.append(positions)
            count = len(positions) + (count % 2)
    spans = []
    for positions in plan:
        merged_at = []
        for pos in positions:
            if pos < 0 or pos + 1 >= len(nodes):
                raise ValueError(f"invalid target plan position {pos} with {len(nodes)} active nodes")
            merged_at.append(nodes[pos] + nodes[pos + 1])
        spans.extend(merged_at)
        for pos, merged in reversed(list(zip(positions, merged_at))):
            nodes[pos:pos + 2] = [merged]
    return spans


def _row_context(index, row):
    return (f"row_index={index} id={row.get('id')} template={row.get('tree_template', row.get('merge_plan', 'balanced'))} "
            f"leaves={row['leaves'].shape[0]} total_tokens={int(row['lengths'].sum().item())}")


def _assert_finite_batch(indexed_rows):
    """Fail before training on corrupted cache rows; include enough context to fix the cache/config."""
    for index, row in indexed_rows:
        if not torch.isfinite(row['leaves']).all():
            raise FloatingPointError(f"non-finite cached leaves before training: {_row_context(index, row)}")
        if not torch.isfinite(row['lengths']).all() or (row['lengths'] <= 0).any():
            raise FloatingPointError(f"invalid cached lengths before training: {_row_context(index, row)}")
        spans = None
        targets = row.get('targets', [])
        provenance = row.get('target_provenance')
        if provenance is not None and len(provenance) != len(targets):
            raise ValueError(f"target/provenance length mismatch: {_row_context(index, row)}")
        for target_index, target in enumerate(targets):
            if provenance is not None:
                item = provenance[target_index]
                required = {'target_index', 'level', 'leaf_start', 'leaf_end',
                            'token_count', 'teacher_eligible'}
                if not required <= set(item) or int(item['target_index']) != target_index:
                    raise ValueError(f"invalid target provenance at {target_index}: {_row_context(index, row)}")
                safe = int(row.get('teacher_max_tokens', 6144))
                if safe > 6144:
                    raise ValueError("k-ary teacher safety bound may not exceed 6144 tokens")
                eligible = int(item['token_count']) <= safe
                if bool(item['teacher_eligible']) != eligible or (target is not None) != eligible:
                    raise ValueError(f"teacher target/provenance mismatch at {target_index}: {_row_context(index, row)}")
            if target is not None and not torch.isfinite(target).all():
                spans = spans if spans is not None else _target_span_lengths(row)
                span = spans[target_index] if target_index < len(spans) else 'unknown'
                raise FloatingPointError(
                    f"non-finite cached target before training: {_row_context(index, row)} "
                    f"target_index={target_index} target_tokens={span}")


def _validate_cache_arity(rows, arity, split):
    """Prevent binary target order from being silently reused by a K-ary model."""
    arity = int(arity)
    if arity == 2:
        return
    for index, row in enumerate(rows):
        if (row.get('cache_namespace') != 'kary_geodesic_v1' or
                int(row.get('arity', -1)) != arity or
                row.get('reduction') != 'contiguous_balanced'):
            raise ValueError(
                f"{split} row {index} lacks matching kary_geodesic_v1 provenance; "
                "binary caches/target order are not valid for K-ary training")
        if 'target_provenance' not in row:
            raise ValueError(f"{split} row {index} lacks target provenance")


def _assert_finite_cache(rows, split):
    try:
        _assert_finite_batch(list(enumerate(rows)))
    except FloatingPointError as exc:
        raise FloatingPointError(f"{split} cache failed finite check: {exc}") from exc


def _format_batch_context(indexed_rows, limit=8):
    return '; '.join(_row_context(index, row) for index, row in indexed_rows[:limit])


def _first_nonfinite_grad(model):
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            grad = param.grad.detach()
            finite = grad[torch.isfinite(grad)]
            if finite.numel():
                return f"{name} finite_min={float(finite.min())} finite_max={float(finite.max())}"
            return f"{name} all_nonfinite"
    return "unknown"


def _merge_level(model, nodes, sizes):
    """One genuine contiguous K-ary level; carry only a final singleton."""
    # Preserve the exact historical binary operation/reshape path.
    if model.arity == 2:
        pair_count = nodes.shape[1] // 2
        child = nodes[:, :pair_count * 2].reshape(-1, 2, model.d)
        child_sizes = sizes[:, :pair_count * 2].reshape(-1, 2)
        parent = model.merge(child, child_sizes)
        parent_nodes = parent.reshape(nodes.shape[0], pair_count, model.d)
        parent_sizes = child_sizes.sum(-1).reshape(nodes.shape[0], pair_count)
        if nodes.shape[1] % 2:
            parent_nodes = torch.cat([parent_nodes, nodes[:, -1:, :]], dim=1)
            parent_sizes = torch.cat([parent_sizes, sizes[:, -1:]], dim=1)
        return parent_nodes, parent_sizes, parent, child, child_sizes, pair_count

    groups = contiguous_reduction_groups(nodes.shape[1], model.arity)
    merged = [(start, width) for start, width in groups if width >= 2]
    children, masses = [], []
    for start, width in merged:
        children.append(F.pad(nodes[:, start:start + width],
                              (0, 0, 0, model.arity - width)))
        masses.append(F.pad(sizes[:, start:start + width],
                            (0, model.arity - width)))
    child = torch.stack(children, 1).reshape(-1, model.arity, model.d)
    child_sizes = torch.stack(masses, 1).reshape(-1, model.arity)
    parent = model.merge(child, child_sizes)
    merge_count = len(merged)
    parent_nodes = parent.reshape(nodes.shape[0], merge_count, model.d)
    parent_sizes = child_sizes.sum(-1).reshape(nodes.shape[0], merge_count)
    if groups[-1][1] == 1:
        parent_nodes = torch.cat([parent_nodes, nodes[:, -1:, :]], dim=1)
        parent_sizes = torch.cat([parent_sizes, sizes[:, -1:]], dim=1)
    return parent_nodes, parent_sizes, parent, child, child_sizes, merge_count


class _DistributionMomentCollector:
    """Accumulate matched parent/teacher vectors for one optimizer minibatch.

    Matching the mean and diagonal covariance is deliberately parameter-free and
    uses only the teacher targets already available for the cosine objective.
    Full 768x768 covariance matrices would add substantial compute/noise to a
    small minibatch, while their diagonal is a stable marginal-variance check.
    """
    def __init__(self, minimum_samples=2):
        self.minimum_samples = int(minimum_samples)
        self.predictions = []
        self.targets = []

    def add(self, predictions, targets):
        if predictions.numel():
            self.predictions.append(predictions.reshape(-1, predictions.shape[-1]))
            self.targets.append(targets.reshape(-1, targets.shape[-1]))

    def pairs(self):
        """Return all graph-connected generated/teacher pairs, if any."""
        if not self.predictions:
            return None, None
        return torch.cat(self.predictions, dim=0), torch.cat(self.targets, dim=0)

    def losses(self, zero, minimum_samples=None):
        prediction, target = self.pairs()
        minimum_samples = self.minimum_samples if minimum_samples is None else int(minimum_samples)
        if prediction is None or prediction.shape[0] < minimum_samples:
            return zero, zero
        mean = (prediction.mean(0) - target.mean(0)).square().sum()
        prediction_var = (prediction - prediction.mean(0)).square().mean(0)
        target_var = (target - target.mean(0)).square().mean(0)
        variance = (prediction_var - target_var).square().sum()
        return mean, variance


class _ParentLogisticDiscriminator(torch.nn.Module):
    """Train-only logistic discriminator matching the validation probe family."""
    def __init__(self, dimension):
        super().__init__()
        self.logit = torch.nn.Linear(int(dimension), 1)
        # The first discriminator update, not an arbitrary random direction,
        # should determine the initial adversarial gradient.
        torch.nn.init.zeros_(self.logit.weight)
        torch.nn.init.zeros_(self.logit.bias)

    def forward(self, embeddings):
        return self.logit(embeddings.float()).squeeze(-1)


def _autoencoder_losses(model, parent, child, child_sizes, cycle_weight,
                        detach_parent=False):
    """Content reconstruction without a length side channel in new modes.

    ``oracle`` is intentionally the byte-for-byte historical objective. Equal
    and predicted modes compare cycles at the *same counterfactual masses* on
    both sides; the equal target is not the original length-weighted parent.
    """
    mode = getattr(model, "decoder_length_mode", "oracle")
    # During staged decoder-only fitting, the parent is a fixed feature.  This
    # is explicit even though frozen encoder parameters would normally make it
    # graph-free, and protects the boundary if a caller supplies a grad-bearing
    # parent directly.
    decoder_parent = parent.detach() if detach_parent else parent
    if getattr(model, "decoder_mode", "legacy") == "strict":
        genuine = (child_sizes > 0).all(1)
        if not genuine.any():
            zero = parent.sum() * 0.0
            return zero, zero
        p, real, sizes = decoder_parent[genuine], child[genuine], child_sizes[genuine]
        decoded, decode_aux = model.decode(p, return_aux=True)
        reconstruction = 1 - F.cosine_similarity(decoded, real.detach(), dim=-1).mean()
        q = sizes.to(p) / sizes.sum(1, keepdim=True).to(p)
        encoder_aux = model._last_merge_aux
        effective = encoder_aux["effective_weights"][genuine] if model.importance_mode == "semantic" else None
        predicted = decode_aux["importance_weights"]
        model.record_importance_target(predicted, q, effective)
        cycle = reconstruction * 0.0
        if cycle_weight:
            recomposed = model.compose_anchor(decoded, predicted)
            cycle = 1 - F.cosine_similarity(recomposed, p.detach(), dim=-1).mean()
        return reconstruction, cycle
    decoded = model.decode(decoder_parent, child_sizes)
    direct = 1 - F.cosine_similarity(decoded, child.detach(), dim=-1).mean()
    reconstruction = direct

    relational_weight = float(getattr(model, "relational_loss_weight", 0.0))
    if relational_weight:
        real = child.detach()
        target_pair = F.cosine_similarity(real[:, 0], real[:, 1], dim=-1)
        decoded_pair = F.cosine_similarity(decoded[:, 0], decoded[:, 1], dim=-1)
        target_parent = F.cosine_similarity(real, parent.detach()[:, None, :], dim=-1)
        decoded_parent = F.cosine_similarity(decoded, parent.detach()[:, None, :], dim=-1)
        relational = F.smooth_l1_loss(decoded_pair, target_pair) + F.smooth_l1_loss(decoded_parent, target_parent)
        reconstruction = reconstruction + relational_weight * relational

    ratio_weight = float(getattr(model, "ratio_loss_weight", 0.0))
    if mode == "predicted" and ratio_weight:
        predicted = model.predict_child_masses(decoder_parent)
        bound = float(model.ratio_log_bound)
        predicted_log_ratio = (predicted[:, 0].log() - predicted[:, 1].log()).clamp(-bound, bound)
        true = child_sizes.to(parent).clamp_min(1e-8)
        target_log_ratio = (true[:, 0].log() - true[:, 1].log()).clamp(-bound, bound)
        # Smooth-L1 over bounded log-ratios cannot be dominated by extreme spans.
        reconstruction = reconstruction + ratio_weight * F.smooth_l1_loss(predicted_log_ratio, target_log_ratio)

    cycle = reconstruction * 0.0
    if cycle_weight:
        if mode == "oracle":
            cycle_lengths = child_sizes
            target = parent.detach()
        else:
            # decoder_masses ignores child_sizes in predicted mode. Detach the
            # counterfactual real-child merge so it is a fixed target.
            cycle_lengths = model.decoder_masses(decoder_parent, child_sizes)
            target = model.merge(child.detach(), cycle_lengths).detach()
        z_tilde = model.merge(decoded, cycle_lengths)
        cycle = 1 - F.cosine_similarity(z_tilde, target, dim=-1).mean()
    for name, value in (("reconstruction", reconstruction), ("cycle", cycle)):
        if not torch.isfinite(value):
            raise FloatingPointError(f"non-finite {name} loss in encode-decode cycle")
    return reconstruction, cycle


def _step_loss_plan(model, row, autoencoder, device, recon_weight, cycle_weight=0.0,
                    collector=None, teacher_weight=1.0, detach_decoder_parent=False):
    """Train one arbitrary valid adjacent-merge tree from a prepared plan."""
    nodes = [x.to(device) for x in row['leaves']]
    sizes = [x.to(device) for x in row['lengths']]
    teacher_losses = []
    recon_losses = []
    cycle_losses = []
    targets = row['targets']
    for step, pos in enumerate(row['merge_plan']):
        pos = int(pos)
        if pos < 0 or pos + 1 >= len(nodes):
            raise ValueError(f"invalid merge_plan position {pos} with {len(nodes)} active nodes")
        child = torch.stack([nodes[pos], nodes[pos + 1]], dim=0).unsqueeze(0)
        child_sizes = torch.stack([sizes[pos], sizes[pos + 1]], dim=0).unsqueeze(0)
        parent = model.merge(child, child_sizes)
        target = targets[step]
        if target is not None:
            target = target.to(device)
            teacher_losses.append((1 - F.cosine_similarity(parent[0], target, dim=-1)).mean())
            if collector is not None:
                collector.add(parent, target.unsqueeze(0))
        if autoencoder:
            reconstruction, cycle = _autoencoder_losses(
                model, parent, child, child_sizes, cycle_weight, detach_decoder_parent)
            recon_losses.append(reconstruction)
            cycle_losses.append(cycle)
        nodes[pos:pos + 2] = [parent[0]]
        sizes[pos:pos + 2] = [child_sizes.sum()]
    zero = row['leaves'].to(device).sum() * 0.0
    teacher = torch.stack(teacher_losses).mean() if teacher_losses else zero
    recon = torch.stack(recon_losses).mean() if recon_losses else zero
    cycle = torch.stack(cycle_losses).mean() if cycle_losses else zero
    return teacher_weight * teacher + (recon_weight * recon if autoencoder else 0.0) + (cycle_weight * cycle if autoencoder else 0.0), {'teacher': teacher, 'reconstruction': recon, 'cycle': cycle}


def _step_loss_template_batch(model, rows, autoencoder, device, recon_weight, cycle_weight=0.0,
                              collector=None, teacher_weight=1.0, detach_decoder_parent=False):
    """GPU-batch rows sharing a deterministic contiguous tree template."""
    leaves = torch.stack([r['leaves'] for r in rows]).to(device)
    sizes = torch.stack([r['lengths'] for r in rows]).to(device)
    plan = _tree_plan_for_row(rows[0])
    nodes, teacher_levels, recon_levels, cycle_levels, offset = leaves, [], [], [], 0
    for positions in plan:
        indices = torch.tensor(positions, device=device)
        child = torch.stack((nodes[:, indices], nodes[:, indices + 1]), dim=2).reshape(-1, 2, model.d)
        child_sizes = torch.stack((sizes[:, indices], sizes[:, indices + 1]), dim=2).reshape(-1, 2)
        parent = model.merge(child, child_sizes).reshape(len(rows), len(positions), model.d)
        targets = [target for row in rows for target in row['targets'][offset:offset + len(positions)]]
        mask_list = [target is not None for target in targets]
        if any(mask_list):
            target = torch.stack([target if target is not None else torch.zeros(model.d) for target in targets]).to(device)
            mask = torch.tensor(mask_list, dtype=torch.bool, device=device)
            matched_parent = parent.reshape(-1, model.d)[mask]
            matched_target = target[mask]
            teacher_levels.append((1 - F.cosine_similarity(matched_parent, matched_target, dim=-1)).mean())
            if collector is not None:
                collector.add(matched_parent, matched_target)
        if autoencoder:
            reconstruction, cycle = _autoencoder_losses(
                model, parent.reshape(-1, model.d), child, child_sizes, cycle_weight,
                detach_decoder_parent)
            recon_levels.append(reconstruction)
            cycle_levels.append(cycle)
        # Positions are disjoint and sorted. Replace each pair while carrying
        # untouched nodes, preserving the contiguous active-node invariant.
        next_nodes, next_sizes, cursor = [], [], 0
        for pair, pos in enumerate(positions):
            if cursor < pos:
                next_nodes.append(nodes[:, cursor:pos]); next_sizes.append(sizes[:, cursor:pos])
            next_nodes.append(parent[:, pair:pair + 1]); next_sizes.append(child_sizes.reshape(len(rows), len(positions), 2)[:, pair].sum(-1, keepdim=True))
            cursor = pos + 2
        if cursor < nodes.shape[1]:
            next_nodes.append(nodes[:, cursor:]); next_sizes.append(sizes[:, cursor:])
        nodes, sizes = torch.cat(next_nodes, 1), torch.cat(next_sizes, 1)
        offset += len(positions)
    teacher = torch.stack(teacher_levels).mean() if teacher_levels else leaves.sum() * 0.0
    recon = torch.stack(recon_levels).mean() if recon_levels else leaves.sum() * 0.0
    cycle = torch.stack(cycle_levels).mean() if cycle_levels else leaves.sum() * 0.0
    return teacher_weight * teacher + (recon_weight * recon if autoencoder else 0.0) + (cycle_weight * cycle if autoencoder else 0.0), {'teacher': teacher, 'reconstruction': recon, 'cycle': cycle}


def _tree_plan_for_row(row):
    # Kept here (rather than serialized per row) so caches stay compact.
    count, template = row['leaves'].shape[0], row['tree_template']
    if template == 'balanced':
        plan = []
        while count > 1:
            positions = list(range(0, count - 1, 2)); plan.append(positions)
            count = len(positions) + (count % 2)
        return plan
    plan = []
    while count > 1:
        if template == 'left': pos = 0
        elif template == 'right': pos = count - 2
        elif template == 'outside_in': pos = 0 if len(plan) % 2 == 0 else count - 2
        else: raise ValueError(f"unknown tree template: {template}")
        plan.append([pos]); count -= 1
    return plan


def _step_loss_batch_base(model, rows, autoencoder, device, recon_weight, cycle_weight=0.0,
                          collector=None, teacher_weight=1.0, detach_decoder_parent=False):
    """Batch the base supervised losses and collect matched vectors when requested."""
    templated = [r for r in rows if 'tree_template' in r]
    if templated:
        groups = {}
        for row in rows:
            key = (row['leaves'].shape[0], row.get('tree_template'))
            groups.setdefault(key, []).append(row)
        losses, teachers, recons, cycles = [], [], [], []
        for group in groups.values():
            if 'tree_template' in group[0]:
                loss, parts = _step_loss_template_batch(
                    model, group, autoencoder, device, recon_weight, cycle_weight, collector,
                    teacher_weight, detach_decoder_parent)
            else:
                loss, parts = _step_loss_batch_base(
                    model, group, autoencoder, device, recon_weight, cycle_weight, collector,
                    teacher_weight, detach_decoder_parent)
            losses.append(loss.repeat(len(group))); teachers.append(parts['teacher'].detach().repeat(len(group)))
            if autoencoder:
                recons.append(parts['reconstruction'].detach().repeat(len(group)))
                cycles.append(parts['cycle'].detach().repeat(len(group)))
        output = {'teacher': torch.cat(teachers).mean()}
        if autoencoder:
            output['reconstruction'] = torch.cat(recons).mean()
            output['cycle'] = torch.cat(cycles).mean()
        return torch.cat(losses).mean(), output
    planned = [r for r in rows if 'merge_plan' in r]
    if planned:
        losses = []
        teachers = []
        recons = []
        cycles = []
        for row in rows:
            if 'merge_plan' in row:
                loss, parts = _step_loss_plan(
                    model, row, autoencoder, device, recon_weight, cycle_weight, collector,
                    teacher_weight, detach_decoder_parent)
            else:
                loss, parts = _step_loss_batch_base(
                    model, [row], autoencoder, device, recon_weight, cycle_weight, collector,
                    teacher_weight, detach_decoder_parent)
            losses.append(loss)
            teachers.append(parts['teacher'].detach())
            if autoencoder:
                recons.append(parts['reconstruction'].detach())
                cycles.append(parts['cycle'].detach())
        parts = {'teacher': torch.stack(teachers).mean()}
        if autoencoder:
            parts['reconstruction'] = torch.stack(recons).mean()
            parts['cycle'] = torch.stack(cycles).mean()
        return torch.stack(losses).mean(), parts

    groups = {}
    for row in rows:
        groups.setdefault(row['leaves'].shape[0], []).append(row)
    group_losses = []
    group_parts = []
    for leaves_count, group in groups.items():
        leaves = torch.stack([r['leaves'] for r in group]).to(device)
        lengths = torch.stack([r['lengths'] for r in group]).to(device)
        nodes, sizes = leaves, lengths
        teacher_levels = []
        recon_levels = []
        cycle_levels = []
        offset = 0
        while nodes.shape[1] > 1:
            next_nodes, next_sizes, parent, child, child_sizes, pairs_per_doc = _merge_level(model, nodes, sizes)
            flat_targets = []
            target_mask = []
            for r in group:
                level_targets = r['targets'][offset:offset + pairs_per_doc]
                flat_targets.extend(level_targets)
                target_mask.extend(t is not None for t in level_targets)
            if any(target_mask):
                target = torch.stack([
                    t if t is not None else torch.zeros(model.d) for t in flat_targets
                ]).to(device)
                mask = torch.tensor(target_mask, dtype=torch.bool, device=device)
                matched_parent, matched_target = parent[mask], target[mask]
                teacher_loss = 1 - F.cosine_similarity(matched_parent, matched_target, dim=-1)
                teacher_levels.append(teacher_loss.mean())
                if collector is not None:
                    collector.add(matched_parent, matched_target)
            if autoencoder:
                reconstruction, cycle = _autoencoder_losses(
                    model, parent, child, child_sizes, cycle_weight, detach_decoder_parent)
                recon_levels.append(reconstruction)
                cycle_levels.append(cycle)
            offset += pairs_per_doc
            nodes, sizes = next_nodes, next_sizes
        teacher = torch.stack(teacher_levels).mean() if teacher_levels else leaves.sum() * 0.0
        recon = torch.stack(recon_levels).mean() if recon_levels else leaves.sum() * 0.0
        cycle = torch.stack(cycle_levels).mean() if cycle_levels else leaves.sum() * 0.0
        total = teacher_weight * teacher + (recon_weight * recon if autoencoder else 0.0) + (cycle_weight * cycle if autoencoder else 0.0)
        group_losses.append(total.repeat(len(group)))
        group_parts.append((teacher.detach().repeat(len(group)), recon.detach().repeat(len(group)), cycle.detach().repeat(len(group))))
    total = torch.cat(group_losses).mean()
    teacher = torch.cat([x[0] for x in group_parts]).mean()
    parts = {'teacher': teacher}
    if autoencoder:
        recon = torch.cat([x[1] for x in group_parts]).mean()
        cycle = torch.cat([x[2] for x in group_parts]).mean()
        parts['reconstruction'] = recon
        parts['cycle'] = cycle
    return total, parts


def step_loss_batch(model, rows, autoencoder, device, recon_weight, cycle_weight=0.0,
                    distribution_mean_weight=0.0, distribution_variance_weight=0.0,
                    distribution_min_samples=2, collector=None, teacher_weight=1.0,
                    auxiliary_weight_multiplier=1.0, importance_weight_multiplier=None,
                    detach_decoder_parent=False):
    """Add minibatch teacher-distribution matching to the existing base loss.

    The collected vectors span every teacher-supervised merge in the sampled
    rows, so variable leaf counts and tree templates do not fragment the
    moment estimate into one-row groups.  ``collector`` lets the train-only
    adversarial screen reuse these exact pairs without a second merger pass.
    """
    weights = (float(distribution_mean_weight), float(distribution_variance_weight))
    teacher_weight = float(teacher_weight)
    auxiliary_weight_multiplier = float(auxiliary_weight_multiplier)
    importance_weight_multiplier = (auxiliary_weight_multiplier if importance_weight_multiplier is None
                                    else float(importance_weight_multiplier))
    all_multipliers = weights + (teacher_weight, auxiliary_weight_multiplier,
                                 importance_weight_multiplier)
    if not all(math.isfinite(weight) and weight >= 0 for weight in all_multipliers):
        raise ValueError(f"objective weights/multipliers must be finite and non-negative, got {all_multipliers}")
    collector = collector or _DistributionMomentCollector(distribution_min_samples)
    # Geodesic angle telemetry is collected for every parameterization. The
    # same collection window also carries opt-in semantic/strict losses.
    model.begin_aux_collection()
    total, parts = _step_loss_batch_base(
        model, rows, autoencoder, device, recon_weight, cycle_weight, collector,
        teacher_weight, detach_decoder_parent)
    mean, variance = collector.losses(total * 0.0, distribution_min_samples)
    total = total + weights[0] * mean + weights[1] * variance
    parts.update({'distribution_mean': mean.detach(), 'distribution_variance': variance.detach()})
    auxiliary, auxiliary_parts = model.end_aux_collection(
        total * 0.0, auxiliary_weight_multiplier, importance_weight_multiplier)
    total = total + auxiliary
    parts.update({key: value.detach() for key, value in auxiliary_parts.items()})
    return total, parts


def _adversarial_discriminator_loss(discriminator, prediction, target):
    """BCE for the logistic probe: teacher parents=1, generated parents=0."""
    fake_logits = discriminator(prediction.detach())
    real_logits = discriminator(target.detach())
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    loss = 0.5 * (fake_loss + real_loss)
    accuracy = 0.5 * ((fake_logits < 0).float().mean() + (real_logits >= 0).float().mean())
    return loss, accuracy


def _update_adversarial_discriminator(discriminator, optimizer, prediction, target, updates):
    """Update only the probe; detached inputs prevent merger gradients here."""
    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)
    losses, accuracies = [], []
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        loss, accuracy = _adversarial_discriminator_loss(discriminator, prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite adversarial discriminator loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(loss.detach())
        accuracies.append(accuracy.detach())
    optimizer.zero_grad(set_to_none=True)
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    return torch.stack(losses).mean(), torch.stack(accuracies).mean()


def _adversarial_merger_loss(discriminator, prediction):
    """Return the fake-class BCE whose negative is a gradient-reversal loss."""
    logits = discriminator(prediction)
    return F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))


@torch.no_grad()
def validation_loss(model, rows, variant, device, cfg):
    vals=[]
    sample = rows[:min(len(rows), 128)]
    for i in range(0, len(sample), int(cfg.get('batch_size', 32))):
        batch = sample[i:i + int(cfg.get('batch_size', 32))]
        _assert_finite_batch(list(enumerate(batch, start=i)))
        # Cycle consistency is deliberately training-only: validation keeps the
        # original teacher-plus-direct-reconstruction selection criterion.
        loss, _ = step_loss_batch(
            model, batch, variant == 'autoencoder', device, cfg['reconstruction_weight'], 0.0,
            float(cfg.get('distribution_mean_weight', 0.0)),
            float(cfg.get('distribution_variance_weight', 0.0)),
            int(cfg.get('distribution_min_samples', 2)))
        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite validation loss near validation row {i}')
        vals.append(float(loss))
    return sum(vals) / len(vals)


@torch.no_grad()
def validation_autoencoder_diagnostics(model, rows, device, cfg):
    """Validation-role-only raw AE diagnostics, separate from selection loss."""
    sample = rows[:min(len(rows), 128)]
    values = {name: [] for name in ('reconstruction', 'cycle', 'importance')}
    for i in range(0, len(sample), int(cfg.get('batch_size', 32))):
        batch = sample[i:i + int(cfg.get('batch_size', 32))]
        _assert_finite_batch(list(enumerate(batch, start=i)))
        _, parts = step_loss_batch(
            model, batch, True, device, float(cfg.get('reconstruction_weight', 0.0)),
            float(cfg.get('cycle_weight', 0.0)), 0.0, 0.0,
            int(cfg.get('distribution_min_samples', 2)))
        for name in values:
            if name in parts:
                values[name].append(float(parts[name]))
    return {name: float(np.mean(parts)) for name, parts in values.items() if parts}


def _root_target(row):
    targets = row.get('targets', [])
    provenance = row.get('target_provenance')
    if provenance is None:
        return next((target for target in reversed(targets) if target is not None), None)
    leaf_count = int(row['leaves'].shape[0])
    matches = [targets[i] for i, item in enumerate(provenance)
               if int(item['leaf_start']) == 0 and int(item['leaf_end']) == leaf_count]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one provenance-backed root target for row {row.get('id')}")
    return matches[0]


@torch.no_grad()
def batched_balanced_roots(model, rows, device, batch_size):
    """Return roots/targets grouped by leaf count, never one GPU merge per doc."""
    groups = {}
    for row in rows:
        target = _root_target(row)
        if target is not None and 'tree_template' not in row and 'merge_plan' not in row:
            groups.setdefault(row['leaves'].shape[0], []).append((row, target))
    fake, real, ids = [], [], []
    for group in groups.values():
        for start in range(0, len(group), batch_size):
            part = group[start:start + batch_size]
            nodes = torch.stack([row['leaves'] for row, _ in part]).to(device)
            sizes = torch.stack([row['lengths'] for row, _ in part]).to(device)
            while nodes.shape[1] > 1:
                nodes, sizes, _, _, _, _ = _merge_level(model, nodes, sizes)
            fake.extend(nodes[:, 0].cpu().numpy())
            real.extend(target.numpy() for _, target in part)
            ids.extend(row['id'] for row, _ in part)
    return fake, real, ids


@torch.no_grad()
def validation_metrics(model, rows, device, cfg):
    sample = rows[:int(cfg.get('validation_metric_documents', len(rows)))]
    fake, real, ids = batched_balanced_roots(model, sample, device, int(cfg.get('validation_root_batch_size', cfg.get('batch_size', 32))))
    if len(real) < 4: return {}
    real, fake, ids = np.asarray(real), np.asarray(fake), np.asarray(ids)
    x=np.vstack((real,fake)); y=np.r_[np.ones(len(real)),np.zeros(len(fake))]; groups=np.r_[ids,ids]
    tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=int(cfg['seed'])).split(x,y,groups))
    clf=LogisticRegression(max_iter=1000,solver='lbfgs').fit(x[tr],y[tr]); p=clf.predict_proba(x[te])[:,1]
    result = {'validation_auc':float(roc_auc_score(y[te],p)),
              'validation_pr_auc':float(average_precision_score(y[te],p)),
              'validation_balanced_accuracy':float(balanced_accuracy_score(y[te],p>=.5)),
              'validation_teacher_cosine':float(np.mean(np.sum(fake*real,axis=1)))}
    if bool(cfg.get('validation_attacks', False)):
        # Validation-only reuse of the predeclared grouped logistic and RBF
        # implementations. No evaluator path or role artifact is opened.
        from .evaluate import evaluate_attacks
        attacks = evaluate_attacks(real, fake, ids, bootstrap=0)
        result.update({
            'validation_logistic_auc': attacks['attacks']['standard_scaled_logistic']['auc'],
            'validation_rbf_auc': attacks['attacks']['standard_scaled_rbf_svm']['auc'],
            'validation_max_attack_auc': attacks['max_attack_auc'],
        })
    return result


def _amp_dtype(cfg, device):
    mp = str(cfg.get('mixed_precision', 'none')).lower()
    if device.type != 'cuda' or mp in ('', 'none', 'false'):
        return None
    if mp == 'bf16' and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    # Avoid fp16 by default: cosine-heavy losses can underflow more easily.
    return None


def _load_training_roles(cfg):
    """Load corpus-separated roles without any protected-data fallback.

    ``training_without_validation`` is an opt-in for protocols that must freeze
    a fixed final checkpoint before a separately prepared development cache is
    opened. It never substitutes TRAIN rows for validation rows.
    """
    corpora = cfg.get('corpora')
    train_only = bool(cfg.get('training_without_validation', False))
    if not corpora:
        rows = torch.load(Path(cfg['cache_dir']) / 'train.pt', weights_only=False)
        val_path = Path(cfg['cache_dir']) / 'validation.pt'
        if train_only and val_path.exists():
            raise ValueError('train-only protocol refuses a validation artifact in its cache')
        return rows, ({} if train_only else ({'all': torch.load(val_path, weights_only=False)} if val_path.exists() else {})), None
    weights, train_by, validation_by = {}, {}, {}
    for corpus in corpora:
        corpus_id = str(corpus['id'])
        weights[corpus_id] = float(corpus['weight'])
        root = Path(cfg['cache_dir']) / corpus_id
        train_by[corpus_id] = torch.load(root / 'train.pt', weights_only=False)
        validation_path = root / 'validation.pt'
        if train_only:
            if validation_path.exists():
                raise ValueError(f'train-only protocol refuses validation artifact: {validation_path}')
        else:
            validation_by[corpus_id] = torch.load(validation_path, weights_only=False)
    if any(not rows for rows in train_by.values()) or (not train_only and any(not rows for rows in validation_by.values())):
        raise RuntimeError('every declared corpus role must be non-empty')
    # A flat copy is useful for dimensions and finite guards; sampling below is
    # explicitly corpus-weighted and does not depend on corpus cache sizes.
    return [row for rows in train_by.values() for row in rows], validation_by, (train_by, weights)


def _stratified_corpus_counts(weights, batch_size):
    """Largest-remainder allocation: deterministic and within one row of the target."""
    names = list(weights)
    values = [float(weights[name]) for name in names]
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not values or any(not math.isfinite(value) or value < 0 for value in values) or sum(values) <= 0:
        raise ValueError("corpus weights must be finite, non-negative, and have positive sum")
    quotas = [batch_size * value / sum(values) for value in values]
    counts = [math.floor(quota) for quota in quotas]
    # Stable corpus order breaks equal-remainder ties, making the allocation
    # independent of hash randomization while the eventual batch is shuffled.
    order = sorted(range(len(names)), key=lambda i: (-(quotas[i] - counts[i]), i))
    for index in order[:batch_size - sum(counts)]:
        counts[index] += 1
    return dict(zip(names, counts))


def _sample_training_batch(indexed_rows, multicorpus, batch_size, stratified=False):
    if multicorpus is None:
        return random.sample(indexed_rows, min(batch_size, len(indexed_rows)))
    train_by, weights = multicorpus
    names = list(weights)
    if not stratified:
        # Do not refactor these two calls: this is the exact legacy RNG path.
        picked = random.choices(names, weights=[weights[name] for name in names], k=batch_size)
        return [(name, random.choice(train_by[name])) for name in picked]
    counts = _stratified_corpus_counts(weights, batch_size)
    picked = [name for name in names for _ in range(counts[name])]
    random.shuffle(picked)
    return [(name, random.choice(train_by[name])) for name in picked]


def _full_arity_decoder_stage_rows(indexed_rows, multicorpus, arity):
    """Filter only the frozen strict-decoder stage to rows with a real K-way merge."""
    arity = int(arity)
    eligible = lambda row: int(row['leaves'].shape[0]) >= arity
    if multicorpus is None:
        selected = [(index, row) for index, row in indexed_rows if eligible(row)]
        if not selected:
            raise ValueError("strict decoder-only stage has no full-arity TRAIN rows")
        return selected, None
    train_by, weights = multicorpus
    selected_by = {name: [row for row in rows if eligible(row)]
                   for name, rows in train_by.items()}
    missing = [name for name, rows in selected_by.items() if not rows]
    if missing:
        raise ValueError(f"strict decoder-only stage has no full-arity rows for: {missing}")
    # Retain the original corpus order and weights. Only within-corpus row
    # eligibility changes, so stratified equal-corpus batches remain exact.
    return indexed_rows, (selected_by, weights)


def _learning_rate_at_step(base_learning_rate, min_learning_rate, total_steps, warmup_steps, step):
    """LR for a zero-based optimizer step; warmup reaches base and cosine reaches min."""
    base, minimum = float(base_learning_rate), float(min_learning_rate)
    total, warmup, step = int(total_steps), int(warmup_steps), int(step)
    if total < 1 or warmup < 0 or warmup >= total:
        raise ValueError("total_steps must be positive and 0 <= warmup_steps < total_steps")
    if not all(math.isfinite(value) and value >= 0 for value in (base, minimum)) or minimum > base:
        raise ValueError("learning rates must be finite with 0 <= min_learning_rate <= learning_rate")
    if step < 0 or step >= total:
        raise ValueError("step is outside the configured training range")
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    decay_steps = total - warmup
    if decay_steps <= 1:
        return minimum
    progress = (step - warmup) / (decay_steps - 1)
    return minimum + 0.5 * (base - minimum) * (1.0 + math.cos(math.pi * progress))


def _trailing_average(history, value, window):
    if int(window) < 1:
        raise ValueError("checkpoint_moving_average_window must be at least 1")
    history.append(float(value))
    return float(np.mean(history[-int(window):]))


def _teacher_cosine_guard(record, validation_by, macro_minimum=None, per_corpus_minimum=None):
    macro_ok = (macro_minimum is None or
                record.get('macro_validation_teacher_cosine', record.get('validation_teacher_cosine', float('-inf')))
                >= float(macro_minimum))
    per_corpus_values = [record.get(f'validation/{name}/teacher_cosine', float('-inf'))
                         for name in validation_by if name != 'all']
    per_corpus_ok = (per_corpus_minimum is None or
                     (bool(per_corpus_values) and min(per_corpus_values) >= float(per_corpus_minimum)))
    return macro_ok and per_corpus_ok, macro_ok, per_corpus_ok


def _quality_rejection_reason(record, saturation_streak, saturation_patience,
                              reject_saturation=False, reject_cosine=False):
    """Return a fail-closed early-stop reason while leaving the best checkpoint intact."""
    if reject_saturation and saturation_streak >= saturation_patience:
        return (f"geodesic_saturation_nonzero_for_{saturation_streak}_steps:"
                f"{float(record.get('geodesic_saturation_fraction', 0.0)):.9g}")
    if (reject_cosine and record.get('checkpoint_selection_eligible') is False and
            record.get('checkpoint_selection_step_guard_passed', True)):
        failed = []
        if not record.get('macro_teacher_cosine_guard_passed', False):
            failed.append('macro')
        if not record.get('per_corpus_teacher_cosine_guard_passed', False):
            failed.append('per_corpus')
        return f"teacher_cosine_guard_failed:{'+'.join(failed) or 'unknown'}"
    return None


def _weighted_loss_contributions(parts, cfg, adversarial_alignment_weight=0.0,
                                 effective_weights=None):
    weights = (dict(effective_weights) if effective_weights is not None else
               _effective_training_weights(cfg, True, False, 1.0))
    # Keep the explicit argument backward-compatible for callers that do not
    # supply staged effective weights.
    if effective_weights is None:
        weights['adversarial_merger'] = -float(adversarial_alignment_weight)
    return {name: float(value) * weights[name] for name, value in parts.items() if name in weights}


def _telemetry_record(parts, cfg, indexed_batch, multicorpus, grad_norm, learning_rate,
                      adversarial_alignment_weight=0.0, effective_weights=None):
    contributions = _weighted_loss_contributions(
        parts, cfg, adversarial_alignment_weight, effective_weights)
    counts = ({name: sum(corpus == name for corpus, _ in indexed_batch)
               for name in multicorpus[1]} if multicorpus is not None else {})
    fields = {'grad_norm': float(grad_norm), 'current_learning_rate': float(learning_rate)}
    if effective_weights is not None:
        fields.update({f'effective_weight/{name}': float(value)
                       for name, value in effective_weights.items()})
    fields.update({f'weighted_contribution/{name}': value for name, value in contributions.items()})
    fields.update({f'corpus_batch_count/{name}': count for name, count in counts.items()})
    fields.update({f'corpus_batch_fraction/{name}': count / len(indexed_batch)
                   for name, count in counts.items()})
    return fields, contributions, counts


def _multicorpus_validation(model, validation_by, variant, device, cfg):
    record, losses, metric_values = {}, [], {}
    diagnostic_values = {}
    for corpus_id, rows in validation_by.items():
        loss = validation_loss(model, rows, variant, device, cfg)
        metrics = validation_metrics(model, rows, device, cfg)
        diagnostics = (validation_autoencoder_diagnostics(model, rows, device, cfg)
                       if variant == 'autoencoder' and
                       bool(cfg.get('validation_autoencoder_diagnostics', False)) else {})
        record[f'validation/{corpus_id}/loss'] = loss
        losses.append(loss)
        for key, value in metrics.items():
            short = key.removeprefix('validation_')
            record[f'validation/{corpus_id}/{short}'] = value
            metric_values.setdefault(short, []).append(value)
        for name, value in diagnostics.items():
            record[f'validation/{corpus_id}/autoencoder_{name}'] = value
            diagnostic_values.setdefault(name, []).append(value)
    record['macro_validation_loss'] = float(np.mean(losses))
    for short, values in metric_values.items():
        # Macro means give each corpus equal selection weight, independent of
        # both requested training mixture and validation cache size.
        if len(values) == len(validation_by):
            record[f'macro_validation_{short}'] = float(np.mean(values))
    for name, values in diagnostic_values.items():
        if len(values) == len(validation_by):
            record[f'macro_validation_autoencoder_{name}'] = float(np.mean(values))
    return record


def _make_writer(cfg, variant, seed):
    if not bool(cfg.get('tensorboard', cfg.get('enable_tensorboard', False))):
        return None
    if SummaryWriter is None:
        print('TensorBoard logging requested but tensorboard is not installed; continuing without it.', flush=True)
        return None
    log_root = Path(cfg.get('tensorboard_dir', Path(cfg['output_dir']) / 'tensorboard'))
    return SummaryWriter(log_dir=str(log_root / f'{variant}_seed{seed}'))


def run(cfg, variant, seed, device):
    random.seed(seed); torch.manual_seed(seed)
    rows, validation_by, multicorpus = _load_training_roles(cfg)
    val_rows = validation_by.get('all', []) if set(validation_by) == {'all'} else []
    cycle_weight = float(cfg.get('cycle_weight', 0.0))
    distribution_mean_weight = float(cfg.get('distribution_mean_weight', 0.0))
    distribution_variance_weight = float(cfg.get('distribution_variance_weight', 0.0))
    distribution_min_samples = int(cfg.get('distribution_min_samples', 2))
    adversarial_alignment_weight = float(cfg.get('adversarial_alignment_weight', 0.0))
    adversarial_discriminator_learning_rate = float(cfg.get('adversarial_discriminator_learning_rate', 0.0))
    adversarial_discriminator_weight_decay = float(cfg.get('adversarial_discriminator_weight_decay', 0.0))
    adversarial_discriminator_updates = int(cfg.get('adversarial_discriminator_updates', 1))
    adversarial_min_samples = int(cfg.get('adversarial_min_samples', 2))
    freeze_encoder_steps = int(cfg.get('freeze_encoder_steps', 0))
    objective_start_step = int(cfg.get('autoencoder_objective_start_step', 0))
    objective_ramp_steps = int(cfg.get('autoencoder_objective_ramp_steps', 0))
    encoder_learning_rate_scale = float(cfg.get('encoder_learning_rate_scale', 1.0))
    decoder_stage_full_arity = bool(cfg.get('decoder_only_require_full_arity_rows', False))
    initialization_path = cfg.get('initialize_from_encoder_checkpoint',
                                  cfg.get('encoder_initialization_checkpoint'))
    initialization_expected_hash = cfg.get('initialize_from_encoder_checkpoint_sha256',
                                            cfg.get('encoder_initialization_checkpoint_sha256'))
    allow_autoencoder_initialization = bool(cfg.get('initialize_encoder_from_autoencoder_checkpoint', False))
    continuation_path = cfg.get('continue_encoder_from_checkpoint')
    continuation_expected_hash = cfg.get('continue_encoder_from_checkpoint_sha256')
    if freeze_encoder_steps < 0 or objective_start_step < 0 or objective_ramp_steps < 0:
        raise ValueError("freeze/start/ramp steps must be non-negative")
    if not math.isfinite(encoder_learning_rate_scale) or encoder_learning_rate_scale <= 0:
        raise ValueError("encoder_learning_rate_scale must be finite and positive")
    if variant != 'autoencoder' and (freeze_encoder_steps or initialization_path or
                                     objective_start_step or objective_ramp_steps):
        raise ValueError("staged autoencoder controls require variant='autoencoder'")
    if continuation_path and variant != 'encoder':
        raise ValueError("continue_encoder_from_checkpoint requires variant='encoder'")
    if continuation_expected_hash and not continuation_path:
        raise ValueError("continue_encoder_from_checkpoint_sha256 requires its checkpoint")
    if continuation_path and initialization_path:
        raise ValueError("encoder initialization and encoder continuation are mutually exclusive")
    if allow_autoencoder_initialization and (variant != 'autoencoder' or not initialization_path):
        raise ValueError("autoencoder-source encoder initialization requires an autoencoder run and checkpoint")
    if objective_ramp_steps and objective_start_step < freeze_encoder_steps:
        raise ValueError("autoencoder_objective_start_step must be at or after encoder unfreeze")
    if not math.isfinite(cycle_weight) or cycle_weight < 0:
        raise ValueError(f"cycle_weight must be a finite non-negative value, got {cycle_weight}")
    if (not math.isfinite(distribution_mean_weight) or distribution_mean_weight < 0 or
            not math.isfinite(distribution_variance_weight) or distribution_variance_weight < 0):
        raise ValueError("distribution weights must be finite and non-negative")
    if distribution_min_samples < 2:
        raise ValueError("distribution_min_samples must be at least 2")
    if not math.isfinite(adversarial_alignment_weight) or adversarial_alignment_weight < 0:
        raise ValueError("adversarial_alignment_weight must be finite and non-negative")
    if adversarial_alignment_weight:
        if not math.isfinite(adversarial_discriminator_learning_rate) or adversarial_discriminator_learning_rate <= 0:
            raise ValueError("adversarial_discriminator_learning_rate must be finite and positive")
        if not math.isfinite(adversarial_discriminator_weight_decay) or adversarial_discriminator_weight_decay < 0:
            raise ValueError("adversarial_discriminator_weight_decay must be finite and non-negative")
        if adversarial_discriminator_updates < 1:
            raise ValueError("adversarial_discriminator_updates must be at least 1")
        if adversarial_min_samples < 2:
            raise ValueError("adversarial_min_samples must be at least 2")
    arity = int(cfg.get('arity', 2))
    _validate_cache_arity(rows, arity, 'train')
    _assert_finite_cache(rows, 'train')
    for corpus_id, corpus_validation in validation_by.items():
        _validate_cache_arity(corpus_validation, arity, f'validation/{corpus_id}')
        _assert_finite_cache(corpus_validation, f'validation/{corpus_id}')
    model = GeodesicMerge(d=rows[0]['leaves'].shape[-1], hidden=cfg['encoder_hidden'],
                          layers=cfg['encoder_layers'], heads=cfg['heads'],
                          max_angle=float(cfg.get('max_geodesic_angle', cfg.get('max_angle', 0.8))),
                          decoder=variant == 'autoencoder',
                          decoder_layers=cfg['decoder_layers'],
                          decoder_length_mode=str(cfg.get('decoder_length_mode', 'oracle')),
                          ratio_log_bound=float(cfg.get('ratio_log_bound', 4.0)),
                          ratio_loss_weight=float(cfg.get('ratio_loss_weight', 0.0)),
                          relational_loss_weight=float(cfg.get('relational_loss_weight', 0.0)),
                          importance_mode=str(cfg.get('importance_mode', 'token')),
                          importance_beta=float(cfg.get('importance_beta', 1.0)),
                          decoder_mode=str(cfg.get('decoder_mode', 'legacy')),
                          importance_loss_weight=float(cfg.get('importance_loss_weight', 0.0)),
                          importance_prior_weight=float(cfg.get('importance_prior_weight', 0.0)),
                          delta_loss_weight=float(cfg.get('delta_loss_weight', 0.0)),
                          sharpness_loss_weight=float(cfg.get('sharpness_loss_weight', 0.0)),
                          slot_loss_weight=float(cfg.get('slot_loss_weight', 0.0)),
                          sharpness_margin=float(cfg.get('sharpness_margin', 0.1)),
                          geodesic_parameterization=str(cfg.get('geodesic_parameterization', 'legacy')),
                          arity=arity).to(device)
    initialization_info = {}
    continuation_info = {}
    if initialization_path:
        initialization_info = _load_encoder_initialization(
            model, initialization_path, initialization_expected_hash,
            allow_autoencoder_source=allow_autoencoder_initialization)
        print(f"Loaded {initialization_info['initialization_shared_key_count']} shared encoder keys "
              f"from {initialization_path} sha256={initialization_info['initialization_checkpoint_sha256']}",
              flush=True)
    if continuation_path:
        continuation_info = _load_encoder_continuation(
            model, continuation_path, continuation_expected_hash)
        print(f"Continued exact encoder state ({continuation_info['continuation_loaded_key_count']} keys) "
              f"from {continuation_path} sha256={continuation_info['continuation_checkpoint_sha256']}; "
              "optimizer intentionally restarted.", flush=True)
    if freeze_encoder_steps:
        _set_encoder_trainable(model, False)
        decoder_trainable = [name for name, parameter in model.named_parameters()
                             if parameter.requires_grad]
        if not decoder_trainable:
            raise ValueError("freeze_encoder_steps requires a decoder/importance prediction module")
        if (float(cfg.get('reconstruction_weight', 0.0)) == 0.0 and
                float(cfg.get('importance_loss_weight', 0.0)) == 0.0):
            raise ValueError("decoder-only stage requires reconstruction or importance loss")
    # Include frozen parameters from the start. AdamW skips parameters with no
    # gradients, then safely begins their state on the first unfrozen update.
    # The opt-in encoder scale is also applied to the cosine floor; omitted
    # configs retain the exact historical one-group optimizer construction.
    opt = torch.optim.AdamW(
        _optimizer_parameter_groups(model, encoder_learning_rate_scale),
        lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    current_group_learning_rates = _set_optimizer_base_learning_rate(
        opt, float(cfg['learning_rate']))
    schedule = str(cfg.get('learning_rate_schedule', cfg.get('lr_schedule', 'none'))).lower()
    if schedule not in ('none', 'cosine'):
        raise ValueError("learning_rate_schedule must be 'none' or 'cosine'")
    warmup_steps = int(cfg.get('warmup_steps', 0))
    min_learning_rate = float(cfg.get('min_learning_rate', 0.0))
    if schedule == 'cosine':
        # Validate all scheduler parameters before creating any artifacts.
        _learning_rate_at_step(cfg['learning_rate'], min_learning_rate, cfg['steps'], warmup_steps, 0)
    stratified_batches = bool(cfg.get('stratified_multicorpus_batches',
                                      cfg.get('stratified_batches', False)))
    discriminator = discriminator_opt = None
    if adversarial_alignment_weight:
        # Linear initialization consumes CPU RNG. Restore it so the matched
        # control keeps the same merger/dropout random stream.
        rng_state = torch.get_rng_state()
        discriminator = _ParentLogisticDiscriminator(model.d).to(device)
        torch.set_rng_state(rng_state)
        discriminator_opt = torch.optim.AdamW(
            discriminator.parameters(), lr=adversarial_discriminator_learning_rate,
            weight_decay=adversarial_discriminator_weight_decay)
    out = Path(cfg['output_dir']); out.mkdir(parents=True, exist_ok=True)
    _require_free_storage(out, cfg.get('minimum_free_bytes_before_train', 0),
                          'training run')
    log = out / f'{variant}_seed{seed}.jsonl'
    amp_dtype = _amp_dtype(cfg, device)
    writer = _make_writer(cfg, variant, seed)
    indexed_rows = list(enumerate(rows))
    decoder_indexed_rows, decoder_multicorpus = indexed_rows, multicorpus
    if decoder_stage_full_arity:
        if variant != 'autoencoder' or not freeze_encoder_steps or str(cfg.get('decoder_mode', 'legacy')) != 'strict':
            raise ValueError("decoder_only_require_full_arity_rows requires a frozen strict autoencoder stage")
        decoder_indexed_rows, decoder_multicorpus = _full_arity_decoder_stage_rows(
            indexed_rows, multicorpus, arity)
    selection_metric = str(cfg.get('checkpoint_selection', 'validation_loss'))
    # The default preserves the historical one-file, best-validation behavior.
    # Setting this false makes that same canonical file track each eligible
    # validation checkpoint, without accumulating per-step files.
    keep_only_best = bool(cfg.get('keep_only_best', True))
    minimum_teacher_cosine = cfg.get('minimum_validation_teacher_cosine')
    minimum_per_corpus_teacher_cosine = cfg.get('minimum_per_corpus_validation_teacher_cosine')
    for name, value in (("minimum_validation_teacher_cosine", minimum_teacher_cosine),
                        ("minimum_per_corpus_validation_teacher_cosine", minimum_per_corpus_teacher_cosine)):
        if value is not None and not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite when provided")
    minimum_teacher_cosine = None if minimum_teacher_cosine is None else float(minimum_teacher_cosine)
    minimum_per_corpus_teacher_cosine = (None if minimum_per_corpus_teacher_cosine is None
                                         else float(minimum_per_corpus_teacher_cosine))
    moving_window = int(cfg.get('checkpoint_moving_average_window', 1))
    if moving_window < 1:
        raise ValueError("checkpoint_moving_average_window must be at least 1")
    selection_start_step = int(cfg.get('checkpoint_selection_start_step', 1))
    if selection_start_step < 1 or selection_start_step > int(cfg['steps']):
        raise ValueError("checkpoint_selection_start_step must be within configured training steps")
    reject_saturation = bool(cfg.get('reject_on_geodesic_saturation', False))
    saturation_patience = int(cfg.get('geodesic_saturation_patience_steps', 1))
    reject_cosine = bool(cfg.get('reject_on_cosine_guard_failure', False))
    defer_checkpoint_writes = bool(cfg.get('defer_selected_checkpoint_write_until_training_complete', False))
    deferred_checkpoint = None
    # Default false preserves fail-fast historical screens. A validation scale
    # run may finish its configured optimizer budget after permanent rejection;
    # no later checkpoint/validation can become eligible in that mode.
    continue_after_rejection = bool(cfg.get('continue_after_quality_rejection', False))
    if saturation_patience < 1:
        raise ValueError("geodesic_saturation_patience_steps must be at least 1")
    selection_history = []
    saturation_streak = 0
    rejected_reason = None
    final_step = 0
    model.train(); best = float('inf'); every = max(1, int(cfg.get('validation_every', cfg['steps'] // 10)))
    encoder_was_frozen = bool(freeze_encoder_steps)
    for s in range(cfg['steps']):
        step = s + 1
        encoder_frozen = variant == 'autoencoder' and step <= freeze_encoder_steps
        if encoder_was_frozen and not encoder_frozen:
            _set_encoder_trainable(model, True)
            encoder_was_frozen = False
            print(f"Unfroze encoder at optimizer step {step}; continuing the existing optimizer/schedule.",
                  flush=True)
        objective_multiplier = _autoencoder_objective_multiplier(
            step, objective_start_step, objective_ramp_steps)
        effective_weights = _effective_training_weights(
            cfg, variant == 'autoencoder', encoder_frozen, objective_multiplier)
        sampling_rows = decoder_indexed_rows if encoder_frozen else indexed_rows
        sampling_multicorpus = decoder_multicorpus if encoder_frozen else multicorpus
        indexed_batch = _sample_training_batch(
            sampling_rows, sampling_multicorpus, int(cfg['batch_size']), stratified=stratified_batches)
        _assert_finite_batch(indexed_batch)
        batch = [row for _, row in indexed_batch]
        opt.zero_grad(); sums = {}
        if schedule == 'cosine':
            current_lr = _learning_rate_at_step(
                cfg['learning_rate'], min_learning_rate, cfg['steps'], warmup_steps, s)
            current_group_learning_rates = _set_optimizer_base_learning_rate(opt, current_lr)
        else:
            current_lr = float(cfg['learning_rate'])
            current_group_learning_rates = {
                str(group.get('group_name', f'group_{index}')): float(group['lr'])
                for index, group in enumerate(opt.param_groups)
            }
        adversarial_active = discriminator is not None and not encoder_frozen
        pair_collector = (_DistributionMomentCollector(min(distribution_min_samples, adversarial_min_samples))
                          if adversarial_active else None)
        auxiliary_multiplier = 0.0 if encoder_frozen else objective_multiplier
        importance_multiplier = 1.0 if encoder_frozen else objective_multiplier
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
            total, p = step_loss_batch(
                model, batch, variant == 'autoencoder', device,
                effective_weights['reconstruction'], effective_weights['cycle'],
                effective_weights['distribution_mean'], effective_weights['distribution_variance'],
                distribution_min_samples, pair_collector,
                teacher_weight=effective_weights['teacher'],
                auxiliary_weight_multiplier=auxiliary_multiplier,
                importance_weight_multiplier=importance_multiplier,
                detach_decoder_parent=encoder_frozen)
        if adversarial_active:
            prediction, target = pair_collector.pairs()
            if prediction is None or prediction.shape[0] < adversarial_min_samples:
                adversarial_discriminator = total.detach() * 0.0
                adversarial_merger = total.detach() * 0.0
                adversarial_accuracy = total.detach() * 0.0
            else:
                adversarial_discriminator, adversarial_accuracy = _update_adversarial_discriminator(
                    discriminator, discriminator_opt, prediction, target, adversarial_discriminator_updates)
                # Subtracting fake-class BCE is precisely a gradient-reversal
                # update: D learns generated=0 while the merger maximizes that
                # same loss with D parameters frozen.
                adversarial_merger = _adversarial_merger_loss(discriminator, prediction)
                total = total + effective_weights['adversarial_merger'] * adversarial_merger
            p.update({'adversarial_discriminator': adversarial_discriminator.detach(),
                      'adversarial_merger': adversarial_merger.detach(),
                      'adversarial_discriminator_accuracy': adversarial_accuracy.detach()})
        if not torch.isfinite(total):
            raise FloatingPointError(
                f'non-finite training loss at step {s + 1}; batch={_format_batch_context(indexed_batch)}')
        for k, v in p.items():
            if not torch.isfinite(v):
                raise FloatingPointError(
                    f'non-finite {k} loss component at step {s + 1}; batch={_format_batch_context(indexed_batch)}')
            sums[k] = float(v)
        total.backward()
        bad_grad = _first_nonfinite_grad(model)
        if bad_grad != "unknown":
            raise FloatingPointError(
                f'non-finite gradient at step {s + 1}; first={bad_grad}; batch={_format_batch_context(indexed_batch)}')
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        telemetry, contributions, corpus_counts = _telemetry_record(
            sums, cfg, indexed_batch, multicorpus, grad_norm, current_lr,
            adversarial_alignment_weight, effective_weights)
        rec = {
            'step': step, 'loss': float(total.detach()), **telemetry, **sums,
            **{f'current_learning_rate/{name}': value
               for name, value in current_group_learning_rates.items()},
            'encoder_frozen': encoder_frozen,
            'encoder_learning_rate_scale': encoder_learning_rate_scale,
            'decoder_only_require_full_arity_rows': decoder_stage_full_arity,
            'autoencoder_objective_multiplier': objective_multiplier,
            'freeze_encoder_steps': freeze_encoder_steps,
            'autoencoder_objective_start_step': objective_start_step,
            'autoencoder_objective_ramp_steps': objective_ramp_steps,
            **initialization_info,
            **continuation_info,
            'distribution_mean_weight': distribution_mean_weight,
            'distribution_variance_weight': distribution_variance_weight,
            'adversarial_alignment_weight': adversarial_alignment_weight,
            'adversarial_discriminator_learning_rate': adversarial_discriminator_learning_rate,
            'adversarial_discriminator_updates': adversarial_discriminator_updates,
        }
        final_step = s + 1
        saturation_streak = (saturation_streak + 1 if
                             float(rec.get('geodesic_saturation_fraction', 0.0)) > 0.0 else 0)
        rec['geodesic_saturation_streak'] = saturation_streak
        step_rejection = _quality_rejection_reason(
            rec, saturation_streak, saturation_patience, reject_saturation, False)
        if step_rejection and rejected_reason is None:
            rejected_reason = step_rejection
        if writer is not None:
            writer.add_scalar('loss/train', rec['loss'], s + 1)
            writer.add_scalar('optimization/model_grad_norm', rec['grad_norm'], s + 1)
            writer.add_scalar('optimization/learning_rate', rec['current_learning_rate'], s + 1)
            for k, v in sums.items():
                if k.startswith('geodesic_'):
                    writer.add_scalar(f'geodesic/{k.removeprefix("geodesic_")}', v, s + 1)
                else:
                    writer.add_scalar(f'loss/{k}', v, s + 1)
            for name, value in contributions.items():
                writer.add_scalar(f'loss_weighted/{name}', value, s + 1)
            for name, count in corpus_counts.items():
                writer.add_scalar(f'batch_corpus/{name}/count', count, s + 1)
                writer.add_scalar(f'batch_corpus/{name}/fraction', count / len(indexed_batch), s + 1)
            writer.add_scalar('weights/distribution_mean', distribution_mean_weight, s + 1)
            writer.add_scalar('weights/distribution_variance', distribution_variance_weight, s + 1)
            writer.add_scalar('weights/adversarial_alignment', effective_weights['adversarial_merger'], s + 1)
            writer.add_scalar('weights/autoencoder_objective_multiplier', objective_multiplier, s + 1)
            writer.add_scalar('stage/encoder_frozen', float(encoder_frozen), s + 1)
            for name, value in effective_weights.items():
                writer.add_scalar(f'weights/effective/{name}', value, s + 1)
            if 'adversarial_discriminator_accuracy' in sums:
                writer.add_scalar('adversarial/discriminator_accuracy', sums['adversarial_discriminator_accuracy'], s + 1)
        if (not rejected_reason and validation_by and
                ((s + 1) % every == 0 or s + 1 == cfg['steps'])):
            was_training = model.training
            model.eval()
            if val_rows:
                rec['validation_loss'] = validation_loss(model, val_rows, variant, device, cfg)
                rec.update(validation_metrics(model, val_rows, device, cfg))
            else:
                rec.update(_multicorpus_validation(model, validation_by, variant, device, cfg))
            if was_training:
                model.train()
            if writer is not None:
                for key, value in rec.items():
                    if key.startswith('validation/') or key.startswith('macro_validation_'):
                        writer.add_scalar(key, value, s + 1)
                if 'validation_loss' in rec:
                    writer.add_scalar('loss/validation', rec['validation_loss'], s + 1)
                for key in ('validation_auc','validation_pr_auc','validation_balanced_accuracy','validation_teacher_cosine'):
                    if key in rec: writer.add_scalar(f'validation/{key.removeprefix("validation_")}', rec[key], s + 1)
            if selection_metric not in rec:
                raise KeyError(f"checkpoint_selection={selection_metric!r} is not a recorded validation metric")
            raw_selection_value = float(rec[selection_metric])
            if selection_metric in ('macro_validation_auc', 'macro_validation_max_attack_auc'):
                smoothed_selection_value = _trailing_average(
                    selection_history, raw_selection_value, moving_window)
                rec[f'{selection_metric}_raw'] = raw_selection_value
                rec[f'{selection_metric}_smoothed'] = smoothed_selection_value
            else:
                # The V2 smoothing option is deliberately scoped to the noisy
                # multicorpus macro AUC; all established alternative criteria
                # retain their exact raw-value semantics.
                smoothed_selection_value = raw_selection_value
            rec['checkpoint_selection_metric'] = selection_metric
            rec['checkpoint_selection_raw_value'] = raw_selection_value
            rec['checkpoint_selection_smoothed_value'] = smoothed_selection_value
            # Preserve this established field while making it the value actually
            # used for selection. Window=1 is exactly the historical behavior.
            rec['checkpoint_selection_value'] = smoothed_selection_value
            rec['checkpoint_moving_average_window'] = moving_window
            rec['minimum_validation_teacher_cosine'] = minimum_teacher_cosine
            rec['minimum_per_corpus_validation_teacher_cosine'] = minimum_per_corpus_teacher_cosine
            cosine_eligible, macro_guard_passed, per_corpus_guard_passed = _teacher_cosine_guard(
                rec, validation_by, minimum_teacher_cosine, minimum_per_corpus_teacher_cosine)
            selection_step_guard_passed = step >= selection_start_step
            eligible = cosine_eligible and selection_step_guard_passed
            rec['macro_teacher_cosine_guard_passed'] = macro_guard_passed
            rec['per_corpus_teacher_cosine_guard_passed'] = per_corpus_guard_passed
            rec['checkpoint_selection_start_step'] = selection_start_step
            rec['checkpoint_selection_step_guard_passed'] = selection_step_guard_passed
            rec['checkpoint_selection_eligible'] = eligible
            validation_rejection = _quality_rejection_reason(
                rec, saturation_streak, saturation_patience, False, reject_cosine)
            if validation_rejection and rejected_reason is None:
                rejected_reason = validation_rejection
                step_rejection = validation_rejection
            if writer is not None:
                writer.add_scalar('checkpoint_selection/raw', raw_selection_value, s + 1)
                writer.add_scalar('checkpoint_selection/smoothed', smoothed_selection_value, s + 1)
                writer.add_scalar('checkpoint_selection/eligible', float(eligible), s + 1)
            improved = eligible and smoothed_selection_value < best
            if improved:
                best = smoothed_selection_value
            if eligible and (improved or not keep_only_best):
                checkpoint_payload = {
                    'state_dict': model.state_dict(), 'variant': variant, 'config': cfg,
                    'architecture': model.architecture_metadata(),
                    'seed': seed, 'selected_step': s + 1,
                    'validation_loss': rec.get('validation_loss', rec.get('macro_validation_loss')),
                    'validation_metrics': {k: v for k, v in rec.items()
                                           if k.startswith('validation/') or k.startswith('macro_validation_')},
                    'checkpoint_selection_metric': selection_metric,
                    'checkpoint_selection_raw_value': raw_selection_value,
                    'checkpoint_selection_smoothed_value': smoothed_selection_value,
                    'checkpoint_selection_value': smoothed_selection_value,
                    'checkpoint_moving_average_window': moving_window,
                    'minimum_validation_teacher_cosine': minimum_teacher_cosine,
                    'minimum_per_corpus_validation_teacher_cosine': minimum_per_corpus_teacher_cosine,
                    'checkpoint_selection_start_step': selection_start_step,
                    'staged_training': {
                        'freeze_encoder_steps': freeze_encoder_steps,
                        'autoencoder_objective_start_step': objective_start_step,
                        'autoencoder_objective_ramp_steps': objective_ramp_steps,
                        'encoder_learning_rate_scale': encoder_learning_rate_scale,
                        **initialization_info,
                    },
                    'continuation': continuation_info or None,
                }
                if defer_checkpoint_writes:
                    # Keep exactly one selected state in host RAM. Updating its
                    # tensors in place avoids both accumulating checkpoints and
                    # transiently holding two host copies after each selection.
                    if deferred_checkpoint is None:
                        cpu_state = {name: value.detach().cpu().clone()
                                     for name, value in checkpoint_payload['state_dict'].items()}
                    else:
                        cpu_state = deferred_checkpoint['state_dict']
                        with torch.no_grad():
                            for name, value in checkpoint_payload['state_dict'].items():
                                cpu_state[name].copy_(value.detach(), non_blocking=False)
                    checkpoint_payload['state_dict'] = cpu_state
                    deferred_checkpoint = checkpoint_payload
                else:
                    _require_checkpoint_storage(cfg=cfg, path=out,
                                                context=f'checkpoint write at step {s + 1}')
                    atomic_torch_save(checkpoint_payload, out / f'{variant}_seed{seed}.pt')
        if rejected_reason:
            rec['run_rejected'] = True
            rec['rejection_reason'] = rejected_reason
            rec['preserved_best_checkpoint'] = (deferred_checkpoint is not None or
                                                 (out / f'{variant}_seed{seed}.pt').exists())
        with log.open('a') as f:
            f.write(json.dumps(rec) + '\n')
        if (s + 1) % every == 0 or step_rejection:
            print(variant, seed, rec, flush=True)
        if step_rejection:
            print(f'REJECTED {variant} seed={seed} step={s + 1}: {rejected_reason}', flush=True)
            if continue_after_rejection:
                print('Continuing configured optimizer steps with checkpoint selection permanently disabled.',
                      flush=True)
        if rejected_reason and not continue_after_rejection:
            break
    if deferred_checkpoint is not None:
        _require_checkpoint_storage(cfg=cfg, path=out,
                                    context='deferred selected checkpoint write after training')
        atomic_torch_save(deferred_checkpoint, out / f'{variant}_seed{seed}.pt')
    if not validation_by and not rejected_reason:
        fixed_final = bool(cfg.get('fixed_final_checkpoint_without_validation', False))
        if fixed_final and final_step != int(cfg['steps']):
            raise RuntimeError('fixed-final train-only checkpoint did not reach its configured final step')
        payload = {
            'state_dict': model.state_dict(), 'variant': variant, 'config': cfg, 'seed': seed,
            'architecture': model.architecture_metadata(),
            'staged_training': {
                'freeze_encoder_steps': freeze_encoder_steps,
                'autoencoder_objective_start_step': objective_start_step,
                'autoencoder_objective_ramp_steps': objective_ramp_steps,
                'encoder_learning_rate_scale': encoder_learning_rate_scale,
                **initialization_info,
            },
            'continuation': continuation_info or None,
        }
        if fixed_final:
            payload.update({
                'selected_step': final_step,
                'checkpoint_selection_metric': 'fixed_final_step_without_validation',
                'checkpoint_selection_value': None,
                'checkpoint_selection_start_step': final_step,
            })
        _require_checkpoint_storage(cfg=cfg, path=out,
                                    context='fixed final train-only checkpoint write')
        atomic_torch_save(payload, out / f'{variant}_seed{seed}.pt')
    status = {
        'status': 'rejected' if rejected_reason else 'completed',
        'variant': variant, 'seed': seed, 'final_step': final_step,
        'configured_steps': int(cfg['steps']), 'rejection_reason': rejected_reason,
        'best_checkpoint_preserved': (out / f'{variant}_seed{seed}.pt').exists(),
        'best_checkpoint_selection_value': None if not math.isfinite(best) else best,
        'freeze_encoder_steps': freeze_encoder_steps,
        'autoencoder_objective_start_step': objective_start_step,
        'autoencoder_objective_ramp_steps': objective_ramp_steps,
        'encoder_learning_rate_scale': encoder_learning_rate_scale,
        'decoder_only_require_full_arity_rows': decoder_stage_full_arity,
        'checkpoint_selection_start_step': selection_start_step,
        'continued_after_quality_rejection': bool(rejected_reason and continue_after_rejection),
        **initialization_info,
        **continuation_info,
    }
    (out / f'{variant}_seed{seed}_run_status.json').write_text(json.dumps(status, indent=2) + '\n')
    if writer is not None:
        writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--variant', choices=['encoder', 'autoencoder'], required=True)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.smoke:
        cfg['output_dir'] = f"{cfg['output_dir']}_smoke"
        if cfg.get('tensorboard_dir'):
            cfg['tensorboard_dir'] = f"{cfg['tensorboard_dir']}_smoke"
        cfg.update(steps=20, batch_size=4)
        if cfg.get('corpora'):
            # An initialized smoke must retain checkpoint-compatible dimensions;
            # otherwise use the historical tiny architecture.
            cfg.update(steps=3, batch_size=2, validation_every=3,
                       validation_metric_documents=16,
                       distribution_min_samples=2)
            if not (cfg.get('initialize_from_encoder_checkpoint',
                            cfg.get('encoder_initialization_checkpoint')) or
                    cfg.get('continue_encoder_from_checkpoint')):
                cfg.update(encoder_hidden=112, encoder_layers=1, decoder_layers=1)
        if a.variant == 'autoencoder' and int(cfg.get('freeze_encoder_steps', 0)):
            # Cover decoder-only, unfreeze, and a full ramp endpoint in 3 steps.
            cfg.update(freeze_encoder_steps=1, autoencoder_objective_start_step=1,
                       autoencoder_objective_ramp_steps=1)
        if str(cfg.get('learning_rate_schedule', cfg.get('lr_schedule', 'none'))).lower() == 'cosine':
            cfg['warmup_steps'] = min(int(cfg.get('warmup_steps', 0)), cfg['steps'] - 1)
        # Smoke models exercise wiring, not production quality gates.
        cfg['reject_on_cosine_guard_failure'] = False
        cfg['reject_on_geodesic_saturation'] = False
    device = torch.device(a.device if a.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    run(cfg, a.variant, a.seed if a.seed is not None else cfg['seed'], device)


if __name__ == '__main__':
    main()
