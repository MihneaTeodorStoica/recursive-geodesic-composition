"""Small, standalone recursive geodesic merger and training-only decoders."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F


class _ZeroInitWeights(torch.autograd.Function):
    """Return token weights exactly while routing gradients through softmax."""
    @staticmethod
    def forward(ctx, mass, semantic):
        return mass.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad, grad


class _StrictDecoder(nn.Module):
    """Versioned parent-only decoder; labels never enter this module."""
    def __init__(self, d, hidden, heads, layers, dropout, arity=2):
        super().__init__()
        self.arity = int(arity)
        self.parent = nn.Linear(d, hidden)
        self.slot_pos = nn.Parameter(torch.randn(1, self.arity, hidden) * .01)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 3, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.child_head = nn.Linear(hidden, d)
        self.importance_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.child_head.weight); nn.init.zeros_(self.child_head.bias)
        nn.init.zeros_(self.importance_head.weight); nn.init.zeros_(self.importance_head.bias)

    def forward(self, parent):
        p = F.normalize(parent, dim=-1)
        u = self.context(self.parent(p)[:, None, :].expand(-1, self.arity, -1) + self.slot_pos)
        children = F.normalize(p[:, None, :] + .25 * torch.tanh(self.child_head(u)) /
                               math.sqrt(parent.shape[-1]), dim=-1)
        importance_logits = self.importance_head(u).squeeze(-1)
        # Kept for the parent-only strict decode API without changing the
        # historical two-value internal return contract.
        self.last_importance_logits = importance_logits
        weights = torch.softmax(importance_logits, dim=1)
        return children, weights


class GeodesicMerge(nn.Module):
    def __init__(self, d=384, hidden=256, layers=2, heads=8, dropout=.05,
                 max_angle=.8, decoder=False, decoder_layers=1,
                 decoder_length_mode="oracle", ratio_log_bound=4.0,
                 ratio_loss_weight=0.0, relational_loss_weight=0.0,
                 importance_mode="token", importance_beta=1.0,
                 decoder_mode="legacy", importance_loss_weight=0.0,
                 importance_prior_weight=0.0, delta_loss_weight=0.0,
                 sharpness_loss_weight=0.0, slot_loss_weight=0.0,
                 sharpness_margin=0.1, geodesic_parameterization="legacy",
                 arity=2, composition_geometry="geodesic"):
        super().__init__()
        if hidden % heads: raise ValueError("hidden must divide heads")
        if isinstance(arity, bool) or int(arity) != arity or int(arity) < 2:
            raise ValueError("arity must be an integer >= 2")
        arity = int(arity)
        if importance_mode not in {"token", "semantic"}:
            raise ValueError(f"unknown importance_mode: {importance_mode}")
        if decoder_mode not in {"legacy", "strict"}:
            raise ValueError(f"unknown decoder_mode: {decoder_mode}")
        if arity != 2 and decoder and decoder_mode != "strict":
            raise ValueError("arity > 2 decoders must use parent-only decoder_mode='strict'")
        if geodesic_parameterization not in {"legacy", "tangent_expmap"}:
            raise ValueError(f"unknown geodesic_parameterization: {geodesic_parameterization}")
        if composition_geometry not in {"geodesic", "euclidean_residual", "free_euclidean"}:
            raise ValueError(f"unknown composition_geometry: {composition_geometry}")
        if composition_geometry != "geodesic" and geodesic_parameterization != "tangent_expmap":
            raise ValueError("Euclidean V4 variants require the parameter-matched tangent-expmap module layout")
        if not math.isfinite(max_angle) or max_angle <= 0:
            raise ValueError("max_angle must be finite and positive")
        if not math.isfinite(importance_beta) or importance_beta <= 0:
            raise ValueError("importance_beta must be finite and positive")
        self.d, self.hidden, self.max_angle = d, hidden, float(max_angle)
        self.arity = arity
        self.importance_mode, self.decoder_mode = importance_mode, decoder_mode
        self.geodesic_parameterization = geodesic_parameterization
        self.composition_geometry = composition_geometry
        self.importance_beta = float(importance_beta)
        self.input = nn.Sequential(nn.Linear(d + 2, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.pos = nn.Parameter(torch.randn(1, arity, hidden) * .01)
        enc = nn.TransformerEncoderLayer(hidden, heads, hidden * 3, dropout,
                                         activation="gelu", batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.trunk = nn.Sequential(nn.LayerNorm(hidden * arity + d), nn.Linear(hidden * arity + d, hidden * 3),
                                   nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 3, hidden * 3), nn.GELU())
        self.direction = nn.Linear(hidden * 3, d)
        if geodesic_parameterization == "legacy":
            self.angle = nn.Linear(hidden * 3, 1)
            nn.init.zeros_(self.angle.weight)
            nn.init.constant_(self.angle.bias, math.log((.04 / max_angle) / (1 - .04 / max_angle)))
        else:
            # A zero tangent is exactly the anchor, but expmap has identity
            # derivative there, so an off-anchor teacher can immediately move
            # this head. There is deliberately no separate scalar angle head.
            self.angle = None
            nn.init.zeros_(self.direction.weight); nn.init.zeros_(self.direction.bias)
            if composition_geometry == "free_euclidean":
                # Free Euclidean output has no anchor fallback, so it requires a
                # nonzero random output head at initialization.
                nn.init.xavier_uniform_(self.direction.weight)
        if decoder_length_mode not in {"oracle", "equal", "predicted"}:
            raise ValueError(f"unknown decoder_length_mode: {decoder_length_mode}")
        if not math.isfinite(ratio_log_bound) or ratio_log_bound <= 0:
            raise ValueError("ratio_log_bound must be finite and positive")
        self.decoder_length_mode = decoder_length_mode
        self.ratio_log_bound = float(ratio_log_bound)
        self.ratio_loss_weight = float(ratio_loss_weight)
        self.relational_loss_weight = float(relational_loss_weight)
        loss_values = (ratio_loss_weight, relational_loss_weight, importance_loss_weight,
                       importance_prior_weight, delta_loss_weight, sharpness_loss_weight,
                       slot_loss_weight, sharpness_margin)
        if not all(math.isfinite(float(x)) and float(x) >= 0 for x in loss_values):
            raise ValueError("auxiliary loss weights/margins must be finite and non-negative")
        self.importance_loss_weight = float(importance_loss_weight)
        self.importance_prior_weight = float(importance_prior_weight)
        self.delta_loss_weight = float(delta_loss_weight)
        self.sharpness_loss_weight = float(sharpness_loss_weight)
        self.slot_loss_weight = float(slot_loss_weight)
        self.sharpness_margin = float(sharpness_margin)
        self.decoder = None
        self.ratio_head = None
        self.strict_decoder = None
        if decoder and decoder_mode == "legacy":
            din = nn.Sequential(nn.Linear(d + 2, hidden), nn.LayerNorm(hidden), nn.GELU())
            dec = nn.TransformerEncoderLayer(hidden, heads, hidden * 3, dropout,
                                             activation="gelu", batch_first=True, norm_first=True)
            self.decoder = nn.ModuleDict({"input": din, "context": nn.TransformerEncoder(dec, decoder_layers, enable_nested_tensor=False),
                                          "head": nn.Linear(hidden, d)})
            nn.init.zeros_(self.decoder["head"].weight); nn.init.zeros_(self.decoder["head"].bias)
            if decoder_length_mode == "predicted":
                self.ratio_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden),
                                                nn.GELU(), nn.Linear(hidden, 1))
                nn.init.zeros_(self.ratio_head[-1].weight)
                nn.init.zeros_(self.ratio_head[-1].bias)
        # Opt-in modules are constructed last: defaults retain historical keys/RNG.
        self.salience = None
        if importance_mode == "semantic":
            self.salience = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
            nn.init.zeros_(self.salience[-1].weight); nn.init.zeros_(self.salience[-1].bias)
        if decoder and decoder_mode == "strict":
            self.strict_decoder = _StrictDecoder(d, hidden, heads, decoder_layers, dropout, arity)
        self._collect_aux = False
        self._merge_aux, self._decode_aux, self._geodesic_angles = [], [], []
        self._last_merge_aux = None

    def begin_aux_collection(self):
        self._collect_aux = True
        self._merge_aux.clear(); self._decode_aux.clear(); self._geodesic_angles.clear()

    def end_aux_collection(self, zero, auxiliary_multiplier=1.0, importance_multiplier=None):
        """Finish one loss collection with optional training-time coefficient scaling.

        Multipliers are external controls rather than model state so old configs,
        checkpoints, and direct callers retain exactly the historical weighting.
        """
        self._collect_aux = False
        auxiliary_multiplier = float(auxiliary_multiplier)
        importance_multiplier = (auxiliary_multiplier if importance_multiplier is None
                                 else float(importance_multiplier))
        values = {}
        if self._merge_aux:
            regs = [self.importance_regularizers(a) for a in self._merge_aux]
            for key in regs[0]: values[key] = torch.stack([x[key] for x in regs]).mean()
        if self._decode_aux:
            values["importance"] = torch.stack([x[0] for x in self._decode_aux]).mean()
            values["predicted_slot0_use"] = torch.stack([x[2][:, 0].mean() for x in self._decode_aux]).mean()
        if self._geodesic_angles:
            angles = torch.cat(self._geodesic_angles)
            values["geodesic_angle_mean"] = angles.mean()
            values["geodesic_angle_p95"] = torch.quantile(angles, .95)
            values["geodesic_angle_max"] = angles.max()
            values["geodesic_saturation_fraction"] = ((angles >= .95 * self.max_angle).float().mean()
                                                        if self.composition_geometry == "geodesic"
                                                        else angles.new_zeros(()))
        for key in ("prior", "delta", "sharpness", "slot", "importance"):
            values.setdefault(key, zero)
        for key in ("geodesic_angle_mean", "geodesic_angle_p95", "geodesic_angle_max",
                    "geodesic_saturation_fraction"):
            values.setdefault(key, zero.detach())
        weighted = (auxiliary_multiplier * (
                        self.importance_prior_weight * values["prior"] +
                        self.delta_loss_weight * values["delta"] +
                        self.sharpness_loss_weight * values["sharpness"] +
                        self.slot_loss_weight * values["slot"]) +
                    importance_multiplier * self.importance_loss_weight * values["importance"])
        return weighted, values

    @staticmethod
    def _tanhc(value):
        """Stable tanh(x)/x with value and derivative well-defined at zero."""
        square = value.square()
        series = 1.0 - square / 3.0 + 2.0 * square.square() / 15.0
        safe_value = torch.where(value == 0, torch.ones_like(value), value)
        ratio = torch.tanh(value) / safe_value
        threshold = math.sqrt(torch.finfo(value.dtype).eps)
        return torch.where(value.abs() < threshold, series, ratio)

    def merge(self, children, lengths, return_aux=False):
        if children.ndim == 2: children, lengths = children[None], lengths[None]; single = True
        else: single = False
        if children.ndim != 3 or children.shape[1:] != (self.arity, self.d):
            raise ValueError(f"expected [B,{self.arity},D]")
        if lengths.ndim != 2 or lengths.shape != children.shape[:2]:
            raise ValueError(f"expected child masses [B,{self.arity}]")
        lengths = lengths.to(children)
        if not torch.isfinite(lengths).all() or (lengths < 0).any():
            raise ValueError("child masses must be finite and non-negative")
        active = lengths > 0
        if (~active.any(dim=1)).any():
            raise ValueError("each merge must have at least one positive child mass")
        if self.arity > 2:
            if (active.sum(1) < 2).any():
                raise ValueError("k-ary merge requires 2..arity active children; carry singletons unchanged")
            # Reduction pads only the trailing slots. Rejecting holes protects
            # learned ordered slot semantics and catches malformed cache rows.
            seen_inactive = (~active).cummax(dim=1).values
            if (active & seen_inactive).any():
                raise ValueError("active children must be a contiguous prefix")
        x = F.normalize(children, dim=-1)
        mass = lengths / lengths.sum(1, keepdim=True).clamp_min(1e-8)
        features = torch.cat((x, torch.stack((mass, mass.clamp_min(1e-8).log()), -1)), -1)
        h = self.context(self.input(features) + self.pos, src_key_padding_mask=~active) * active[..., None]
        corrections = torch.zeros_like(mass)
        weights = mass
        if self.salience is not None:
            corrections = self.importance_beta * torch.tanh(self.salience(h).squeeze(-1)) * active
            logits = mass.clamp_min(1e-8).log() + corrections
            semantic_weights = torch.softmax(logits.masked_fill(~active, -torch.inf), dim=1)
            # Preserve bitwise zero-init equality while retaining the softmax
            # gradient that lets the initially-zero salience head start moving.
            weights = (_ZeroInitWeights.apply(mass, semantic_weights)
                       if not torch.any(corrections.detach()) else semantic_weights)
        euclidean_anchor = (x * weights[..., None]).sum(1)
        anchor = F.normalize(euclidean_anchor, dim=-1)
        trunk_anchor = (euclidean_anchor if self.composition_geometry == "euclidean_residual" else
                        torch.zeros_like(anchor) if self.composition_geometry == "free_euclidean" else anchor)
        z = self.trunk(torch.cat((h.flatten(1), trunk_anchor), -1))
        raw_tangent = self.direction(z)
        tangent = raw_tangent - (raw_tangent * anchor).sum(-1, keepdim=True) * anchor
        raw_tangent_norm = tangent.norm(dim=-1, keepdim=True)
        if self.composition_geometry == "euclidean_residual":
            geodesic_correction = raw_tangent
            root = F.normalize(euclidean_anchor + raw_tangent, dim=-1)
            angle = torch.acos((root * anchor).sum(-1, keepdim=True).clamp(-1.0, 1.0))
        elif self.composition_geometry == "free_euclidean":
            geodesic_correction = raw_tangent
            root = F.normalize(raw_tangent, dim=-1)
            angle = torch.acos((root * anchor).sum(-1, keepdim=True).clamp(-1.0, 1.0))
        elif self.geodesic_parameterization == "legacy":
            direction = tangent / raw_tangent_norm.clamp_min(1e-6)
            angle = self.max_angle * torch.sigmoid(self.angle(z))
            root = F.normalize(torch.cos(angle) * anchor + torch.sin(angle) * direction, dim=-1)
            geodesic_correction = angle * direction
        else:
            # Radial tanh smoothly maps any projected tangent into the open
            # geodesic ball. tanhc(0)=1 preserves the full teacher gradient at
            # the zero-initialized correction.
            geodesic_correction = tangent * self._tanhc(raw_tangent_norm / self.max_angle)
            angle = geodesic_correction.norm(dim=-1, keepdim=True)
            # torch.sinc(x/pi) is sin(x)/x, including its stable value at zero.
            root = (torch.cos(angle) * anchor +
                    torch.sinc(angle / math.pi) * geodesic_correction)
        aux = {"anchor": anchor, "angle": angle.squeeze(-1),
               "geodesic_correction": geodesic_correction,
               "correction_norm": geodesic_correction.norm(dim=-1),
               "raw_tangent": raw_tangent, "raw_tangent_norm": raw_tangent_norm.squeeze(-1),
               "geodesic_saturated": ((angle.squeeze(-1) >= .95 * self.max_angle)
                                        if self.composition_geometry == "geodesic"
                                        else torch.zeros_like(angle.squeeze(-1), dtype=torch.bool)),
               "composition_geometry": self.composition_geometry,
               "euclidean_anchor": euclidean_anchor,
               "residual_norm": raw_tangent.norm(dim=-1),
               "effective_weights": weights, "token_fractions": mass,
               "corrections": corrections, "active_mask": active}
        self._last_merge_aux = aux
        if self._collect_aux:
            self._geodesic_angles.append(aux["angle"].detach().reshape(-1))
            if self.salience is not None: self._merge_aux.append(aux)
        if single: root = root[0]
        if return_aux:
            if self.salience is None:
                keys = ("anchor", "angle", "geodesic_correction", "correction_norm",
                        "raw_tangent", "raw_tangent_norm", "geodesic_saturated")
                return root, {key: aux[key] for key in keys}
            return root, aux
        return root

    @staticmethod
    def compose_anchor(children, weights):
        return F.normalize((F.normalize(children, dim=-1) * weights[..., None]).sum(1), dim=-1)

    def importance_regularizers(self, aux):
        q, w, delta, active = (aux[k] for k in ("token_fractions", "effective_weights", "corrections", "active_mask"))
        eps = 1e-8
        prior = (q * (q.clamp_min(eps).log() - w.clamp_min(eps).log())).sum(1).mean()
        delta_loss = delta.square().sum() / active.sum().clamp_min(1)
        hq = -(q * q.clamp_min(eps).log()).sum(1)
        hw = -(w * w.clamp_min(eps).log()).sum(1)
        sharpness = F.relu(hq - hw - self.sharpness_margin).square().mean()
        slot = (w.mean(0) - q.mean(0)).square().sum()
        ratio = w / q.clamp_min(eps)
        return {"prior": prior, "delta": delta_loss, "sharpness": sharpness, "slot": slot,
                "correction_saturation": ((delta.abs() >= .95 * self.importance_beta) & active).sum() / active.sum().clamp_min(1),
                "entropy_change": (hw - hq).mean(), "slot0_use": w[:, 0].mean(),
                "effective_token_ratio_p05": torch.quantile(ratio[active], .05),
                "effective_token_ratio_p95": torch.quantile(ratio[active], .95)}

    def predict_child_masses(self, parent):
        """Predict two strictly-positive, normalized masses from only the parent."""
        if self.strict_decoder is not None:
            single = parent.ndim == 1
            _, weights = self._strict_forward(parent)
            return weights[0] if single else weights
        if self.ratio_head is None: raise RuntimeError("ratio prediction was not enabled")
        single = parent.ndim == 1
        if single: parent = parent[None]
        log_ratio = self.ratio_log_bound * torch.tanh(self.ratio_head(F.normalize(parent, dim=-1)))
        masses = torch.softmax(torch.cat((0.5 * log_ratio, -0.5 * log_ratio), dim=-1), dim=-1)
        return masses[0] if single else masses

    def decoder_masses(self, parent, oracle_lengths=None):
        if self.decoder_mode == "strict": return self.predict_child_masses(parent)
        batch = 1 if parent.ndim == 1 else parent.shape[0]
        if self.decoder_length_mode == "predicted": return self.predict_child_masses(parent)
        if self.decoder_length_mode == "equal":
            return parent.new_ones((2,) if parent.ndim == 1 else (batch, 2))
        if oracle_lengths is None: raise ValueError("oracle decoder mode requires child lengths")
        return oracle_lengths.to(parent)

    def _strict_forward(self, parent):
        if self.strict_decoder is None: raise RuntimeError("strict decoder was not enabled")
        if parent.ndim == 1: parent = parent[None]
        return self.strict_decoder(parent)

    def decode(self, parent, lengths=None, return_aux=False):
        single = parent.ndim == 1
        if self.strict_decoder is not None:
            out, weights = self._strict_forward(parent)
            logits = self.strict_decoder.last_importance_logits
            if single: out, weights, logits = out[0], weights[0], logits[0]
            return (out, {"importance_weights": weights,
                          "importance_logits": logits}) if return_aux else out
        if self.decoder is None: raise RuntimeError("decoder was not enabled")
        lengths = self.decoder_masses(parent, lengths)
        if single: parent, lengths = parent[None], lengths[None]
        if (~(lengths > 0).any(dim=1)).any(): raise ValueError("decoder masks may not be all zero")
        lengths = lengths.to(parent); mass = lengths / lengths.sum(1, keepdim=True).clamp_min(1e-8)
        p = F.normalize(parent, dim=-1)[:, None, :].expand(-1, 2, -1)
        x = torch.cat((p, torch.stack((mass, mass.clamp_min(1e-8).log()), -1)), -1)
        h = self.decoder["context"](self.decoder["input"](x) + self.pos, src_key_padding_mask=lengths <= 0)
        out = F.normalize(p + .25 * torch.tanh(self.decoder["head"](h)) / math.sqrt(self.d), dim=-1)
        return out[0] if single else out

    def record_importance_target(self, predicted, token_fractions, effective_weights=None):
        target = effective_weights if self.importance_mode == "semantic" else token_fractions
        target = target.detach()
        loss = (target * (target.clamp_min(1e-8).log() - predicted.clamp_min(1e-8).log())).sum(1).mean()
        if self._collect_aux: self._decode_aux.append((loss, target, predicted))
        return loss

    def architecture_metadata(self):
        version = (3 if self.geodesic_parameterization == "tangent_expmap" else
                   2 if (self.importance_mode == "semantic" or self.decoder_mode == "strict") else 1)
        version = max(version, 4 if self.arity != 2 else version)
        return {"version": version, "arity": self.arity,
                "importance_mode": self.importance_mode, "decoder_mode": self.decoder_mode,
                "composition_geometry": self.composition_geometry,
                "geodesic_parameterization": self.geodesic_parameterization,
                "max_geodesic_angle": self.max_angle,
                "importance_beta": self.importance_beta,
                "loss_coefficients": {k: getattr(self, k) for k in
                    ("importance_loss_weight", "importance_prior_weight", "delta_loss_weight",
                     "sharpness_loss_weight", "slot_loss_weight")}}

    def load_compatible_state_dict(self, state_dict):
        """Strict legacy load, or explicit zero-init/copy migration for opt-in modules."""
        if (self.importance_mode == "token" and self.decoder_mode == "legacy" and
                self.geodesic_parameterization == "legacy"):
            return self.load_state_dict(state_dict, strict=True)
        own = self.state_dict()
        migrating_legacy_geometry = (self.geodesic_parameterization == "tangent_expmap" and
                                     "angle.weight" in state_dict)
        copied = {k: v for k, v in state_dict.items()
                  if k in own and own[k].shape == v.shape and
                  not (migrating_legacy_geometry and k.startswith("direction."))}
        if self.strict_decoder is not None and "decoder.input.0.weight" in state_dict:
            copied["strict_decoder.parent.weight"] = state_dict["decoder.input.0.weight"][:, :self.d]
            copied["strict_decoder.parent.bias"] = state_dict["decoder.input.0.bias"]
            for suffix in ("weight", "bias"):
                key = f"decoder.head.{suffix}"
                if key in state_dict: copied[f"strict_decoder.child_head.{suffix}"] = state_dict[key]
        return self.load_state_dict(copied, strict=False)


def weighted_average(children, lengths):
    if children.ndim == 2: children, lengths = children[None], lengths[None]; single = True
    else: single = False
    mass = lengths.to(children) / lengths.sum(1, keepdim=True).clamp_min(1e-8)
    out = F.normalize((F.normalize(children, dim=-1) * mass[..., None]).sum(1), dim=-1)
    return out[0] if single else out


def contiguous_reduction_groups(count, arity):
    """Deterministic contiguous groups: merge sizes 2..K, carry final singleton."""
    count, arity = int(count), int(arity)
    if count < 0 or arity < 2:
        raise ValueError("count must be non-negative and arity must be >= 2")
    groups = []
    start = 0
    while start < count:
        size = min(arity, count - start)
        groups.append((start, size))
        start += size
    return groups


def recursive_reduce(model, leaves, lengths, decode=False):
    """True K-ary contiguous balanced reduction; singleton remainders are carried."""
    levels = []
    nodes, sizes = leaves, lengths
    while nodes.shape[1] > 1:
        old_nodes, old_sizes = nodes, sizes
        groups = contiguous_reduction_groups(nodes.shape[1], model.arity)
        merged = [(start, size) for start, size in groups if size >= 2]
        children, child_sizes = [], []
        for start, size in merged:
            pad = model.arity - size
            children.append(F.pad(old_nodes[:, start:start + size], (0, 0, 0, pad)))
            child_sizes.append(F.pad(old_sizes[:, start:start + size], (0, pad)))
        child = torch.stack(children, 1).reshape(-1, model.arity, model.d)
        merge_sizes = torch.stack(child_sizes, 1).reshape(-1, model.arity)
        parent = model.merge(child, merge_sizes)
        levels.append((parent, child, merge_sizes))
        parent = parent.reshape(leaves.shape[0], len(merged), model.d)
        parent_sizes = merge_sizes.sum(1).reshape(leaves.shape[0], len(merged))
        nodes, sizes = parent, parent_sizes
        if groups[-1][1] == 1:
            nodes = torch.cat([nodes, old_nodes[:, -1:, :]], dim=1)
            sizes = torch.cat([sizes, old_sizes[:, -1:]], dim=1)
    return nodes[:, 0], levels
