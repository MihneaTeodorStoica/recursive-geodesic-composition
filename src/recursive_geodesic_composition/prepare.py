"""Create immutable, split-separated cached tensors from a HF revision."""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
import torch
from .data import FrozenTeacher, stream_document_ranges, tokenize_text, chunk_token_ids


def random_chunk_token_ids(token_ids, minimum, maximum, rng):
    """Split token_ids into contiguous, deterministic variable-size chunks."""
    if minimum < 4 or maximum < minimum:
        raise ValueError("random chunk bounds must satisfy 4 <= min <= max")
    chunks = []
    start = 0
    while start < len(token_ids):
        remaining = len(token_ids) - start
        if remaining <= maximum:
            if remaining >= 4:
                chunks.append(token_ids[start:])
            break
        size = rng.randint(minimum, maximum)
        chunks.append(token_ids[start:start + size])
        start += size
    return chunks


def _random_seed(seed, document_id, augmentation, purpose=""):
    value = f"{seed}:{document_id}:{augmentation}:{purpose}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _usable_tree_count(chunk_lengths, max_leaves, teacher_max_tokens):
    count = 1
    limit = len(chunk_lengths) if max_leaves is None else min(len(chunk_lengths), max_leaves)
    while count * 2 <= limit:
        candidate = count * 2
        if sum(chunk_lengths[:candidate]) > teacher_max_tokens:
            break
        count = candidate
    return count


def _embed_targets(teacher, spans, teacher_max, batch_size=32, max_padded_tokens=32768, context="targets"):
    """Embed eligible full spans in safe padded batches, retaining span order.

    A count-only batch of 32 8192-token examples can exhaust memory.  The
    padded-token budget bounds ``batch_len * longest_len`` as well as count.
    """
    embedded = []
    batch, batch_meta, longest = [], [], 0
    def flush():
        nonlocal batch, batch_meta, longest
        if batch:
            outputs = teacher.embed(batch, len(batch))
            for target_index, span_len, output in zip(batch_meta, map(len, batch), outputs):
                if not torch.isfinite(output).all():
                    raise FloatingPointError(
                        f"teacher produced non-finite target embedding: {context} "
                        f"target_index={target_index} target_tokens={span_len} "
                        f"teacher_max_tokens={teacher_max}")
                embedded.append(output)
            batch, batch_meta, longest = [], [], 0
    for target_index, span in enumerate(spans):
        if len(span) > teacher_max:
            continue
        candidate_longest = max(longest, len(span))
        if batch and (len(batch) >= batch_size or (len(batch) + 1) * candidate_longest > max_padded_tokens):
            flush()
        batch.append(span); batch_meta.append(target_index); longest = max(longest, len(span))
    flush()
    iterator = iter(embedded)
    return [next(iterator) if len(span) <= teacher_max else None for span in spans]


def _balanced_plan(count):
    """Parallel adjacent-pair reductions; positions refer to current nodes."""
    plan = []
    while count > 1:
        positions = list(range(0, count - 1, 2))
        plan.append(positions)
        count = len(positions) + (count % 2)
    return plan


def _tree_plan(count, template):
    """Small deterministic family of valid contiguous merge templates."""
    if template == "balanced":
        return _balanced_plan(count)
    plan = []
    while count > 1:
        if template == "left":
            pos = 0
        elif template == "right":
            pos = count - 2
        elif template == "outside_in":
            pos = 0 if len(plan) % 2 == 0 else count - 2
        else:
            raise ValueError(f"unknown tree template: {template}")
        plan.append([pos])
        count -= 1
    return plan


def _targets_for_plan(chunks, plan, teacher, teacher_max, batch_size=32, max_padded_tokens=32768, context="targets"):
    """Generate exact full-span targets, batching variable-length inputs safely."""
    spans, nodes = [], list(chunks)
    for positions in plan:
        merged_at = []
        for pos in positions:
            if pos < 0 or pos + 1 >= len(nodes):
                raise ValueError("invalid contiguous tree template")
            merged_at.append(nodes[pos] + nodes[pos + 1])
        spans.extend(merged_at)
        for pos, merged in reversed(list(zip(positions, merged_at))):
            nodes[pos:pos + 2] = [merged]
    return _embed_targets(teacher, spans, teacher_max, batch_size, max_padded_tokens, context)


def _balanced_targets(chunks, teacher, teacher_max, batch_size=32, max_padded_tokens=32768, context="targets"):
    return _targets_for_plan(chunks, _balanced_plan(len(chunks)), teacher, teacher_max,
                             batch_size, max_padded_tokens, context)


def _random_merge_plan(chunks, teacher, teacher_max, rng):
    """Compatibility helper for tests/old callers; targets are still batched."""
    plan, nodes = [], list(chunks)
    while len(nodes) > 1:
        pos = rng.randrange(len(nodes) - 1)
        plan.append([pos])
        nodes[pos:pos + 2] = [nodes[pos] + nodes[pos + 1]]
    targets = _targets_for_plan(chunks, plan, teacher, teacher_max)
    return [level[0] for level in plan], targets


def _make_row(doc_id, chunks, teacher, cfg, *, cap_to_teacher=True, max_leaves=8, tree_rng=None):
    teacher_max = int(cfg.get("teacher_max_tokens", 8192))
    if cap_to_teacher:
        count = _usable_tree_count([len(x) for x in chunks], max_leaves, teacher_max)
        if count < 2:
            return None
        chunks = chunks[:count]
    elif max_leaves is not None:
        chunks = chunks[:int(max_leaves)]
    if len(chunks) < 2:
        return None

    leaves = teacher.embed(chunks, 32)
    if not torch.isfinite(leaves).all():
        raise FloatingPointError(f"teacher produced non-finite leaf embedding: doc_id={doc_id} leaves={len(chunks)}")
    lengths = torch.tensor([len(x) for x in chunks], dtype=torch.float32)
    row = {"id": doc_id, "leaves": leaves, "lengths": lengths}
    if tree_rng is not None:
        templates = cfg.get("tree_templates", ["balanced", "left", "right", "outside_in"])
        template = templates[tree_rng.randrange(len(templates))]
        plan = _tree_plan(len(chunks), template)
        row["tree_template"] = template
        row["targets"] = _targets_for_plan(
            chunks, plan, teacher, teacher_max, int(cfg.get("teacher_target_batch_size", 32)),
            int(cfg.get("teacher_target_max_padded_tokens", 32768)),
            context=f"doc_id={doc_id} template={template} leaves={len(chunks)} total_tokens={sum(len(x) for x in chunks)}")
    else:
        row["targets"] = _balanced_targets(
            chunks, teacher, teacher_max, int(cfg.get("teacher_target_batch_size", 32)),
            int(cfg.get("teacher_target_max_padded_tokens", 32768)),
            context=f"doc_id={doc_id} template=balanced leaves={len(chunks)} total_tokens={sum(len(x) for x in chunks)}")
    return row


def _span_buckets(cfg):
    buckets = cfg.get("train_span_length_buckets") or cfg.get("span_length_buckets")
    if not buckets:
        return None
    parsed = []
    for item in buckets:
        if isinstance(item, dict):
            lo, hi = int(item["min"]), int(item["max"])
        else:
            lo, hi = int(item[0]), int(item[1])
        if lo < 8 or hi < lo:
            raise ValueError("span buckets must satisfy 8 <= min <= max")
        parsed.append((lo, hi))
    return parsed


def _random_span(token_ids, cfg, rng):
    buckets = _span_buckets(cfg)
    if not buckets:
        return token_ids
    feasible = [(lo, min(hi, len(token_ids))) for lo, hi in buckets if len(token_ids) >= lo]
    if not feasible:
        return []
    lo, hi = rng.choice(feasible)
    length = rng.randint(lo, hi)
    start = rng.randint(0, len(token_ids) - length)
    # Random start and random end are represented by choosing both start and
    # length/end deterministically from this document's augmentation RNG.
    return token_ids[start:start + length]


def _training_segmentations(cfg, doc, token_ids, augmentations):
    random_spans = bool(cfg.get("random_training_spans", False))
    if not random_spans:
        segmentations = [chunk_token_ids(token_ids, cfg["chunk_tokens"])]
        if augmentations:
            tree_limit = int(cfg.get("max_tree_leaves", 8))
            context_limit = int(cfg.get("teacher_max_tokens", 8192))
            maximum = min(int(cfg.get("random_chunk_max", cfg["chunk_tokens"])),
                          context_limit // tree_limit)
            minimum = int(cfg.get("random_chunk_min", maximum))
            if minimum < 4 or minimum > maximum:
                raise ValueError("random chunk bounds do not fit teacher/tree limits")
            for augmentation in range(augmentations):
                rng = random.Random(_random_seed(cfg.get("seed", 0), doc.id, augmentation))
                segmentations.append(random_chunk_token_ids(token_ids, minimum, maximum, rng))
        return segmentations

    segmentations = []
    total = max(1, augmentations)
    minimum = int(cfg.get("random_chunk_min", 256))
    maximum = int(cfg.get("random_chunk_max", cfg["chunk_tokens"]))
    for augmentation in range(total):
        rng = random.Random(_random_seed(cfg.get("seed", 0), doc.id, augmentation, "span"))
        span = _random_span(token_ids, cfg, rng)
        if not span:
            continue
        # Chunk boundaries are random, contiguous, and cover the sampled span.
        chunk_rng = random.Random(_random_seed(cfg.get("seed", 0), doc.id, augmentation, "chunks"))
        segmentations.append(random_chunk_token_ids(span, minimum, maximum, chunk_rng))
    return segmentations


def prepare_split(cfg, teacher, name, start, count, out, documents=None, corpus_id=None):
    rows = []
    requested = count
    augmentations = int(cfg.get(f"{name}_augmentations", 0))
    random_train = name == "train" and bool(cfg.get("random_training_spans", False))
    documents = documents if documents is not None else stream_document_ranges(
        cfg["dataset_name"], cfg["dataset_config"], cfg.get("dataset_revision"),
        {name: (start, requested)})[name]
    for doc in documents:
        if corpus_id is not None:
            # IDs enter validation grouping, so corpus namespace is part of the
            # immutable identity rather than relying on source IDs being unique.
            doc = type(doc)(f"{corpus_id}:{doc.id}", doc.text)
        token_ids = tokenize_text(teacher.tokenizer, doc.text)
        if random_train:
            segmentations = _training_segmentations(cfg, doc, token_ids, augmentations)
        else:
            segmentations = [chunk_token_ids(token_ids, cfg["chunk_tokens"])]
            # Keep validation/test fixed for now.  Only legacy configs that
            # explicitly ask for non-train augmentations get old behavior.
            if augmentations:
                tree_limit = int(cfg.get("max_tree_leaves", 8))
                context_limit = int(cfg.get("teacher_max_tokens", 8192))
                maximum = min(int(cfg.get("random_chunk_max", cfg["chunk_tokens"])),
                              context_limit // tree_limit)
                minimum = int(cfg.get("random_chunk_min", maximum))
                for augmentation in range(augmentations):
                    rng = random.Random(_random_seed(cfg.get("seed", 0), doc.id, augmentation))
                    segmentations.append(random_chunk_token_ids(token_ids, minimum, maximum, rng))
        for aug_index, chunks in enumerate(segmentations):
            if random_train:
                tree_rng = None
                if bool(cfg.get("varied_training_trees", False)):
                    tree_rng = random.Random(_random_seed(cfg.get("seed", 0), doc.id, aug_index, "tree"))
                row = _make_row(doc.id, chunks, teacher, cfg, cap_to_teacher=False,
                                max_leaves=cfg.get("train_max_tree_leaves"), tree_rng=tree_rng)
            else:
                row = _make_row(doc.id, chunks, teacher, cfg, cap_to_teacher=True,
                                max_leaves=int(cfg.get("max_tree_leaves", 8)))
            if row is not None:
                rows.append(row)
    if not rows:
        raise RuntimeError(f"No usable documents in {name}")
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, path)
    Path(str(path) + ".json").write_text(json.dumps({
        "split": name, "source_start": start, "requested": requested,
        "usable": len(rows), "augmentations": augmentations,
        "random_training_spans": random_train,
        "varied_training_trees": bool(cfg.get("varied_training_trees", False)) if name == "train" else False,
        "span_length_buckets": cfg.get("train_span_length_buckets") or cfg.get("span_length_buckets"),
        "corpus_id": corpus_id,
        "dataset": cfg["dataset_name"], "config": cfg.get("dataset_config"),
        "revision": cfg.get("dataset_revision"), "model": cfg["model_name"],
        "model_revision": cfg.get("model_revision")}, indent=2))
    print(f"{name}: {len(rows)} usable documents -> {path}")


def _prepare_multicorpus(cfg, teacher, smoke=False):
    """Prepare only train/validation roles in corpus-separated namespaces."""
    corpora = cfg.get("corpora")
    if not isinstance(corpora, list) or not corpora:
        raise ValueError("multicorpus config requires a non-empty corpora list")
    ids = [str(item["id"]) for item in corpora]
    if len(ids) != len(set(ids)) or any(not x or "/" in x or x in ("test", "official") for x in ids):
        raise ValueError("corpus IDs must be unique, non-empty path components")
    weights = [float(item["weight"]) for item in corpora]
    if any(x <= 0 for x in weights) or abs(sum(weights) - 1.0) > 1e-8:
        raise ValueError("corpus mixture weights must be positive and sum to 1")
    for corpus in corpora:
        corpus_id = str(corpus["id"])
        local = dict(cfg)
        local.update(dataset_name=corpus["dataset_name"],
                     dataset_config=corpus.get("dataset_config"),
                     dataset_revision=corpus["dataset_revision"])
        if smoke:
            local["train_augmentations"] = 1
        ranges = {
            role: (int(corpus[f"{role}_start"]), int(corpus[f"{role}_documents"]))
            for role in ("train", "validation")
        }
        train_range, validation_range = ranges["train"], ranges["validation"]
        if max(train_range[0], validation_range[0]) < min(sum(train_range), sum(validation_range)):
            raise ValueError(f"train and validation source ranges overlap for {corpus_id}")
        if smoke:
            # Enough source rows for each corpus to retain >=4 usable roots,
            # which exercises per-corpus AUC rather than only loss/cosine.
            ranges = {"train": (0, 6), "validation": (16, 16)}
        documents = stream_document_ranges(
            local["dataset_name"], local.get("dataset_config"), local["dataset_revision"], ranges)
        for role in ("train", "validation"):
            start, count = ranges[role]
            prepare_split(local, teacher, role, start, count,
                          Path(cfg["cache_dir"]) / corpus_id / f"{role}.pt",
                          documents[role], corpus_id=corpus_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.smoke and "corpora" not in cfg:
        cfg.update(train_documents=32, validation_documents=16, test_documents=16,
                   train_start=0, validation_start=100, test_start=200)
    teacher = FrozenTeacher(cfg["model_name"], cfg.get("model_revision"), a.device)
    if "corpora" in cfg:
        _prepare_multicorpus(cfg, teacher, a.smoke)
        return
    # Legacy single-corpus preparation retains its original three-role API.
    # Multi-corpus configs take the guarded branch above and never request test.
    splits = ("train", "validation", "test")
    documents = stream_document_ranges(
        cfg["dataset_name"], cfg["dataset_config"], cfg.get("dataset_revision"),
        {name: (cfg[f"{name}_start"], cfg[f"{name}_documents"]) for name in splits})
    for split in splits:
        prepare_split(cfg, teacher, split, cfg[f"{split}_start"],
                      cfg[f"{split}_documents"], Path(cfg["cache_dir"]) / f"{split}.pt",
                      documents[split])


if __name__ == "__main__":
    main()
