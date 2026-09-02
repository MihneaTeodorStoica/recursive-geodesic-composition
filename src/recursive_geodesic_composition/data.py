"""HF streaming ingestion and frozen BGE chunk embedding."""
from __future__ import annotations
from dataclasses import dataclass
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

@dataclass
class Document:
    id: str
    text: str

def stream_documents(name, config, revision=None, limit=None, start=0):
    """Stream one source-index range (the historical public API)."""
    if limit is not None:
        yield from stream_document_ranges(name, config, revision, {"split": (start, limit)})["split"]
        return
    ds = load_dataset(name, config, split="train", streaming=True, revision=revision)
    for i, row in enumerate(ds):
        if i < start:
            continue
        text = row.get("text") or row.get("content") or ""
        if isinstance(text, str) and text.strip():
            yield Document(str(row.get("id", i)), text)


def stream_document_ranges(name, config, revision, ranges):
    """Return requested source-index ranges after making exactly one HF pass.

    ``ranges`` maps a name to ``(start, count)``.  Starts/counts are source
    dataset indices, matching the old ``stream_documents`` behaviour, rather
    than usable-document indices.  This matters for reproducible cached splits.
    """
    pending = {key: (int(start), int(count)) for key, (start, count) in ranges.items()}
    result = {key: [] for key in pending}
    if not pending:
        return result
    last = max(start + count for start, count in pending.values())
    ds = load_dataset(name, config, split="train", streaming=True, revision=revision)
    for i, row in enumerate(ds):
        if i >= last:
            break
        matching = [key for key, (start, count) in pending.items() if start <= i < start + count]
        if not matching:
            continue
        text = row.get("text") or row.get("content") or ""
        if not isinstance(text, str) or not text.strip():
            continue
        document = Document(str(row.get("id", i)), text)
        for key in matching:
            result[key].append(document)
    return result

def tokenize_text(tokenizer, text):
    return tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]


def chunk_token_ids(token_ids, max_tokens=510):
    """Make the original, non-overlapping fixed-size segmentation."""
    return [token_ids[i:i + max_tokens] for i in range(0, len(token_ids), max_tokens)
            if len(token_ids[i:i + max_tokens]) >= 4]


def chunk_text(tokenizer, text, max_tokens=510):
    return chunk_token_ids(tokenize_text(tokenizer, text), max_tokens)

class FrozenTeacher:
    """Frozen embedding teacher with explicit, provenance-visible pooling.

    ``masked_mean`` is the historical V1 behavior and remains the default so
    publication reproduction code is not reinterpreted.  The documented
    GTE-ModernBERT embedding path is ``cls``: L2-normalized
    ``outputs.last_hidden_state[:, 0]`` after normal tokenizer special-token
    handling.
    """

    VALID_POOLING = {"masked_mean", "cls"}

    def __init__(self, name="BAAI/bge-small-en-v1.5", revision=None, device="cuda",
                 pooling="masked_mean", trust_remote_code=None):
        if pooling not in self.VALID_POOLING:
            raise ValueError(f"unknown teacher pooling {pooling!r}; expected one of {sorted(self.VALID_POOLING)}")
        self.name = name
        self.revision = revision
        self.pooling = pooling
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        if trust_remote_code is None:
            trust_remote_code = (name == "Alibaba-NLP/gte-modernbert-base")
        self.trust_remote_code = bool(trust_remote_code)
        self.tokenizer = AutoTokenizer.from_pretrained(
            name, revision=revision, trust_remote_code=self.trust_remote_code)
        self.model = AutoModel.from_pretrained(
            name, revision=revision, trust_remote_code=self.trust_remote_code).to(self.device).eval()
        for p in self.model.parameters(): p.requires_grad_(False)

    def provenance(self):
        return {
            "teacher_id": self.name,
            "teacher_revision": self.revision,
            "teacher_pooling": self.pooling,
            "tokenizer_class": type(self.tokenizer).__name__,
            "model_class": type(self.model).__name__,
            "trust_remote_code": self.trust_remote_code,
            "max_position_embeddings": int(getattr(self.model.config, "max_position_embeddings", 0) or 0),
            "hidden_size": int(getattr(self.model.config, "hidden_size", 0) or 0),
        }

    @torch.inference_mode()
    def embed(self, token_chunks, batch_size=32):
        result = []
        for i in range(0, len(token_chunks), batch_size):
            batch = token_chunks[i:i + batch_size]
            # The teacher has a finite context window. Long-document targets use
            # the deterministic prefix; recursive training still sees every leaf.
            limit = int(getattr(self.model.config, "max_position_embeddings", 512)) - 2
            batch = [tokens[:limit] for tokens in batch]
            if self.pooling == "cls":
                cls_id = getattr(self.tokenizer, "cls_token_id", None)
                sep_id = getattr(self.tokenizer, "sep_token_id", None)
                if cls_id is None or sep_id is None:
                    raise ValueError("CLS pooling requires tokenizer cls_token_id and sep_token_id")
                # Repository chunkers store raw token ids (add_special_tokens=False)
                # for exact token accounting; reconstitute the model-card input.
                batch = [[cls_id, *list(tokens), sep_id] for tokens in batch]
            encoded = self.tokenizer.pad({"input_ids": batch}, padding=True, return_tensors="pt").to(self.device)
            hidden = self.model(**encoded).last_hidden_state
            if self.pooling == "masked_mean":
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            elif self.pooling == "cls":
                pooled = hidden[:, 0]
            else:
                raise AssertionError(f"unvalidated teacher pooling: {self.pooling}")
            result.append(torch.nn.functional.normalize(pooled.float(), dim=-1).cpu())
        return torch.cat(result) if result else torch.empty((0, self.model.config.hidden_size))
