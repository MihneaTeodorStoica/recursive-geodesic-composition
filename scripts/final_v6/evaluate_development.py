"""Evaluate one frozen v6 seed once on fresh safe-direct and full-document development data."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

from recursive_geodesic_composition.deployment_regime_study import fixed_attacks  # noqa: E402
from recursive_geodesic_composition.evaluate_kary_validation import ParentOnlyRetentionProbe, build_model  # noqa: E402
from protocol import (CORPORA, DEVELOPMENT_SCAN, DEV_CACHE, ROOT, SAFE_RANGE,  # noqa: E402
                      TARGET_DOCUMENTS, atomic_json, load_config, sha256_file,
                      stable_hash)
from recursive_geodesic_composition.validation_output_calibration import diagnostics  # noqa: E402
from public_replay import REPLAY_ID, development_freeze, package_file_hash, public_cache_manifest


def load_cache(cfg):
    cache = ROOT / DEV_CACHE; manifest_path = cache / "CACHE_MANIFEST.json"
    if not manifest_path.is_file() or (cache / "COMPLETE").read_text().strip() != f"{sha256_file(manifest_path)}  CACHE_MANIFEST.json":
        raise RuntimeError("atomic development cache incomplete")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("source_scan") != list(DEVELOPMENT_SCAN) or manifest.get("protected_data_accessed") is not False:
        raise ValueError("development manifest boundary changed")
    safe_by, full_by = {}, {}
    for corpus in CORPORA:
        safe_path, index_path = cache / corpus / "safe.pt", cache / corpus / "full_index.json"
        if sha256_file(safe_path) != manifest["artifacts"][f"{corpus}/safe.pt"] or sha256_file(index_path) != manifest["artifacts"][f"{corpus}/full_index.json"]:
            raise ValueError("development cache index hash mismatch")
        safe = torch.load(safe_path, map_location="cpu", weights_only=False)
        index = json.loads(index_path.read_text())["documents"]
        if len(safe) != TARGET_DOCUMENTS or len(index) != TARGET_DOCUMENTS: raise ValueError("development count mismatch")
        for rank, (row, item) in enumerate(zip(safe, index)):
            if row["id"] != item["id"] or row["source_index"] != item["source_index"] or row["selected_rank"] != rank:
                raise ValueError("safe/full document alignment mismatch")
            if not DEVELOPMENT_SCAN[0] <= row["source_index"] < DEVELOPMENT_SCAN[1]: raise ValueError("source escaped development")
            count, tokens = len(row["leaves"]), int(row["lengths"].sum())
            if count not in (2, 3, 4) or not SAFE_RANGE[0] <= tokens <= SAFE_RANGE[1]: raise ValueError("unsafe direct row")
        safe_by[corpus], full_by[corpus] = safe, index
    return safe_by, full_by, {"path": str(manifest_path.relative_to(ROOT)), "sha256": sha256_file(manifest_path),
                              "document_order_sha256": stable_hash([r["id"] for c in CORPORA for r in safe_by[c]])}


@torch.inference_mode()
def safe_extract(model, safe_by, device, batch_size):
    records = [(corpus, row) for corpus in CORPORA for row in safe_by[corpus]]
    roots = [None] * len(records); angles, corrections = [], []
    geo_sat = imp_sat = active_total = 0; dtype = next(model.parameters()).dtype
    for count in (2, 3, 4):
        indexes = [i for i, (_, row) in enumerate(records) if len(row["leaves"]) == count]
        for start in range(0, len(indexes), batch_size):
            part_indexes = indexes[start:start + batch_size]; part = [records[i][1] for i in part_indexes]
            children = torch.stack([r["leaves"].float() for r in part]).to(device=device, dtype=dtype)
            masses = torch.stack([r["lengths"].float() for r in part]).to(device=device, dtype=dtype)
            if count < 4: children = F.pad(children, (0, 0, 0, 4-count)); masses = F.pad(masses, (0, 4-count))
            values = model.merge(children, masses); aux = model._last_merge_aux; active = aux["active_mask"]
            if not torch.isfinite(values).all(): raise FloatingPointError("nonfinite safe root")
            angles.extend(aux["angle"].float().cpu().tolist()); corrections.extend(aux["corrections"][active].abs().float().cpu().tolist())
            geo_sat += int(aux["geodesic_saturated"].sum()); imp_sat += int(((aux["corrections"].abs() >= .95 * model.importance_beta) & active).sum()); active_total += int(active.sum())
            for index, value in zip(part_indexes, values.float().cpu()): roots[index] = value.numpy()
    root_array = np.stack(roots).astype(np.float32)
    return {"roots": root_array, "targets": np.stack([r["teacher_target"].numpy() for _, r in records]).astype(np.float32),
            "groups": np.asarray([r["id"] for _, r in records]), "corpora": np.asarray([c for c, _ in records]),
            "leaf_counts": np.asarray([len(r["leaves"]) for _, r in records]), "records": records,
            "telemetry": {"geodesic_saturation_count": geo_sat, "importance_saturation_count": imp_sat,
                          "importance_active_count": active_total, "angle_maximum": max(angles),
                          "importance_correction_abs_maximum": max(corrections)}}


def scope_safe(data, cfg, attack_cfg):
    scopes = {"pooled": np.arange(len(data["roots"]))}; scopes.update({c: np.flatnonzero(data["corpora"] == c) for c in CORPORA})
    result = {}
    for offset, (scope, indexes) in enumerate(scopes.items()):
        cosine = F.cosine_similarity(torch.from_numpy(data["roots"][indexes]), torch.from_numpy(data["targets"][indexes]), dim=1)
        result[scope] = {"rows": len(indexes), "teacher_cosine": float(cosine.mean()),
                         "attacks": fixed_attacks(data["targets"][indexes], data["roots"][indexes], data["groups"][indexes],
                                                  attack_cfg, int(cfg["bootstrap_replicates"]), int(cfg["bootstrap_seed"]) + offset * 100),
                         "health": diagnostics(data["roots"][indexes], np.linalg.norm(data["roots"][indexes], axis=1), {"collapse": cfg["collapse"]})}
    return result


@torch.inference_mode()
def builtin(model, data, device):
    exact = np.flatnonzero(data["leaf_counts"] == 4); roots = torch.from_numpy(data["roots"][exact]).float()
    children = torch.stack([data["records"][i][1]["leaves"].float() for i in exact]); predictions = []
    for start in range(0, len(roots), 512): predictions.append(model.decode(roots[start:start+512].to(device)).float().cpu())
    values = F.cosine_similarity(torch.cat(predictions), children, dim=-1).mean(-1).numpy(); labels = data["corpora"][exact]
    return {"pooled": {"rows": len(values), "ordered_child_cosine": float(values.mean())}} | {
        corpus: {"rows": int((labels == corpus).sum()), "ordered_child_cosine": float(values[labels == corpus].mean())} for corpus in CORPORA}


def load_probe(path, cfg, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    probe = ParentOnlyRetentionProbe(768, 4, int(cfg["matched_probe"]["hidden"])).to(device)
    probe.load_state_dict(checkpoint["state_dict"], strict=True); probe.eval(); return probe


@torch.inference_mode()
def retention(cfg, freeze, candidate, source, data, device):
    exact = np.flatnonzero(data["leaf_counts"] == 4); rows = [(data["corpora"][i], data["records"][i][1]) for i in exact]
    children = torch.stack([row["leaves"].float() for _, row in rows]); labels = np.asarray([c for c, _ in rows])
    dtype = next(candidate.parameters()).dtype
    def model_roots(model):
        result = []
        for start in range(0, len(rows), 128):
            part = rows[start:start+128]
            result.append(model.merge(torch.stack([r["leaves"].float() for _, r in part]).to(device=device,dtype=dtype),
                                      torch.stack([r["lengths"].float() for _, r in part]).to(device=device,dtype=dtype)).float().cpu())
        return torch.cat(result)
    roots = {"candidate": model_roots(candidate), "source": model_roots(source)}
    scores = {}
    for arm in ("source", "candidate"):
        artifact = freeze["matched_probe"]["probe_artifacts"][arm]
        path = ROOT / artifact["path"]
        if sha256_file(path) != artifact["sha256"]: raise ValueError("frozen probe hash mismatch")
        probe = load_probe(path, cfg, device); predictions = []
        for start in range(0, len(rows), 512): predictions.append(probe(roots[arm][start:start+512].to(device)).cpu())
        values = F.cosine_similarity(torch.cat(predictions), children, dim=-1).mean(-1).numpy()
        scores[arm] = {"pooled": float(values.mean())} | {c: float(values[labels == c].mean()) for c in CORPORA}
    floors = freeze["matched_probe"]["frozen_floors"]
    passes = {scope: bool(scores["candidate"][scope] >= floors[scope]) for scope in floors}
    return {"fit_role": "fresh_v6_train", "development_fit_updates": 0, "scores": scores,
            "frozen_floors": floors, "floor_pass": passes, "all_floors_pass": all(passes.values())}


class Stats:
    def __init__(self): self.values=[]
    def add(self, values): self.values.extend(np.asarray(values).reshape(-1).astype(float).tolist())
    def result(self):
        x=np.asarray(self.values); return {"count": len(x), "mean": float(x.mean()), "maximum": float(x.max()), "minimum": float(x.min())}


@torch.inference_mode()
def reduce_full(model, shard, device, telemetry):
    nodes=shard["leaves"].float(); masses=shard["lengths"].float(); dtype=next(model.parameters()).dtype
    while len(nodes)>1:
        groups=[(s,min(4,len(nodes)-s)) for s in range(0,len(nodes),4)]; parents={}
        merge=[(i,s,w) for i,(s,w) in enumerate(groups) if w>=2]
        for offset in range(0,len(merge),64):
            part=merge[offset:offset+64]
            child=torch.stack([F.pad(nodes[s:s+w],(0,0,0,4-w)) for _,s,w in part]).to(device=device,dtype=dtype)
            weight=torch.stack([F.pad(masses[s:s+w],(0,4-w)) for _,s,w in part]).to(device=device,dtype=dtype)
            roots=model.merge(child,weight); aux=model._last_merge_aux; decoded=model.decode(roots); active=aux["active_mask"]
            if not torch.isfinite(roots).all() or not torch.isfinite(decoded).all(): raise FloatingPointError("nonfinite full recursion")
            telemetry["angles"].add(aux["angle"].float().cpu()); telemetry["decoder"].add(F.cosine_similarity(decoded,child,dim=-1)[active].float().cpu())
            telemetry["geo_sat"] += int(aux["geodesic_saturated"].sum()); telemetry["imp_sat"] += int(((aux["corrections"].abs()>=.95*model.importance_beta)&active).sum()); telemetry["merges"] += len(part)
            for local,(index,s,w) in enumerate(part): parents[index]=(roots[local].float().cpu(),masses[s:s+w].sum())
        next_nodes=[]; next_masses=[]
        for index,(s,w) in enumerate(groups):
            if w==1: next_nodes.append(nodes[s]); next_masses.append(masses[s])
            else: next_nodes.append(parents[index][0]); next_masses.append(parents[index][1])
        nodes=torch.stack(next_nodes); masses=torch.stack(next_masses)
    return nodes[0]


@torch.inference_mode()
def full_diagnostic(model, full_by, cfg, device):
    telemetry={"angles":Stats(),"decoder":Stats(),"geo_sat":0,"imp_sat":0,"merges":0}; roots=[]; labels=[]; started=time.perf_counter()
    for corpus in CORPORA:
        for item in full_by[corpus]:
            path=ROOT/DEV_CACHE/item["shard"]
            if path.stat().st_size != item["shard_bytes"] or sha256_file(path)!=item["shard_sha256"]: raise ValueError("full shard hash mismatch")
            shard=torch.load(path,map_location="cpu",weights_only=False)
            if shard["id"]!=item["id"] or shard["truncated"] or shard["token_cap"] is not None: raise ValueError("full shard invariant changed")
            roots.append(reduce_full(model,shard,device,telemetry).numpy()); labels.append(corpus)
    values=np.stack(roots).astype(np.float32); labels=np.asarray(labels); elapsed=time.perf_counter()-started
    health={"pooled":diagnostics(values,np.linalg.norm(values,axis=1),{"collapse":cfg["collapse"]})}
    health.update({c:diagnostics(values[labels==c],np.linalg.norm(values[labels==c],axis=1),{"collapse":cfg["collapse"]}) for c in CORPORA})
    return {"documents":len(values),"every_available_token":True,"teacher_comparison":False,"auc":False,
            "health":health,"telemetry":{"geodesic":telemetry["angles"].result(),"builtin_decoder_child_cosine":telemetry["decoder"].result(),
                                           "geodesic_saturation_count":telemetry["geo_sat"],"importance_saturation_count":telemetry["imp_sat"],
                                           "internal_merges":telemetry["merges"],"seconds":elapsed}}


def run(config_path, device_name, *, replay=False):
    cfg=load_config(config_path); seed=int(cfg["seed"]); output=ROOT/cfg["output_dir"]
    replay_provenance = None
    if replay:
        config_relative = str(Path(config_path).resolve().relative_to(ROOT))
        config_digest = package_file_hash(ROOT, config_relative)
        package_file_hash(ROOT, cfg["attack_config"])
        freeze, freeze_provenance = development_freeze(ROOT, seed)
        _, cache_release = public_cache_manifest(ROOT, DEV_CACHE)
        for artifact in (freeze["checkpoint"], freeze["source_checkpoint"],
                         *freeze["matched_probe"]["probe_artifacts"].values()):
            if sha256_file(ROOT / artifact["path"]) != artifact["sha256"]:
                raise ValueError("public development replay artifact changed")
        safe_by,full_by,cache_provenance=load_cache(cfg)
        output = ROOT / "replays" / "development" / str(seed)
        if output.exists():
            raise FileExistsError("development replay output is create-only")
        freeze_path = ROOT / freeze_provenance["normalized_freeze_path"]
        replay_provenance = {"contract": REPLAY_ID, "freeze": freeze_provenance,
                             "cache": cache_release, "config_path": config_relative,
                             "config_sha256": config_digest,
                             "historical_results_overwritten": False,
                             "original_collection_ordinal_claimed": False}
    else:
        freeze_path=output/"PRE_DEVELOPMENT_FROZEN.json"; freeze=json.loads(freeze_path.read_text())
    checkpoint_path=ROOT/freeze["checkpoint"]["path"]
    if sha256_file(checkpoint_path)!=freeze["checkpoint"]["sha256"]: raise ValueError("checkpoint changed after freeze")
    if replay and device_name != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for replay but unavailable; request --device cpu explicitly")
    result_path=output/"DEVELOPMENT_EVALUATION.json"; started=output/"DEVELOPMENT_EVALUATION_STARTED"
    if result_path.exists() or started.exists(): raise FileExistsError("development evaluation is one-shot")
    if replay:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()
    fd=os.open(started,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
    with os.fdopen(fd,"w") as h: h.write(f"freeze_sha256={sha256_file(freeze_path)}\n"); h.flush(); os.fsync(h.fileno())
    device=torch.device(device_name if device_name=="cpu" or torch.cuda.is_available() else "cpu")
    if not replay:
        safe_by,full_by,cache_provenance=load_cache(cfg)
    candidate,_=build_model(checkpoint_path,device)
    source_path=ROOT/freeze["source_checkpoint"]["path"]; source,_=build_model(source_path,device)
    data=safe_extract(candidate,safe_by,device,int(cfg["root_batch_size"])); attack_path=ROOT/cfg["attack_config"]
    attack_cfg=yaml.safe_load(attack_path.read_text())
    result={"study":cfg["study"],"seed":seed,"development_only":True,"protected_data_accessed":False,
            "evaluation_ordinal":1,"checkpoint":freeze["checkpoint"],"safe_direct":{
                "scopes":scope_safe(data,cfg,attack_cfg),"telemetry":data["telemetry"],
                "builtin_decoder_exact4":builtin(candidate,data,device),
                "train_fitted_matched_probe":retention(cfg,freeze,candidate,source,data,device)},
            "full_document_diagnostic":full_diagnostic(candidate,full_by,cfg,device),
            "provenance":{"cache":cache_provenance,"freeze_sha256":sha256_file(freeze_path),
                          "attack_config":cfg["attack_config"],"attack_config_sha256":sha256_file(attack_path),
                          "evaluator_sha256":sha256_file(__file__),"maximum_source_index_opened_exclusive":max(v["source_index"] for rows in safe_by.values() for v in rows)+1,
                          "reserve_or_future_opened":False}}
    if replay:
        result.pop("evaluation_ordinal")
        result["public_replay"] = replay_provenance
        result["environment"] = {"python": sys.version, "torch": torch.__version__,
                                 "numpy": np.__version__, "device_requested": device_name,
                                 "device_used": str(device), "cpu_threads": torch.get_num_threads()}
        archived_relative = f"results/clean_v6_development/seeds/{seed}/DEVELOPMENT_EVALUATION.json"
        archived_digest = package_file_hash(ROOT, archived_relative)
        archived = json.loads((ROOT / archived_relative).read_text())
        compared = {}
        for scope, values in result["safe_direct"]["scopes"].items():
            old = archived["safe_direct"]["scopes"][scope]
            for metric, fresh, original in (
                ("teacher_cosine", values["teacher_cosine"], old["teacher_cosine"]),
                ("strongest_attack_auc", values["attacks"]["strongest_orientation_safe_auc"],
                 old["attacks"]["strongest_orientation_safe_auc"]),
            ):
                compared[f"{scope}/{metric}"] = {"fresh": fresh, "archived": original,
                                                "difference": fresh - original}
        result["comparison_to_archived"] = {
            "path": archived_relative, "sha256": archived_digest, "metrics": compared,
            "same_checkpoint_and_cached_tensors": True,
            "bitwise_runtime_or_timing_equivalence_claimed": False,
        }
    atomic_json(result_path,result); (output/"DEVELOPMENT_EVALUATION_COMPLETE").write_text(f"{sha256_file(result_path)}  DEVELOPMENT_EVALUATION.json\n")
    return result_path


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--device",default="cuda")
    parser.add_argument("--replay", action="store_true", help="Use the pinned anonymous cache/freeze in a fresh replay namespace")
    args=parser.parse_args(); print(run(Path(args.config).resolve(),args.device,replay=args.replay))
