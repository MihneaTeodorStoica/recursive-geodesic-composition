#!/usr/bin/env python3
"""Independent scalar/metric audits of all completed additional review studies."""
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits
from review_common import ROOT, verify_protocol, require_hash, sha256, atomic_json, hashes, summarize
from recursive_geodesic_composition.continuation_comparison import paired_auc
from recursive_geodesic_composition.extended_controls import cluster_bootstrap


def scalar_audit(root, summaries, folder=None):
    cfg=verify_protocol(root)["configuration"]
    docs=json.loads(((folder or root)/"SCORES.json").read_text())
    offset={"input_order_v1":0,"tree_families_v1":100,"probe_families_v1":200,"decoder_gradient_v1":400}[root.name]
    for seed,cols in docs["scores"].items():
        assert summarize(cols,docs["labels"],cfg,offset)==summaries[seed]
    if root.name=="probe_families_v1":
        order=[f"{s}_{p}" for s in cfg["seeds"] for p in cfg["probes"]["seeds"]]
    else: order=list(map(str,cfg["seeds"]))
    for key,values in docs["scores"]["mean"].items():
        np.testing.assert_array_equal(np.mean([docs["scores"][s][key] for s in order],axis=0),values)
    assert len(docs["groups"])==len(set(docs["groups"]))
    return docs


def normalized(x):
    v=np.asarray(x,dtype=np.float64)
    return v/np.linalg.norm(v,axis=-1,keepdims=True)


def order_audit(root,r):
    docs=scalar_audit(root,r["summaries"])
    payload=torch.load(root/"vectors.pt",weights_only=True,map_location="cpu")
    children=normalized(payload["children"].numpy());perms=payload["permutations"].numpy()
    for seed,modes in payload["outputs"].items():
        for mode,output in modes.items():
            for family in ("decoder","probe"):
                # Different implementation and reduction order, one document at a time.
                follow,identity,fixed=[],[],[]
                for i,pred in enumerate(output[family]):
                    p=normalized(pred.numpy());target=children[i][perms]
                    ordered=np.sum(p*target,axis=-1).mean(-1)
                    shuffled=np.sum(p.mean(1)*target.mean(1),axis=-1)
                    follow.append((ordered-shuffled).mean());identity.append(ordered[0]-shuffled[0])
                    fixed.append(np.mean([np.sum(p[0]*children[i][perm],axis=-1).mean() for perm in perms])-np.sum(p[0].mean(0)*children[i].mean(0)))
                for metric,value in (("following_gap",follow),("identity_gap",identity),("fixed_parent_gap",fixed)):
                    np.testing.assert_allclose(value,docs["scores"][seed][f"{mode}_{family}_{metric}"],rtol=0,atol=1e-14)
                assert np.max(np.abs(fixed))<1e-14
    return {"all_permutation_statistics_independently_recomputed":True,"fixed_parent_cancellation":True}


def tree_audit(root,r):
    docs=scalar_audit(root,r["summaries"])
    payload=torch.load(root/"vectors.pt",weights_only=True,map_location="cpu")
    target=F.normalize(payload["targets"].double(),dim=-1)
    for seed,methods in payload["outputs"].items():
        for key,values in methods.items():
            score=(F.normalize(values.double(),dim=-1)*target).sum(-1).numpy()
            np.testing.assert_allclose(score,docs["scores"][seed][key],atol=1e-14,rtol=0)
            torch.testing.assert_close(values.norm(dim=-1),torch.ones(len(values)),atol=2e-5,rtol=0)
    for size,(leaves,masses) in payload["inputs"].items():
        assert all(int(n.sum())==int(payload["inputs"][1024][1][i].sum()) for i,n in enumerate(masses))
        for arity in (2,4):
            counts=r["accounting"][f"{size}_k{arity}"]
            for i,n in enumerate(masses):
                k=len(n);depth=nodes=0
                while k>1:
                    nodes+=k//arity+int(k%arity>1);k=(k+arity-1)//arity;depth+=1
                assert depth==counts["depths"][i] and nodes==counts["merges"][i]
    return {"double_precision_teacher_cosines":True,"all_depth_merge_and_mass_counts":True}


def probe_audit(root,r):
    docs=scalar_audit(root,r["summaries"])
    manifest=json.loads((root/"FROZEN_PROBES.json").read_text());train=json.loads((root/"TRAIN_MANIFEST.json").read_text())
    data=train["documents"];ids={row["id"] for row in data}
    assert ids.isdisjoint(docs["groups"])
    assert len(data)==len(ids)==len({row["token_sha256"] for row in data})==800
    assert sum(row["role"]=="fit" for row in data)==640
    assert all(140000<=row["source_index"]<152000 for row in data)
    initial={};schedules={}
    for key,cell in manifest["training"].items():
        family,pseed,method=key.split("_",2)
        initial.setdefault((family,pseed),set()).add(cell["initial_state_sha256"])
        schedules.setdefault(pseed,set()).add(cell["schedule_sha256"])
        assert len(cell["losses"])==400 and cell["selection"]=="fixed_step_400_no_validation_selection"
        assert np.isfinite(cell["losses"]).all()
    assert all(len(x)==1 for x in list(initial.values())+list(schedules.values()))
    for item in manifest["models"].values(): require_hash(root/item["path"],item["sha256"])
    pred=torch.load(root/"predictions.pt",weights_only=True,map_location="cpu")
    old=torch.load(ROOT/"results/reconstruction_calibration_v1/vectors.pt",weights_only=True,map_location="cpu")
    for key,value in pred.items():
        family,pseed,method=key.split("_",2)
        values=(F.normalize(value.double(),dim=-1)*F.normalize(old["children"].double(),dim=-1)).sum(-1).mean(-1).numpy()
        seeds=[method.split("_")[1]] if method.startswith("learned_") else list(map(str,verify_protocol(root)["configuration"]["seeds"]))
        tag="learned" if method.startswith("learned_") else method
        for seed in seeds:
            np.testing.assert_allclose(values,docs["scores"][f"{seed}_{pseed}"][f"{family}_{tag}_ordered"],atol=1e-14,rtol=0)
    return {"train_validation_test_roles_and_exact_hashes_disjoint":True,"initialization_and_schedules_matched_across_representations":True,
            "all_45_probe_predictions_rescored_independently":True,"no_validation_or_test_selection":True}


def gradient_audit(root,r):
    scalar_audit(root,r["teacher"]);scalar_audit(root,r["reconstruction"],root/"reconstruction")
    frozen=json.loads((root/"FROZEN_MODELS.json").read_text());cfg=verify_protocol(root)["configuration"]
    for seed in cfg["seeds"]:
        a,b=(frozen["training"][f"{arm}_{seed}"] for arm in ("attached","detached"))
        assert a["warm_history"]==b["warm_history"] and a["schedule_sha256"]==b["schedule_sha256"]
        assert a["joint_history"][0]["loss"]==b["joint_history"][0]["loss"]
        assert not a["decoder_parent_detached_in_joint"] and b["decoder_parent_detached_in_joint"]
        for arm in ("attached","detached"):
            item=frozen["fitted"][f"{arm}_{seed}"];require_hash(root/item["path"],item["sha256"])
            assert item["shared_warm_sha256"]==frozen["shared_warm_checkpoints"][str(seed)]
        assert [x["step"] for x in a["joint_history"]]==list(range(301,401))
    for i,scope in enumerate(["pooled","fineweb_edu","fineweb","openwebtext","wikipedia_en_20231101"]):
        bundles={"attached":[],"detached":[]}
        for seed in cfg["seeds"]:
            pair={arm:json.loads((root/f"attacks/{arm}_seed{seed}__{scope}.json").read_text()) for arm in bundles}
            for field in ("classifier_train_groups","classifier_holdout_groups"): assert pair["attached"][field]==pair["detached"][field]
            for arm,cell in pair.items(): bundles[arm].append(cell["bundles"][cell["metrics"]["strongest_attack"]])
        got=paired_auc(bundles["detached"],bundles["attached"],2000,cfg["bootstrap"]["seed"]+410+i)
        assert got=={k:v for k,v in r["source_detection"][scope].items() if k not in ("labels","families")}
    return {"identical_joint_starts_and_forward_losses":True,"only_configured_gradient_routing_differs":True,"all_paired_auc_intervals_reproduced":True}


def manual_metrics(similarity, positives):
    output={k:[] for k in ("ndcg_at_10","recall_at_1","recall_at_10","mrr_at_10")}
    for row,positive in zip(similarity,positives):
        rank=np.argsort(-row,kind="stable");positions=np.flatnonzero(np.isin(rank,positive));hits=positions[positions<10]
        dcg=sum(1/np.log2(int(p)+2) for p in hits)
        ideal=sum(1/np.log2(p+2) for p in range(min(10,len(positive))))
        output["ndcg_at_10"].append(dcg/ideal)
        output["recall_at_1"].append(float(0 in positions)/len(positive))
        output["recall_at_10"].append(len(hits)/len(positive))
        output["mrr_at_10"].append(1/(int(hits[0])+1) if len(hits) else 0.)
    return output


def retrieval_audit(root,r):
    cfg=verify_protocol(root)["configuration"];maximum=0.
    for i,task in enumerate(cfg["retrieval"]["tasks"]):
        folder=root/task;entry=r["tasks"][task]
        require_hash(folder/"vectors.pt",entry["vectors_sha256"]);require_hash(folder/"SCORES.json",entry["scores_sha256"])
        docs=json.loads((folder/"SCORES.json").read_text());p=torch.load(folder/"vectors.pt",weights_only=True,map_location="cpu")
        assert docs["doc_ids"]==sorted(docs["doc_ids"])
        for method,vectors in p["representations"].items():
            q=p["cls_queries"] if method=="context_cls" else p["mean_queries"]
            similarity=normalized(q.numpy())@normalized(vectors.numpy()).T
            # Independent Torch double matrix products, before a separate rank metric implementation.
            actual=(F.normalize(q.double(),dim=-1)@F.normalize(vectors.double(),dim=-1).T).numpy()
            error=float(np.max(abs(actual-similarity)));maximum=max(maximum,error);assert error<1e-14
            metrics=manual_metrics(similarity,docs["relevance"])
            for key,values in metrics.items(): np.testing.assert_allclose(values,docs["scores"][f"{method}/{key}"],atol=1e-14,rtol=0)
        keys=list(docs["scores"]);values=np.column_stack([docs["scores"][k] for k in keys])
        groups=np.asarray(docs["query_clusters"]);lengths=np.asarray(docs["positive_document_max_lengths"])
        scopes={"all":np.ones(len(groups),bool)}
        bounds=[0,*cfg["retrieval"]["length_bins"],np.inf]
        offsets={"all":300+i*10}
        for j,(lo,hi) in enumerate(zip(bounds[:-1],bounds[1:])):
            mask=(lengths>lo)&(lengths<=hi)
            if mask.any():
                key=f"length_{int(lo)+1}_{int(hi) if np.isfinite(hi) else 'plus'}";scopes[key]=mask;offsets[key]=301+i*10+j
        for scope,mask in scopes.items():
            draws=cluster_bootstrap(values[mask],groups[mask],2000,cfg["bootstrap"]["seed"]+offsets[scope])
            for j,key in enumerate(keys):
                cell=entry["summaries"][scope][key]
                np.testing.assert_allclose(np.percentile(draws[:,j],[2.5,97.5]),cell["ci95"],atol=1e-14,rtol=0)
                assert abs(values[mask,j].mean()-cell["mean"])<1e-14
        print(f"Independent retrieval metrics/intervals verified: {task}",flush=True)
    return {"all_retrieval_metrics_independently_recomputed":True,"all_cluster_intervals_recomputed":True,"maximum_similarity_error":maximum}


def main(names):
    functions={"input_order":order_audit,"tree_families":tree_audit,"probe_families":probe_audit,"decoder_gradient":gradient_audit,"longembed_review":retrieval_audit}
    with threadpool_limits(limits=4):
        torch.set_num_threads(4)
        for name in names:
            root=ROOT/f"results/{name}_v1";protocol=verify_protocol(root)
            r=json.loads((root/"RESULTS.json").read_text())
            for path,digest in json.loads((root/"ARTIFACT_HASHES.json").read_text()).items(): require_hash(root/path,digest)
            values=functions[name](root,r)
            checks={k:v for k,v in values.items() if isinstance(v,bool)}
            diagnostics={k:v for k,v in values.items() if not isinstance(v,bool)}
            assert all(checks.values())
            audit={"passed":True,"checks":checks,"diagnostics":diagnostics,"results_sha256":sha256(root/"RESULTS.json"),"audit_sha256":sha256(__file__),
                   "prior_results_unchanged":True,"inference_replay_record":r["independent_replay_errors"]}
            atomic_json(root/"AUDIT.json",audit);hashes(root)
            print(f"AUDIT PASSED: {name}",flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("studies",nargs="*",default=["input_order","tree_families","probe_families","decoder_gradient","longembed_review"])
    main(parser.parse_args().studies)
