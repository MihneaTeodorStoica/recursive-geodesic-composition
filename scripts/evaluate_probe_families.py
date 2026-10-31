#!/usr/bin/env python3
"""Matched TRAIN-only bounded, unbounded and direct probe families."""
import torch.nn.functional as F
from review_common import *
from recursive_geodesic_composition.extended_controls import FlexibleProbe, slot_scores
from recursive_geodesic_composition.safe_baselines import baseline_parent


def main():
    cache=ROOT/"cache_review_train_v1"
    data_manifest=json.loads((cache/"RESULTS.json").read_text());verify_protocol(cache)
    require_hash(cache/"rows.pt",data_manifest["rows_sha256"])
    root,cfg,device,cal,baseline,_,published=begin("probe_families",__file__,extra=[cache/"RESULTS.json"])
    train=torch.load(cache/"rows.pt",weights_only=True,map_location="cpu")
    children=torch.stack([r["leaves"] for r in train]);masses=torch.stack([r["lengths"] for r in train])
    fit=np.array([r["role"]=="fit" for r in train]);val=~fit
    assert fit.sum()==4*cfg["train"]["fit_per_corpus"] and val.sum()==4*cfg["train"]["validation_per_corpus"]
    labels=np.array([r["corpus"] for r in train])
    centroid=children[fit].mean(0)
    parents={"weighted":baseline_parent(children,masses,"token_weighted_mean"),"unweighted":baseline_parent(children,masses,"unweighted_mean")}
    checkpoints=checkpoint_entries(baseline)
    with torch.inference_mode():
        for entry in checkpoints:
            model,cp=build_model(ROOT/entry["path"],device);del cp
            parents[f"learned_{entry['seed']}"]=torch.cat([model.merge(x.to(device),n.to(device)).cpu() for x,n in zip(children.split(64),masses.split(64))])
            del model
    atomic_tensor(root/"train_features.pt",{"parents":parents,"children":children,"masses":masses,"fit":torch.from_numpy(fit),"centroid":centroid})
    atomic_json(root/"TRAIN_MANIFEST.json",{"preparation":data_manifest,"protocol":json.loads((cache/"PROTOCOL.json").read_text()),
        "documents":[{k:r[k] for k in ("id","corpus","source_index","role","token_sha256")} for r in train],
        "features_sha256":sha256(root/"train_features.pt"),"test_data_used_for_fitting":False})
    section=cfg["probes"];models={};training={};schedules={}
    for pseed in section["seeds"]:
        generator=torch.Generator().manual_seed(pseed+100)
        per_corpus=[np.flatnonzero(fit & (labels==c)) for c in published["corpus_order"]]
        schedule=torch.stack([torch.cat([torch.tensor(idx)[torch.randint(len(idx),(32,),generator=generator)] for idx in per_corpus]) for _ in range(section["updates"])])
        schedules[str(pseed)]=stable_hash(schedule.tolist())
        for family in section["families"]:
            for method,features in parents.items():
                torch.manual_seed(pseed)
                probe=FlexibleProbe(768,section["hidden"],family,centroid).to(device)
                initial=stable_hash({k:sha256_tensor(v) for k,v in probe.state_dict().items()})
                optimizer=torch.optim.AdamW(probe.parameters(),lr=section["learning_rate"],weight_decay=section["weight_decay"])
                x,y=features.to(device).clone(),children.to(device).clone()
                losses=[]
                for indexes in schedule:
                    index=indexes.to(device);pred=probe(x[index])
                    loss=1-F.cosine_similarity(pred,y[index],dim=-1).mean()+section["mse_weight"]*F.mse_loss(pred,y[index])
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(probe.parameters(),section["clip_norm"],error_if_nonfinite=True);optimizer.step()
                    losses.append(float(loss.detach()))
                probe.eval()
                with torch.inference_mode():
                    validation=F.cosine_similarity(probe(x[val]),y[val],dim=-1).mean().item()
                key=f"{family}_{pseed}_{method}"
                path=root/"probes"/f"{key}.pt"
                atomic_tensor(path,{"state_dict":probe.cpu().state_dict(),"family":family,"probe_seed":pseed,"representation":method})
                models[key]={"path":str(path.relative_to(root)),"sha256":sha256(path)}
                training[key]={"initial_state_sha256":initial,"schedule_sha256":schedules[str(pseed)],"losses":losses,"validation_cosine":validation,
                               "selection":"fixed_step_400_no_validation_selection","parameters":sum(p.numel() for p in probe.parameters())}
                del probe,optimizer,x,y
                print(f"Frozen {key}; no test scoring yet",flush=True)
    atomic_json(root/"FROZEN_PROBES.json",{"models":models,"training":training,"train_manifest_sha256":sha256(root/"TRAIN_MANIFEST.json")})
    old,rows,_=exact_inputs(cal,baseline,published)
    assert set(old["groups"]).isdisjoint(r["id"] for r in train)
    test_parents={"weighted":old["parents"]["weighted"],"unweighted":old["parents"]["unweighted"]} | {f"learned_{s}":old["parents"][str(s)] for s in cfg["seeds"]}
    pred_scores,outputs,replay,geometry={},{},{},{}
    with torch.inference_mode():
        for key,item in models.items():
            require_hash(root/item["path"],item["sha256"])
            state=torch.load(root/item["path"],weights_only=True,map_location="cpu")
            probe=FlexibleProbe(768,section["hidden"],state["family"],centroid).to(device)
            probe.load_state_dict(state["state_dict"]);probe.eval()
            p=test_parents[state["representation"]]
            pred=torch.cat([probe(v.to(device)).cpu() for v in p.split(64)])
            validate_vectors(pred.reshape(-1,768).numpy(),unit=True)
            alt=torch.cat([probe(v.to(device)).cpu() for v in p.split(17)])
            error=float((pred-alt).abs().max());assert error<=cfg["replay_tolerance"]
            replay[key]=error
            ordered,shuffled=slot_scores(pred,old["children"])
            copy,_=slot_scores(p[:,None].expand(-1,4,-1),old["children"])
            pred_scores[key]={"ordered":ordered.numpy(),"order_gap":(ordered-shuffled).numpy(),"copy":copy.numpy(),"gain":(ordered-copy).numpy()}
            angle=F.cosine_similarity(pred.double(),p.double()[:,None],dim=-1).clamp(-1,1).acos()
            geometry[key]={"maximum_degrees":float(angle.max()*180/np.pi),"fraction_beyond_old_cap":float((angle>np.arcsin(.25)+1e-6).double().mean())}
            outputs[key]=pred;del probe
    centroid_score,centroid_shuffle=slot_scores(centroid[None].expand(606,-1,-1),old["children"])
    paired={}
    for seed in cfg["seeds"]:
        for pseed in section["seeds"]:
            cols={"train_slot_centroid":centroid_score.numpy(),"centroid_order_gap":(centroid_score-centroid_shuffle).numpy()}
            for family in section["families"]:
                for method,tag in (("learned",f"learned_{seed}"),("weighted","weighted"),("unweighted","unweighted")):
                    for metric,values in pred_scores[f"{family}_{pseed}_{tag}"].items(): cols[f"{family}_{method}_{metric}"]=values
                for pool in ("weighted","unweighted"):
                    cols[f"{family}_learned_delta_{pool}"]=cols[f"{family}_learned_ordered"]-cols[f"{family}_{pool}_ordered"]
            for family in ("unbounded","direct"):
                cols[f"{family}_delta_bounded"]=cols[f"{family}_learned_ordered"]-cols["bounded_learned_ordered"]
            paired[f"{seed}_{pseed}"]=cols
    atomic_tensor(root/"predictions.pt",outputs)
    summaries=save_scores(root,paired,old["groups"],old["labels"],cfg,200)
    for e in checkpoints: require_hash(ROOT/e["path"],e["sha256"])
    seal(root,{"documents":606,"train_documents":int(fit.sum()),"validation_documents":int(val.sum()),"probes":len(models),"families":section["families"],
        "summaries":summaries,"geometry":geometry,"independent_replay_errors":replay,"frozen_probes_sha256":sha256(root/"FROZEN_PROBES.json"),
        "scores_sha256":sha256(root/"SCORES.json"),"prediction_sha256":sha256(root/"predictions.pt"),"merger_updates":0,"probe_updates_per_fit":400})
    p=summaries["mean"]["pooled"];table=[]
    for family in section["families"]:
        for metric in ("learned_ordered","learned_gain","learned_delta_weighted","learned_delta_unweighted"):
            v=p[f"{family}_{metric}"];table.append((f"{family}: {metric.replace('_',' ')}",v["mean"],v["ci95"]))
    report_table(root,"Matched probe-family comparison",table,
        "All probes are new TRAIN-only fits with equal budgets and schedules. Bounded and unbounded residual probes share zero-last-layer initialization; the non-residual direct MLP starts at TRAIN slot centroids. Scores average three merger checkpoints and three probe initializations within each document; pooling fits are reused across checkpoint pairings. Intervals condition on these fits, not hypothetical training seeds. No validation or test selection is performed.","tab:probe-families")


def sha256_tensor(tensor):
    import hashlib
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


if __name__ == "__main__": main()
