#!/usr/bin/env python3
"""Matched attached/detached decoder-parent gradient paths in NEW TRAIN prefixes."""
import copy
import sys
import torch.nn.functional as F
from review_common import *
from evaluate_continuation_comparison import artifacts
from evaluate_safe_baselines import padded_batches, attack_result
from recursive_geodesic_composition.train import (step_loss_batch, _set_encoder_trainable,
    _optimizer_parameter_groups, _set_optimizer_base_learning_rate, _learning_rate_at_step,
    _effective_training_weights, _autoencoder_objective_multiplier)
from recursive_geodesic_composition.extended_controls import slot_scores
from recursive_geodesic_composition.continuation_comparison import paired_auc
sys.path.insert(0,str(ROOT/"scripts/final_v6"))
from freeze_training_and_probes import initial_model


def train_step(model,opt,batch,cfg,device,step,detach):
    frozen=step<=300
    multiplier=_autoencoder_objective_multiplier(step,300,100)
    w=_effective_training_weights(cfg,True,frozen,multiplier)
    lr=_learning_rate_at_step(cfg["learning_rate"],cfg["min_learning_rate"],400,50,step-1)
    _set_optimizer_base_learning_rate(opt,lr)
    loss,parts=step_loss_batch(model,batch,True,device,w["reconstruction"],w["cycle"],
        w["distribution_mean"],w["distribution_variance"],8,teacher_weight=w["teacher"],
        auxiliary_weight_multiplier=0. if frozen else multiplier,
        importance_weight_multiplier=1. if frozen else multiplier,
        detach_decoder_parent=frozen or detach)
    if not torch.isfinite(loss): raise FloatingPointError("nonfinite training loss")
    opt.zero_grad(set_to_none=True);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
    return {"step":step,"loss":float(loss.detach()),"gradient_norm":float(norm),"learning_rate":lr,"parts":{k:float(v) for k,v in parts.items()}}


def main():
    cache=ROOT/"cache_review_train_v1";verify_protocol(cache)
    prep=json.loads((cache/"RESULTS.json").read_text());require_hash(cache/"rows.pt",prep["rows_sha256"])
    extra=[cache/"RESULTS.json",ROOT/"src/recursive_geodesic_composition/train.py",ROOT/"scripts/final_v6/freeze_training_and_probes.py"]
    root,cfg,device,cal,baseline,official,published=begin("decoder_gradient",__file__,extra=extra)
    rows=torch.load(cache/"rows.pt",weights_only=True,map_location="cpu")
    rows=[r|{"targets":[r["teacher_target"]]} for r in rows if r["role"]=="fit"]
    assert len(rows)==4*cfg["train"]["fit_per_corpus"]
    entries=artifacts(baseline);fitted={};training={};warm_ids={}
    for seed in cfg["seeds"]:
        original=yaml.safe_load((ROOT/f"configs/final_v6/seed{seed}.yaml").read_text())
        source=entries[str(seed)]["source"]
        original["initialize_from_encoder_checkpoint"]=source["path"]
        original["initialize_from_encoder_checkpoint_sha256"]=source["sha256"]
        model,info=initial_model(original);model=model.to(device).train()
        _set_encoder_trainable(model,False)
        opt=torch.optim.AdamW(_optimizer_parameter_groups(model,.02),lr=original["learning_rate"],weight_decay=original["weight_decay"])
        generator=torch.Generator().manual_seed(cfg["ablation"]["seed"]+seed)
        by_corpus=[[i for i,r in enumerate(rows) if r["corpus"]==c] for c in published["corpus_order"]]
        schedule=[[group[j] for group in by_corpus for j in torch.randint(len(group),(2,),generator=generator).tolist()] for _ in range(400)]
        history=[]
        for step in range(1,301): history.append(train_step(model,opt,[rows[i] for i in schedule[step-1]],original,device,step,True))
        warm={"state_dict":copy.deepcopy(model.state_dict()),"optimizer":copy.deepcopy(opt.state_dict()),
              "cpu_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state(),"initialization":info,"schedule":schedule}
        atomic_tensor(root/f"warm_seed{seed}.pt",warm)
        warm_ids[str(seed)]=sha256(root/f"warm_seed{seed}.pt")
        for arm in cfg["ablation"]["arms"]:
            model.load_state_dict(warm["state_dict"],strict=True);opt.load_state_dict(copy.deepcopy(warm["optimizer"]))
            _set_encoder_trainable(model,True);model.train()
            torch.set_rng_state(warm["cpu_rng"]);torch.cuda.set_rng_state(warm["cuda_rng"])
            arm_history=[]
            for step in range(301,401):
                arm_history.append(train_step(model,opt,[rows[i] for i in schedule[step-1]],original,device,step,arm=="detached"))
            path=root/f"{arm}_seed{seed}.pt"
            atomic_tensor(path,{"state_dict":model.state_dict(),"config":original,"seed":seed,"variant":"autoencoder","selected_step":400,
                               "analysis_role":"new_gradient_intervention_not_original_clean_v6"})
            fitted[f"{arm}_{seed}"]={"path":str(path.relative_to(root)),"sha256":sha256(path),"shared_warm_sha256":warm_ids[str(seed)]}
            training[f"{arm}_{seed}"]={"warm_history":history,"joint_history":arm_history,"schedule_sha256":stable_hash(schedule),
                                      "decoder_parent_detached_in_joint":arm=="detached"}
            print(f"Frozen intervention {arm}/{seed}",flush=True)
        assert training[f"attached_{seed}"]["joint_history"][0]["loss"] == training[f"detached_{seed}"]["joint_history"][0]["loss"], "joint arms did not start at the same forward computation"
        del model,opt,warm;torch.cuda.empty_cache()
    atomic_json(root/"FROZEN_MODELS.json",{"fitted":fitted,"training":training,"train_preparation":prep,
        "shared_warm_checkpoints":warm_ids,"no_validation_or_test_selection":True})
    # Evaluation starts only after all six final models have been fixed.
    safe,_=load_safe_cache(baseline,published);groups=[r["id"] for r in safe];labels=[r["corpus"] for r in safe]
    assert set(groups).isdisjoint(r["id"] for r in rows)
    target=torch.stack([r["teacher_target"] for r in safe]);exact=np.array([i for i,r in enumerate(safe) if len(r["lengths"])==4])
    children=torch.stack([safe[i]["leaves"] for i in exact]);batches,indices=padded_batches(safe,device,64)
    outputs={};replay={};scores={};reconstruction={}
    with torch.inference_mode():
        for seed in cfg["seeds"]:
            cols,rec={},{}
            for arm in cfg["ablation"]["arms"]:
                entry=fitted[f"{arm}_{seed}"];require_hash(root/entry["path"],entry["sha256"])
                model,cp=build_model(root/entry["path"],device);del cp
                values=torch.empty(4000,768)
                for (x,n),ids in zip(batches,indices): values[ids]=model.merge(x,n).cpu()
                alternate=[]
                for start in range(0,4000,17):
                    part=safe[start:start+17];x=torch.zeros(len(part),4,768,device=device);n=torch.zeros(len(part),4,device=device)
                    for i,row in enumerate(part):
                        k=len(row["lengths"]);x[i,:k]=row["leaves"].to(device);n[i,:k]=row["lengths"].to(device)
                    alternate.append(model.merge(x,n).cpu())
                err=float((torch.cat(alternate)-values).abs().max());assert err<=cfg["replay_tolerance"]
                replay[f"{arm}_{seed}"]=err
                pred=torch.cat([model.decode(x.to(device)).cpu() for x in values[exact].split(64)])
                ordered,shuffled=slot_scores(pred,children);copy_score,_=slot_scores(values[exact,None].expand(-1,4,-1),children)
                for name,v in (("ordered",ordered),("copy",copy_score),("gain",ordered-copy_score),("order_gap",ordered-shuffled)):
                    rec[f"{arm}_{name}"]=v.numpy()
                outputs[f"{arm}_{seed}"]=values
                cols[arm]=cosine_rows(values.numpy(),target.numpy())
                del model
            cols["attached_delta_detached"]=cols["attached"]-cols["detached"]
            for metric in ("ordered","copy","gain","order_gap"):
                rec[f"{metric}_delta"]=rec[f"attached_{metric}"]-rec[f"detached_{metric}"]
            scores[str(seed)],reconstruction[str(seed)]=cols,rec
    atomic_tensor(root/"vectors.pt",{"parents":outputs,"targets":target,"groups":groups,"labels":labels})
    atomic_json(root/"INFERENCE_MANIFEST.json",{"config_sha256":sha256(CONFIG),"frozen_models_sha256":sha256(root/"FROZEN_MODELS.json"),"vectors_sha256":sha256(root/"vectors.pt")})
    teacher_summary=save_scores(root,scores,groups,labels,cfg,400)
    # Preserve separate populations rather than overwriting the all-prefix file.
    rec_root=root/"reconstruction";rec_root.mkdir()
    rec_summary=save_scores(rec_root,reconstruction,[groups[i] for i in exact],[labels[i] for i in exact],cfg,400)
    adapter=baseline|{"output_dir":str(root.relative_to(ROOT)),"config_sha256":sha256(CONFIG)}
    scopes={"pooled":np.ones(4000,bool)}|{c:np.array(labels)==c for c in published["corpus_order"]}
    auc={}
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        for i,(scope,mask) in enumerate(scopes.items()):
            bundles={"attached":[],"detached":[]};families={}
            for seed in cfg["seeds"]:
                families[str(seed)]={}
                for arm in cfg["ablation"]["arms"]:
                    result=attack_result(f"{arm}_seed{seed}",scope,target.numpy()[mask],outputs[f"{arm}_{seed}"].numpy()[mask],np.asarray(groups)[mask],adapter,official,published,i)
                    selected=result["metrics"]["strongest_attack"];families[str(seed)][arm]=selected
                    bundles[arm].append(result["bundles"][selected])
            # Generic function's source/final labels mean detached/attached here.
            auc[scope]=paired_auc(bundles["detached"],bundles["attached"],2000,cfg["bootstrap"]["seed"]+410+i)|{"labels":{"source":"detached","final":"attached"},"families":families}
    for pair in entries.values():
        for e in pair.values(): require_hash(ROOT/e["path"],e["sha256"])
    seal(root,{"teacher":teacher_summary,"reconstruction":rec_summary,"source_detection":auc,
               "independent_replay_errors":replay,"training_models":6,"teacher_documents":4000,"reconstruction_documents":606,
               "new_regime_not_historical_reproduction":True,"scores_sha256":sha256(root/"SCORES.json"),
               "frozen_models_sha256":sha256(root/"FROZEN_MODELS.json")})
    t=teacher_summary["mean"]["pooled"]["attached_delta_detached"];p=rec_summary["mean"]["pooled"]
    a=auc["pooled"]["three_seed_mean"]["final_minus_source"]
    table=[("Teacher: attached minus detached",t["mean"],t["ci95"]),("AUC: attached minus detached",a["mean"],a["ci95"])]
    for k in ("ordered_delta","gain_delta","order_gap_delta"):
        c=p[k];table.append((k.replace("_"," "),c["mean"],c["ci95"]))
    report_table(root,"Decoder-parent gradient intervention",table,
        "New 640-document exact-four TRAIN regime. Each checkpoint pair shares the identical 300-update warm model, optimizer, RNG and all 100 joint minibatches. The only joint-stage switch is decoder-parent detachment; original teacher, distribution and encoder regularizers remain active. Both arms continue fitting the decoder. This is not a recreation of historical full-tree clean-v6 training, not training from scratch, and not removal of all decoder computations. Fixed final step, no validation selection.","tab:decoder-gradient")


if __name__ == "__main__": main()
