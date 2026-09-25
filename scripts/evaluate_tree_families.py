#!/usr/bin/env python3
"""Five ordered binary trees plus teacher-valid segmentation/depth controls."""
import torch.nn.functional as F
from review_common import *
from recursive_geodesic_composition.extended_controls import binary_trees, binary_tree, ragged_reduce
from recursive_geodesic_composition.safe_baselines import baseline_parent


def pools(leaves, masses, method):
    maximum=max(len(x) for x in leaves)
    x=torch.stack([F.pad(v,(0,0,0,maximum-len(v))) for v in leaves])
    n=torch.stack([F.pad(v,(0,maximum-len(v))) for v in masses])
    return baseline_parent(x,n,method)


def main():
    root,cfg,device,cal,baseline,official,published=begin("tree_families",__file__)
    old,rows,_=exact_inputs(cal,baseline,published)
    tokens={r["id"]:r["safe_token_ids"] for r in read_tokens(baseline,published)}
    inputs={1024:([r["leaves"] for r in rows],[r["lengths"] for r in rows])}
    teacher=make_teacher(official,device)
    for size in (512,256):
        leaves,masses=[],[]
        for i,row in enumerate(rows):
            ids=tokens[row["id"]]
            chunks=[ids[j:j+size] for j in range(0,len(ids),size)]
            leaves.append(teacher.embed(chunks,16))
            masses.append(torch.tensor([len(c) for c in chunks],dtype=torch.float32))
            assert sum(len(c) for c in chunks)==int(row["lengths"].sum())
            if (i+1)%100==0: print(f"Segmentation {size}: {i+1}/606",flush=True)
        inputs[size]=(leaves,masses)
    del teacher
    torch.cuda.empty_cache()
    targets=torch.stack([r["teacher_target"] for r in rows]).numpy()
    score,tensors,replay,accounting={}, {}, {}, {}
    trees=binary_trees()
    with torch.inference_mode():
        for entry in checkpoint_entries(baseline):
            seed=str(entry["seed"])
            model,checkpoint=build_model(ROOT/entry["path"],device);del checkpoint
            cols,outputs={},{}
            original=cosine_rows(old["parents"][seed].numpy(),targets)
            for size,(xs,ns) in inputs.items():
                for kind,method in (("weighted","token_weighted_mean"),("unweighted","unweighted_mean")):
                    value=pools(xs,ns,method)
                    outputs[f"{size}_{kind}"]=value
                    cols[f"{size}_{kind}"]=cosine_rows(value.numpy(),targets)
                xd,nd=[x.to(device) for x in xs],[n.to(device) for n in ns]
                for arity in (2,4):
                    key=f"{size}_k{arity}"
                    value,counts=ragged_reduce(model,xd,nd,arity,64)
                    value=value.cpu();validate_vectors(value.numpy(),unit=True)
                    rebuilt,other=ragged_reduce(model,xd[::-1],nd[::-1],arity,17)
                    error=float((rebuilt.flip(0).cpu()-value).abs().max())
                    assert error<=cfg["replay_tolerance"]
                    assert counts["total_masses"]==other["total_masses"][::-1]
                    replay[f"{seed}_{key}"]=error
                    if size==1024 and arity==4:
                        error=float((value-old["parents"][seed]).abs().max())
                        assert error<=cfg["replay_tolerance"]
                        replay[f"{seed}_original_parent"]=error
                    outputs[key]=value;accounting[key]=counts
                    cols[key]=cosine_rows(value.numpy(),targets)
                    cols[f"{key}_delta_weighted"]=cols[key]-cols[f"{size}_weighted"]
                    cols[f"{key}_delta_original"]=cols[key]-original
            x=torch.stack(inputs[1024][0]).to(device);n=torch.stack(inputs[1024][1]).to(device)
            for j,tree in enumerate(trees):
                name=f"tree_{j+1}"
                out=[]
                for start in range(0,606,64):
                    v,m=binary_tree(model,x[start:start+64],n[start:start+64],tree)
                    assert torch.equal(m,n[start:start+64].sum(1));out.append(v.cpu())
                value=torch.cat(out)
                alternate=torch.cat([binary_tree(model,x[s:s+17],n[s:s+17],tree)[0].cpu() for s in range(0,606,17)])
                error=float((value-alternate).abs().max());assert error<=cfg["replay_tolerance"]
                replay[f"{seed}_{name}"]=error
                outputs[name]=value;cols[name]=cosine_rows(value.numpy(),targets)
                cols[f"{name}_delta_original"]=cols[name]-original
                cols[f"{name}_delta_weighted"]=cols[name]-cols["1024_weighted"]
            score[seed],tensors[seed]=cols,outputs
            require_hash(ROOT/entry["path"],entry["sha256"])
            del model;torch.cuda.empty_cache()
            print(f"All tree families complete: {seed}",flush=True)
    atomic_tensor(root/"vectors.pt",{"inputs":inputs,"outputs":tensors,"targets":torch.from_numpy(targets)})
    summaries=save_scores(root,score,old["groups"],old["labels"],cfg,100)
    seal(root,{"documents":606,"trees":trees,"accounting":accounting,"summaries":summaries,
               "independent_replay_errors":replay,"vectors_sha256":sha256(root/"vectors.pt"),"scores_sha256":sha256(root/"SCORES.json"),
               "new_training_updates":0,"new_segmentation_teacher_encodings":True,"same_original_teacher_targets":True})
    table=[];p=summaries["mean"]["pooled"]
    for key in [f"tree_{i+1}_delta_original" for i in range(5)]+[f"{size}_k{k}_delta_weighted" for size in (1024,512,256) for k in (2,4)]:
        v=p[key];table.append((key.replace("_"," "),v["mean"],v["ci95"]))
    report_table(root,"Tree and segmentation robustness",table,
      "Same 606 texts and direct teacher targets. Five binary trees preserve original leaf order. Segmentation at 512/256 tokens changes leaf representations as well as reduction depth; 256 is below the configured training chunk-size range. No depth-only causal interpretation is made. All methods within a segmentation share the same new leaves; flat pools normalize only once.","tab:tree-families")


if __name__ == "__main__": main()
