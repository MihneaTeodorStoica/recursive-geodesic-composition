#!/usr/bin/env python3
"""Complete the fixed LongEmbed protocol, preserving every completed task byte.

Recovery changes only physical-line JSONL parsing and resumable orchestration.
"""
import time
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, HfApi
from threadpoolctl import threadpool_limits
from recursive_geodesic_composition.data import tokenize_text
from recursive_geodesic_composition.extended_controls import ragged_reduce, retrieval_metrics, positive_components, cluster_bootstrap
from recursive_geodesic_composition.train import batched_balanced_roots
from review_common import *
from recursive_geodesic_composition.stable_pooling import stable_masked_mean


from evaluate_longembed_review import mean_encode, cls_encode, grouped_summary

def main():
    root=ROOT/"results/longembed_review_v1"
    if root.exists():
        verify_protocol(root)
        if (root/"RESULTS.json").exists() or (root/"RESUME_PROTOCOL.json").exists():
            raise FileExistsError("completed/resumed study is create-only")
        cfg,_,baseline,official,_=context();device=configure_device(cfg,"cuda")
    else:
        root,cfg,device,_,baseline,official,_=begin("longembed_review",__file__,extra=[ROOT/"scripts/evaluate_longembed_review.py",ROOT/"src/recursive_geodesic_composition/train.py",ROOT/"src/recursive_geodesic_composition/stable_pooling.py"])
    preserved={str(p.relative_to(root)):sha256(p) for p in sorted(root.glob("*/*")) if p.is_file()}
    atomic_json(root/"RESUME_PROTOCOL.json",{"reason":"physical-line JSONL parsing; Unicode separators are valid inside JSON strings", "implementation_sha256":sha256(__file__),
        "preserved_task_files":preserved,"no_recomputation_or_replacement_of_completed_scores":True,"original_protocol_sha256":sha256(root/"PROTOCOL.json")})
    section=cfg["retrieval"];checkpoints=checkpoint_entries(baseline)
    teacher=make_teacher(official,device)
    tasks={};all_metadata={};replay={};all_files={};table=[]
    api=HfApi()
    remote={x.path:x for x in api.list_repo_tree(section["dataset"],repo_type="dataset",revision=section["revision"],recursive=True) if hasattr(x,"size")}
    for task_index,task in enumerate(section["tasks"]):
        raw={}
        for split in ("corpus","queries","qrels"):
            file=f"{task}/{split}.jsonl"
            path=Path(hf_hub_download(section["dataset"],file,repo_type="dataset",revision=section["revision"],local_dir=ROOT/"artifacts/review_longembed"))
            digest=sha256(path);entry=remote[file]
            if entry.lfs:
                assert digest==entry.lfs.sha256
            else:
                import hashlib
                body=path.read_bytes()
                assert hashlib.sha1(f"blob {len(body)}\0".encode()+body).hexdigest()==entry.blob_id
            all_files[file]={"sha256":digest,"bytes":path.stat().st_size,"git_blob_id":entry.blob_id}
            with path.open(encoding="utf-8") as handle:
                raw[split]=[json.loads(line) for line in handle if line.strip()]
        if (root/task/"RESULTS.json").exists():
            saved=json.loads((root/task/"RESULTS.json").read_text())
            require_hash(root/task/"vectors.pt",saved["vectors_sha256"])
            require_hash(root/task/"SCORES.json",saved["scores_sha256"])
            payload=torch.load(root/task/"vectors.pt",weights_only=True,map_location="cpu")
            task_replay={}
            with torch.inference_mode():
                for entry in checkpoints:
                    model,cp=build_model(ROOT/entry["path"],device);del cp
                    old_rows=[{"id":i,"leaves":x,"lengths":n,"targets":[payload["representations"]["context_mean"][i]]} for i,(x,n) in enumerate(zip(payload["leaves"],payload["masses"]))]
                    a,_,idx=batched_balanced_roots(model,[r for r in old_rows if len(r["leaves"])>1],device,17)
                    alt=torch.stack([x[0] if len(x)==1 else torch.zeros(768) for x in payload["leaves"]])
                    if idx: alt[idx]=torch.tensor(np.asarray(a))
                    error=float((alt-payload["representations"][f"learned_{entry['seed']}"]).abs().max())
                    assert error<=cfg["replay_tolerance"]
                    task_replay[str(entry["seed"])]=error;del model
            tasks[task]=saved;replay[task]=task_replay
            for key in ("first","weighted","unweighted","context_mean","context_cls","learned_mean"):
                c=saved["summaries"]["all"][key+"/ndcg_at_10"];table.append((f"{task.replace('_',' ')}: {key.replace('_',' ')}",c["mean"],c["ci95"]))
            c=saved["summaries"]["all"]["delta_weighted"];table.append((f"{task.replace('_',' ')}: learned minus weighted",c["mean"],c["ci95"]))
            print(f"Retained completed task without rescoring: {task}",flush=True)
            del payload,raw
            continue
        docs=sorted(raw["corpus"],key=lambda r:r["doc_id"]);queries=sorted(raw["queries"],key=lambda r:r["qid"])
        doc_ids=[d["doc_id"] for d in docs];qids=[q["qid"] for q in queries]
        assert len(doc_ids)==len(set(doc_ids)) and len(qids)==len(set(qids))
        di,qi={v:i for i,v in enumerate(doc_ids)},{v:i for i,v in enumerate(qids)}
        relevance=np.zeros((len(queries),len(docs)),dtype=np.int8)
        for row in raw["qrels"]:
            relevance[qi[row["qid"]],di[row["doc_id"]]]=1
        assert (relevance.sum(1)>0).all()
        groups=positive_components(relevance)
        start=time.perf_counter()
        tokens=[tokenize_text(teacher.tokenizer,d["text"]) for d in docs]
        qtokens=[tokenize_text(teacher.tokenizer,q["text"]) for q in queries]
        assert all(tokens) and all(qtokens)
        lengths=np.array([len(t) for t in tokens]);q_lengths=np.array([len(t) for t in qtokens])
        chunks=[[ids[s:s+1024] for s in range(0,len(ids),1024)] for ids in tokens]
        counts=[len(c) for c in chunks]
        flat=[c for row in chunks for c in row]
        print(f"{task}: {len(docs)} docs, {len(queries)} queries, {sum(lengths)} tokens, {len(flat)} leaves",flush=True)
        torch.cuda.synchronize();t=time.perf_counter()
        leaf_tensor,leaf_diagnostic=mean_encode(teacher,flat,16)
        torch.cuda.synchronize();leaf_seconds=time.perf_counter()-t
        leaves=list(leaf_tensor.split(counts));masses=[torch.tensor([len(c) for c in row],dtype=torch.float32) for row in chunks]
        mean_queries,query_diagnostic=mean_encode(teacher,qtokens,16)
        mean_context,context_diagnostic=mean_encode(teacher,[ids[:8190] for ids in tokens],1)
        cls_context=cls_encode(teacher,[d["text"] for d in docs],1)
        cls_queries=cls_encode(teacher,[q["text"] for q in queries],16)
        encoding_seconds=time.perf_counter()-start
        representations={"first":torch.stack([x[0] for x in leaves]),
            "weighted":torch.stack([F.normalize((F.normalize(x,dim=-1)*(n/n.sum())[:,None]).sum(0),dim=-1) for x,n in zip(leaves,masses)]),
            "unweighted":torch.stack([F.normalize(F.normalize(x,dim=-1).mean(0),dim=-1) for x in leaves]),
            "context_mean":mean_context,"context_cls":cls_context}
        accounting=None;timing={};task_replay={}
        with torch.inference_mode():
            for entry in checkpoints:
                model,cp=build_model(ROOT/entry["path"],device);del cp
                xd,nd=[x.to(device) for x in leaves],[n.to(device) for n in masses]
                torch.cuda.synchronize();t=time.perf_counter()
                value,accounting=ragged_reduce(model,xd,nd,4,64)
                torch.cuda.synchronize();timing[str(entry["seed"])]=time.perf_counter()-t
                representations[f"learned_{entry['seed']}"]=value.cpu()
                # Independent implementation: old training reducer, grouped by
                # leaf count rather than packing all corpus nodes by level.
                rows=[{"id":i,"leaves":x,"lengths":n,"targets":[mean_context[i]]} for i,(x,n) in enumerate(zip(leaves,masses))]
                # Old helper skips singleton rows; those roots are carried exactly.
                multi=[r for r in rows if len(r["leaves"])>1]
                a,_,ids=batched_balanced_roots(model,multi,device,17)
                alternate=torch.stack([x[0] if len(x)==1 else torch.zeros(768) for x in leaves])
                if ids: alternate[ids]=torch.tensor(np.asarray(a))
                error=float((alternate-value.cpu()).abs().max());assert error<=cfg["replay_tolerance"]
                task_replay[str(entry["seed"])]=error
                del model,xd,nd
                require_hash(ROOT/entry["path"],entry["sha256"])
                torch.cuda.empty_cache()
        for name,values in representations.items(): validate_vectors(values.numpy(),name,unit=True)
        measured={}
        with threadpool_limits(limits=4):
            for method,values in representations.items():
                q=(cls_queries if method=="context_cls" else mean_queries).double().numpy()
                d=values.double().numpy()
                q=q/np.linalg.norm(q,axis=1,keepdims=True);d=d/np.linalg.norm(d,axis=1,keepdims=True)
                similarity=q@d.T
                measured[method]=retrieval_metrics(similarity,relevance)
        measured["learned_mean"]={k:np.mean([measured[f"learned_{s}"][k] for s in cfg["seeds"]],axis=0) for k in measured["first"]}
        columns={f"{method}/{metric}":v for method,metrics in measured.items() for metric,v in metrics.items()}
        for reference in ("weighted","unweighted","context_mean","context_cls"):
            columns[f"delta_{reference}"]=columns["learned_mean/ndcg_at_10"]-columns[f"{reference}/ndcg_at_10"]
        summaries={"all":grouped_summary(columns,groups,cfg,300+task_index*10)}
        positive_lengths=np.max(np.where(relevance>0,lengths[None],0),axis=1)
        bounds=[0,*section["length_bins"],np.inf]
        for i,(lo,hi) in enumerate(zip(bounds[:-1],bounds[1:])):
            mask=(positive_lengths>lo)&(positive_lengths<=hi)
            if mask.any():
                summaries[f"length_{int(lo)+1}_{int(hi) if np.isfinite(hi) else 'plus'}"]=grouped_summary({k:v[mask] for k,v in columns.items()},groups[mask],cfg,301+task_index*10+i)
        metadata={"documents":len(docs),"queries":len(queries),"positive_document_clusters":len(np.unique(groups)),"tokens":int(lengths.sum()),
            "max_tokens":int(lengths.max()),"above_8192":int((lengths>8192).sum()),"above_32768":int((lengths>32768).sum()),
            "leaves":sum(counts),"max_leaves":max(counts),"above_training_leaf_cap_96":sum(n>96 for n in counts),
            "max_depth":max(accounting["depths"]),"depth_histogram":{str(d):accounting["depths"].count(d) for d in sorted(set(accounting["depths"]))},
            "merges_per_checkpoint":sum(accounting["merges"]),"query_max_tokens":int(q_lengths.max()),"truncated_mean_queries":int((q_lengths>8190).sum()),
            "leaf_encoding_seconds":leaf_seconds,"all_encoding_and_tokenization_seconds":encoding_seconds,"composition_seconds_single_pass_not_benchmark":timing,
            "pooling_precision_diagnostics":{"leaves":leaf_diagnostic,"queries":query_diagnostic,"context":context_diagnostic}}
        task_root=root/task;task_root.mkdir()
        atomic_tensor(task_root/"vectors.pt",{"leaves":leaves,"masses":masses,"representations":representations,"mean_queries":mean_queries,"cls_queries":cls_queries})
        atomic_json(task_root/"SCORES.json",{"query_ids":qids,"doc_ids":doc_ids,"relevance":[np.flatnonzero(row).tolist() for row in relevance],
            "query_clusters":groups.tolist(),"document_token_lengths":lengths.tolist(),"positive_document_max_lengths":positive_lengths.tolist(),
            "scores":{k:v.tolist() for k,v in columns.items()}})
        tasks[task] = {"summaries":summaries,"metadata":metadata,"vectors_sha256":sha256(task_root/"vectors.pt"),"scores_sha256":sha256(task_root/"SCORES.json")}
        replay[task]=task_replay
        atomic_json(task_root/"RESULTS.json",tasks[task])
        for key in ("first","weighted","unweighted","context_mean","context_cls","learned_mean"):
            c=summaries["all"][key+"/ndcg_at_10"];table.append((f"{task.replace('_',' ')}: {key.replace('_',' ')}",c["mean"],c["ci95"]))
        c=summaries["all"]["delta_weighted"];table.append((f"{task.replace('_',' ')}: learned minus weighted",c["mean"],c["ci95"]))
        print(f"Completed all methods: {task}",flush=True)
        del representations,leaves,leaf_tensor,tokens,qtokens,flat,chunks,raw
    for file,digest in preserved.items(): require_hash(root/file,digest)
    seal(root,{"resume_protocol_sha256":sha256(root/"RESUME_PROTOCOL.json"),"tasks":tasks,"dataset_files":all_files,"independent_replay_errors":replay,"no_model_fitting_or_selection":True,
        "metric_scale":"0_to_1","mean_accumulation":"FP32 uniformly for ALL benchmark leaves, queries and context references; FP16 backbone weights unchanged",
        "precision_recovery_before_any_retrieval_scores":True,"confidence":"query-weighted document-cluster bootstrap; fixed corpus and checkpoints; no simultaneous guarantee"})
    report_table(root,"LongEmbed frozen-model retrieval",table,
        "All four real tasks and all candidates/queries at the pinned revision. nDCG@10 on a 0--1 scale. Mean-pool queries match all mean-based representations; CLS queries match the documented first-token context reference only. Teacher context references truncate, whereas recursive and flat full-document representations use every token. Intervals cluster queries by connected positive-document components while preserving query weighting and fixed candidate corpora. No task fitting or model selection.","tab:longembed-review")


if __name__ == "__main__": main()
