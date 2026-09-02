#!/usr/bin/env python3
"""Inspect only public corpus metadata/text lengths, then freeze retrieval protocol.

This stage deliberately never opens queries or qrels and never loads a model.
"""
from __future__ import annotations
import gzip, hashlib, json, os, statistics, sys, time, urllib.request, zipfile
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/v3_cls/downstream"
LONGEMBED_REV = "10039a580487dacecf79db69166e17ace3ede392"
MLDR_REV = "d67138e705d963e346253a80e59676ddb418810a"
NFC_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def summary(lengths):
    values=sorted(lengths)
    def q(x): return values[min(len(values)-1, round((len(values)-1)*x))]
    return {"documents_inspected":len(values),"characters":{"min":values[0],"median":q(.5),"p90":q(.9),"max":values[-1],"mean":statistics.fmean(values)}}

def nfc_lengths():
    raw=OUT/'source'/"nfcorpus.zip"; extracted=OUT/'source'/"nfcorpus"
    raw.parent.mkdir(parents=True,exist_ok=True)
    if not raw.exists(): urllib.request.urlretrieve(NFC_URL,raw)
    if not extracted.exists():
        with zipfile.ZipFile(raw) as z: z.extractall(raw.parent)
    corpus=extracted/'corpus.jsonl'; lengths=[]
    with corpus.open() as f:
        for line in f:
            row=json.loads(line); lengths.append(len((row.get('title','')+'\n'+row.get('text','')).strip()))
    return summary(lengths), {"url":NFC_URL,"archive_sha256":sha(raw),"corpus_bytes":corpus.stat().st_size}

def hf_lengths(repo, config, revision, limit=None):
    ds=load_dataset(repo,config,split='corpus',revision=revision,streaming=True)
    lengths=[]
    for i,row in enumerate(ds):
        text=row.get('text')
        if not isinstance(text,str): raise ValueError('malformed corpus text during blind inspection')
        lengths.append(len(text))
        if limit is not None and i+1>=limit: break
    return summary(lengths)

def mldr_lengths(limit=128):
    # datasets>=4 no longer executes MLDR.py, so inspect a bounded prefix of the
    # immutable gzip directly. Closing the streaming response prevents a full 985 MB download.
    url=f"https://huggingface.co/datasets/Shitao/MLDR/resolve/{MLDR_REV}/mldr-v1.0-en/corpus.jsonl.gz"
    lengths=[]
    with urllib.request.urlopen(url) as response, gzip.GzipFile(fileobj=response) as stream:
        for _ in range(limit):
            row=json.loads(stream.readline())
            text=row.get('text')
            if not isinstance(text,str): raise ValueError('malformed MLDR corpus text during blind inspection')
            lengths.append(len(text))
    return summary(lengths), url

def exclusive_write(path,obj):
    flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL
    fd=os.open(path,flags,0o444)
    with os.fdopen(fd,'w') as f:
        json.dump(obj,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inspection={"schema":"v3_cls_downstream_blind_corpus_inspection_v1","created_unix":time.time(),
      "restriction":"Only dataset metadata and corpus text character lengths were inspected; no queries, qrels, model, embeddings, or retrieval results were opened.","datasets":{}}
    inspection["datasets"]["nfcorpus"],inspection["datasets"]["nfcorpus_source"]=nfc_lengths()
    mldr, mldr_url=mldr_lengths(128)
    inspection["datasets"]["mldr_en_sample"]={**mldr,"sample":"first 128 corpus rows only","revision":MLDR_REV,"immutable_corpus_url":mldr_url,"compressed_corpus_bytes":984905843,"published_corpus_documents":200000,"published_average_document_length":3308}
    inspection["datasets"]["longembed_narrativeqa"]={**hf_lengths('dwzhu/LongEmbed','narrativeqa',LONGEMBED_REV),"revision":LONGEMBED_REV,"expected_documents":355}
    inspect_path=OUT/'BLIND_LENGTH_INSPECTION.json'; exclusive_write(inspect_path,inspection)
    precommit={
      "schema":"v3_cls_downstream_precommit_v1","created_unix":time.time(),"blind_inspection":"results/v3_cls/downstream/BLIND_LENGTH_INSPECTION.json","blind_inspection_sha256":sha(inspect_path),
      "benchmarks":[
        {"id":"nfcorpus","kind":"BEIR","split":"test","source":NFC_URL,"archive_sha256":inspection['datasets']['nfcorpus_source']['archive_sha256']},
        {"id":"longembed_narrativeqa","kind":"LongEmbed real-world long-document retrieval","source":"dwzhu/LongEmbed","config":"narrativeqa","revision":LONGEMBED_REV,"splits":{"corpus":"corpus","queries":"queries","qrels":"qrels"}}
      ],
      "long_benchmark_decision":{"preferred":"MLDR English","selected":"LongEmbed NarrativeQA","reason":"MLDR English's published 200,000-document corpus is infeasible for an exact float32 1024-token leaf cache and three checkpoint passes on the available 5.7 GB workspace. LongEmbed NarrativeQA is an established public English long-context retrieval benchmark with 355 genuinely long documents; all 355 corpus lengths were inspected without opening labels."},
      "teacher":{"id":"Alibaba-NLP/gte-modernbert-base","revision":"e7f32e3c00f91d699e8c43b53106206bcc72bb22","pooling":"normalized CLS","raw_token_limit":8190},
      "document_input":"NFCorpus: strip(title + newline + text); NarrativeQA: text verbatim","leaf_cache":{"raw_tokens_per_leaf":1024,"contiguous":True,"retain_final_short_leaf":True,"dtype":"float32","reuse":"one shared hash-pinned cache per dataset for every recursive checkpoint, token-weighted mean, and unweighted mean"},
      "methods":["recursive_seed424242","recursive_seed271828","recursive_seed314159","token_weighted_leaf_mean","unweighted_leaf_mean"],
      "direct_teacher_rule":"Include direct_teacher only if every query and every document in the benchmark is <=8190 raw tokenizer tokens; otherwise omit the method for the entire benchmark (no truncation or mixed representation).",
      "ranking":"cosine similarity of L2-normalized query/direct-CLS and document embeddings; deterministic doc-id ascending tie break; preserve top 100",
      "metrics":["nDCG@10","Recall@10","Recall@100","MRR@10"],
      "metric_implementation":"pytrec_eval ndcg_cut.10, recall.10, recall.100; explicit reciprocal rank at cutoff 10 over positive qrels",
      "stratification":{"query_assignment":"highest-relevance qrel document; ties resolved by lexicographically smallest document id","tokens":[[1,1024],[1025,4096],[4097,8190],[8191,32768],[32769,None]],"leaves":[[1,1],[2,4],[5,8],[9,32],[33,None]],"depth":"deterministic contiguous K=4 recursion depth, exact integer value"},
      "qrels_policy":"Fail on malformed/duplicate/nonfinite qrels, unknown query/document IDs, missing positive judgments, query-set mismatch, duplicate IDs, missing embeddings, or nonfinite scores. Zero-grade judgments are retained but not relevant. No silent drops.",
      "checkpoints":{
        "424242":{"path":"results/v3_cls/adversarial_v2_candidate_seed424242/autoencoder_seed424242.pt","sha256":"5f44f72a48d64830dc411e8f5559c9a89a9737fcabd8a6bf6bd37bff9fc5c0d8"},
        "271828":{"path":"results/v3_cls/adversarial_v2_final_seed271828/autoencoder_seed271828.pt","sha256":"3f9efe8cfdabd0d8997319969581fa2220c123c27abff3848ce1e0e7a0769fec"},
        "314159":{"path":"results/v3_cls/adversarial_v2_final_seed314159/autoencoder_seed314159.pt","sha256":"9677f5ac5078856457d91ca0df2ddc222c76a2752bcb698cbd570e7ed1bebbc7"}}
    }
    path=OUT/'PRECOMMIT.json'; exclusive_write(path,precommit)
    print(path)
if __name__=='__main__': main()
