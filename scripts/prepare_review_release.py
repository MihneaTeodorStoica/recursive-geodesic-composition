#!/usr/bin/env python3
"""Create a separate v2 evidence/asset manifest; never modify v1-paper files."""
from __future__ import annotations
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RELEASE=ROOT/"artifacts/review_release_v2"
MANIFEST=ROOT/"reproducibility/REVIEW_RELEASE_MANIFEST.json"
CHECKSUMS=ROOT/"reproducibility/REVIEW_RELEASE_ASSETS.sha256"
PREVIOUS={"official":"2cabef20750aa0a80148784012a1518ea2d1c1008ac7d617467b68f4b9cfb917",
 "safe":"f16d23f504d3155dce680243a680d6c26ed2835b46f6aef9100b7fd9c1c57da5",
 "recursive":"74944d0960b9311ef8b897ef130474b6a04d463cac4ea6222cb8d96aa6614e29",
 "reconstruction":"6f378128f85bcac3ae762c9a931985e3fd5a110d08cc8334bcfef87746bc46c7",
 "continuation":"84c151264e962577a35cd12ad47ace1a63852d5abcc8ddcaa1784827d23b16c1"}
PRIOR=["results/official_v6/results.json","results/safe_baselines_v1/RESULTS.json","results/recursive_reuse_v1/RESULTS.json",
       "results/reconstruction_calibration_v1/RESULTS.json","results/continuation_comparison_v1/RESULTS.json"]
STUDIES=["input_order","tree_families","probe_families","decoder_gradient","longembed_review"]


def sha256(path):
    d=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(8<<20),b""): d.update(block)
    return d.hexdigest()


def manifest():
    assert {path:sha256(ROOT/path) for path in PRIOR}==dict(zip(PRIOR,PREVIOUS.values()))
    paper=ROOT/"paper/main.pdf"
    if not paper.is_file(): raise FileNotFoundError("Build and inspect the reviewed PDF before packaging")
    outputs={};local_only={};assets=[]
    for name in STUDIES:
        study=ROOT/f"results/{name}_v1"
        audit=json.loads((study/"AUDIT.json").read_text())
        result=sha256(study/"RESULTS.json")
        assert audit["passed"] and audit["results_sha256"]==result
        outputs[name]={"result_path":str((study/"RESULTS.json").relative_to(ROOT)),"sha256":result,
                       "audit_sha256":sha256(study/"AUDIT.json"),"protocol_sha256":sha256(study/"PROTOCOL.json")}
        expected=json.loads((study/"ARTIFACT_HASHES.json").read_text())
        for subpath,digest in expected.items():
            if subpath.endswith(".pt"):
                path=study/subpath
                assert path.exists() and sha256(path)==digest
                local_only[str(path.relative_to(ROOT))]={"sha256":digest,"byte_size":path.stat().st_size,
                    "release_asset":False,"reproducibility_note":"Regenerate from pinned inputs, or download separately trained weights if listed below"}
    cache=ROOT/"cache_review_train_v1"
    for name in ("rows.pt","tokens.pt"):
        path=cache/name
        local_only[str(path.relative_to(ROOT))]={"sha256":sha256(path),"byte_size":path.stat().st_size,
                  "release_asset":False,"reproducibility_note":"Regenerate from pinned TRAIN selection; not a released dataset cache"}
    # Only new model weights, not embeddings or tokenized source material, are distributed.
    paths=[ROOT/f"results/decoder_gradient_v1/{arm}_seed{seed}.pt" for seed in (424242,271828,314159)
           for arm in ("warm","attached","detached")]
    paths+=sorted((ROOT/"results/probe_families_v1/probes").glob("*.pt"))
    assert len(paths)==54 and len(set(paths))==54
    for path in paths:
        assert path.is_file() and str(path.relative_to(ROOT)) in local_only
        asset="review_"+path.stem+".pt"
        role="shared_warm_checkpoint" if path.name.startswith("warm") else (
             "new_attached_or_detached_joint_model" if path.parent.name=="decoder_gradient_v1" else "new_train_only_fitted_probe")
        assets.append({"asset_filename":asset,"protocol_path":str(path.relative_to(ROOT)),"role":role,
                       "byte_size":path.stat().st_size,"sha256":local_only[str(path.relative_to(ROOT))]["sha256"]})
        local_only[str(path.relative_to(ROOT))]["release_asset"]=True
    assets.append({"asset_filename":"One_Vector_from_Many_reviewed.pdf","protocol_path":"paper/main.pdf","role":"reviewed_35_page_pdf",
                   "byte_size":paper.stat().st_size,"sha256":sha256(paper)})
    assert len({a["asset_filename"] for a in assets})==len(assets)
    return {"schema_version":1,"release":{"repository":"anonymous-artifact-source",
        "tag":"v2-paper-review","previous_historical_release":"v1-paper"},"results":outputs,
        "preserved_prior_result_sha256":dict(zip(PRIOR,PREVIOUS.values())),"published_assets":sorted(assets,key=lambda a:a["asset_filename"]),
        "local_artifact_identities":local_only,"license_and_scope":"Only newly fitted model/probe weights and reviewed PDF are mirrored; no source text, tokenized source corpus, or derived embedding caches are published. All JSON study scores and audit reports live in Git. The original released checkpoint/probe weights remain in v1-paper; new models do not replace them.",
        "numerical_reproduction":"TRAIN checkpoints and derived vectors have recorded exact SHA-256; GPU re-training/regeneration is not asserted byte-identical across environments."}


def main():
    data=manifest();RELEASE.mkdir(parents=True,exist_ok=True)
    for item in data["published_assets"]:
        source=ROOT/item["protocol_path"];destination=RELEASE/item["asset_filename"]
        if not destination.exists():
            if source.suffix==".pdf": shutil.copyfile(source,destination)
            else: os.link(source,destination)
        assert destination.stat().st_size==item["byte_size"] and sha256(destination)==item["sha256"]
    CHECKSUMS.write_text("".join(f"{a['sha256']}  {a['asset_filename']}\n" for a in data["published_assets"]))
    checksum=RELEASE/CHECKSUMS.name
    shutil.copyfile(CHECKSUMS,checksum)
    data["checksum_asset"]={"filename":checksum.name,"sha256":sha256(checksum),"byte_size":checksum.stat().st_size}
    MANIFEST.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")
    print(f"Prepared {len(data['published_assets'])} weights/PDF assets + checksum; {len(data['local_artifact_identities'])} locally hashed tensor files",flush=True)


if __name__=="__main__":main()
