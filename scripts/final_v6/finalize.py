"""Aggregate all v6 development outcomes and apply the frozen promotion gate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from protocol import (AGGREGATE_OUTPUT, CORPORA, DEV_CACHE, RESERVE_BUFFER, ROOT,  # noqa: E402
                      SEEDS, atomic_json, sha256_file)


def run():
    output=ROOT/AGGREGATE_OUTPUT; destination=output/"FINAL_RESULTS.json"
    if destination.exists() or (output/"DONE").exists(): raise FileExistsError("v6 results are append-only")
    declaration_path=output/"PRE_DEVELOPMENT_FROZEN.json"
    json.loads(declaration_path.read_text())
    seed_results=[]; freezes=[]
    for seed in SEEDS:
        directory=ROOT/f"results_kary4_strict_autoencoder_v6_clean_development_seed{seed}"
        freeze_path,evaluation_path=directory/"PRE_DEVELOPMENT_FROZEN.json",directory/"DEVELOPMENT_EVALUATION.json"
        freeze,evaluation=json.loads(freeze_path.read_text()),json.loads(evaluation_path.read_text())
        if evaluation["checkpoint"]!=freeze["checkpoint"]: raise ValueError("checkpoint changed between freeze and development")
        marker=directory/"DEVELOPMENT_EVALUATION_COMPLETE"
        if marker.read_text().strip()!=f"{sha256_file(evaluation_path)}  DEVELOPMENT_EVALUATION.json": raise ValueError("evaluation completion hash mismatch")
        freezes.append(freeze); seed_results.append(evaluation)
    aucs=[item["safe_direct"]["scopes"]["pooled"]["attacks"]["strongest_orientation_safe_auc"] for item in seed_results]
    mean_auc=sum(aucs)/len(aucs)
    every_cosine=all(item["safe_direct"]["scopes"]["pooled"]["teacher_cosine"]>.98 for item in seed_results)
    every_builtin=all(item["safe_direct"]["builtin_decoder_exact4"]["pooled"]["ordered_child_cosine"]>=.90 for item in seed_results)
    retention=all(item["safe_direct"]["train_fitted_matched_probe"]["all_floors_pass"] for item in seed_results)
    changed=all(freeze["training"][role]["nonzero"] and freeze["training"][role]["changed_parameter_tensors"]==freeze["training"][role]["parameter_tensors"]
                for freeze in freezes for role in ("encoder_change","decoder_change"))
    no_collapse=all(not item["safe_direct"]["scopes"][scope]["health"]["collapse"] and item["safe_direct"]["scopes"][scope]["health"]["finite_fraction"]==1.0
                    and not item["full_document_diagnostic"]["health"][scope]["collapse"] and item["full_document_diagnostic"]["health"][scope]["finite_fraction"]==1.0
                    for item in seed_results for scope in ("pooled",*CORPORA))
    no_saturation=all(freeze["training"]["maximum_geodesic_saturation_fraction"]==0 and freeze["training"]["maximum_importance_saturation_fraction"]==0
                      and item["safe_direct"]["telemetry"]["geodesic_saturation_count"]==0 and item["safe_direct"]["telemetry"]["importance_saturation_count"]==0
                      and item["full_document_diagnostic"]["telemetry"]["geodesic_saturation_count"]==0 and item["full_document_diagnostic"]["telemetry"]["importance_saturation_count"]==0
                      for freeze,item in zip(freezes,seed_results))
    conditions={"three_seed_mean_strongest_auc_below_0.60":mean_auc<.60,
                "every_seed_strongest_auc_below_0.61":all(value<.61 for value in aucs),
                "every_seed_pooled_teacher_cosine_above_0.98":every_cosine,
                "every_seed_pooled_builtin_child_cosine_at_least_0.90":every_builtin,
                "every_seed_all_frozen_train_retention_floors_pass":retention,
                "every_encoder_and_decoder_parameter_tensor_changed":changed,
                "safe_and_full_finite_without_collapse":no_collapse,
                "training_safe_and_full_without_saturation":no_saturation}
    manifest=ROOT/DEV_CACHE/"CACHE_MANIFEST.json"; cache=json.loads(manifest.read_text())
    if int(cache["maximum_source_index_opened_exclusive"])>RESERVE_BUFFER[0]: raise ValueError("reserve boundary was reached")
    result={"study":AGGREGATE_OUTPUT.removeprefix("results_"),"development_only":True,"protected_data_accessed":False,
            "reserve_buffer_opened":False,"future_protected_opened":False,"promotion":{"conditions":conditions,"passed":all(conditions.values()),
            "disposition":"development_pass_allows_a_separate_future_decision_but_does_not_freeze_or_access_protected_data"},
            "aggregate":{"mean_pooled_strongest_auc":mean_auc,"pooled_strongest_auc_by_seed":dict(zip(map(str,SEEDS),aucs))},
            "seeds":seed_results,"training_freezes":freezes,
            "provenance":{"pre_development_declaration":str(declaration_path.relative_to(ROOT)),"pre_development_declaration_sha256":sha256_file(declaration_path),
                          "development_cache_manifest":str(manifest.relative_to(ROOT)),"development_cache_manifest_sha256":sha256_file(manifest),
                          "maximum_source_index_opened_exclusive":cache["maximum_source_index_opened_exclusive"],"paper_edited":False,
                          "protected_work_performed":False}}
    atomic_json(destination,result)
    lines=["# CLEAN DEVELOPMENT ONLY v6 retraining and validation","",f"**Promotion: {'PASS' if result['promotion']['passed'] else 'FAIL'}**","",
           "No reserve-buffer or protected-data access occurred. Checkpoints were fixed, probes were fitted, and retention floors were recorded before the development cache existed.","",
           "## Frozen gate","",f"- Three-seed mean strongest AUC: `{mean_auc:.9f}` (target: `<0.60`)",
           f"- Every-seed AUC guard passed: `{conditions['every_seed_strongest_auc_below_0.61']}`",f"- Pooled cosine guard passed: `{every_cosine}`",
           f"- Built-in decoder guard passed: `{every_builtin}`",f"- TRAIN-fitted matched-probe retention passed: `{retention}`",
           f"- Every encoder and decoder parameter tensor changed: `{changed}`",f"- Finite/noncollapsed/no-saturation: `{no_collapse and no_saturation}`","",
           "| seed | strongest AUC | teacher cosine | built-in child cosine | retention |", "|---:|---:|---:|---:|:---:|"]
    for seed,item,value in zip(SEEDS,seed_results,aucs):
        lines.append(f"| {seed} | {value:.9f} | {item['safe_direct']['scopes']['pooled']['teacher_cosine']:.9f} | {item['safe_direct']['builtin_decoder_exact4']['pooled']['ordered_child_cosine']:.9f} | {item['safe_direct']['train_fitted_matched_probe']['all_floors_pass']} |")
    lines += ["","The unrestricted full-document diagnostic used every available token from the same selected documents, with no teacher comparison or AUC.",""]
    (output/"REPORT.md").write_text("\n".join(lines))
    checkpoint_lines=[f"{freeze['checkpoint']['sha256']}  ../{freeze['checkpoint']['path']}" for freeze in freezes]
    (output/"CHECKPOINTS.sha256").write_text("\n".join(checkpoint_lines)+"\n")
    sums=[]
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"SHA256SUMS","DONE"}: sums.append(f"{sha256_file(path)}  {path.relative_to(output)}")
    (output/"SHA256SUMS").write_text("\n".join(sums)+"\n"); (output/"DONE").write_text(f"{sha256_file(destination)}  FINAL_RESULTS.json\n")
    return destination


if __name__=="__main__": print(run())
