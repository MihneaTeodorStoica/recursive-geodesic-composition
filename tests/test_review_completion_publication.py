"""Fast, offline checks that publication statements match frozen review results."""
import hashlib
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
STUDIES=("input_order","tree_families","probe_families","decoder_gradient","longembed_review")
PREVIOUS={
 "official_v6/results.json":"236081e0c721602c742b573e3dfd06a556f9448ddaab299f10007788de51a7c0",
 "safe_baselines_v1/RESULTS.json":"b1734a0733826c2a00f838be5beaf3d74935bdcfd21b0e03d6867826cddba557",
 "recursive_reuse_v1/RESULTS.json":"3aa555d2073bc1b26f2d4f74fe65f83ec493be9b4fda034b62278c27ba305e5d",
 "reconstruction_calibration_v1/RESULTS.json":"3beb1d99ff83d8e13bf9a3d7f052c7750c3c858883431ce9351e78867ab3b06c",
 "continuation_comparison_v1/RESULTS.json":"5088db03712fb902ffeea55ded39db555e359ccc2962df33b3b144276e443194"}


def digest(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name,expected", list(PREVIOUS.items()))
def test_immutable_prior_result_bytes(name,expected):
 assert digest(ROOT/"results"/name)==expected


@pytest.mark.parametrize("name",STUDIES)
def test_new_studies_audited_with_unchanged_results(name):
 root=ROOT/"results"/f"{name}_v1"
 result=json.loads((root/"RESULTS.json").read_text())
 protocol=json.loads((root/"PROTOCOL.json").read_text())
 audit=json.loads((root/"AUDIT.json").read_text())
 assert result["protocol_sha256"]==digest(root/"PROTOCOL.json")
 assert audit["passed"] and all(audit["checks"].values())
 assert audit["results_sha256"]==digest(root/"RESULTS.json")
 assert audit["prior_results_unchanged"]
 assert protocol["immutable_previous_results"]=={f"results/{p}":s for p,s in PREVIOUS.items()}
 for path,expected in json.loads((root/"ARTIFACT_HASHES.json").read_text()).items():
  target=root/path
  if target.suffix!='.pt': assert digest(target)==expected


def test_retrieval_all_tasks_correctly_reported():
 root=ROOT/"results/longembed_review_v1"
 result=json.loads((root/"RESULTS.json").read_text())
 expected={"2wikimqa":(300,300,0.012497749992407008),"qmsum":(197,1527,-0.019949063466936392),
           "summ_screen_fd":(336,336,-0.027004546244337197),"narrativeqa":(355,10449,-0.004480849872787795)}
 assert set(result["tasks"])==set(expected)
 for task,(docs,queries,diff) in expected.items():
  entry=result["tasks"][task]
  assert entry["metadata"]["documents"]==docs and entry["metadata"]["queries"]==queries
  assert entry["summaries"]["all"]["delta_weighted"]["mean"]==diff
  for method in ("first","weighted","unweighted","context_mean","context_cls","learned_mean"):
   assert method+"/ndcg_at_10" in entry["summaries"]["all"]
 assert result["precision_recovery_before_any_retrieval_scores"] is True


def test_train_and_numerical_recoveries_explicit():
 directory=ROOT/"reproducibility/protocols/review_completion_v1"
 recovery=json.loads((directory/"TRAIN_POPULATION_RECOVERY.json").read_text())
 assert recovery["no_embeddings_or_model_fits_or_new_evaluation_scores_before_change"]
 assert recovery["changes"]["fit_per_corpus"]==[500,160]
 assert recovery["changes"]["validation_per_corpus"]==[100,40]
 precision=json.loads((directory/"CONTEXT_REDUCTION_RECOVERY.json").read_text())
 assert precision["no_retrieval_scores_available_before_repair"] and precision["no_documents_dropped"]
 assert digest(directory/"recovery/evaluate_longembed_pre_scoring.py")==precision["failed_evaluator_sha256"]
 assert digest(directory/"recovery/evaluate_longembed_initial.py")==precision["recovered_evaluator_sha256"]
 resume=json.loads((ROOT/"results/longembed_review_v1/RESUME_PROTOCOL.json").read_text())
 assert resume["no_recomputation_or_replacement_of_completed_scores"]
 assert digest(ROOT/"scripts/complete_longembed_review.py")==resume["implementation_sha256"]


def test_review_release_is_separate_and_checksums_are_pinned():
 manifest=json.loads((ROOT/"reproducibility/REVIEW_RELEASE_MANIFEST.json").read_text())
 assert manifest["release"]["tag"]=="v2-paper-review"
 assert manifest["release"]["previous_historical_release"]=="v1-paper"
 assert manifest["preserved_prior_result_sha256"]=={f"results/{p}":s for p,s in PREVIOUS.items()}
 assert len(manifest["published_assets"])==55
 assert len({x["asset_filename"] for x in manifest["published_assets"]})==55
 assert all(x["byte_size"]>0 and len(x["sha256"])==64 for x in manifest["published_assets"])
 for name in STUDIES:
  assert manifest["results"][name]["sha256"]==digest(ROOT/"results"/f"{name}_v1/RESULTS.json")
 checksum=ROOT/"reproducibility/REVIEW_RELEASE_ASSETS.sha256"
 assert digest(checksum)==manifest["checksum_asset"]["sha256"]
 assert checksum.read_text()=="".join(f"{x['sha256']}  {x['asset_filename']}\n" for x in manifest["published_assets"])
 assert all(x["role"]!="final_clean_v6_checkpoint" for x in manifest["published_assets"])


def test_gradient_study_is_not_relabelled_historical():
 result=json.loads((ROOT/"results/decoder_gradient_v1/RESULTS.json").read_text())
 assert result["new_regime_not_historical_reproduction"] and result["training_models"]==6
 assert result["teacher_documents"]==4000 and result["reconstruction_documents"]==606
