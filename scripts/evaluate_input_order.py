#!/usr/bin/env python3
"""All-24 input permutations, coupled masses, and a fixed-parent negative control."""
import torch.nn.functional as F
from review_common import *
from evaluate_reconstruction_calibration import probe_entries, load_probe
from recursive_geodesic_composition.extended_controls import PERMUTATIONS, slot_scores


def main():
    root, cfg, device, cal, baseline, _, published = begin("input_order", __file__)
    old, rows, old_manifest = exact_inputs(cal, baseline, published)
    children = old["children"]
    masses = torch.stack([r["lengths"] for r in rows])
    permutations = torch.tensor(PERMUTATIONS)
    x = children[:, permutations].reshape(-1,4,768)
    n = masses[:, permutations].reshape(-1,4)
    checkpoints, probes = checkpoint_entries(baseline), probe_entries(baseline)
    scores, tensors, replay = {}, {}, {}
    with torch.inference_mode():
        for entry, probe_entry in zip(checkpoints, probes):
            seed = str(entry["seed"])
            model, checkpoint = build_model(ROOT / entry["path"], device)
            del checkpoint
            probe, _ = load_probe(probe_entry, device)
            cols, stored = {}, {}
            for mode in cfg["order"]["mass_modes"]:
                lengths = n if mode == "actual" else torch.ones_like(n)
                parents = torch.cat([model.merge(x[s:s+64].to(device), lengths[s:s+64].to(device)).cpu() for s in range(0,len(x),64)])
                validate_vectors(parents.numpy(), unit=True)
                identity = parents.reshape(606,24,768)[:,0]
                if mode == "actual":
                    error = float((identity-old["parents"][seed]).abs().max())
                    assert error <= cfg["replay_tolerance"]
                    replay[f"{seed}_identity_previous_parent"] = error
                repeated = identity[:,None].expand(-1,24,-1).reshape(-1,768)
                cols[f"{mode}_parent_agreement"] = cosine_rows(parents.numpy(), repeated.numpy()).reshape(606,24).mean(1)
                # No teacher cosine against unencoded permuted text is claimed.
                stored[mode] = {"parents": parents.reshape(606,24,768)}
                for family, call in (("decoder", model.decode), ("probe", probe)):
                    pred = torch.cat([call(p.to(device)).cpu() for p in parents.split(64)])
                    ordered, shuffled = slot_scores(pred, x)
                    original = children[:,None].expand(-1,24,-1,-1).reshape(-1,4,768)
                    stale, _ = slot_scores(pred, original)
                    vals = {"following": ordered, "shuffled": shuffled, "following_gap": ordered-shuffled, "stale": stale}
                    for metric, value in vals.items():
                        cols[f"{mode}_{family}_{metric}"] = value.reshape(606,24).mean(1).numpy()
                    fixed = pred.reshape(606,24,4,768)[:,0,None].expand(-1,24,-1,-1).reshape(-1,4,768)
                    fixed_order, fixed_shuffle = slot_scores(fixed, x)
                    cols[f"{mode}_{family}_fixed_parent_gap"] = (fixed_order-fixed_shuffle).reshape(606,24).mean(1).numpy()
                    cols[f"{mode}_{family}_identity_gap"] = (ordered-shuffled).reshape(606,24)[:,0].numpy()
                    assert abs(cols[f"{mode}_{family}_fixed_parent_gap"]).max() < 1e-14
                    stored[mode][family] = pred.reshape(606,24,4,768)
                    if mode == "actual":
                        old_name = "strict_decoder" if family == "decoder" else "frozen_probe"
                        error = float((stored[mode][family][:,0]-old["predictions"][f"{old_name}_seed{seed}"]).abs().max())
                        assert error <= cfg["replay_tolerance"]
                        replay[f"{seed}_identity_previous_{family}"] = error
                # Independently replay the full set with B=17, perm-major packing.
                alt_x = x.reshape(606,24,4,768).transpose(0,1).reshape(-1,4,768)
                alt_n = lengths.reshape(606,24,4).transpose(0,1).reshape(-1,4)
                rebuilt = torch.cat([model.merge(alt_x[s:s+17].to(device),alt_n[s:s+17].to(device)).cpu() for s in range(0,len(x),17)])
                rebuilt = rebuilt.reshape(24,606,768).transpose(0,1).reshape(-1,768)
                error = float((rebuilt-parents).abs().max())
                assert error <= cfg["replay_tolerance"]
                replay[f"{seed}_{mode}_parent"] = error
                for family, call in (("decoder", model.decode), ("probe", probe)):
                    pred = torch.cat([call(p.to(device)).cpu() for p in rebuilt.split(17)])
                    error = float((pred-stored[mode][family].reshape(-1,4,768)).abs().max())
                    assert error <= cfg["replay_tolerance"]
                    replay[f"{seed}_{mode}_{family}"] = error
            scores[seed], tensors[seed] = cols, stored
            for e in (entry,probe_entry): require_hash(ROOT/e["path"],e["sha256"])
            del model, probe
            torch.cuda.empty_cache()
            print(f"Completed order permutations {seed}",flush=True)
    atomic_tensor(root/"vectors.pt", {"outputs":tensors,"children":children,"masses":masses,"permutations":permutations})
    summaries=save_scores(root,scores,old["groups"],old["labels"],cfg)
    seal(root,{"documents":606,"permutations":PERMUTATIONS,"summaries":summaries,"independent_replay_errors":replay,
               "vectors_sha256":sha256(root/"vectors.pt"),"previous_manifest_sha256":sha256(ROOT/cal["output_dir"]/"INFERENCE_MANIFEST.json"),
               "scores_sha256":sha256(root/"SCORES.json"),"new_training_updates":0,"new_teacher_encoding_calls":0})
    pooled=summaries["mean"]["pooled"]
    table=[]
    for mode in ("actual","equal"):
        for family in ("decoder","probe"):
            for metric in ("identity_gap","following_gap","fixed_parent_gap"):
                v=pooled[f"{mode}_{family}_{metric}"]
                table.append((f"{mode} masses, {family}: {metric.replace('_',' ')}",v["mean"],v["ci95"]))
    report_table(root,"Input-order controls",table,
        "All 24 input permutations on 606 prefixes; counts travel with leaves or are set equal. Following gap averages ordered-minus-all-target-assignment cosine after recomputing each parent. The fixed-parent control changes only scoring and must average to zero. No reordered-text teacher target is constructed. Intervals pair documents across checkpoints.","tab:input-order")


if __name__ == "__main__": main()
