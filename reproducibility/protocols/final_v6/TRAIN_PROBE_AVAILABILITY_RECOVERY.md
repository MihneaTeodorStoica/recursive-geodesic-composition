# Preserved TRAIN-probe availability recovery

After the immutable fresh TRAIN cache was published, and before any v6 model update or development access, an outcome-blind availability audit counted unique exact-four-child TRAIN groups. FineWeb had 184 such groups, below the original disjoint allocation of 120 fit plus 120 floor groups.

The original configs and cache-manifest hash are preserved. The additive `configs_train_availability_recovery/` configs fix 80 fit and 80 floor groups per corpus. This leaves disjoint groups, changes no model-training recipe or promotion criterion, uses TRAIN only, and was fixed before model outcomes or development data existed. All later v6 training/freezing/evaluation uses these additive configs.
