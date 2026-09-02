# V2.2 TRAIN pilot amendment V2 — C accumulator

After TRAIN access, C was found impossible with two rows/corpus/step and minimum four. BASELINE/A/B are immutable; B remains health-ineligible. C alone uses a non-rolling two-step block accumulator. Step 1 stores all per-corpus generated/teacher pairs detached and applies zero covariance loss (the residual-mean term remains current-step). Step 2 concatenates detached step-1 context before current gradient-bearing pairs, computes each corpus teacher-whitened covariance with the unchanged relative ridge 1e-4, averages exactly 1/4, then clears all four buffers. Incomplete buffers at run end are discarded. No history gradient exists.

Unchanged: batch 8 (2 rows/corpus), seed/init/order, optimizer, 250 steps, coefficient 64, residual mean .01, condition <=1e8, gradient <=1000, three-step saturation gate. Only original C is retried; its failed empty namespace is preserved by renaming to `c_failed_pre_amendment`.
