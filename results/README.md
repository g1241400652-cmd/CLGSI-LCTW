# Numerical results

This directory contains compact numerical results for the manuscript. It does not redistribute CMU-MOSI/CMU-MOSEI recordings or features, model weights, or training logs.

| File | Contents | Manuscript location |
|---|---|---|
| `validation_runs.json` | Seven validation metrics for each of 12 selected runs, recomputed from saved predictions | Table 3 |
| `analysis_counts.json` | Per-run validation transitions, near-boundary counts, and training gate totals | Tables 4 and 8; boundary analysis |
| `reported_aggregates.json` | Reported test means and component-control summaries | Tables 6 and 7 |
| `training_protocol.json` | Seeds, batches, scheduler, selection rule and LCTW parameters | Section 4.2 |

Validation metrics are stored as fractions for classification scores. Table 3 multiplies those scores by 100 and reports the arithmetic mean and sample standard deviation (`ddof=1`) across runs. Test classification means in `reported_aggregates.json` are already percentages, matching Table 6. The files state these units explicitly.

The component summaries are paired differences (control minus full LCTW), not differences between independently calculated standard deviations. Test means and component summaries are aggregate-level data; they should not be treated as individual runs or used to infer unreported per-run values.

Has0 includes zero labels and uses `>= 0`; Non0 removes zero labels and uses `> 0`. Near-boundary analysis uses the CLGSI prediction with absolute value below 0.25. Validation counts pool run-sample observations. Training counts include repeated presentations across epochs, so they are not counts of unique samples.

Run `python results/verify_results.py` from the repository root to check validation summaries and count consistency. This numerical check does not train a model or independently reproduce the reported test experiments.
