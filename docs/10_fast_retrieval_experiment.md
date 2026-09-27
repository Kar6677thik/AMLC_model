# Faster retrieval after optimized-002

Team: RestoreBuildRun. All tests, probes, training and inference run on the Windows compute laptop. This revision requires a new model run, normally `optimized-003`; keep both earlier runs and their source/environment versions intact.

## Evidence behind this change

Optimized-002 improved development F0.5 from 0.840499 to **0.867298**, candidate recall from 0.746263 to **0.797114**, and precision from 0.969175 to **0.971421**. India F0.5 rose from 0.769727 to 0.798994. GPU execution passed the CUDA check; XGBoost fitting took **38.8 seconds**. Training feature preparation took 1,355.8 seconds.

The test benchmark fell to **52.24 anchors/second**, implying **9.21 hours** of scoring. Of 95.58 benchmark seconds, only 0.664 seconds were model scoring. Retrieval accounted for 355.35 of 364.02 summed worker seconds. More GPU batching cannot materially fix that bottleneck.

The supplied `optimized-002-retrieval.json` measures the same 1,000 development anchors:

| Retrieval | Oracle F0.5 | Pair recall | Mean scored candidates | Records examined |
| --- | ---: | ---: | ---: | ---: |
| Original v1, 20/source | 0.883494 | 0.744065 | 34.064 | 212,150 |
| v2, 20/source | 0.907626 | 0.794731 | 39.724 | 927,762 |
| v2, 32/source | 0.908385 | 0.797336 | 63.104 | 927,762 |
| v2, 48/source | 0.908678 | 0.798494 | 93.760 | 927,762 |

Moving from 32 to 20 per source saves 37% of scored pairs for a 0.000759 oracle loss on this sample, but leaves retrieval work unchanged. Therefore this revision targets posting lookup, intersections and pre-ranking work as well as the final cap. The audit timings are single sequential measurements with differing cache warmth; they do not prove smaller caps are slower or faster end to end.

## Implementation

Retrieval v3 selects posting lists in rounds across exact/name, address, name-token and trigram channels. With a four-key budget, all available channels can still contribute. It first ranks ordinary candidates. Frequent-key intersections run only if fewer than eight usable candidates exist for that source, or the best lexical score is below 0.70. These values are heuristics to evaluate, not learned probabilities. Fallback candidates are merged by ID and re-ranked with their accumulated key hits; unchanged candidates retain their scores. Intersections that exceed the posting cap are discarded rather than arbitrarily truncated.

The new timers distinguish key/posting lookup, intersections, record fetching, ranking and final selection. They are nested parts of total retrieval worker time, so do not add them to retrieval time again. Four feature workers and 1,024-anchor model batches remain. The SQLite builder, normalization and stored-key settings are unchanged, enabling reuse of completed baseline indexes.

The pre-training probe evaluates five recipes on identical development and test samples:

| Recipe | Keys/source | Posting limit | Query expansion | Final cap/source | Intersections |
| --- | ---: | ---: | ---: | ---: | --- |
| v2-control-32 | 10 | 400 | 3 | 32 | Always, up to 2 |
| adaptive-32 | 10 | 400 | 3 | 32 | Conditional, up to 2 |
| balanced-32 | 6 | 200 | 2 | 32 | Conditional, up to 1 |
| balanced-20 | 6 | 200 | 2 | 20 | Conditional, up to 1 |
| lean-20 | 4 | 120 | 2 | 20 | Conditional, up to 1 |

The adaptive recipe also changes posting-channel allocation, so its comparison with v2 is not a pure intersection-only ablation. Balanced-32 versus balanced-20 isolates the final cap. No recipe is assumed better without measurement.

## Commands on the compute laptop

Transfer the updated code before starting. From the repository root, replace the dataset path:

```powershell
.\scripts\Run-Fast.ps1 -Dataset 'D:\AMLC\student_resource\dataset' -RunId optimized-003 -Phase Probe
```

The default phase installs dependencies, runs synthetic tests, verifies/reuses both indexes, then probes 2,000 development anchors and approximately 3,000 evenly spaced test anchors. This phase is CPU retrieval/feature work and does not train a model. Results are written progressively to `runs/probes/optimized-003-fast/probe.json`.

A recipe qualifies only if:

- Test retrieval/feature throughput is at least **150 anchors/second**.
- Development oracle F0.5 is within **0.005** of the freshly measured v2 control.
- Overall candidate recall is within **0.01** of the control.
- Every sampled country's oracle is within **0.01** and candidate recall is within **0.01** of its control value.

The fastest qualifying recipe is saved as `selected_config.json`. If no recipe qualifies, the probe reports rejection reasons and does **not** select a configuration or start training. Send `probe.json` for analysis. There is no automatic relaxation of quality gates.

If a configuration qualifies, continue:

```powershell
.\scripts\Run-Fast.ps1 -Dataset 'D:\AMLC\student_resource\dataset' -RunId optimized-003 -Phase Evaluate -SkipInstall
```

This verifies the probe's source hash, index signatures and frozen configuration, runs CUDA checks, trains on 100,000 fit anchors and evaluates the full 10,000-anchor development sample. It then benchmarks 5,000 test anchors with the actual model. It stops before full prediction. Probe timing excludes model scoring and output; this second benchmark is required even though model cost was small in optimized-002.

After reviewing the trained-model results:

```powershell
.\scripts\Run-Fast.ps1 -Dataset 'D:\AMLC\student_resource\dataset' -RunId optimized-003 -Phase Predict -SkipInstall
```

Prediction requires development F0.5 above **0.862298** (at most 0.005 below optimized-002 and still above baseline), and enough wall-clock time before **2026-09-27 23:59 IST** for **1.35 times** the measured scoring estimate plus a **60-minute release buffer**. This is a more aggressive timing margin than the old 2x gate; it is explicit, configurable, and not a deadline guarantee. `-TimingMargin`, `-ReserveMinutes` and `-Deadline` control these values. The launcher checks the deadline again after training, rather than assuming time remaining at probe completion is unchanged.

Before training, a preliminary timing gate reserves another 30 minutes for training/evaluation. This allowance is an assumption, and the probe excludes model time. If the gate fails, retain the earlier submission and inspect the report rather than blindly overriding it. `-Phase All` starts from a fresh probe and continues through the same gates automatically. After an interruption, use Evaluate or Predict as appropriate; do not overwrite a completed probe. Incomplete training still requires a new run directory.

`-Work` points to the original completed index directory if different from `artifacts/baseline`. `-MinQueriesPerSecond` controls the probe speed floor, and `-MinDevScore` controls the final development-score floor. Changing source or selected configuration after probing requires a fresh probe/run. Existing optimized-002 artifacts cannot be resumed under modified source; use their recorded commit/environment for that model.

## Validation and limitations

Synthetic tests now cover conditional intersection execution, disabled intersection budgets, serial/Windows-spawn equality for v3, country-regression rejection, and probe/config integrity. The existing CUDA, model serialization, metric, output corruption, package and resume checks remain. Tests and challenge computation have **not** run on this development PC; the compute launcher executes the tests first. Static review alone does not establish a speedup.

The probe is a small development selection exercise, not an unbiased performance estimate. Later recipes may benefit from warmer OS caches. The full trained-model benchmark, country slices and singleton errors must still be reviewed. If time permits, evaluate the selected model once on the locked holdout without retuning. No local France accuracy is available, and no new public score is predicted.

Output candidates remain exactly the records scored by the final matcher. After prediction, run the organizer validator, package with real member names and retain the submitted baseline until the replacement is accepted. No launcher uploads a submission.
