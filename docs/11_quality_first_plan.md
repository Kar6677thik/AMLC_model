# Quality-first plan after the failed speed/recall probe

Team: RestoreBuildRun. Deadline remains 2026-09-27 23:59 IST. Computation runs on the Windows compute laptop with 32 GB RAM and 8,151 MiB NVIDIA VRAM. This plan seeks the best **measured, finishable** model among a small set of experiments; it cannot guarantee 0.99 or predict a public leaderboard score.

## 1. What the supplied probe actually shows

The probe selected no recipe because none passed both its speed and quality gates:

| Recipe | Development oracle F0.5 | Candidate pair recall | Test anchors/second |
| --- | ---: | ---: | ---: |
| Full v2 control | 0.912366 | 0.803236 | 74.41 |
| Adaptive-32 | 0.911150 | 0.800925 | 68.61 |
| Balanced-32 | 0.904230 | 0.785467 | 194.22 |
| Balanced-20 | 0.903516 | 0.783155 | 206.16 |
| Lean-20 | 0.889472 | 0.747616 | 378.46 |

These are the same 2,000 development anchors and approximately 3,000 test anchors across recipes. They are oracle/throughput results, not trained-model scores. The earlier 10,000-anchor optimized-002 development result remains 0.867298, with oracle 0.909838. Sample sizes differ, so do not treat the 0.912366 versus 0.909838 difference as a model improvement.

For full v2, retrieval ranking consumed **94.64 of 146.12 summed retrieval worker seconds** (~65%); record fetching consumed 29.61 (~20%); intersections only 4.98 (~3%). Removing intersections was not the main speed opportunity. Adaptive channel allocation also increased the raw pool and failed to improve speed. The probe correctly rejected the smaller recipes' recall losses rather than silently trading them away.

## 2. Preserve retrieval quality while removing unused string work

Previously, ranking every raw target built the complete matching feature view: token/numeric sets, core business name, abbreviation variants, initials, postal tokens and first numbers. Ranking only uses folded text, sorted unique tokens and address digits. Most raw targets are discarded before the final matcher, so computing all those other fields for them is wasted work.

The compact ranking backend computes only those required fields, with bounded caches. It retains the **same scalar RapidFuzz functions, floating-point arithmetic, ranking formula, channel budgets, intersection policy, final cap and tie-breaking** as v2. There is no approximate pruning, altered precision or new identity heuristic in this optimization. Complete 46-feature views are still calculated for every candidate actually passed to the model.

The legacy backend remains available for exact synthetic comparison. Tests compare candidate records, scores, hit counts and feature matrices, including Unicode and digit cases. On real data, the new control's complete development candidate lists are compared row by row with optimized-002's saved `dev_scores.jsonl`; a difference stops the sweep.

Six feature workers are the initial setting for this experiment, within the 32 GB RAM budget. This is a configurable hypothesis, not a measured six-worker speedup. Native thread pools remain bounded and scoring stays at 1,024 anchors/batch. The database builder and persisted normalization/key settings remain unchanged, so completed SQLite indexes are reused.

## 3. Spend the saved time on model quality

Reuse optimized-002's **fit-only** `train_x.npy`, `train_y.npy`, `train_w.npy` and `training_anchors.tsv`. This avoids another 23-minute feature extraction pass and leaves all original artifacts unchanged. Feature import is restricted to the reviewed optimized-002 source hash (or the current source), learned v2 mode, the exact 46-feature schema, matching dataset/split signatures and matching feature dependency versions. Array shapes, dtypes, finite values, binary labels, positive counts and positive weights are checked. Fresh file hashes record the imported data's provenance; the original run did not store array hashes, so these are lineage records rather than proof against earlier finite-value edits.

Train and compare:

1. **Control:** the original GPU recipe, refitted using the preserved arrays. This establishes the new source's development quality and actual full-index throughput.
2. **Deeper model:** 900 trees, depth 8, 128 bins, with the original learning rate and regularization. This tests whether a more expressive matcher closes some of the gap to the candidate oracle. Higher training fit does not establish better validation performance.
3. **Equal-weight ensemble:** average the control and deeper model's scores, then calibrate its own development threshold. Both members see exactly the same candidates, so this adds GPU model work without repeating retrieval or expanding exported candidates.

Each completed model is evaluated on the full configured 10,000 development anchors. Models are checked in descending development macro F0.5 order; the highest-scoring one that meets score and deadline gates is selected, with ties preferring the control. No larger model or ensemble is assumed better. Optional trials run only if the control's measured ETA leaves a further experiment allowance before inference and release. An optional model failure preserves the completed control and is recorded in the report.

Development labels select model/threshold; they are never appended to tree training. The sampled fit partition, shared-target quarantine and exact candidate export contract remain. Holdout remains untouched during model selection. If the final runtime budget permits, evaluate the chosen recipe once on holdout without retuning. All development scores remain selection-biased and do not establish France or public-leaderboard performance.

## 4. Run the experiment

Transfer this code update first. Do not run it concurrently with another writer or alter code during the study. On the compute laptop, from the repository root:

```powershell
.\scripts\Run-Quality.ps1 -Dataset 'D:\AMLC\student_resource\dataset' -StudyId quality-004 -Phase Evaluate
```

Replace the dataset path. By default it reads `runs/optimized-002` and reuses `artifacts/baseline`. Use `-SourceRun` and `-Work` if those directories differ. **Do not delete optimized-002's saved arrays or development score file.** The package extra is installed, synthetic tests run, and real CUDA execution is checked before model fitting.

The launcher writes:

- `runs/quality-004/quality.json`: measured trial scores, timings, candidate equivalence result, selection and deadline checks.
- `runs/quality-004/control/`: full control model and reports.
- `runs/quality-004/deep/`: deeper model and reports, if time allowed.
- `runs/quality-004/ensemble/`: ensemble manifest, both model files and reports, if time allowed.

It stops before full inference. Send `quality.json` plus the selected model's `dev_report.json` and `benchmark_test.json` for review. If a model qualifies, generate the submission files with:

```powershell
.\scripts\Run-Quality.ps1 -Dataset 'D:\AMLC\student_resource\dataset' -StudyId quality-004 -Phase Predict -SkipInstall
```

`-Phase All` runs the same experiment and proceeds automatically if gates pass. `-Workers` defaults to six; use a new StudyId to compare four if memory/I/O contention is observed. Completed studies are immutable. After an incomplete experiment, inspect its report and preserve completed models; a fresh experiment needs a new StudyId.

## 5. Deadline and release policy

The best completed model must score at least `max(baseline development score, optimized-002 development score - 0.002)` to be selected. Its measured full-model scoring estimate, multiplied by **1.35**, must leave **60 minutes** for export, validators, packaging and upload before **23:59 IST**. These planning margins are configurable and are not a completion guarantee. The check runs again immediately before prediction; a selected model can become infeasible while waiting.

The launcher never silently relaxes gates or starts an unfinishable model after a failed selection. If compact full v2 still misses the time budget, retain the submitted baseline and use the measured reports to make the next explicit trade-off; do not assume the smaller balanced recipe's oracle loss equals its trained-model loss. The already supplied probe is retained as evidence for a later fallback experiment.

Prediction validates the final TSVs and exact score ledger. Then run the organizer validator and package using the selected run subdirectory and `output/quality-004`. Ensemble packages include each member, its hash, its full training configuration and weight. Inference from the packaged assets requires no original training arrays. A separate `ber rebuild-ensemble` command regenerates its fit features from organizer data, refits the members and calibrates the rebuilt ensemble, so the original imported arrays are not required for a full rebuild.

## 6. What this does not establish

The current candidate oracle is a ceiling for the evaluated sample and fixed candidate set, not a universal limit for this challenge or its hidden test set. Reaching 0.99 would require substantially better recovery and classification than we have measured. The earlier public-score range estimate was not supported by sufficient evidence and should not guide submission decisions.

Hybrid multilingual retrieval, new address indexes and neural reranking could broaden candidate recovery in a longer project, but their benefit is unmeasured and their full-corpus cost is not justified by this evening's budget. This revision prioritizes preserving the demonstrated retrieval gain and testing stronger matching without redoing expensive preparation.

Static review and PowerShell parsing run on the development PC. Runtime regression tests, GPU tests, model selection and challenge benchmarks remain for the compute laptop. No new score or speedup is claimed until those results exist.
