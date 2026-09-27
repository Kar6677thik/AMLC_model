# Baseline analysis and next experiment

Team: **RestoreBuildRun**. Target: Windows ThinkPad P16, Ultra 7, 32 GB RAM, NVIDIA RTX PRO 1000 Blackwell laptop GPU, **8,151 MiB dedicated VRAM**, driver 596.58. All runtime tests and challenge computation belong on this PC.

## What baseline-001 establishes

The user reports a submitted score of **0.806** and about five hours end to end. The transferred reports establish these separate local measurements:

| Measurement | Baseline |
| --- | ---: |
| Development anchors | 10,000 |
| Development macro F0.5 | 0.8404988086 |
| Candidate oracle macro F0.5 | 0.8863254306 |
| Candidate pair recall | 0.74626308 |
| Match micro precision / recall | 0.96918 / 0.69774 |
| India candidate recall | 0.64641 |
| US candidate recall | 0.81326 |
| Mean candidates per anchor | 34.1143 |
| Anchors at the 40-candidate total cap | 65% |
| Training, including feature extraction | 184.94 seconds |
| Full-index test benchmark | 107.43 anchors/second |
| Test S1 anchors | 1,732,544 |
| Estimated scoring alone | 4.48 hours |

Evidence: supplied `baseline-001/training.json`, `dev_report.json`, `benchmark_test.json` and manifests. Public and development scores are different measurements; their numerical difference is not a reliable prediction of a new submission score.

The largest accuracy constraint is missed candidates, particularly for India. No matcher can recover a positive omitted by retrieval. There is also about 0.046 development macro F0.5 between the existing matcher and its candidate oracle. Precision is already high, so raising recall while controlling false positives is more promising than indiscriminately lowering the threshold.

The timing points to retrieval, text features and inference across 1.73 million anchors. Accelerating three minutes of training alone cannot remove a multi-hour inference bottleneck.

## Implemented experiment

1. **Broader retrieval on the existing index.** Query three times as many possible token/gram keys, then select at most ten small posting lists per source with coverage across exact, name, address and trigram channels. Try at most two bounded frequent-key intersections even if another channel found candidates. Rank using name, address and numeric evidence, with quotas that retain strong address and strong name alternatives. Score at most 32 candidates per source (64 total), up from 20. Never inject validation positives or truncate frequent postings by entity ID.
2. **Reuse text work and CPU cores.** Bounded caches retain folded strings, token sets, digit sets and key frequencies. Batched SQL frequency/posting queries reduce call overhead. Four Windows-spawn processes perform retrieval and features against read-only SQLite connections, with bounded queues and deterministic output order.
3. **46 features, up from 32.** Add abbreviation variants, business-name core similarity, initials, conflicting name digits, address digits, postal-like tokens and first-number comparisons. These are generic features learned from supplied labels, not external business information. Missing or conflicting postal-like tokens are evidence, not hard exclusion rules.
4. **Larger training sample and a CUDA backend.** Train on 100,000 fit anchors with XGBoost histogram trees on `cuda:0`. Use depth 6, 128 bins and 600 rounds as a starting recipe. A separate LightGBM CPU configuration retains the other improvements for comparison or recovery.
5. **Larger scoring batches.** Accumulate 1,024 anchors per model call. At 64 candidates and 46 float32 features, the numeric input alone is about 11.5 MiB per full batch. This is not a total GPU memory estimate: trees, histograms and working buffers also consume memory. The training feature array has a maximum of about 1.18 GB on disk, plus labels/weights and library working memory.
6. **Measure and reject regressions.** Record CPU worker retrieval/feature times separately from model wall time. Test real CUDA execution before training and reject detected CPU fallback. Benchmark 5,000 spaced test anchors. Full inference through the launcher requires development F0.5 above baseline and a conservative scoring estimate within the chosen time budget.

SQLite and string matching remain CPU work; continuous high GPU utilization is not expected. More workers or a larger batch is not automatically faster. Four workers and 1,024 anchors are starting settings for this hardware, subject to the benchmark.

## Run on the compute PC

Transfer the updated repository first. Preserve `runs/baseline-001`, `output/baseline-001`, the completed SQLite indexes and the baseline submission. Use a **new run ID**. The old model has a different feature/source signature and cannot run under this new source; reproducing it requires its original code and recorded environment. The reported baseline Git commit is `5d4b1e42a1cf6db1e1754f4033bf41970f16a406`.

From the repository root, replace the dataset path:

```powershell
.\scripts\Run-Optimized.ps1 -Dataset 'D:\AMLC\student_resource\dataset' -RunId optimized-002 -Phase Evaluate -AuditRetrieval
```

This installs the optional XGBoost dependency, executes synthetic tests and a real CUDA smoke check, optionally audits retrieval, trains, tunes the development threshold and benchmarks inference. It stops before full prediction. Default `-Work` is `artifacts/baseline`; specify the existing work directory if the first run used another location. The SQLite builder and normalization source are unchanged, and stored index settings match baseline, so compatible completed indexes are reused. Dataset hashes are still checked. A `.building.sqlite` file is not a completed reusable index.

Inspect these files:

- `runs/probes/optimized-002-retrieval.json`: same 1,000 development anchors across baseline retrieval and v2 caps of 20, 32 and 48 per source. Compare candidate oracle, recall, counts, country slices and time. Audit timing includes feature computation; it does not measure trained-model accuracy.
- `runs/optimized-002/training.json`: device, fit time, feature time and feature importance.
- `runs/optimized-002/dev_report.json`: macro F0.5, precision/recall, singleton errors and country slices, plus the newly calibrated threshold.
- `runs/optimized-002/benchmark_test.json`: real test-index throughput, device, timings and conservative ETA. This excludes full output export, validation, packaging and upload.

If development score improves and the ETA leaves enough deadline buffer:

```powershell
.\scripts\Run-Optimized.ps1 -Dataset 'D:\AMLC\student_resource\dataset' -RunId optimized-002 -Phase Predict -SkipInstall
```

The default gate allows at most **three hours for the 2x scoring estimate**. It is a planning budget, not a promise of completion or an automatic wall-clock deadline check. Adjust `-MaxScoringHours` only against the actual remaining time; leave additional time for export, validators, packaging and upload before 23:59 IST. `-Phase All` runs the same checks and then predicts automatically if they pass.

Use `nvidia-smi -l 2` in a second PowerShell window to observe dedicated VRAM and GPU activity. The launcher also prints a CUDA verification report; Task Manager's default 3D graph alone is not a useful CUDA verification.

## Choosing the next configuration

- If retrieval recall improves but time is too high, compare the audit's 20 versus 32 candidates per source before increasing the cap further. Candidate efficiency is also part of the challenge ranking.
- If model quality disappoints, compare the optimized CPU recipe using `-Backend cpu -RunId optimized-cpu-002 -Phase Evaluate`. This is a separate trained model with its own threshold and outputs, not a silent CUDA fallback.
- If memory pressure appears, lower training anchors first; if GPU fit exhausts VRAM, lower `max_bin` or `max_depth`. If inference memory is the issue, lower `prediction_batch_anchors`. Configuration changes require a new model run; never alter caps or features only at test inference.
- If workers are limited by disk or RAM, compare two versus four workers with a new run. If GPU model time is already a small fraction, increasing scoring batches will have little impact.
- Select settings using development data, then evaluate the chosen recipe once on the locked holdout with `ber evaluate --partition holdout`. Do not tune on holdout. No local France accuracy is available because provided labels cover US/India.
- Retain the already submitted baseline if the replacement does not establish a convincing improvement or cannot finish before the deadline. No new leaderboard score or speedup is claimed until measured.

Learned multilingual retrieval and neural reranking remain deferred. Building another full-corpus embedding index under the remaining deadline and an 8 GB laptop GPU is a larger operational risk than improving the measured lexical bottleneck first.

## Release and verification

The launcher validates both new TSVs and the exact scored-candidate ledger. Run the unchanged organizer validator using the new output paths, then use `ber package` with real team member names, as documented in the package README. Neither launcher uploads files. Preserve the existing submission until replacement files pass all checks.

Added synthetic coverage checks Unicode/digit views, equality and ordering across serial/Windows-spawn extraction, and XGBoost save/load/package behavior. Existing metric, leakage, resume, corruption and package checks remain. **These tests and CUDA execution have not been run on the development PC; the compute launcher runs them before challenge work.** Static source review is not runtime verification.

XGBoost's official [GPU guide](https://xgboost.readthedocs.io/en/stable/gpu/index.html) documents CUDA histogram training and GPU prediction, and its [installation guide](https://xgboost.readthedocs.io/en/stable/install.html) documents Windows GPU wheels. XGBoost uses the [Apache 2.0 license](https://github.com/dmlc/xgboost/blob/master/LICENSE); final packages record exact installed dependency versions.
