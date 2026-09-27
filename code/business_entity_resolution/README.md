# RestoreBuildRun entity resolution

Implemented: streaming input audit, SQLite-backed lexical retrieval, grouped fit/dev/holdout partitions, 46 pair features, LightGBM CPU and XGBoost CUDA matchers (or rules), macro F0.5 threshold tuning, resumable inference, strict output validation, and final zip generation. Optimized configurations add broader retrieval and Windows-spawn feature workers.

**Execution status:** baseline-001 produced development macro F0.5 0.84050; the user reports submission score 0.806 and roughly five hours. The optimization changes are statically reviewed but unexecuted on the development PC. Tests, CUDA checks and challenge-data runs must execute on the separate compute PC. No improved score, speedup or passing-test claim is made yet.

## Next run: 8 GB NVIDIA GPU

From the repository root on the compute PC:

```powershell
.\scripts\Run-Optimized.ps1 -Dataset 'D:\AMLC\student_resource\dataset' -RunId optimized-002 -Phase Evaluate -AuditRetrieval
```

This runs synthetic tests, verifies actual CUDA training/prediction, audits retrieval, trains on 100,000 fit anchors, tunes the threshold and benchmarks 5,000 test queries. Defaults are four feature workers, 1,024 anchors per scoring batch and 32 candidates per target source. XGBoost CUDA support comes from the `[gpu]` package extra; the check rejects detected CPU fallback. See the [optimization analysis](../../docs/09_baseline_analysis_and_optimization.md) for measurements, configuration choices and limitations.

Review the new development and benchmark reports, then run:

```powershell
.\scripts\Run-Optimized.ps1 -Dataset 'D:\AMLC\student_resource\dataset' -RunId optimized-002 -Phase Predict -SkipInstall
```

Full inference requires development F0.5 above 0.8404988086 and a conservative scoring ETA within `-MaxScoringHours` (default 3). Allow additional time for exports, validators and upload. Use `-Backend cpu -RunId optimized-cpu-002 -Phase Evaluate` for a separate LightGBM comparison. There is no silent GPU fallback.

Compatible completed indexes in `artifacts/baseline` are reused; pass `-Work` if they live elsewhere. Preserve baseline outputs and use a new run ID. The new feature/source signature cannot resume a model trained with the original code. The commands below describe the general pipeline; substitute the optimized config and new run/output paths as appropriate.

## General CPU launcher on the Windows compute PC

Install Python **3.10 or newer** (3.11 is a reasonable starting point) and Git. Copy the supplied dataset separately; Git excludes it. The baseline needs no GPU, CUDA, external service, or pretrained model download.

Expected data layout:

```text
<DATASET>/train/train_source1.tsv
<DATASET>/train/train_source2.tsv
<DATASET>/train/train_source3.tsv
<DATASET>/train/train_ground_truth.tsv
<DATASET>/test/test_source1.tsv
<DATASET>/test/test_source2.tsv
<DATASET>/test/test_source3.tsv
```

From the repository root in PowerShell, replace the dataset path with the actual location:

```powershell
.\scripts\Run-Compute.ps1 -Dataset 'D:\AMLC\student_resource\dataset' -RunId baseline-new
```

The launcher creates `.venv`, installs the editable package, runs synthetic tests, prepares training data, fits the model, tunes the threshold, prepares the test index, benchmarks 1,000 spaced test queries, predicts all test S1 entities, and strictly validates both outputs. Read the printed throughput estimate: it excludes output writing, validation, packaging, and upload. Keep the deadline buffer in the plan.

Rerun the same command to reuse completed training/evaluation and resume completed inference shards. It will not silently overwrite incomplete training or reuse a different model version. `-SkipInstall` reuses the existing environment. Use `-Python 'C:\path\to\python.exe'` if `python` is not on PATH.

Outputs and reports are under `output/<RunId>/` and `runs/<RunId>/`; SQLite indexes are under `artifacts/baseline/`. Read `dev_report.json`, `benchmark_test.json`, and `validation.json` before selecting a release. The launcher does not upload anything.

## Manual setup and tests

From this package directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Activate that environment or substitute its Python executable in all subsequent commands. Bootstrap dependencies are bounded ranges. Every run captures exact installed versions in `environment.txt`; final packaging uses that lock. Inference rejects changed dependency versions or changed source code.

The synthetic tests cover hand-computed metric cases, singleton credit, Unicode handling, shared-target group splits, held-out target exclusion, both model modes, frozen holdout decisions, partial inference resumption, strict corruption detection, and archive layout. They do not establish real-data accuracy or throughput.

## Stage commands

Use explicit paths; none of the commands require a fixed working directory after package installation.

```powershell
$Data = 'D:\AMLC\student_resource\dataset'
$Work = 'D:\AMLC\artifacts\baseline'
$Run = 'D:\AMLC\runs\baseline-001'
$Out = 'D:\AMLC\output\baseline-001'

python -m ber prepare --dataset $Data --work $Work --config configs/baseline.json --split train
python -m ber train --dataset $Data --work $Work --run $Run --config configs/baseline.json --mode learned
python -m ber evaluate --work $Work --run $Run --partition dev
python -m ber benchmark --dataset $Data --work $Work --run $Run --limit 1000
python -m ber predict --dataset $Data --work $Work --run $Run --output $Out
python -m ber validate --work $Work --output $Out
```

`train` automatically calls `prepare`, so the separate prepare command is optional; it is useful for inspecting the input manifest before fitting. `prepare` also audits IDs/labels, computes hashes and generates splits. `predict` prepares/reuses the test index. The benchmark queries the full test index with evenly spaced S1 rows; inspect reported country coverage rather than assuming the sample is fully representative.

For a one-command fresh run:

```powershell
python -m ber run --dataset $Data --work $Work --run $Run --output $Out --config configs/baseline.json --mode learned
```

`run` is for a **new** run directory. After an interruption, resume with the appropriate individual stage, or use the repository launcher. Index builds and training restart from scratch after failure; only completed prediction shards resume. A failed index leaves a named `.building.sqlite` file for inspection. Use a new work directory rather than deleting unrelated artifacts.

Do not run concurrent writers against the same work, run, or output directory. These stages are intended to execute sequentially on the compute PC.

The rules control uses `--mode rules` and a new run/output directory. It is calibrated on development data too. GPU execution uses `configs/optimized-gpu.json` and requires the optional XGBoost dependency. Neural retrieval and neural pair scoring remain deferred.

## Validation policy

The split uses stable hashes of connected components induced by positive labels, approximately 70% fit / 15% dev / 15% holdout. It is **not yet stratified**. Shared targets connect anchors. Held-out positive targets and their exact normalized duplicate target records are quarantined from supervised training negatives. Fit-side candidate lists cannot use those reserved target records.

Unlabeled target text and posting frequencies are available in the retrieval corpus during evaluation, as at inference. This is an explicit transductive retrieval design; the model does not fit held-out labels. Group folds, leave-country-out experiments, and a full-data refit are not yet implemented. This release uses the model trained on the configured sample of the fit partition.

Development defaults to 10,000 anchors (or all if fewer). `evaluate --limit 0` evaluates all anchors in the selected partition, with the scope recorded. Do not rerun threshold searches on holdout: the command rejects repeat evaluation reports and only reads the existing threshold for holdout.

Once the recipe is selected, an optional locked holdout evaluation is:

```powershell
python -m ber evaluate --work $Work --run $Run --partition holdout
```

Score is the mean per-S1 `5*TP/(5*TP + 4*FP + FN)`, with explicit singleton rules. Reports include the candidate oracle ceiling, recall, candidate counts, singleton errors, country slices and multiplicity slices. No France validation accuracy can be claimed from the provided labels.

## Candidate generation and resource bounds

The source-specific inverted index uses hashed lexical keys for exact/composite names, name tokens, selected trigrams, and name/address combinations. Hash collisions can only add candidates, never establish identity. Countries remain generic string evidence; there is no country whitelist or mandatory country equality filter.

Each query reads bounded posting lists, with a small bounded intersection fallback for frequent keys. Lists that cannot be safely bounded are skipped and counted in diagnostics. After lexical ranking, the default cap is 20 targets from each source. The matcher scores every retained candidate; exported candidates are not narrowed to final matches.

Indexes and training arrays are disk-backed. The default CPU baseline sample is 25,000 fit anchors, at most 1 million pairs and about 184 MB of float32 feature storage, plus labels/weights and library working memory. The optimized sample is 100,000 anchors with at most 6.4 million pairs (about 1.18 GB of feature storage). Actual index size and runtime depend on measured row counts and key frequencies. This is not a promise of performance on the full dataset or a billion-record benchmark.

For experiments, copy the config, change one parameter, and use a new run/output directory. Index-affecting changes (`name_tokens`, `address_tokens`, `name_grams`, `seed`, normalization/builder code) require a new work directory. Candidate caps or model settings can reuse a compatible index, but still require fresh model/evaluation runs. Never change inference caps after calibration.

## Output validation and organizer check

The output directory contains:

- `matching_results.tsv`: final accepted targets, one row per S1.
- `candidate_pairs.tsv`: exact candidate IDs scored, one row per S1.
- `scoring_ledger.tsv.gz`: internal scores for every candidate, including empty rows.
- `prediction.json` and `validation.json`: signatures, counts, hashes and checks.
- `.shards/`: completed resumable chunks; retain until release is confirmed.

Internal validation streams outputs in input S1 order and checks exact coverage, duplicates, target membership, candidate/score-ledger equality, frozen-threshold reproduction and file hashes. It uses the prepared test database, avoiding a full in-memory target-ID set.

From the repository root, also run the unchanged organizer validator after releasing other large processes:

```powershell
.\.venv\Scripts\python.exe student_resource/utils/validate_submission.py --matching output/baseline-001/matching_results.tsv --candidate output/baseline-001/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
```

Use the actual dataset/output paths. The organizer validator can consume significant RAM and treats some candidate errors as warnings; the internal checks remain hard failures.

## Final package

After selecting the release, provide real member names:

```powershell
python -m ber package --work $Work --run $Run --output $Out --destination 'D:\AMLC\dist\release-001' --members 'Member One, Member Two' --team RestoreBuildRun
```

This revalidates outputs and produces `RestoreBuildRun_submission.zip` with both TSVs, source, exact environment pins, frozen configuration, model/decision assets, reports, licenses, and a methodology document populated from measured results. Review that document before submitting; it does not claim manual error analysis, clean reproduction or portal acceptance that has not occurred. `release.json` explicitly records those outstanding human checks.

Extracted-package inference (from `code/business_entity_resolution/`):

```powershell
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m ber predict --dataset $Data --work 'D:\AMLC\reproduce\work' --run assets --output 'D:\AMLC\reproduce\output'
python -m ber validate --work 'D:\AMLC\reproduce\work' --output 'D:\AMLC\reproduce\output'
```

For a full rebuild, run `ber run` with `configs/final.json`, a fresh run directory and the original mode from `assets/run.json`. If the original development evaluation used a custom `--limit`, reproduce that same limit with separate train/evaluate/predict commands. Compare the matching and candidate TSV hashes against `assets/prediction.json`; score-ledger gzip byte hashes may differ because gzip embeds creation metadata.

The package excludes raw data, training feature arrays and the large score ledger. Keep the original ledger locally for audit. The source regenerates it; neither its omission from the zip nor the reduced packaging footprint changes the candidate export.
