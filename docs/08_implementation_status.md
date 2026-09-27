# 08 — Implementation status and handoff

Team: RestoreBuildRun. Status as of 2026-09-27: first baseline authored; compute-PC verification pending.

## Implemented

| Capability | Location under `code/business_entity_resolution/src/ber/` |
| --- | --- |
| Strict TSV schemas and ID-list handling | `io.py` |
| Unicode-aware normalization and lexical keys | `normalize.py` |
| Input hashes, source/label integrity checks, input statistics | `database.py` |
| Connected-label-component fit/dev/holdout split; held-out target quarantine | `database.py` |
| Disk-backed bounded posting retrieval and source-specific candidate caps | `blocking.py` |
| Exact singleton-aware macro F0.5 and candidate oracle metrics | `metrics.py` |
| 32 pair features; disk-backed training arrays; LightGBM/rules modes | `features.py`, `model.py` |
| Development threshold search; holdout evaluation without retuning | `evaluate.py` |
| Full-index test-query timing estimate | `benchmark.py` |
| Resumable inference shards; exact candidate export and score ledger | `predict.py` |
| Streaming strict validation of coverage, IDs, candidates, decisions and hashes | `validate.py` |
| Required final zip layout and measured-results methodology generation | `package.py` |
| Unified stage commands | `cli.py` |

The repository also includes a Windows launcher, synthetic integration tests, baseline config, bootstrap dependency bounds, MIT implementation license, third-party references, Git ignore rules, and a detailed execution README.

## What has and has not been verified

Source/config/documentation were reviewed on the development PC. No Python program, unit test, dataset scan, training run, or prediction workload was executed here, in accordance with the user's compute-location instruction. Consequently, the code is **not yet test-verified** and no runtime, accuracy or submission-validity result is claimed.

The compute launcher runs the synthetic tests first and stops on failure. Its tests cover scoring, Unicode, split isolation, rules and learned inference, frozen holdout decisions, restart behavior, corruption rejection and packaging. This is the first required execution gate.

No remote push or portal upload has occurred. No GPU model or VRAM has been confirmed. Python and dependency compatibility must be established on the ThinkPad.

## Deliberate first-release limits

- No neural embedding, ANN, cross-encoder or fine-tuning stage yet.
- Split assignment is component-safe but not stratified; inspect measured partition/country/singleton distributions before relying on them.
- Posting frequencies use all unlabeled target records in the indexed corpus; this is a documented inference-context choice, not fit-only IDF.
- No leave-country-out experiment, group cross-validation, confidence intervals or all-label model refit yet.
- Candidate caps and posting suppression limit recall. Evaluate the candidate oracle ceiling before interpreting classifier errors.
- Runtime memory/throughput limits are design targets, not empirical findings. The benchmark measures the complete index with sampled queries but excludes export/validation/packaging costs.
- Only prediction shards resume. Interrupted index builds or training require a fresh compatible work/run location or explicit inspection of incomplete artifacts.
- Source hashes and runtime dependency versions must match the trained run. Freeze code during long inference; intentional code/model changes require a new run.
- Generated final methodology is populated from actual reports but still requires team review. Packaging does not mark reproduction or portal submission complete.

## Compute-PC sequence

1. Synchronize the source repository and transfer the original dataset separately.
2. Run the launcher described in [the implementation README](../code/business_entity_resolution/README.md).
3. If synthetic tests fail, stop and return the complete traceback; no challenge computation should start before they pass.
4. Inspect `artifacts/baseline/train_manifest.json`, `runs/baseline-001/dev_report.json`, and `runs/baseline-001/benchmark_test.json`.
5. Use the measured candidate ceiling, singleton errors and throughput to decide whether a targeted change fits tonight's time budget. Keep the completed baseline immutable.
6. Finish both test outputs and strict validation. Run the organizer validator with the actual dataset path.
7. Evaluate the locked holdout only for the selected recipe, package with real team member names, reproduce extracted contents, and upload before the contingency buffer expires.

The code can be implemented without the exact GPU details; the remaining immediate prerequisite is getting it onto the compute PC and executing the tests there.
