# Decision refinement for completed runs

This separate, standard-library program reuses the completed model's scores. It does not edit `src/ber`, retrain, retrieve candidates, or invalidate the existing run's source hash. Run it **on the compute laptop**. Preserve the completed original output and submission.

## 1. Measure before changing predictions

From the repository root:

```powershell
.\scripts\Run-Refinement.ps1 -Phase Study
```

Defaults: `runs/balanced-005`, `artifacts/baseline`, new `runs/refinement-006`. Override `-RunId`, `-Work`, and `-StudyId` if needed. Existing experiment directories are never overwritten; after an interrupted experiment, use a fresh StudyId.

The launcher first runs the synthetic regression suite. It then verifies the development score-cache hash and training signature, reproduces the original macro F0.5, and annotates provisional matches using the original normalized records. It builds a separate disk-backed Source-1 name/address index; the original index is read-only. This is CPU/SSD work, not another GPU prediction pass. Runtime is unmeasured; it scans S1 once and the cached development scores.

Read `runs/refinement-006/study.json`. It reports individual country, ownership, missing-address, street-conflict, same-address rival-name, and per-business score-gap ablations, plus an explicit country+ownership combination. Each includes macro F0.5, precision, recall, country scores, singleton false-positive rate, and true/false links removed. The threshold remains the original threshold; no new candidate or below-threshold match can be added.

Selection chooses once on a deterministic half of development. The chosen policy must improve **both halves by at least 0.0005 macro F0.5** and lose no more than **0.002 in any represented country**. Otherwise `selected` is `baseline`; **keep the original output**. These are practical gates, not confidence intervals. Both halves were already used for original model/threshold selection; neither constitutes untouched holdout evidence. Ownership competition in sampled development omits anchors outside that sample, so its test behavior has additional uncertainty.

## 2. Apply a passing policy

The original prediction must be complete and have a successful original `ber validate` report. Then:

```powershell
.\scripts\Run-Refinement.ps1 -Phase Apply
```

Use the same overrides as Study, plus `-Original` if the original output is not `output/balanced-005`. This streams the existing ledger, checks its original hashes, rows, threshold decisions and candidate alignment, and writes **new** `output/refinement-006` files. It inherits full candidate membership validation only after checking that the original validation hashes still match every original file. Text lookup is limited to provisional matches.

The candidate TSV is copied byte-for-byte. The matching TSV may only lose links. A disk-backed ownership pass, when selected, keeps the unique highest-scoring claimant per target; tied best scores yield no owner. A business can still have multiple targets from either source. `refinement.json` records hashes, provenance and per-country changes. Keep `decisions.sqlite` for the local audit; it is not submitted.

Text-dependent rules and score-gap policies apply **only to countries represented in development**. France is not assigned a learned veto based on unlabeled hypotheses. Country and target-ownership constraints are structural: they are eligible only if the original training manifest reports zero country-conflicting positives and zero shared-target components. Unknown country values do not produce a country conflict. Candidate generation/export remains unchanged even for rejected cross-country pairs.

Street parsing uses bounded explicit road-name spans and abstains for unsupported formats. Address normalization lost punctuation, so this is a conservative experimental flag, not a complete address parser. It does not compare city/postal/full-address overlap as a street-identity score. Missing-address ambiguity uses exact normalized full names across all S1 records in a country; it does not strip assumed filler words. Same-address rival-name evidence checks whether the target name exactly identifies another S1 name at that address. The set-gap option raises each business's effective cutoff to `max(original_threshold, highest_provisional_score - gap)`; it is a validated ranking heuristic, **not** expected F0.5 calculated from calibrated probabilities.

## 3. France and decoy audit

Only if time remains for analysis:

```powershell
.\scripts\Run-Refinement.ps1 -Phase Audit
```

Send `runs/refinement-006-audit/audit.json` for review. The audit separately reports countries' near-name differences among distinct S1 businesses at identical normalized addresses, token separation counts, repeated one-token substitutions and bounded examples. Groups over 32 records are skipped and counted to prevent quadratic blowups. Counts are descriptive, not calibrated probabilities. No text models, translated examples, external data, automated filler deletion, or fabricated France labels are used. Repeated substitutions are **suspected patterns for inspection**, not proven decoys. Lack of observed separation does not prove a token is harmless.

## 4. Validate and package the selected output

Do not run the original `ber validate` or `ber package` against refined output: they correctly require exact original global-threshold decisions. Preserve them for the original output. Run the organizer validator on the refined files, using the actual test path:

```powershell
.\.venv\Scripts\python.exe student_resource/utils/validate_submission.py --matching output/refinement-006/matching_results.tsv --candidate output/refinement-006/candidate_pairs.tsv --test-dir 'D:\AMLC\student_resource\dataset\test' --check-ids
```

First create the ordinary, verified package of the **original** completed balanced-005 result, if one does not already exist. Supply real member names:

```powershell
.\.venv\Scripts\python.exe -m ber package --work artifacts/baseline --run runs/balanced-005 --output output/balanced-005 --destination dist/balanced-005-original --members 'Actual team member names' --team RestoreBuildRun
```

Then extend that package with the refined matches, policy, study report, new source and methodology:

```powershell
.\.venv\Scripts\python.exe code/refinement/refine.py package --baseline-zip dist/balanced-005-original/RestoreBuildRun_submission.zip --output output/refinement-006 --study runs/refinement-006 --destination dist/refinement-006/RestoreBuildRun_submission.zip
```

The extension verifies baseline TSV hashes and prediction identity and preserves its trained models, exact environment, code and licenses. The original prediction/validation manifests are renamed `baseline_*` because they describe provisional matches. Final hashes and policy are under `code/refinement/assets`. The original methodology is retained as `Baseline_methodology.md`, with a new primary document explaining final decisions and limitations. Packaging is not portal submission or proof of clean reproduction.

For extracted-package inference, reproduce the original predictions as described in `code/business_entity_resolution/README.md`. From the extracted root, apply the bundled frozen policy:

```powershell
python code/refinement/refine.py apply --work 'D:\AMLC\reproduce\work' --run code/business_entity_resolution/assets --original 'D:\AMLC\reproduce\output' --policy code/refinement/assets/policy.json --output 'D:\AMLC\reproduce\refined'
```

Run original `ber validate` before this command. Original `run.json` and `decision.json` must be byte-identical; do not retrain and reuse an old policy. Compare final matching/candidate hashes, then run the organizer validator. Gzip ledger timestamps may differ across inference invocations; each regenerated prediction records its own ledger hash.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s code/refinement -v
```

Coverage includes macro F0.5/empty credit, ownership conflicts and tied-score abstention, multiple targets per anchor, France non-transfer, adaptive cutoffs, street parsing, ambiguity/rival evidence, complete streamed refinement, tamper rejection, and final ZIP provenance. Local implementation work does not establish these tests passed: execution belongs on the compute PC.
