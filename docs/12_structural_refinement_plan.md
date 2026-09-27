# Structural decision refinement after balanced-005

Team: **RestoreBuildRun**. The user confirms balanced-005 prediction is complete. Keep that output as the release fallback. All tests, dataset scans and experiments run on the Windows compute laptop; local work is implementation and static review only.

## Evidence and target

The newly supplied 0.99 approach is an unverified external result, not a measured capability of this repository. Its strongest transferable ideas concern precision, abstention and structural evidence. The current balanced-005 development result is **0.861759 macro F0.5**, **0.975408 precision**, **0.729706 recall**, and **0.902108 candidate oracle** on 10,000 development anchors. Earlier public 0.806 and current development metrics measure different data; neither predicts the next public score.

For this fixed development candidate set, even a perfect downstream classifier cannot exceed its measured oracle. A decision-layer improvement cannot recover absent targets. Therefore the immediate goal is **measured precision/abstention gains without another retrieval/model run**, not a promise of 0.99. A later route toward that level would also need large retrieval improvements and evidence on France.

## 1. Preserve the completed work

Add a separate `code/refinement` program and `scripts/Run-Refinement.ps1`. Do not change `src/ber`, its feature schema, normalizer, database builder, model, threshold, saved development scores or original output. The BER source hash remains unchanged. No GPU install, refit, re-embedding or repeat full inference is required. This stage is CPU/SSD work over existing text/scores; moving it to GPU would not remove its main SQLite/string/I/O costs.

Every experiment/output uses a fresh directory. Original SQLite indices are opened read-only; full-S1 ambiguity/address summaries and proposed links live in sidecars. The final candidate file remains byte-identical to the original because it must represent every pair fed into the matcher. Filtering the candidate TSV down to accepted links would misrepresent the pipeline and violate the challenge contract.

## 2. Measure explicit ablations

| Experiment | Actual implemented behavior | Evidence and limits |
| --- | --- | --- |
| Country | Reject a provisional link only when both country values exist and differ | Eligible only if the training audit reports no positive country mismatches; countries remain open strings |
| Ownership | A target can have one winning S1; unique highest retained score wins, best-score ties abstain | Eligible only with zero shared-target components in labels; S1 can retain many S2/S3 targets |
| Country + ownership | Remove cross-country contenders before global ownership resolution | Explicit combined ablation prevents an ineligible claimant stealing another country's target |
| Missing address | Abstain when either address is empty and an anchor/target full normalized name is shared by multiple S1s in that country | Exact-name evidence only; precision gain must exceed lost true matches |
| Street conflict | Reject two confidently parsed, nonempty, disjoint street-name spans | Short syntax patterns, abstention on unknown layouts; normalization has lost punctuation, so not a complete parser |
| Same-address rival name | Reject when target name exactly equals a different S1 name at the identical address, while differing from proposed owner | Direct structural negative evidence, not a generic suffix blacklist |
| Per-business score gap | Effective threshold is max(original threshold, best provisional score minus gap), comparing gaps 0.4, 0.2, 0.1 | A conservative adaptive-set heuristic; raw weighted tree scores are not calibrated probabilities |

All changes are deletion-only, so there is no unsupported recovery claim. Country/ownership are structural and country-agnostic. Text and score-gap rules are automatically restricted to countries represented in labeled development; no US/India-selected text veto silently transfers to unlabeled France. Individual results and the explicit structural combination are reported separately. Do not stack every attractive-looking heuristic into one submission.

## 3. Selection and validation

1. Verify train signature, score-cache hash and exact original macro-score reproduction.
2. Build exact normalized S1 name counts from the complete training reference. This uses unlabeled reference structure, not fit/holdout labels.
3. Hash development anchors into deterministic tune/check halves. Choose the best policy on tune only, preferring unchanged baseline on ties. Gate that single winner on check; never search check results for another winner.
4. Require at least +0.0005 macro F0.5 on each half and no country regression larger than 0.002. Otherwise retain the original output. Report all metrics, singleton errors and removed true/false links by country.
5. Treat both halves as **reused development**, because original model/threshold selection already used them. They do not provide an unbiased holdout or a guarantee. Keep the original holdout untouched during selection.
6. Ownership comparisons on cached dev scores see only sampled dev anchors; full test ownership sees all S1s. Report this limitation. Watch the final country-level removal counts for unexpected scale shifts; no France improvement can be measured without labels.
7. Apply only the frozen passing policy to the complete original ledger. Validate original file hashes, original threshold reproduction, S1 ordering, candidate equality and recorded original membership checks. Resolve ownership globally on disk, not separately per batch or source.
8. Produce a separate output and final provenance manifest. Run the organizer validator and package with the refinement-aware extension. Original BER validation remains correct for the original files and is not weakened to accept arbitrary edited matches.

## 4. France and generated-decoy investigation

Use only organizer-supplied records. Audit distinct S1 names at the same exact normalized address, separately for every country. Count near-name token differences and repeated one-token substitutions; include examples and token occurrence context. Bound groups to 32 rows, skip/count larger groups, and limit examples. This avoids quadratic comparisons at malls or generic addresses.

The critical inference is asymmetric: a token repeatedly distinguishing distinct co-address S1 businesses is **evidence against dropping it**. A token that seldom separates them is **not proof that it is filler**. There are no positive France labels in these comparisons. Frequencies are correlated pair counts, not calibrated probabilities. Similarly, unusually frequent edits can motivate scrutiny but cannot establish that individual links are artificial decoys. No automatic learned filler list, synthetic French supervision, or decoy blacklist is enabled by this implementation.

Inspect `audit.json` next, focusing on high-support separators, substitutions, country differences and representative source/target examples. Before adding a France rule, formulate exactly which links it would change and count them. Test one change per subsequent release. Keep a frozen control and record public-score feedback without treating repeated leaderboard probing as an unbiased evaluation.

## 5. Beyond tonight: addressing the remaining recall ceiling

If time permits a later full experiment, classify real misses into blocking omissions versus reranking errors. Expand only channels with measured recall gains: rare distinguishing name tokens at shared addresses, street names isolated from locality text, high-frequency-name disambiguation with house/postal/street evidence, and language-preserving variants. Measure country/multiplicity candidate oracles before paying for a new model run. Newly recovered pairs must actually enter the scorer and candidate export.

For fuller per-business set optimization, fit score calibration only on reserved labeled calibration anchors and compare expected F0.5 decisions against threshold/gap controls. Account for candidate misses, correlated duplicate records, empty-business prior and target exclusivity. Do not insert uncalibrated tree outputs into an expected-utility formula and call it a probability model. France requires separate transfer evidence, and a calibrated US/India model is not automatically calibrated there.

## 6. Immediate compute sequence

See [refinement commands and release instructions](../code/refinement/README.md). Start with:

```powershell
.\scripts\Run-Refinement.ps1 -Phase Study
```

Return `runs/refinement-006/study.json`. If a policy passes, `-Phase Apply` creates `output/refinement-006`. If baseline is retained, submit/preserve the completed balanced-005 result; do not relax the gates just to force a change. The optional `-Phase Audit` produces `runs/refinement-006-audit/audit.json` for further France analysis. Run phases sequentially to avoid contention. Preserve time for packaging and upload before **2026-09-27 23:59 IST**; no new full training or scoring job is justified by the remaining window.

Synthetic tests cover decision semantics, full ledger application, provenance, selection rejection and packaging. They are supplied for execution on the compute PC. Static code review is not a claim of measured runtime, passing tests, improved accuracy or a successful portal submission.
